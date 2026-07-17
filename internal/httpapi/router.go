// Package httpapi exposes the protocol transformer as an OpenAI-compatible
// Chat Completions gateway.
package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"time"

	"chat-completion-transformer/internal/capabilities"
	"chat-completion-transformer/internal/upstream"
	"chat-completion-transformer/pkg/transformer"

	"github.com/gin-gonic/gin"
)

// Limits bounds the client request and provider response bodies.
type Limits struct {
	MaxBodyBytes        int64
	MaxStreamBytes      int64
	ResponseBodyTimeout time.Duration
}

const defaultResponseBodyTimeout = 30 * time.Second

type handler struct {
	transformer      *transformer.Transformer
	upstream         *upstream.Client
	maxBodyBytes     int64
	maxResponseBytes int64
	responseTimeout  time.Duration
}

// NewRouter creates the gateway's public HTTP surface.
func NewRouter(service *transformer.Transformer, client *upstream.Client, limits Limits) (*gin.Engine, error) {
	if service == nil {
		return nil, errors.New("transformer is required")
	}
	if client == nil {
		return nil, errors.New("upstream client is required")
	}
	if limits.MaxBodyBytes <= 0 {
		return nil, errors.New("max body bytes must be positive")
	}
	if limits.MaxStreamBytes <= 0 {
		return nil, errors.New("max stream bytes must be positive")
	}
	if limits.ResponseBodyTimeout < 0 {
		return nil, errors.New("response body timeout must not be negative")
	}
	if limits.ResponseBodyTimeout == 0 {
		limits.ResponseBodyTimeout = defaultResponseBodyTimeout
	}

	h := &handler{
		transformer:      service,
		upstream:         client,
		maxBodyBytes:     limits.MaxBodyBytes,
		maxResponseBytes: limits.MaxStreamBytes,
		responseTimeout:  limits.ResponseBodyTimeout,
	}
	router := gin.New()
	router.HandleMethodNotAllowed = true
	router.Use(recoveryMiddleware())
	router.GET("/healthz", h.health)
	router.POST("/v1/chat/completions", h.chatCompletions)
	router.NoRoute(func(c *gin.Context) {
		writeAPIError(c, http.StatusNotFound, "The requested endpoint does not exist.", "invalid_request_error", nil, "not_found")
	})
	router.NoMethod(func(c *gin.Context) {
		writeAPIError(c, http.StatusMethodNotAllowed, "The requested method is not allowed for this endpoint.", "invalid_request_error", nil, "method_not_allowed")
	})
	return router, nil
}

func (h *handler) health(c *gin.Context) {
	if c.Request.Context().Err() != nil {
		return
	}
	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

func (h *handler) chatCompletions(c *gin.Context) {
	ctx := c.Request.Context()
	authorizationValues := c.Request.Header.Values("Authorization")
	if len(authorizationValues) != 1 {
		writeAPIError(c, http.StatusUnauthorized, "A valid Authorization: Bearer token is required.", "invalid_request_error", nil, "invalid_api_key")
		return
	}
	authorization, ok := bearerAuthorization(authorizationValues[0])
	if !ok {
		writeAPIError(c, http.StatusUnauthorized, "A valid Authorization: Bearer token is required.", "invalid_request_error", nil, "invalid_api_key")
		return
	}
	c.Request.Header.Set("Authorization", authorization)

	body, ok := h.readRequestBody(c)
	if !ok {
		return
	}

	decoded := h.transformer.DecodeChatRequest(body)
	if ctx.Err() != nil {
		return
	}
	if !decoded.OK || decoded.Value == nil {
		writeDiagnosticError(c, http.StatusBadRequest, "invalid_request_error", decoded.Diagnostics)
		return
	}

	request := *decoded.Value
	if request.CandidateCount != nil && *request.CandidateCount > 1 {
		writeAPIError(c, http.StatusBadRequest, "This gateway supports one completion per request.", "invalid_request_error", "n", string(transformer.DiagnosticCandidateCountUnsupported))
		return
	}
	endpoint, err := h.upstream.Endpoint(request.ModelAlias)
	if err != nil {
		if errors.Is(err, upstream.ErrRouteNotFound) {
			writeAPIError(c, http.StatusNotFound, "The model `"+request.ModelAlias+"` does not exist or you do not have access to it.", "invalid_request_error", "model", "model_not_found")
			return
		}
		writeAPIError(c, http.StatusInternalServerError, err.Error(), "server_error", nil, "gateway_configuration_error")
		return
	}

	encoded, err := h.encodeRequest(ctx, endpoint, request)
	if err != nil {
		if ctx.Err() != nil {
			return
		}
		writeAPIError(c, http.StatusInternalServerError, err.Error(), "server_error", nil, "request_transform_failed")
		return
	}
	if !encoded.OK || encoded.Value == nil {
		writeDiagnosticError(c, http.StatusBadRequest, "invalid_request_error", appendDiagnostics(decoded.Diagnostics, encoded.Diagnostics...))
		return
	}

	diagnostics := appendDiagnostics(decoded.Diagnostics, encoded.Diagnostics...)
	response, err := h.upstream.Do(ctx, request.ModelAlias, *encoded.Value, request.Stream, c.Request.Header)
	if err != nil {
		writeUpstreamRequestError(c, err)
		return
	}
	if response == nil || response.Body == nil {
		writeAPIError(c, http.StatusBadGateway, "The upstream returned no response.", "upstream_error", nil, "upstream_empty_response")
		return
	}
	defer response.Body.Close()
	copyUpstreamHeaders(c.Writer.Header(), response.Header)

	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		if response.StatusCode < http.StatusBadRequest {
			writeUnexpectedUpstreamStatus(c)
			return
		}
		h.writeUpstreamError(c, response)
		return
	}
	if request.Stream {
		if !isEventStream(response.Header.Get("Content-Type")) {
			writeUnexpectedStreamResponse(c)
			return
		}
		h.streamResponse(c, response.Body, endpoint, request.ModelAlias, request.StreamIncludeUsage, diagnostics)
		return
	}
	if !isJSON(response.Header.Get("Content-Type")) {
		writeUnexpectedJSONResponse(c)
		return
	}
	h.bufferedResponse(c, response.Body, endpoint, diagnostics)
}

func (h *handler) encodeRequest(
	ctx context.Context,
	endpoint capabilities.Endpoint,
	request transformer.CanonicalRequest,
) (transformer.Result[json.RawMessage], error) {
	if endpoint == capabilities.EndpointResponses {
		return h.transformer.EncodeResponsesRequest(ctx, request)
	}
	if endpoint == capabilities.EndpointMessages {
		return h.transformer.EncodeAnthropicRequest(ctx, request)
	}
	return transformer.Result[json.RawMessage]{}, errors.New("unsupported upstream endpoint " + string(endpoint))
}

func (h *handler) bufferedResponse(c *gin.Context, body httpBody, endpoint capabilities.Endpoint, diagnostics []transformer.Diagnostic) {
	ctx := c.Request.Context()
	raw, err := h.readUpstreamBody(ctx, body)
	if err != nil {
		if ctx.Err() != nil {
			return
		}
		writeUpstreamBodyError(c, err, h.maxResponseBytes)
		return
	}

	decoded := h.decodeResponse(endpoint, raw)
	diagnostics = appendDiagnostics(diagnostics, decoded.Diagnostics...)
	if !decoded.OK || decoded.Value == nil {
		writeDiagnosticError(c, http.StatusBadGateway, "upstream_error", diagnostics)
		return
	}
	if failure := responseFailure(*decoded.Value); failure != nil {
		writeAPIError(c, http.StatusBadGateway, failure.Message, failure.Type, failure.Param, failure.Code)
		return
	}

	encoded := h.transformer.EncodeChatResponse(*decoded.Value)
	diagnostics = appendDiagnostics(diagnostics, encoded.Diagnostics...)
	if !encoded.OK || encoded.Value == nil {
		writeDiagnosticError(c, http.StatusBadGateway, "upstream_error", diagnostics)
		return
	}

	setDiagnosticsHeader(c.Writer.Header(), diagnostics)
	c.Data(http.StatusOK, "application/json", *encoded.Value)
}

func responseFailure(response transformer.CanonicalResponse) *apiError {
	for _, output := range response.Outputs {
		if output.FinishReason != transformer.FinishReasonError &&
			output.FinishReason != transformer.FinishReasonPause &&
			output.FinishReason != transformer.FinishReasonUnknown {
			continue
		}

		failure := &apiError{
			Message: "The upstream returned a non-terminal response.",
			Type:    "upstream_error",
			Code:    "upstream_response_not_terminal",
		}
		if output.FinishReason == transformer.FinishReasonError {
			failure.Message = "The upstream response failed."
			failure.Code = "upstream_response_failed"
		}
		if output.FinishReason == transformer.FinishReasonPause {
			failure.Message = "The upstream response requires continuation that Chat Completions cannot represent."
			failure.Code = "upstream_response_paused"
		}
		if output.ProviderReason != nil && *output.ProviderReason != "" {
			failure.Code = *output.ProviderReason
		}
		if output.FinishReason != transformer.FinishReasonError {
			return failure
		}
		var providerError struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		}
		if err := json.Unmarshal(response.Extensions["error"], &providerError); err == nil {
			if providerError.Message != "" {
				failure.Message = providerError.Message
			}
			if providerError.Code != "" {
				failure.Code = providerError.Code
			}
		}
		return failure
	}
	return nil
}

func (h *handler) decodeResponse(endpoint capabilities.Endpoint, body []byte) transformer.Result[transformer.CanonicalResponse] {
	if endpoint == capabilities.EndpointResponses {
		return h.transformer.DecodeResponsesResponse(body)
	}
	if endpoint == capabilities.EndpointMessages {
		return h.transformer.DecodeAnthropicResponse(body)
	}
	return transformer.Result[transformer.CanonicalResponse]{
		Diagnostics: []transformer.Diagnostic{{
			Severity: transformer.SeverityError,
			Code:     transformer.DiagnosticCode("unsupported_upstream_endpoint"),
			Message:  "unsupported upstream endpoint " + string(endpoint),
		}},
	}
}
