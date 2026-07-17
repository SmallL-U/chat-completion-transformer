package httpapi

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"log"
	"mime"
	"net"
	"net/http"
	"runtime/debug"
	"strconv"
	"strings"

	"chat-completion-transformer/pkg/transformer"

	"github.com/gin-gonic/gin"
)

const diagnosticsTrailer = "X-Transformer-Diagnostics"

const (
	maxGatewayDiagnostics = 8
	maxDiagnosticMessage  = 256
	maxDiagnosticPath     = 128
	maxDiagnosticCode     = 64
	maxDiagnosticsHeader  = 4096
)

var errBodyTooLarge = errors.New("body exceeds the configured size limit")

type httpBody interface {
	io.Reader
	io.Closer
}

type apiErrorResponse struct {
	Error apiError `json:"error"`
}

type apiError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   any    `json:"param"`
	Code    any    `json:"code"`
}

func (h *handler) readRequestBody(c *gin.Context) ([]byte, bool) {
	if c.Request.ContentLength > h.maxBodyBytes {
		writeAPIError(c, http.StatusRequestEntityTooLarge, requestTooLargeMessage(h.maxBodyBytes), "invalid_request_error", nil, "request_too_large")
		return nil, false
	}

	body, err := readBody(c.Request.Context(), c.Request.Body, h.maxBodyBytes)
	if err == nil {
		return body, true
	}
	if c.Request.Context().Err() != nil {
		return nil, false
	}
	if errors.Is(err, errBodyTooLarge) {
		writeAPIError(c, http.StatusRequestEntityTooLarge, requestTooLargeMessage(h.maxBodyBytes), "invalid_request_error", nil, "request_too_large")
		return nil, false
	}

	writeAPIError(c, http.StatusBadRequest, "Unable to read the request body.", "invalid_request_error", nil, "request_body_read_failed")
	return nil, false
}

type bodyRead struct {
	chunk []byte
	err   error
}

type bodyPump struct {
	results  <-chan bodyRead
	requests chan<- struct{}
	cancel   context.CancelFunc
	body     io.ReadCloser
	done     <-chan struct{}
}

func newBodyPump(ctx context.Context, body io.ReadCloser) *bodyPump {
	pumpContext, cancel := context.WithCancel(ctx)
	results := make(chan bodyRead)
	requests := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(results)
		defer close(done)
		buffer := make([]byte, readBufferBytes)
		for {
			select {
			case <-pumpContext.Done():
				return
			case <-requests:
			}
			count, err := body.Read(buffer)
			read := bodyRead{err: err}
			if count > 0 {
				read.chunk = append([]byte(nil), buffer[:count]...)
			}
			select {
			case results <- read:
			case <-pumpContext.Done():
				return
			}
			if err != nil {
				return
			}
		}
	}()
	return &bodyPump{results: results, requests: requests, cancel: cancel, body: body, done: done}
}

func (pump *bodyPump) read(ctx context.Context) (bodyRead, bool) {
	select {
	case <-ctx.Done():
		return bodyRead{}, false
	case pump.requests <- struct{}{}:
	}

	select {
	case <-ctx.Done():
		return bodyRead{}, false
	case read, open := <-pump.results:
		return read, open
	}
}

func (pump *bodyPump) stop() {
	pump.cancel()
	_ = pump.body.Close()
	<-pump.done
}

func readBody(ctx context.Context, body io.ReadCloser, limit int64) ([]byte, error) {
	pump := newBodyPump(ctx, body)
	defer pump.stop()

	var result []byte
	for {
		read, open := pump.read(ctx)
		if !open {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			return result, nil
		}
		if int64(len(read.chunk)) > limit-int64(len(result)) {
			return nil, errBodyTooLarge
		}
		result = append(result, read.chunk...)
		if read.err == nil {
			continue
		}
		if errors.Is(read.err, io.EOF) {
			return result, nil
		}
		return nil, read.err
	}
}

func writeAPIError(c *gin.Context, status int, message, errorType string, param, code any) {
	if c.Request.Context().Err() != nil {
		return
	}
	c.JSON(status, apiErrorResponse{Error: apiError{
		Message: message,
		Type:    errorType,
		Param:   param,
		Code:    code,
	}})
}

func recoveryMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		defer func() {
			recovered := recover()
			if recovered == nil {
				return
			}
			log.Printf("recovered HTTP panic (%T):\n%s", recovered, debug.Stack())
			c.Abort()
			if c.Writer.Written() {
				return
			}
			writeAPIError(c, http.StatusInternalServerError, "The server encountered an internal error.", "server_error", nil, "internal_server_error")
		}()
		c.Next()
	}
}

func bearerAuthorization(value string) (string, bool) {
	parts := strings.Fields(value)
	if len(parts) != 2 || !strings.EqualFold(parts[0], "Bearer") || parts[1] == "" {
		return "", false
	}
	return "Bearer " + parts[1], true
}

func writeDiagnosticError(c *gin.Context, status int, errorType string, diagnostics []transformer.Diagnostic) {
	errorValue := diagnosticAPIError(errorType, diagnostics)
	writeAPIError(c, status, errorValue.Message, errorValue.Type, errorValue.Param, errorValue.Code)
}

func diagnosticAPIError(errorType string, diagnostics []transformer.Diagnostic) apiError {
	message := "The request could not be processed."
	var param any
	var code any = "invalid_request"
	for _, diagnostic := range diagnostics {
		if diagnostic.Severity != transformer.SeverityError {
			continue
		}
		diagnostic = sanitizeDiagnostic(diagnostic)
		message = diagnostic.Message
		code = string(diagnostic.Code)
		if diagnostic.Path != nil {
			param = *diagnostic.Path
		}
		break
	}
	return apiError{Message: message, Type: errorType, Param: param, Code: code}
}

func writeUpstreamRequestError(c *gin.Context, err error) {
	ctx := c.Request.Context()
	if ctx.Err() != nil || errors.Is(err, context.Canceled) {
		return
	}
	if errors.Is(err, context.DeadlineExceeded) || isTimeout(err) {
		writeAPIError(c, http.StatusGatewayTimeout, "The upstream request timed out.", "upstream_error", nil, "upstream_timeout")
		return
	}
	writeAPIError(c, http.StatusBadGateway, "The upstream request failed.", "upstream_error", nil, "upstream_request_failed")
}

func isTimeout(err error) bool {
	var networkError net.Error
	return errors.As(err, &networkError) && networkError.Timeout()
}

func (h *handler) writeUpstreamError(c *gin.Context, response *http.Response) {
	body, err := h.readUpstreamBody(c.Request.Context(), response.Body)
	if err != nil {
		if c.Request.Context().Err() != nil {
			return
		}
		writeUpstreamBodyError(c, err, h.maxResponseBytes)
		return
	}

	parsed := parseUpstreamError(response.StatusCode, body)
	writeAPIError(c, response.StatusCode, parsed.Message, parsed.Type, parsed.Param, parsed.Code)
}

func writeUnexpectedUpstreamStatus(c *gin.Context) {
	writeAPIError(c, http.StatusBadGateway, "The upstream returned an unexpected HTTP status.", "upstream_error", nil, "upstream_invalid_status")
}

func isEventStream(contentType string) bool {
	mediaType, _, err := mime.ParseMediaType(contentType)
	return err == nil && strings.EqualFold(mediaType, "text/event-stream")
}

func isJSON(contentType string) bool {
	mediaType, _, err := mime.ParseMediaType(contentType)
	if err != nil {
		return false
	}
	mediaType = strings.ToLower(mediaType)
	return mediaType == "application/json" || strings.HasPrefix(mediaType, "application/") && strings.HasSuffix(mediaType, "+json")
}

func writeUnexpectedStreamResponse(c *gin.Context) {
	writeAPIError(c, http.StatusBadGateway, "The upstream did not return an event stream.", "upstream_error", nil, "upstream_invalid_content_type")
}

func writeUnexpectedJSONResponse(c *gin.Context) {
	writeAPIError(c, http.StatusBadGateway, "The upstream did not return a JSON response.", "upstream_error", nil, "upstream_invalid_content_type")
}

func (h *handler) readUpstreamBody(ctx context.Context, body io.ReadCloser) ([]byte, error) {
	readContext, cancel := context.WithTimeout(ctx, h.responseTimeout)
	defer cancel()
	return readBody(readContext, body, h.maxResponseBytes)
}

func parseUpstreamError(status int, body []byte) apiError {
	result := apiError{
		Message: http.StatusText(status),
		Type:    "upstream_error",
		Code:    "upstream_error",
	}

	var envelope struct {
		Type    string          `json:"type"`
		Message string          `json:"message"`
		Error   json.RawMessage `json:"error"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return result
	}
	if len(envelope.Error) == 0 {
		if envelope.Message != "" {
			result.Message = envelope.Message
		}
		if envelope.Type != "" && envelope.Type != "error" {
			result.Type = envelope.Type
		}
		return result
	}

	var nested struct {
		Message string          `json:"message"`
		Type    string          `json:"type"`
		Param   json.RawMessage `json:"param"`
		Code    json.RawMessage `json:"code"`
	}
	if err := json.Unmarshal(envelope.Error, &nested); err != nil {
		return result
	}
	if nested.Message != "" {
		result.Message = nested.Message
	}
	if nested.Type != "" {
		result.Type = nested.Type
	}
	if param, valid := nullableString(nested.Param); valid {
		result.Param = param
	}
	if code, valid := nullableString(nested.Code); len(nested.Code) > 0 && valid {
		result.Code = code
	} else if len(nested.Code) == 0 && nested.Type != "" {
		result.Code = nested.Type
	}
	return result
}

func nullableString(raw json.RawMessage) (any, bool) {
	if len(raw) == 0 {
		return nil, false
	}
	if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, true
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, false
	}
	return value, true
}

func writeUpstreamBodyError(c *gin.Context, err error, limit int64) {
	if errors.Is(err, context.DeadlineExceeded) {
		writeAPIError(c, http.StatusGatewayTimeout, "The upstream response body timed out.", "upstream_error", nil, "upstream_response_timeout")
		return
	}
	if errors.Is(err, errBodyTooLarge) {
		writeAPIError(c, http.StatusBadGateway, "The upstream response exceeds the configured limit of "+strconv.FormatInt(limit, 10)+" bytes.", "upstream_error", nil, "upstream_response_too_large")
		return
	}
	writeAPIError(c, http.StatusBadGateway, "Unable to read the upstream response.", "upstream_error", nil, "upstream_response_read_failed")
}

func requestTooLargeMessage(limit int64) string {
	return "The request body exceeds the configured limit of " + strconv.FormatInt(limit, 10) + " bytes."
}

func responseTooLargeDiagnostic(limit int64) transformer.Diagnostic {
	return transformer.Diagnostic{
		Severity: transformer.SeverityError,
		Code:     transformer.DiagnosticCode("upstream_response_too_large"),
		Message:  "The upstream response exceeds the configured limit of " + strconv.FormatInt(limit, 10) + " bytes.",
	}
}

func upstreamReadDiagnostic() transformer.Diagnostic {
	return transformer.Diagnostic{
		Severity: transformer.SeverityError,
		Code:     transformer.DiagnosticCode("upstream_response_read_failed"),
		Message:  "Unable to read the upstream response.",
	}
}

func responseWriteDiagnostic() transformer.Diagnostic {
	return transformer.Diagnostic{
		Severity: transformer.SeverityError,
		Code:     transformer.DiagnosticCode("response_write_failed"),
		Message:  "Unable to write the response.",
	}
}

func appendDiagnostics(prefix []transformer.Diagnostic, suffix ...transformer.Diagnostic) []transformer.Diagnostic {
	diagnostics := make([]transformer.Diagnostic, 0, maxGatewayDiagnostics)
	for _, diagnostic := range prefix {
		diagnostics = appendBoundedDiagnostic(diagnostics, diagnostic)
	}
	for _, diagnostic := range suffix {
		diagnostics = appendBoundedDiagnostic(diagnostics, diagnostic)
	}
	return diagnostics
}

func appendBoundedDiagnostic(diagnostics []transformer.Diagnostic, diagnostic transformer.Diagnostic) []transformer.Diagnostic {
	diagnostic = sanitizeDiagnostic(diagnostic)
	if len(diagnostics) < maxGatewayDiagnostics {
		return append(diagnostics, diagnostic)
	}
	if diagnostic.Severity != transformer.SeverityError {
		return diagnostics
	}
	for index := len(diagnostics) - 1; index >= 0; index-- {
		if diagnostics[index].Severity == transformer.SeverityError {
			continue
		}
		diagnostics[index] = diagnostic
		return diagnostics
	}
	diagnostics[len(diagnostics)-1] = diagnostic
	return diagnostics
}

func sanitizeDiagnostic(diagnostic transformer.Diagnostic) transformer.Diagnostic {
	diagnostic.Code = transformer.DiagnosticCode(truncateText(string(diagnostic.Code), maxDiagnosticCode))
	diagnostic.Message = truncateText(diagnostic.Message, maxDiagnosticMessage)
	if diagnostic.Path != nil {
		path := truncateText(*diagnostic.Path, maxDiagnosticPath)
		diagnostic.Path = &path
	}
	diagnostic.SourceValue = nil
	return diagnostic
}

func truncateText(value string, limit int) string {
	characters := []rune(value)
	if len(characters) <= limit {
		return value
	}
	return string(characters[:limit]) + "…"
}

func setDiagnosticsHeader(header http.Header, diagnostics []transformer.Diagnostic) {
	diagnostics = appendDiagnostics(nil, diagnostics...)
	for len(diagnostics) > 0 {
		encoded, err := json.Marshal(diagnostics)
		if err != nil {
			return
		}
		value := base64.RawURLEncoding.EncodeToString(encoded)
		if len(value) <= maxDiagnosticsHeader {
			header.Set(diagnosticsTrailer, value)
			return
		}
		if len(diagnostics) == 1 {
			return
		}
		diagnostics = dropDiagnosticForHeader(diagnostics)
	}
}

func dropDiagnosticForHeader(diagnostics []transformer.Diagnostic) []transformer.Diagnostic {
	index := len(diagnostics) - 1
	for candidate := len(diagnostics) - 1; candidate >= 0; candidate-- {
		if diagnostics[candidate].Severity != transformer.SeverityError {
			index = candidate
			break
		}
	}
	return append(diagnostics[:index], diagnostics[index+1:]...)
}

func copyUpstreamHeaders(target, source http.Header) {
	for name, values := range source {
		lower := strings.ToLower(name)
		if lower != "retry-after" && lower != "request-id" && lower != "x-request-id" &&
			!strings.HasPrefix(lower, "x-ratelimit-") && !strings.HasPrefix(lower, "anthropic-ratelimit-") {
			continue
		}
		for _, value := range values {
			target.Add(name, value)
		}
	}
}
