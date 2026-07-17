// Package httpapi exposes the protocol transformer over HTTP.
package httpapi

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strconv"

	"chat-completion-transformer/pkg/transformer"

	"github.com/gin-gonic/gin"
)

const (
	diagnosticsTrailer = "X-Transformer-Diagnostics"
	readBufferBytes    = 32 << 10
)

var errBodyTooLarge = errors.New("request body exceeds the configured size limit")

// Limits bounds buffered JSON requests and streamed provider responses.
type Limits struct {
	MaxBodyBytes   int64
	MaxStreamBytes int64
}

type handler struct {
	transformer    *transformer.Transformer
	maxBodyBytes   int64
	maxStreamBytes int64
}

type streamDecoder interface {
	Feed([]byte) transformer.Result[[]transformer.CanonicalEvent]
	Close() transformer.Result[[]transformer.CanonicalEvent]
}

// NewRouter creates a Gin router whose handlers keep all work tied to the
// incoming request context.
func NewRouter(service *transformer.Transformer, limits Limits) (*gin.Engine, error) {
	if service == nil {
		return nil, errors.New("transformer is required")
	}
	if limits.MaxBodyBytes <= 0 {
		return nil, errors.New("max body bytes must be positive")
	}
	if limits.MaxStreamBytes <= 0 {
		return nil, errors.New("max stream bytes must be positive")
	}

	h := &handler{
		transformer:    service,
		maxBodyBytes:   limits.MaxBodyBytes,
		maxStreamBytes: limits.MaxStreamBytes,
	}
	router := gin.New()
	router.Use(gin.Recovery())
	router.GET("/healthz", h.health)

	api := router.Group("/v1/transform")
	api.POST("/chat-completions/to/openai-responses", func(c *gin.Context) {
		h.chatRequestToProvider(c, service.EncodeResponsesRequest)
	})
	api.POST("/chat-completions/to/anthropic-messages", func(c *gin.Context) {
		h.chatRequestToProvider(c, service.EncodeAnthropicRequest)
	})
	api.POST("/openai-responses/to/chat-completions", func(c *gin.Context) {
		h.providerResponseToChat(c, service.DecodeResponsesResponse)
	})
	api.POST("/anthropic-messages/to/chat-completions", func(c *gin.Context) {
		h.providerResponseToChat(c, service.DecodeAnthropicResponse)
	})
	api.POST("/openai-responses/sse/to/chat-completions", func(c *gin.Context) {
		h.providerStreamToChat(c, service.CreateResponsesStreamDecoder())
	})
	api.POST("/anthropic-messages/sse/to/chat-completions", func(c *gin.Context) {
		h.providerStreamToChat(c, service.CreateAnthropicStreamDecoder())
	})

	return router, nil
}

func (h *handler) health(c *gin.Context) {
	if c.Request.Context().Err() != nil {
		return
	}
	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

func (h *handler) chatRequestToProvider(
	c *gin.Context,
	encode func(context.Context, transformer.CanonicalRequest) (transformer.Result[json.RawMessage], error),
) {
	ctx := c.Request.Context()
	body, ok := h.readJSONBody(c)
	if !ok {
		return
	}
	if ctx.Err() != nil {
		return
	}

	decoded := h.transformer.DecodeChatRequest(body)
	if ctx.Err() != nil {
		return
	}
	if !decoded.OK || decoded.Value == nil {
		c.JSON(http.StatusBadRequest, decoded)
		return
	}

	encoded, err := encode(ctx, *decoded.Value)
	if err != nil {
		if ctx.Err() != nil {
			return
		}
		diagnostics := appendDiagnostics(decoded.Diagnostics, internalDiagnostic(err))
		c.JSON(http.StatusInternalServerError, failedResult(diagnostics))
		return
	}
	if ctx.Err() != nil {
		return
	}

	result := mergeDiagnostics(encoded, decoded.Diagnostics)
	status := http.StatusOK
	if !result.OK {
		status = http.StatusUnprocessableEntity
	}
	c.JSON(status, result)
}

func (h *handler) providerResponseToChat(
	c *gin.Context,
	decode func([]byte) transformer.Result[transformer.CanonicalResponse],
) {
	ctx := c.Request.Context()
	body, ok := h.readJSONBody(c)
	if !ok {
		return
	}
	if ctx.Err() != nil {
		return
	}

	decoded := decode(body)
	if ctx.Err() != nil {
		return
	}
	if !decoded.OK || decoded.Value == nil {
		c.JSON(http.StatusBadRequest, decoded)
		return
	}

	encoded := h.transformer.EncodeChatResponse(*decoded.Value)
	if ctx.Err() != nil {
		return
	}
	result := mergeDiagnostics(encoded, decoded.Diagnostics)
	status := http.StatusOK
	if !result.OK {
		status = http.StatusUnprocessableEntity
	}
	c.JSON(status, result)
}

func (h *handler) readJSONBody(c *gin.Context) ([]byte, bool) {
	ctx := c.Request.Context()
	if c.Request.ContentLength > h.maxBodyBytes {
		c.JSON(http.StatusRequestEntityTooLarge, failedResult([]transformer.Diagnostic{bodyTooLargeDiagnostic(h.maxBodyBytes)}))
		return nil, false
	}

	body, err := readBody(ctx, c.Request.Body, h.maxBodyBytes)
	if err == nil {
		return body, true
	}
	if ctx.Err() != nil {
		return nil, false
	}
	if errors.Is(err, errBodyTooLarge) {
		c.JSON(http.StatusRequestEntityTooLarge, failedResult([]transformer.Diagnostic{bodyTooLargeDiagnostic(h.maxBodyBytes)}))
		return nil, false
	}

	c.JSON(http.StatusBadRequest, failedResult([]transformer.Diagnostic{readDiagnostic(err)}))
	return nil, false
}

func (h *handler) providerStreamToChat(c *gin.Context, decoder streamDecoder) {
	ctx := c.Request.Context()
	state := newStreamState(c, h.transformer.CreateChatStreamEncoder(transformer.StreamEncoderOptions{}))
	if c.Request.ContentLength > h.maxStreamBytes {
		state.diagnostics = append(state.diagnostics, bodyTooLargeDiagnostic(h.maxStreamBytes))
		state.abort(http.StatusRequestEntityTooLarge)
		return
	}
	if err := http.NewResponseController(c.Writer).EnableFullDuplex(); err != nil {
		state.diagnostics = append(state.diagnostics, fullDuplexDiagnostic(err))
		state.abort(http.StatusInternalServerError)
		return
	}

	pump := newBodyPump(ctx, c.Request.Body)
	defer pump.stop()

	var total int64
	for {
		select {
		case <-ctx.Done():
			return
		case read, open := <-pump.results:
			if !open {
				h.finishStream(state, decoder)
				return
			}
			if len(read.chunk) > 0 {
				if int64(len(read.chunk)) > h.maxStreamBytes-total {
					state.diagnostics = append(state.diagnostics, bodyTooLargeDiagnostic(h.maxStreamBytes))
					state.abort(http.StatusRequestEntityTooLarge)
					return
				}
				total += int64(len(read.chunk))
				if !state.consume(decoder.Feed(read.chunk)) {
					state.abort(http.StatusBadRequest)
					return
				}
			}
			if read.err == nil {
				continue
			}
			if errors.Is(read.err, io.EOF) {
				h.finishStream(state, decoder)
				return
			}
			if ctx.Err() != nil {
				return
			}

			state.diagnostics = append(state.diagnostics, readDiagnostic(read.err))
			state.abort(http.StatusBadRequest)
			return
		}
	}
}

func (h *handler) finishStream(state *streamState, decoder streamDecoder) {
	if state.ctx.Err() != nil {
		return
	}
	if !state.consume(decoder.Close()) {
		state.abort(http.StatusBadRequest)
		return
	}
	if !state.consumeEncoded(state.encoder.Close()) {
		state.abort(http.StatusUnprocessableEntity)
		return
	}
	state.complete()
}

type streamState struct {
	c           *gin.Context
	ctx         context.Context
	encoder     *transformer.ChatStreamEncoder
	diagnostics []transformer.Diagnostic
	started     bool
}

func newStreamState(c *gin.Context, encoder *transformer.ChatStreamEncoder) *streamState {
	return &streamState{
		c:           c,
		ctx:         c.Request.Context(),
		encoder:     encoder,
		diagnostics: make([]transformer.Diagnostic, 0),
	}
}

func (s *streamState) consume(result transformer.Result[[]transformer.CanonicalEvent]) bool {
	s.diagnostics = append(s.diagnostics, result.Diagnostics...)
	if !result.OK || result.Value == nil {
		return false
	}
	for _, event := range *result.Value {
		if s.ctx.Err() != nil {
			return false
		}
		if !s.consumeEncoded(s.encoder.Encode(event)) {
			return false
		}
	}
	return true
}

func (s *streamState) consumeEncoded(result transformer.Result[[][]byte]) bool {
	s.diagnostics = append(s.diagnostics, result.Diagnostics...)
	if !result.OK || result.Value == nil {
		return false
	}
	for _, frame := range *result.Value {
		if !s.writeFrame(frame) {
			return false
		}
	}
	return true
}

func (s *streamState) writeFrame(frame []byte) bool {
	if s.ctx.Err() != nil {
		return false
	}
	if len(frame) == 0 {
		return true
	}
	if !s.started {
		s.start()
	}
	if _, err := s.c.Writer.Write(frame); err != nil {
		if s.ctx.Err() == nil {
			s.diagnostics = append(s.diagnostics, writeDiagnostic(err))
		}
		return false
	}
	s.c.Writer.Flush()
	return s.ctx.Err() == nil
}

func (s *streamState) start() {
	header := s.c.Writer.Header()
	header.Set("Content-Type", "text/event-stream")
	header.Set("Cache-Control", "no-cache")
	header.Add("Trailer", diagnosticsTrailer)
	s.started = true
}

func (s *streamState) abort(status int) {
	if s.ctx.Err() != nil {
		return
	}
	if !s.started {
		s.c.JSON(status, failedResult(s.diagnostics))
		return
	}
	s.setTrailer()
}

func (s *streamState) complete() {
	if s.ctx.Err() != nil {
		return
	}
	if !s.started {
		s.start()
		s.c.Writer.WriteHeaderNow()
	}
	s.setTrailer()
}

func (s *streamState) setTrailer() {
	encoded, err := json.Marshal(s.diagnostics)
	if err != nil {
		return
	}
	s.c.Writer.Header().Set(diagnosticsTrailer, base64.RawURLEncoding.EncodeToString(encoded))
}

type bodyRead struct {
	chunk []byte
	err   error
}

type bodyPump struct {
	results <-chan bodyRead
	cancel  context.CancelFunc
	body    io.ReadCloser
	done    <-chan struct{}
}

func newBodyPump(ctx context.Context, body io.ReadCloser) *bodyPump {
	pumpContext, cancel := context.WithCancel(ctx)
	results := make(chan bodyRead)
	done := make(chan struct{})
	go func() {
		defer close(results)
		defer close(done)
		buffer := make([]byte, readBufferBytes)
		for {
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

	return &bodyPump{results: results, cancel: cancel, body: body, done: done}
}

func (p *bodyPump) stop() {
	p.cancel()
	_ = p.body.Close()
	<-p.done
}

func readBody(ctx context.Context, body io.ReadCloser, limit int64) ([]byte, error) {
	pump := newBodyPump(ctx, body)
	defer pump.stop()

	var result []byte
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case read, open := <-pump.results:
			if !open {
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
}

func mergeDiagnostics[T any](result transformer.Result[T], prefix []transformer.Diagnostic) transformer.Result[T] {
	result.Diagnostics = appendDiagnostics(prefix, result.Diagnostics...)
	result.Lossless = result.OK && len(result.Diagnostics) == 0
	return result
}

func appendDiagnostics(prefix []transformer.Diagnostic, suffix ...transformer.Diagnostic) []transformer.Diagnostic {
	diagnostics := make([]transformer.Diagnostic, 0, len(prefix)+len(suffix))
	diagnostics = append(diagnostics, prefix...)
	diagnostics = append(diagnostics, suffix...)
	return diagnostics
}

func failedResult(diagnostics []transformer.Diagnostic) transformer.Result[json.RawMessage] {
	if diagnostics == nil {
		diagnostics = make([]transformer.Diagnostic, 0)
	}
	return transformer.Result[json.RawMessage]{Diagnostics: diagnostics, OK: false}
}

func bodyTooLargeDiagnostic(limit int64) transformer.Diagnostic {
	return transformer.Diagnostic{
		Severity: transformer.SeverityError,
		Code:     transformer.DiagnosticCode("request_body_too_large"),
		Message:  "request body exceeds the configured limit of " + strconv.FormatInt(limit, 10) + " bytes",
	}
}

func readDiagnostic(err error) transformer.Diagnostic {
	return transformer.Diagnostic{
		Severity: transformer.SeverityError,
		Code:     transformer.DiagnosticCode("request_body_read_failed"),
		Message:  err.Error(),
	}
}

func writeDiagnostic(err error) transformer.Diagnostic {
	return transformer.Diagnostic{
		Severity: transformer.SeverityError,
		Code:     transformer.DiagnosticCode("response_write_failed"),
		Message:  err.Error(),
	}
}

func fullDuplexDiagnostic(err error) transformer.Diagnostic {
	return transformer.Diagnostic{
		Severity: transformer.SeverityError,
		Code:     transformer.DiagnosticCode("full_duplex_unavailable"),
		Message:  "enable full-duplex streaming: " + err.Error(),
	}
}

func internalDiagnostic(err error) transformer.Diagnostic {
	return transformer.Diagnostic{
		Severity: transformer.SeverityError,
		Code:     transformer.DiagnosticCode("internal_error"),
		Message:  err.Error(),
	}
}
