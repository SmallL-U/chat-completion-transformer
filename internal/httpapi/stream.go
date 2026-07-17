package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"

	"chat-completion-transformer/internal/capabilities"
	"chat-completion-transformer/pkg/transformer"

	"github.com/gin-gonic/gin"
)

const readBufferBytes = 32 << 10

type streamDecoder interface {
	Feed([]byte) transformer.Result[[]transformer.CanonicalEvent]
	Close() transformer.Result[[]transformer.CanonicalEvent]
}

func (h *handler) streamResponse(
	c *gin.Context,
	body io.ReadCloser,
	endpoint capabilities.Endpoint,
	modelAlias string,
	includeUsage bool,
	diagnostics []transformer.Diagnostic,
) {
	decoder, err := h.streamDecoder(endpoint)
	if err != nil {
		writeAPIError(c, http.StatusBadGateway, err.Error(), "upstream_error", nil, "unsupported_upstream_endpoint")
		return
	}

	encoder := h.transformer.CreateChatStreamEncoder(transformer.StreamEncoderOptions{FallbackModel: modelAlias, IncludeUsage: includeUsage})
	state := newStreamState(c, encoder, diagnostics)
	pump := newBodyPump(state.ctx, body)
	defer pump.stop()
	var total int64
	for {
		read, open := pump.read(state.ctx)
		if !open {
			if state.ctx.Err() != nil {
				return
			}
			h.finishStream(state, decoder)
			return
		}

		count := len(read.chunk)
		if count > 0 {
			if int64(count) > h.maxResponseBytes-total {
				state.diagnostics = appendDiagnostics(state.diagnostics, responseTooLargeDiagnostic(h.maxResponseBytes))
				state.abort(http.StatusBadGateway)
				return
			}
			total += int64(count)
			if !state.consume(decoder.Feed(read.chunk)) {
				state.abort(http.StatusBadGateway)
				return
			}
			if state.finished {
				state.complete()
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
		if state.ctx.Err() != nil {
			return
		}

		state.diagnostics = appendDiagnostics(state.diagnostics, upstreamReadDiagnostic())
		state.abort(http.StatusBadGateway)
		return
	}
}

func (h *handler) streamDecoder(endpoint capabilities.Endpoint) (streamDecoder, error) {
	if endpoint == capabilities.EndpointResponses {
		return h.transformer.CreateResponsesStreamDecoder(), nil
	}
	if endpoint == capabilities.EndpointMessages {
		return h.transformer.CreateAnthropicStreamDecoder(), nil
	}
	return nil, errors.New("unsupported upstream endpoint " + string(endpoint))
}

func (h *handler) finishStream(state *streamState, decoder streamDecoder) {
	if state.ctx.Err() != nil {
		return
	}
	if !state.consume(decoder.Close()) {
		state.abort(http.StatusBadGateway)
		return
	}
	if !state.consumeEncoded(state.encoder.Close()) {
		state.abort(http.StatusBadGateway)
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
	finished    bool
}

func newStreamState(c *gin.Context, encoder *transformer.ChatStreamEncoder, diagnostics []transformer.Diagnostic) *streamState {
	return &streamState{
		c:           c,
		ctx:         c.Request.Context(),
		encoder:     encoder,
		diagnostics: appendDiagnostics(nil, diagnostics...),
	}
}

func (s *streamState) consume(result transformer.Result[[]transformer.CanonicalEvent]) bool {
	s.diagnostics = appendDiagnostics(s.diagnostics, result.Diagnostics...)
	if !result.OK || result.Value == nil {
		return false
	}
	for _, event := range *result.Value {
		if s.ctx.Err() != nil {
			return false
		}
		if isNonTerminalFinish(event) {
			s.diagnostics = appendDiagnostics(s.diagnostics, transformer.Diagnostic{
				Severity: transformer.SeverityError,
				Code:     transformer.DiagnosticCode("upstream_response_not_terminal"),
				Message:  "The upstream stream did not finish with a Chat Completions terminal reason.",
			})
			return false
		}
		if !s.consumeEncoded(s.encoder.Encode(event)) {
			return false
		}
		if s.finished {
			return true
		}
	}
	return true
}

func isNonTerminalFinish(event transformer.CanonicalEvent) bool {
	if event.Type != transformer.EventFinish || event.Reason == nil {
		return false
	}
	return *event.Reason == transformer.FinishReasonPause ||
		*event.Reason == transformer.FinishReasonUnknown ||
		*event.Reason == transformer.FinishReasonError
}

func (s *streamState) consumeEncoded(result transformer.Result[[][]byte]) bool {
	s.diagnostics = appendDiagnostics(s.diagnostics, result.Diagnostics...)
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
			s.diagnostics = appendDiagnostics(s.diagnostics, responseWriteDiagnostic())
		}
		return false
	}
	s.c.Writer.Flush()
	if bytes.Equal(frame, []byte("data: [DONE]\n\n")) {
		s.finished = true
	}
	return s.ctx.Err() == nil
}

func (s *streamState) start() {
	header := s.c.Writer.Header()
	header.Set("Content-Type", "text/event-stream")
	header.Set("Cache-Control", "no-cache")
	header.Set("X-Accel-Buffering", "no")
	header.Add("Trailer", diagnosticsTrailer)
	s.started = true
}

func (s *streamState) abort(status int) {
	if s.ctx.Err() != nil {
		return
	}
	if !s.started {
		writeDiagnosticError(s.c, status, "upstream_error", s.diagnostics)
		return
	}
	s.writeErrorFrame()
	s.setTrailer()
}

func (s *streamState) writeErrorFrame() {
	payload, err := json.Marshal(apiErrorResponse{Error: diagnosticAPIError("upstream_error", s.diagnostics)})
	if err != nil {
		return
	}
	frame := make([]byte, 0, len(payload)+8)
	frame = append(frame, "data: "...)
	frame = append(frame, payload...)
	frame = append(frame, '\n', '\n')
	if _, err := s.c.Writer.Write(frame); err != nil {
		return
	}
	s.c.Writer.Flush()
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
	setDiagnosticsHeader(s.c.Writer.Header(), s.diagnostics)
}
