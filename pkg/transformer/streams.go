package transformer

import (
	"chat-completion-transformer/internal/anthropicmessages"
	"chat-completion-transformer/internal/canonical"
	"chat-completion-transformer/internal/chatcompletions"
	"chat-completion-transformer/internal/openairesponses"
)

type StreamEncoderOptions struct {
	Created       int64
	FallbackID    string
	FallbackModel string
	IncludeUsage  bool
}

type ResponsesStreamDecoder struct {
	inner *openairesponses.StreamDecoder
}

func (t *Transformer) CreateResponsesStreamDecoder() *ResponsesStreamDecoder {
	if t == nil {
		return &ResponsesStreamDecoder{}
	}
	return &ResponsesStreamDecoder{inner: openairesponses.NewStreamDecoder(t.maxSSEEventBytes)}
}

func (d *ResponsesStreamDecoder) Feed(chunk []byte) Result[[]CanonicalEvent] {
	if d == nil || d.inner == nil {
		return canonical.Failure[[]canonical.Event]([]canonical.Diagnostic{notInitializedDiagnostic()})
	}
	return d.inner.Feed(chunk)
}

func (d *ResponsesStreamDecoder) Close() Result[[]CanonicalEvent] {
	if d == nil || d.inner == nil {
		return canonical.Failure[[]canonical.Event]([]canonical.Diagnostic{notInitializedDiagnostic()})
	}
	return d.inner.Close()
}

type AnthropicStreamDecoder struct {
	inner *anthropicmessages.StreamDecoder
}

func (t *Transformer) CreateAnthropicStreamDecoder() *AnthropicStreamDecoder {
	if t == nil {
		return &AnthropicStreamDecoder{}
	}
	return &AnthropicStreamDecoder{inner: anthropicmessages.NewStreamDecoder(t.maxSSEEventBytes)}
}

func (d *AnthropicStreamDecoder) Feed(chunk []byte) Result[[]CanonicalEvent] {
	if d == nil || d.inner == nil {
		return canonical.Failure[[]canonical.Event]([]canonical.Diagnostic{notInitializedDiagnostic()})
	}
	return d.inner.Feed(chunk)
}

func (d *AnthropicStreamDecoder) Close() Result[[]CanonicalEvent] {
	if d == nil || d.inner == nil {
		return canonical.Failure[[]canonical.Event]([]canonical.Diagnostic{notInitializedDiagnostic()})
	}
	return d.inner.Close()
}

type ChatStreamEncoder struct {
	inner *chatcompletions.StreamEncoder
}

func (t *Transformer) CreateChatStreamEncoder(options StreamEncoderOptions) *ChatStreamEncoder {
	if t == nil {
		return &ChatStreamEncoder{}
	}
	return &ChatStreamEncoder{inner: chatcompletions.NewStreamEncoder(chatcompletions.StreamEncodeOptions{
		Mode:          t.mode,
		Created:       options.Created,
		FallbackID:    options.FallbackID,
		FallbackModel: options.FallbackModel,
		IncludeUsage:  options.IncludeUsage,
	})}
}

func (e *ChatStreamEncoder) Encode(event CanonicalEvent) Result[[][]byte] {
	if e == nil || e.inner == nil {
		return canonical.Failure[[][]byte]([]canonical.Diagnostic{notInitializedDiagnostic()})
	}
	return e.inner.Encode(event)
}

func (e *ChatStreamEncoder) Close() Result[[][]byte] {
	if e == nil || e.inner == nil {
		return canonical.Failure[[][]byte]([]canonical.Diagnostic{notInitializedDiagnostic()})
	}
	return e.inner.Close()
}
