package chatcompletions

import (
	"bytes"
	"encoding/json"
	"fmt"
	"time"

	"chat-completion-transformer/internal/canonical"
)

const (
	diagnosticInvalidStreamEvent canonical.DiagnosticCode = "invalid_stream_event"
	diagnosticStreamState        canonical.DiagnosticCode = "invalid_stream_state"
	diagnosticStreamTruncated    canonical.DiagnosticCode = "stream_truncated"
)

type StreamEncodeOptions struct {
	Mode          canonical.Mode
	Created       int64
	FallbackID    string
	FallbackModel string
}

// StreamEncoder converts one canonical event stream into Chat Completions SSE.
// It is stateful and must not be shared between upstream responses.
type StreamEncoder struct {
	options      StreamEncodeOptions
	id           string
	model        string
	roleSent     bool
	finished     bool
	toolIndexes  map[int]int
	openTools    map[int]bool
	nextTool     int
	pendingUsage *canonical.Usage
}

func NewStreamEncoder(options StreamEncodeOptions) *StreamEncoder {
	if options.Created == 0 {
		options.Created = time.Now().Unix()
	}
	return &StreamEncoder{
		options:     options,
		id:          options.FallbackID,
		model:       options.FallbackModel,
		toolIndexes: make(map[int]int),
		openTools:   make(map[int]bool),
	}
}

// Encode returns zero or more complete SSE frames for one canonical event.
func (e *StreamEncoder) Encode(event canonical.Event) canonical.Result[[][]byte] {
	if e.finished {
		return canonical.Failure[[][]byte]([]canonical.Diagnostic{
			diagnostic(canonical.SeverityError, diagnosticStreamState, "received an event after stream finish", "", event.Value),
		})
	}

	frames := make([][]byte, 0, 3)
	diagnostics := make([]canonical.Diagnostic, 0)
	switch event.Type {
	case canonical.EventResponseStart:
		if e.roleSent {
			diagnostics = append(diagnostics, diagnostic(canonical.SeverityError, diagnosticStreamState, "received response_start after Chat output began", "", event.Value))
			break
		}
		id := e.id
		if event.ID != "" {
			id = event.ID
		}
		model := e.model
		if event.Model != nil {
			model = *event.Model
		}
		created := e.options.Created
		if event.CreatedAt != nil {
			created = *event.CreatedAt
		}
		if id == "" || model == "" {
			diagnostics = append(diagnostics, diagnostic(canonical.SeverityError, diagnosticStreamState, "stream response ID and model must be known at response_start", "", event.Value))
			break
		}
		e.id = id
		e.model = model
		e.options.Created = created
		frames = append(frames, e.ensureRole(&diagnostics)...)
	case canonical.EventTextDelta:
		frames = append(frames, e.ensureRole(&diagnostics)...)
		frames = append(frames, e.chunkFrame(map[string]any{"content": event.Delta}, nil, nil))
	case canonical.EventRefusalDelta:
		frames = append(frames, e.ensureRole(&diagnostics)...)
		frames = append(frames, e.chunkFrame(map[string]any{"refusal": event.Delta}, nil, nil))
	case canonical.EventToolCallStart:
		if event.OutputIndex == nil || event.CallID == "" || event.Name == "" {
			diagnostics = append(diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidStreamEvent, "tool_call_start requires output_index, call_id, and name", "", event.Value))
			break
		}
		providerIndex := *event.OutputIndex
		if providerIndex < 0 {
			diagnostics = append(diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidStreamEvent, "tool_call_start output_index must not be negative", "output_index", event.Value))
			break
		}
		if _, exists := e.toolIndexes[providerIndex]; exists {
			diagnostics = append(diagnostics, diagnostic(canonical.SeverityError, diagnosticStreamState, fmt.Sprintf("duplicate tool start for provider output %d", providerIndex), "output_index", nil))
			break
		}
		frames = append(frames, e.ensureRole(&diagnostics)...)
		if canonical.HasErrors(diagnostics) {
			break
		}
		chatIndex := e.nextTool
		e.nextTool++
		e.toolIndexes[providerIndex] = chatIndex
		e.openTools[providerIndex] = true
		frames = append(frames, e.chunkFrame(map[string]any{
			"tool_calls": []any{map[string]any{
				"index": chatIndex,
				"id":    event.CallID,
				"type":  "function",
				"function": map[string]any{
					"name":      event.Name,
					"arguments": "",
				},
			}},
		}, nil, nil))
	case canonical.EventToolArgumentsDelta:
		if event.OutputIndex == nil {
			diagnostics = append(diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidStreamEvent, "tool_arguments_delta requires output_index", "", event.Value))
			break
		}
		chatIndex, exists := e.toolIndexes[*event.OutputIndex]
		if !exists || !e.openTools[*event.OutputIndex] {
			diagnostics = append(diagnostics, diagnostic(canonical.SeverityError, diagnosticStreamState, "received tool arguments outside an open tool call", "output_index", nil))
			break
		}
		frames = append(frames, e.chunkFrame(map[string]any{
			"tool_calls": []any{map[string]any{
				"index":    chatIndex,
				"function": map[string]any{"arguments": event.Delta},
			}},
		}, nil, nil))
	case canonical.EventToolCallEnd:
		if event.OutputIndex == nil || !e.openTools[*event.OutputIndex] {
			diagnostics = append(diagnostics, diagnostic(canonical.SeverityError, diagnosticStreamState, "tool_call_end does not match an open tool call", "output_index", nil))
			break
		}
		e.openTools[*event.OutputIndex] = false
	case canonical.EventUsage:
		if event.Usage == nil {
			diagnostics = append(diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidStreamEvent, "usage event requires usage", "", event.Value))
			break
		}
		if len(event.Usage.Extensions) > 0 {
			diagnostics = append(diagnostics, lossyDiagnostic(e.options.Mode, diagnosticResponseExtensionLossy, "provider usage extensions cannot be represented by Chat Completions", "usage.extensions", mustMarshal(event.Usage.Extensions)))
		}
		if canonical.HasErrors(diagnostics) {
			break
		}
		e.pendingUsage = mergeUsage(e.pendingUsage, *event.Usage)
	case canonical.EventFinish:
		if lenOpenTools(e.openTools) > 0 {
			diagnostics = append(diagnostics, diagnostic(canonical.SeverityError, diagnosticStreamState, "stream finished before every tool call ended", "", nil))
			break
		}
		if event.Reason == nil {
			diagnostics = append(diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidStreamEvent, "finish event requires a reason", "reason", nil))
			break
		}
		output := canonical.Output{Index: 0, FinishReason: *event.Reason, ProviderReason: event.ProviderReason}
		finishReason, finishDiagnostics := encodeFinishReason(output, e.options.Mode)
		diagnostics = append(diagnostics, finishDiagnostics...)
		if hasMeaningfulJSON(event.Value) {
			diagnostics = append(diagnostics, lossyDiagnostic(e.options.Mode, diagnosticResponseExtensionLossy, "provider finish details cannot be represented by Chat Completions SSE", "finish.extensions", event.Value))
		}
		if canonical.HasErrors(diagnostics) {
			break
		}
		frames = append(frames, e.ensureRole(&diagnostics)...)
		if canonical.HasErrors(diagnostics) {
			break
		}
		frames = append(frames, e.chunkFrame(map[string]any{}, finishReason, nil))
		if e.pendingUsage != nil {
			frames = append(frames, e.usageFrame(*e.pendingUsage))
		}
		frames = append(frames, []byte("data: [DONE]\n\n"))
		e.finished = true
	case canonical.EventError:
		diagnostics = append(diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidStreamEvent, "upstream stream reported an error", "", event.Error))
	case canonical.EventOpaque:
		diagnostics = append(diagnostics, lossyDiagnostic(e.options.Mode, canonical.DiagnosticUnsupportedContentPart, "unknown provider stream event was preserved but has no Chat chunk representation", "", event.Value))
	default:
		diagnostics = append(diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidStreamEvent, fmt.Sprintf("unknown canonical event type %q", event.Type), "type", nil))
	}

	if canonical.HasErrors(diagnostics) {
		return canonical.Failure[[][]byte](diagnostics)
	}
	return canonical.Success(frames, diagnostics)
}

func hasMeaningfulJSON(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	return len(trimmed) > 0 && !bytes.Equal(trimmed, []byte("null")) && !bytes.Equal(trimmed, []byte("{}"))
}

func (e *StreamEncoder) Close() canonical.Result[[][]byte] {
	if e.finished {
		return canonical.Success([][]byte(nil), nil)
	}
	return canonical.Failure[[][]byte]([]canonical.Diagnostic{
		diagnostic(canonical.SeverityError, diagnosticStreamTruncated, "canonical stream closed before a finish event", "", nil),
	})
}

func (e *StreamEncoder) ensureRole(diagnostics *[]canonical.Diagnostic) [][]byte {
	if e.roleSent {
		return nil
	}
	if e.id == "" || e.model == "" {
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticStreamState, "stream response ID and model must be known before the first Chat chunk", "", nil))
		return nil
	}
	e.roleSent = true
	return [][]byte{e.chunkFrame(map[string]any{"role": "assistant", "content": ""}, nil, nil)}
}

func (e *StreamEncoder) chunkFrame(delta map[string]any, finishReason any, usage any) []byte {
	choice := map[string]any{
		"index":         0,
		"delta":         delta,
		"finish_reason": finishReason,
		"logprobs":      nil,
	}
	chunk := map[string]any{
		"id":      e.id,
		"object":  "chat.completion.chunk",
		"created": e.options.Created,
		"model":   e.model,
		"choices": []any{choice},
	}
	if usage != nil {
		chunk["usage"] = usage
	}
	return sseFrame(chunk)
}

func (e *StreamEncoder) usageFrame(usage canonical.Usage) []byte {
	chunk := map[string]any{
		"id":      e.id,
		"object":  "chat.completion.chunk",
		"created": e.options.Created,
		"model":   e.model,
		"choices": []any{},
		"usage":   encodeChatUsage(usage),
	}
	return sseFrame(chunk)
}

func sseFrame(value any) []byte {
	encoded, _ := json.Marshal(value)
	frame := make([]byte, 0, len(encoded)+8)
	frame = append(frame, "data: "...)
	frame = append(frame, encoded...)
	frame = append(frame, '\n', '\n')
	return frame
}

func lenOpenTools(tools map[int]bool) int {
	count := 0
	for _, open := range tools {
		if open {
			count++
		}
	}
	return count
}

func mergeUsage(current *canonical.Usage, update canonical.Usage) *canonical.Usage {
	merged := canonical.Usage{}
	if current != nil {
		merged = *current
	}
	if update.InputTokens != nil {
		merged.InputTokens = update.InputTokens
	}
	if update.OutputTokens != nil {
		merged.OutputTokens = update.OutputTokens
	}
	if update.TotalTokens != nil {
		merged.TotalTokens = update.TotalTokens
	}
	return &merged
}
