package anthropicmessages

import (
	"encoding/json"
	"strings"
	"testing"

	"chat-completion-transformer/internal/canonical"
)

func TestStreamDecoderReassemblesEveryToolArgumentSplit(t *testing.T) {
	arguments := `{"city":"Beijing","unit":"c"}`
	for split := 0; split <= len(arguments); split++ {
		stream := toolStream(arguments[:split], arguments[split:])
		decoder := NewStreamDecoder(0)
		result := decoder.Feed(stream)
		if !result.OK || result.Value == nil {
			t.Fatalf("split %d feed result = %#v", split, result)
		}
		closed := decoder.Close()
		if !closed.OK {
			t.Fatalf("split %d close result = %#v", split, closed)
		}

		var joined strings.Builder
		var starts, ends int
		var lastUsage *canonical.Usage
		var finish *canonical.Event
		for index := range *result.Value {
			event := (*result.Value)[index]
			switch event.Type {
			case canonical.EventToolCallStart:
				starts++
			case canonical.EventToolArgumentsDelta:
				joined.WriteString(event.Delta)
			case canonical.EventToolCallEnd:
				ends++
			case canonical.EventUsage:
				lastUsage = event.Usage
			case canonical.EventFinish:
				copy := event
				finish = &copy
			}
		}
		if starts != 1 || ends != 1 || joined.String() != arguments {
			t.Fatalf("split %d: starts=%d ends=%d arguments=%q", split, starts, ends, joined.String())
		}
		if lastUsage == nil || lastUsage.TotalTokens == nil || *lastUsage.TotalTokens != 6 {
			t.Fatalf("split %d usage = %#v", split, lastUsage)
		}
		if finish == nil || finish.Reason == nil || *finish.Reason != canonical.FinishReasonToolCalls || finish.ProviderReason == nil || *finish.ProviderReason != "tool_use" {
			t.Fatalf("split %d finish = %#v", split, finish)
		}
		var opaque map[string]json.RawMessage
		if err := json.Unmarshal(finish.Value, &opaque); err != nil {
			t.Fatalf("split %d finish value: %v", split, err)
		}
		if string(opaque["stop_details"]) != `{"kind":"tool"}` || string(opaque["stop_sequence"]) != "null" {
			t.Fatalf("split %d opaque = %#v", split, opaque)
		}
	}
}

func TestStreamDecoderAcceptsEveryNetworkSplit(t *testing.T) {
	stream := textStream()
	for split := 0; split <= len(stream); split++ {
		decoder := NewStreamDecoder(0)
		first := decoder.Feed(stream[:split])
		if !first.OK {
			t.Fatalf("split %d first = %#v", split, first)
		}
		second := decoder.Feed(stream[split:])
		if !second.OK {
			t.Fatalf("split %d second = %#v", split, second)
		}
		if closed := decoder.Close(); !closed.OK {
			t.Fatalf("split %d close = %#v", split, closed)
		}
	}
}

func TestStreamDecoderValidatesToolJSONOnlyAtBlockStop(t *testing.T) {
	decoder := NewStreamDecoder(0)
	prefix := append([]byte{}, messageStartFrame()...)
	prefix = append(prefix, sseJSON(map[string]any{
		"type":          "content_block_start",
		"index":         0,
		"content_block": map[string]any{"type": "tool_use", "id": "call_1", "name": "lookup", "input": map[string]any{}},
	})...)
	prefix = append(prefix, sseJSON(map[string]any{
		"type":  "content_block_delta",
		"index": 0,
		"delta": map[string]any{"type": "input_json_delta", "partial_json": "["},
	})...)
	result := decoder.Feed(prefix)
	if !result.OK {
		t.Fatalf("delta result = %#v", result)
	}

	result = decoder.Feed(sseJSON(map[string]any{"type": "content_block_stop", "index": 0}))
	if result.OK || !hasDiagnostic(result.Diagnostics, canonical.DiagnosticInvalidToolArgumentsJSON) {
		t.Fatalf("stop result = %#v", result)
	}
}

func TestStreamDecoderEmitsInitialEmptyToolInput(t *testing.T) {
	stream := append([]byte{}, messageStartFrame()...)
	stream = append(stream, sseJSON(map[string]any{
		"type":          "content_block_start",
		"index":         0,
		"content_block": map[string]any{"type": "tool_use", "id": "call_1", "name": "lookup", "input": map[string]any{}},
	})...)
	stream = append(stream, sseJSON(map[string]any{"type": "content_block_stop", "index": 0})...)
	stream = append(stream, sseJSON(map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": "tool_use", "stop_sequence": nil},
		"usage": map[string]any{"output_tokens": 1},
	})...)
	stream = append(stream, sseJSON(map[string]any{"type": "message_stop"})...)

	decoder := NewStreamDecoder(0)
	result := decoder.Feed(stream)
	if !result.OK || result.Value == nil {
		t.Fatalf("result = %#v", result)
	}
	var arguments strings.Builder
	for _, event := range *result.Value {
		if event.Type == canonical.EventToolArgumentsDelta {
			arguments.WriteString(event.Delta)
		}
	}
	if arguments.String() != "{}" {
		t.Fatalf("arguments = %q", arguments.String())
	}
}

func TestStreamDecoderPreservesUnknownEventsAndBlocks(t *testing.T) {
	stream := append([]byte{}, messageStartFrame()...)
	stream = append(stream, sseJSON(map[string]any{"type": "future_event", "value": 1})...)
	stream = append(stream, sseJSON(map[string]any{
		"type":          "content_block_start",
		"index":         4,
		"content_block": map[string]any{"type": "thinking", "thinking": ""},
	})...)
	stream = append(stream, sseJSON(map[string]any{
		"type":  "content_block_delta",
		"index": 4,
		"delta": map[string]any{"type": "thinking_delta", "thinking": "x"},
	})...)
	stream = append(stream, sseJSON(map[string]any{"type": "content_block_stop", "index": 4})...)
	stream = append(stream, sseJSON(map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": "end_turn", "stop_sequence": nil},
		"usage": map[string]any{"output_tokens": 1},
	})...)
	stream = append(stream, sseJSON(map[string]any{"type": "message_stop"})...)

	decoder := NewStreamDecoder(0)
	result := decoder.Feed(stream)
	if !result.OK || result.Value == nil {
		t.Fatalf("result = %#v", result)
	}
	opaque := 0
	for _, event := range *result.Value {
		if event.Type != canonical.EventOpaque {
			continue
		}
		opaque++
		if event.Provider == nil || *event.Provider != "anthropic.messages" || len(event.Value) == 0 {
			t.Fatalf("opaque event = %#v", event)
		}
	}
	if opaque != 4 {
		t.Fatalf("opaque events = %d, all = %#v", opaque, *result.Value)
	}
	if closed := decoder.Close(); !closed.OK {
		t.Fatalf("close = %#v", closed)
	}
}

func TestStreamDecoderReportsIllegalStateWithoutPanic(t *testing.T) {
	decoder := NewStreamDecoder(0)
	result := decoder.Feed(sseJSON(map[string]any{
		"type":  "content_block_delta",
		"index": 0,
		"delta": map[string]any{"type": "text_delta", "text": "x"},
	}))
	if result.OK || !hasDiagnostic(result.Diagnostics, diagnosticStreamState) {
		t.Fatalf("result = %#v", result)
	}
}

func TestStreamDecoderPingErrorAndTruncation(t *testing.T) {
	decoder := NewStreamDecoder(0)
	result := decoder.Feed(sseJSON(map[string]any{"type": "ping"}))
	if !result.OK || result.Value == nil || len(*result.Value) != 0 {
		t.Fatalf("ping result = %#v", result)
	}
	result = decoder.Feed(sseJSON(map[string]any{
		"type":  "error",
		"error": map[string]any{"type": "overloaded_error", "message": "busy"},
	}))
	if !result.OK || result.Value == nil || len(*result.Value) != 2 || (*result.Value)[0].Type != canonical.EventError {
		t.Fatalf("error result = %#v", result)
	}
	finish := (*result.Value)[1]
	if finish.Type != canonical.EventFinish || finish.Reason == nil || *finish.Reason != canonical.FinishReasonError || finish.ProviderReason == nil || *finish.ProviderReason != "overloaded_error" {
		t.Fatalf("error finish = %#v", finish)
	}
	if closed := decoder.Close(); !closed.OK {
		t.Fatalf("error close = %#v", closed)
	}

	decoder = NewStreamDecoder(0)
	result = decoder.Feed(messageStartFrame())
	if !result.OK {
		t.Fatalf("start result = %#v", result)
	}
	closed := decoder.Close()
	if closed.OK || !hasDiagnostic(closed.Diagnostics, diagnosticStreamTruncated) {
		t.Fatalf("truncated close = %#v", closed)
	}
}

func toolStream(first, second string) []byte {
	stream := append([]byte{}, messageStartFrame()...)
	stream = append(stream, sseJSON(map[string]any{"type": "ping"})...)
	stream = append(stream, sseJSON(map[string]any{
		"type":          "content_block_start",
		"index":         3,
		"content_block": map[string]any{"type": "tool_use", "id": "call_1", "name": "weather", "input": map[string]any{}},
	})...)
	for _, partial := range []string{first, second} {
		stream = append(stream, sseJSON(map[string]any{
			"type":  "content_block_delta",
			"index": 3,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": partial},
		})...)
	}
	stream = append(stream, sseJSON(map[string]any{"type": "content_block_stop", "index": 3})...)
	stream = append(stream, sseJSON(map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   "tool_use",
			"stop_sequence": nil,
			"stop_details":  map[string]any{"kind": "tool"},
		},
		"usage": map[string]any{"output_tokens": 4, "cache_read_input_tokens": 1},
	})...)
	stream = append(stream, sseJSON(map[string]any{"type": "message_stop"})...)
	return stream
}

func textStream() []byte {
	stream := append([]byte{}, messageStartFrame()...)
	stream = append(stream, sseJSON(map[string]any{
		"type":          "content_block_start",
		"index":         0,
		"content_block": map[string]any{"type": "text", "text": ""},
	})...)
	stream = append(stream, sseJSON(map[string]any{
		"type":  "content_block_delta",
		"index": 0,
		"delta": map[string]any{"type": "text_delta", "text": "hello"},
	})...)
	stream = append(stream, sseJSON(map[string]any{"type": "content_block_stop", "index": 0})...)
	stream = append(stream, sseJSON(map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": "end_turn", "stop_sequence": nil},
		"usage": map[string]any{"output_tokens": 1},
	})...)
	stream = append(stream, sseJSON(map[string]any{"type": "message_stop"})...)
	return stream
}

func messageStartFrame() []byte {
	return sseJSON(map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test",
			"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
			"usage": map[string]any{"input_tokens": 2, "output_tokens": 0},
		},
	})
}

func sseJSON(value any) []byte {
	typeName := value.(map[string]any)["type"].(string)
	encoded, _ := json.Marshal(value)
	frame := make([]byte, 0, len(encoded)+len(typeName)+16)
	frame = append(frame, "event: "...)
	frame = append(frame, typeName...)
	frame = append(frame, '\n')
	frame = append(frame, "data: "...)
	frame = append(frame, encoded...)
	frame = append(frame, '\n', '\n')
	return frame
}
