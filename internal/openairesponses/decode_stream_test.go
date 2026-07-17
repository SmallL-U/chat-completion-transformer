package openairesponses

import (
	"math/rand"
	"strings"
	"testing"

	"chat-completion-transformer/internal/canonical"
)

func TestStreamDecoderDecodesTextRefusalToolsUsageAndUnknown(t *testing.T) {
	stream := strings.Join([]string{
		sseFrame("response.created", `{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","created_at":123,"model":"gpt-test","status":"in_progress","output":[]}}`),
		sseFrame("response.output_text.delta", `{"type":"response.output_text.delta","sequence_number":1,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"hel"}`),
		sseFrame("response.output_text.done", `{"type":"response.output_text.done","sequence_number":2,"item_id":"msg_1","output_index":0,"content_index":0,"text":"hello"}`),
		sseFrame("response.refusal.delta", `{"type":"response.refusal.delta","sequence_number":3,"item_id":"msg_1","output_index":0,"content_index":1,"delta":"no"}`),
		sseFrame("response.refusal.done", `{"type":"response.refusal.done","sequence_number":4,"item_id":"msg_1","output_index":0,"content_index":1,"refusal":"nope"}`),
		sseFrame("response.output_item.added", `{"type":"response.output_item.added","sequence_number":5,"output_index":1,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"weather","arguments":""}}`),
		sseFrame("response.function_call_arguments.delta", `{"type":"response.function_call_arguments.delta","sequence_number":6,"item_id":"fc_1","output_index":1,"delta":"{\"city\":"}`),
		sseFrame("response.function_call_arguments.delta", `{"type":"response.function_call_arguments.delta","sequence_number":7,"item_id":"fc_1","output_index":1,"delta":"\"Beijing\"}"}`),
		sseFrame("response.function_call_arguments.done", `{"type":"response.function_call_arguments.done","sequence_number":8,"item_id":"fc_1","output_index":1,"name":"weather","arguments":"{\"city\":\"Beijing\"}"}`),
		sseFrame("response.output_item.done", `{"type":"response.output_item.done","sequence_number":9,"output_index":1,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"weather","arguments":"{\"city\":\"Beijing\"}"}}`),
		sseFrame("response.reasoning_summary_text.delta", `{"type":"response.reasoning_summary_text.delta","sequence_number":10,"item_id":"rs_1","output_index":2,"summary_index":0,"delta":"thought"}`),
		sseFrame("response.completed", completedEventJSON()),
	}, "")

	decoder := NewStreamDecoder(0)
	result := decoder.Feed([]byte(stream))
	if !result.OK || result.Value == nil {
		t.Fatalf("Feed() = %#v", result)
	}
	if closeResult := decoder.Close(); !closeResult.OK {
		t.Fatalf("Close() = %#v", closeResult)
	}

	events := *result.Value
	if len(events) != 12 {
		t.Fatalf("event count = %d, want 12: %#v", len(events), events)
	}
	if events[0].Type != canonical.EventResponseStart || events[0].CreatedAt == nil || *events[0].CreatedAt != 123 {
		t.Fatalf("start event = %#v", events[0])
	}
	if events[1].Type != canonical.EventTextDelta || events[1].Delta != "hel" || events[2].Delta != "lo" {
		t.Fatalf("text events = %#v", events[1:3])
	}
	if events[3].Type != canonical.EventRefusalDelta || events[4].Delta != "pe" {
		t.Fatalf("refusal events = %#v", events[3:5])
	}
	if events[5].Type != canonical.EventToolCallStart || events[5].CallID != "call_1" {
		t.Fatalf("tool start = %#v", events[5])
	}
	if events[8].Type != canonical.EventToolCallEnd {
		t.Fatalf("tool end = %#v", events[8])
	}
	if events[9].Type != canonical.EventOpaque {
		t.Fatalf("unknown event = %#v", events[9])
	}
	if events[10].Type != canonical.EventUsage || events[10].Usage == nil || *events[10].Usage.TotalTokens != 14 {
		t.Fatalf("usage event = %#v", events[10])
	}
	if events[11].Type != canonical.EventFinish || events[11].Reason == nil || *events[11].Reason != canonical.FinishReasonToolCalls {
		t.Fatalf("finish event = %#v", events[11])
	}
}

func TestStreamDecoderAcceptsEveryTwoWayByteSplit(t *testing.T) {
	stream := compactToolStream()
	for split := 0; split <= len(stream); split++ {
		decoder := NewStreamDecoder(0)
		events := feedStreamParts(t, decoder, []byte(stream[:split]), []byte(stream[split:]))
		if closeResult := decoder.Close(); !closeResult.OK {
			t.Fatalf("split %d: Close() = %#v", split, closeResult)
		}
		if arguments := joinedToolArguments(events); arguments != `{"x":1}` {
			t.Fatalf("split %d: arguments = %q", split, arguments)
		}
	}
}

func TestStreamDecoderAcceptsRandomChunking(t *testing.T) {
	stream := []byte(compactToolStream())
	random := rand.New(rand.NewSource(7))
	for iteration := 0; iteration < 100; iteration++ {
		decoder := NewStreamDecoder(0)
		events := make([]canonical.Event, 0)
		for offset := 0; offset < len(stream); {
			size := random.Intn(17) + 1
			end := offset + size
			if end > len(stream) {
				end = len(stream)
			}
			events = append(events, feedStreamParts(t, decoder, stream[offset:end])...)
			offset = end
		}
		if closeResult := decoder.Close(); !closeResult.OK {
			t.Fatalf("iteration %d: Close() = %#v", iteration, closeResult)
		}
		if arguments := joinedToolArguments(events); arguments != `{"x":1}` {
			t.Fatalf("iteration %d: arguments = %q", iteration, arguments)
		}
	}
}

func TestStreamDecoderReportsStateErrorsWithoutPanicking(t *testing.T) {
	decoder := NewStreamDecoder(0)
	result := decoder.Feed([]byte(sseFrame("response.output_text.delta", `{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"bad"}`)))
	if result.OK {
		t.Fatalf("delta before created unexpectedly succeeded: %#v", result)
	}
	assertDiagnosticCode(t, result.Diagnostics, DiagnosticInvalidStreamState)

	decoder = NewStreamDecoder(0)
	result = decoder.Feed([]byte(sseFrame("response.created", `{"type":"response.created","response":{"id":"resp","output":[]}}`)))
	if !result.OK {
		t.Fatalf("created = %#v", result)
	}
	closeResult := decoder.Close()
	if closeResult.OK {
		t.Fatal("unterminated Responses stream unexpectedly closed successfully")
	}
	assertDiagnosticCode(t, closeResult.Diagnostics, DiagnosticInvalidStreamState)
}

func TestStreamDecoderMapsFailedAndTopLevelError(t *testing.T) {
	for _, test := range []struct {
		name   string
		stream string
	}{
		{
			name: "failed",
			stream: sseFrame("response.created", `{"type":"response.created","response":{"id":"resp","output":[]}}`) +
				sseFrame("response.failed", `{"type":"response.failed","response":{"id":"resp","status":"failed","output":[],"error":{"code":"server_error","message":"failed"}}}`),
		},
		{
			name:   "top level error",
			stream: sseFrame("error", `{"type":"error","code":"rate_limit","message":"slow down","param":null}`),
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			decoder := NewStreamDecoder(0)
			result := decoder.Feed([]byte(test.stream))
			if !result.OK || result.Value == nil {
				t.Fatalf("Feed() = %#v", result)
			}
			var sawError, sawFinish bool
			for _, event := range *result.Value {
				sawError = sawError || event.Type == canonical.EventError
				sawFinish = sawFinish || event.Type == canonical.EventFinish
			}
			if !sawError || !sawFinish {
				t.Fatalf("events = %#v", *result.Value)
			}
			if closeResult := decoder.Close(); !closeResult.OK {
				t.Fatalf("Close() = %#v", closeResult)
			}
		})
	}
}

func TestStreamDecoderMapsIncompleteUsageAndReason(t *testing.T) {
	stream := sseFrame("response.created", `{"type":"response.created","response":{"id":"resp","output":[]}}`) +
		sseFrame("response.incomplete", `{"type":"response.incomplete","response":{"id":"resp","status":"incomplete","output":[],"incomplete_details":{"reason":"max_output_tokens"},"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}`)
	decoder := NewStreamDecoder(0)
	result := decoder.Feed([]byte(stream))
	if !result.OK || result.Value == nil {
		t.Fatalf("Feed() = %#v", result)
	}
	events := *result.Value
	if len(events) != 3 || events[1].Type != canonical.EventUsage || events[2].Type != canonical.EventFinish {
		t.Fatalf("events = %#v", events)
	}
	if events[2].Reason == nil || *events[2].Reason != canonical.FinishReasonLength || events[2].ProviderReason == nil || *events[2].ProviderReason != "max_output_tokens" {
		t.Fatalf("finish = %#v", events[2])
	}
	if closeResult := decoder.Close(); !closeResult.OK {
		t.Fatalf("Close() = %#v", closeResult)
	}
}

func TestStreamDecoderReconcilesTerminalOutputWhenDoneEventsAreMissing(t *testing.T) {
	stream := sseFrame("response.created", `{"type":"response.created","response":{"id":"resp","output":[]}}`) +
		sseFrame("response.output_text.delta", `{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"hel"}`) +
		sseFrame("response.output_item.added", `{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc","call_id":"call","name":"f","arguments":""}}`) +
		sseFrame("response.function_call_arguments.delta", `{"type":"response.function_call_arguments.delta","item_id":"fc","output_index":1,"delta":"{\"x\":"}`) +
		sseFrame("response.completed", `{"type":"response.completed","response":{"id":"resp","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"hello"}]},{"type":"function_call","id":"fc","call_id":"call","name":"f","arguments":"{\"x\":1}"}]}}`)

	decoder := NewStreamDecoder(0)
	result := decoder.Feed([]byte(stream))
	if !result.OK || result.Value == nil {
		t.Fatalf("Feed() = %#v", result)
	}
	assertDiagnosticCode(t, result.Diagnostics, DiagnosticInvalidStreamState)
	if result.Lossless {
		t.Fatal("reconciled stream unexpectedly reported lossless")
	}
	if got := joinedText(*result.Value); got != "hello" {
		t.Fatalf("joined text = %q", got)
	}
	if got := joinedToolArguments(*result.Value); got != `{"x":1}` {
		t.Fatalf("joined arguments = %q", got)
	}
	if !hasEventType(*result.Value, canonical.EventToolCallEnd) || !hasEventType(*result.Value, canonical.EventFinish) {
		t.Fatalf("events = %#v", *result.Value)
	}
	if closeResult := decoder.Close(); !closeResult.OK {
		t.Fatalf("Close() = %#v", closeResult)
	}
}

func TestStreamDecoderRejectsTerminalOutputThatConflictsWithDeltas(t *testing.T) {
	stream := sseFrame("response.created", `{"type":"response.created","response":{"id":"resp","output":[]}}`) +
		sseFrame("response.output_text.delta", `{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"hello"}`) +
		sseFrame("response.completed", `{"type":"response.completed","response":{"id":"resp","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"goodbye"}]}]}}`)
	decoder := NewStreamDecoder(0)
	result := decoder.Feed([]byte(stream))
	if result.OK {
		t.Fatalf("conflicting terminal response unexpectedly succeeded: %#v", result)
	}
	assertDiagnosticCode(t, result.Diagnostics, DiagnosticInvalidStreamState)
}

func compactToolStream() string {
	return sseFrame("response.created", `{"type":"response.created","response":{"id":"resp","created_at":1,"model":"gpt-test","status":"in_progress","output":[]}}`) +
		sseFrame("response.output_item.added", `{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc","call_id":"call","name":"f","arguments":""}}`) +
		sseFrame("response.function_call_arguments.delta", `{"type":"response.function_call_arguments.delta","item_id":"fc","output_index":0,"delta":"{\"x\":"}`) +
		sseFrame("response.function_call_arguments.delta", `{"type":"response.function_call_arguments.delta","item_id":"fc","output_index":0,"delta":"1}"}`) +
		sseFrame("response.function_call_arguments.done", `{"type":"response.function_call_arguments.done","item_id":"fc","output_index":0,"arguments":"{\"x\":1}"}`) +
		sseFrame("response.completed", `{"type":"response.completed","response":{"id":"resp","status":"completed","output":[{"type":"function_call","id":"fc","call_id":"call","name":"f","arguments":"{\"x\":1}"}]}}`)
}

func completedEventJSON() string {
	return `{"type":"response.completed","sequence_number":11,"response":{"id":"resp_1","created_at":123,"model":"gpt-test","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]},{"type":"refusal","refusal":"nope"}]},{"type":"function_call","id":"fc_1","call_id":"call_1","name":"weather","arguments":"{\"city\":\"Beijing\"}"}],"usage":{"input_tokens":10,"output_tokens":4,"total_tokens":14,"input_tokens_details":{"cached_tokens":2}}}}`
}

func sseFrame(name string, data string) string {
	return "event: " + name + "\n" + "data: " + data + "\n\n"
}

func feedStreamParts(t *testing.T, decoder *StreamDecoder, parts ...[]byte) []canonical.Event {
	t.Helper()
	events := make([]canonical.Event, 0)
	for _, part := range parts {
		result := decoder.Feed(part)
		if !result.OK || result.Value == nil {
			t.Fatalf("Feed(%q) = %#v", part, result)
		}
		events = append(events, (*result.Value)...)
	}
	return events
}

func joinedToolArguments(events []canonical.Event) string {
	var builder strings.Builder
	for _, event := range events {
		if event.Type == canonical.EventToolArgumentsDelta {
			builder.WriteString(event.Delta)
		}
	}
	return builder.String()
}

func joinedText(events []canonical.Event) string {
	var builder strings.Builder
	for _, event := range events {
		if event.Type == canonical.EventTextDelta {
			builder.WriteString(event.Delta)
		}
	}
	return builder.String()
}

func hasEventType(events []canonical.Event, eventType canonical.EventType) bool {
	for _, event := range events {
		if event.Type == eventType {
			return true
		}
	}
	return false
}
