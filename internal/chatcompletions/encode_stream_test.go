package chatcompletions

import (
	"encoding/json"
	"strings"
	"testing"

	"chat-completion-transformer/internal/canonical"
)

func TestStreamEncoderTextToolsUsageAndFinish(t *testing.T) {
	encoder := NewStreamEncoder(StreamEncodeOptions{Mode: canonical.ModeCompatible, Created: 123})
	model := "target-model"
	indexNine := 9
	indexTwo := 2
	reason := canonical.FinishReasonToolCalls
	inputTokens := int64(10)
	outputTokens := int64(5)
	totalTokens := int64(15)
	events := []canonical.Event{
		{Type: canonical.EventResponseStart, ID: "response_1", Model: &model},
		{Type: canonical.EventTextDelta, OutputIndex: &indexTwo, Delta: "hello"},
		{Type: canonical.EventToolCallStart, OutputIndex: &indexNine, CallID: "call_a", Name: "lookup"},
		{Type: canonical.EventToolArgumentsDelta, OutputIndex: &indexNine, Delta: `{"q":`},
		{Type: canonical.EventToolCallStart, OutputIndex: &indexTwo, CallID: "call_b", Name: "lookup"},
		{Type: canonical.EventToolArgumentsDelta, OutputIndex: &indexTwo, Delta: `"x"}`},
		{Type: canonical.EventToolCallEnd, OutputIndex: &indexNine},
		{Type: canonical.EventToolCallEnd, OutputIndex: &indexTwo},
		{Type: canonical.EventUsage, Usage: &canonical.Usage{InputTokens: &inputTokens, OutputTokens: &outputTokens, TotalTokens: &totalTokens}},
		{Type: canonical.EventFinish, Reason: &reason},
	}

	var frames [][]byte
	for _, event := range events {
		result := encoder.Encode(event)
		if !result.OK || result.Value == nil {
			t.Fatalf("event %#v: result = %#v", event, result)
		}
		frames = append(frames, (*result.Value)...)
	}
	if len(frames) != 9 {
		t.Fatalf("frames = %d\n%s", len(frames), joinFrames(frames))
	}
	if string(frames[len(frames)-1]) != "data: [DONE]\n\n" {
		t.Fatalf("last frame = %q", frames[len(frames)-1])
	}

	chunks := make([]map[string]any, 0, len(frames)-1)
	for _, frame := range frames[:len(frames)-1] {
		payload := strings.TrimSuffix(strings.TrimPrefix(string(frame), "data: "), "\n\n")
		var chunk map[string]any
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, chunk)
	}
	firstChoices := chunks[0]["choices"].([]any)
	firstDelta := firstChoices[0].(map[string]any)["delta"].(map[string]any)
	if firstDelta["role"] != "assistant" {
		t.Fatalf("first delta = %#v", firstDelta)
	}
	usageChoices := chunks[len(chunks)-1]["choices"].([]any)
	if len(usageChoices) != 0 {
		t.Fatalf("usage choices = %#v", usageChoices)
	}

	joined := joinFrames(frames)
	if strings.Count(joined, `"role":"assistant"`) != 1 {
		t.Fatalf("role count in %s", joined)
	}
	if !strings.Contains(joined, `"index":0`) || !strings.Contains(joined, `"index":1`) {
		t.Fatalf("tool indexes missing in %s", joined)
	}
}

func TestStreamEncoderRejectsInvalidState(t *testing.T) {
	encoder := NewStreamEncoder(StreamEncodeOptions{FallbackID: "id", FallbackModel: "model"})
	index := 1
	result := encoder.Encode(canonical.Event{Type: canonical.EventToolArgumentsDelta, OutputIndex: &index, Delta: "{"})
	if result.OK || !containsDiagnostic(result.Diagnostics, diagnosticStreamState) {
		t.Fatalf("result = %#v", result)
	}
}

func TestStreamEncoderFailedToolStartDoesNotConsumeRole(t *testing.T) {
	encoder := NewStreamEncoder(StreamEncodeOptions{FallbackID: "id", FallbackModel: "model"})
	index := 1
	failed := encoder.Encode(canonical.Event{Type: canonical.EventToolCallStart, OutputIndex: &index, CallID: "call_1"})
	if failed.OK || !containsDiagnostic(failed.Diagnostics, diagnosticInvalidStreamEvent) {
		t.Fatalf("failed = %#v", failed)
	}

	result := encoder.Encode(canonical.Event{Type: canonical.EventToolCallStart, OutputIndex: &index, CallID: "call_1", Name: "lookup"})
	if !result.OK || result.Value == nil || len(*result.Value) != 2 {
		t.Fatalf("result = %#v", result)
	}
	if !strings.Contains(string((*result.Value)[0]), `"role":"assistant"`) {
		t.Fatalf("role frame = %s", (*result.Value)[0])
	}
}

func TestStreamEncoderResponseStartIsAtomicAndUnique(t *testing.T) {
	encoder := NewStreamEncoder(StreamEncodeOptions{})
	model := "model"
	missingModel := encoder.Encode(canonical.Event{Type: canonical.EventResponseStart, ID: "id"})
	if missingModel.OK {
		t.Fatalf("missing model = %#v", missingModel)
	}
	missingID := encoder.Encode(canonical.Event{Type: canonical.EventResponseStart, Model: &model})
	if missingID.OK {
		t.Fatalf("missing ID reused state from failed event: %#v", missingID)
	}

	started := encoder.Encode(canonical.Event{Type: canonical.EventResponseStart, ID: "id", Model: &model})
	if !started.OK || started.Value == nil || len(*started.Value) != 1 {
		t.Fatalf("started = %#v", started)
	}
	duplicate := encoder.Encode(canonical.Event{Type: canonical.EventResponseStart, ID: "other", Model: &model})
	if duplicate.OK || !containsDiagnostic(duplicate.Diagnostics, diagnosticStreamState) {
		t.Fatalf("duplicate = %#v", duplicate)
	}
}

func TestStreamEncoderStrictFinishFailureCanBeRetried(t *testing.T) {
	encoder := NewStreamEncoder(StreamEncodeOptions{Mode: canonical.ModeStrict, FallbackID: "id", FallbackModel: "model"})
	reason := canonical.FinishReasonStop
	providerReason := "provider_stop"
	failed := encoder.Encode(canonical.Event{Type: canonical.EventFinish, Reason: &reason, ProviderReason: &providerReason})
	if failed.OK || !containsDiagnostic(failed.Diagnostics, diagnosticProviderReasonLossy) {
		t.Fatalf("failed = %#v", failed)
	}

	result := encoder.Encode(canonical.Event{Type: canonical.EventFinish, Reason: &reason})
	if !result.OK || result.Value == nil || len(*result.Value) != 3 {
		t.Fatalf("result = %#v", result)
	}
	if string((*result.Value)[2]) != "data: [DONE]\n\n" || !strings.Contains(string((*result.Value)[0]), `"role":"assistant"`) {
		t.Fatalf("frames = %s", joinFrames(*result.Value))
	}
}

func TestStreamEncoderMergesUsageEvents(t *testing.T) {
	encoder := NewStreamEncoder(StreamEncodeOptions{FallbackID: "id", FallbackModel: "model"})
	inputTokens := int64(10)
	outputTokens := int64(5)
	totalTokens := int64(15)
	updates := []canonical.Event{
		{Type: canonical.EventUsage, Usage: &canonical.Usage{InputTokens: &inputTokens}},
		{Type: canonical.EventUsage, Usage: &canonical.Usage{OutputTokens: &outputTokens, TotalTokens: &totalTokens}},
	}
	for _, event := range updates {
		result := encoder.Encode(event)
		if !result.OK {
			t.Fatalf("usage event = %#v", result)
		}
	}
	reason := canonical.FinishReasonStop
	result := encoder.Encode(canonical.Event{Type: canonical.EventFinish, Reason: &reason})
	if !result.OK || result.Value == nil || len(*result.Value) != 4 {
		t.Fatalf("finish = %#v", result)
	}

	usageFrame := (*result.Value)[2]
	payload := strings.TrimSuffix(strings.TrimPrefix(string(usageFrame), "data: "), "\n\n")
	var chunk struct {
		Choices []any            `json:"choices"`
		Usage   map[string]int64 `json:"usage"`
	}
	if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
		t.Fatal(err)
	}
	if len(chunk.Choices) != 0 || chunk.Usage["prompt_tokens"] != 10 || chunk.Usage["completion_tokens"] != 5 || chunk.Usage["total_tokens"] != 15 {
		t.Fatalf("usage chunk = %#v", chunk)
	}
}

func TestStreamEncoderDetectsTruncation(t *testing.T) {
	encoder := NewStreamEncoder(StreamEncodeOptions{FallbackID: "id", FallbackModel: "model"})
	result := encoder.Close()
	if result.OK || !containsDiagnostic(result.Diagnostics, diagnosticStreamTruncated) {
		t.Fatalf("result = %#v", result)
	}
}

func TestStreamEncoderDiagnosesProviderFinishDetails(t *testing.T) {
	encoder := NewStreamEncoder(StreamEncodeOptions{
		Mode:          canonical.ModeCompatible,
		FallbackID:    "id",
		FallbackModel: "model",
	})
	reason := canonical.FinishReasonStop
	result := encoder.Encode(canonical.Event{
		Type:   canonical.EventFinish,
		Reason: &reason,
		Value:  json.RawMessage(`{"stop_details":{"type":"refusal"}}`),
	})
	if !result.OK || result.Lossless || !containsDiagnostic(result.Diagnostics, diagnosticResponseExtensionLossy) {
		t.Fatalf("result = %#v", result)
	}
}

func joinFrames(frames [][]byte) string {
	var builder strings.Builder
	for _, frame := range frames {
		builder.Write(frame)
	}
	return builder.String()
}
