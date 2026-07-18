package chatcompletions

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"chat-completion-transformer/internal/canonical"
)

func TestStreamEncoderTextToolsUsageAndFinish(t *testing.T) {
	encoder := NewStreamEncoder(StreamEncodeOptions{Mode: canonical.ModeCompatible, Created: 123, IncludeUsage: true})
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
	for index, chunk := range chunks[:len(chunks)-1] {
		usage, exists := chunk["usage"]
		if !exists || usage != nil {
			t.Fatalf("chunk %d usage = %#v, present = %t; want null", index, usage, exists)
		}
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

func TestStreamEncoderRebuildsDeltaTemplateAfterResponseStart(t *testing.T) {
	encoder := NewStreamEncoder(StreamEncodeOptions{Created: 1})
	failed := encoder.Encode(canonical.Event{Type: canonical.EventTextDelta, Delta: "ignored"})
	if failed.OK {
		t.Fatalf("failed delta = %#v", failed)
	}

	model := "target-model"
	started := encoder.Encode(canonical.Event{Type: canonical.EventResponseStart, ID: "response-id", Model: &model})
	if !started.OK {
		t.Fatalf("start = %#v", started)
	}
	result := encoder.Encode(canonical.Event{Type: canonical.EventTextDelta, Delta: "hello"})
	if !result.OK || result.Value == nil || len(*result.Value) != 1 {
		t.Fatalf("delta = %#v", result)
	}
	frame := string((*result.Value)[0])
	if !strings.Contains(frame, `"id":"response-id"`) || !strings.Contains(frame, `"model":"target-model"`) {
		t.Fatalf("frame = %s", frame)
	}
}

func TestStreamEncoderMergesUsageEvents(t *testing.T) {
	encoder := NewStreamEncoder(StreamEncodeOptions{FallbackID: "id", FallbackModel: "model", IncludeUsage: true})
	inputTokens := int64(10)
	outputTokens := int64(5)
	totalTokens := int64(15)
	cachedTokens := int64(4)
	cacheWriteTokens := int64(0)
	updates := []canonical.Event{
		{Type: canonical.EventUsage, Usage: &canonical.Usage{
			InputTokens:       &inputTokens,
			CachedInputTokens: &cachedTokens,
			Extensions: canonical.Object{
				canonical.UsageExtensionAnthropicCacheCreation: json.RawMessage(`{"ephemeral_5m_input_tokens":0}`),
			},
		}},
		{Type: canonical.EventUsage, Usage: &canonical.Usage{
			OutputTokens:          &outputTokens,
			TotalTokens:           &totalTokens,
			CacheWriteInputTokens: &cacheWriteTokens,
		}},
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
		Choices []any `json:"choices"`
		Usage   struct {
			PromptTokens        int64 `json:"prompt_tokens"`
			CompletionTokens    int64 `json:"completion_tokens"`
			TotalTokens         int64 `json:"total_tokens"`
			PromptTokensDetails struct {
				CachedTokens     *int64 `json:"cached_tokens"`
				CacheWriteTokens *int64 `json:"cache_write_tokens"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
		t.Fatal(err)
	}
	if len(chunk.Choices) != 0 || chunk.Usage.PromptTokens != 10 || chunk.Usage.CompletionTokens != 5 || chunk.Usage.TotalTokens != 15 {
		t.Fatalf("usage chunk = %#v", chunk)
	}
	if chunk.Usage.PromptTokensDetails.CachedTokens == nil || *chunk.Usage.PromptTokensDetails.CachedTokens != 4 {
		t.Fatalf("usage chunk = %#v", chunk)
	}
	if chunk.Usage.PromptTokensDetails.CacheWriteTokens == nil || *chunk.Usage.PromptTokensDetails.CacheWriteTokens != 0 {
		t.Fatalf("usage chunk = %#v", chunk)
	}
}

func TestStreamEncoderCopiesUsageValues(t *testing.T) {
	encoder := NewStreamEncoder(StreamEncodeOptions{FallbackID: "id", FallbackModel: "model", IncludeUsage: true})
	input := int64(10)
	output := int64(5)
	total := int64(15)
	cached := int64(4)
	cacheWrite := int64(1)
	usage := canonical.Usage{
		InputTokens:           &input,
		OutputTokens:          &output,
		TotalTokens:           &total,
		CachedInputTokens:     &cached,
		CacheWriteInputTokens: &cacheWrite,
	}
	result := encoder.Encode(canonical.Event{Type: canonical.EventUsage, Usage: &usage})
	if !result.OK {
		t.Fatalf("usage = %#v", result)
	}

	input, output, total, cached, cacheWrite = -10, -5, -15, -4, -1
	reason := canonical.FinishReasonStop
	result = encoder.Encode(canonical.Event{Type: canonical.EventFinish, Reason: &reason})
	if !result.OK || result.Value == nil || len(*result.Value) != 4 {
		t.Fatalf("finish = %#v", result)
	}
	usageFrame := string((*result.Value)[2])
	for _, expected := range []string{
		`"prompt_tokens":10`,
		`"completion_tokens":5`,
		`"total_tokens":15`,
		`"cached_tokens":4`,
		`"cache_write_tokens":1`,
	} {
		if !strings.Contains(usageFrame, expected) {
			t.Fatalf("usage frame = %s, want %s", usageFrame, expected)
		}
	}
}

func TestStreamEncoderOmitsUsageByDefault(t *testing.T) {
	encoder := NewStreamEncoder(StreamEncodeOptions{Mode: canonical.ModeStrict, FallbackID: "id", FallbackModel: "model"})
	inputTokens := int64(10)
	cachedTokens := int64(3)
	usage := encoder.Encode(canonical.Event{Type: canonical.EventUsage, Usage: &canonical.Usage{
		InputTokens:       &inputTokens,
		CachedInputTokens: &cachedTokens,
		Extensions: canonical.Object{
			canonical.UsageExtensionAnthropicCacheCreation: json.RawMessage(`{"ephemeral_5m_input_tokens":1}`),
		},
	}})
	if !usage.OK {
		t.Fatalf("usage = %#v", usage)
	}

	reason := canonical.FinishReasonStop
	result := encoder.Encode(canonical.Event{Type: canonical.EventFinish, Reason: &reason})
	if !result.OK || result.Value == nil || len(*result.Value) != 3 {
		t.Fatalf("finish = %#v", result)
	}
	if strings.Contains(joinFrames(*result.Value), `"usage":`) {
		t.Fatalf("usage was emitted without include_usage: %s", joinFrames(*result.Value))
	}
}

func TestStreamEncoderRejectsUnknownUsageExtensionInStrictMode(t *testing.T) {
	encoder := NewStreamEncoder(StreamEncodeOptions{Mode: canonical.ModeStrict, FallbackID: "id", FallbackModel: "model", IncludeUsage: true})
	result := encoder.Encode(canonical.Event{Type: canonical.EventUsage, Usage: &canonical.Usage{
		Extensions: canonical.Object{"future_usage": json.RawMessage(`true`)},
	}})
	if result.OK || !containsDiagnostic(result.Diagnostics, diagnosticResponseExtensionLossy) {
		t.Fatalf("result = %#v", result)
	}
}

func TestStreamEncoderRejectsInvalidPromptCacheDetails(t *testing.T) {
	negative := int64(-1)
	tests := []struct {
		name  string
		usage canonical.Usage
	}{
		{name: "cached input", usage: canonical.Usage{CachedInputTokens: &negative}},
		{name: "cache write", usage: canonical.Usage{CacheWriteInputTokens: &negative}},
		{name: "null cache creation breakdown", usage: canonical.Usage{Extensions: canonical.Object{
			canonical.UsageExtensionAnthropicCacheCreation: json.RawMessage(`null`),
		}}},
		{name: "negative cache creation breakdown", usage: canonical.Usage{Extensions: canonical.Object{
			canonical.UsageExtensionAnthropicCacheCreation: json.RawMessage(`{"ephemeral_5m_input_tokens":-1}`),
		}}},
		{name: "wrong cache creation breakdown type", usage: canonical.Usage{Extensions: canonical.Object{
			canonical.UsageExtensionAnthropicCacheCreation: json.RawMessage(`{"ephemeral_1h_input_tokens":"1"}`),
		}}},
		{name: "overflowing cache creation breakdown", usage: canonical.Usage{Extensions: canonical.Object{
			canonical.UsageExtensionAnthropicCacheCreation: json.RawMessage(`{"ephemeral_1h_input_tokens":9223372036854775808}`),
		}}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for _, mode := range []canonical.Mode{canonical.ModeStrict, canonical.ModeCompatible, canonical.ModeEmulate} {
				t.Run(string(mode), func(t *testing.T) {
					encoder := NewStreamEncoder(StreamEncodeOptions{
						Mode:          mode,
						FallbackID:    "id",
						FallbackModel: "model",
						IncludeUsage:  true,
					})
					result := encoder.Encode(canonical.Event{Type: canonical.EventUsage, Usage: &test.usage})
					if result.OK || result.Value != nil || !containsDiagnostic(result.Diagnostics, canonical.DiagnosticInvalidCacheUsage) {
						t.Fatalf("result = %#v", result)
					}
				})
			}
		})
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

func TestStreamEncoderDeltaFrameMatchesMarshalPath(t *testing.T) {
	for _, test := range []struct {
		name         string
		field        string
		delta        string
		includeUsage bool
	}{
		{name: "content", field: "content", delta: "hello"},
		{name: "refusal", field: "refusal", delta: "cannot help", includeUsage: true},
		{name: "escaping", field: "content", delta: "<script>\n\"quoted\" \\ slash & \u2028 世界", includeUsage: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			encoder := NewStreamEncoder(StreamEncodeOptions{
				Created:       123,
				FallbackID:    "id",
				FallbackModel: "model",
				IncludeUsage:  test.includeUsage,
			})
			base := append([]byte(nil), encoder.deltaChunkBase()...)
			got := encoder.deltaFrame(test.field, test.delta)
			want := encoder.chunkFrame(map[string]any{test.field: test.delta}, nil, nil)
			if !bytes.Equal(got, want) {
				t.Fatalf("delta frame mismatch\ngot:  %s\nwant: %s", got, want)
			}
			if !bytes.Equal(base, encoder.deltaBase) {
				t.Fatalf("cached delta base was mutated\nbefore: %s\nafter:  %s", base, encoder.deltaBase)
			}
		})
	}
}

func TestStreamEncoderDeltaFrameFallsBackWhenSJSONFails(t *testing.T) {
	encoder := NewStreamEncoder(StreamEncodeOptions{Created: 123, FallbackID: "id", FallbackModel: "model", IncludeUsage: true})
	encoder.deltaBase = []byte(`[]`)
	got := encoder.deltaFrame("content", "hello")
	want := encoder.chunkFrame(map[string]any{"content": "hello"}, nil, nil)
	if !bytes.Equal(got, want) {
		t.Fatalf("fallback frame mismatch\ngot:  %s\nwant: %s", got, want)
	}

	got = encoder.deltaFrame("content.injected", "blocked")
	want = encoder.chunkFrame(map[string]any{"content.injected": "blocked"}, nil, nil)
	if !bytes.Equal(got, want) {
		t.Fatalf("unrecognized path fallback mismatch\ngot:  %s\nwant: %s", got, want)
	}
}

func joinFrames(frames [][]byte) string {
	var builder strings.Builder
	for _, frame := range frames {
		builder.Write(frame)
	}
	return builder.String()
}
