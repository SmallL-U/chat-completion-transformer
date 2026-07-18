package anthropicmessages

import (
	"encoding/json"
	"fmt"
	"testing"

	"chat-completion-transformer/internal/canonical"
)

func TestDecodeResponseMapsEveryStopReason(t *testing.T) {
	tests := []struct {
		provider  string
		canonical canonical.FinishReason
	}{
		{provider: "end_turn", canonical: canonical.FinishReasonStop},
		{provider: "stop_sequence", canonical: canonical.FinishReasonStop},
		{provider: "max_tokens", canonical: canonical.FinishReasonLength},
		{provider: "model_context_window_exceeded", canonical: canonical.FinishReasonLength},
		{provider: "tool_use", canonical: canonical.FinishReasonToolCalls},
		{provider: "refusal", canonical: canonical.FinishReasonRefusal},
		{provider: "pause_turn", canonical: canonical.FinishReasonPause},
	}
	for _, test := range tests {
		t.Run(test.provider, func(t *testing.T) {
			input := []byte(fmt.Sprintf(`{
				"id":"msg_1",
				"type":"message",
				"role":"assistant",
				"model":"claude-test",
				"content":[],
				"stop_reason":%q,
				"stop_sequence":null,
				"usage":{"input_tokens":2,"output_tokens":3}
			}`, test.provider))
			result := DecodeResponse(input)
			if !result.OK || result.Value == nil {
				t.Fatalf("result = %#v", result)
			}
			output := result.Value.Outputs[0]
			if output.FinishReason != test.canonical || output.ProviderReason == nil || *output.ProviderReason != test.provider {
				t.Fatalf("output = %#v", output)
			}
		})
	}
}

func TestDecodeResponseContentUsageAndOpaqueFields(t *testing.T) {
	input := []byte(`{
		"id":"msg_1",
		"type":"message",
		"role":"assistant",
		"model":"claude-test",
		"content":[
			{"type":"text","text":"hello"},
			{"type":"tool_use","id":"call_1","name":"lookup","input":{"q":"x"}},
			{"type":"thinking","thinking":"secret"}
		],
		"stop_reason":"tool_use",
		"stop_sequence":"END",
		"stop_details":{"type":"refusal","reason":"policy"},
		"usage":{
			"input_tokens":2,
			"output_tokens":3,
			"cache_creation_input_tokens":2,
			"cache_read_input_tokens":1,
			"cache_creation":{"ephemeral_5m_input_tokens":2,"ephemeral_1h_input_tokens":0}
		},
		"container":{"id":"container_1"}
	}`)
	result := DecodeResponse(input)
	if !result.OK || result.Value == nil {
		t.Fatalf("result = %#v", result)
	}
	response := *result.Value
	output := response.Outputs[0]
	if len(output.Content) != 1 || output.Content[0].Text != "hello" {
		t.Fatalf("content = %#v", output.Content)
	}
	if len(output.ToolCalls) != 1 || output.ToolCalls[0].ArgumentsRaw != `{"q":"x"}` || output.ToolCalls[0].ArgumentsParsed == nil {
		t.Fatalf("tool calls = %#v", output.ToolCalls)
	}
	if len(output.ProviderItems) != 1 {
		t.Fatalf("provider items = %#v", output.ProviderItems)
	}
	if string(output.Extensions["stop_sequence"]) != `"END"` || string(output.Extensions["stop_details"]) != `{"type":"refusal","reason":"policy"}` {
		t.Fatalf("output extensions = %#v", output.Extensions)
	}
	if response.Usage == nil || response.Usage.InputTokens == nil || *response.Usage.InputTokens != 5 || response.Usage.TotalTokens == nil || *response.Usage.TotalTokens != 8 {
		t.Fatalf("usage = %#v", response.Usage)
	}
	if response.Usage.CachedInputTokens == nil || *response.Usage.CachedInputTokens != 1 || response.Usage.CacheWriteInputTokens == nil || *response.Usage.CacheWriteInputTokens != 2 {
		t.Fatalf("usage = %#v", response.Usage)
	}
	assertCacheCreationBreakdown(t, response.Usage.Extensions, 2, 0)
	if _, exists := response.Usage.Extensions["cache_read_input_tokens"]; exists {
		t.Fatalf("consumed cache read remained in extensions = %#v", response.Usage.Extensions)
	}
	if _, exists := response.Extensions["container"]; !exists {
		t.Fatalf("response extensions = %#v", response.Extensions)
	}
}

func TestDecodeResponseCacheUsageNilZeroAndOverflow(t *testing.T) {
	tests := []struct {
		name      string
		usage     string
		wantOK    bool
		wantInput int64
		wantTotal int64
		wantRead  *int64
		wantWrite *int64
	}{
		{
			name:      "cache fields omitted",
			usage:     `{"input_tokens":2,"output_tokens":3}`,
			wantOK:    true,
			wantInput: 2,
			wantTotal: 5,
		},
		{
			name:      "explicit zero cache fields",
			usage:     `{"input_tokens":2,"output_tokens":3,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}`,
			wantOK:    true,
			wantInput: 2,
			wantTotal: 5,
			wantRead:  int64Pointer(0),
			wantWrite: int64Pointer(0),
		},
		{
			name:   "negative cache write",
			usage:  `{"input_tokens":2,"output_tokens":3,"cache_creation_input_tokens":-1}`,
			wantOK: false,
		},
		{
			name:   "null cache read",
			usage:  `{"input_tokens":2,"output_tokens":3,"cache_read_input_tokens":null}`,
			wantOK: false,
		},
		{
			name:   "input cache overflow",
			usage:  `{"input_tokens":9223372036854775807,"output_tokens":0,"cache_read_input_tokens":1}`,
			wantOK: false,
		},
		{
			name:   "total overflow",
			usage:  `{"input_tokens":9223372036854775807,"output_tokens":1}`,
			wantOK: false,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			input := []byte(`{
				"id":"msg_1","type":"message","role":"assistant","model":"claude-test",
				"content":[],"stop_reason":"end_turn","usage":` + test.usage + `
			}`)
			result := DecodeResponse(input)
			if result.OK != test.wantOK {
				t.Fatalf("result = %#v", result)
			}
			if !test.wantOK {
				if !hasDiagnostic(result.Diagnostics, canonical.DiagnosticInvalidCacheUsage) {
					t.Fatalf("diagnostics = %#v", result.Diagnostics)
				}
				return
			}
			usage := result.Value.Usage
			if usage == nil || usage.InputTokens == nil || *usage.InputTokens != test.wantInput || usage.TotalTokens == nil || *usage.TotalTokens != test.wantTotal {
				t.Fatalf("usage = %#v", usage)
			}
			assertOptionalInt64(t, "cache read", usage.CachedInputTokens, test.wantRead)
			assertOptionalInt64(t, "cache write", usage.CacheWriteInputTokens, test.wantWrite)
		})
	}
}

func TestDecodeResponseRejectsInvalidCacheCreationBreakdown(t *testing.T) {
	tests := []struct {
		name      string
		breakdown string
	}{
		{name: "null", breakdown: `null`},
		{name: "array", breakdown: `[]`},
		{name: "string", breakdown: `"invalid"`},
		{name: "five-minute null", breakdown: `{"ephemeral_5m_input_tokens":null}`},
		{name: "five-minute negative", breakdown: `{"ephemeral_5m_input_tokens":-1}`},
		{name: "one-hour wrong type", breakdown: `{"ephemeral_1h_input_tokens":"1"}`},
		{name: "one-hour overflow", breakdown: `{"ephemeral_1h_input_tokens":9223372036854775808}`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			input := []byte(`{
				"id":"msg_1","type":"message","role":"assistant","model":"claude-test",
				"content":[],"stop_reason":"end_turn",
				"usage":{"input_tokens":1,"output_tokens":1,"cache_creation":` + test.breakdown + `}
			}`)
			result := DecodeResponse(input)
			if result.OK || result.Value != nil || !hasDiagnostic(result.Diagnostics, canonical.DiagnosticInvalidCacheUsage) {
				t.Fatalf("result = %#v", result)
			}
		})
	}
}

func assertOptionalInt64(t *testing.T, name string, got, want *int64) {
	t.Helper()
	if got == nil && want == nil {
		return
	}
	if got == nil || want == nil || *got != *want {
		t.Fatalf("%s = %v, want %v", name, got, want)
	}
}

func int64Pointer(value int64) *int64 {
	return &value
}

func assertCacheCreationBreakdown(t *testing.T, extensions canonical.Object, fiveMinute, oneHour int64) {
	t.Helper()
	var breakdown struct {
		FiveMinute int64 `json:"ephemeral_5m_input_tokens"`
		OneHour    int64 `json:"ephemeral_1h_input_tokens"`
	}
	raw := extensions[canonical.UsageExtensionAnthropicCacheCreation]
	if err := json.Unmarshal(raw, &breakdown); err != nil {
		t.Fatalf("decode cache creation breakdown %s: %v", raw, err)
	}
	if breakdown.FiveMinute != fiveMinute || breakdown.OneHour != oneHour {
		t.Fatalf("cache creation breakdown = %#v", breakdown)
	}
}

func TestDecodeResponsePreservesUnknownStopReason(t *testing.T) {
	input := []byte(`{
		"id":"msg_1","type":"message","role":"assistant","model":"claude-test",
		"content":[],"stop_reason":"future_reason",
		"usage":{"input_tokens":1,"output_tokens":1}
	}`)
	result := DecodeResponse(input)
	if !result.OK || result.Value == nil || !hasDiagnostic(result.Diagnostics, diagnosticUnknownStop) {
		t.Fatalf("result = %#v", result)
	}
	output := result.Value.Outputs[0]
	if output.FinishReason != canonical.FinishReasonUnknown || output.ProviderReason == nil || *output.ProviderReason != "future_reason" {
		t.Fatalf("output = %#v", output)
	}
}

func TestDecodeResponseRejectsNonObjectToolInput(t *testing.T) {
	input := []byte(`{
		"id":"msg_1","type":"message","role":"assistant","model":"claude-test",
		"content":[{"type":"tool_use","id":"call_1","name":"lookup","input":[]}],
		"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}
	}`)
	result := DecodeResponse(input)
	if result.OK || !hasDiagnostic(result.Diagnostics, diagnosticInvalidResponse) {
		t.Fatalf("result = %#v", result)
	}
}

func TestDecodeResponseToolArgumentsRemainValidJSON(t *testing.T) {
	input := []byte(`{
		"id":"msg_1","type":"message","role":"assistant","model":"claude-test",
		"content":[{"type":"tool_use","id":"call_1","name":"lookup","input":{"number":9007199254740993}}],
		"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}
	}`)
	result := DecodeResponse(input)
	if !result.OK || result.Value == nil {
		t.Fatalf("result = %#v", result)
	}
	arguments := result.Value.Outputs[0].ToolCalls[0].ArgumentsRaw
	var decoded map[string]json.RawMessage
	if err := json.Unmarshal([]byte(arguments), &decoded); err != nil {
		t.Fatal(err)
	}
	if string(decoded["number"]) != "9007199254740993" {
		t.Fatalf("arguments = %s", arguments)
	}
}
