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
		"usage":{"input_tokens":2,"output_tokens":3,"cache_read_input_tokens":1},
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
	if response.Usage == nil || response.Usage.TotalTokens == nil || *response.Usage.TotalTokens != 5 {
		t.Fatalf("usage = %#v", response.Usage)
	}
	if string(response.Usage.Extensions["cache_read_input_tokens"]) != "1" {
		t.Fatalf("usage extensions = %#v", response.Usage.Extensions)
	}
	if _, exists := response.Extensions["container"]; !exists {
		t.Fatalf("response extensions = %#v", response.Extensions)
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
