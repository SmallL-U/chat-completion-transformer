package openairesponses

import (
	"encoding/json"
	"testing"

	"chat-completion-transformer/internal/canonical"
)

func TestDecodeResponseTraversesEveryOutputItem(t *testing.T) {
	result := DecodeResponse([]byte(`{
  "id":"resp_1",
  "created_at":123,
  "model":"gpt-test",
  "status":"completed",
  "output":[
    {"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"hidden"}]},
    {"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[
      {"type":"output_text","text":"hello","annotations":[]},
      {"type":"refusal","refusal":"cannot do that"},
      {"type":"future_content","value":7}
    ]},
    {"type":"function_call","id":"fc_1","call_id":"call_1","name":"weather","arguments":"{\"city\":\"Beijing\"}"},
    {"type":"function_call","id":"fc_2","call_id":"call_2","name":"time","arguments":"not-json"},
    {"type":"web_search_call","id":"web_1","status":"completed"}
  ],
  "usage":{"input_tokens":10,"output_tokens":4,"total_tokens":14,"output_tokens_details":{"reasoning_tokens":2}},
  "metadata":{"tenant":"demo"}
}`))
	if !result.OK || result.Value == nil {
		t.Fatalf("DecodeResponse() = %#v", result)
	}
	response := *result.Value
	if response.ID != "resp_1" || response.CreatedAt == nil || *response.CreatedAt != 123 {
		t.Fatalf("response identity = %#v", response)
	}
	if len(response.Outputs) != 1 {
		t.Fatalf("outputs = %#v", response.Outputs)
	}
	output := response.Outputs[0]
	if output.FinishReason != canonical.FinishReasonToolCalls {
		t.Fatalf("finish reason = %q", output.FinishReason)
	}
	if len(output.Content) != 3 || output.Content[0].Kind != canonical.PartText || output.Content[1].Kind != canonical.PartRefusal || output.Content[2].Kind != canonical.PartOpaque {
		t.Fatalf("content = %#v", output.Content)
	}
	if len(output.ToolCalls) != 2 || output.ToolCalls[0].ArgumentsParsed == nil || output.ToolCalls[1].ArgumentsParsed != nil {
		t.Fatalf("tool calls = %#v", output.ToolCalls)
	}
	if len(output.ProviderItems) != 5 {
		t.Fatalf("provider items = %d, want unknown items and raw known-item metadata", len(output.ProviderItems))
	}
	if response.Usage == nil || response.Usage.TotalTokens == nil || *response.Usage.TotalTokens != 14 {
		t.Fatalf("usage = %#v", response.Usage)
	}
	if _, exists := response.Usage.Extensions["output_tokens_details"]; !exists {
		t.Fatalf("usage extensions = %#v", response.Usage.Extensions)
	}
	if _, exists := response.Extensions["metadata"]; !exists {
		t.Fatalf("response extensions = %#v", response.Extensions)
	}
}

func TestDecodeResponsePreservesUnknownFieldsOnKnownItems(t *testing.T) {
	result := DecodeResponse([]byte(`{
  "id":"resp",
  "status":"completed",
  "output":[
    {"type":"message","id":"msg","future_message":{"x":1},"content":[
      {"type":"output_text","text":"hello","future_text":7}
    ]},
    {"type":"function_call","id":"fc","call_id":"call","name":"lookup","arguments":"{}","future_call":true}
  ]
}`))
	if !result.OK || result.Value == nil {
		t.Fatalf("DecodeResponse() = %#v", result)
	}
	output := result.Value.Outputs[0]
	if len(output.ProviderItems) != 2 {
		t.Fatalf("provider items = %#v", output.ProviderItems)
	}
	message, err := canonical.DecodeObject(output.ProviderItems[0])
	if err != nil || message["future_message"] == nil {
		t.Fatalf("preserved message = %s, err = %v", output.ProviderItems[0], err)
	}
	var content []canonical.Object
	if err := json.Unmarshal(message["content"], &content); err != nil || content[0]["future_text"] == nil {
		t.Fatalf("preserved output_text = %s, err = %v", message["content"], err)
	}
	call, err := canonical.DecodeObject(output.ProviderItems[1])
	if err != nil || call["future_call"] == nil || call["id"] == nil {
		t.Fatalf("preserved function call = %s, err = %v", output.ProviderItems[1], err)
	}
}

func TestDecodeResponseMapsStatuses(t *testing.T) {
	for _, test := range []struct {
		name         string
		status       string
		extra        string
		wantReason   canonical.FinishReason
		wantProvider string
	}{
		{name: "completed", status: "completed", wantReason: canonical.FinishReasonStop},
		{name: "max output incomplete", status: "incomplete", extra: `,"incomplete_details":{"reason":"max_output_tokens"}`, wantReason: canonical.FinishReasonLength, wantProvider: "max_output_tokens"},
		{name: "content filter incomplete", status: "incomplete", extra: `,"incomplete_details":{"reason":"content_filter"}`, wantReason: canonical.FinishReasonContentFilter, wantProvider: "content_filter"},
		{name: "failed", status: "failed", extra: `,"error":{"code":"server_error","message":"failed"}`, wantReason: canonical.FinishReasonError, wantProvider: "failed"},
		{name: "queued", status: "queued", wantReason: canonical.FinishReasonUnknown, wantProvider: "queued"},
	} {
		t.Run(test.name, func(t *testing.T) {
			raw := `{"id":"resp","status":"` + test.status + `","output":[]` + test.extra + `}`
			result := DecodeResponse([]byte(raw))
			if !result.OK || result.Value == nil {
				t.Fatalf("DecodeResponse() = %#v", result)
			}
			output := result.Value.Outputs[0]
			if output.FinishReason != test.wantReason {
				t.Fatalf("reason = %q, want %q", output.FinishReason, test.wantReason)
			}
			provider := ""
			if output.ProviderReason != nil {
				provider = *output.ProviderReason
			}
			if provider != test.wantProvider {
				t.Fatalf("provider reason = %q, want %q", provider, test.wantProvider)
			}
		})
	}
}

func TestDecodeResponseReturnsDiagnosticsForProtocolErrors(t *testing.T) {
	for _, raw := range []string{
		`not-json`,
		`{"id":"resp","output":null}`,
		`{"id":"resp","output":[7]}`,
		`{"id":"resp","output":[{"type":"function_call","call_id":7}]}`,
	} {
		result := DecodeResponse([]byte(raw))
		if result.OK {
			t.Fatalf("DecodeResponse(%q) unexpectedly succeeded: %#v", raw, result)
		}
		assertDiagnosticCode(t, result.Diagnostics, DiagnosticInvalidResponse)
	}
}

func TestDecodeResponseMapsPromptCacheUsageDetails(t *testing.T) {
	result := DecodeResponse([]byte(`{
  "id":"resp",
  "status":"completed",
  "output":[],
  "usage":{
    "input_tokens":20,
    "output_tokens":5,
    "total_tokens":25,
    "input_tokens_details":{
      "cached_tokens":12,
      "cache_write_tokens":0,
      "future_detail":7
    }
  }
}`))
	if !result.OK || result.Value == nil || result.Value.Usage == nil {
		t.Fatalf("DecodeResponse() = %#v", result)
	}
	usage := result.Value.Usage
	if usage.CachedInputTokens == nil || *usage.CachedInputTokens != 12 {
		t.Fatalf("cached input tokens = %#v", usage.CachedInputTokens)
	}
	if usage.CacheWriteInputTokens == nil || *usage.CacheWriteInputTokens != 0 {
		t.Fatalf("cache write input tokens = %#v", usage.CacheWriteInputTokens)
	}
	if usage.InputTokens == nil || *usage.InputTokens != 20 || usage.TotalTokens == nil || *usage.TotalTokens != 25 {
		t.Fatalf("aggregate usage changed = %#v", usage)
	}
	details, err := canonical.DecodeObject(usage.Extensions["input_tokens_details"])
	if err != nil || len(details) != 1 || details["future_detail"] == nil {
		t.Fatalf("residual input token details = %s, err = %v", usage.Extensions["input_tokens_details"], err)
	}
}

func TestDecodeResponseConsumesKnownPromptCacheUsageDetails(t *testing.T) {
	result := DecodeResponse([]byte(`{
  "id":"resp",
  "status":"completed",
  "output":[],
  "usage":{
    "input_tokens":3,
    "output_tokens":2,
    "total_tokens":5,
    "input_tokens_details":{"cached_tokens":0}
  }
}`))
	if !result.OK || result.Value == nil || result.Value.Usage == nil {
		t.Fatalf("DecodeResponse() = %#v", result)
	}
	usage := result.Value.Usage
	if usage.CachedInputTokens == nil || *usage.CachedInputTokens != 0 {
		t.Fatalf("cached input tokens = %#v", usage.CachedInputTokens)
	}
	if usage.CacheWriteInputTokens != nil {
		t.Fatalf("unreported cache write tokens = %#v", usage.CacheWriteInputTokens)
	}
	if _, exists := usage.Extensions["input_tokens_details"]; exists {
		t.Fatalf("known details were retained as extensions: %#v", usage.Extensions)
	}
}

func TestDecodeResponseRejectsInvalidPromptCacheUsage(t *testing.T) {
	for _, test := range []struct {
		name    string
		details string
	}{
		{name: "details null", details: `null`},
		{name: "details type", details: `[]`},
		{name: "cached null", details: `{"cached_tokens":null}`},
		{name: "cached type", details: `{"cached_tokens":"1"}`},
		{name: "cached negative", details: `{"cached_tokens":-1}`},
		{name: "write null", details: `{"cache_write_tokens":null}`},
		{name: "write negative", details: `{"cache_write_tokens":-1}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			raw := `{"id":"resp","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,"input_tokens_details":` + test.details + `}}`
			result := DecodeResponse([]byte(raw))
			if result.OK || result.Value != nil {
				t.Fatalf("invalid cache usage unexpectedly succeeded: %#v", result)
			}
			assertDiagnosticCode(t, result.Diagnostics, canonical.DiagnosticInvalidCacheUsage)
		})
	}
}

func TestDecodeResponseRejectsNegativeAggregateUsage(t *testing.T) {
	result := DecodeResponse([]byte(`{"id":"resp","status":"completed","output":[],"usage":{"input_tokens":-1}}`))
	if result.OK || result.Value != nil {
		t.Fatalf("negative aggregate usage unexpectedly succeeded: %#v", result)
	}
	assertDiagnosticCode(t, result.Diagnostics, DiagnosticInvalidResponse)
}
