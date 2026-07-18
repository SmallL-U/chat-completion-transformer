package chatcompletions

import (
	"encoding/json"
	"testing"

	"chat-completion-transformer/internal/canonical"
)

func TestEncodeResponsePreservesTextToolsAndUsage(t *testing.T) {
	model := "target-model"
	inputTokens := int64(10)
	outputTokens := int64(5)
	totalTokens := int64(15)
	result := EncodeResponse(canonical.Response{
		ID:    "response_1",
		Model: &model,
		Outputs: []canonical.Output{{
			Index:        0,
			Content:      []canonical.Part{{Kind: canonical.PartText, Text: "answer"}},
			ToolCalls:    []canonical.ToolCall{{ID: "call_1", Name: "lookup", ArgumentsRaw: `{"q":"x"}`}},
			FinishReason: canonical.FinishReasonToolCalls,
		}},
		Usage: &canonical.Usage{InputTokens: &inputTokens, OutputTokens: &outputTokens, TotalTokens: &totalTokens},
	}, ResponseEncodeOptions{Mode: canonical.ModeStrict, Created: 123})

	if !result.OK || result.Value == nil || !result.Lossless {
		t.Fatalf("result = %#v", result)
	}
	var response struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		Choices []struct {
			Message struct {
				Content   *string `json:"content"`
				ToolCalls []struct {
					ID       string `json:"id"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage map[string]int64 `json:"usage"`
	}
	if err := json.Unmarshal(*result.Value, &response); err != nil {
		t.Fatal(err)
	}
	if response.ID != "response_1" || response.Object != "chat.completion" || response.Created != 123 {
		t.Fatalf("response = %#v", response)
	}
	choice := response.Choices[0]
	if choice.Message.Content == nil || *choice.Message.Content != "answer" || choice.FinishReason != "tool_calls" {
		t.Fatalf("choice = %#v", choice)
	}
	if len(choice.Message.ToolCalls) != 1 || choice.Message.ToolCalls[0].Function.Arguments != `{"q":"x"}` {
		t.Fatalf("tool calls = %#v", choice.Message.ToolCalls)
	}
	if response.Usage["total_tokens"] != 15 {
		t.Fatalf("usage = %#v", response.Usage)
	}
}

func TestEncodeResponseDoesNotCallRefusalContentFilter(t *testing.T) {
	model := "target-model"
	response := canonical.Response{
		ID:    "response_1",
		Model: &model,
		Outputs: []canonical.Output{{
			Content:      []canonical.Part{{Kind: canonical.PartRefusal, Text: "cannot help"}},
			FinishReason: canonical.FinishReasonRefusal,
		}},
	}

	strict := EncodeResponse(response, ResponseEncodeOptions{Mode: canonical.ModeStrict, Created: 1})
	if strict.OK || !containsDiagnostic(strict.Diagnostics, diagnosticFinishReasonLossy) {
		t.Fatalf("strict = %#v", strict)
	}

	compatible := EncodeResponse(response, ResponseEncodeOptions{Mode: canonical.ModeCompatible, Created: 1})
	if !compatible.OK || compatible.Value == nil || compatible.Lossless {
		t.Fatalf("compatible = %#v", compatible)
	}
	var value struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Refusal string `json:"refusal"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(*compatible.Value, &value); err != nil {
		t.Fatal(err)
	}
	if value.Choices[0].FinishReason != "stop" || value.Choices[0].Message.Refusal != "cannot help" {
		t.Fatalf("value = %#v", value)
	}
}

func TestEncodeResponseMapsContentFilterExactly(t *testing.T) {
	model := "target-model"
	result := EncodeResponse(canonical.Response{
		ID:    "response_1",
		Model: &model,
		Outputs: []canonical.Output{{
			FinishReason: canonical.FinishReasonContentFilter,
		}},
	}, ResponseEncodeOptions{Mode: canonical.ModeStrict, Created: 1})
	if !result.OK || result.Value == nil || !result.Lossless {
		t.Fatalf("result = %#v", result)
	}

	var response struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(*result.Value, &response); err != nil {
		t.Fatal(err)
	}
	if len(response.Choices) != 1 || response.Choices[0].FinishReason != "content_filter" {
		t.Fatalf("response = %#v", response)
	}
}

func TestEncodeResponseIncludesTextRefusalToolsAndProviderReasonDiagnostic(t *testing.T) {
	model := "target-model"
	providerReason := "provider \"quoted\"\nreason"
	result := EncodeResponse(canonical.Response{
		ID:    "response_1",
		Model: &model,
		Outputs: []canonical.Output{{
			Index: 0,
			Content: []canonical.Part{
				{Kind: canonical.PartText, Text: "answer"},
				{Kind: canonical.PartRefusal, Text: "cannot disclose"},
			},
			ToolCalls:      []canonical.ToolCall{{ID: "call_1", Name: "lookup", ArgumentsRaw: `{"q":"x"}`}},
			FinishReason:   canonical.FinishReasonToolCalls,
			ProviderReason: &providerReason,
		}},
	}, ResponseEncodeOptions{Mode: canonical.ModeCompatible, Created: 123})
	if !result.OK || result.Value == nil || result.Lossless {
		t.Fatalf("result = %#v", result)
	}

	var value struct {
		Choices []struct {
			Message struct {
				Content   string `json:"content"`
				Refusal   string `json:"refusal"`
				ToolCalls []struct {
					ID       string `json:"id"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(*result.Value, &value); err != nil {
		t.Fatal(err)
	}
	choice := value.Choices[0]
	if choice.Message.Content != "answer" || choice.Message.Refusal != "cannot disclose" || choice.FinishReason != "tool_calls" {
		t.Fatalf("choice = %#v", choice)
	}
	if len(choice.Message.ToolCalls) != 1 || choice.Message.ToolCalls[0].ID != "call_1" || choice.Message.ToolCalls[0].Function.Arguments != `{"q":"x"}` {
		t.Fatalf("tool calls = %#v", choice.Message.ToolCalls)
	}

	for _, item := range result.Diagnostics {
		if item.Code != diagnosticProviderReasonLossy {
			continue
		}
		if !json.Valid(item.SourceValue) {
			t.Fatalf("provider reason source is not JSON: %q", item.SourceValue)
		}
		var got string
		if err := json.Unmarshal(item.SourceValue, &got); err != nil || got != providerReason {
			t.Fatalf("provider reason = %q, err = %v", got, err)
		}
		return
	}
	t.Fatalf("provider reason diagnostic missing: %#v", result.Diagnostics)
}

func TestEncodeResponseMapsUnknownFinishReasonWithDiagnostic(t *testing.T) {
	model := "target-model"
	result := EncodeResponse(canonical.Response{
		ID:    "response_1",
		Model: &model,
		Outputs: []canonical.Output{{
			Index:        0,
			FinishReason: canonical.FinishReason("future_reason"),
		}},
	}, ResponseEncodeOptions{Mode: canonical.ModeCompatible, Created: 123})
	if !result.OK || result.Value == nil || !containsDiagnostic(result.Diagnostics, diagnosticFinishReasonLossy) {
		t.Fatalf("result = %#v", result)
	}
	var value struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(*result.Value, &value); err != nil {
		t.Fatal(err)
	}
	if value.Choices[0].FinishReason != "stop" {
		t.Fatalf("value = %#v", value)
	}
}

func TestEncodeResponseRejectsMissingEnvelope(t *testing.T) {
	result := EncodeResponse(canonical.Response{}, ResponseEncodeOptions{Mode: canonical.ModeCompatible})
	if result.OK || !canonical.HasErrors(result.Diagnostics) {
		t.Fatalf("result = %#v", result)
	}
}

func TestEncodeResponseWritesPromptTokenDetails(t *testing.T) {
	model := "target-model"
	input := int64(1200)
	output := int64(80)
	total := int64(1280)
	cached := int64(900)
	write := int64(0)
	result := EncodeResponse(canonical.Response{
		ID:    "response_1",
		Model: &model,
		Outputs: []canonical.Output{{
			Index:        0,
			FinishReason: canonical.FinishReasonStop,
		}},
		Usage: &canonical.Usage{
			InputTokens:           &input,
			OutputTokens:          &output,
			TotalTokens:           &total,
			CachedInputTokens:     &cached,
			CacheWriteInputTokens: &write,
			Extensions: canonical.Object{
				canonical.UsageExtensionAnthropicCacheCreation: json.RawMessage(`{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":0}`),
			},
		},
	}, ResponseEncodeOptions{Mode: canonical.ModeStrict, Created: 1})
	if !result.OK || result.Value == nil || !result.Lossless {
		t.Fatalf("result = %#v", result)
	}

	var response struct {
		Usage struct {
			PromptTokens        int64 `json:"prompt_tokens"`
			CompletionTokens    int64 `json:"completion_tokens"`
			TotalTokens         int64 `json:"total_tokens"`
			PromptTokensDetails struct {
				CachedTokens     *int64 `json:"cached_tokens"`
				CacheWriteTokens *int64 `json:"cache_write_tokens"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(*result.Value, &response); err != nil {
		t.Fatal(err)
	}
	if response.Usage.PromptTokens != input || response.Usage.TotalTokens != total {
		t.Fatalf("usage = %#v", response.Usage)
	}
	if response.Usage.PromptTokensDetails.CachedTokens == nil || *response.Usage.PromptTokensDetails.CachedTokens != cached {
		t.Fatalf("details = %#v", response.Usage.PromptTokensDetails)
	}
	if response.Usage.PromptTokensDetails.CacheWriteTokens == nil || *response.Usage.PromptTokensDetails.CacheWriteTokens != 0 {
		t.Fatalf("details = %#v", response.Usage.PromptTokensDetails)
	}
}

func TestEncodeResponseOmitsUnreportedPromptTokenDetail(t *testing.T) {
	model := "target-model"
	cached := int64(0)
	result := EncodeResponse(canonical.Response{
		ID:      "response_1",
		Model:   &model,
		Outputs: []canonical.Output{{FinishReason: canonical.FinishReasonStop}},
		Usage:   &canonical.Usage{CachedInputTokens: &cached},
	}, ResponseEncodeOptions{Mode: canonical.ModeStrict, Created: 1})
	if !result.OK || result.Value == nil {
		t.Fatalf("result = %#v", result)
	}
	var response struct {
		Usage struct {
			Details map[string]json.RawMessage `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(*result.Value, &response); err != nil {
		t.Fatal(err)
	}
	if string(response.Usage.Details["cached_tokens"]) != "0" {
		t.Fatalf("details = %#v", response.Usage.Details)
	}
	if _, exists := response.Usage.Details["cache_write_tokens"]; exists {
		t.Fatalf("details = %#v", response.Usage.Details)
	}
}

func TestEncodeResponseRejectsInvalidPromptCacheDetails(t *testing.T) {
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
					model := "model"
					result := EncodeResponse(canonical.Response{
						ID:    "response",
						Model: &model,
						Outputs: []canonical.Output{{
							Index:        0,
							FinishReason: canonical.FinishReasonStop,
						}},
						Usage: &test.usage,
					}, ResponseEncodeOptions{Mode: mode})
					if result.OK || result.Value != nil || !containsDiagnostic(result.Diagnostics, canonical.DiagnosticInvalidCacheUsage) {
						t.Fatalf("result = %#v", result)
					}
				})
			}
		})
	}
}

func TestEncodeResponseStillRejectsUnknownUsageExtensionsInStrictMode(t *testing.T) {
	model := "target-model"
	result := EncodeResponse(canonical.Response{
		ID:      "response_1",
		Model:   &model,
		Outputs: []canonical.Output{{FinishReason: canonical.FinishReasonStop}},
		Usage: &canonical.Usage{Extensions: canonical.Object{
			"future_usage": json.RawMessage(`true`),
		}},
	}, ResponseEncodeOptions{Mode: canonical.ModeStrict, Created: 1})
	if result.OK || !containsDiagnostic(result.Diagnostics, diagnosticResponseExtensionLossy) {
		t.Fatalf("result = %#v", result)
	}
}
