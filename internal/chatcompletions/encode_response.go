package chatcompletions

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"chat-completion-transformer/internal/canonical"
)

const (
	diagnosticInvalidCanonicalResponse canonical.DiagnosticCode = "invalid_canonical_response"
	diagnosticResponseContentLossy     canonical.DiagnosticCode = "response_content_lossy"
	diagnosticFinishReasonLossy        canonical.DiagnosticCode = "finish_reason_lossy"
	diagnosticProviderReasonLossy      canonical.DiagnosticCode = "provider_reason_lossy"
	diagnosticResponseExtensionLossy   canonical.DiagnosticCode = "response_extension_lossy"
)

type ResponseEncodeOptions struct {
	Mode          canonical.Mode
	Created       int64
	FallbackModel string
}

// EncodeResponse converts a complete canonical provider response into a Chat
// Completions response object. Diagnostics carry details Chat cannot express.
func EncodeResponse(response canonical.Response, options ResponseEncodeOptions) canonical.Result[json.RawMessage] {
	diagnostics := make([]canonical.Diagnostic, 0)
	if response.ID == "" {
		diagnostics = append(diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidCanonicalResponse, "response ID is required", "id", nil))
	}
	if len(response.Outputs) == 0 {
		diagnostics = append(diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidCanonicalResponse, "at least one response output is required", "outputs", nil))
	}

	model := options.FallbackModel
	if response.Model != nil {
		model = *response.Model
	}
	if model == "" {
		diagnostics = append(diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidCanonicalResponse, "response model is required", "model", nil))
	}

	choices := make([]map[string]any, 0, len(response.Outputs))
	for index := range response.Outputs {
		choice, choiceDiagnostics := encodeChoice(response.Outputs[index], options.Mode)
		diagnostics = append(diagnostics, choiceDiagnostics...)
		choices = append(choices, choice)
	}

	created := options.Created
	if created == 0 && response.CreatedAt != nil {
		created = *response.CreatedAt
	}
	if created == 0 {
		created = time.Now().Unix()
	}
	value := map[string]any{
		"id":      response.ID,
		"object":  "chat.completion",
		"created": created,
		"model":   model,
		"choices": choices,
	}
	if response.Usage != nil {
		cacheUsageDiagnostics := validateCanonicalCacheUsage(*response.Usage)
		diagnostics = append(diagnostics, cacheUsageDiagnostics...)
		if !canonical.HasErrors(cacheUsageDiagnostics) {
			value["usage"] = encodeChatUsage(*response.Usage)
		}
		unknownExtensions := unknownUsageExtensions(response.Usage.Extensions)
		if len(unknownExtensions) > 0 {
			diagnostics = append(diagnostics, lossyDiagnostic(options.Mode, diagnosticResponseExtensionLossy, "provider usage extensions cannot be represented by Chat Completions", "usage.extensions", mustMarshal(unknownExtensions)))
		}
	}
	if len(response.Extensions) > 0 {
		diagnostics = append(diagnostics, lossyDiagnostic(options.Mode, diagnosticResponseExtensionLossy, "provider response extensions cannot be represented by Chat Completions", "extensions", mustMarshal(response.Extensions)))
	}

	if canonical.HasErrors(diagnostics) {
		return canonical.Failure[json.RawMessage](diagnostics)
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		diagnostics = append(diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidCanonicalResponse, fmt.Sprintf("encode Chat Completions response: %v", err), "", nil))
		return canonical.Failure[json.RawMessage](diagnostics)
	}
	raw := json.RawMessage(encoded)
	return canonical.Success(raw, diagnostics)
}

func encodeChoice(output canonical.Output, mode canonical.Mode) (map[string]any, []canonical.Diagnostic) {
	diagnostics := make([]canonical.Diagnostic, 0)
	message := map[string]any{"role": "assistant"}

	var text strings.Builder
	var refusal strings.Builder
	hasText := false
	hasRefusal := false
	for partIndex, part := range output.Content {
		switch part.Kind {
		case canonical.PartText:
			hasText = true
			text.WriteString(part.Text)
		case canonical.PartRefusal:
			hasRefusal = true
			refusal.WriteString(part.Text)
		default:
			path := fmt.Sprintf("outputs.%d.content.%d", output.Index, partIndex)
			source := part.Value
			if len(source) == 0 {
				source = mustMarshal(part)
			}
			diagnostics = append(diagnostics, lossyDiagnostic(mode, diagnosticResponseContentLossy, fmt.Sprintf("canonical %s content cannot be represented in an assistant Chat message", part.Kind), path, source))
		}
	}
	if hasText {
		message["content"] = text.String()
	} else {
		message["content"] = nil
	}
	if hasRefusal {
		message["refusal"] = refusal.String()
	}

	if len(output.ToolCalls) > 0 {
		calls := make([]map[string]any, 0, len(output.ToolCalls))
		for _, call := range output.ToolCalls {
			if call.ID == "" || call.Name == "" {
				diagnostics = append(diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidCanonicalResponse, "tool call ID and name are required", fmt.Sprintf("outputs.%d.tool_calls", output.Index), nil))
				continue
			}
			calls = append(calls, map[string]any{
				"id":   call.ID,
				"type": "function",
				"function": map[string]any{
					"name":      call.Name,
					"arguments": call.ArgumentsRaw,
				},
			})
		}
		message["tool_calls"] = calls
	}

	finishReason, finishDiagnostics := encodeFinishReason(output, mode)
	diagnostics = append(diagnostics, finishDiagnostics...)
	if len(output.ProviderItems) > 0 {
		diagnostics = append(diagnostics, lossyDiagnostic(mode, diagnosticResponseExtensionLossy, "provider-specific output items cannot be represented by Chat Completions", fmt.Sprintf("outputs.%d", output.Index), mustMarshal(output.ProviderItems)))
	}
	if len(output.Extensions) > 0 {
		diagnostics = append(diagnostics, lossyDiagnostic(mode, diagnosticResponseExtensionLossy, "provider output extensions cannot be represented by Chat Completions", fmt.Sprintf("outputs.%d.extensions", output.Index), mustMarshal(output.Extensions)))
	}

	return map[string]any{
		"index":         output.Index,
		"message":       message,
		"finish_reason": finishReason,
		"logprobs":      nil,
	}, diagnostics
}

func encodeFinishReason(output canonical.Output, mode canonical.Mode) (string, []canonical.Diagnostic) {
	diagnostics := make([]canonical.Diagnostic, 0, 1)
	var reason string
	switch output.FinishReason {
	case canonical.FinishReasonStop:
		reason = "stop"
	case canonical.FinishReasonLength:
		reason = "length"
	case canonical.FinishReasonToolCalls:
		reason = "tool_calls"
	case canonical.FinishReasonContentFilter:
		reason = "content_filter"
	case canonical.FinishReasonRefusal:
		reason = "stop"
		diagnostics = append(diagnostics, lossyDiagnostic(mode, diagnosticFinishReasonLossy, "Chat Completions has no refusal finish reason; refusal content is preserved and finish_reason is stop", fmt.Sprintf("outputs.%d.finish_reason", output.Index), nil))
	default:
		reason = "stop"
		diagnostics = append(diagnostics, lossyDiagnostic(mode, diagnosticFinishReasonLossy, fmt.Sprintf("canonical finish reason %q has no exact Chat Completions equivalent", output.FinishReason), fmt.Sprintf("outputs.%d.finish_reason", output.Index), nil))
	}
	if output.ProviderReason != nil {
		diagnostics = append(diagnostics, lossyDiagnostic(mode, diagnosticProviderReasonLossy, fmt.Sprintf("original provider finish reason %q is available only in diagnostics", *output.ProviderReason), fmt.Sprintf("outputs.%d.provider_reason", output.Index), mustMarshal(*output.ProviderReason)))
	}
	return reason, diagnostics
}

func encodeChatUsage(usage canonical.Usage) map[string]any {
	value := make(map[string]any)
	if usage.InputTokens != nil {
		value["prompt_tokens"] = *usage.InputTokens
	}
	if usage.OutputTokens != nil {
		value["completion_tokens"] = *usage.OutputTokens
	}
	if usage.TotalTokens != nil {
		value["total_tokens"] = *usage.TotalTokens
	}
	if usage.CachedInputTokens == nil && usage.CacheWriteInputTokens == nil {
		return value
	}
	details := make(map[string]any)
	if usage.CachedInputTokens != nil {
		details["cached_tokens"] = *usage.CachedInputTokens
	}
	if usage.CacheWriteInputTokens != nil {
		details["cache_write_tokens"] = *usage.CacheWriteInputTokens
	}
	value["prompt_tokens_details"] = details
	return value
}

func validateCanonicalCacheUsage(usage canonical.Usage) []canonical.Diagnostic {
	diagnostics := make([]canonical.Diagnostic, 0, 3)
	if usage.CachedInputTokens != nil && *usage.CachedInputTokens < 0 {
		diagnostics = append(diagnostics, diagnostic(
			canonical.SeverityError,
			canonical.DiagnosticInvalidCacheUsage,
			"cached input token count must be a non-negative integer",
			"usage.cached_input_tokens",
			mustMarshal(*usage.CachedInputTokens),
		))
	}
	if usage.CacheWriteInputTokens != nil && *usage.CacheWriteInputTokens < 0 {
		diagnostics = append(diagnostics, diagnostic(
			canonical.SeverityError,
			canonical.DiagnosticInvalidCacheUsage,
			"cache write input token count must be a non-negative integer",
			"usage.cache_write_input_tokens",
			mustMarshal(*usage.CacheWriteInputTokens),
		))
	}
	cacheCreation, exists := usage.Extensions[canonical.UsageExtensionAnthropicCacheCreation]
	if exists {
		diagnostics = append(diagnostics, validateCanonicalCacheCreationBreakdown(cacheCreation)...)
	}
	return diagnostics
}

func validateCanonicalCacheCreationBreakdown(raw json.RawMessage) []canonical.Diagnostic {
	path := "usage.extensions." + canonical.UsageExtensionAnthropicCacheCreation
	object, err := canonical.DecodeObject(raw)
	if err != nil {
		return []canonical.Diagnostic{diagnostic(
			canonical.SeverityError,
			canonical.DiagnosticInvalidCacheUsage,
			"cache creation breakdown must be an object",
			path,
			raw,
		)}
	}

	diagnostics := make([]canonical.Diagnostic, 0, 2)
	for _, name := range []string{"ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"} {
		valueRaw, exists := object[name]
		if !exists {
			continue
		}
		var value int64
		if !bytes.Equal(bytes.TrimSpace(valueRaw), []byte("null")) {
			if err := json.Unmarshal(valueRaw, &value); err == nil && value >= 0 {
				continue
			}
		}
		diagnostics = append(diagnostics, diagnostic(
			canonical.SeverityError,
			canonical.DiagnosticInvalidCacheUsage,
			"cache creation token count must be a non-negative integer",
			path+"."+name,
			valueRaw,
		))
	}
	return diagnostics
}

func unknownUsageExtensions(extensions canonical.Object) canonical.Object {
	unknown := make(canonical.Object)
	for name, raw := range extensions {
		if name == canonical.UsageExtensionAnthropicCacheCreation {
			continue
		}
		unknown[name] = cloneRaw(raw)
	}
	if len(unknown) == 0 {
		return nil
	}
	return unknown
}

func lossyDiagnostic(mode canonical.Mode, code canonical.DiagnosticCode, message, path string, source json.RawMessage) canonical.Diagnostic {
	severity := canonical.SeverityWarning
	if mode == canonical.ModeStrict {
		severity = canonical.SeverityError
	}
	return diagnostic(severity, code, message, path, source)
}

func mustMarshal(value any) json.RawMessage {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil
	}
	return encoded
}
