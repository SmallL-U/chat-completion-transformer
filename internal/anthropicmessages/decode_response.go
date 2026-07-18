package anthropicmessages

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"

	"chat-completion-transformer/internal/canonical"
)

const (
	diagnosticInvalidResponse canonical.DiagnosticCode = "invalid_anthropic_response"
	diagnosticUnknownStop     canonical.DiagnosticCode = "unknown_stop_reason"
)

// DecodeResponse validates a complete Anthropic Messages response and
// converts it to the canonical response representation.
func DecodeResponse(input []byte) canonical.Result[canonical.Response] {
	object, err := canonical.DecodeObject(input)
	if err != nil {
		return canonical.Failure[canonical.Response]([]canonical.Diagnostic{
			makeDiagnostic(canonical.SeverityError, diagnosticInvalidResponse, err.Error(), "", input),
		})
	}

	diagnostics := make([]canonical.Diagnostic, 0)
	response := canonical.Response{}
	response.ID = responseRequiredString(takeRaw(object, "id"), "id", &diagnostics)
	model := responseRequiredString(takeRaw(object, "model"), "model", &diagnostics)
	if model != "" {
		response.Model = &model
	}
	validateResponseLiteral(takeRaw(object, "type"), "message", "type", &diagnostics)
	validateResponseLiteral(takeRaw(object, "role"), "assistant", "role", &diagnostics)

	output := canonical.Output{Index: 0}
	output.Content, output.ToolCalls, output.ProviderItems = decodeResponseContent(takeRaw(object, "content"), &diagnostics)
	decodeResponseStop(object, &output, &diagnostics)
	response.Outputs = []canonical.Output{output}
	response.Usage = decodeResponseUsage(takeRaw(object, "usage"), &diagnostics)
	response.Extensions = cloneObject(object)

	if canonical.HasErrors(diagnostics) {
		return canonical.Failure[canonical.Response](diagnostics)
	}
	return canonical.Success(response, diagnostics)
}

func decodeResponseContent(raw json.RawMessage, diagnostics *[]canonical.Diagnostic) ([]canonical.Part, []canonical.ToolCall, []json.RawMessage) {
	if len(bytes.TrimSpace(raw)) == 0 {
		*diagnostics = append(*diagnostics, makeDiagnostic(canonical.SeverityError, diagnosticInvalidResponse, "content is required", "content", nil))
		return nil, nil, nil
	}

	var blocks []json.RawMessage
	if err := json.Unmarshal(raw, &blocks); err != nil || blocks == nil {
		*diagnostics = append(*diagnostics, makeDiagnostic(canonical.SeverityError, diagnosticInvalidResponse, "content must be an array", "content", raw))
		return nil, nil, nil
	}

	parts := make([]canonical.Part, 0, len(blocks))
	calls := make([]canonical.ToolCall, 0)
	providerItems := make([]json.RawMessage, 0)
	for index, block := range blocks {
		path := fmt.Sprintf("content.%d", index)
		object, err := canonical.DecodeObject(block)
		if err != nil {
			*diagnostics = append(*diagnostics, makeDiagnostic(canonical.SeverityError, diagnosticInvalidResponse, "content block must be an object", path, block))
			continue
		}
		typeName := responseRequiredString(takeRaw(object, "type"), path+".type", diagnostics)
		switch typeName {
		case "text":
			text := responseRequiredString(takeRaw(object, "text"), path+".text", diagnostics)
			parts = append(parts, canonical.Part{Kind: canonical.PartText, Text: text})
			if len(object) > 0 {
				providerItems = append(providerItems, cloneRaw(block))
			}
		case "tool_use":
			call := decodeResponseToolUse(object, path, diagnostics)
			calls = append(calls, call)
			if len(object) > 0 {
				providerItems = append(providerItems, cloneRaw(block))
			}
		default:
			providerItems = append(providerItems, cloneRaw(block))
		}
	}
	return parts, calls, providerItems
}

func decodeResponseToolUse(object canonical.Object, path string, diagnostics *[]canonical.Diagnostic) canonical.ToolCall {
	call := canonical.ToolCall{
		ID:   responseRequiredString(takeRaw(object, "id"), path+".id", diagnostics),
		Name: responseRequiredString(takeRaw(object, "name"), path+".name", diagnostics),
	}
	input := takeRaw(object, "input")
	parsed, err := canonical.DecodeObject(input)
	if err != nil {
		*diagnostics = append(*diagnostics, makeDiagnostic(
			canonical.SeverityError,
			diagnosticInvalidResponse,
			fmt.Sprintf("tool_use input must be a JSON object: %v", err),
			path+".input",
			input,
		))
		return call
	}
	call.ArgumentsRaw = string(bytes.TrimSpace(input))
	call.ArgumentsParsed = parsed
	return call
}

func decodeResponseStop(object canonical.Object, output *canonical.Output, diagnostics *[]canonical.Diagnostic) {
	raw, exists := object["stop_reason"]
	delete(object, "stop_reason")
	if !exists || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		output.FinishReason = canonical.FinishReasonUnknown
		*diagnostics = append(*diagnostics, makeDiagnostic(canonical.SeverityError, diagnosticInvalidResponse, "stop_reason is required in a complete response", "stop_reason", raw))
	} else {
		var reason string
		if err := json.Unmarshal(raw, &reason); err != nil || reason == "" {
			output.FinishReason = canonical.FinishReasonUnknown
			*diagnostics = append(*diagnostics, makeDiagnostic(canonical.SeverityError, diagnosticInvalidResponse, "stop_reason must be a non-empty string", "stop_reason", raw))
		} else {
			mapped, known := mapStopReason(reason)
			output.FinishReason = mapped
			output.ProviderReason = &reason
			if !known {
				*diagnostics = append(*diagnostics, makeDiagnostic(canonical.SeverityWarning, diagnosticUnknownStop, fmt.Sprintf("unknown Anthropic stop reason %q was preserved", reason), "stop_reason", raw))
			}
		}
	}

	for _, name := range []string{"stop_sequence", "stop_details"} {
		raw, exists := object[name]
		delete(object, name)
		if !exists {
			continue
		}
		if output.Extensions == nil {
			output.Extensions = make(canonical.Object)
		}
		output.Extensions[name] = cloneRaw(raw)
	}
}

func mapStopReason(reason string) (canonical.FinishReason, bool) {
	switch reason {
	case "end_turn", "stop_sequence":
		return canonical.FinishReasonStop, true
	case "max_tokens", "model_context_window_exceeded":
		return canonical.FinishReasonLength, true
	case "tool_use":
		return canonical.FinishReasonToolCalls, true
	case "refusal":
		return canonical.FinishReasonRefusal, true
	case "pause_turn":
		return canonical.FinishReasonPause, true
	default:
		return canonical.FinishReasonUnknown, false
	}
}

func decodeResponseUsage(raw json.RawMessage, diagnostics *[]canonical.Diagnostic) *canonical.Usage {
	object, err := canonical.DecodeObject(raw)
	if err != nil {
		*diagnostics = append(*diagnostics, makeDiagnostic(canonical.SeverityError, diagnosticInvalidResponse, "usage must be an object", "usage", raw))
		return nil
	}
	rawInput := responseInt64(takeRaw(object, "input_tokens"), "usage.input_tokens", diagnostics)
	output := responseInt64(takeRaw(object, "output_tokens"), "usage.output_tokens", diagnostics)
	cacheWrite := takeCacheToken(object, "cache_creation_input_tokens", "usage.cache_creation_input_tokens", diagnostics)
	cacheRead := takeCacheToken(object, "cache_read_input_tokens", "usage.cache_read_input_tokens", diagnostics)
	usage := normalizeAnthropicUsage(rawInput, output, cacheWrite, cacheRead, takeUsageExtensions(object, "usage", diagnostics), "usage", diagnostics)
	return &usage
}

func takeCacheToken(object canonical.Object, name, path string, diagnostics *[]canonical.Diagnostic) *int64 {
	raw, exists := object[name]
	delete(object, name)
	if !exists {
		return nil
	}
	var value int64
	if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		*diagnostics = append(*diagnostics, makeDiagnostic(
			canonical.SeverityError,
			canonical.DiagnosticInvalidCacheUsage,
			"cache token count must be a non-negative integer",
			path,
			raw,
		))
		return nil
	}
	if err := json.Unmarshal(raw, &value); err != nil || value < 0 {
		*diagnostics = append(*diagnostics, makeDiagnostic(
			canonical.SeverityError,
			canonical.DiagnosticInvalidCacheUsage,
			"cache token count must be a non-negative integer",
			path,
			raw,
		))
		return nil
	}
	return &value
}

func takeUsageExtensions(object canonical.Object, path string, diagnostics *[]canonical.Diagnostic) canonical.Object {
	cacheCreation, hasCacheCreation := object["cache_creation"]
	delete(object, "cache_creation")
	extensions := cloneObject(object)
	if !hasCacheCreation {
		return extensions
	}
	validateCacheCreationBreakdown(cacheCreation, path+".cache_creation", diagnostics)
	if extensions == nil {
		extensions = make(canonical.Object)
	}
	extensions[canonical.UsageExtensionAnthropicCacheCreation] = cloneRaw(cacheCreation)
	return extensions
}

func validateCacheCreationBreakdown(raw json.RawMessage, path string, diagnostics *[]canonical.Diagnostic) {
	object, err := canonical.DecodeObject(raw)
	if err != nil {
		*diagnostics = append(*diagnostics, makeDiagnostic(
			canonical.SeverityError,
			canonical.DiagnosticInvalidCacheUsage,
			"cache creation breakdown must be an object",
			path,
			raw,
		))
		return
	}
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
		*diagnostics = append(*diagnostics, makeDiagnostic(
			canonical.SeverityError,
			canonical.DiagnosticInvalidCacheUsage,
			"cache creation token count must be a non-negative integer",
			path+"."+name,
			valueRaw,
		))
	}
}

func normalizeAnthropicUsage(
	rawInput *int64,
	output *int64,
	cacheWrite *int64,
	cacheRead *int64,
	extensions canonical.Object,
	path string,
	diagnostics *[]canonical.Diagnostic,
) canonical.Usage {
	usage := canonical.Usage{
		OutputTokens:          output,
		CachedInputTokens:     cacheRead,
		CacheWriteInputTokens: cacheWrite,
		Extensions:            extensions,
	}
	if rawInput == nil {
		return usage
	}

	input, ok := checkedTokenSum(*rawInput, pointerValue(cacheWrite), pointerValue(cacheRead))
	if !ok {
		*diagnostics = append(*diagnostics, makeDiagnostic(
			canonical.SeverityError,
			canonical.DiagnosticInvalidCacheUsage,
			"input and cache token counts overflow int64",
			path,
			nil,
		))
		return usage
	}
	usage.InputTokens = &input
	if output == nil {
		return usage
	}

	total, ok := checkedTokenSum(input, *output)
	if !ok {
		*diagnostics = append(*diagnostics, makeDiagnostic(
			canonical.SeverityError,
			canonical.DiagnosticInvalidCacheUsage,
			"input and output token counts overflow int64",
			path,
			nil,
		))
		return usage
	}
	usage.TotalTokens = &total
	return usage
}

func pointerValue(value *int64) int64 {
	if value == nil {
		return 0
	}
	return *value
}

func checkedTokenSum(values ...int64) (int64, bool) {
	var total int64
	for _, value := range values {
		if value < 0 || total > math.MaxInt64-value {
			return 0, false
		}
		total += value
	}
	return total, true
}

func responseInt64(raw json.RawMessage, path string, diagnostics *[]canonical.Diagnostic) *int64 {
	var value int64
	if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		*diagnostics = append(*diagnostics, makeDiagnostic(canonical.SeverityError, diagnosticInvalidResponse, "token count must be a non-negative integer", path, raw))
		return nil
	}
	if err := json.Unmarshal(raw, &value); err != nil || value < 0 {
		*diagnostics = append(*diagnostics, makeDiagnostic(canonical.SeverityError, diagnosticInvalidResponse, "token count must be a non-negative integer", path, raw))
		return nil
	}
	return &value
}

func responseRequiredString(raw json.RawMessage, path string, diagnostics *[]canonical.Diagnostic) string {
	var value string
	if err := json.Unmarshal(raw, &value); err != nil || value == "" {
		*diagnostics = append(*diagnostics, makeDiagnostic(canonical.SeverityError, diagnosticInvalidResponse, "value must be a non-empty string", path, raw))
		return ""
	}
	return value
}

func validateResponseLiteral(raw json.RawMessage, expected, path string, diagnostics *[]canonical.Diagnostic) {
	var value string
	if err := json.Unmarshal(raw, &value); err == nil && value == expected {
		return
	}
	*diagnostics = append(*diagnostics, makeDiagnostic(canonical.SeverityError, diagnosticInvalidResponse, fmt.Sprintf("value must be %q", expected), path, raw))
}

func takeRaw(object canonical.Object, name string) json.RawMessage {
	value := object[name]
	delete(object, name)
	return value
}

func cloneObject(object canonical.Object) canonical.Object {
	if len(object) == 0 {
		return nil
	}
	cloned := make(canonical.Object, len(object))
	for name, raw := range object {
		cloned[name] = cloneRaw(raw)
	}
	return cloned
}
