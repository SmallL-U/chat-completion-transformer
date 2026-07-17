package anthropicmessages

import (
	"bytes"
	"encoding/json"
	"fmt"

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
	usage := canonical.Usage{}
	usage.InputTokens = responseInt64(takeRaw(object, "input_tokens"), "usage.input_tokens", diagnostics)
	usage.OutputTokens = responseInt64(takeRaw(object, "output_tokens"), "usage.output_tokens", diagnostics)
	if usage.InputTokens != nil && usage.OutputTokens != nil {
		total := *usage.InputTokens + *usage.OutputTokens
		usage.TotalTokens = &total
	}
	usage.Extensions = cloneObject(object)
	return &usage
}

func responseInt64(raw json.RawMessage, path string, diagnostics *[]canonical.Diagnostic) *int64 {
	var value int64
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
