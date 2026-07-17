package openairesponses

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"chat-completion-transformer/internal/canonical"
)

// DecodeResponse converts one complete OpenAI Responses object into the
// canonical response representation. Provider-specific and unknown fields are
// retained as raw JSON.
func DecodeResponse(input []byte) canonical.Result[canonical.Response] {
	object, err := canonical.DecodeObject(input)
	if err != nil {
		return canonical.Failure[canonical.Response]([]canonical.Diagnostic{
			diagnostic(canonical.SeverityError, DiagnosticInvalidResponse, err.Error(), "", input),
		})
	}

	diagnostics := make([]canonical.Diagnostic, 0)
	response := canonical.Response{}
	response.ID = decodeStringField(object, "id", "id", true, &diagnostics)
	response.CreatedAt = decodeOptionalInt64(object["created_at"], "created_at", &diagnostics)
	if modelRaw := object["model"]; hasJSONValue(modelRaw) {
		model := decodeString(modelRaw, "model", &diagnostics)
		if model != "" {
			response.Model = &model
		}
	}

	output := canonical.Output{Index: 0}
	decodeResponseOutput(object["output"], &output, &diagnostics)
	status := decodeOptionalString(object["status"], "status", &diagnostics)
	incompleteReason := decodeIncompleteReason(object["incomplete_details"], &diagnostics)
	output.FinishReason, output.ProviderReason = responseFinishReason(status, incompleteReason, object["error"], output)
	response.Outputs = []canonical.Output{output}

	if usageRaw := object["usage"]; hasJSONValue(usageRaw) {
		response.Usage = decodeUsage(usageRaw, "usage", &diagnostics)
	}

	response.Extensions = cloneObjectExcept(object, "id", "created_at", "model", "output", "usage")
	return canonical.Success(response, diagnostics)
}

func decodeResponseOutput(raw json.RawMessage, output *canonical.Output, diagnostics *[]canonical.Diagnostic) {
	if !hasJSONValue(raw) {
		*diagnostics = append(*diagnostics, diagnostic(
			canonical.SeverityError,
			DiagnosticInvalidResponse,
			"response output is required",
			"output",
			raw,
		))
		return
	}

	var items []json.RawMessage
	if err := json.Unmarshal(raw, &items); err != nil || items == nil {
		*diagnostics = append(*diagnostics, diagnostic(
			canonical.SeverityError,
			DiagnosticInvalidResponse,
			"response output must be an array",
			"output",
			raw,
		))
		return
	}

	for index, itemRaw := range items {
		item, err := canonical.DecodeObject(itemRaw)
		if err != nil {
			output.ProviderItems = append(output.ProviderItems, cloneRaw(itemRaw))
			*diagnostics = append(*diagnostics, diagnostic(
				canonical.SeverityError,
				DiagnosticInvalidResponse,
				"response output item must be an object",
				fmt.Sprintf("output.%d", index),
				itemRaw,
			))
			continue
		}

		typeName := decodeStringField(item, "type", fmt.Sprintf("output.%d.type", index), true, diagnostics)
		switch typeName {
		case "message":
			decodeMessageOutput(item, itemRaw, index, output, diagnostics)
		case "function_call":
			decodeFunctionCallOutput(item, itemRaw, index, output, diagnostics)
		default:
			output.ProviderItems = append(output.ProviderItems, cloneRaw(itemRaw))
		}
	}
}

func decodeMessageOutput(
	item canonical.Object,
	itemRaw json.RawMessage,
	outputIndex int,
	output *canonical.Output,
	diagnostics *[]canonical.Diagnostic,
) {
	preserveItem := objectHasKeysExcept(item, "type", "content")
	path := fmt.Sprintf("output.%d.content", outputIndex)
	raw := item["content"]
	var parts []json.RawMessage
	if err := json.Unmarshal(raw, &parts); err != nil || parts == nil {
		output.ProviderItems = append(output.ProviderItems, cloneRaw(itemRaw))
		*diagnostics = append(*diagnostics, diagnostic(
			canonical.SeverityError,
			DiagnosticInvalidResponse,
			"message output content must be an array",
			path,
			raw,
		))
		return
	}

	for partIndex, partRaw := range parts {
		partPath := fmt.Sprintf("%s.%d", path, partIndex)
		partObject, err := canonical.DecodeObject(partRaw)
		if err != nil {
			output.Content = append(output.Content, opaqueResponsePart(partRaw))
			*diagnostics = append(*diagnostics, diagnostic(
				canonical.SeverityError,
				DiagnosticInvalidResponse,
				"message output content part must be an object",
				partPath,
				partRaw,
			))
			continue
		}

		typeName := decodeStringField(partObject, "type", partPath+".type", true, diagnostics)
		switch typeName {
		case "output_text":
			text := decodeStringField(partObject, "text", partPath+".text", true, diagnostics)
			output.Content = append(output.Content, canonical.Part{Kind: canonical.PartText, Text: text})
			preserveItem = preserveItem || objectHasKeysExcept(partObject, "type", "text")
		case "refusal":
			refusal := decodeStringField(partObject, "refusal", partPath+".refusal", true, diagnostics)
			output.Content = append(output.Content, canonical.Part{Kind: canonical.PartRefusal, Text: refusal})
			preserveItem = preserveItem || objectHasKeysExcept(partObject, "type", "refusal")
		default:
			output.Content = append(output.Content, opaqueResponsePart(partRaw))
		}
	}
	if preserveItem {
		output.ProviderItems = append(output.ProviderItems, cloneRaw(itemRaw))
	}
}

func decodeFunctionCallOutput(
	item canonical.Object,
	itemRaw json.RawMessage,
	outputIndex int,
	output *canonical.Output,
	diagnostics *[]canonical.Diagnostic,
) {
	path := fmt.Sprintf("output.%d", outputIndex)
	arguments := decodeStringField(item, "arguments", path+".arguments", true, diagnostics)
	call := canonical.ToolCall{
		ID:           decodeStringField(item, "call_id", path+".call_id", true, diagnostics),
		Name:         decodeStringField(item, "name", path+".name", true, diagnostics),
		ArgumentsRaw: arguments,
	}
	if parsed, err := canonical.DecodeObject([]byte(arguments)); err == nil {
		call.ArgumentsParsed = parsed
	}
	output.ToolCalls = append(output.ToolCalls, call)
	if objectHasKeysExcept(item, "type", "call_id", "name", "arguments") {
		output.ProviderItems = append(output.ProviderItems, cloneRaw(itemRaw))
	}
}

func decodeUsage(raw json.RawMessage, path string, diagnostics *[]canonical.Diagnostic) *canonical.Usage {
	object, err := canonical.DecodeObject(raw)
	if err != nil {
		*diagnostics = append(*diagnostics, diagnostic(
			canonical.SeverityError,
			DiagnosticInvalidResponse,
			"response usage must be an object",
			path,
			raw,
		))
		return nil
	}

	usage := &canonical.Usage{}
	usage.InputTokens = decodeOptionalInt64(object["input_tokens"], path+".input_tokens", diagnostics)
	usage.OutputTokens = decodeOptionalInt64(object["output_tokens"], path+".output_tokens", diagnostics)
	usage.TotalTokens = decodeOptionalInt64(object["total_tokens"], path+".total_tokens", diagnostics)
	usage.Extensions = cloneObjectExcept(object, "input_tokens", "output_tokens", "total_tokens")
	return usage
}

func responseFinishReason(
	status string,
	incompleteReason string,
	errorRaw json.RawMessage,
	output canonical.Output,
) (canonical.FinishReason, *string) {
	if hasNonNullJSON(errorRaw) {
		reason := status
		if reason == "" {
			reason = "error"
		}
		return canonical.FinishReasonError, &reason
	}

	switch status {
	case "failed", "cancelled":
		reason := status
		return canonical.FinishReasonError, &reason
	case "incomplete":
		reason := incompleteReason
		if reason == "" {
			reason = status
		}
		if reason == "max_output_tokens" || reason == "max_tokens" {
			return canonical.FinishReasonLength, &reason
		}
		if reason == "content_filter" {
			return canonical.FinishReasonContentFilter, &reason
		}
		return canonical.FinishReasonUnknown, &reason
	case "queued", "in_progress":
		reason := status
		return canonical.FinishReasonUnknown, &reason
	}

	if len(output.ToolCalls) > 0 {
		return canonical.FinishReasonToolCalls, nil
	}
	for _, part := range output.Content {
		if part.Kind == canonical.PartRefusal {
			return canonical.FinishReasonRefusal, nil
		}
	}
	return canonical.FinishReasonStop, nil
}

func decodeIncompleteReason(raw json.RawMessage, diagnostics *[]canonical.Diagnostic) string {
	if !hasJSONValue(raw) {
		return ""
	}
	object, err := canonical.DecodeObject(raw)
	if err != nil {
		*diagnostics = append(*diagnostics, diagnostic(
			canonical.SeverityError,
			DiagnosticInvalidResponse,
			"incomplete_details must be an object or null",
			"incomplete_details",
			raw,
		))
		return ""
	}
	return decodeOptionalString(object["reason"], "incomplete_details.reason", diagnostics)
}

func decodeStringField(
	object canonical.Object,
	name string,
	path string,
	required bool,
	diagnostics *[]canonical.Diagnostic,
) string {
	raw, exists := object[name]
	if !exists || !hasJSONValue(raw) {
		if required {
			*diagnostics = append(*diagnostics, diagnostic(
				canonical.SeverityError,
				DiagnosticInvalidResponse,
				fmt.Sprintf("%s is required", path),
				path,
				raw,
			))
		}
		return ""
	}
	return decodeString(raw, path, diagnostics)
}

func decodeString(raw json.RawMessage, path string, diagnostics *[]canonical.Diagnostic) string {
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		*diagnostics = append(*diagnostics, diagnostic(
			canonical.SeverityError,
			DiagnosticInvalidResponse,
			fmt.Sprintf("%s must be a string", path),
			path,
			raw,
		))
	}
	return value
}

func decodeOptionalString(raw json.RawMessage, path string, diagnostics *[]canonical.Diagnostic) string {
	if !hasJSONValue(raw) {
		return ""
	}
	return decodeString(raw, path, diagnostics)
}

func decodeOptionalInt64(raw json.RawMessage, path string, diagnostics *[]canonical.Diagnostic) *int64 {
	if !hasJSONValue(raw) {
		return nil
	}
	var value int64
	if err := json.Unmarshal(raw, &value); err != nil {
		*diagnostics = append(*diagnostics, diagnostic(
			canonical.SeverityError,
			DiagnosticInvalidResponse,
			fmt.Sprintf("%s must be an integer", path),
			path,
			raw,
		))
		return nil
	}
	return &value
}

func cloneObjectExcept(object canonical.Object, names ...string) canonical.Object {
	excluded := make(map[string]struct{}, len(names))
	for _, name := range names {
		excluded[name] = struct{}{}
	}
	result := make(canonical.Object)
	for name, raw := range object {
		if _, skip := excluded[name]; skip {
			continue
		}
		result[name] = cloneRaw(raw)
	}
	if len(result) == 0 {
		return nil
	}
	return result
}

func objectHasKeysExcept(object canonical.Object, names ...string) bool {
	known := make(map[string]struct{}, len(names))
	for _, name := range names {
		known[name] = struct{}{}
	}
	for name := range object {
		if _, exists := known[name]; !exists {
			return true
		}
	}
	return false
}

func opaqueResponsePart(raw json.RawMessage) canonical.Part {
	return canonical.Part{
		Kind:     canonical.PartOpaque,
		Provider: "openai.responses",
		Value:    cloneRaw(raw),
	}
}

func hasJSONValue(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	return len(trimmed) > 0 && !bytes.Equal(trimmed, []byte("null"))
}

func hasNonNullJSON(raw json.RawMessage) bool {
	return hasJSONValue(raw) && !strings.EqualFold(string(bytes.TrimSpace(raw)), "null")
}
