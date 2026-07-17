package chatcompletions

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"chat-completion-transformer/internal/canonical"
)

const (
	diagnosticInvalidJSON           canonical.DiagnosticCode = "invalid_json"
	diagnosticInvalidRequest        canonical.DiagnosticCode = "invalid_request"
	diagnosticUnsupportedToolType   canonical.DiagnosticCode = "unsupported_tool_type"
	diagnosticRequestFieldPreserved canonical.DiagnosticCode = "request_field_preserved"
	diagnosticMessageFieldPreserved canonical.DiagnosticCode = "message_field_preserved"
	diagnosticContentFieldPreserved canonical.DiagnosticCode = "content_field_preserved"
)

// DecodeRequest validates an untrusted Chat Completions request and normalizes
// it into the provider-independent representation.
func DecodeRequest(input []byte) canonical.Result[canonical.Request] {
	object, err := canonical.DecodeObject(input)
	if err != nil {
		return canonical.Failure[canonical.Request]([]canonical.Diagnostic{
			diagnostic(canonical.SeverityError, diagnosticInvalidJSON, err.Error(), "", input),
		})
	}

	diagnostics := make([]canonical.Diagnostic, 0)
	request := canonical.Request{}

	request.ModelAlias = requiredString(take(object, "model"), "model", &diagnostics)
	request.Turns = decodeMessages(take(object, "messages"), &diagnostics)
	request.Tools = decodeTools(take(object, "tools"), object, &diagnostics)
	request.ToolChoice = decodeToolChoice(take(object, "tool_choice"), object, &diagnostics)
	request.ParallelToolCalls = optional[bool](take(object, "parallel_tool_calls"), "parallel_tool_calls", &diagnostics)
	request.MaxOutputTokens = decodeMaxTokens(take(object, "max_completion_tokens"), object, &diagnostics)
	request.Temperature = optional[float64](take(object, "temperature"), "temperature", &diagnostics)
	request.TopP = optional[float64](take(object, "top_p"), "top_p", &diagnostics)
	request.StopSequences = decodeStop(take(object, "stop"), &diagnostics)
	request.CandidateCount = optional[int](take(object, "n"), "n", &diagnostics)
	request.OutputFormat = decodeOutputFormat(take(object, "response_format"), object, &diagnostics)
	if stream := optional[bool](take(object, "stream"), "stream", &diagnostics); stream != nil {
		request.Stream = *stream
	}
	streamOptionsRaw := take(object, "stream_options")
	streamIncludeUsage, streamOptionExtensions := decodeStreamOptions(streamOptionsRaw, &diagnostics)
	request.StreamIncludeUsage = streamIncludeUsage
	if len(streamOptionsRaw) > 0 && !request.Stream {
		diagnostics = append(diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "stream_options requires stream to be true", "stream_options", streamOptionsRaw))
	}
	request.Metadata = decodeMetadata(take(object, "metadata"), &diagnostics)

	if request.CandidateCount != nil && *request.CandidateCount < 1 {
		diagnostics = append(diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "n must be at least 1", "n", nil))
	}
	if request.MaxOutputTokens != nil && *request.MaxOutputTokens < 1 {
		diagnostics = append(diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "max output tokens must be at least 1", "max_completion_tokens", nil))
	}

	request.Extensions = object
	for name, raw := range streamOptionExtensions {
		request.Extensions["chat_completions.stream_options."+name] = raw
	}
	for name, raw := range object {
		diagnostics = append(diagnostics, diagnostic(
			canonical.SeverityWarning,
			diagnosticRequestFieldPreserved,
			fmt.Sprintf("Chat Completions field %q is preserved as an extension", name),
			name,
			raw,
		))
	}

	return canonical.Success(request, diagnostics)
}

func decodeStreamOptions(raw json.RawMessage, diagnostics *[]canonical.Diagnostic) (bool, canonical.Object) {
	if len(raw) == 0 {
		return false, nil
	}

	object, err := canonical.DecodeObject(raw)
	if err != nil {
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "stream_options must be an object", "stream_options", raw))
		return false, nil
	}

	includeUsage := optional[bool](take(object, "include_usage"), "stream_options.include_usage", diagnostics)
	return includeUsage != nil && *includeUsage, object
}

func decodeMessages(raw json.RawMessage, diagnostics *[]canonical.Diagnostic) []canonical.Turn {
	if len(raw) == 0 {
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "messages is required", "messages", nil))
		return nil
	}

	var messages []json.RawMessage
	if err := json.Unmarshal(raw, &messages); err != nil || messages == nil {
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "messages must be an array", "messages", raw))
		return nil
	}
	if len(messages) == 0 {
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "messages must contain at least one message", "messages", raw))
		return nil
	}

	turns := make([]canonical.Turn, 0, len(messages))
	for index := 0; index < len(messages); index++ {
		role := peekRole(messages[index], fmt.Sprintf("messages.%d.role", index), diagnostics)
		if role != "tool" {
			turns = append(turns, decodeMessage(messages[index], index, diagnostics))
			continue
		}

		results := make([]canonical.ToolResult, 0)
		for index < len(messages) {
			if peekRole(messages[index], fmt.Sprintf("messages.%d.role", index), diagnostics) != "tool" {
				break
			}
			results = append(results, decodeToolResult(messages[index], index, diagnostics))
			index++
		}
		index--
		turns = append(turns, canonical.Turn{Kind: canonical.TurnToolResults, Results: results})
	}

	return turns
}

func peekRole(raw json.RawMessage, path string, diagnostics *[]canonical.Diagnostic) string {
	object, err := canonical.DecodeObject(raw)
	if err != nil {
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "message must be an object", strings.TrimSuffix(path, ".role"), raw))
		return ""
	}

	return requiredString(object["role"], path, diagnostics)
}

func decodeMessage(raw json.RawMessage, index int, diagnostics *[]canonical.Diagnostic) canonical.Turn {
	path := fmt.Sprintf("messages.%d", index)
	object, err := canonical.DecodeObject(raw)
	if err != nil {
		return canonical.Turn{Kind: canonical.TurnMessage}
	}

	roleRaw := take(object, "role")
	roleName := requiredString(roleRaw, path+".role", diagnostics)
	role, ok := canonicalRole(roleName)
	if !ok || roleName == "tool" {
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, fmt.Sprintf("unsupported message role %q", roleName), path+".role", roleRaw))
	}

	turn := canonical.Turn{
		Kind:    canonical.TurnMessage,
		Role:    role,
		Content: decodeContent(take(object, "content"), path+".content", diagnostics),
		Name:    optional[string](take(object, "name"), path+".name", diagnostics),
	}

	hasAssistantAlternative := false
	if role == canonical.RoleAssistant {
		turn.ToolCalls, turn.Content = decodeToolCalls(take(object, "tool_calls"), path+".tool_calls", turn.Content, diagnostics)
		hasAssistantAlternative = len(turn.ToolCalls) > 0
		if refusal := optional[string](take(object, "refusal"), path+".refusal", diagnostics); refusal != nil {
			turn.Content = append(turn.Content, canonical.Part{Kind: canonical.PartRefusal, Text: *refusal})
			hasAssistantAlternative = true
		}
	}
	if len(turn.Content) == 0 && !hasAssistantAlternative {
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "message content is required unless an assistant message contains a tool call or refusal", path+".content", nil))
	}

	if len(object) > 0 {
		turn.Content = append(turn.Content, canonical.Part{Kind: canonical.PartOpaque, Provider: "chat_completions.message", Value: cloneRaw(raw)})
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityWarning, diagnosticMessageFieldPreserved, "unknown message fields are preserved as opaque content", path, raw))
	}

	return turn
}

func decodeToolResult(raw json.RawMessage, index int, diagnostics *[]canonical.Diagnostic) canonical.ToolResult {
	path := fmt.Sprintf("messages.%d", index)
	object, err := canonical.DecodeObject(raw)
	if err != nil {
		return canonical.ToolResult{}
	}

	delete(object, "role")
	result := canonical.ToolResult{
		CallID:  requiredString(take(object, "tool_call_id"), path+".tool_call_id", diagnostics),
		Content: decodeContent(take(object, "content"), path+".content", diagnostics),
	}
	if len(result.Content) == 0 {
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "tool message content is required", path+".content", nil))
	}
	if len(object) > 0 {
		result.Content = append(result.Content, canonical.Part{Kind: canonical.PartOpaque, Provider: "chat_completions.tool_message", Value: cloneRaw(raw)})
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityWarning, diagnosticMessageFieldPreserved, "unknown tool message fields are preserved as opaque content", path, raw))
	}

	return result
}

func decodeContent(raw json.RawMessage, path string, diagnostics *[]canonical.Diagnostic) []canonical.Part {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}

	if trimmed[0] == '"' {
		var text string
		if err := json.Unmarshal(trimmed, &text); err != nil {
			*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "content must be a valid string", path, raw))
			return nil
		}
		return []canonical.Part{{Kind: canonical.PartText, Text: text}}
	}

	var blocks []json.RawMessage
	if err := json.Unmarshal(trimmed, &blocks); err != nil || blocks == nil {
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "content must be a string, array, or null", path, raw))
		return nil
	}

	parts := make([]canonical.Part, 0, len(blocks))
	for index, block := range blocks {
		parts = append(parts, decodeContentBlock(block, fmt.Sprintf("%s.%d", path, index), diagnostics)...)
	}
	return parts
}

func decodeContentBlock(raw json.RawMessage, path string, diagnostics *[]canonical.Diagnostic) []canonical.Part {
	object, err := canonical.DecodeObject(raw)
	if err != nil {
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "content part must be an object", path, raw))
		return []canonical.Part{opaqueChatPart(raw)}
	}

	typeName := requiredString(take(object, "type"), path+".type", diagnostics)
	var part canonical.Part
	nestedFields := false
	switch typeName {
	case "text":
		part = canonical.Part{Kind: canonical.PartText, Text: requiredString(take(object, "text"), path+".text", diagnostics)}
	case "refusal":
		part = canonical.Part{Kind: canonical.PartRefusal, Text: requiredString(take(object, "refusal"), path+".refusal", diagnostics)}
	case "image_url":
		part, nestedFields = decodeImagePart(take(object, "image_url"), path+".image_url", raw, diagnostics)
	case "input_audio":
		part, nestedFields = decodeAudioPart(take(object, "input_audio"), path+".input_audio", raw, diagnostics)
	case "file":
		part, nestedFields = decodeFilePart(take(object, "file"), path+".file", raw, diagnostics)
	default:
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityWarning, canonical.DiagnosticUnsupportedContentPart, fmt.Sprintf("content part type %q is preserved as opaque", typeName), path, raw))
		return []canonical.Part{opaqueChatPart(raw)}
	}

	if part.Kind == canonical.PartOpaque || (!nestedFields && len(object) == 0) {
		return []canonical.Part{part}
	}

	*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityWarning, diagnosticContentFieldPreserved, "unmapped content part fields are preserved in an opaque part", path, raw))
	return []canonical.Part{part, opaqueChatPart(raw)}
}

func decodeImagePart(raw json.RawMessage, path string, original json.RawMessage, diagnostics *[]canonical.Diagnostic) (canonical.Part, bool) {
	object, err := canonical.DecodeObject(raw)
	if err != nil {
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "image_url must be an object", path, raw))
		return opaqueChatPart(original), false
	}

	value := requiredString(take(object, "url"), path+".url", diagnostics)
	source := canonical.AssetSource{Kind: canonical.AssetSourceURL, URL: value}
	if mediaType, data, ok := parseBase64DataURL(value); ok {
		source = canonical.AssetSource{Kind: canonical.AssetSourceBase64, MediaType: mediaType, Data: data}
	}

	part := canonical.Part{Kind: canonical.PartImage, Source: &source}
	if detail := optional[string](take(object, "detail"), path+".detail", diagnostics); detail != nil {
		parsed := canonical.ImageDetail(*detail)
		if parsed != canonical.ImageDetailAuto && parsed != canonical.ImageDetailLow && parsed != canonical.ImageDetailHigh {
			*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, fmt.Sprintf("unsupported image detail %q", *detail), path+".detail", raw))
		} else {
			part.Detail = &parsed
		}
	}
	return part, len(object) > 0
}

func decodeAudioPart(raw json.RawMessage, path string, original json.RawMessage, diagnostics *[]canonical.Diagnostic) (canonical.Part, bool) {
	object, err := canonical.DecodeObject(raw)
	if err != nil {
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "input_audio must be an object", path, raw))
		return opaqueChatPart(original), false
	}

	data := requiredString(take(object, "data"), path+".data", diagnostics)
	format := requiredString(take(object, "format"), path+".format", diagnostics)
	mediaType := "audio/" + format
	if format == "mp3" {
		mediaType = "audio/mpeg"
	}
	source := canonical.AssetSource{Kind: canonical.AssetSourceBase64, MediaType: mediaType, Data: data}
	return canonical.Part{Kind: canonical.PartAudio, Source: &source}, len(object) > 0
}

func decodeFilePart(raw json.RawMessage, path string, original json.RawMessage, diagnostics *[]canonical.Diagnostic) (canonical.Part, bool) {
	object, err := canonical.DecodeObject(raw)
	if err != nil {
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "file must be an object", path, raw))
		return opaqueChatPart(original), false
	}

	part := canonical.Part{Kind: canonical.PartFile}
	part.Filename = optional[string](take(object, "filename"), path+".filename", diagnostics)
	fileID := optional[string](take(object, "file_id"), path+".file_id", diagnostics)
	fileDataRaw := take(object, "file_data")
	if fileID != nil {
		part.Source = &canonical.AssetSource{Kind: canonical.AssetSourceFileID, FileID: *fileID}
		return part, len(bytes.TrimSpace(fileDataRaw)) > 0 && !bytes.Equal(bytes.TrimSpace(fileDataRaw), []byte("null")) || len(object) > 0
	}

	fileData := requiredString(fileDataRaw, path+".file_data", diagnostics)
	mediaType, data, ok := parseBase64DataURL(fileData)
	if !ok {
		mediaType = "application/octet-stream"
		data = fileData
	}
	part.Source = &canonical.AssetSource{Kind: canonical.AssetSourceBase64, MediaType: mediaType, Data: data}
	return part, len(object) > 0
}

func decodeToolCalls(raw json.RawMessage, path string, content []canonical.Part, diagnostics *[]canonical.Diagnostic) ([]canonical.ToolCall, []canonical.Part) {
	if len(bytes.TrimSpace(raw)) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, content
	}

	var values []json.RawMessage
	if err := json.Unmarshal(raw, &values); err != nil || values == nil {
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "tool_calls must be an array", path, raw))
		return nil, content
	}

	calls := make([]canonical.ToolCall, 0, len(values))
	for index, value := range values {
		callPath := fmt.Sprintf("%s.%d", path, index)
		object, err := canonical.DecodeObject(value)
		if err != nil {
			*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "tool call must be an object", callPath, value))
			continue
		}
		typeName := requiredString(take(object, "type"), callPath+".type", diagnostics)
		if typeName != "function" {
			content = append(content, canonical.Part{Kind: canonical.PartOpaque, Provider: "chat_completions.tool_call", Value: cloneRaw(value)})
			*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityWarning, diagnosticUnsupportedToolType, fmt.Sprintf("tool call type %q is preserved as opaque", typeName), callPath, value))
			continue
		}

		id := requiredString(take(object, "id"), callPath+".id", diagnostics)
		functionRaw := take(object, "function")
		function, err := canonical.DecodeObject(functionRaw)
		if err != nil {
			*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "function tool call is malformed", callPath+".function", functionRaw))
			continue
		}
		name := requiredString(take(function, "name"), callPath+".function.name", diagnostics)
		arguments := requiredString(take(function, "arguments"), callPath+".function.arguments", diagnostics)
		calls = append(calls, canonical.ToolCall{
			ID:              id,
			Name:            name,
			ArgumentsRaw:    arguments,
			ArgumentsParsed: parseJSONObjectString(arguments),
		})
		if len(object) == 0 && len(function) == 0 {
			continue
		}
		content = append(content, canonical.Part{Kind: canonical.PartOpaque, Provider: "chat_completions.tool_call", Value: cloneRaw(value)})
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityWarning, diagnosticMessageFieldPreserved, "unmapped function tool call fields are preserved as opaque content", callPath, value))
	}

	return calls, content
}

func decodeTools(raw json.RawMessage, extensions canonical.Object, diagnostics *[]canonical.Diagnostic) []canonical.ToolDefinition {
	if len(bytes.TrimSpace(raw)) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil
	}

	var values []json.RawMessage
	if err := json.Unmarshal(raw, &values); err != nil || values == nil {
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "tools must be an array", "tools", raw))
		return nil
	}

	tools := make([]canonical.ToolDefinition, 0, len(values))
	unsupported := make([]json.RawMessage, 0)
	preserveRaw := false
	for index, value := range values {
		path := fmt.Sprintf("tools.%d", index)
		object, err := canonical.DecodeObject(value)
		if err != nil {
			*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "tool must be an object", path, value))
			continue
		}
		typeName := requiredString(take(object, "type"), path+".type", diagnostics)
		if typeName != "function" {
			preserveRaw = true
			unsupported = append(unsupported, cloneRaw(value))
			*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityWarning, diagnosticUnsupportedToolType, fmt.Sprintf("tool type %q cannot be represented as a canonical function tool", typeName), path, value))
			continue
		}

		functionRaw := take(object, "function")
		function, err := canonical.DecodeObject(functionRaw)
		if err != nil {
			*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "function tool is malformed", path+".function", functionRaw))
			continue
		}
		schema := defaultInputSchema()
		if parameters := take(function, "parameters"); len(bytes.TrimSpace(parameters)) > 0 && !bytes.Equal(bytes.TrimSpace(parameters), []byte("null")) {
			schema, err = canonical.DecodeObject(parameters)
			if err != nil {
				*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "function parameters must be an object", path+".function.parameters", parameters))
				continue
			}
		}
		tools = append(tools, canonical.ToolDefinition{
			Name:        requiredString(take(function, "name"), path+".function.name", diagnostics),
			Description: optional[string](take(function, "description"), path+".function.description", diagnostics),
			InputSchema: schema,
			Strict:      optional[bool](take(function, "strict"), path+".function.strict", diagnostics),
		})
		if len(object) > 0 || len(function) > 0 {
			preserveRaw = true
		}
	}
	if preserveRaw {
		extensions["tools"] = cloneRaw(raw)
	}
	if len(unsupported) > 0 {
		encoded, _ := json.Marshal(unsupported)
		if _, exists := extensions["chat_completions.unsupported_tools"]; !exists {
			extensions["chat_completions.unsupported_tools"] = encoded
		}
	}
	return tools
}

func decodeToolChoice(raw json.RawMessage, extensions canonical.Object, diagnostics *[]canonical.Diagnostic) *canonical.ToolChoice {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	if trimmed[0] == '"' {
		var mode string
		if err := json.Unmarshal(trimmed, &mode); err != nil {
			*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "tool_choice is malformed", "tool_choice", raw))
			return nil
		}
		choice := canonical.ToolChoice{Mode: canonical.ToolChoiceMode(mode)}
		if choice.Mode != canonical.ToolChoiceAuto && choice.Mode != canonical.ToolChoiceNone && choice.Mode != canonical.ToolChoiceRequired {
			*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, fmt.Sprintf("unsupported tool choice %q", mode), "tool_choice", raw))
		}
		return &choice
	}

	object, err := canonical.DecodeObject(trimmed)
	if err != nil {
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "tool_choice must be a string or object", "tool_choice", raw))
		return nil
	}
	typeRaw := take(object, "type")
	if requiredString(typeRaw, "tool_choice.type", diagnostics) != "function" {
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "only named function tool_choice is supported", "tool_choice.type", typeRaw))
		return nil
	}
	functionRaw := take(object, "function")
	function, err := canonical.DecodeObject(functionRaw)
	if err != nil {
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "tool_choice.function must be an object", "tool_choice.function", functionRaw))
		return nil
	}
	name := requiredString(take(function, "name"), "tool_choice.function.name", diagnostics)
	if len(object) > 0 || len(function) > 0 {
		extensions["tool_choice"] = cloneRaw(raw)
	}
	return &canonical.ToolChoice{Mode: canonical.ToolChoiceNamed, Name: &name}
}

func decodeOutputFormat(raw json.RawMessage, extensions canonical.Object, diagnostics *[]canonical.Diagnostic) *canonical.OutputFormat {
	if len(bytes.TrimSpace(raw)) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil
	}
	object, err := canonical.DecodeObject(raw)
	if err != nil {
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "response_format must be an object", "response_format", raw))
		return nil
	}
	typeRaw := take(object, "type")
	typeName := requiredString(typeRaw, "response_format.type", diagnostics)
	format := &canonical.OutputFormat{Type: canonical.OutputFormatType(typeName)}
	switch format.Type {
	case canonical.OutputFormatText, canonical.OutputFormatJSONObject:
		if len(object) > 0 {
			extensions["response_format"] = cloneRaw(raw)
		}
		return format
	case canonical.OutputFormatJSONSchema:
		schemaConfigRaw := take(object, "json_schema")
		schemaConfig, err := canonical.DecodeObject(schemaConfigRaw)
		if err != nil {
			*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "response_format.json_schema must be an object", "response_format.json_schema", schemaConfigRaw))
			return format
		}
		name := requiredString(take(schemaConfig, "name"), "response_format.json_schema.name", diagnostics)
		format.Name = &name
		format.Description = optional[string](take(schemaConfig, "description"), "response_format.json_schema.description", diagnostics)
		format.Strict = optional[bool](take(schemaConfig, "strict"), "response_format.json_schema.strict", diagnostics)
		schemaRaw := take(schemaConfig, "schema")
		format.Schema, err = canonical.DecodeObject(schemaRaw)
		if err != nil {
			*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "response_format.json_schema.schema must be an object", "response_format.json_schema.schema", schemaRaw))
		}
		if len(object) > 0 || len(schemaConfig) > 0 {
			extensions["response_format"] = cloneRaw(raw)
		}
		return format
	default:
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, fmt.Sprintf("unsupported response format %q", typeName), "response_format.type", typeRaw))
		return format
	}
}

func decodeStop(raw json.RawMessage, diagnostics *[]canonical.Diagnostic) []string {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	if trimmed[0] == '"' {
		var value string
		if err := json.Unmarshal(trimmed, &value); err == nil {
			return []string{value}
		}
	}
	var values []string
	if err := json.Unmarshal(trimmed, &values); err != nil || values == nil {
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "stop must be a string or array of strings", "stop", raw))
		return nil
	}
	return values
}

func decodeMetadata(raw json.RawMessage, diagnostics *[]canonical.Diagnostic) map[string]string {
	if len(bytes.TrimSpace(raw)) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil
	}
	var metadata map[string]string
	if err := json.Unmarshal(raw, &metadata); err != nil || metadata == nil {
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, "metadata must be an object of strings", "metadata", raw))
		return nil
	}
	return metadata
}

func decodeMaxTokens(primary json.RawMessage, extensions canonical.Object, diagnostics *[]canonical.Diagnostic) *int {
	if len(bytes.TrimSpace(primary)) > 0 && !bytes.Equal(bytes.TrimSpace(primary), []byte("null")) {
		return optional[int](primary, "max_completion_tokens", diagnostics)
	}
	return optional[int](take(extensions, "max_tokens"), "max_tokens", diagnostics)
}

func optional[T any](raw json.RawMessage, path string, diagnostics *[]canonical.Diagnostic) *T {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	var value T
	if err := json.Unmarshal(trimmed, &value); err != nil {
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, fmt.Sprintf("%s has an invalid type", path), path, raw))
		return nil
	}
	return &value
}

func requiredString(raw json.RawMessage, path string, diagnostics *[]canonical.Diagnostic) string {
	value := optional[string](raw, path, diagnostics)
	if value == nil || *value == "" {
		*diagnostics = append(*diagnostics, diagnostic(canonical.SeverityError, diagnosticInvalidRequest, fmt.Sprintf("%s is required", path), path, raw))
		return ""
	}
	return *value
}

func canonicalRole(role string) (canonical.Role, bool) {
	switch role {
	case "system":
		return canonical.RoleSystem, true
	case "developer":
		return canonical.RoleDeveloper, true
	case "user":
		return canonical.RoleUser, true
	case "assistant":
		return canonical.RoleAssistant, true
	default:
		return canonical.Role(role), false
	}
}

func parseJSONObjectString(raw string) canonical.Object {
	object, err := canonical.DecodeObject([]byte(raw))
	if err != nil {
		return nil
	}
	return object
}

func parseBase64DataURL(value string) (string, string, bool) {
	header, data, ok := strings.Cut(value, ",")
	if !ok || !strings.HasPrefix(header, "data:") || !strings.HasSuffix(strings.ToLower(header), ";base64") {
		return "", "", false
	}
	mediaType := header[len("data:") : len(header)-len(";base64")]
	if mediaType == "" || data == "" {
		return "", "", false
	}
	return mediaType, data, true
}

func defaultInputSchema() canonical.Object {
	return canonical.Object{
		"type":       json.RawMessage(`"object"`),
		"properties": json.RawMessage(`{}`),
	}
}

func take(object canonical.Object, key string) json.RawMessage {
	if object == nil {
		return nil
	}
	value := object[key]
	delete(object, key)
	return value
}

func opaqueChatPart(raw json.RawMessage) canonical.Part {
	return canonical.Part{Kind: canonical.PartOpaque, Provider: "chat_completions", Value: cloneRaw(raw)}
}

func cloneRaw(raw json.RawMessage) json.RawMessage {
	return append(json.RawMessage(nil), raw...)
}

func diagnostic(severity canonical.Severity, code canonical.DiagnosticCode, message, path string, source json.RawMessage) canonical.Diagnostic {
	diagnostic := canonical.Diagnostic{Severity: severity, Code: code, Message: message, SourceValue: cloneRaw(source)}
	if path != "" {
		diagnostic.Path = &path
	}
	return diagnostic
}
