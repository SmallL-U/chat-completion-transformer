package anthropicmessages

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"chat-completion-transformer/internal/assets"
	"chat-completion-transformer/internal/canonical"
	"chat-completion-transformer/internal/capabilities"
)

const (
	diagnosticInvalidCanonicalRequest canonical.DiagnosticCode = "invalid_canonical_request"
	diagnosticInvalidProfile          canonical.DiagnosticCode = "invalid_capability_profile"
	diagnosticMissingMaxTokens        canonical.DiagnosticCode = "missing_max_output_tokens"
	diagnosticAssetResolutionFailed   canonical.DiagnosticCode = "asset_resolution_failed"
	diagnosticImageSourceUnsupported  canonical.DiagnosticCode = "image_source_unsupported"
	diagnosticFileSourceUnsupported   canonical.DiagnosticCode = "file_source_unsupported"
	diagnosticContentFieldLossy       canonical.DiagnosticCode = "content_field_lossy"
	diagnosticToolStrictUnsupported   canonical.DiagnosticCode = "tool_strict_unsupported"
	diagnosticParallelUnsupported     canonical.DiagnosticCode = "parallel_tool_calls_unsupported"
	diagnosticStructuredUnsupported   canonical.DiagnosticCode = "structured_output_unsupported"
	diagnosticSchemaUnsupported       canonical.DiagnosticCode = "schema_unsupported"
	diagnosticMetadataUnsupported     canonical.DiagnosticCode = "metadata_unsupported"
	diagnosticRequestExtension        canonical.DiagnosticCode = "request_extension_unsupported"
	diagnosticContextCanceled         canonical.DiagnosticCode = "context_canceled"
)

// RequestEncodeOptions describes one concrete Anthropic Messages target.
// Resolver is optional; the built-in resolver validates native sources without
// fetching remote URLs.
type RequestEncodeOptions struct {
	TargetModel            string
	Mode                   canonical.Mode
	Profile                capabilities.Profile
	DefaultMaxOutputTokens int
	Resolver               assets.Resolver
}

type anthropicCacheControl struct {
	Type string  `json:"type"`
	TTL  *string `json:"ttl,omitempty"`
	path string
}

type anthropicCacheBlock struct {
	control *anthropicCacheControl
}

type requestEncoder struct {
	ctx         context.Context
	options     RequestEncodeOptions
	resolver    assets.Resolver
	diagnostics []canonical.Diagnostic
	system      []any
	messages    []any
	roleWarning bool
}

// EncodeRequest converts a canonical request into an Anthropic Messages JSON
// request. It never starts background work; cancellation is propagated to the
// asset resolver used by the caller.
func EncodeRequest(ctx context.Context, request canonical.Request, options RequestEncodeOptions) canonical.Result[json.RawMessage] {
	if ctx == nil {
		return canonical.Failure[json.RawMessage]([]canonical.Diagnostic{
			makeDiagnostic(canonical.SeverityError, diagnosticInvalidCanonicalRequest, "context is required", "", nil),
		})
	}
	if err := ctx.Err(); err != nil {
		return canceledResult(err)
	}

	resolver := options.Resolver
	if resolver == nil {
		resolver = assets.NativeResolver{}
	}
	encoder := requestEncoder{ctx: ctx, options: options, resolver: resolver}
	encoder.validateTarget()
	encoder.diagnostics = append(encoder.diagnostics, canonical.ValidateToolHistory(request.Turns)...)
	encoder.encodeTurns(request.Turns)

	value := encoder.encodeFields(request)
	if err := ctx.Err(); err != nil {
		encoder.diagnostics = append(encoder.diagnostics, makeDiagnostic(
			canonical.SeverityError,
			diagnosticContextCanceled,
			fmt.Sprintf("request encoding canceled: %v", err),
			"",
			nil,
		))
	}
	if canonical.HasErrors(encoder.diagnostics) {
		return canonical.Failure[json.RawMessage](encoder.diagnostics)
	}

	encoded, err := json.Marshal(value)
	if err != nil {
		encoder.diagnostics = append(encoder.diagnostics, makeDiagnostic(
			canonical.SeverityError,
			diagnosticInvalidCanonicalRequest,
			fmt.Sprintf("encode Anthropic Messages request: %v", err),
			"",
			nil,
		))
		return canonical.Failure[json.RawMessage](encoder.diagnostics)
	}
	raw := json.RawMessage(encoded)
	return canonical.Success(raw, encoder.diagnostics)
}

func canceledResult(err error) canonical.Result[json.RawMessage] {
	return canonical.Failure[json.RawMessage]([]canonical.Diagnostic{
		makeDiagnostic(
			canonical.SeverityError,
			diagnosticContextCanceled,
			fmt.Sprintf("request encoding canceled: %v", err),
			"",
			nil,
		),
	})
}

func (e *requestEncoder) validateTarget() {
	if e.options.TargetModel == "" {
		e.addError(diagnosticInvalidCanonicalRequest, "target model is required", "model", nil)
	}
	profile := e.options.Profile
	if profile.Provider != capabilities.ProviderAnthropic {
		e.addError(diagnosticInvalidProfile, "capability profile must target Anthropic", "profile.provider", nil)
	}
	if !isMessagesEndpoint(profile.Endpoint) {
		e.addError(diagnosticInvalidProfile, "capability profile must target a Messages endpoint", "profile.endpoint", nil)
	}
	if profile.Model == "" {
		e.addError(diagnosticInvalidProfile, "capability profile model is required", "profile.model", nil)
		return
	}
	if e.options.TargetModel == "" || profile.Model == e.options.TargetModel {
		return
	}
	e.addError(diagnosticInvalidProfile, "capability profile model does not match target model", "profile.model", quotedRaw(profile.Model))
}

func isMessagesEndpoint(endpoint capabilities.Endpoint) bool {
	return endpoint == capabilities.EndpointMessages ||
		endpoint == capabilities.EndpointBedrockMessages ||
		endpoint == capabilities.EndpointVertexMessages
}

func (e *requestEncoder) encodeTurns(turns []canonical.Turn) {
	leading := true
	for index := 0; index < len(turns); index++ {
		turn := turns[index]
		e.validateIgnoredTurnCacheExtensions(turn, index)
		if isInstructionTurn(turn) {
			e.encodeInstruction(turn, index, leading)
			continue
		}

		leading = false
		if turn.Kind == canonical.TurnToolResults {
			index = e.encodeToolResults(turns, index)
			continue
		}
		if turn.Kind != canonical.TurnMessage {
			e.addError(diagnosticInvalidCanonicalRequest, fmt.Sprintf("unsupported turn kind %q", turn.Kind), fmt.Sprintf("turns.%d.kind", index), quotedRaw(string(turn.Kind)))
			continue
		}
		e.encodeOrdinaryMessage(turn, index)
	}
	if len(e.messages) > 0 {
		return
	}
	e.addError(diagnosticInvalidCanonicalRequest, "Anthropic Messages requires at least one user or assistant message", "turns", nil)
}

func (e *requestEncoder) validateIgnoredTurnCacheExtensions(turn canonical.Turn, turnIndex int) {
	if turn.Kind != canonical.TurnMessage {
		e.validateIgnoredPartCacheExtensions(turn.Content, fmt.Sprintf("turns.%d.content", turnIndex))
	}
	if turn.Kind == canonical.TurnToolResults {
		return
	}
	for resultIndex, result := range turn.Results {
		e.validateIgnoredPartCacheExtensions(
			result.Content,
			fmt.Sprintf("turns.%d.results.%d.content", turnIndex, resultIndex),
		)
	}
}

func (e *requestEncoder) validateIgnoredPartCacheExtensions(parts []canonical.Part, basePath string) {
	for partIndex, part := range parts {
		for _, name := range sortedObjectKeys(part.Extensions) {
			raw := part.Extensions[name]
			path := fmt.Sprintf("%s.%d.extensions.%s", basePath, partIndex, name)
			switch name {
			case "cache_control":
				if e.decodeCacheControl(raw, path) == nil {
					continue
				}
				e.addError(
					canonical.DiagnosticCacheBreakpointUnsupported,
					"Anthropic cache_control cannot be attached to content ignored for this canonical turn kind",
					path,
					raw,
				)
			case "prompt_cache_breakpoint":
				if err := validateOpenAIPromptCacheBreakpoint(raw); err != nil {
					e.addError(canonical.DiagnosticInvalidCacheControl, err.Error(), path, raw)
					continue
				}
				e.addError(
					canonical.DiagnosticCacheBreakpointUnsupported,
					"OpenAI prompt cache breakpoint cannot be attached to content ignored for this canonical turn kind",
					path,
					raw,
				)
			case "prompt_cache_key", "prompt_cache_options", "prompt_cache_retention":
				if err := validateOpenAICacheDirective(name, raw); err != nil {
					e.addError(canonical.DiagnosticInvalidCacheControl, err.Error(), path, raw)
					continue
				}
				e.addError(
					canonical.DiagnosticInvalidCacheControl,
					fmt.Sprintf("cache directive %q is only valid at the request top level", name),
					path,
					raw,
				)
			}
		}
	}
}

func isInstructionTurn(turn canonical.Turn) bool {
	if turn.Kind != canonical.TurnMessage {
		return false
	}
	return turn.Role == canonical.RoleSystem || turn.Role == canonical.RoleDeveloper
}

func (e *requestEncoder) encodeInstruction(turn canonical.Turn, index int, leading bool) {
	path := fmt.Sprintf("turns.%d", index)
	if turn.Role == canonical.RoleDeveloper {
		e.addRolePriorityWarning(path + ".role")
	}
	if turn.Name != nil {
		e.addLossy(diagnosticContentFieldLossy, "message name cannot be represented by Anthropic Messages", path+".name", quotedRaw(*turn.Name))
	}
	if len(turn.ToolCalls) > 0 {
		e.addError(diagnosticInvalidCanonicalRequest, "system and developer turns cannot contain tool calls", path+".tool_calls", nil)
	}

	blocks := e.encodeParts(turn.Content, canonical.RoleSystem, path+".content")
	if leading {
		e.system = append(e.system, blocks...)
		return
	}
	if e.options.Profile.MidConversationSystem {
		e.messages = append(e.messages, map[string]any{"role": "system", "content": blocks})
		return
	}

	e.addLossy(
		canonical.DiagnosticMidConversationSystemUnsupported,
		"target profile does not support mid-conversation system messages",
		path,
		nil,
	)
	if e.options.Mode == canonical.ModeStrict {
		return
	}
	marker := map[string]any{"type": "text", "text": "[Originally inserted mid-conversation]"}
	e.system = append(e.system, marker)
	e.system = append(e.system, blocks...)
}

func (e *requestEncoder) addRolePriorityWarning(path string) {
	if e.roleWarning {
		return
	}
	e.roleWarning = true
	e.diagnostics = append(e.diagnostics, makeDiagnostic(
		canonical.SeverityWarning,
		canonical.DiagnosticRolePriorityCollapsed,
		"developer instructions are merged into Anthropic's single system instruction layer",
		path,
		nil,
	))
}

func (e *requestEncoder) encodeOrdinaryMessage(turn canonical.Turn, index int) {
	path := fmt.Sprintf("turns.%d", index)
	if turn.Role != canonical.RoleUser && turn.Role != canonical.RoleAssistant {
		e.addError(diagnosticInvalidCanonicalRequest, fmt.Sprintf("unsupported message role %q", turn.Role), path+".role", quotedRaw(string(turn.Role)))
		return
	}
	if turn.Name != nil {
		e.addLossy(diagnosticContentFieldLossy, "message name cannot be represented by Anthropic Messages", path+".name", quotedRaw(*turn.Name))
	}
	if turn.Role != canonical.RoleAssistant && len(turn.ToolCalls) > 0 {
		e.addError(diagnosticInvalidCanonicalRequest, "only assistant turns may contain tool calls", path+".tool_calls", nil)
	}

	content := e.encodeParts(turn.Content, turn.Role, path+".content")
	if turn.Role == canonical.RoleAssistant {
		content = append(content, e.encodeToolCalls(turn.ToolCalls, path+".tool_calls")...)
	}
	if len(content) == 0 {
		e.addError(diagnosticInvalidCanonicalRequest, "Anthropic message content cannot be empty", path+".content", nil)
	}
	e.messages = append(e.messages, map[string]any{"role": string(turn.Role), "content": content})
}

func (e *requestEncoder) encodeToolCalls(calls []canonical.ToolCall, path string) []any {
	blocks := make([]any, 0, len(calls))
	for index, call := range calls {
		callPath := fmt.Sprintf("%s.%d", path, index)
		if call.ID == "" || call.Name == "" {
			e.addError(diagnosticInvalidCanonicalRequest, "tool call ID and name are required", callPath, nil)
			continue
		}
		input, err := canonical.DecodeObject([]byte(call.ArgumentsRaw))
		if err != nil {
			e.addError(
				canonical.DiagnosticInvalidToolArgumentsJSON,
				fmt.Sprintf("tool call arguments must be a complete JSON object: %v", err),
				callPath+".arguments_raw",
				quotedRaw(call.ArgumentsRaw),
			)
			continue
		}
		blocks = append(blocks, map[string]any{
			"type":  "tool_use",
			"id":    call.ID,
			"name":  call.Name,
			"input": input,
		})
	}
	return blocks
}

func (e *requestEncoder) encodeToolResults(turns []canonical.Turn, index int) int {
	turn := turns[index]
	path := fmt.Sprintf("turns.%d", index)
	content := make([]any, 0, len(turn.Results))
	for resultIndex, result := range turn.Results {
		resultPath := fmt.Sprintf("%s.results.%d", path, resultIndex)
		if result.CallID == "" {
			e.addError(diagnosticInvalidCanonicalRequest, "tool result call ID is required", resultPath+".call_id", nil)
			continue
		}
		block := map[string]any{
			"type":        "tool_result",
			"tool_use_id": result.CallID,
		}
		blocks := e.encodeParts(result.Content, canonical.Role("tool_result"), resultPath+".content")
		if len(blocks) > 0 {
			block["content"] = blocks
		}
		if result.IsError != nil && *result.IsError {
			block["is_error"] = true
		}
		content = append(content, block)
	}

	// A following user turn belongs to the same Anthropic message. Keeping the
	// tool_result blocks first is required for reliable parallel tool use.
	if index+1 < len(turns) && isPlainUserTurn(turns[index+1]) {
		user := turns[index+1]
		e.validateIgnoredTurnCacheExtensions(user, index+1)
		userPath := fmt.Sprintf("turns.%d", index+1)
		if user.Name != nil {
			e.addLossy(diagnosticContentFieldLossy, "message name cannot be represented by Anthropic Messages", userPath+".name", quotedRaw(*user.Name))
		}
		content = append(content, e.encodeParts(user.Content, canonical.RoleUser, userPath+".content")...)
		index++
	}
	if len(content) == 0 {
		e.addError(diagnosticInvalidCanonicalRequest, "tool results message cannot be empty", path+".results", nil)
	}
	e.messages = append(e.messages, map[string]any{"role": "user", "content": content})
	return index
}

func isPlainUserTurn(turn canonical.Turn) bool {
	return turn.Kind == canonical.TurnMessage && turn.Role == canonical.RoleUser && len(turn.ToolCalls) == 0
}

func (e *requestEncoder) encodeParts(parts []canonical.Part, role canonical.Role, path string) []any {
	blocks := make([]any, 0, len(parts))
	for index, part := range parts {
		partPath := fmt.Sprintf("%s.%d", path, index)
		block := e.encodePart(part, role, partPath)
		e.encodePartExtensions(part, role, partPath, block)
		if block == nil {
			continue
		}
		blocks = append(blocks, block)
	}
	return blocks
}

func (e *requestEncoder) encodePart(part canonical.Part, role canonical.Role, path string) any {
	switch part.Kind {
	case canonical.PartText:
		if !e.options.Profile.Content.Text {
			e.addLossy(canonical.DiagnosticUnsupportedContentPart, "target profile does not support text content", path, nil)
			return nil
		}
		return map[string]any{"type": "text", "text": part.Text}
	case canonical.PartImage:
		return e.encodeImage(part, role, path)
	case canonical.PartFile:
		return e.encodeDocument(part, role, path)
	case canonical.PartRefusal:
		e.addLossy(canonical.DiagnosticUnsupportedContentPart, "Anthropic request history has no refusal content block", path, quotedRaw(part.Text))
		if e.options.Mode == canonical.ModeStrict {
			return nil
		}
		return map[string]any{"type": "text", "text": part.Text}
	case canonical.PartAudio:
		e.addLossy(canonical.DiagnosticUnsupportedContentPart, "Anthropic Messages does not support canonical audio content", path, nil)
		return nil
	case canonical.PartOpaque:
		e.addLossy(canonical.DiagnosticUnsupportedContentPart, "opaque provider content cannot be encoded as an Anthropic request block", path, part.Value)
		return nil
	default:
		e.addError(diagnosticInvalidCanonicalRequest, fmt.Sprintf("unknown canonical content kind %q", part.Kind), path+".kind", quotedRaw(string(part.Kind)))
		return nil
	}
}

func (e *requestEncoder) encodeImage(part canonical.Part, role canonical.Role, path string) any {
	if role != canonical.RoleUser && role != canonical.Role("tool_result") {
		e.addLossy(canonical.DiagnosticUnsupportedContentPart, "Anthropic image blocks are only supported in user input and tool results", path, nil)
		return nil
	}
	if !e.options.Profile.Content.Image {
		e.addLossy(canonical.DiagnosticUnsupportedContentPart, "target profile does not support image content", path, nil)
		return nil
	}
	if part.Detail != nil {
		e.addLossy(diagnosticContentFieldLossy, "Anthropic Messages has no image detail setting", path+".detail", quotedRaw(string(*part.Detail)))
	}
	source := e.resolveSource(part.Source, path+".source")
	if source == nil {
		return nil
	}
	encoded := e.encodeSource(*source, true, path+".source")
	if encoded == nil {
		return nil
	}
	return map[string]any{"type": "image", "source": encoded}
}

func (e *requestEncoder) encodeDocument(part canonical.Part, role canonical.Role, path string) any {
	if role != canonical.RoleUser && role != canonical.Role("tool_result") {
		e.addLossy(canonical.DiagnosticUnsupportedContentPart, "Anthropic document blocks are only supported in user input and tool results", path, nil)
		return nil
	}
	if !e.options.Profile.Content.File {
		e.addLossy(canonical.DiagnosticUnsupportedContentPart, "target profile does not support file content", path, nil)
		return nil
	}
	if part.Filename != nil {
		e.addLossy(diagnosticContentFieldLossy, "canonical filename is not sent as part of an Anthropic document source", path+".filename", quotedRaw(*part.Filename))
	}
	source := e.resolveSource(part.Source, path+".source")
	if source == nil {
		return nil
	}
	encoded := e.encodeSource(*source, false, path+".source")
	if encoded == nil {
		return nil
	}
	return map[string]any{"type": "document", "source": encoded}
}

func (e *requestEncoder) resolveSource(source *canonical.AssetSource, path string) *assets.ResolvedAsset {
	if source == nil {
		e.addError(diagnosticInvalidCanonicalRequest, "content asset source is required", path, nil)
		return nil
	}
	resolved, err := e.resolver.ResolveForAnthropic(e.ctx, *source)
	if err != nil {
		e.addLossy(diagnosticAssetResolutionFailed, fmt.Sprintf("resolve Anthropic asset: %v", err), path, mustRaw(source))
		return nil
	}
	return &resolved
}

func (e *requestEncoder) encodeSource(source assets.ResolvedAsset, image bool, path string) any {
	switch source.Kind {
	case canonical.AssetSourceURL:
		if !e.sourceEnabled(image, canonical.AssetSourceURL) {
			e.addSourceUnsupported(image, "URL", path, quotedRaw(source.URL))
			return nil
		}
		return map[string]any{"type": "url", "url": source.URL}
	case canonical.AssetSourceBase64:
		if !e.sourceEnabled(image, canonical.AssetSourceBase64) {
			e.addSourceUnsupported(image, "base64", path, quotedRaw(source.MediaType))
			return nil
		}
		return map[string]any{"type": "base64", "media_type": source.MediaType, "data": source.Data}
	case canonical.AssetSourceFileID:
		if !e.sourceEnabled(image, canonical.AssetSourceFileID) {
			e.addSourceUnsupported(image, "file", path, quotedRaw(source.FileID))
			return nil
		}
		return map[string]any{"type": "file", "file_id": source.FileID}
	default:
		e.addError(diagnosticInvalidCanonicalRequest, fmt.Sprintf("unsupported resolved asset kind %q", source.Kind), path+".kind", quotedRaw(string(source.Kind)))
		return nil
	}
}

func (e *requestEncoder) sourceEnabled(image bool, kind canonical.AssetSourceKind) bool {
	capability := e.options.Profile.Files
	if image {
		capability = e.options.Profile.Images
	}
	switch kind {
	case canonical.AssetSourceURL:
		return capability.URL
	case canonical.AssetSourceBase64:
		return capability.Base64
	case canonical.AssetSourceFileID:
		return e.options.Profile.Endpoint == capabilities.EndpointMessages && capability.FileID
	default:
		return false
	}
}

func (e *requestEncoder) addSourceUnsupported(image bool, sourceType, path string, source json.RawMessage) {
	contentType := "file"
	code := diagnosticFileSourceUnsupported
	if image {
		contentType = "image"
		code = diagnosticImageSourceUnsupported
	}
	message := fmt.Sprintf("target profile does not support %s %s sources", sourceType, contentType)
	if sourceType == "file" {
		message = fmt.Sprintf("%s file sources require an Anthropic direct Messages Files API profile", contentType)
	}
	e.addLossy(code, message, path, source)
}

func (e *requestEncoder) encodeFields(request canonical.Request) map[string]any {
	value := map[string]any{
		"model":    e.options.TargetModel,
		"messages": e.messages,
		"stream":   request.Stream,
	}
	if len(e.system) > 0 {
		value["system"] = e.system
	}

	maxTokens := e.encodeMaxTokens(request.MaxOutputTokens)
	if maxTokens > 0 {
		value["max_tokens"] = maxTokens
	}
	e.encodeTools(request, value)
	e.encodeSampling(request, value)
	e.encodeStopSequences(request.StopSequences, value)
	e.encodeCandidateCount(request.CandidateCount)
	e.encodeOutputFormat(request.OutputFormat, value)
	e.encodeMetadata(request.Metadata, value)
	e.encodeExtensions(request.Extensions, value)
	e.validateCacheControls(value)
	return value
}

func (e *requestEncoder) encodeMaxTokens(requested *int) int {
	value := e.options.DefaultMaxOutputTokens
	if requested != nil {
		value = *requested
	}
	if value > 0 {
		return value
	}
	e.addError(diagnosticMissingMaxTokens, "Anthropic Messages requires a positive max_tokens value", "max_output_tokens", nil)
	return 0
}

func (e *requestEncoder) encodeTools(request canonical.Request, value map[string]any) {
	if len(request.Tools) == 0 {
		if request.ToolChoice != nil {
			e.addLossy(diagnosticContentFieldLossy, "tool choice is ignored because no tools are defined", "tool_choice", mustRaw(request.ToolChoice))
		}
		if request.ParallelToolCalls != nil {
			e.addLossy(diagnosticParallelUnsupported, "parallel_tool_calls is ignored because no tools are defined", "parallel_tool_calls", mustRaw(request.ParallelToolCalls))
		}
		return
	}

	tools := make([]any, 0, len(request.Tools))
	names := make(map[string]struct{}, len(request.Tools))
	for index, tool := range request.Tools {
		path := fmt.Sprintf("tools.%d", index)
		if tool.Name == "" {
			e.addError(diagnosticInvalidCanonicalRequest, "tool name is required", path+".name", nil)
			continue
		}
		if _, exists := names[tool.Name]; exists {
			e.addError(diagnosticInvalidCanonicalRequest, fmt.Sprintf("duplicate tool name %q", tool.Name), path+".name", quotedRaw(tool.Name))
			continue
		}
		names[tool.Name] = struct{}{}
		if tool.InputSchema == nil {
			e.addError(diagnosticInvalidCanonicalRequest, "tool input schema must be a JSON object", path+".input_schema", nil)
			continue
		}

		encoded := map[string]any{"name": tool.Name, "input_schema": tool.InputSchema}
		if tool.Description != nil {
			encoded["description"] = *tool.Description
		}
		if tool.Strict != nil && *tool.Strict {
			if !e.options.Profile.StrictTools {
				e.addLossy(diagnosticToolStrictUnsupported, "target profile does not support strict tool schemas", path+".strict", []byte("true"))
			} else if e.validateSchema(tool.InputSchema, path+".input_schema", false) {
				encoded["strict"] = true
			}
		}
		e.encodeToolExtensions(tool.Extensions, path+".extensions", encoded)
		tools = append(tools, encoded)
	}
	if len(tools) > 0 {
		value["tools"] = tools
	}

	choice := e.encodeToolChoice(request.ToolChoice, request.ParallelToolCalls, names)
	if choice != nil {
		value["tool_choice"] = choice
	}
}

func (e *requestEncoder) encodeToolChoice(choice *canonical.ToolChoice, parallel *bool, names map[string]struct{}) map[string]any {
	result := map[string]any{"type": "auto"}
	if choice != nil {
		switch choice.Mode {
		case canonical.ToolChoiceAuto:
			result["type"] = "auto"
		case canonical.ToolChoiceNone:
			result["type"] = "none"
		case canonical.ToolChoiceRequired:
			result["type"] = "any"
		case canonical.ToolChoiceNamed:
			if choice.Name == nil || *choice.Name == "" {
				e.addError(diagnosticInvalidCanonicalRequest, "named tool choice requires a name", "tool_choice.name", nil)
				return nil
			}
			if _, exists := names[*choice.Name]; !exists {
				e.addError(diagnosticInvalidCanonicalRequest, "named tool choice does not reference a defined tool", "tool_choice.name", quotedRaw(*choice.Name))
				return nil
			}
			result["type"] = "tool"
			result["name"] = *choice.Name
		default:
			e.addError(diagnosticInvalidCanonicalRequest, fmt.Sprintf("unknown tool choice mode %q", choice.Mode), "tool_choice.mode", quotedRaw(string(choice.Mode)))
			return nil
		}
	}
	if parallel == nil {
		return result
	}
	if !*parallel {
		result["disable_parallel_tool_use"] = true
		return result
	}
	if e.options.Profile.ParallelToolCalls {
		return result
	}
	e.addLossy(diagnosticParallelUnsupported, "target profile does not support parallel tool calls", "parallel_tool_calls", []byte("true"))
	return result
}

func (e *requestEncoder) encodeSampling(request canonical.Request, value map[string]any) {
	if request.Temperature != nil {
		if *request.Temperature < 0 || *request.Temperature > 1 {
			e.addError(diagnosticInvalidCanonicalRequest, "temperature must be between 0 and 1", "temperature", mustRaw(*request.Temperature))
		} else if e.options.Profile.Temperature {
			value["temperature"] = *request.Temperature
		} else {
			e.addLossy(canonical.DiagnosticSamplingParameterUnsupported, "target profile does not support temperature", "temperature", mustRaw(*request.Temperature))
		}
	}
	if request.TopP == nil {
		return
	}
	if *request.TopP < 0 || *request.TopP > 1 {
		e.addError(diagnosticInvalidCanonicalRequest, "top_p must be between 0 and 1", "top_p", mustRaw(*request.TopP))
		return
	}
	if e.options.Profile.TopP {
		value["top_p"] = *request.TopP
		return
	}
	e.addLossy(canonical.DiagnosticSamplingParameterUnsupported, "target profile does not support top_p", "top_p", mustRaw(*request.TopP))
}

func (e *requestEncoder) encodeStopSequences(stops []string, value map[string]any) {
	if len(stops) == 0 {
		return
	}
	for index, stop := range stops {
		if stop != "" {
			continue
		}
		e.addError(diagnosticInvalidCanonicalRequest, "stop sequence cannot be empty", fmt.Sprintf("stop_sequences.%d", index), quotedRaw(stop))
	}
	if !e.options.Profile.StopSequences {
		e.addLossy(canonical.DiagnosticSamplingParameterUnsupported, "target profile does not support stop sequences", "stop_sequences", mustRaw(stops))
		return
	}
	value["stop_sequences"] = stops
}

func (e *requestEncoder) encodeCandidateCount(count *int) {
	if count == nil || *count == 1 {
		return
	}
	if *count < 1 {
		e.addError(diagnosticInvalidCanonicalRequest, "candidate count must be positive", "candidate_count", mustRaw(*count))
		return
	}
	e.addLossy(canonical.DiagnosticCandidateCountUnsupported, "Anthropic Messages returns one candidate per request", "candidate_count", mustRaw(*count))
}

func (e *requestEncoder) encodeOutputFormat(format *canonical.OutputFormat, value map[string]any) {
	if format == nil || format.Type == canonical.OutputFormatText {
		return
	}
	if format.Type == canonical.OutputFormatJSONObject {
		e.addLossy(canonical.DiagnosticResponseFormatLossy, "Anthropic has no schema-less json_object output mode", "output_format", mustRaw(format))
		if e.options.Mode == canonical.ModeEmulate {
			e.system = append(e.system, map[string]any{
				"type": "text",
				"text": "Return one valid JSON object. Do not include markdown or text outside the JSON object.",
			})
			value["system"] = e.system
		}
		return
	}
	if format.Type != canonical.OutputFormatJSONSchema {
		e.addError(diagnosticInvalidCanonicalRequest, fmt.Sprintf("unknown output format %q", format.Type), "output_format.type", quotedRaw(string(format.Type)))
		return
	}
	if !e.options.Profile.StructuredOutput {
		e.addLossy(diagnosticStructuredUnsupported, "target profile does not support structured output", "output_format", mustRaw(format))
		return
	}
	if format.Schema == nil {
		e.addError(diagnosticInvalidCanonicalRequest, "structured output schema must be a JSON object", "output_format.schema", nil)
		return
	}
	if !e.validateSchema(format.Schema, "output_format.schema", true) {
		return
	}
	if format.Name != nil {
		e.addLossy(canonical.DiagnosticResponseFormatLossy, "Anthropic output_config does not carry the canonical schema name", "output_format.name", quotedRaw(*format.Name))
	}
	if format.Description != nil {
		e.addLossy(canonical.DiagnosticResponseFormatLossy, "Anthropic output_config does not carry the canonical schema description", "output_format.description", quotedRaw(*format.Description))
	}
	if format.Strict != nil && !*format.Strict {
		e.addLossy(canonical.DiagnosticResponseFormatLossy, "Anthropic structured output remains constrained when canonical strict is false", "output_format.strict", []byte("false"))
	}
	value["output_config"] = map[string]any{
		"format": map[string]any{"type": "json_schema", "schema": format.Schema},
	}
}

func (e *requestEncoder) validateSchema(schema canonical.Object, path string, rootObject bool) bool {
	valid := true
	if rootObject {
		typeRaw, exists := schema["type"]
		if exists {
			var typeName string
			if err := json.Unmarshal(typeRaw, &typeName); err != nil || typeName != "object" {
				e.addLossy(diagnosticSchemaUnsupported, "Anthropic structured output requires an object root schema", path+".type", typeRaw)
				valid = false
			}
		}
	}
	if e.findUnsupportedSchemaKeywords(schema, path) {
		valid = false
	}
	return valid
}

func (e *requestEncoder) findUnsupportedSchemaKeywords(value any, path string) bool {
	encoded, err := json.Marshal(value)
	if err != nil {
		e.addError(diagnosticInvalidCanonicalRequest, fmt.Sprintf("encode JSON schema: %v", err), path, nil)
		return true
	}
	var decoded any
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.UseNumber()
	if err := decoder.Decode(&decoded); err != nil {
		e.addError(diagnosticInvalidCanonicalRequest, fmt.Sprintf("decode JSON schema: %v", err), path, encoded)
		return true
	}
	return e.walkSchema(decoded, path)
}

func (e *requestEncoder) walkSchema(value any, path string) bool {
	object, ok := value.(map[string]any)
	if !ok {
		array, arrayOK := value.([]any)
		if !arrayOK {
			return false
		}
		unsupported := false
		for index, item := range array {
			if e.walkSchema(item, fmt.Sprintf("%s.%d", path, index)) {
				unsupported = true
			}
		}
		return unsupported
	}

	unsupported := false
	for key, item := range object {
		itemPath := path + "." + key
		if unsupportedSchemaKeyword(key) {
			e.addLossy(diagnosticSchemaUnsupported, fmt.Sprintf("Anthropic structured schemas do not support %q", key), itemPath, mustRaw(item))
			unsupported = true
		}
		if e.walkSchema(item, itemPath) {
			unsupported = true
		}
	}
	return unsupported
}

func unsupportedSchemaKeyword(keyword string) bool {
	switch keyword {
	case "if", "then", "else", "not", "oneOf", "patternProperties", "dependentSchemas", "dependentRequired", "unevaluatedProperties", "unevaluatedItems":
		return true
	default:
		return false
	}
}

func (e *requestEncoder) encodeMetadata(metadata map[string]string, value map[string]any) {
	if len(metadata) == 0 {
		return
	}
	keys := sortedStringKeys(metadata)
	if !e.options.Profile.Metadata {
		for _, key := range keys {
			e.addLossy(diagnosticMetadataUnsupported, "target profile does not support Anthropic request metadata", "metadata."+key, quotedRaw(metadata[key]))
		}
		return
	}

	userID, exists := metadata["user_id"]
	if exists && userID != "" {
		value["metadata"] = map[string]any{"user_id": userID}
	}
	for _, key := range keys {
		if key == "user_id" && metadata[key] != "" {
			continue
		}
		e.addLossy(diagnosticMetadataUnsupported, "Anthropic metadata only supports a non-empty user_id", "metadata."+key, quotedRaw(metadata[key]))
	}
}

func (e *requestEncoder) encodeExtensions(extensions canonical.Object, value map[string]any) {
	if len(extensions) == 0 {
		return
	}
	keys := make([]string, 0, len(extensions))
	for key := range extensions {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		switch key {
		case "cache_control":
			e.encodeTopLevelCacheControl(extensions[key], value)
		case "prompt_cache_key", "prompt_cache_options", "prompt_cache_retention":
			e.encodeOpenAICacheDirective(key, extensions[key])
		case "prompt_cache_breakpoint":
			e.addError(
				canonical.DiagnosticCacheBreakpointUnsupported,
				"prompt_cache_breakpoint is only valid on supported OpenAI input content blocks",
				"extensions."+key,
				extensions[key],
			)
		case "top_k":
			e.encodeTopKExtension(extensions[key], value)
		default:
			e.addLossy(diagnosticRequestExtension, fmt.Sprintf("canonical extension %q is not passed through to Anthropic", key), "extensions."+key, extensions[key])
		}
	}
}

func (e *requestEncoder) encodeTopLevelCacheControl(raw json.RawMessage, value map[string]any) {
	control := e.decodeCacheControl(raw, "extensions.cache_control")
	if control == nil {
		return
	}
	if !e.cacheControlEnabled("extensions.cache_control", raw) {
		return
	}
	value["cache_control"] = control
}

func (e *requestEncoder) encodePartExtensions(part canonical.Part, role canonical.Role, path string, block any) {
	if len(part.Extensions) == 0 {
		return
	}

	keys := sortedObjectKeys(part.Extensions)
	for _, key := range keys {
		raw := part.Extensions[key]
		extensionPath := path + ".extensions." + key
		switch key {
		case "cache_control":
			e.encodePartCacheControl(raw, extensionPath, part, role, block)
		case "prompt_cache_breakpoint":
			if err := validateOpenAIPromptCacheBreakpoint(raw); err != nil {
				e.addError(canonical.DiagnosticInvalidCacheControl, err.Error(), extensionPath, raw)
				continue
			}
			if !validOpenAIPromptCacheBreakpointPosition(part, role) {
				e.addError(
					canonical.DiagnosticCacheBreakpointUnsupported,
					"OpenAI prompt cache breakpoints are only valid on input text, image, and file blocks",
					extensionPath,
					raw,
				)
				continue
			}
			e.addLossy(
				canonical.DiagnosticCacheControlProviderMismatch,
				"OpenAI prompt cache breakpoint is not passed through to Anthropic",
				extensionPath,
				raw,
			)
		case "prompt_cache_key", "prompt_cache_options", "prompt_cache_retention":
			e.addError(
				canonical.DiagnosticInvalidCacheControl,
				fmt.Sprintf("cache directive %q is only valid at the request top level", key),
				extensionPath,
				raw,
			)
		default:
			e.addLossy(diagnosticContentFieldLossy, fmt.Sprintf("canonical part extension %q is not passed through to Anthropic", key), extensionPath, raw)
		}
	}
}

func (e *requestEncoder) encodeOpenAICacheDirective(name string, raw json.RawMessage) {
	err := validateOpenAICacheDirective(name, raw)
	path := "extensions." + name
	if err != nil {
		e.addError(canonical.DiagnosticInvalidCacheControl, err.Error(), path, raw)
		return
	}
	e.addLossy(
		canonical.DiagnosticCacheControlProviderMismatch,
		fmt.Sprintf("OpenAI cache directive %q is not passed through to Anthropic", name),
		path,
		raw,
	)
}

func validateOpenAICacheDirective(name string, raw json.RawMessage) error {
	var err error
	switch name {
	case "prompt_cache_key":
		var key string
		key, err = decodeOpenAICacheString(raw, name)
		if err == nil && strings.TrimSpace(key) == "" {
			err = fmt.Errorf("prompt_cache_key must be a non-empty string")
		}
	case "prompt_cache_options":
		err = validateOpenAIPromptCacheOptions(raw)
	case "prompt_cache_retention":
		err = validateOpenAIPromptCacheRetention(raw)
	}
	return err
}

func validateOpenAIPromptCacheRetention(raw json.RawMessage) error {
	value, err := decodeOpenAICacheString(raw, "prompt_cache_retention")
	if err != nil {
		return err
	}
	if value == "in_memory" || value == "24h" {
		return nil
	}
	return fmt.Errorf("prompt_cache_retention must be %q or %q", "in_memory", "24h")
}

func validateOpenAIPromptCacheOptions(raw json.RawMessage) error {
	object, err := canonical.DecodeObject(raw)
	if err != nil {
		return fmt.Errorf("prompt_cache_options must be an object: %w", err)
	}
	for _, name := range sortedObjectKeys(object) {
		if name != "mode" && name != "ttl" {
			return fmt.Errorf("prompt_cache_options contains unsupported field %q", name)
		}
	}
	if modeRaw, exists := object["mode"]; exists {
		mode, decodeErr := decodeOpenAICacheString(modeRaw, "prompt_cache_options.mode")
		if decodeErr != nil {
			return decodeErr
		}
		if mode != "implicit" && mode != "explicit" {
			return fmt.Errorf("prompt_cache_options.mode must be %q or %q", "implicit", "explicit")
		}
	}
	if ttlRaw, exists := object["ttl"]; exists {
		ttl, decodeErr := decodeOpenAICacheString(ttlRaw, "prompt_cache_options.ttl")
		if decodeErr != nil {
			return decodeErr
		}
		if ttl != "30m" {
			return fmt.Errorf("prompt_cache_options.ttl must be %q", "30m")
		}
	}
	return nil
}

func validateOpenAIPromptCacheBreakpoint(raw json.RawMessage) error {
	object, err := canonical.DecodeObject(raw)
	if err != nil {
		return fmt.Errorf("prompt_cache_breakpoint must be an object: %w", err)
	}
	for _, name := range sortedObjectKeys(object) {
		if name != "mode" {
			return fmt.Errorf("prompt_cache_breakpoint contains unsupported field %q", name)
		}
	}
	modeRaw, exists := object["mode"]
	if !exists {
		return fmt.Errorf("prompt_cache_breakpoint.mode is required")
	}
	mode, err := decodeOpenAICacheString(modeRaw, "prompt_cache_breakpoint.mode")
	if err != nil {
		return err
	}
	if mode != "explicit" {
		return fmt.Errorf("prompt_cache_breakpoint.mode must be %q", "explicit")
	}
	return nil
}

func decodeOpenAICacheString(raw json.RawMessage, path string) (string, error) {
	if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return "", fmt.Errorf("%s must be a string", path)
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return "", fmt.Errorf("%s must be a string", path)
	}
	return value, nil
}

func validOpenAIPromptCacheBreakpointPosition(part canonical.Part, role canonical.Role) bool {
	if role == canonical.RoleAssistant || role == canonical.Role("tool_result") {
		return false
	}
	if part.Kind == canonical.PartText && part.Text == "" {
		return false
	}
	if part.Kind != canonical.PartText && part.Kind != canonical.PartImage && part.Kind != canonical.PartFile {
		return false
	}
	return true
}

func (e *requestEncoder) encodePartCacheControl(raw json.RawMessage, path string, part canonical.Part, role canonical.Role, block any) {
	control := e.decodeCacheControl(raw, path)
	if control == nil {
		return
	}
	if role == canonical.Role("tool_result") {
		e.addError(
			canonical.DiagnosticCacheBreakpointUnsupported,
			"cache_control on tool result sub-content is not supported by this transformer",
			path,
			raw,
		)
		return
	}
	if part.Kind == canonical.PartText && part.Text == "" {
		e.addError(canonical.DiagnosticCacheBreakpointUnsupported, "empty text blocks cannot be cache breakpoints", path, raw)
		return
	}
	if part.Kind != canonical.PartText && part.Kind != canonical.PartImage && part.Kind != canonical.PartFile {
		e.addError(canonical.DiagnosticCacheBreakpointUnsupported, fmt.Sprintf("cache_control is not supported on canonical %q parts", part.Kind), path, raw)
		return
	}
	if (part.Kind == canonical.PartImage || part.Kind == canonical.PartFile) && role != canonical.RoleUser {
		e.addError(canonical.DiagnosticCacheBreakpointUnsupported, "cache_control on image and file blocks is only supported in user messages", path, raw)
		return
	}
	if !e.cacheControlEnabled(path, raw) {
		return
	}
	encoded, ok := block.(map[string]any)
	if !ok {
		e.addError(canonical.DiagnosticCacheBreakpointUnsupported, "cache_control cannot be attached to an omitted Anthropic content block", path, raw)
		return
	}
	encoded["cache_control"] = control
}

func (e *requestEncoder) encodeToolExtensions(extensions canonical.Object, path string, encoded map[string]any) {
	if len(extensions) == 0 {
		return
	}

	keys := sortedObjectKeys(extensions)
	for _, key := range keys {
		raw := extensions[key]
		extensionPath := path + "." + key
		if key == "prompt_cache_key" || key == "prompt_cache_options" || key == "prompt_cache_retention" {
			e.addError(
				canonical.DiagnosticInvalidCacheControl,
				fmt.Sprintf("cache directive %q is only valid at the request top level", key),
				extensionPath,
				raw,
			)
			continue
		}
		if key == "prompt_cache_breakpoint" {
			e.addError(
				canonical.DiagnosticCacheBreakpointUnsupported,
				"prompt_cache_breakpoint is not valid on tool definitions",
				extensionPath,
				raw,
			)
			continue
		}
		if key != "cache_control" {
			e.addLossy(diagnosticRequestExtension, fmt.Sprintf("canonical tool extension %q is not passed through to Anthropic", key), extensionPath, raw)
			continue
		}
		control := e.decodeCacheControl(raw, extensionPath)
		if control == nil {
			continue
		}
		if !e.cacheControlEnabled(extensionPath, raw) {
			continue
		}
		encoded["cache_control"] = control
	}
}

func (e *requestEncoder) decodeCacheControl(raw json.RawMessage, path string) *anthropicCacheControl {
	object, err := canonical.DecodeObject(raw)
	if err != nil {
		e.addError(canonical.DiagnosticInvalidCacheControl, fmt.Sprintf("cache_control must be an object: %v", err), path, raw)
		return nil
	}

	typeRaw, hasType := object["type"]
	delete(object, "type")
	if !hasType {
		e.addError(canonical.DiagnosticInvalidCacheControl, "cache_control.type is required", path+".type", nil)
		return nil
	}
	var typeName string
	if err := json.Unmarshal(typeRaw, &typeName); err != nil || typeName != "ephemeral" {
		e.addError(canonical.DiagnosticInvalidCacheControl, "cache_control.type must be \"ephemeral\"", path+".type", typeRaw)
		return nil
	}

	var ttl *string
	if ttlRaw, exists := object["ttl"]; exists {
		delete(object, "ttl")
		var value string
		if err := json.Unmarshal(ttlRaw, &value); err != nil || (value != "5m" && value != "1h") {
			e.addError(canonical.DiagnosticInvalidCacheControl, "cache_control.ttl must be \"5m\" or \"1h\"", path+".ttl", ttlRaw)
			return nil
		}
		ttl = &value
	}

	if len(object) > 0 {
		key := sortedObjectKeys(object)[0]
		e.addError(canonical.DiagnosticInvalidCacheControl, fmt.Sprintf("unknown cache_control field %q", key), path+"."+key, object[key])
		return nil
	}
	return &anthropicCacheControl{Type: typeName, TTL: ttl, path: path}
}

func (e *requestEncoder) cacheControlEnabled(path string, raw json.RawMessage) bool {
	cache := e.options.Profile.PromptCache
	if cache.Mode == capabilities.PromptCacheAnthropic && e.options.Profile.Endpoint == capabilities.EndpointMessages {
		return true
	}
	if cache.Mode == capabilities.PromptCacheOpenAILegacy || cache.Mode == capabilities.PromptCacheOpenAI56 {
		e.addLossy(canonical.DiagnosticCacheControlProviderMismatch, "Anthropic cache_control cannot be used with an OpenAI prompt cache profile", path, raw)
		return false
	}
	e.addLossy(canonical.DiagnosticCacheControlUnsupported, "target profile does not support Anthropic prompt caching", path, raw)
	return false
}

func (e *requestEncoder) validateCacheControls(value map[string]any) {
	automatic, hasAutomatic := value["cache_control"].(*anthropicCacheControl)
	blocks := collectAnthropicCacheBlocks(value)

	explicitCount := 0
	lastCacheable := -1
	fifthExplicitPath := "cache_control"
	for index, block := range blocks {
		lastCacheable = index
		if block.control != nil {
			explicitCount++
			if explicitCount == 5 {
				fifthExplicitPath = block.control.path
			}
		}
	}
	if explicitCount > 4 {
		e.addError(canonical.DiagnosticInvalidCacheControl, "Anthropic requests support at most 4 explicit cache breakpoints", fifthExplicitPath, nil)
	}
	if !hasAutomatic || lastCacheable < 0 {
		e.validateCacheTTLOrder(blocks, nil, -1)
		return
	}

	lastControl := blocks[lastCacheable].control
	if lastControl != nil && effectiveCacheTTL(lastControl) != effectiveCacheTTL(automatic) {
		e.addError(
			canonical.DiagnosticInvalidCacheControl,
			"automatic and explicit cache controls on the last cacheable block must use the same TTL",
			automatic.path,
			nil,
		)
		return
	}
	if lastControl != nil {
		e.validateCacheTTLOrder(blocks, nil, -1)
		return
	}
	if explicitCount > 3 {
		e.addError(canonical.DiagnosticInvalidCacheControl, "automatic caching leaves room for at most 3 explicit cache breakpoints", automatic.path, nil)
	}
	e.validateCacheTTLOrder(blocks, automatic, lastCacheable)
}

func (e *requestEncoder) validateCacheTTLOrder(blocks []anthropicCacheBlock, automatic *anthropicCacheControl, automaticIndex int) {
	seenFiveMinute := false
	for index, block := range blocks {
		controls := []*anthropicCacheControl{block.control}
		if index == automaticIndex {
			controls = append(controls, automatic)
		}
		for _, control := range controls {
			if control == nil {
				continue
			}
			if effectiveCacheTTL(control) == "5m" {
				seenFiveMinute = true
				continue
			}
			if !seenFiveMinute {
				continue
			}
			e.addError(canonical.DiagnosticInvalidCacheControl, "1h cache controls must appear before 5m cache controls", control.path, nil)
			return
		}
	}
}

func collectAnthropicCacheBlocks(value map[string]any) []anthropicCacheBlock {
	blocks := make([]anthropicCacheBlock, 0)
	appendBlock := func(raw any) {
		block, ok := raw.(map[string]any)
		if !ok || !anthropicBlockCacheable(block) {
			return
		}
		control, _ := block["cache_control"].(*anthropicCacheControl)
		blocks = append(blocks, anthropicCacheBlock{control: control})
	}

	if tools, ok := value["tools"].([]any); ok {
		for _, tool := range tools {
			appendBlock(tool)
		}
	}
	if system, ok := value["system"].([]any); ok {
		for _, block := range system {
			appendBlock(block)
		}
	}
	if messages, ok := value["messages"].([]any); ok {
		for _, rawMessage := range messages {
			message, ok := rawMessage.(map[string]any)
			if !ok {
				continue
			}
			content, _ := message["content"].([]any)
			for _, block := range content {
				appendBlock(block)
			}
		}
	}
	return blocks
}

func anthropicBlockCacheable(block map[string]any) bool {
	if _, isTool := block["name"]; isTool {
		_, hasSchema := block["input_schema"]
		return hasSchema
	}
	typeName, _ := block["type"].(string)
	if typeName == "text" {
		text, _ := block["text"].(string)
		return text != ""
	}
	return typeName == "image" || typeName == "document" || typeName == "tool_use" || typeName == "tool_result"
}

func effectiveCacheTTL(control *anthropicCacheControl) string {
	if control != nil && control.TTL != nil {
		return *control.TTL
	}
	return "5m"
}

func sortedObjectKeys(values canonical.Object) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func (e *requestEncoder) encodeTopKExtension(raw json.RawMessage, value map[string]any) {
	var topK int
	if err := json.Unmarshal(raw, &topK); err != nil || topK < 1 {
		e.addError(diagnosticInvalidCanonicalRequest, "top_k extension must be a positive integer", "extensions.top_k", raw)
		return
	}
	if !e.options.Profile.TopK {
		e.addLossy(canonical.DiagnosticSamplingParameterUnsupported, "target profile does not support top_k", "extensions.top_k", raw)
		return
	}
	value["top_k"] = topK
}

func sortedStringKeys(values map[string]string) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func (e *requestEncoder) addError(code canonical.DiagnosticCode, message, path string, source json.RawMessage) {
	e.diagnostics = append(e.diagnostics, makeDiagnostic(canonical.SeverityError, code, message, path, source))
}

func (e *requestEncoder) addLossy(code canonical.DiagnosticCode, message, path string, source json.RawMessage) {
	severity := canonical.SeverityWarning
	if e.options.Mode == canonical.ModeStrict {
		severity = canonical.SeverityError
	}
	e.diagnostics = append(e.diagnostics, makeDiagnostic(severity, code, message, path, source))
}

func makeDiagnostic(severity canonical.Severity, code canonical.DiagnosticCode, message, path string, source json.RawMessage) canonical.Diagnostic {
	diagnostic := canonical.Diagnostic{Severity: severity, Code: code, Message: message, SourceValue: cloneRaw(source)}
	if path != "" {
		diagnostic.Path = &path
	}
	return diagnostic
}

func quotedRaw(value string) json.RawMessage {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil
	}
	return encoded
}

func mustRaw(value any) json.RawMessage {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil
	}
	return encoded
}

func cloneRaw(value json.RawMessage) json.RawMessage {
	return append(json.RawMessage(nil), value...)
}
