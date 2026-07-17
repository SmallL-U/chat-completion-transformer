package openairesponses

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"chat-completion-transformer/internal/assets"
	"chat-completion-transformer/internal/canonical"
	"chat-completion-transformer/internal/capabilities"
)

// InstructionPolicy controls whether leading system and developer messages
// remain in the input item list or become the top-level instructions string.
type InstructionPolicy string

const (
	InstructionPolicyPreserveMessages InstructionPolicy = "preserve_messages"
	InstructionPolicyExtractLeading   InstructionPolicy = "extract_leading"
)

// EncodeOptions supplies routing and endpoint-specific behavior. TargetModel
// is deliberately separate from the canonical model alias.
type EncodeOptions struct {
	TargetModel       string
	Mode              canonical.Mode
	Profile           capabilities.Profile
	Resolver          assets.Resolver
	InstructionPolicy InstructionPolicy
}

// EncodeRequest converts a canonical request into an OpenAI Responses request.
func EncodeRequest(
	ctx context.Context,
	request canonical.Request,
	options EncodeOptions,
) canonical.Result[map[string]any] {
	if ctx == nil {
		return canonical.Failure[map[string]any]([]canonical.Diagnostic{
			diagnostic(canonical.SeverityError, DiagnosticInvalidEncodeOptions, "context is required", "", nil),
		})
	}
	if err := ctx.Err(); err != nil {
		return canonical.Failure[map[string]any]([]canonical.Diagnostic{
			diagnostic(canonical.SeverityError, DiagnosticRequestCanceled, err.Error(), "", nil),
		})
	}

	mode := options.Mode
	if mode == "" {
		mode = canonical.ModeCompatible
	}

	diagnostics := canonical.ValidateToolHistory(request.Turns)
	diagnostics = append(diagnostics, validateRequestTurns(request.Turns)...)
	if mode != canonical.ModeStrict && mode != canonical.ModeCompatible && mode != canonical.ModeEmulate {
		diagnostics = append(diagnostics, diagnostic(
			canonical.SeverityError,
			DiagnosticInvalidEncodeOptions,
			fmt.Sprintf("unknown transform mode %q", mode),
			"mode",
			mode,
		))
	}
	if options.TargetModel == "" {
		diagnostics = append(diagnostics, diagnostic(
			canonical.SeverityError,
			canonical.DiagnosticModelMappingMissing,
			"an OpenAI Responses target model is required",
			"model",
			request.ModelAlias,
		))
	}
	if options.Profile.Provider != capabilities.ProviderOpenAI || options.Profile.Endpoint != capabilities.EndpointResponses {
		diagnostics = append(diagnostics, diagnostic(
			canonical.SeverityError,
			DiagnosticInvalidEncodeOptions,
			"the capability profile must describe the OpenAI Responses endpoint",
			"profile",
			map[string]any{"provider": options.Profile.Provider, "endpoint": options.Profile.Endpoint},
		))
	}
	if options.TargetModel != "" && options.Profile.Model != options.TargetModel {
		diagnostics = append(diagnostics, diagnostic(
			canonical.SeverityError,
			DiagnosticInvalidEncodeOptions,
			"the capability profile model must match the target model",
			"profile.model",
			options.Profile.Model,
		))
	}

	policy := options.InstructionPolicy
	if policy == "" {
		policy = InstructionPolicyPreserveMessages
	}
	if policy != InstructionPolicyPreserveMessages && policy != InstructionPolicyExtractLeading {
		diagnostics = append(diagnostics, lossyDiagnostic(
			mode,
			DiagnosticInstructionPolicyFallback,
			fmt.Sprintf("unknown instruction policy %q; preserving messages", policy),
			"instruction_policy",
			policy,
		))
		policy = InstructionPolicyPreserveMessages
	}

	instructions, firstInputTurn, extractionDiagnostics := extractInstructions(request.Turns, policy, mode)
	diagnostics = append(diagnostics, extractionDiagnostics...)

	resolver := options.Resolver
	if resolver == nil {
		resolver = assets.NativeResolver{}
	}
	input, inputDiagnostics := encodeInput(ctx, request.Turns[firstInputTurn:], options.Profile, resolver, mode, firstInputTurn)
	diagnostics = append(diagnostics, inputDiagnostics...)

	tools, toolDiagnostics := encodeTools(request.Tools, options.Profile, mode)
	diagnostics = append(diagnostics, toolDiagnostics...)

	value := map[string]any{
		"model":  options.TargetModel,
		"input":  input,
		"tools":  tools,
		"stream": request.Stream,
	}
	if instructions != "" {
		value["instructions"] = instructions
	}
	if request.ToolChoice != nil {
		value["tool_choice"] = encodeToolChoice(*request.ToolChoice)
		if !validToolChoiceMode(request.ToolChoice.Mode) {
			diagnostics = append(diagnostics, diagnostic(
				canonical.SeverityError,
				DiagnosticUnsupportedRequestField,
				fmt.Sprintf("unknown tool choice mode %q", request.ToolChoice.Mode),
				"tool_choice.mode",
				request.ToolChoice.Mode,
			))
		}
		if request.ToolChoice.Mode == canonical.ToolChoiceNamed {
			validateNamedToolChoice(request, &diagnostics)
		}
	}
	if request.ParallelToolCalls != nil {
		if options.Profile.ParallelToolCalls {
			value["parallel_tool_calls"] = *request.ParallelToolCalls
		} else {
			diagnostics = append(diagnostics, lossyDiagnostic(
				mode,
				DiagnosticUnsupportedRequestField,
				"the selected Responses profile does not support parallel_tool_calls",
				"parallel_tool_calls",
				*request.ParallelToolCalls,
			))
		}
	}
	if request.MaxOutputTokens != nil {
		value["max_output_tokens"] = *request.MaxOutputTokens
		if *request.MaxOutputTokens < 1 {
			diagnostics = append(diagnostics, diagnostic(
				canonical.SeverityError,
				DiagnosticUnsupportedRequestField,
				"max_output_tokens must be at least 1",
				"max_output_tokens",
				*request.MaxOutputTokens,
			))
		}
	}
	if request.Temperature != nil {
		if options.Profile.Temperature {
			value["temperature"] = *request.Temperature
		} else {
			diagnostics = append(diagnostics, unsupportedSamplingDiagnostic(mode, "temperature", *request.Temperature))
		}
	}
	if request.TopP != nil {
		if options.Profile.TopP {
			value["top_p"] = *request.TopP
		} else {
			diagnostics = append(diagnostics, unsupportedSamplingDiagnostic(mode, "top_p", *request.TopP))
		}
	}
	if len(request.StopSequences) > 0 {
		if options.Profile.StopSequences {
			value["stop"] = append([]string(nil), request.StopSequences...)
		} else {
			diagnostics = append(diagnostics, lossyDiagnostic(
				mode,
				DiagnosticUnsupportedRequestField,
				"the selected Responses profile does not support stop sequences",
				"stop_sequences",
				request.StopSequences,
			))
		}
	}
	if request.CandidateCount != nil && *request.CandidateCount > 1 {
		diagnostics = append(diagnostics, lossyDiagnostic(
			mode,
			canonical.DiagnosticCandidateCountUnsupported,
			"OpenAI Responses does not support n; the caller must issue multiple requests",
			"candidate_count",
			*request.CandidateCount,
		))
	}
	if request.CandidateCount != nil && *request.CandidateCount < 1 {
		diagnostics = append(diagnostics, diagnostic(
			canonical.SeverityError,
			DiagnosticUnsupportedRequestField,
			"candidate_count must be at least 1",
			"candidate_count",
			*request.CandidateCount,
		))
	}
	if request.OutputFormat != nil && request.OutputFormat.Type != canonical.OutputFormatText {
		if options.Profile.StructuredOutput {
			value["text"] = map[string]any{"format": encodeOutputFormat(*request.OutputFormat)}
		} else {
			diagnostics = append(diagnostics, lossyDiagnostic(
				mode,
				canonical.DiagnosticResponseFormatLossy,
				"the selected Responses profile does not support structured output",
				"output_format",
				request.OutputFormat,
			))
		}
	}
	diagnostics = append(diagnostics, validateOutputFormat(request.OutputFormat)...)
	if len(request.Metadata) > 0 {
		if options.Profile.Metadata {
			metadata := make(map[string]string, len(request.Metadata))
			for key, item := range request.Metadata {
				metadata[key] = item
			}
			value["metadata"] = metadata
		} else {
			diagnostics = append(diagnostics, lossyDiagnostic(
				mode,
				DiagnosticUnsupportedRequestField,
				"the selected Responses profile does not support metadata",
				"metadata",
				request.Metadata,
			))
		}
	}

	extensionNames := make([]string, 0, len(request.Extensions))
	for name := range request.Extensions {
		extensionNames = append(extensionNames, name)
	}
	sort.Strings(extensionNames)
	for _, name := range extensionNames {
		diagnostics = append(diagnostics, lossyDiagnostic(
			mode,
			DiagnosticUnsupportedExtension,
			fmt.Sprintf("canonical extension %q is not forwarded to OpenAI Responses", name),
			"extensions."+name,
			request.Extensions[name],
		))
	}

	return canonical.Success(value, diagnostics)
}

func extractInstructions(
	turns []canonical.Turn,
	policy InstructionPolicy,
	mode canonical.Mode,
) (string, int, []canonical.Diagnostic) {
	if policy != InstructionPolicyExtractLeading {
		return "", 0, nil
	}

	prefixEnd := 0
	for prefixEnd < len(turns) && isInstructionTurn(turns[prefixEnd]) {
		prefixEnd++
	}
	if prefixEnd == 0 {
		return "", 0, nil
	}

	for index := prefixEnd; index < len(turns); index++ {
		if !isInstructionTurn(turns[index]) {
			continue
		}
		diagnostic := lossyDiagnostic(
			mode,
			DiagnosticInstructionPolicyFallback,
			"cannot extract instructions when system or developer messages appear mid-conversation",
			fmt.Sprintf("turns.%d", index),
			nil,
		)
		return "", 0, []canonical.Diagnostic{diagnostic}
	}

	roles := make(map[canonical.Role]struct{})
	messages := make([]string, 0, prefixEnd)
	for index := 0; index < prefixEnd; index++ {
		turn := turns[index]
		roles[turn.Role] = struct{}{}
		var builder strings.Builder
		for partIndex, part := range turn.Content {
			if part.Kind == canonical.PartText {
				builder.WriteString(part.Text)
				continue
			}
			diagnostic := lossyDiagnostic(
				mode,
				DiagnosticInstructionPolicyFallback,
				"only text system and developer messages can be extracted into instructions",
				fmt.Sprintf("turns.%d.content.%d", index, partIndex),
				part,
			)
			return "", 0, []canonical.Diagnostic{diagnostic}
		}
		messages = append(messages, builder.String())
	}

	diagnostics := make([]canonical.Diagnostic, 0, 1)
	if len(roles) > 1 {
		diagnostics = append(diagnostics, lossyDiagnostic(
			mode,
			canonical.DiagnosticRolePriorityCollapsed,
			"extracting both system and developer messages collapses their role priority",
			"turns",
			nil,
		))
	}

	return strings.Join(messages, "\n\n"), prefixEnd, diagnostics
}

func isInstructionTurn(turn canonical.Turn) bool {
	if turn.Kind != canonical.TurnMessage {
		return false
	}
	return turn.Role == canonical.RoleSystem || turn.Role == canonical.RoleDeveloper
}

func encodeInput(
	ctx context.Context,
	turns []canonical.Turn,
	profile capabilities.Profile,
	resolver assets.Resolver,
	mode canonical.Mode,
	offset int,
) ([]any, []canonical.Diagnostic) {
	input := make([]any, 0, len(turns))
	diagnostics := make([]canonical.Diagnostic, 0)
	seenNonInstruction := false

	for relativeIndex, turn := range turns {
		turnIndex := relativeIndex + offset
		if isInstructionTurn(turn) && seenNonInstruction && !profile.MidConversationSystem {
			diagnostics = append(diagnostics, lossyDiagnostic(
				mode,
				canonical.DiagnosticMidConversationSystemUnsupported,
				"the selected Responses profile does not support mid-conversation system or developer messages",
				fmt.Sprintf("turns.%d", turnIndex),
				turn,
			))
			continue
		}
		if !isInstructionTurn(turn) {
			seenNonInstruction = true
		}
		if turn.Kind == canonical.TurnToolResults {
			items, itemDiagnostics := encodeToolResults(ctx, turn.Results, profile, resolver, mode, turnIndex)
			input = append(input, items...)
			diagnostics = append(diagnostics, itemDiagnostics...)
			continue
		}
		if turn.Kind != canonical.TurnMessage {
			diagnostics = append(diagnostics, lossyDiagnostic(
				mode,
				DiagnosticUnsupportedRequestField,
				fmt.Sprintf("unknown canonical turn kind %q", turn.Kind),
				fmt.Sprintf("turns.%d.kind", turnIndex),
				turn.Kind,
			))
			continue
		}
		if !validRole(turn.Role) {
			diagnostics = append(diagnostics, diagnostic(
				canonical.SeverityError,
				DiagnosticUnsupportedRequestField,
				fmt.Sprintf("unknown canonical role %q", turn.Role),
				fmt.Sprintf("turns.%d.role", turnIndex),
				turn.Role,
			))
			continue
		}

		content, hasContent, contentDiagnostics := encodeMessageContent(
			ctx,
			turn.Role,
			turn.Content,
			profile,
			resolver,
			mode,
			fmt.Sprintf("turns.%d.content", turnIndex),
		)
		diagnostics = append(diagnostics, contentDiagnostics...)
		if hasContent {
			item := map[string]any{"role": string(turn.Role), "content": content}
			if hasRefusal(turn.Content) {
				item["type"] = "message"
				item["status"] = "completed"
			}
			input = append(input, item)
		}
		if turn.Name != nil {
			diagnostics = append(diagnostics, lossyDiagnostic(
				mode,
				DiagnosticUnsupportedRequestField,
				"OpenAI Responses message items do not preserve Chat Completions message names",
				fmt.Sprintf("turns.%d.name", turnIndex),
				*turn.Name,
			))
		}
		if turn.Role != canonical.RoleAssistant {
			if len(turn.ToolCalls) > 0 {
				diagnostics = append(diagnostics, diagnostic(
					canonical.SeverityError,
					DiagnosticUnsupportedRequestField,
					"only assistant turns may contain tool calls",
					fmt.Sprintf("turns.%d.tool_calls", turnIndex),
					turn.ToolCalls,
				))
			}
			continue
		}
		for callIndex, call := range turn.ToolCalls {
			input = append(input, map[string]any{
				"type":      "function_call",
				"call_id":   call.ID,
				"name":      call.Name,
				"arguments": call.ArgumentsRaw,
			})
			if call.ID == "" || call.Name == "" {
				diagnostics = append(diagnostics, diagnostic(
					canonical.SeverityError,
					DiagnosticUnsupportedRequestField,
					"function calls require non-empty call IDs and names",
					fmt.Sprintf("turns.%d.tool_calls.%d", turnIndex, callIndex),
					call,
				))
			}
		}
	}

	return input, diagnostics
}

func encodeMessageContent(
	ctx context.Context,
	role canonical.Role,
	parts []canonical.Part,
	profile capabilities.Profile,
	resolver assets.Resolver,
	mode canonical.Mode,
	path string,
) (any, bool, []canonical.Diagnostic) {
	if len(parts) == 0 {
		return nil, false, nil
	}
	if allText(parts) {
		if !profile.Content.Text {
			diagnostic := unsupportedPartDiagnostic(mode, "text", path, parts)
			return nil, false, []canonical.Diagnostic{diagnostic}
		}
		var builder strings.Builder
		for _, part := range parts {
			builder.WriteString(part.Text)
		}
		return builder.String(), true, nil
	}

	content := make([]any, 0, len(parts))
	diagnostics := make([]canonical.Diagnostic, 0)
	for index, part := range parts {
		partPath := fmt.Sprintf("%s.%d", path, index)
		item, ok, itemDiagnostics := encodeContentPart(ctx, role, part, profile, resolver, mode, partPath)
		diagnostics = append(diagnostics, itemDiagnostics...)
		if ok {
			content = append(content, item)
		}
	}
	if len(content) == 0 {
		return nil, false, diagnostics
	}
	return content, true, diagnostics
}

func encodeContentPart(
	ctx context.Context,
	role canonical.Role,
	part canonical.Part,
	profile capabilities.Profile,
	resolver assets.Resolver,
	mode canonical.Mode,
	path string,
) (any, bool, []canonical.Diagnostic) {
	switch part.Kind {
	case canonical.PartText:
		if !profile.Content.Text {
			return nil, false, []canonical.Diagnostic{unsupportedPartDiagnostic(mode, "text", path, part)}
		}
		typeName := "input_text"
		if role == canonical.RoleAssistant {
			typeName = "output_text"
			return map[string]any{"type": typeName, "text": part.Text, "annotations": []any{}}, true, nil
		}
		return map[string]any{"type": typeName, "text": part.Text}, true, nil
	case canonical.PartRefusal:
		if role != canonical.RoleAssistant {
			return nil, false, []canonical.Diagnostic{unsupportedPartDiagnostic(mode, "refusal on a non-assistant message", path, part)}
		}
		return map[string]any{"type": "refusal", "refusal": part.Text}, true, nil
	case canonical.PartImage:
		if role == canonical.RoleAssistant || !profile.Content.Image {
			return nil, false, []canonical.Diagnostic{unsupportedPartDiagnostic(mode, "image", path, part)}
		}
		item, itemDiagnostic := encodeImage(ctx, part, profile, resolver, mode, path)
		if itemDiagnostic != nil {
			return nil, false, []canonical.Diagnostic{*itemDiagnostic}
		}
		return item, true, nil
	case canonical.PartFile:
		if role == canonical.RoleAssistant || !profile.Content.File {
			return nil, false, []canonical.Diagnostic{unsupportedPartDiagnostic(mode, "file", path, part)}
		}
		item, itemDiagnostic := encodeFile(ctx, part, profile, resolver, mode, path)
		if itemDiagnostic != nil {
			return nil, false, []canonical.Diagnostic{*itemDiagnostic}
		}
		return item, true, nil
	case canonical.PartAudio:
		return nil, false, []canonical.Diagnostic{unsupportedPartDiagnostic(mode, "audio", path, part)}
	case canonical.PartOpaque:
		return nil, false, []canonical.Diagnostic{unsupportedPartDiagnostic(mode, "opaque provider content", path, part.Value)}
	default:
		return nil, false, []canonical.Diagnostic{unsupportedPartDiagnostic(mode, fmt.Sprintf("unknown part kind %q", part.Kind), path, part)}
	}
}

func encodeImage(
	ctx context.Context,
	part canonical.Part,
	profile capabilities.Profile,
	resolver assets.Resolver,
	mode canonical.Mode,
	path string,
) (map[string]any, *canonical.Diagnostic) {
	if part.Source == nil {
		diagnostic := diagnostic(canonical.SeverityError, DiagnosticAssetResolutionFailed, "image source is required", path+".source", nil)
		return nil, &diagnostic
	}
	resolved, err := resolver.ResolveForResponses(ctx, *part.Source)
	if err != nil {
		if contextError := ctx.Err(); contextError != nil {
			diagnostic := diagnostic(canonical.SeverityError, DiagnosticRequestCanceled, contextError.Error(), path+".source", nil)
			return nil, &diagnostic
		}
		diagnostic := lossyDiagnostic(mode, DiagnosticAssetResolutionFailed, fmt.Sprintf("resolve image: %v", err), path+".source", part.Source)
		return nil, &diagnostic
	}
	if !supportsImageSource(profile, resolved.Kind) {
		diagnostic := unsupportedPartDiagnostic(mode, fmt.Sprintf("image source %q", resolved.Kind), path+".source", part.Source)
		return nil, &diagnostic
	}

	item := map[string]any{"type": "input_image"}
	if part.Detail != nil {
		item["detail"] = string(*part.Detail)
	}
	switch resolved.Kind {
	case canonical.AssetSourceURL:
		item["image_url"] = resolved.URL
	case canonical.AssetSourceBase64:
		item["image_url"] = dataURL(resolved.MediaType, resolved.Data)
	case canonical.AssetSourceFileID:
		item["file_id"] = resolved.FileID
	}
	return item, nil
}

func encodeFile(
	ctx context.Context,
	part canonical.Part,
	profile capabilities.Profile,
	resolver assets.Resolver,
	mode canonical.Mode,
	path string,
) (map[string]any, *canonical.Diagnostic) {
	if part.Source == nil {
		diagnostic := diagnostic(canonical.SeverityError, DiagnosticAssetResolutionFailed, "file source is required", path+".source", nil)
		return nil, &diagnostic
	}
	resolved, err := resolver.ResolveForResponses(ctx, *part.Source)
	if err != nil {
		if contextError := ctx.Err(); contextError != nil {
			diagnostic := diagnostic(canonical.SeverityError, DiagnosticRequestCanceled, contextError.Error(), path+".source", nil)
			return nil, &diagnostic
		}
		diagnostic := lossyDiagnostic(mode, DiagnosticAssetResolutionFailed, fmt.Sprintf("resolve file: %v", err), path+".source", part.Source)
		return nil, &diagnostic
	}
	if !supportsFileSource(profile, resolved.Kind) {
		diagnostic := unsupportedPartDiagnostic(mode, fmt.Sprintf("file source %q", resolved.Kind), path+".source", part.Source)
		return nil, &diagnostic
	}

	item := map[string]any{"type": "input_file"}
	if part.Filename != nil {
		item["filename"] = *part.Filename
	}
	switch resolved.Kind {
	case canonical.AssetSourceURL:
		item["file_url"] = resolved.URL
	case canonical.AssetSourceBase64:
		item["file_data"] = dataURL(resolved.MediaType, resolved.Data)
	case canonical.AssetSourceFileID:
		item["file_id"] = resolved.FileID
	default:
		diagnostic := unsupportedPartDiagnostic(mode, fmt.Sprintf("file source %q", resolved.Kind), path+".source", part.Source)
		return nil, &diagnostic
	}
	return item, nil
}

func encodeToolResults(
	ctx context.Context,
	results []canonical.ToolResult,
	profile capabilities.Profile,
	resolver assets.Resolver,
	mode canonical.Mode,
	turnIndex int,
) ([]any, []canonical.Diagnostic) {
	items := make([]any, 0, len(results))
	diagnostics := make([]canonical.Diagnostic, 0)
	for resultIndex, result := range results {
		path := fmt.Sprintf("turns.%d.results.%d", turnIndex, resultIndex)
		output, outputDiagnostics := encodeToolOutput(ctx, result.Content, profile, resolver, mode, path+".content")
		diagnostics = append(diagnostics, outputDiagnostics...)
		items = append(items, map[string]any{
			"type":    "function_call_output",
			"call_id": result.CallID,
			"output":  output,
		})
		if result.CallID == "" {
			diagnostics = append(diagnostics, diagnostic(canonical.SeverityError, DiagnosticUnsupportedRequestField, "function call output requires a call ID", path+".call_id", nil))
		}
		if result.IsError != nil && *result.IsError {
			diagnostics = append(diagnostics, lossyDiagnostic(
				mode,
				DiagnosticUnsupportedRequestField,
				"Responses function_call_output has no portable is_error flag",
				path+".is_error",
				true,
			))
		}
	}
	return items, diagnostics
}

func encodeToolOutput(
	ctx context.Context,
	parts []canonical.Part,
	profile capabilities.Profile,
	resolver assets.Resolver,
	mode canonical.Mode,
	path string,
) (any, []canonical.Diagnostic) {
	if len(parts) == 0 {
		return "", nil
	}
	if allText(parts) {
		if !profile.Content.Text {
			return "", []canonical.Diagnostic{unsupportedPartDiagnostic(mode, "text", path, parts)}
		}
		var builder strings.Builder
		for _, part := range parts {
			builder.WriteString(part.Text)
		}
		return builder.String(), nil
	}

	content := make([]any, 0, len(parts))
	diagnostics := make([]canonical.Diagnostic, 0)
	for index, part := range parts {
		partPath := fmt.Sprintf("%s.%d", path, index)
		item, ok, itemDiagnostics := encodeContentPart(ctx, canonical.RoleUser, part, profile, resolver, mode, partPath)
		diagnostics = append(diagnostics, itemDiagnostics...)
		if ok {
			content = append(content, item)
		}
	}
	return content, diagnostics
}

func encodeTools(
	tools []canonical.ToolDefinition,
	profile capabilities.Profile,
	mode canonical.Mode,
) ([]any, []canonical.Diagnostic) {
	encoded := make([]any, 0, len(tools))
	diagnostics := make([]canonical.Diagnostic, 0)
	seenNames := make(map[string]int, len(tools))
	for index, tool := range tools {
		path := fmt.Sprintf("tools.%d", index)
		validSchema := true
		if strings.TrimSpace(tool.Name) == "" {
			diagnostics = append(diagnostics, diagnostic(
				canonical.SeverityError,
				DiagnosticUnsupportedRequestField,
				"function tool name must not be empty",
				path+".name",
				tool.Name,
			))
		} else if firstIndex, exists := seenNames[tool.Name]; exists {
			diagnostics = append(diagnostics, diagnostic(
				canonical.SeverityError,
				DiagnosticUnsupportedRequestField,
				fmt.Sprintf("function tool name %q duplicates tools.%d", tool.Name, firstIndex),
				path+".name",
				tool.Name,
			))
		} else {
			seenNames[tool.Name] = index
		}
		if tool.InputSchema == nil {
			validSchema = false
			diagnostics = append(diagnostics, diagnostic(
				canonical.SeverityError,
				DiagnosticUnsupportedSchema,
				"function tool input schema is required",
				path+".input_schema",
				nil,
			))
		} else if !schemaHasObjectRoot(tool.InputSchema) {
			validSchema = false
			diagnostics = append(diagnostics, diagnostic(
				canonical.SeverityError,
				DiagnosticUnsupportedSchema,
				"function tool input schema must have an object root",
				path+".input_schema.type",
				tool.InputSchema["type"],
			))
		}

		strict := false
		if tool.Strict != nil && *tool.Strict {
			if validSchema {
				diagnostics = append(diagnostics, validateStrictSchema(tool.InputSchema, path+".input_schema")...)
			}
			if profile.StrictTools {
				strict = true
			} else {
				diagnostics = append(diagnostics, lossyDiagnostic(
					mode,
					DiagnosticUnsupportedRequestField,
					"the selected Responses profile does not support strict function tools",
					fmt.Sprintf("tools.%d.strict", index),
					true,
				))
			}
		}

		parameters := tool.InputSchema
		if parameters == nil {
			parameters = canonical.Object{}
		}
		item := map[string]any{
			"type":       "function",
			"name":       tool.Name,
			"parameters": parameters,
			"strict":     strict,
		}
		if tool.Description != nil {
			item["description"] = *tool.Description
		}
		encoded = append(encoded, item)
	}
	return encoded, diagnostics
}

func validateRequestTurns(turns []canonical.Turn) []canonical.Diagnostic {
	if len(turns) == 0 {
		return []canonical.Diagnostic{diagnostic(
			canonical.SeverityError,
			DiagnosticUnsupportedRequestField,
			"at least one conversation turn is required",
			"turns",
			nil,
		)}
	}

	diagnostics := make([]canonical.Diagnostic, 0)
	for index, turn := range turns {
		if turn.Kind != canonical.TurnMessage {
			continue
		}
		if turn.Role == canonical.RoleAssistant && (len(turn.Content) > 0 || len(turn.ToolCalls) > 0) {
			continue
		}
		if turn.Role != canonical.RoleAssistant && len(turn.Content) > 0 {
			continue
		}
		diagnostics = append(diagnostics, diagnostic(
			canonical.SeverityError,
			DiagnosticUnsupportedRequestField,
			"message turn must contain content or assistant tool calls",
			fmt.Sprintf("turns.%d", index),
			turn,
		))
	}
	return diagnostics
}

func schemaHasObjectRoot(schema canonical.Object) bool {
	var typeName string
	if err := json.Unmarshal(schema["type"], &typeName); err != nil {
		return false
	}
	return typeName == "object"
}

func validateNamedToolChoice(request canonical.Request, diagnostics *[]canonical.Diagnostic) {
	if request.ToolChoice == nil || request.ToolChoice.Mode != canonical.ToolChoiceNamed {
		return
	}
	if request.ToolChoice.Name == nil || *request.ToolChoice.Name == "" {
		*diagnostics = append(*diagnostics, diagnostic(
			canonical.SeverityError,
			DiagnosticUnsupportedRequestField,
			"named tool choice requires a non-empty function name",
			"tool_choice.name",
			nil,
		))
		return
	}
	for _, tool := range request.Tools {
		if tool.Name == *request.ToolChoice.Name {
			return
		}
	}
	*diagnostics = append(*diagnostics, diagnostic(
		canonical.SeverityError,
		DiagnosticUnsupportedRequestField,
		"named tool choice must reference a declared function tool",
		"tool_choice.name",
		*request.ToolChoice.Name,
	))
}

func validateOutputFormat(format *canonical.OutputFormat) []canonical.Diagnostic {
	if format == nil {
		return nil
	}
	if format.Type != canonical.OutputFormatText && format.Type != canonical.OutputFormatJSONObject && format.Type != canonical.OutputFormatJSONSchema {
		return []canonical.Diagnostic{diagnostic(
			canonical.SeverityError,
			DiagnosticUnsupportedSchema,
			fmt.Sprintf("unknown output format type %q", format.Type),
			"output_format.type",
			format.Type,
		)}
	}
	if format.Type != canonical.OutputFormatJSONSchema {
		return nil
	}
	diagnostics := make([]canonical.Diagnostic, 0)
	if format.Name == nil || *format.Name == "" {
		diagnostics = append(diagnostics, diagnostic(
			canonical.SeverityError,
			DiagnosticUnsupportedSchema,
			"JSON Schema output format requires a non-empty name",
			"output_format.name",
			nil,
		))
	}
	if format.Schema == nil {
		diagnostics = append(diagnostics, diagnostic(
			canonical.SeverityError,
			DiagnosticUnsupportedSchema,
			"JSON Schema output format requires an object schema",
			"output_format.schema",
			nil,
		))
		return diagnostics
	}
	if format.Strict != nil && *format.Strict {
		diagnostics = append(diagnostics, validateStrictSchema(format.Schema, "output_format.schema")...)
	}
	return diagnostics
}

func validateStrictSchema(schema canonical.Object, path string) []canonical.Diagnostic {
	diagnostics := make([]canonical.Diagnostic, 0)
	var typeName string
	if raw, ok := schema["type"]; ok {
		_ = json.Unmarshal(raw, &typeName)
	}
	if typeName != "object" {
		diagnostics = append(diagnostics, diagnostic(
			canonical.SeverityError,
			DiagnosticUnsupportedSchema,
			"strict Responses schemas must have an object root",
			path+".type",
			schema["type"],
		))
	}
	var additionalProperties bool
	if err := json.Unmarshal(schema["additionalProperties"], &additionalProperties); err != nil || additionalProperties {
		diagnostics = append(diagnostics, diagnostic(
			canonical.SeverityError,
			DiagnosticUnsupportedSchema,
			"strict Responses object schemas require additionalProperties: false",
			path+".additionalProperties",
			schema["additionalProperties"],
		))
	}

	unsupported := []string{"default", "allOf", "oneOf", "not", "if", "then", "else", "dependentRequired", "dependentSchemas", "patternProperties", "unevaluatedProperties"}
	for _, keyword := range unsupported {
		if raw, exists := schema[keyword]; exists {
			diagnostics = append(diagnostics, diagnostic(
				canonical.SeverityError,
				DiagnosticUnsupportedSchema,
				fmt.Sprintf("strict Responses schemas do not support %q", keyword),
				path+"."+keyword,
				raw,
			))
		}
	}
	return diagnostics
}

func encodeToolChoice(choice canonical.ToolChoice) any {
	if choice.Mode != canonical.ToolChoiceNamed {
		return string(choice.Mode)
	}
	name := ""
	if choice.Name != nil {
		name = *choice.Name
	}
	return map[string]any{"type": "function", "name": name}
}

func encodeOutputFormat(format canonical.OutputFormat) map[string]any {
	if format.Type == canonical.OutputFormatJSONObject {
		return map[string]any{"type": "json_object"}
	}
	if format.Type != canonical.OutputFormatJSONSchema {
		return map[string]any{"type": "text"}
	}

	value := map[string]any{
		"type":   "json_schema",
		"schema": format.Schema,
	}
	if format.Name != nil {
		value["name"] = *format.Name
	}
	if format.Description != nil {
		value["description"] = *format.Description
	}
	if format.Strict != nil {
		value["strict"] = *format.Strict
	}
	return value
}

func validToolChoiceMode(mode canonical.ToolChoiceMode) bool {
	return mode == canonical.ToolChoiceAuto ||
		mode == canonical.ToolChoiceNone ||
		mode == canonical.ToolChoiceRequired ||
		mode == canonical.ToolChoiceNamed
}

func validRole(role canonical.Role) bool {
	return role == canonical.RoleSystem ||
		role == canonical.RoleDeveloper ||
		role == canonical.RoleUser ||
		role == canonical.RoleAssistant
}

func unsupportedSamplingDiagnostic(mode canonical.Mode, name string, value float64) canonical.Diagnostic {
	return lossyDiagnostic(
		mode,
		canonical.DiagnosticSamplingParameterUnsupported,
		fmt.Sprintf("the selected Responses profile does not support %s", name),
		name,
		value,
	)
}

func unsupportedPartDiagnostic(mode canonical.Mode, name string, path string, source any) canonical.Diagnostic {
	return lossyDiagnostic(
		mode,
		canonical.DiagnosticUnsupportedContentPart,
		fmt.Sprintf("OpenAI Responses cannot represent %s in this position or profile", name),
		path,
		source,
	)
}

func supportsImageSource(profile capabilities.Profile, kind canonical.AssetSourceKind) bool {
	switch kind {
	case canonical.AssetSourceURL:
		return profile.Images.URL
	case canonical.AssetSourceBase64:
		return profile.Images.Base64
	case canonical.AssetSourceFileID:
		return profile.Images.FileID
	default:
		return false
	}
}

func supportsFileSource(profile capabilities.Profile, kind canonical.AssetSourceKind) bool {
	switch kind {
	case canonical.AssetSourceURL:
		return profile.Files.URL
	case canonical.AssetSourceBase64:
		return profile.Files.Base64
	case canonical.AssetSourceFileID:
		return profile.Files.FileID
	default:
		return false
	}
}

func allText(parts []canonical.Part) bool {
	for _, part := range parts {
		if part.Kind != canonical.PartText {
			return false
		}
	}
	return true
}

func hasRefusal(parts []canonical.Part) bool {
	for _, part := range parts {
		if part.Kind == canonical.PartRefusal {
			return true
		}
	}
	return false
}

func dataURL(mediaType string, data string) string {
	return "data:" + mediaType + ";base64," + data
}
