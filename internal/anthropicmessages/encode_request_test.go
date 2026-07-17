package anthropicmessages

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"chat-completion-transformer/internal/assets"
	"chat-completion-transformer/internal/canonical"
	"chat-completion-transformer/internal/capabilities"
)

func TestEncodeRequestParallelToolsAndLeadingInstructions(t *testing.T) {
	request := canonical.Request{
		Turns: []canonical.Turn{
			messageTurn(canonical.RoleSystem, "be concise"),
			messageTurn(canonical.RoleDeveloper, "use tools"),
			messageTurn(canonical.RoleUser, "weather"),
			{
				Kind:    canonical.TurnMessage,
				Role:    canonical.RoleAssistant,
				Content: []canonical.Part{{Kind: canonical.PartText, Text: "checking"}},
				ToolCalls: []canonical.ToolCall{
					{ID: "call_a", Name: "weather", ArgumentsRaw: `{"city":"Beijing"}`},
					{ID: "call_b", Name: "weather", ArgumentsRaw: `{"city":"Shanghai"}`},
				},
			},
			{
				Kind: canonical.TurnToolResults,
				Results: []canonical.ToolResult{
					{CallID: "call_a", Content: textParts("20C")},
					{CallID: "call_b", Content: textParts("24C")},
				},
			},
			messageTurn(canonical.RoleUser, "compare"),
		},
		Tools: []canonical.ToolDefinition{{
			Name:        "weather",
			InputSchema: objectSchema(),
		}},
		ToolChoice:        &canonical.ToolChoice{Mode: canonical.ToolChoiceRequired},
		ParallelToolCalls: boolPointer(false),
		MaxOutputTokens:   intPointerValue(256),
	}

	result := EncodeRequest(context.Background(), request, testEncodeOptions())
	if !result.OK || result.Value == nil {
		t.Fatalf("result = %#v", result)
	}
	if !hasDiagnostic(result.Diagnostics, canonical.DiagnosticRolePriorityCollapsed) {
		t.Fatalf("diagnostics = %#v", result.Diagnostics)
	}

	var encoded map[string]any
	if err := json.Unmarshal(*result.Value, &encoded); err != nil {
		t.Fatal(err)
	}
	system := encoded["system"].([]any)
	if len(system) != 2 {
		t.Fatalf("system = %#v", system)
	}
	messages := encoded["messages"].([]any)
	if len(messages) != 3 {
		t.Fatalf("messages = %#v", messages)
	}
	assistant := messages[1].(map[string]any)["content"].([]any)
	if len(assistant) != 3 || assistant[0].(map[string]any)["type"] != "text" || assistant[1].(map[string]any)["type"] != "tool_use" {
		t.Fatalf("assistant content = %#v", assistant)
	}
	results := messages[2].(map[string]any)["content"].([]any)
	if len(results) != 3 || results[0].(map[string]any)["type"] != "tool_result" || results[1].(map[string]any)["type"] != "tool_result" || results[2].(map[string]any)["type"] != "text" {
		t.Fatalf("tool result content = %#v", results)
	}
	choice := encoded["tool_choice"].(map[string]any)
	if choice["type"] != "any" || choice["disable_parallel_tool_use"] != true {
		t.Fatalf("tool_choice = %#v", choice)
	}
}

func TestEncodeRequestRejectsInvalidToolArgumentsAndHistory(t *testing.T) {
	tests := []struct {
		name  string
		turns []canonical.Turn
		code  canonical.DiagnosticCode
	}{
		{
			name: "arguments are not object",
			turns: []canonical.Turn{
				messageTurn(canonical.RoleUser, "go"),
				{Kind: canonical.TurnMessage, Role: canonical.RoleAssistant, ToolCalls: []canonical.ToolCall{{ID: "call", Name: "tool", ArgumentsRaw: `[]`}}},
				{Kind: canonical.TurnToolResults, Results: []canonical.ToolResult{{CallID: "call", Content: textParts("done")}}},
			},
			code: canonical.DiagnosticInvalidToolArgumentsJSON,
		},
		{
			name: "orphan result",
			turns: []canonical.Turn{
				messageTurn(canonical.RoleUser, "go"),
				{Kind: canonical.TurnToolResults, Results: []canonical.ToolResult{{CallID: "missing", Content: textParts("done")}}},
			},
			code: canonical.DiagnosticOrphanToolResult,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := canonical.Request{Turns: test.turns, MaxOutputTokens: intPointerValue(64)}
			result := EncodeRequest(context.Background(), request, testEncodeOptions())
			if result.OK || !hasDiagnostic(result.Diagnostics, test.code) {
				t.Fatalf("result = %#v", result)
			}
		})
	}
}

func TestEncodeRequestDiagnosesUnsupportedFields(t *testing.T) {
	temperature := 0.5
	topP := 0.8
	strict := true
	count := 2
	parallel := true
	request := canonical.Request{
		Turns:             []canonical.Turn{messageTurn(canonical.RoleUser, "hello")},
		Tools:             []canonical.ToolDefinition{{Name: "lookup", InputSchema: objectSchema(), Strict: &strict}},
		ParallelToolCalls: &parallel,
		MaxOutputTokens:   intPointerValue(64),
		Temperature:       &temperature,
		TopP:              &topP,
		StopSequences:     []string{"done"},
		CandidateCount:    &count,
		OutputFormat:      &canonical.OutputFormat{Type: canonical.OutputFormatJSONObject},
		Metadata:          map[string]string{"trace": "abc"},
		Extensions:        canonical.Object{"top_k": json.RawMessage(`20`), "future": json.RawMessage(`true`)},
	}
	options := testEncodeOptions()
	options.Profile.Temperature = false
	options.Profile.TopP = false
	options.Profile.StopSequences = false
	options.Profile.StrictTools = false
	options.Profile.ParallelToolCalls = false
	options.Profile.Metadata = false

	result := EncodeRequest(context.Background(), request, options)
	if !result.OK || result.Value == nil {
		t.Fatalf("result = %#v", result)
	}
	for _, code := range []canonical.DiagnosticCode{
		canonical.DiagnosticSamplingParameterUnsupported,
		diagnosticToolStrictUnsupported,
		diagnosticParallelUnsupported,
		canonical.DiagnosticCandidateCountUnsupported,
		canonical.DiagnosticResponseFormatLossy,
		diagnosticMetadataUnsupported,
		diagnosticRequestExtension,
	} {
		if !hasDiagnostic(result.Diagnostics, code) {
			t.Errorf("missing %q in %#v", code, result.Diagnostics)
		}
	}

	var encoded map[string]any
	if err := json.Unmarshal(*result.Value, &encoded); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"temperature", "top_p", "stop_sequences", "output_config", "metadata", "top_k", "future"} {
		if _, exists := encoded[field]; exists {
			t.Errorf("unexpected field %q in %#v", field, encoded)
		}
	}
}

func TestEncodeRequestMapsSupportedTopKExtension(t *testing.T) {
	request := canonical.Request{
		Turns:           []canonical.Turn{messageTurn(canonical.RoleUser, "hello")},
		MaxOutputTokens: intPointerValue(64),
		Extensions:      canonical.Object{"top_k": json.RawMessage(`20`)},
	}
	options := testEncodeOptions()
	options.Profile.TopK = true
	result := EncodeRequest(context.Background(), request, options)
	if !result.OK || result.Value == nil {
		t.Fatalf("result = %#v", result)
	}
	var encoded map[string]any
	if err := json.Unmarshal(*result.Value, &encoded); err != nil {
		t.Fatal(err)
	}
	if encoded["top_k"] != float64(20) {
		t.Fatalf("encoded = %#v", encoded)
	}
}

func TestEncodeRequestStructuredSchemaAndImageSources(t *testing.T) {
	format := &canonical.OutputFormat{
		Type: canonical.OutputFormatJSONSchema,
		Name: stringPointer("answer"),
		Schema: canonical.Object{
			"type":       json.RawMessage(`"object"`),
			"properties": json.RawMessage(`{"answer":{"type":"string"}}`),
		},
	}
	request := canonical.Request{
		Turns: []canonical.Turn{{
			Kind: canonical.TurnMessage,
			Role: canonical.RoleUser,
			Content: []canonical.Part{
				{Kind: canonical.PartImage, Source: &canonical.AssetSource{Kind: canonical.AssetSourceURL, URL: "https://example.com/a.png"}},
				{Kind: canonical.PartImage, Source: &canonical.AssetSource{Kind: canonical.AssetSourceBase64, MediaType: "image/png", Data: "aA=="}},
				{Kind: canonical.PartImage, Source: &canonical.AssetSource{Kind: canonical.AssetSourceFileID, FileID: "file_1"}},
			},
		}},
		MaxOutputTokens: intPointerValue(64),
		OutputFormat:    format,
	}
	result := EncodeRequest(context.Background(), request, testEncodeOptions())
	if !result.OK || result.Value == nil {
		t.Fatalf("result = %#v", result)
	}
	var encoded map[string]any
	if err := json.Unmarshal(*result.Value, &encoded); err != nil {
		t.Fatal(err)
	}
	content := encoded["messages"].([]any)[0].(map[string]any)["content"].([]any)
	wantTypes := []string{"url", "base64", "file"}
	for index, want := range wantTypes {
		source := content[index].(map[string]any)["source"].(map[string]any)
		if source["type"] != want {
			t.Fatalf("source %d = %#v", index, source)
		}
	}
	if _, exists := encoded["output_config"]; !exists {
		t.Fatalf("encoded = %#v", encoded)
	}

	options := testEncodeOptions()
	options.Profile.Endpoint = capabilities.EndpointBedrockMessages
	options.Profile.Images.FileID = true
	options.Profile.Model = options.TargetModel
	request.Turns[0].Content = append(textParts("describe"), request.Turns[0].Content[2])
	result = EncodeRequest(context.Background(), request, options)
	if !result.OK || !hasDiagnostic(result.Diagnostics, diagnosticImageSourceUnsupported) {
		t.Fatalf("bedrock result = %#v", result)
	}
}

func TestEncodeRequestGatesDocumentSourcesAndEmulatesJSONObject(t *testing.T) {
	request := canonical.Request{
		Turns: []canonical.Turn{{
			Kind: canonical.TurnMessage,
			Role: canonical.RoleUser,
			Content: []canonical.Part{
				{Kind: canonical.PartText, Text: "read"},
				{Kind: canonical.PartFile, Source: &canonical.AssetSource{Kind: canonical.AssetSourceURL, URL: "https://example.com/a.pdf"}},
				{Kind: canonical.PartFile, Source: &canonical.AssetSource{Kind: canonical.AssetSourceBase64, MediaType: "application/pdf", Data: "aA=="}},
				{Kind: canonical.PartFile, Source: &canonical.AssetSource{Kind: canonical.AssetSourceFileID, FileID: "file_1"}},
			},
		}},
		MaxOutputTokens: intPointerValue(64),
		OutputFormat:    &canonical.OutputFormat{Type: canonical.OutputFormatJSONObject},
	}
	options := testEncodeOptions()
	options.Mode = canonical.ModeEmulate
	result := EncodeRequest(context.Background(), request, options)
	if !result.OK || result.Value == nil || !hasDiagnostic(result.Diagnostics, canonical.DiagnosticResponseFormatLossy) {
		t.Fatalf("result = %#v", result)
	}
	var encoded map[string]any
	if err := json.Unmarshal(*result.Value, &encoded); err != nil {
		t.Fatal(err)
	}
	if len(encoded["system"].([]any)) != 1 {
		t.Fatalf("system = %#v", encoded["system"])
	}
	content := encoded["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if len(content) != 4 {
		t.Fatalf("content = %#v", content)
	}

	options.Profile.Endpoint = capabilities.EndpointVertexMessages
	options.Profile.Model = options.TargetModel
	result = EncodeRequest(context.Background(), request, options)
	if !result.OK || !hasDiagnostic(result.Diagnostics, diagnosticFileSourceUnsupported) {
		t.Fatalf("vertex result = %#v", result)
	}
}

func TestEncodeRequestUsesCallerContextForResolver(t *testing.T) {
	type contextKey struct{}
	ctx := context.WithValue(context.Background(), contextKey{}, "request")
	resolver := &checkingResolver{expected: ctx}
	request := canonical.Request{
		Turns: []canonical.Turn{{
			Kind: canonical.TurnMessage,
			Role: canonical.RoleUser,
			Content: []canonical.Part{{
				Kind:   canonical.PartImage,
				Source: &canonical.AssetSource{Kind: canonical.AssetSourceURL, URL: "https://example.com/a.png"},
			}},
		}},
		MaxOutputTokens: intPointerValue(64),
	}
	options := testEncodeOptions()
	options.Resolver = resolver
	result := EncodeRequest(ctx, request, options)
	if !result.OK || !resolver.called {
		t.Fatalf("result = %#v, resolver = %#v", result, resolver)
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	result = EncodeRequest(canceled, request, options)
	if result.OK || !hasDiagnostic(result.Diagnostics, diagnosticContextCanceled) {
		t.Fatalf("canceled result = %#v", result)
	}
}

type checkingResolver struct {
	expected context.Context
	called   bool
}

func (r *checkingResolver) ResolveForResponses(context.Context, canonical.AssetSource) (assets.ResolvedAsset, error) {
	return assets.ResolvedAsset{}, errors.New("not used")
}

func (r *checkingResolver) ResolveForAnthropic(ctx context.Context, source canonical.AssetSource) (assets.ResolvedAsset, error) {
	r.called = true
	if ctx != r.expected {
		return assets.ResolvedAsset{}, errors.New("unexpected context")
	}
	return assets.ResolvedAsset{Kind: source.Kind, URL: source.URL}, nil
}

func testEncodeOptions() RequestEncodeOptions {
	return RequestEncodeOptions{
		TargetModel: "claude-test",
		Mode:        canonical.ModeCompatible,
		Profile: capabilities.Profile{
			Provider:              capabilities.ProviderAnthropic,
			Endpoint:              capabilities.EndpointMessages,
			Model:                 "claude-test",
			MidConversationSystem: true,
			Temperature:           true,
			TopP:                  true,
			StopSequences:         true,
			Metadata:              true,
			StructuredOutput:      true,
			StrictTools:           true,
			ParallelToolCalls:     true,
			Images: capabilities.ImageCapabilities{
				URL: true, Base64: true, FileID: true,
			},
			Files: capabilities.ImageCapabilities{
				URL: true, Base64: true, FileID: true,
			},
			Content: capabilities.ContentCapabilities{
				Text: true, Image: true, File: true,
			},
		},
	}
}

func messageTurn(role canonical.Role, text string) canonical.Turn {
	return canonical.Turn{Kind: canonical.TurnMessage, Role: role, Content: textParts(text)}
}

func textParts(text string) []canonical.Part {
	return []canonical.Part{{Kind: canonical.PartText, Text: text}}
}

func objectSchema() canonical.Object {
	return canonical.Object{"type": json.RawMessage(`"object"`), "properties": json.RawMessage(`{}`)}
}

func boolPointer(value bool) *bool {
	return &value
}

func intPointerValue(value int) *int {
	return &value
}

func stringPointer(value string) *string {
	return &value
}

func hasDiagnostic(diagnostics []canonical.Diagnostic, code canonical.DiagnosticCode) bool {
	for _, diagnostic := range diagnostics {
		if diagnostic.Code == code {
			return true
		}
	}
	return false
}
