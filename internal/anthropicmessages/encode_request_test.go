package anthropicmessages

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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

func TestEncodeRequestMapsAnthropicCacheControls(t *testing.T) {
	hour := json.RawMessage(`{"type":"ephemeral","ttl":"1h"}`)
	fiveMinutes := json.RawMessage(`{"type":"ephemeral"}`)
	explicitFiveMinutes := json.RawMessage(`{"type":"ephemeral","ttl":"5m"}`)
	request := canonical.Request{
		Turns: []canonical.Turn{
			{
				Kind: canonical.TurnMessage,
				Role: canonical.RoleSystem,
				Content: []canonical.Part{{
					Kind:       canonical.PartText,
					Text:       "stable system",
					Extensions: canonical.Object{"cache_control": hour},
				}},
			},
			{
				Kind: canonical.TurnMessage,
				Role: canonical.RoleUser,
				Content: []canonical.Part{{
					Kind:       canonical.PartText,
					Text:       "stable user",
					Extensions: canonical.Object{"cache_control": explicitFiveMinutes},
				}},
			},
			{
				Kind: canonical.TurnMessage,
				Role: canonical.RoleAssistant,
				Content: []canonical.Part{{
					Kind:       canonical.PartText,
					Text:       "stable assistant",
					Extensions: canonical.Object{"cache_control": fiveMinutes},
				}},
			},
		},
		Tools: []canonical.ToolDefinition{{
			Name:        "lookup",
			InputSchema: objectSchema(),
			Extensions:  canonical.Object{"cache_control": hour},
		}},
		MaxOutputTokens: intPointerValue(64),
		Extensions:      canonical.Object{"cache_control": fiveMinutes},
	}

	result := EncodeRequest(context.Background(), request, cacheEncodeOptions())
	if !result.OK || result.Value == nil {
		t.Fatalf("result = %#v", result)
	}
	var encoded map[string]any
	if err := json.Unmarshal(*result.Value, &encoded); err != nil {
		t.Fatal(err)
	}
	assertCacheControl(t, encoded["cache_control"], "")
	assertCacheControl(t, encoded["tools"].([]any)[0].(map[string]any)["cache_control"], "1h")
	assertCacheControl(t, encoded["system"].([]any)[0].(map[string]any)["cache_control"], "1h")
	messages := encoded["messages"].([]any)
	assertCacheControl(t, messages[0].(map[string]any)["content"].([]any)[0].(map[string]any)["cache_control"], "5m")
	assertCacheControl(t, messages[1].(map[string]any)["content"].([]any)[0].(map[string]any)["cache_control"], "")
}

func TestEncodeRequestMapsCacheControlToUserAssets(t *testing.T) {
	hour := json.RawMessage(`{"type":"ephemeral","ttl":"1h"}`)
	request := canonical.Request{
		Turns: []canonical.Turn{{
			Kind: canonical.TurnMessage,
			Role: canonical.RoleUser,
			Content: []canonical.Part{
				{
					Kind:       canonical.PartImage,
					Source:     &canonical.AssetSource{Kind: canonical.AssetSourceBase64, MediaType: "image/png", Data: "aA=="},
					Extensions: canonical.Object{"cache_control": hour},
				},
				{
					Kind:       canonical.PartFile,
					Source:     &canonical.AssetSource{Kind: canonical.AssetSourceBase64, MediaType: "application/pdf", Data: "aA=="},
					Extensions: canonical.Object{"cache_control": hour},
				},
			},
		}},
		MaxOutputTokens: intPointerValue(64),
	}

	result := EncodeRequest(context.Background(), request, cacheEncodeOptions())
	if !result.OK || result.Value == nil {
		t.Fatalf("result = %#v", result)
	}
	var encoded map[string]any
	if err := json.Unmarshal(*result.Value, &encoded); err != nil {
		t.Fatal(err)
	}
	content := encoded["messages"].([]any)[0].(map[string]any)["content"].([]any)
	assertCacheControl(t, content[0].(map[string]any)["cache_control"], "1h")
	assertCacheControl(t, content[1].(map[string]any)["cache_control"], "1h")
}

func TestEncodeRequestMapsChatPromptCacheBreakpointsToUserAssets(t *testing.T) {
	breakpoint := json.RawMessage(`{"mode":"explicit"}`)
	request := canonical.Request{
		Turns: []canonical.Turn{{
			Kind: canonical.TurnMessage,
			Role: canonical.RoleUser,
			Content: []canonical.Part{
				{
					Kind:       canonical.PartImage,
					Source:     &canonical.AssetSource{Kind: canonical.AssetSourceBase64, MediaType: "image/png", Data: "aA=="},
					Extensions: canonical.Object{"prompt_cache_breakpoint": breakpoint},
				},
				{
					Kind:       canonical.PartFile,
					Source:     &canonical.AssetSource{Kind: canonical.AssetSourceBase64, MediaType: "application/pdf", Data: "aA=="},
					Extensions: canonical.Object{"prompt_cache_breakpoint": breakpoint},
				},
			},
		}},
		MaxOutputTokens: intPointerValue(64),
		Extensions: canonical.Object{
			"prompt_cache_options": json.RawMessage(`{"mode":"explicit","ttl":"30m"}`),
		},
	}

	result := EncodeRequest(context.Background(), request, cacheEncodeOptions())
	if !result.OK || result.Value == nil {
		t.Fatalf("result = %#v", result)
	}
	var encoded map[string]any
	if err := json.Unmarshal(*result.Value, &encoded); err != nil {
		t.Fatal(err)
	}
	content := encoded["messages"].([]any)[0].(map[string]any)["content"].([]any)
	assertCacheControl(t, content[0].(map[string]any)["cache_control"], "1h")
	assertCacheControl(t, content[1].(map[string]any)["cache_control"], "1h")
}

func TestEncodeRequestRejectsNativeAndStandardCacheControlConflict(t *testing.T) {
	native := json.RawMessage(`{"type":"ephemeral","ttl":"1h"}`)
	standard := json.RawMessage(`{"mode":"explicit"}`)
	tests := []struct {
		name    string
		request canonical.Request
	}{
		{
			name: "top level",
			request: canonical.Request{
				Turns:           []canonical.Turn{messageTurn(canonical.RoleUser, "hello")},
				MaxOutputTokens: intPointerValue(64),
				Extensions: canonical.Object{
					"cache_control":        native,
					"prompt_cache_options": json.RawMessage(`{"mode":"implicit"}`),
				},
			},
		},
		{
			name: "content part",
			request: canonical.Request{
				Turns: []canonical.Turn{{
					Kind: canonical.TurnMessage,
					Role: canonical.RoleUser,
					Content: []canonical.Part{{
						Kind: canonical.PartText,
						Text: "hello",
						Extensions: canonical.Object{
							"cache_control":           native,
							"prompt_cache_breakpoint": standard,
						},
					}},
				}},
				MaxOutputTokens: intPointerValue(64),
				Extensions: canonical.Object{
					"prompt_cache_options": json.RawMessage(`{"mode":"explicit"}`),
				},
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := EncodeRequest(context.Background(), test.request, cacheEncodeOptions())
			if result.OK || !hasDiagnostic(result.Diagnostics, canonical.DiagnosticInvalidCacheControl) {
				t.Fatalf("result = %#v", result)
			}
		})
	}
}

func TestEncodeRequestRejectsMalformedCacheControl(t *testing.T) {
	tests := []struct {
		name string
		raw  json.RawMessage
	}{
		{name: "null", raw: json.RawMessage(`null`)},
		{name: "missing type", raw: json.RawMessage(`{}`)},
		{name: "wrong type", raw: json.RawMessage(`{"type":"persistent"}`)},
		{name: "empty ttl", raw: json.RawMessage(`{"type":"ephemeral","ttl":""}`)},
		{name: "null ttl", raw: json.RawMessage(`{"type":"ephemeral","ttl":null}`)},
		{name: "wrong ttl", raw: json.RawMessage(`{"type":"ephemeral","ttl":"2h"}`)},
		{name: "unknown field", raw: json.RawMessage(`{"type":"ephemeral","future":true}`)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for _, mode := range []canonical.Mode{canonical.ModeStrict, canonical.ModeCompatible, canonical.ModeEmulate} {
				t.Run(string(mode), func(t *testing.T) {
					request := canonical.Request{
						Turns:           []canonical.Turn{messageTurn(canonical.RoleUser, "hello")},
						MaxOutputTokens: intPointerValue(64),
						Extensions:      canonical.Object{"cache_control": test.raw},
					}
					options := cacheEncodeOptions()
					options.Mode = mode
					result := EncodeRequest(context.Background(), request, options)
					if result.OK || !hasDiagnostic(result.Diagnostics, canonical.DiagnosticInvalidCacheControl) {
						t.Fatalf("result = %#v", result)
					}
				})
			}
		})
	}
}

func TestEncodeRequestMapsImplicitChatPromptCacheToAnthropic(t *testing.T) {
	breakpoint := json.RawMessage(`{"mode":"explicit"}`)
	parts := make([]canonical.Part, 5)
	for index := range parts {
		parts[index] = canonical.Part{
			Kind:       canonical.PartText,
			Text:       fmt.Sprintf("stable-%d", index),
			Extensions: canonical.Object{"prompt_cache_breakpoint": breakpoint},
		}
	}
	request := canonical.Request{
		Turns:           []canonical.Turn{{Kind: canonical.TurnMessage, Role: canonical.RoleUser, Content: parts}},
		MaxOutputTokens: intPointerValue(64),
		Extensions: canonical.Object{
			"prompt_cache_options": json.RawMessage(`{"mode":"implicit","ttl":"30m"}`),
		},
	}

	result := EncodeRequest(context.Background(), request, cacheEncodeOptions())
	if !result.OK || result.Value == nil {
		t.Fatalf("result = %#v", result)
	}
	var encoded map[string]any
	if err := json.Unmarshal(*result.Value, &encoded); err != nil {
		t.Fatal(err)
	}
	assertCacheControl(t, encoded["cache_control"], "1h")
	content := encoded["messages"].([]any)[0].(map[string]any)["content"].([]any)
	for index, raw := range content {
		_, exists := raw.(map[string]any)["cache_control"]
		if exists != (index >= 2) {
			t.Fatalf("content[%d] cache_control exists = %v", index, exists)
		}
		if exists {
			assertCacheControl(t, raw.(map[string]any)["cache_control"], "1h")
		}
	}
	for _, name := range []string{"prompt_cache_options", "prompt_cache_key", "prompt_cache_retention"} {
		if _, exists := encoded[name]; exists {
			t.Fatalf("encoded standard field %q = %#v", name, encoded[name])
		}
	}
}

func TestEncodeRequestMapsExplicitChatPromptCacheToAnthropic(t *testing.T) {
	breakpoint := json.RawMessage(`{"mode":"explicit"}`)
	parts := make([]canonical.Part, 5)
	for index := range parts {
		parts[index] = canonical.Part{
			Kind:       canonical.PartText,
			Text:       fmt.Sprintf("stable-%d", index),
			Extensions: canonical.Object{"prompt_cache_breakpoint": breakpoint},
		}
	}
	request := canonical.Request{
		Turns:           []canonical.Turn{{Kind: canonical.TurnMessage, Role: canonical.RoleUser, Content: parts}},
		MaxOutputTokens: intPointerValue(64),
		Extensions: canonical.Object{
			"prompt_cache_options": json.RawMessage(`{"mode":"explicit"}`),
		},
	}

	options := cacheEncodeOptions()
	options.Mode = canonical.ModeStrict
	result := EncodeRequest(context.Background(), request, options)
	if !result.OK || result.Value == nil {
		t.Fatalf("result = %#v", result)
	}
	var encoded map[string]any
	if err := json.Unmarshal(*result.Value, &encoded); err != nil {
		t.Fatal(err)
	}
	if _, exists := encoded["cache_control"]; exists {
		t.Fatalf("automatic cache_control = %#v", encoded["cache_control"])
	}
	content := encoded["messages"].([]any)[0].(map[string]any)["content"].([]any)
	for index, raw := range content {
		_, exists := raw.(map[string]any)["cache_control"]
		if exists != (index >= 1) {
			t.Fatalf("content[%d] cache_control exists = %v", index, exists)
		}
	}
}

func TestEncodeRequestDiagnosesUnmappedChatCacheHints(t *testing.T) {
	for _, retention := range []string{"in_memory", "24h"} {
		t.Run(retention, func(t *testing.T) {
			request := canonical.Request{
				Turns:           []canonical.Turn{messageTurn(canonical.RoleUser, "hello")},
				MaxOutputTokens: intPointerValue(64),
				Extensions: canonical.Object{
					"prompt_cache_key":       json.RawMessage(`"stable"`),
					"prompt_cache_retention": json.RawMessage(`"` + retention + `"`),
				},
			}

			for _, mode := range []canonical.Mode{canonical.ModeCompatible, canonical.ModeEmulate} {
				options := cacheEncodeOptions()
				options.Mode = mode
				result := EncodeRequest(context.Background(), request, options)
				if !result.OK || result.Value == nil || !hasDiagnostic(result.Diagnostics, canonical.DiagnosticCacheControlProviderMismatch) {
					t.Fatalf("%s result = %#v", mode, result)
				}
				var encoded map[string]any
				if err := json.Unmarshal(*result.Value, &encoded); err != nil {
					t.Fatal(err)
				}
				if _, exists := encoded["cache_control"]; exists {
					t.Fatalf("cache_control = %#v", encoded["cache_control"])
				}
			}

			options := cacheEncodeOptions()
			options.Mode = canonical.ModeStrict
			result := EncodeRequest(context.Background(), request, options)
			if result.OK || !hasDiagnostic(result.Diagnostics, canonical.DiagnosticCacheControlProviderMismatch) {
				t.Fatalf("strict result = %#v", result)
			}
		})
	}
}

func TestEncodeRequestTreatsAllPromptCacheKeyStringsAsValidUnmappedHints(t *testing.T) {
	for _, raw := range []json.RawMessage{json.RawMessage(`""`), json.RawMessage(`"  "`)} {
		t.Run(string(raw), func(t *testing.T) {
			request := canonical.Request{
				Turns:           []canonical.Turn{messageTurn(canonical.RoleUser, "hello")},
				MaxOutputTokens: intPointerValue(64),
				Extensions: canonical.Object{
					"prompt_cache_key": raw,
				},
			}

			result := EncodeRequest(context.Background(), request, cacheEncodeOptions())
			if !result.OK || result.Value == nil || !hasDiagnostic(result.Diagnostics, canonical.DiagnosticCacheControlProviderMismatch) {
				t.Fatalf("result = %#v", result)
			}
			if hasDiagnostic(result.Diagnostics, canonical.DiagnosticInvalidCacheControl) {
				t.Fatalf("string key was treated as malformed: %#v", result.Diagnostics)
			}
		})
	}
}

func TestEncodeRequestDoesNotInjectCacheControlWithoutStandardSignal(t *testing.T) {
	request := canonical.Request{
		Turns:           []canonical.Turn{messageTurn(canonical.RoleUser, "hello")},
		MaxOutputTokens: intPointerValue(64),
	}
	result := EncodeRequest(context.Background(), request, cacheEncodeOptions())
	if !result.OK || result.Value == nil {
		t.Fatalf("result = %#v", result)
	}
	var encoded map[string]any
	if err := json.Unmarshal(*result.Value, &encoded); err != nil {
		t.Fatal(err)
	}
	if _, exists := encoded["cache_control"]; exists {
		t.Fatalf("cache_control = %#v", encoded["cache_control"])
	}
}

func TestEncodeRequestUsesImplicitModeForBreakpointWithoutOptions(t *testing.T) {
	request := canonical.Request{
		Turns: []canonical.Turn{
			{
				Kind: canonical.TurnMessage,
				Role: canonical.RoleSystem,
				Content: []canonical.Part{{
					Kind:       canonical.PartText,
					Text:       "stable",
					Extensions: canonical.Object{"prompt_cache_breakpoint": json.RawMessage(`{"mode":"explicit"}`)},
				}},
			},
			messageTurn(canonical.RoleUser, "dynamic"),
		},
		MaxOutputTokens: intPointerValue(64),
	}

	result := EncodeRequest(context.Background(), request, cacheEncodeOptions())
	if !result.OK || result.Value == nil {
		t.Fatalf("result = %#v", result)
	}
	var encoded map[string]any
	if err := json.Unmarshal(*result.Value, &encoded); err != nil {
		t.Fatal(err)
	}
	assertCacheControl(t, encoded["cache_control"], "1h")
	system := encoded["system"].([]any)[0].(map[string]any)
	assertCacheControl(t, system["cache_control"], "1h")
}

func TestEncodeRequestKeepsImplicitAutomaticControlWhenBreakpointBlockIsUnsupported(t *testing.T) {
	request := canonical.Request{
		Turns: []canonical.Turn{{
			Kind: canonical.TurnMessage,
			Role: canonical.RoleUser,
			Content: []canonical.Part{
				{Kind: canonical.PartText, Text: "stable"},
				{
					Kind:       canonical.PartAudio,
					Extensions: canonical.Object{"prompt_cache_breakpoint": json.RawMessage(`{"mode":"explicit"}`)},
				},
			},
		}},
		MaxOutputTokens: intPointerValue(64),
	}

	result := EncodeRequest(context.Background(), request, cacheEncodeOptions())
	if !result.OK || result.Value == nil || !hasDiagnostic(result.Diagnostics, canonical.DiagnosticCacheBreakpointUnsupported) {
		t.Fatalf("result = %#v", result)
	}
	var encoded map[string]any
	if err := json.Unmarshal(*result.Value, &encoded); err != nil {
		t.Fatal(err)
	}
	assertCacheControl(t, encoded["cache_control"], "1h")
}

func TestEncodeRequestGatesMappedChatPromptCacheByProfile(t *testing.T) {
	request := canonical.Request{
		Turns:           []canonical.Turn{messageTurn(canonical.RoleUser, "hello")},
		MaxOutputTokens: intPointerValue(64),
		Extensions: canonical.Object{
			"prompt_cache_options": json.RawMessage(`{"mode":"implicit"}`),
		},
	}

	options := testEncodeOptions()
	result := EncodeRequest(context.Background(), request, options)
	if !result.OK || result.Value == nil || !hasDiagnostic(result.Diagnostics, canonical.DiagnosticCacheControlUnsupported) {
		t.Fatalf("compatible result = %#v", result)
	}
	var encoded map[string]any
	if err := json.Unmarshal(*result.Value, &encoded); err != nil {
		t.Fatal(err)
	}
	if _, exists := encoded["cache_control"]; exists {
		t.Fatalf("cache_control = %#v", encoded["cache_control"])
	}

	options.Mode = canonical.ModeStrict
	result = EncodeRequest(context.Background(), request, options)
	if result.OK || !hasDiagnostic(result.Diagnostics, canonical.DiagnosticCacheControlUnsupported) {
		t.Fatalf("strict result = %#v", result)
	}
}

func TestEncodeRequestMapsAssistantAndToolMessageBreakpoints(t *testing.T) {
	breakpoint := json.RawMessage(`{"mode":"explicit"}`)
	request := canonical.Request{
		Turns: []canonical.Turn{
			messageTurn(canonical.RoleUser, "go"),
			{
				Kind: canonical.TurnMessage,
				Role: canonical.RoleAssistant,
				Content: []canonical.Part{{
					Kind:       canonical.PartText,
					Text:       "calling",
					Extensions: canonical.Object{"prompt_cache_breakpoint": breakpoint},
				}},
				ToolCalls: []canonical.ToolCall{{ID: "call_1", Name: "lookup", ArgumentsRaw: `{}`}},
			},
			{
				Kind: canonical.TurnToolResults,
				Results: []canonical.ToolResult{{
					CallID: "call_1",
					Content: []canonical.Part{{
						Kind:       canonical.PartText,
						Text:       "done",
						Extensions: canonical.Object{"prompt_cache_breakpoint": breakpoint},
					}},
				}},
			},
		},
		MaxOutputTokens: intPointerValue(64),
		Extensions: canonical.Object{
			"prompt_cache_options": json.RawMessage(`{"mode":"explicit"}`),
		},
	}

	result := EncodeRequest(context.Background(), request, cacheEncodeOptions())
	if !result.OK || result.Value == nil {
		t.Fatalf("result = %#v", result)
	}
	var encoded map[string]any
	if err := json.Unmarshal(*result.Value, &encoded); err != nil {
		t.Fatal(err)
	}
	messages := encoded["messages"].([]any)
	assistant := messages[1].(map[string]any)["content"].([]any)[0].(map[string]any)
	assertCacheControl(t, assistant["cache_control"], "1h")
	toolResult := messages[2].(map[string]any)["content"].([]any)[0].(map[string]any)
	assertCacheControl(t, toolResult["cache_control"], "1h")
	toolResultContent := toolResult["content"].([]any)[0].(map[string]any)
	if _, exists := toolResultContent["cache_control"]; exists {
		t.Fatalf("nested tool-result cache_control = %#v", toolResultContent["cache_control"])
	}
}

func TestEncodeRequestDiagnosesNonFinalToolMessageBreakpoint(t *testing.T) {
	breakpoint := json.RawMessage(`{"mode":"explicit"}`)
	request := canonical.Request{
		Turns: []canonical.Turn{
			messageTurn(canonical.RoleUser, "go"),
			{
				Kind:      canonical.TurnMessage,
				Role:      canonical.RoleAssistant,
				ToolCalls: []canonical.ToolCall{{ID: "call_1", Name: "lookup", ArgumentsRaw: `{}`}},
			},
			{
				Kind: canonical.TurnToolResults,
				Results: []canonical.ToolResult{{
					CallID: "call_1",
					Content: []canonical.Part{
						{Kind: canonical.PartText, Text: "stable", Extensions: canonical.Object{"prompt_cache_breakpoint": breakpoint}},
						{Kind: canonical.PartText, Text: "dynamic"},
					},
				}},
			},
		},
		MaxOutputTokens: intPointerValue(64),
	}

	options := cacheEncodeOptions()
	result := EncodeRequest(context.Background(), request, options)
	if !result.OK || result.Value == nil || !hasDiagnostic(result.Diagnostics, canonical.DiagnosticCacheBreakpointUnsupported) {
		t.Fatalf("compatible result = %#v", result)
	}
	var encoded map[string]any
	if err := json.Unmarshal(*result.Value, &encoded); err != nil {
		t.Fatal(err)
	}
	messages := encoded["messages"].([]any)
	toolResult := messages[2].(map[string]any)["content"].([]any)[0].(map[string]any)
	if _, exists := toolResult["cache_control"]; exists {
		t.Fatalf("tool result cache_control = %#v", toolResult["cache_control"])
	}

	options.Mode = canonical.ModeStrict
	result = EncodeRequest(context.Background(), request, options)
	if result.OK || !hasDiagnostic(result.Diagnostics, canonical.DiagnosticCacheBreakpointUnsupported) {
		t.Fatalf("strict result = %#v", result)
	}
}

func TestEncodeRequestRejectsMalformedOpenAICacheDirectives(t *testing.T) {
	tests := []struct {
		name       string
		directive  string
		raw        json.RawMessage
		breakpoint bool
	}{
		{name: "key null", directive: "prompt_cache_key", raw: json.RawMessage(`null`)},
		{name: "key wrong type", directive: "prompt_cache_key", raw: json.RawMessage(`42`)},
		{name: "retention null", directive: "prompt_cache_retention", raw: json.RawMessage(`null`)},
		{name: "retention wrong enum", directive: "prompt_cache_retention", raw: json.RawMessage(`"forever"`)},
		{name: "options null", directive: "prompt_cache_options", raw: json.RawMessage(`null`)},
		{name: "options unknown field", directive: "prompt_cache_options", raw: json.RawMessage(`{"future":true}`)},
		{name: "options null mode", directive: "prompt_cache_options", raw: json.RawMessage(`{"mode":null}`)},
		{name: "options wrong mode", directive: "prompt_cache_options", raw: json.RawMessage(`{"mode":"future"}`)},
		{name: "options wrong ttl", directive: "prompt_cache_options", raw: json.RawMessage(`{"ttl":"24h"}`)},
		{name: "breakpoint null", directive: "prompt_cache_breakpoint", raw: json.RawMessage(`null`), breakpoint: true},
		{name: "breakpoint missing mode", directive: "prompt_cache_breakpoint", raw: json.RawMessage(`{}`), breakpoint: true},
		{name: "breakpoint unknown field", directive: "prompt_cache_breakpoint", raw: json.RawMessage(`{"mode":"explicit","future":true}`), breakpoint: true},
		{name: "breakpoint null mode", directive: "prompt_cache_breakpoint", raw: json.RawMessage(`{"mode":null}`), breakpoint: true},
		{name: "breakpoint wrong mode", directive: "prompt_cache_breakpoint", raw: json.RawMessage(`{"mode":"implicit"}`), breakpoint: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for _, mode := range []canonical.Mode{canonical.ModeStrict, canonical.ModeCompatible, canonical.ModeEmulate} {
				t.Run(string(mode), func(t *testing.T) {
					request := canonical.Request{
						Turns:           []canonical.Turn{messageTurn(canonical.RoleUser, "hello")},
						MaxOutputTokens: intPointerValue(64),
					}
					if test.breakpoint {
						request.Turns[0].Content[0].Extensions = canonical.Object{test.directive: test.raw}
					} else {
						request.Extensions = canonical.Object{test.directive: test.raw}
					}
					options := cacheEncodeOptions()
					options.Mode = mode
					result := EncodeRequest(context.Background(), request, options)
					if result.OK || !hasDiagnostic(result.Diagnostics, canonical.DiagnosticInvalidCacheControl) {
						t.Fatalf("result = %#v", result)
					}
				})
			}
		})
	}
}

func TestEncodeRequestGatesAnthropicCacheControl(t *testing.T) {
	request := canonical.Request{
		Turns:           []canonical.Turn{messageTurn(canonical.RoleUser, "hello")},
		MaxOutputTokens: intPointerValue(64),
		Extensions:      canonical.Object{"cache_control": json.RawMessage(`{"type":"ephemeral"}`)},
	}

	options := testEncodeOptions()
	result := EncodeRequest(context.Background(), request, options)
	if !result.OK || result.Value == nil || !hasDiagnostic(result.Diagnostics, canonical.DiagnosticCacheControlUnsupported) {
		t.Fatalf("compatible result = %#v", result)
	}
	var encoded map[string]any
	if err := json.Unmarshal(*result.Value, &encoded); err != nil {
		t.Fatal(err)
	}
	if _, exists := encoded["cache_control"]; exists {
		t.Fatalf("encoded = %#v", encoded)
	}

	options.Mode = canonical.ModeStrict
	result = EncodeRequest(context.Background(), request, options)
	if result.OK || !hasDiagnostic(result.Diagnostics, canonical.DiagnosticCacheControlUnsupported) {
		t.Fatalf("strict result = %#v", result)
	}

	options = cacheEncodeOptions()
	options.Profile.Endpoint = capabilities.EndpointBedrockMessages
	result = EncodeRequest(context.Background(), request, options)
	if !result.OK || !hasDiagnostic(result.Diagnostics, canonical.DiagnosticCacheControlUnsupported) {
		t.Fatalf("Bedrock result = %#v", result)
	}
}

func TestEncodeRequestValidatesCacheBreakpointLimitsAndTTLOrder(t *testing.T) {
	fiveMinutes := json.RawMessage(`{"type":"ephemeral"}`)
	hour := json.RawMessage(`{"type":"ephemeral","ttl":"1h"}`)
	tests := []struct {
		name          string
		controls      []json.RawMessage
		automatic     json.RawMessage
		trailingPlain bool
		wantOK        bool
	}{
		{name: "four explicit", controls: []json.RawMessage{fiveMinutes, fiveMinutes, fiveMinutes, fiveMinutes}, wantOK: true},
		{name: "five explicit", controls: []json.RawMessage{fiveMinutes, fiveMinutes, fiveMinutes, fiveMinutes, fiveMinutes}},
		{name: "automatic plus three explicit", controls: []json.RawMessage{fiveMinutes, fiveMinutes, fiveMinutes}, automatic: fiveMinutes, trailingPlain: true, wantOK: true},
		{name: "automatic plus four explicit", controls: []json.RawMessage{fiveMinutes, fiveMinutes, fiveMinutes, fiveMinutes}, automatic: fiveMinutes, trailingPlain: true},
		{name: "automatic no op with four explicit", controls: []json.RawMessage{fiveMinutes, fiveMinutes, fiveMinutes, fiveMinutes}, automatic: fiveMinutes, wantOK: true},
		{name: "automatic TTL conflict", controls: []json.RawMessage{hour}, automatic: fiveMinutes},
		{name: "long before short", controls: []json.RawMessage{hour, fiveMinutes}, wantOK: true},
		{name: "long after short", controls: []json.RawMessage{fiveMinutes, hour}},
		{name: "automatic long after explicit short", controls: []json.RawMessage{fiveMinutes}, automatic: hour, trailingPlain: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			parts := make([]canonical.Part, 0, len(test.controls)+1)
			for index, control := range test.controls {
				parts = append(parts, canonical.Part{
					Kind:       canonical.PartText,
					Text:       "stable-" + string(rune('a'+index)),
					Extensions: canonical.Object{"cache_control": control},
				})
			}
			if test.trailingPlain {
				parts = append(parts, canonical.Part{Kind: canonical.PartText, Text: "latest"})
			}
			request := canonical.Request{
				Turns:           []canonical.Turn{{Kind: canonical.TurnMessage, Role: canonical.RoleUser, Content: parts}},
				MaxOutputTokens: intPointerValue(64),
			}
			if test.automatic != nil {
				request.Extensions = canonical.Object{"cache_control": test.automatic}
			}
			result := EncodeRequest(context.Background(), request, cacheEncodeOptions())
			if result.OK != test.wantOK {
				t.Fatalf("result = %#v", result)
			}
			if !test.wantOK && !hasDiagnostic(result.Diagnostics, canonical.DiagnosticInvalidCacheControl) {
				t.Fatalf("diagnostics = %#v", result.Diagnostics)
			}
		})
	}
}

func TestEncodeRequestRejectsUnsupportedCacheBreakpointLocations(t *testing.T) {
	control := json.RawMessage(`{"type":"ephemeral"}`)
	request := canonical.Request{
		Turns: []canonical.Turn{
			messageTurn(canonical.RoleUser, "go"),
			{
				Kind:      canonical.TurnMessage,
				Role:      canonical.RoleAssistant,
				ToolCalls: []canonical.ToolCall{{ID: "call_1", Name: "lookup", ArgumentsRaw: `{}`}},
			},
			{
				Kind: canonical.TurnToolResults,
				Results: []canonical.ToolResult{{
					CallID: "call_1",
					Content: []canonical.Part{{
						Kind:       canonical.PartText,
						Text:       "done",
						Extensions: canonical.Object{"cache_control": control},
					}},
				}},
			},
		},
		MaxOutputTokens: intPointerValue(64),
	}
	result := EncodeRequest(context.Background(), request, cacheEncodeOptions())
	if result.OK || !hasDiagnostic(result.Diagnostics, canonical.DiagnosticCacheBreakpointUnsupported) {
		t.Fatalf("tool result = %#v", result)
	}

	request = canonical.Request{
		Turns: []canonical.Turn{{
			Kind: canonical.TurnMessage,
			Role: canonical.RoleUser,
			Content: []canonical.Part{{
				Kind:       canonical.PartText,
				Text:       "",
				Extensions: canonical.Object{"cache_control": control},
			}},
		}},
		MaxOutputTokens: intPointerValue(64),
	}
	result = EncodeRequest(context.Background(), request, cacheEncodeOptions())
	if result.OK || !hasDiagnostic(result.Diagnostics, canonical.DiagnosticCacheBreakpointUnsupported) {
		t.Fatalf("empty text = %#v", result)
	}
}

func TestEncodeRequestRejectsCacheDirectivesAtCanonicalInvalidPositions(t *testing.T) {
	validBreakpoint := json.RawMessage(`{"mode":"explicit"}`)
	validControl := json.RawMessage(`{"type":"ephemeral"}`)
	baseRequest := func() canonical.Request {
		return canonical.Request{
			Turns:           []canonical.Turn{messageTurn(canonical.RoleUser, "hello")},
			MaxOutputTokens: intPointerValue(64),
		}
	}
	tests := []struct {
		name      string
		request   func() canonical.Request
		cacheMode capabilities.PromptCacheMode
		code      canonical.DiagnosticCode
	}{
		{
			name: "top-level OpenAI breakpoint",
			request: func() canonical.Request {
				request := baseRequest()
				request.Extensions = canonical.Object{"prompt_cache_breakpoint": validBreakpoint}
				return request
			},
			code: canonical.DiagnosticCacheBreakpointUnsupported,
		},
		{
			name: "part top-level directive",
			request: func() canonical.Request {
				request := baseRequest()
				request.Turns[0].Content[0].Extensions = canonical.Object{"prompt_cache_key": json.RawMessage(`"key"`)}
				return request
			},
			code: canonical.DiagnosticInvalidCacheControl,
		},
		{
			name: "tool breakpoint",
			request: func() canonical.Request {
				request := baseRequest()
				request.Tools = []canonical.ToolDefinition{{
					Name:        "lookup",
					InputSchema: canonical.Object{},
					Extensions:  canonical.Object{"prompt_cache_breakpoint": validBreakpoint},
				}}
				return request
			},
			code: canonical.DiagnosticCacheBreakpointUnsupported,
		},
		{
			name: "assistant image OpenAI breakpoint",
			request: func() canonical.Request {
				request := baseRequest()
				request.Turns = append(request.Turns, canonical.Turn{
					Kind: canonical.TurnMessage,
					Role: canonical.RoleAssistant,
					Content: []canonical.Part{{
						Kind: canonical.PartImage,
						Source: &canonical.AssetSource{
							Kind: canonical.AssetSourceURL,
							URL:  "https://example.com/image.png",
						},
						Extensions: canonical.Object{"prompt_cache_breakpoint": validBreakpoint},
					}},
				})
				return request
			},
			code: canonical.DiagnosticCacheBreakpointUnsupported,
		},
		{
			name: "tool-result cache control before capability gate",
			request: func() canonical.Request {
				request := baseRequest()
				request.Turns = append(request.Turns,
					canonical.Turn{
						Kind:      canonical.TurnMessage,
						Role:      canonical.RoleAssistant,
						ToolCalls: []canonical.ToolCall{{ID: "call", Name: "lookup", ArgumentsRaw: `{}`}},
					},
					canonical.Turn{
						Kind: canonical.TurnToolResults,
						Results: []canonical.ToolResult{{
							CallID: "call",
							Content: []canonical.Part{{
								Kind:       canonical.PartText,
								Text:       "done",
								Extensions: canonical.Object{"cache_control": validControl},
							}},
						}},
					},
				)
				return request
			},
			cacheMode: capabilities.PromptCacheNone,
			code:      canonical.DiagnosticCacheBreakpointUnsupported,
		},
		{
			name: "assistant image cache control before capability gate",
			request: func() canonical.Request {
				request := baseRequest()
				request.Turns = append(request.Turns, canonical.Turn{
					Kind: canonical.TurnMessage,
					Role: canonical.RoleAssistant,
					Content: []canonical.Part{{
						Kind: canonical.PartImage,
						Source: &canonical.AssetSource{
							Kind: canonical.AssetSourceURL,
							URL:  "https://example.com/image.png",
						},
						Extensions: canonical.Object{"cache_control": validControl},
					}},
				})
				return request
			},
			cacheMode: capabilities.PromptCacheNone,
			code:      canonical.DiagnosticCacheBreakpointUnsupported,
		},
		{
			name: "cache control in ignored tool-results content",
			request: func() canonical.Request {
				request := baseRequest()
				request.Turns = append(request.Turns,
					canonical.Turn{
						Kind:      canonical.TurnMessage,
						Role:      canonical.RoleAssistant,
						ToolCalls: []canonical.ToolCall{{ID: "call", Name: "lookup", ArgumentsRaw: `{}`}},
					},
					canonical.Turn{
						Kind: canonical.TurnToolResults,
						Content: []canonical.Part{{
							Kind:       canonical.PartText,
							Text:       "ignored",
							Extensions: canonical.Object{"cache_control": validControl},
						}},
						Results: []canonical.ToolResult{{CallID: "call", Content: textParts("done")}},
					},
				)
				return request
			},
			code: canonical.DiagnosticCacheBreakpointUnsupported,
		},
		{
			name: "OpenAI breakpoint in ignored message results",
			request: func() canonical.Request {
				request := baseRequest()
				request.Turns[0].Results = []canonical.ToolResult{{
					CallID: "ignored",
					Content: []canonical.Part{{
						Kind:       canonical.PartText,
						Text:       "ignored",
						Extensions: canonical.Object{"prompt_cache_breakpoint": validBreakpoint},
					}},
				}}
				return request
			},
			code: canonical.DiagnosticCacheBreakpointUnsupported,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for _, mode := range []canonical.Mode{canonical.ModeStrict, canonical.ModeCompatible, canonical.ModeEmulate} {
				t.Run(string(mode), func(t *testing.T) {
					options := cacheEncodeOptions()
					options.Mode = mode
					if test.cacheMode != capabilities.PromptCacheUnset {
						options.Profile.PromptCache.Mode = test.cacheMode
					}
					result := EncodeRequest(context.Background(), test.request(), options)
					if result.OK || !hasDiagnostic(result.Diagnostics, test.code) {
						t.Fatalf("result = %#v", result)
					}
				})
			}
		})
	}
}

func assertCacheControl(t *testing.T, value any, ttl string) {
	t.Helper()
	control, ok := value.(map[string]any)
	if !ok || control["type"] != "ephemeral" {
		t.Fatalf("cache control = %#v", value)
	}
	actualTTL, exists := control["ttl"]
	if ttl == "" && !exists {
		return
	}
	if !exists || actualTTL != ttl {
		t.Fatalf("cache control = %#v, want ttl %q", control, ttl)
	}
}

func cacheEncodeOptions() RequestEncodeOptions {
	options := testEncodeOptions()
	options.Profile.PromptCache.Mode = capabilities.PromptCacheAnthropic
	return options
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
