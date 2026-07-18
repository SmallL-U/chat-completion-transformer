package openairesponses

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"chat-completion-transformer/internal/canonical"
	"chat-completion-transformer/internal/capabilities"
)

func TestEncodeRequestMapsResponsesFields(t *testing.T) {
	schema := canonical.Object{
		"type":       json.RawMessage(`"object"`),
		"properties": json.RawMessage(`{"city":{"type":"string"}}`),
	}
	request := canonical.Request{
		ModelAlias: "fast",
		Turns: []canonical.Turn{
			{
				Kind:    canonical.TurnMessage,
				Role:    canonical.RoleSystem,
				Content: []canonical.Part{{Kind: canonical.PartText, Text: "be concise"}},
			},
			{
				Kind: canonical.TurnMessage,
				Role: canonical.RoleUser,
				Content: []canonical.Part{
					{Kind: canonical.PartText, Text: "what is here?"},
					{
						Kind: canonical.PartImage,
						Source: &canonical.AssetSource{
							Kind:      canonical.AssetSourceBase64,
							MediaType: "image/png",
							Data:      "aGVsbG8=",
						},
						Detail: pointer(canonical.ImageDetailLow),
					},
				},
			},
			{
				Kind:    canonical.TurnMessage,
				Role:    canonical.RoleAssistant,
				Content: []canonical.Part{{Kind: canonical.PartText, Text: "checking"}},
				ToolCalls: []canonical.ToolCall{{
					ID:           "call_1",
					Name:         "weather",
					ArgumentsRaw: `{"city":"Beijing"}`,
				}},
			},
			{
				Kind: canonical.TurnToolResults,
				Results: []canonical.ToolResult{{
					CallID:  "call_1",
					Content: []canonical.Part{{Kind: canonical.PartText, Text: "sunny"}},
				}},
			},
		},
		Tools: []canonical.ToolDefinition{{
			Name:        "weather",
			Description: pointer("look up weather"),
			InputSchema: schema,
		}},
		ToolChoice:        &canonical.ToolChoice{Mode: canonical.ToolChoiceNamed, Name: pointer("weather")},
		ParallelToolCalls: pointer(false),
		MaxOutputTokens:   pointer(512),
		Temperature:       pointer(0.2),
		TopP:              pointer(0.9),
		StopSequences:     []string{"END"},
		CandidateCount:    pointer(1),
		OutputFormat: &canonical.OutputFormat{
			Type:   canonical.OutputFormatJSONSchema,
			Name:   pointer("weather_response"),
			Schema: schema,
			Strict: pointer(false),
		},
		Stream:   true,
		Metadata: map[string]string{"tenant": "demo"},
	}

	result := EncodeRequest(context.Background(), request, EncodeOptions{
		TargetModel: "gpt-test",
		Mode:        canonical.ModeCompatible,
		Profile:     testResponsesProfile(),
	})
	if !result.OK || result.Value == nil {
		t.Fatalf("EncodeRequest() = %#v", result)
	}
	if len(result.Diagnostics) != 0 || !result.Lossless {
		t.Fatalf("diagnostics = %#v", result.Diagnostics)
	}

	value := *result.Value
	if value["model"] != "gpt-test" || value["stream"] != true {
		t.Fatalf("routing fields = %#v", value)
	}
	tools := value["tools"].([]any)
	tool := tools[0].(map[string]any)
	if strict, exists := tool["strict"]; !exists || strict != false {
		t.Fatalf("tool strict = %#v, want explicit false", strict)
	}
	if tool["parameters"] == nil {
		t.Fatal("tool parameters were omitted")
	}

	input := value["input"].([]any)
	if len(input) != 5 {
		t.Fatalf("input length = %d, want 5: %#v", len(input), input)
	}
	user := input[1].(map[string]any)
	content := user["content"].([]any)
	image := content[1].(map[string]any)
	if image["image_url"] != "data:image/png;base64,aGVsbG8=" || image["detail"] != "low" {
		t.Fatalf("image content = %#v", image)
	}
	call := input[3].(map[string]any)
	if call["type"] != "function_call" || call["call_id"] != "call_1" {
		t.Fatalf("function call = %#v", call)
	}
	output := input[4].(map[string]any)
	if output["type"] != "function_call_output" || output["output"] != "sunny" {
		t.Fatalf("function output = %#v", output)
	}
	choice := value["tool_choice"].(map[string]any)
	if choice["name"] != "weather" || value["stop"] == nil || value["metadata"] == nil {
		t.Fatalf("request options = %#v", value)
	}
	text := value["text"].(map[string]any)
	format := text["format"].(map[string]any)
	if format["type"] != "json_schema" || format["name"] != "weather_response" {
		t.Fatalf("text format = %#v", format)
	}
}

func TestEncodeRequestReportsLossesByMode(t *testing.T) {
	for _, test := range []struct {
		name   string
		mode   canonical.Mode
		wantOK bool
	}{
		{name: "compatible warns", mode: canonical.ModeCompatible, wantOK: true},
		{name: "strict fails", mode: canonical.ModeStrict, wantOK: false},
	} {
		t.Run(test.name, func(t *testing.T) {
			profile := testResponsesProfile()
			profile.Temperature = false
			profile.StopSequences = false
			profile.Metadata = false
			candidateCount := 2
			request := canonical.Request{
				ModelAlias:     "fast",
				Turns:          []canonical.Turn{{Kind: canonical.TurnMessage, Role: canonical.RoleUser, Content: []canonical.Part{{Kind: canonical.PartText, Text: "hi"}}}},
				Temperature:    pointer(0.5),
				StopSequences:  []string{"stop"},
				CandidateCount: &candidateCount,
				Metadata:       map[string]string{"key": "value"},
				Extensions:     canonical.Object{"seed": json.RawMessage(`7`)},
			}
			result := EncodeRequest(context.Background(), request, EncodeOptions{
				TargetModel: "gpt-test",
				Mode:        test.mode,
				Profile:     profile,
			})
			if result.OK != test.wantOK {
				t.Fatalf("OK = %v, want %v; diagnostics = %#v", result.OK, test.wantOK, result.Diagnostics)
			}
			assertDiagnosticCode(t, result.Diagnostics, canonical.DiagnosticSamplingParameterUnsupported)
			assertDiagnosticCode(t, result.Diagnostics, canonical.DiagnosticCandidateCountUnsupported)
			assertDiagnosticCode(t, result.Diagnostics, DiagnosticUnsupportedRequestField)
			assertDiagnosticCode(t, result.Diagnostics, DiagnosticUnsupportedExtension)
			if result.Value != nil {
				if _, exists := (*result.Value)["temperature"]; exists {
					t.Fatal("unsupported temperature was forwarded")
				}
				if _, exists := (*result.Value)["metadata"]; exists {
					t.Fatal("unsupported metadata was forwarded")
				}
			}
		})
	}
}

func TestEncodeRequestExtractLeadingInstructions(t *testing.T) {
	request := canonical.Request{
		Turns: []canonical.Turn{
			{Kind: canonical.TurnMessage, Role: canonical.RoleSystem, Content: []canonical.Part{{Kind: canonical.PartText, Text: "first"}}},
			{Kind: canonical.TurnMessage, Role: canonical.RoleSystem, Content: []canonical.Part{{Kind: canonical.PartText, Text: "second"}}},
			{Kind: canonical.TurnMessage, Role: canonical.RoleUser, Content: []canonical.Part{{Kind: canonical.PartText, Text: "hello"}}},
		},
	}
	result := EncodeRequest(context.Background(), request, EncodeOptions{
		TargetModel:       "gpt-test",
		Profile:           testResponsesProfile(),
		InstructionPolicy: InstructionPolicyExtractLeading,
	})
	if !result.OK || result.Value == nil {
		t.Fatalf("EncodeRequest() = %#v", result)
	}
	if (*result.Value)["instructions"] != "first\n\nsecond" {
		t.Fatalf("instructions = %#v", (*result.Value)["instructions"])
	}
	if got := len((*result.Value)["input"].([]any)); got != 1 {
		t.Fatalf("input length = %d, want 1", got)
	}
}

func TestEncodeRequestKeepsMultimodalFunctionOutput(t *testing.T) {
	request := canonical.Request{
		Turns: []canonical.Turn{
			{
				Kind: canonical.TurnMessage,
				Role: canonical.RoleAssistant,
				ToolCalls: []canonical.ToolCall{{
					ID:           "call_1",
					Name:         "inspect",
					ArgumentsRaw: `{}`,
				}},
			},
			{
				Kind: canonical.TurnToolResults,
				Results: []canonical.ToolResult{{
					CallID: "call_1",
					Content: []canonical.Part{
						{Kind: canonical.PartText, Text: "result"},
						{Kind: canonical.PartImage, Source: &canonical.AssetSource{Kind: canonical.AssetSourceURL, URL: "https://example.com/image.png"}},
						{Kind: canonical.PartFile, Source: &canonical.AssetSource{Kind: canonical.AssetSourceFileID, FileID: "file_1"}, Filename: pointer("report.pdf")},
					},
				}},
			},
		},
	}
	result := EncodeRequest(context.Background(), request, EncodeOptions{TargetModel: "gpt-test", Profile: testResponsesProfile()})
	if !result.OK || result.Value == nil {
		t.Fatalf("EncodeRequest() = %#v", result)
	}
	input := (*result.Value)["input"].([]any)
	output := input[1].(map[string]any)["output"].([]any)
	if len(output) != 3 {
		t.Fatalf("function output = %#v", output)
	}
	if output[0].(map[string]any)["type"] != "input_text" ||
		output[1].(map[string]any)["type"] != "input_image" ||
		output[2].(map[string]any)["type"] != "input_file" {
		t.Fatalf("function output = %#v", output)
	}
}

func TestEncodeRequestValidatesContextAndProfile(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result := EncodeRequest(ctx, canonical.Request{}, EncodeOptions{})
	if result.OK {
		t.Fatal("canceled request unexpectedly succeeded")
	}
	assertDiagnosticCode(t, result.Diagnostics, DiagnosticRequestCanceled)

	result = EncodeRequest(context.Background(), canonical.Request{}, EncodeOptions{
		TargetModel: "gpt-test",
		Profile: capabilities.Profile{
			Provider: capabilities.ProviderAnthropic,
			Endpoint: capabilities.EndpointMessages,
			Model:    "other",
		},
	})
	if result.OK {
		t.Fatal("mismatched profile unexpectedly succeeded")
	}
	assertDiagnosticCode(t, result.Diagnostics, DiagnosticInvalidEncodeOptions)
}

func TestEncodeRequestGatesEveryFileSourceCapability(t *testing.T) {
	tests := []struct {
		name      string
		source    canonical.AssetSource
		files     capabilities.ImageCapabilities
		wantField string
		wantValue string
	}{
		{name: "url", source: canonical.AssetSource{Kind: canonical.AssetSourceURL, URL: "https://example.com/report.pdf"}, files: capabilities.ImageCapabilities{URL: true}, wantField: "file_url", wantValue: "https://example.com/report.pdf"},
		{name: "base64", source: canonical.AssetSource{Kind: canonical.AssetSourceBase64, MediaType: "application/pdf", Data: "eA=="}, files: capabilities.ImageCapabilities{Base64: true}, wantField: "file_data", wantValue: "data:application/pdf;base64,eA=="},
		{name: "file id", source: canonical.AssetSource{Kind: canonical.AssetSourceFileID, FileID: "file_1"}, files: capabilities.ImageCapabilities{FileID: true}, wantField: "file_id", wantValue: "file_1"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := canonical.Request{Turns: []canonical.Turn{{
				Kind:    canonical.TurnMessage,
				Role:    canonical.RoleUser,
				Content: []canonical.Part{{Kind: canonical.PartFile, Source: &test.source}},
			}}}
			profile := testResponsesProfile()
			profile.Files = capabilities.ImageCapabilities{}
			unsupported := EncodeRequest(context.Background(), request, EncodeOptions{TargetModel: "gpt-test", Profile: profile})
			if !unsupported.OK || unsupported.Value == nil {
				t.Fatalf("unsupported compatible transform = %#v", unsupported)
			}
			assertDiagnosticCode(t, unsupported.Diagnostics, canonical.DiagnosticUnsupportedContentPart)
			if got := len((*unsupported.Value)["input"].([]any)); got != 0 {
				t.Fatalf("unsupported file source produced %d input items", got)
			}

			profile.Files = test.files
			supported := EncodeRequest(context.Background(), request, EncodeOptions{TargetModel: "gpt-test", Profile: profile})
			if !supported.OK || supported.Value == nil {
				t.Fatalf("supported transform = %#v", supported)
			}
			input := (*supported.Value)["input"].([]any)
			part := input[0].(map[string]any)["content"].([]any)[0].(map[string]any)
			if part[test.wantField] != test.wantValue {
				t.Fatalf("file part = %#v", part)
			}
		})
	}
}

func TestEncodeRequestRejectsInvalidCanonicalTools(t *testing.T) {
	objectSchema := canonical.Object{"type": json.RawMessage(`"object"`)}
	tests := []struct {
		name     string
		tools    []canonical.ToolDefinition
		wantCode canonical.DiagnosticCode
	}{
		{name: "empty name", tools: []canonical.ToolDefinition{{Name: "", InputSchema: objectSchema}}, wantCode: DiagnosticUnsupportedRequestField},
		{name: "duplicate name", tools: []canonical.ToolDefinition{{Name: "lookup", InputSchema: objectSchema}, {Name: "lookup", InputSchema: objectSchema}}, wantCode: DiagnosticUnsupportedRequestField},
		{name: "nil schema", tools: []canonical.ToolDefinition{{Name: "lookup"}}, wantCode: DiagnosticUnsupportedSchema},
		{name: "non object schema", tools: []canonical.ToolDefinition{{Name: "lookup", InputSchema: canonical.Object{"type": json.RawMessage(`"string"`)}}}, wantCode: DiagnosticUnsupportedSchema},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := canonical.Request{
				Turns: []canonical.Turn{{Kind: canonical.TurnMessage, Role: canonical.RoleUser, Content: []canonical.Part{{Kind: canonical.PartText, Text: "hi"}}}},
				Tools: test.tools,
			}
			result := EncodeRequest(context.Background(), request, EncodeOptions{TargetModel: "gpt-test", Profile: testResponsesProfile()})
			if result.OK || result.Value != nil {
				t.Fatalf("invalid tools unexpectedly succeeded: %#v", result)
			}
			assertDiagnosticCode(t, result.Diagnostics, test.wantCode)
		})
	}
}

func TestEncodeRequestRejectsEmptyCanonicalMessages(t *testing.T) {
	tests := []struct {
		name  string
		turns []canonical.Turn
	}{
		{name: "no turns"},
		{name: "system", turns: []canonical.Turn{{Kind: canonical.TurnMessage, Role: canonical.RoleSystem}}},
		{name: "developer", turns: []canonical.Turn{{Kind: canonical.TurnMessage, Role: canonical.RoleDeveloper}}},
		{name: "user", turns: []canonical.Turn{{Kind: canonical.TurnMessage, Role: canonical.RoleUser}}},
		{name: "assistant", turns: []canonical.Turn{{Kind: canonical.TurnMessage, Role: canonical.RoleAssistant}}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := EncodeRequest(context.Background(), canonical.Request{Turns: test.turns}, EncodeOptions{TargetModel: "gpt-test", Profile: testResponsesProfile()})
			if result.OK || result.Value != nil {
				t.Fatalf("empty canonical message unexpectedly succeeded: %#v", result)
			}
			assertDiagnosticCode(t, result.Diagnostics, DiagnosticUnsupportedRequestField)
		})
	}
}

func TestEncodeRequestMapsPromptCacheRetentionByCapability(t *testing.T) {
	for _, cacheMode := range []capabilities.PromptCacheMode{
		capabilities.PromptCacheOpenAILegacy,
		capabilities.PromptCacheOpenAI56,
	} {
		for _, retention := range []string{"in_memory", "24h"} {
			t.Run(string(cacheMode)+"/"+retention, func(t *testing.T) {
				request := canonical.Request{
					Turns: []canonical.Turn{{
						Kind:    canonical.TurnMessage,
						Role:    canonical.RoleUser,
						Content: []canonical.Part{{Kind: canonical.PartText, Text: "hello"}},
					}},
					Extensions: canonical.Object{
						"prompt_cache_key":       json.RawMessage(`"tenant:demo"`),
						"prompt_cache_retention": json.RawMessage(`"` + retention + `"`),
					},
				}
				profile := testResponsesProfile()
				profile.PromptCache = capabilities.PromptCacheCapabilities{
					Mode:                 cacheMode,
					InMemoryRetention:    true,
					ExtendedRetention24h: true,
				}
				result := EncodeRequest(context.Background(), request, EncodeOptions{TargetModel: "gpt-test", Profile: profile})
				if !result.OK || result.Value == nil || len(result.Diagnostics) != 0 {
					t.Fatalf("EncodeRequest() = %#v", result)
				}
				if (*result.Value)["prompt_cache_key"] != "tenant:demo" || (*result.Value)["prompt_cache_retention"] != retention {
					t.Fatalf("cache fields = %#v", *result.Value)
				}
			})
		}
	}
}

func TestEncodeRequestPreservesPromptCacheKeyStrings(t *testing.T) {
	for _, key := range []string{"", " ", "tenant:demo"} {
		t.Run(fmt.Sprintf("%q", key), func(t *testing.T) {
			encodedKey, err := json.Marshal(key)
			if err != nil {
				t.Fatal(err)
			}
			request := canonical.Request{
				Turns: []canonical.Turn{{
					Kind:    canonical.TurnMessage,
					Role:    canonical.RoleUser,
					Content: []canonical.Part{{Kind: canonical.PartText, Text: "hello"}},
				}},
				Extensions: canonical.Object{"prompt_cache_key": encodedKey},
			}
			profile := testResponsesProfile()
			profile.PromptCache.Mode = capabilities.PromptCacheOpenAI56
			result := EncodeRequest(context.Background(), request, EncodeOptions{TargetModel: "gpt-test", Profile: profile})
			if !result.OK || result.Value == nil || len(result.Diagnostics) != 0 {
				t.Fatalf("EncodeRequest() = %#v", result)
			}
			if (*result.Value)["prompt_cache_key"] != key {
				t.Fatalf("prompt_cache_key = %#v, want %q", (*result.Value)["prompt_cache_key"], key)
			}
		})
	}
}

func TestEncodeRequestMapsOpenAI56CacheOptionsAndBreakpoints(t *testing.T) {
	breakpoint := canonical.Object{"prompt_cache_breakpoint": json.RawMessage(`{"mode":"explicit"}`)}
	request := canonical.Request{
		Turns: []canonical.Turn{
			{
				Kind: canonical.TurnMessage,
				Role: canonical.RoleSystem,
				Content: []canonical.Part{{
					Kind:       canonical.PartText,
					Text:       "stable instructions",
					Extensions: breakpoint,
				}},
			},
			{
				Kind: canonical.TurnMessage,
				Role: canonical.RoleUser,
				Content: []canonical.Part{
					{
						Kind:       canonical.PartImage,
						Source:     &canonical.AssetSource{Kind: canonical.AssetSourceURL, URL: "https://example.com/image.png"},
						Extensions: breakpoint,
					},
					{
						Kind:       canonical.PartFile,
						Source:     &canonical.AssetSource{Kind: canonical.AssetSourceFileID, FileID: "file_1"},
						Extensions: breakpoint,
					},
				},
			},
		},
		Extensions: canonical.Object{
			"prompt_cache_key":     json.RawMessage(`"tenant:demo"`),
			"prompt_cache_options": json.RawMessage(`{"mode":"explicit","ttl":"30m"}`),
		},
	}
	profile := testResponsesProfile()
	profile.PromptCache.Mode = capabilities.PromptCacheOpenAI56
	result := EncodeRequest(context.Background(), request, EncodeOptions{
		TargetModel:       "gpt-test",
		Profile:           profile,
		InstructionPolicy: InstructionPolicyExtractLeading,
	})
	if !result.OK || result.Value == nil || len(result.Diagnostics) != 0 {
		t.Fatalf("EncodeRequest() = %#v", result)
	}
	value := *result.Value
	if _, exists := value["instructions"]; exists {
		t.Fatalf("breakpoint-bearing instructions were collapsed: %#v", value)
	}
	options := value["prompt_cache_options"].(map[string]any)
	if options["mode"] != "explicit" || options["ttl"] != "30m" {
		t.Fatalf("prompt_cache_options = %#v", options)
	}
	input := value["input"].([]any)
	if len(input) != 2 {
		t.Fatalf("input = %#v", input)
	}
	systemContent := input[0].(map[string]any)["content"].([]any)
	assertExplicitBreakpoint(t, systemContent[0].(map[string]any))
	userContent := input[1].(map[string]any)["content"].([]any)
	assertExplicitBreakpoint(t, userContent[0].(map[string]any))
	assertExplicitBreakpoint(t, userContent[1].(map[string]any))
}

func TestEncodeRequestPreservesPlainTextFastPath(t *testing.T) {
	request := canonical.Request{Turns: []canonical.Turn{{
		Kind: canonical.TurnMessage,
		Role: canonical.RoleUser,
		Content: []canonical.Part{
			{Kind: canonical.PartText, Text: "hello "},
			{Kind: canonical.PartText, Text: "world"},
		},
	}}}
	result := EncodeRequest(context.Background(), request, EncodeOptions{TargetModel: "gpt-test", Profile: testResponsesProfile()})
	if !result.OK || result.Value == nil {
		t.Fatalf("EncodeRequest() = %#v", result)
	}
	content := (*result.Value)["input"].([]any)[0].(map[string]any)["content"]
	if content != "hello world" {
		t.Fatalf("content = %#v, want text string fast path", content)
	}
}

func TestEncodeRequestRejectsMalformedPromptCacheDirectives(t *testing.T) {
	tests := []struct {
		name       string
		cacheMode  capabilities.PromptCacheMode
		extensions canonical.Object
		part       canonical.Part
	}{
		{
			name:       "key null",
			cacheMode:  capabilities.PromptCacheOpenAILegacy,
			extensions: canonical.Object{"prompt_cache_key": json.RawMessage(`null`)},
		},
		{
			name:       "key type",
			cacheMode:  capabilities.PromptCacheOpenAILegacy,
			extensions: canonical.Object{"prompt_cache_key": json.RawMessage(`42`)},
		},
		{
			name:       "legacy spelling",
			cacheMode:  capabilities.PromptCacheOpenAILegacy,
			extensions: canonical.Object{"prompt_cache_retention": json.RawMessage(`"in-memory"`)},
		},
		{
			name:       "options null",
			cacheMode:  capabilities.PromptCacheOpenAI56,
			extensions: canonical.Object{"prompt_cache_options": json.RawMessage(`null`)},
		},
		{
			name:       "options future field",
			cacheMode:  capabilities.PromptCacheOpenAI56,
			extensions: canonical.Object{"prompt_cache_options": json.RawMessage(`{"mode":"implicit","future":true}`)},
		},
		{
			name:       "options mode",
			cacheMode:  capabilities.PromptCacheOpenAI56,
			extensions: canonical.Object{"prompt_cache_options": json.RawMessage(`{"mode":"future"}`)},
		},
		{
			name:       "options ttl",
			cacheMode:  capabilities.PromptCacheOpenAI56,
			extensions: canonical.Object{"prompt_cache_options": json.RawMessage(`{"ttl":"24h"}`)},
		},
		{
			name:      "breakpoint mode",
			cacheMode: capabilities.PromptCacheOpenAI56,
			part: canonical.Part{
				Kind:       canonical.PartText,
				Text:       "hello",
				Extensions: canonical.Object{"prompt_cache_breakpoint": json.RawMessage(`{"mode":"implicit"}`)},
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			part := test.part
			if part.Kind == "" {
				part = canonical.Part{Kind: canonical.PartText, Text: "hello"}
			}
			request := canonical.Request{
				Turns:      []canonical.Turn{{Kind: canonical.TurnMessage, Role: canonical.RoleUser, Content: []canonical.Part{part}}},
				Extensions: test.extensions,
			}
			profile := testResponsesProfile()
			profile.PromptCache = capabilities.PromptCacheCapabilities{
				Mode:                 test.cacheMode,
				InMemoryRetention:    test.cacheMode == capabilities.PromptCacheOpenAILegacy,
				ExtendedRetention24h: test.cacheMode == capabilities.PromptCacheOpenAILegacy,
			}
			result := EncodeRequest(context.Background(), request, EncodeOptions{TargetModel: "gpt-test", Profile: profile})
			if result.OK || result.Value != nil {
				t.Fatalf("malformed cache directive unexpectedly succeeded: %#v", result)
			}
			assertDiagnosticCode(t, result.Diagnostics, canonical.DiagnosticInvalidCacheControl)
		})
	}
}

func TestEncodeRequestGatesCacheFieldsByProfileAndMode(t *testing.T) {
	tests := []struct {
		name      string
		mode      canonical.Mode
		cacheMode capabilities.PromptCacheMode
		inMemory  bool
		extended  bool
		field     string
		raw       json.RawMessage
		wantOK    bool
		wantCode  canonical.DiagnosticCode
	}{
		{name: "legacy rejects options compatibly", mode: canonical.ModeCompatible, cacheMode: capabilities.PromptCacheOpenAILegacy, field: "prompt_cache_options", raw: json.RawMessage(`{"mode":"implicit"}`), wantOK: true, wantCode: canonical.DiagnosticCacheControlUnsupported},
		{name: "legacy rejects options strictly", mode: canonical.ModeStrict, cacheMode: capabilities.PromptCacheOpenAILegacy, field: "prompt_cache_options", raw: json.RawMessage(`{"mode":"implicit"}`), wantOK: false, wantCode: canonical.DiagnosticCacheControlUnsupported},
		{name: "legacy profile gates in-memory retention", mode: canonical.ModeCompatible, cacheMode: capabilities.PromptCacheOpenAILegacy, extended: true, field: "prompt_cache_retention", raw: json.RawMessage(`"in_memory"`), wantOK: true, wantCode: canonical.DiagnosticCacheControlUnsupported},
		{name: "5.6 profile gates 24h retention", mode: canonical.ModeCompatible, cacheMode: capabilities.PromptCacheOpenAI56, inMemory: true, field: "prompt_cache_retention", raw: json.RawMessage(`"24h"`), wantOK: true, wantCode: canonical.DiagnosticCacheControlUnsupported},
		{name: "none rejects key compatibly", mode: canonical.ModeCompatible, cacheMode: capabilities.PromptCacheNone, field: "prompt_cache_key", raw: json.RawMessage(`"tenant"`), wantOK: true, wantCode: canonical.DiagnosticCacheControlUnsupported},
		{name: "wrong provider cache control", mode: canonical.ModeCompatible, cacheMode: capabilities.PromptCacheOpenAI56, field: "cache_control", raw: json.RawMessage(`{"type":"ephemeral"}`), wantOK: true, wantCode: canonical.DiagnosticCacheControlProviderMismatch},
		{name: "wrong provider cache control strictly", mode: canonical.ModeStrict, cacheMode: capabilities.PromptCacheOpenAI56, field: "cache_control", raw: json.RawMessage(`{"type":"ephemeral"}`), wantOK: false, wantCode: canonical.DiagnosticCacheControlProviderMismatch},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := canonical.Request{
				Turns:      []canonical.Turn{{Kind: canonical.TurnMessage, Role: canonical.RoleUser, Content: []canonical.Part{{Kind: canonical.PartText, Text: "hello"}}}},
				Extensions: canonical.Object{test.field: test.raw},
			}
			profile := testResponsesProfile()
			profile.PromptCache = capabilities.PromptCacheCapabilities{
				Mode:                 test.cacheMode,
				InMemoryRetention:    test.inMemory,
				ExtendedRetention24h: test.extended,
			}
			result := EncodeRequest(context.Background(), request, EncodeOptions{TargetModel: "gpt-test", Mode: test.mode, Profile: profile})
			if result.OK != test.wantOK {
				t.Fatalf("OK = %v, want %v: %#v", result.OK, test.wantOK, result)
			}
			assertDiagnosticCode(t, result.Diagnostics, test.wantCode)
			if result.Value != nil {
				if _, exists := (*result.Value)[test.field]; exists {
					t.Fatalf("unsupported cache field was forwarded: %#v", *result.Value)
				}
			}
		})
	}
}

func TestEncodeRequestGatesPromptCacheBreakpointByProfile(t *testing.T) {
	for _, test := range []struct {
		name   string
		mode   canonical.Mode
		wantOK bool
	}{
		{name: "compatible drops", mode: canonical.ModeCompatible, wantOK: true},
		{name: "strict rejects", mode: canonical.ModeStrict, wantOK: false},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := canonical.Request{Turns: []canonical.Turn{{
				Kind: canonical.TurnMessage,
				Role: canonical.RoleUser,
				Content: []canonical.Part{{
					Kind:       canonical.PartText,
					Text:       "hello",
					Extensions: canonical.Object{"prompt_cache_breakpoint": json.RawMessage(`{"mode":"explicit"}`)},
				}},
			}}}
			profile := testResponsesProfile()
			profile.PromptCache.Mode = capabilities.PromptCacheOpenAILegacy
			result := EncodeRequest(context.Background(), request, EncodeOptions{TargetModel: "gpt-test", Mode: test.mode, Profile: profile})
			if result.OK != test.wantOK {
				t.Fatalf("OK = %v, want %v: %#v", result.OK, test.wantOK, result)
			}
			assertDiagnosticCode(t, result.Diagnostics, canonical.DiagnosticCacheBreakpointUnsupported)
			if result.Value != nil {
				part := (*result.Value)["input"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
				if _, exists := part["prompt_cache_breakpoint"]; exists {
					t.Fatalf("unsupported breakpoint was forwarded: %#v", part)
				}
			}
		})
	}
}

func TestEncodeRequestRejectsMalformedCrossProviderCacheControl(t *testing.T) {
	invalidValues := []struct {
		name string
		raw  json.RawMessage
	}{
		{name: "null", raw: json.RawMessage(`null`)},
		{name: "invalid type", raw: json.RawMessage(`{"type":"persistent"}`)},
		{name: "invalid ttl", raw: json.RawMessage(`{"type":"ephemeral","ttl":"30m"}`)},
		{name: "unknown field", raw: json.RawMessage(`{"type":"ephemeral","future":true}`)},
	}
	for _, scope := range []string{"request", "part", "tool"} {
		for _, invalid := range invalidValues {
			t.Run(scope+"/"+invalid.name, func(t *testing.T) {
				request := canonical.Request{Turns: []canonical.Turn{{
					Kind:    canonical.TurnMessage,
					Role:    canonical.RoleUser,
					Content: []canonical.Part{{Kind: canonical.PartText, Text: "hello"}},
				}}}
				switch scope {
				case "request":
					request.Extensions = canonical.Object{"cache_control": invalid.raw}
				case "part":
					request.Turns[0].Content[0].Extensions = canonical.Object{"cache_control": invalid.raw}
				case "tool":
					request.Tools = []canonical.ToolDefinition{{
						Name:        "lookup",
						InputSchema: canonical.Object{"type": json.RawMessage(`"object"`)},
						Extensions:  canonical.Object{"cache_control": invalid.raw},
					}}
				}

				result := EncodeRequest(context.Background(), request, EncodeOptions{TargetModel: "gpt-test", Mode: canonical.ModeCompatible, Profile: testResponsesProfile()})
				if result.OK || result.Value != nil {
					t.Fatalf("malformed cross-provider cache_control unexpectedly succeeded: %#v", result)
				}
				assertDiagnosticCode(t, result.Diagnostics, canonical.DiagnosticInvalidCacheControl)
			})
		}
	}
}

func TestEncodeRequestMapsAssistantTextPromptCacheBreakpoint(t *testing.T) {
	breakpoint := canonical.Object{"prompt_cache_breakpoint": json.RawMessage(`{"mode":"explicit"}`)}
	request := canonical.Request{Turns: []canonical.Turn{
		{Kind: canonical.TurnMessage, Role: canonical.RoleUser, Content: []canonical.Part{{Kind: canonical.PartText, Text: "hello"}}},
		{Kind: canonical.TurnMessage, Role: canonical.RoleAssistant, Content: []canonical.Part{
			{Kind: canonical.PartText, Text: "stable answer"},
			{Kind: canonical.PartText, Text: "cache here", Extensions: breakpoint},
		}},
	}}
	profile := testResponsesProfile()
	profile.PromptCache.Mode = capabilities.PromptCacheOpenAI56
	result := EncodeRequest(context.Background(), request, EncodeOptions{TargetModel: "gpt-test", Profile: profile})
	if !result.OK || result.Value == nil || len(result.Diagnostics) != 0 {
		t.Fatalf("EncodeRequest() = %#v", result)
	}

	input := (*result.Value)["input"].([]any)
	message := input[1].(map[string]any)
	content := message["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("assistant content = %#v", content)
	}
	for index, part := range content {
		if part.(map[string]any)["type"] != "input_text" {
			t.Fatalf("assistant content[%d] = %#v", index, part)
		}
	}
	assertExplicitBreakpoint(t, content[1].(map[string]any))
}

func TestEncodeRequestDiagnosesAudioBreakpointUnsupportedByResponses(t *testing.T) {
	breakpoint := canonical.Object{"prompt_cache_breakpoint": json.RawMessage(`{"mode":"explicit"}`)}
	request := canonical.Request{Turns: []canonical.Turn{{
		Kind: canonical.TurnMessage,
		Role: canonical.RoleUser,
		Content: []canonical.Part{
			{Kind: canonical.PartText, Text: "hello"},
			{Kind: canonical.PartAudio, Extensions: breakpoint},
		},
	}}}
	for _, mode := range []canonical.Mode{canonical.ModeStrict, canonical.ModeCompatible, canonical.ModeEmulate} {
		t.Run(string(mode), func(t *testing.T) {
			profile := testResponsesProfile()
			profile.PromptCache.Mode = capabilities.PromptCacheOpenAI56
			result := EncodeRequest(context.Background(), request, EncodeOptions{TargetModel: "gpt-test", Mode: mode, Profile: profile})
			if result.OK != (mode != canonical.ModeStrict) {
				t.Fatalf("EncodeRequest() = %#v", result)
			}
			assertDiagnosticCode(t, result.Diagnostics, canonical.DiagnosticCacheBreakpointUnsupported)
			if result.Value == nil {
				return
			}
			encoded, err := json.Marshal(result.Value)
			if err != nil {
				t.Fatal(err)
			}
			if strings.Contains(string(encoded), "prompt_cache_breakpoint") {
				t.Fatalf("unsupported breakpoint was forwarded: %s", encoded)
			}
		})
	}
}

func TestEncodeRequestMapsToolResultPromptCacheBreakpoint(t *testing.T) {
	breakpoint := canonical.Object{"prompt_cache_breakpoint": json.RawMessage(`{"mode":"explicit"}`)}
	request := canonical.Request{Turns: []canonical.Turn{
		{Kind: canonical.TurnMessage, Role: canonical.RoleUser, Content: []canonical.Part{{Kind: canonical.PartText, Text: "hello"}}},
		{
			Kind: canonical.TurnMessage,
			Role: canonical.RoleAssistant,
			ToolCalls: []canonical.ToolCall{{
				ID:           "call_1",
				Name:         "lookup",
				ArgumentsRaw: `{}`,
			}},
		},
		{
			Kind: canonical.TurnToolResults,
			Results: []canonical.ToolResult{{
				CallID:  "call_1",
				Content: []canonical.Part{{Kind: canonical.PartText, Text: "result", Extensions: breakpoint}},
			}},
		},
	}}
	profile := testResponsesProfile()
	profile.PromptCache.Mode = capabilities.PromptCacheOpenAI56
	result := EncodeRequest(context.Background(), request, EncodeOptions{TargetModel: "gpt-test", Profile: profile})
	if !result.OK || result.Value == nil || len(result.Diagnostics) != 0 {
		t.Fatalf("EncodeRequest() = %#v", result)
	}

	input := (*result.Value)["input"].([]any)
	output := input[len(input)-1].(map[string]any)["output"].([]any)
	assertExplicitBreakpoint(t, output[0].(map[string]any))
}

func TestEncodeRequestRejectsBreakpointOutsideChatSchema(t *testing.T) {
	breakpoint := canonical.Object{"prompt_cache_breakpoint": json.RawMessage(`{"mode":"explicit"}`)}
	request := canonical.Request{Turns: []canonical.Turn{
		{Kind: canonical.TurnMessage, Role: canonical.RoleUser, Content: []canonical.Part{{Kind: canonical.PartText, Text: "hello"}}},
		{Kind: canonical.TurnMessage, Role: canonical.RoleAssistant, Content: []canonical.Part{{Kind: canonical.PartRefusal, Text: "no", Extensions: breakpoint}}},
	}}
	for _, mode := range []canonical.Mode{canonical.ModeStrict, canonical.ModeCompatible, canonical.ModeEmulate} {
		t.Run(string(mode), func(t *testing.T) {
			profile := testResponsesProfile()
			profile.PromptCache.Mode = capabilities.PromptCacheOpenAI56
			result := EncodeRequest(context.Background(), request, EncodeOptions{TargetModel: "gpt-test", Mode: mode, Profile: profile})
			if result.OK || result.Value != nil {
				t.Fatalf("invalid Chat breakpoint position unexpectedly succeeded: %#v", result)
			}
			assertDiagnosticCode(t, result.Diagnostics, canonical.DiagnosticCacheBreakpointUnsupported)
		})
	}
}

func TestEncodeRequestRejectsCanonicalCacheDirectivePositionLoss(t *testing.T) {
	breakpoint := json.RawMessage(`{"mode":"explicit"}`)
	cacheControl := json.RawMessage(`{"type":"ephemeral"}`)
	baseRequest := func() canonical.Request {
		return canonical.Request{Turns: []canonical.Turn{{
			Kind:    canonical.TurnMessage,
			Role:    canonical.RoleUser,
			Content: []canonical.Part{{Kind: canonical.PartText, Text: "hello"}},
		}}}
	}
	tests := []struct {
		name      string
		request   func() canonical.Request
		profile   func() capabilities.Profile
		code      canonical.DiagnosticCode
		path      string
		modeAware bool
	}{
		{
			name: "breakpoint on omitted content",
			request: func() canonical.Request {
				request := baseRequest()
				request.Turns[0].Content[0].Extensions = canonical.Object{"prompt_cache_breakpoint": breakpoint}
				return request
			},
			profile: func() capabilities.Profile {
				profile := testResponsesProfile()
				profile.Content.Text = false
				profile.PromptCache.Mode = capabilities.PromptCacheOpenAI56
				return profile
			},
			code:      canonical.DiagnosticCacheBreakpointUnsupported,
			path:      "turns.0.content.0.extensions.prompt_cache_breakpoint",
			modeAware: true,
		},
		{
			name: "top-level directive on part",
			request: func() canonical.Request {
				request := baseRequest()
				request.Turns[0].Content[0].Extensions = canonical.Object{"prompt_cache_key": json.RawMessage(`"key"`)}
				return request
			},
			profile: testResponsesProfile,
			code:    canonical.DiagnosticInvalidCacheControl,
		},
		{
			name: "breakpoint on tool definition",
			request: func() canonical.Request {
				request := baseRequest()
				request.Tools = []canonical.ToolDefinition{{
					Name:        "lookup",
					InputSchema: canonical.Object{},
					Extensions:  canonical.Object{"prompt_cache_breakpoint": breakpoint},
				}}
				return request
			},
			profile: testResponsesProfile,
			code:    canonical.DiagnosticCacheBreakpointUnsupported,
		},
		{
			name: "Anthropic cache control in tool result content",
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
								Text:       "result",
								Extensions: canonical.Object{"cache_control": cacheControl},
							}},
						}},
					},
				)
				return request
			},
			profile: testResponsesProfile,
			code:    canonical.DiagnosticCacheBreakpointUnsupported,
		},
		{
			name: "Anthropic cache control on assistant image",
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
						Extensions: canonical.Object{"cache_control": cacheControl},
					}},
				})
				return request
			},
			profile: testResponsesProfile,
			code:    canonical.DiagnosticCacheBreakpointUnsupported,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for _, mode := range []canonical.Mode{canonical.ModeStrict, canonical.ModeCompatible, canonical.ModeEmulate} {
				t.Run(string(mode), func(t *testing.T) {
					result := EncodeRequest(context.Background(), test.request(), EncodeOptions{
						TargetModel: "gpt-test",
						Mode:        mode,
						Profile:     test.profile(),
					})
					wantOK := test.modeAware && mode != canonical.ModeStrict
					if result.OK != wantOK {
						t.Fatalf("EncodeRequest().OK = %v, want %v: %#v", result.OK, wantOK, result)
					}
					if wantOK && result.Value == nil {
						t.Fatalf("EncodeRequest().Value = nil: %#v", result)
					}
					if !wantOK && result.Value != nil {
						t.Fatalf("EncodeRequest().Value = %#v, want nil", result.Value)
					}
					assertDiagnosticCode(t, result.Diagnostics, test.code)
					if test.path != "" && !hasDiagnosticPath(result.Diagnostics, test.code, test.path) {
						t.Fatalf("diagnostics = %#v, want path %q", result.Diagnostics, test.path)
					}
				})
			}
		})
	}
}

func TestEncodeRequestValidatesCacheExtensionsOnOmittedTurns(t *testing.T) {
	partWithBreakpoint := func(raw json.RawMessage) canonical.Part {
		return canonical.Part{
			Kind:       canonical.PartText,
			Text:       "stable",
			Extensions: canonical.Object{"prompt_cache_breakpoint": raw},
		}
	}
	tests := []struct {
		name      string
		turns     []canonical.Turn
		code      canonical.DiagnosticCode
		path      string
		modeAware bool
	}{
		{
			name: "malformed breakpoint on omitted mid-conversation instruction",
			turns: []canonical.Turn{
				{Kind: canonical.TurnMessage, Role: canonical.RoleUser, Content: []canonical.Part{{Kind: canonical.PartText, Text: "hello"}}},
				{Kind: canonical.TurnMessage, Role: canonical.RoleSystem, Content: []canonical.Part{partWithBreakpoint(json.RawMessage(`null`))}},
			},
			code: canonical.DiagnosticInvalidCacheControl,
		},
		{
			name: "valid breakpoint on omitted mid-conversation instruction",
			turns: []canonical.Turn{
				{Kind: canonical.TurnMessage, Role: canonical.RoleUser, Content: []canonical.Part{{Kind: canonical.PartText, Text: "hello"}}},
				{Kind: canonical.TurnMessage, Role: canonical.RoleDeveloper, Content: []canonical.Part{partWithBreakpoint(json.RawMessage(`{"mode":"explicit"}`))}},
			},
			code:      canonical.DiagnosticCacheBreakpointUnsupported,
			path:      "turns.1.content.0.extensions.prompt_cache_breakpoint",
			modeAware: true,
		},
		{
			name: "malformed breakpoint on unknown turn",
			turns: []canonical.Turn{
				{Kind: canonical.TurnMessage, Role: canonical.RoleUser, Content: []canonical.Part{{Kind: canonical.PartText, Text: "hello"}}},
				{Kind: canonical.TurnKind("future"), Role: canonical.RoleUser, Content: []canonical.Part{partWithBreakpoint(json.RawMessage(`null`))}},
			},
			code: canonical.DiagnosticInvalidCacheControl,
		},
		{
			name: "malformed breakpoint in ignored message results",
			turns: []canonical.Turn{{
				Kind:    canonical.TurnMessage,
				Role:    canonical.RoleUser,
				Content: []canonical.Part{{Kind: canonical.PartText, Text: "hello"}},
				Results: []canonical.ToolResult{{
					CallID:  "ignored",
					Content: []canonical.Part{partWithBreakpoint(json.RawMessage(`null`))},
				}},
			}},
			code: canonical.DiagnosticInvalidCacheControl,
		},
		{
			name: "valid breakpoint in ignored unknown-turn results",
			turns: []canonical.Turn{
				{Kind: canonical.TurnMessage, Role: canonical.RoleUser, Content: []canonical.Part{{Kind: canonical.PartText, Text: "hello"}}},
				{
					Kind: canonical.TurnKind("future"),
					Results: []canonical.ToolResult{{
						CallID:  "ignored",
						Content: []canonical.Part{partWithBreakpoint(json.RawMessage(`{"mode":"explicit"}`))},
					}},
				},
			},
			code:      canonical.DiagnosticCacheBreakpointUnsupported,
			path:      "turns.1.results.0.content.0.extensions.prompt_cache_breakpoint",
			modeAware: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for _, mode := range []canonical.Mode{canonical.ModeStrict, canonical.ModeCompatible, canonical.ModeEmulate} {
				t.Run(string(mode), func(t *testing.T) {
					profile := testResponsesProfile()
					profile.PromptCache.Mode = capabilities.PromptCacheOpenAI56
					result := EncodeRequest(context.Background(), canonical.Request{Turns: test.turns}, EncodeOptions{
						TargetModel: "gpt-test",
						Mode:        mode,
						Profile:     profile,
					})
					wantOK := test.modeAware && mode != canonical.ModeStrict
					if result.OK != wantOK {
						t.Fatalf("EncodeRequest().OK = %v, want %v: %#v", result.OK, wantOK, result)
					}
					if wantOK && result.Value == nil {
						t.Fatalf("EncodeRequest().Value = nil: %#v", result)
					}
					if !wantOK && result.Value != nil {
						t.Fatalf("EncodeRequest().Value = %#v, want nil", result.Value)
					}
					assertDiagnosticCode(t, result.Diagnostics, test.code)
					if test.path != "" && !hasDiagnosticPath(result.Diagnostics, test.code, test.path) {
						t.Fatalf("diagnostics = %#v, want path %q", result.Diagnostics, test.path)
					}
				})
			}
		})
	}
}

func hasDiagnosticPath(diagnostics []canonical.Diagnostic, code canonical.DiagnosticCode, path string) bool {
	for _, item := range diagnostics {
		if item.Code == code && item.Path != nil && *item.Path == path {
			return true
		}
	}
	return false
}

func TestEncodeRequestDiagnosesPartAndToolCacheExtensions(t *testing.T) {
	cacheControl := json.RawMessage(`{"type":"ephemeral"}`)
	request := canonical.Request{
		Turns: []canonical.Turn{{
			Kind: canonical.TurnMessage,
			Role: canonical.RoleUser,
			Content: []canonical.Part{{
				Kind: canonical.PartText,
				Text: "hello",
				Extensions: canonical.Object{
					"cache_control": cacheControl,
					"future":        json.RawMessage(`true`),
				},
			}},
		}},
		Tools: []canonical.ToolDefinition{{
			Name:        "lookup",
			InputSchema: canonical.Object{"type": json.RawMessage(`"object"`)},
			Extensions:  canonical.Object{"cache_control": cacheControl},
		}},
	}
	result := EncodeRequest(context.Background(), request, EncodeOptions{TargetModel: "gpt-test", Profile: testResponsesProfile()})
	if !result.OK || result.Value == nil {
		t.Fatalf("EncodeRequest() = %#v", result)
	}
	if countDiagnosticCode(result.Diagnostics, canonical.DiagnosticCacheControlProviderMismatch) != 2 {
		t.Fatalf("provider mismatch diagnostics = %#v", result.Diagnostics)
	}
	assertDiagnosticCode(t, result.Diagnostics, DiagnosticUnsupportedExtension)
}

func assertExplicitBreakpoint(t *testing.T, part map[string]any) {
	t.Helper()
	breakpoint, ok := part["prompt_cache_breakpoint"].(map[string]any)
	if !ok || breakpoint["mode"] != "explicit" {
		t.Fatalf("prompt_cache_breakpoint = %#v", part["prompt_cache_breakpoint"])
	}
}

func countDiagnosticCode(diagnostics []canonical.Diagnostic, code canonical.DiagnosticCode) int {
	count := 0
	for _, item := range diagnostics {
		if item.Code == code {
			count++
		}
	}
	return count
}

func testResponsesProfile() capabilities.Profile {
	return capabilities.Profile{
		Provider:          capabilities.ProviderOpenAI,
		Endpoint:          capabilities.EndpointResponses,
		Model:             "gpt-test",
		Temperature:       true,
		TopP:              true,
		StopSequences:     true,
		Metadata:          true,
		StructuredOutput:  true,
		StrictTools:       true,
		ParallelToolCalls: true,
		Images: capabilities.ImageCapabilities{
			URL:    true,
			Base64: true,
			FileID: true,
		},
		Files: capabilities.ImageCapabilities{
			URL:    true,
			Base64: true,
			FileID: true,
		},
		Content: capabilities.ContentCapabilities{
			Text:  true,
			Image: true,
			File:  true,
		},
	}
}

func assertDiagnosticCode(t *testing.T, diagnostics []canonical.Diagnostic, code canonical.DiagnosticCode) {
	t.Helper()
	for _, item := range diagnostics {
		if item.Code == code {
			return
		}
	}
	t.Fatalf("diagnostic %q not found in %#v", code, diagnostics)
}

func pointer[T any](value T) *T {
	return &value
}
