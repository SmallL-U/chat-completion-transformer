package chatcompletions

import (
	"encoding/json"
	"testing"

	"chat-completion-transformer/internal/canonical"
)

func TestDecodeRequestPreservesMessageAndToolSemantics(t *testing.T) {
	result := DecodeRequest([]byte(`{
		"model":"general",
		"messages":[
			{"role":"system","content":"rules"},
			{"role":"user","name":"alice","content":[{"type":"text","text":"weather"}]},
			{"role":"assistant","content":"checking","tool_calls":[
				{"id":"call_a","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Beijing\"}"}},
				{"id":"call_b","type":"function","function":{"name":"weather","arguments":"not json"}}
			]},
			{"role":"tool","tool_call_id":"call_a","content":"20 C"},
			{"role":"tool","tool_call_id":"call_b","content":"24 C"}
		],
		"tools":[{"type":"function","function":{"name":"weather"}}],
		"parallel_tool_calls":false,
		"max_tokens":100,
		"max_completion_tokens":80,
		"stop":"END",
		"stream":true,
		"stream_options":{"include_usage":true}
	}`))

	if !result.OK || result.Value == nil {
		t.Fatalf("result = %#v", result)
	}
	request := *result.Value
	if request.ModelAlias != "general" || len(request.Turns) != 4 {
		t.Fatalf("request = %#v", request)
	}
	assistant := request.Turns[2]
	if len(assistant.Content) != 1 || assistant.Content[0].Text != "checking" || len(assistant.ToolCalls) != 2 {
		t.Fatalf("assistant = %#v", assistant)
	}
	if assistant.ToolCalls[0].ArgumentsRaw != `{"city":"Beijing"}` || assistant.ToolCalls[0].ArgumentsParsed == nil {
		t.Fatalf("valid call = %#v", assistant.ToolCalls[0])
	}
	if assistant.ToolCalls[1].ArgumentsParsed != nil {
		t.Fatalf("invalid arguments were parsed: %#v", assistant.ToolCalls[1])
	}
	results := request.Turns[3]
	if results.Kind != canonical.TurnToolResults || len(results.Results) != 2 {
		t.Fatalf("results = %#v", results)
	}
	if request.MaxOutputTokens == nil || *request.MaxOutputTokens != 80 {
		t.Fatalf("max tokens = %v", request.MaxOutputTokens)
	}
	if len(request.StopSequences) != 1 || request.StopSequences[0] != "END" || !request.Stream || !request.StreamIncludeUsage {
		t.Fatalf("stop/stream = %#v/%t include_usage=%t", request.StopSequences, request.Stream, request.StreamIncludeUsage)
	}
	if request.ParallelToolCalls == nil || *request.ParallelToolCalls {
		t.Fatalf("parallel = %v", request.ParallelToolCalls)
	}
	if len(request.Tools) != 1 || string(request.Tools[0].InputSchema["type"]) != `"object"` {
		t.Fatalf("tools = %#v", request.Tools)
	}
}

func TestDecodeRequestChoicesAndFormats(t *testing.T) {
	tests := []struct {
		name       string
		choice     string
		wantMode   canonical.ToolChoiceMode
		wantName   string
		format     string
		wantFormat canonical.OutputFormatType
	}{
		{name: "auto", choice: `"auto"`, wantMode: canonical.ToolChoiceAuto, format: `{"type":"text"}`, wantFormat: canonical.OutputFormatText},
		{name: "none", choice: `"none"`, wantMode: canonical.ToolChoiceNone, format: `{"type":"json_object"}`, wantFormat: canonical.OutputFormatJSONObject},
		{name: "required", choice: `"required"`, wantMode: canonical.ToolChoiceRequired, format: `{"type":"text"}`, wantFormat: canonical.OutputFormatText},
		{name: "named", choice: `{"type":"function","function":{"name":"lookup"}}`, wantMode: canonical.ToolChoiceNamed, wantName: "lookup", format: `{"type":"json_schema","json_schema":{"name":"answer","description":"shape","strict":true,"schema":{"type":"object"}}}`, wantFormat: canonical.OutputFormatJSONSchema},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			input := `{"model":"general","messages":[{"role":"user","content":"hello"}],"tool_choice":` + test.choice + `,"response_format":` + test.format + `}`
			result := DecodeRequest([]byte(input))
			if !result.OK || result.Value == nil {
				t.Fatalf("result = %#v", result)
			}
			if result.Value.ToolChoice == nil || result.Value.ToolChoice.Mode != test.wantMode {
				t.Fatalf("choice = %#v", result.Value.ToolChoice)
			}
			if test.wantName != "" && (result.Value.ToolChoice.Name == nil || *result.Value.ToolChoice.Name != test.wantName) {
				t.Fatalf("choice name = %#v", result.Value.ToolChoice)
			}
			if result.Value.OutputFormat == nil || result.Value.OutputFormat.Type != test.wantFormat {
				t.Fatalf("format = %#v", result.Value.OutputFormat)
			}
		})
	}
}

func TestDecodeRequestRejectsStreamOptionsWithoutStreaming(t *testing.T) {
	result := DecodeRequest([]byte(`{"model":"general","messages":[{"role":"user","content":"hello"}],"stream_options":{"include_usage":true}}`))
	if result.OK || !containsDiagnostic(result.Diagnostics, diagnosticInvalidRequest) {
		t.Fatalf("result = %#v", result)
	}
}

func TestDecodeRequestMultimodalAndOpaqueParts(t *testing.T) {
	result := DecodeRequest([]byte(`{
		"model":"general",
		"messages":[{"role":"user","content":[
			{"type":"image_url","image_url":{"url":"data:image/png;base64,aW1hZ2U=","detail":"high"}},
			{"type":"input_audio","input_audio":{"data":"YXVkaW8=","format":"wav"}},
			{"type":"future_part","payload":{"precise":9007199254740993}}
		]}]
	}`))

	if !result.OK || result.Value == nil {
		t.Fatalf("result = %#v", result)
	}
	parts := result.Value.Turns[0].Content
	if len(parts) != 3 || parts[0].Kind != canonical.PartImage || parts[1].Kind != canonical.PartAudio || parts[2].Kind != canonical.PartOpaque {
		t.Fatalf("parts = %#v", parts)
	}
	if parts[0].Source == nil || parts[0].Source.Kind != canonical.AssetSourceBase64 || parts[0].Source.MediaType != "image/png" {
		t.Fatalf("image = %#v", parts[0])
	}
	if !json.Valid(parts[2].Value) || !containsDiagnostic(result.Diagnostics, canonical.DiagnosticUnsupportedContentPart) {
		t.Fatalf("opaque/diagnostics = %#v / %#v", parts[2], result.Diagnostics)
	}
	if result.Lossless {
		t.Fatal("opaque content should carry a lossy/unsupported diagnostic")
	}
}

func TestDecodeRequestPreservesUnsupportedFieldsAndTools(t *testing.T) {
	result := DecodeRequest([]byte(`{
		"model":"general",
		"messages":[{"role":"user","content":"hello"}],
		"seed":42,
		"tools":[{"type":"custom","custom":{"name":"shell"}}]
	}`))

	if !result.OK || result.Value == nil {
		t.Fatalf("result = %#v", result)
	}
	if _, ok := result.Value.Extensions["seed"]; !ok {
		t.Fatalf("extensions = %#v", result.Value.Extensions)
	}
	if string(result.Value.Extensions["seed"]) != "42" {
		t.Fatalf("seed raw JSON = %s", result.Value.Extensions["seed"])
	}
	if _, ok := result.Value.Extensions["chat_completions.unsupported_tools"]; !ok {
		t.Fatalf("unsupported tools = %#v", result.Value.Extensions)
	}
	if !containsDiagnostic(result.Diagnostics, diagnosticUnsupportedToolType) {
		t.Fatalf("diagnostics = %#v", result.Diagnostics)
	}
}

func TestDecodeRequestRecognizesTopLevelCacheExtensions(t *testing.T) {
	result := DecodeRequest([]byte(`{
		"model":"general",
		"messages":[{"role":"user","content":"hello"}],
		"cache_control":{"type":"ephemeral"},
		"prompt_cache_key":"tenant:prompt-v1",
		"prompt_cache_options":{"mode":"explicit","ttl":"30m"},
		"prompt_cache_retention":"in_memory"
	}`))

	if !result.OK || result.Value == nil || !result.Lossless {
		t.Fatalf("result = %#v", result)
	}
	want := map[string]string{
		"cache_control":          `{"type":"ephemeral"}`,
		"prompt_cache_key":       `"tenant:prompt-v1"`,
		"prompt_cache_options":   `{"mode":"explicit","ttl":"30m"}`,
		"prompt_cache_retention": `"in_memory"`,
	}
	for name, expected := range want {
		if got := string(result.Value.Extensions[name]); got != expected {
			t.Fatalf("extension %s = %s, want %s", name, got, expected)
		}
	}
	if containsDiagnostic(result.Diagnostics, diagnosticRequestFieldPreserved) {
		t.Fatalf("diagnostics = %#v", result.Diagnostics)
	}
}

func TestDecodeRequestAttachesCacheExtensionsToParts(t *testing.T) {
	result := DecodeRequest([]byte(`{
		"model":"general",
		"messages":[{"role":"user","content":[
			{"type":"text","text":"rules","cache_control":{"type":"ephemeral"}},
			{"type":"image_url","image_url":{"url":"https://example.com/image.png"},"prompt_cache_breakpoint":{"mode":"explicit"}},
			{"type":"file","file":{"file_id":"file_123"},"cache_control":{"type":"ephemeral","ttl":"1h"}}
		]}]
	}`))

	if !result.OK || result.Value == nil || !result.Lossless {
		t.Fatalf("result = %#v", result)
	}
	parts := result.Value.Turns[0].Content
	if len(parts) != 3 {
		t.Fatalf("parts = %#v", parts)
	}
	if string(parts[0].Extensions["cache_control"]) != `{"type":"ephemeral"}` {
		t.Fatalf("text extensions = %#v", parts[0].Extensions)
	}
	if string(parts[1].Extensions["prompt_cache_breakpoint"]) != `{"mode":"explicit"}` {
		t.Fatalf("image extensions = %#v", parts[1].Extensions)
	}
	if string(parts[2].Extensions["cache_control"]) != `{"type":"ephemeral","ttl":"1h"}` {
		t.Fatalf("file extensions = %#v", parts[2].Extensions)
	}
	for _, part := range parts {
		if part.Kind == canonical.PartOpaque {
			t.Fatalf("cache extension created opaque sibling: %#v", parts)
		}
	}
	if containsDiagnostic(result.Diagnostics, diagnosticContentFieldPreserved) {
		t.Fatalf("diagnostics = %#v", result.Diagnostics)
	}
}

func TestDecodeRequestAttachesCacheControlToToolDefinition(t *testing.T) {
	result := DecodeRequest([]byte(`{
		"model":"general",
		"messages":[{"role":"user","content":"hello"}],
		"tools":[{"type":"function","cache_control":{"type":"ephemeral"},"function":{"name":"lookup"}}]
	}`))
	if !result.OK || result.Value == nil || !result.Lossless {
		t.Fatalf("result = %#v", result)
	}
	if len(result.Value.Tools) != 1 || string(result.Value.Tools[0].Extensions["cache_control"]) != `{"type":"ephemeral"}` {
		t.Fatalf("tools = %#v", result.Value.Tools)
	}
	if _, exists := result.Value.Extensions["tools"]; exists {
		t.Fatalf("request extensions = %#v", result.Value.Extensions)
	}

	alias := DecodeRequest([]byte(`{
		"model":"general",
		"messages":[{"role":"user","content":"hello"}],
		"tools":[{"type":"function","function":{"name":"lookup","cache_control":{"type":"ephemeral"}}}]
	}`))
	if alias.OK || !containsDiagnostic(alias.Diagnostics, canonical.DiagnosticInvalidCacheControl) {
		t.Fatalf("function.cache_control alias result = %#v", alias)
	}
}

func TestDecodeRequestRejectsCacheDirectivesAtInvalidPositions(t *testing.T) {
	tests := []struct {
		name string
		body string
		code canonical.DiagnosticCode
	}{
		{
			name: "top-level breakpoint",
			body: `{"model":"general","messages":[{"role":"user","content":"hello"}],"prompt_cache_breakpoint":{"mode":"explicit"}}`,
			code: canonical.DiagnosticCacheBreakpointUnsupported,
		},
		{
			name: "message wrapper",
			body: `{"model":"general","messages":[{"role":"user","content":"hello","cache_control":{"type":"ephemeral"}}]}`,
			code: canonical.DiagnosticInvalidCacheControl,
		},
		{
			name: "content top-level directive",
			body: `{"model":"general","messages":[{"role":"user","content":[{"type":"text","text":"hello","prompt_cache_key":"key"}]}]}`,
			code: canonical.DiagnosticInvalidCacheControl,
		},
		{
			name: "unknown content breakpoint",
			body: `{"model":"general","messages":[{"role":"user","content":[{"type":"future","prompt_cache_breakpoint":{"mode":"explicit"}}]}]}`,
			code: canonical.DiagnosticCacheBreakpointUnsupported,
		},
		{
			name: "nested image directive",
			body: `{"model":"general","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/image.png","cache_control":{"type":"ephemeral"}}}]}]}`,
			code: canonical.DiagnosticInvalidCacheControl,
		},
		{
			name: "tool top-level OpenAI directive",
			body: `{"model":"general","messages":[{"role":"user","content":"hello"}],"tools":[{"type":"function","prompt_cache_options":{"mode":"implicit"},"function":{"name":"lookup"}}]}`,
			code: canonical.DiagnosticInvalidCacheControl,
		},
		{
			name: "assistant tool call directive",
			body: `{"model":"general","messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"call","type":"function","cache_control":{"type":"ephemeral"},"function":{"name":"lookup","arguments":"{}"}}]}]}`,
			code: canonical.DiagnosticInvalidCacheControl,
		},
		{
			name: "tool message wrapper",
			body: `{"model":"general","messages":[{"role":"tool","tool_call_id":"call","content":"result","cache_control":{"type":"ephemeral"}}]}`,
			code: canonical.DiagnosticInvalidCacheControl,
		},
		{
			name: "stream options",
			body: `{"model":"general","messages":[{"role":"user","content":"hello"}],"stream":true,"stream_options":{"include_usage":true,"prompt_cache_key":"key"}}`,
			code: canonical.DiagnosticInvalidCacheControl,
		},
		{
			name: "tool choice",
			body: `{"model":"general","messages":[{"role":"user","content":"hello"}],"tool_choice":{"type":"function","cache_control":{"type":"ephemeral"},"function":{"name":"lookup"}}}`,
			code: canonical.DiagnosticInvalidCacheControl,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := DecodeRequest([]byte(test.body))
			if result.OK || result.Value != nil || !containsDiagnostic(result.Diagnostics, test.code) {
				t.Fatalf("result = %#v", result)
			}
		})
	}
}

func TestDecodeRequestPreservesFutureFieldsInsideKnownStructures(t *testing.T) {
	const contentBlock = `{"type":"text","text":"hello","future":{"precise":9007199254740993}}`
	const toolCall = `{"id":"call_1","type":"function","future_call":true,"function":{"name":"lookup","arguments":"{\"q\":1}","future_function":"x"}}`
	const tools = `[{"type":"function","future_tool":true,"function":{"name":"lookup"}}]`
	const toolChoice = `{"type":"function","future_choice":true,"function":{"name":"lookup"}}`
	const responseFormat = `{"type":"text","future_format":true}`

	input := []byte(`{"model":"general","messages":[` +
		`{"role":"user","content":[` + contentBlock + `]},` +
		`{"role":"assistant","content":null,"tool_calls":[` + toolCall + `]}` +
		`],"tools":` + tools + `,"tool_choice":` + toolChoice + `,"response_format":` + responseFormat + `,"max_completion_tokens":100,"max_tokens":200}`)
	result := DecodeRequest(input)
	if !result.OK || result.Value == nil {
		t.Fatalf("result = %#v", result)
	}

	request := *result.Value
	userParts := request.Turns[0].Content
	if len(userParts) != 2 || userParts[0].Kind != canonical.PartText || userParts[1].Kind != canonical.PartOpaque {
		t.Fatalf("user parts = %#v", userParts)
	}
	if string(userParts[1].Value) != contentBlock {
		t.Fatalf("content block raw JSON = %s", userParts[1].Value)
	}

	assistant := request.Turns[1]
	if len(assistant.ToolCalls) != 1 || assistant.ToolCalls[0].ArgumentsRaw != `{"q":1}` {
		t.Fatalf("assistant tool calls = %#v", assistant.ToolCalls)
	}
	if len(assistant.Content) != 1 || assistant.Content[0].Kind != canonical.PartOpaque || string(assistant.Content[0].Value) != toolCall {
		t.Fatalf("assistant content = %#v", assistant.Content)
	}

	wantExtensions := map[string]string{
		"tools":           tools,
		"tool_choice":     toolChoice,
		"response_format": responseFormat,
		"max_tokens":      "200",
	}
	for name, want := range wantExtensions {
		if got := string(request.Extensions[name]); got != want {
			t.Fatalf("extension %s = %s, want %s", name, got, want)
		}
	}
	if !containsDiagnostic(result.Diagnostics, diagnosticContentFieldPreserved) || !containsDiagnostic(result.Diagnostics, diagnosticMessageFieldPreserved) || !containsDiagnostic(result.Diagnostics, diagnosticRequestFieldPreserved) {
		t.Fatalf("diagnostics = %#v", result.Diagnostics)
	}
	if result.Lossless {
		t.Fatal("preserved future fields must make the transform lossy")
	}
}

func TestDecodeRequestRejectsMalformedInput(t *testing.T) {
	tests := []string{
		`not json`,
		`{}`,
		`{"model":"general","messages":{}}`,
		`{"model":"general","messages":[]}`,
		`{"model":"general","messages":[{"role":"system"}]}`,
		`{"model":"general","messages":[{"role":"developer","content":null}]}`,
		`{"model":"general","messages":[{"role":"user","content":[]}]}`,
		`{"model":"general","messages":[{"role":"assistant","content":null}]}`,
		`{"model":"general","messages":[{"role":"assistant","tool_calls":[]}]}`,
		`{"model":"general","messages":[{"role":"tool","tool_call_id":"call_1","content":null}]}`,
		`{"model":"general","messages":[{"role":"unknown","content":"x"}]}`,
		`{"model":"general","messages":[],"n":0}`,
		`{"model":"general","messages":[],"stop":[1]}`,
	}

	for _, input := range tests {
		result := DecodeRequest([]byte(input))
		if result.OK || result.Value != nil || !canonical.HasErrors(result.Diagnostics) {
			t.Fatalf("input %s: result = %#v", input, result)
		}
	}
}

func containsDiagnostic(diagnostics []canonical.Diagnostic, code canonical.DiagnosticCode) bool {
	for _, item := range diagnostics {
		if item.Code == code {
			return true
		}
	}
	return false
}
