package httpapi

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"chat-completion-transformer/internal/capabilities"
	"chat-completion-transformer/internal/upstream"
	"chat-completion-transformer/pkg/transformer"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
	"go.uber.org/zap/zaptest/observer"
)

const testBodyLimit = 1 << 20

func TestRouterPublicSurface(t *testing.T) {
	provider := httptest.NewServer(http.NotFoundHandler())
	defer provider.Close()
	router := newTestRouter(t, provider.URL, testBodyLimit)

	healthRequest := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	healthResponse := httptest.NewRecorder()
	router.ServeHTTP(healthResponse, healthRequest)
	if healthResponse.Code != http.StatusOK || strings.TrimSpace(healthResponse.Body.String()) != `{"status":"ok"}` {
		t.Fatalf("health response = %d %s", healthResponse.Code, healthResponse.Body.String())
	}

	methodRequest := httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil)
	methodResponse := httptest.NewRecorder()
	router.ServeHTTP(methodResponse, methodRequest)
	if methodResponse.Code != http.StatusMethodNotAllowed || !strings.Contains(methodResponse.Body.String(), "method_not_allowed") {
		t.Fatalf("method response = %d %s", methodResponse.Code, methodResponse.Body.String())
	}

	legacyPaths := []string{
		"/v1/transform/chat-completions/to/openai-responses",
		"/v1/transform/chat-completions/to/anthropic-messages",
		"/v1/transform/openai-responses/to/chat-completions",
		"/v1/transform/anthropic-messages/to/chat-completions",
		"/v1/transform/openai-responses/sse/to/chat-completions",
		"/v1/transform/anthropic-messages/sse/to/chat-completions",
	}
	for _, path := range legacyPaths {
		request := httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{}`))
		response := httptest.NewRecorder()
		router.ServeHTTP(response, request)
		if response.Code != http.StatusNotFound || !strings.Contains(response.Body.String(), "not_found") {
			t.Fatalf("POST %s = %d %s, want OpenAI-shaped 404", path, response.Code, response.Body.String())
		}
	}
}

func TestRecoveryReturnsOpenAIErrorEnvelope(t *testing.T) {
	core, logs := observer.New(zap.ErrorLevel)
	router := gin.New()
	router.Use(recoveryMiddleware(zap.New(core)))
	router.GET("/panic", func(*gin.Context) {
		panic("test panic")
	})

	request := httptest.NewRequest(http.MethodGet, "/panic", nil)
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	if response.Code != http.StatusInternalServerError || !strings.Contains(response.Body.String(), "internal_server_error") {
		t.Fatalf("response = %d %s", response.Code, response.Body.String())
	}
	entries := logs.All()
	if len(entries) != 1 {
		t.Fatalf("panic log entries = %d, want 1", len(entries))
	}
	fields := entries[0].ContextMap()
	panicType, panicTypeOK := fields["panic_type"].(string)
	stack, stackOK := fields["stack"].(string)
	if !panicTypeOK || panicType != "string" || !stackOK || stack == "" {
		t.Fatalf("panic log fields = %#v", fields)
	}
}

func TestRequestLoggingUsesRoutePattern(t *testing.T) {
	core, logs := observer.New(zap.InfoLevel)
	router := gin.New()
	router.Use(requestLoggingMiddleware(zap.New(core)))
	router.GET("/items/:id", func(c *gin.Context) {
		c.Status(http.StatusNoContent)
	})

	request := httptest.NewRequest(http.MethodGet, "/items/private-value?token=secret", nil)
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)

	entries := logs.All()
	if len(entries) != 1 {
		t.Fatalf("request log entries = %d, want 1", len(entries))
	}
	fields := entries[0].ContextMap()
	if fields["path"] != "/items/:id" || fields["status"] != int64(http.StatusNoContent) {
		t.Fatalf("request log fields = %#v", fields)
	}
	encodedFields, err := json.Marshal(fields)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	if strings.Contains(string(encodedFields), "private-value") || strings.Contains(string(encodedFields), "secret") {
		t.Fatalf("request log contains raw URL data: %s", encodedFields)
	}
}

func TestRequestLoggingRedactsUnmatchedPath(t *testing.T) {
	core, logs := observer.New(zap.InfoLevel)
	router := gin.New()
	router.Use(requestLoggingMiddleware(zap.New(core)))

	request := httptest.NewRequest(http.MethodGet, "/private-value?token=secret", nil)
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)

	entries := logs.All()
	if len(entries) != 1 {
		t.Fatalf("request log entries = %d, want 1", len(entries))
	}
	fields := entries[0].ContextMap()
	if fields["path"] != "<unmatched>" {
		t.Fatalf("request log fields = %#v", fields)
	}
	encodedFields, err := json.Marshal(fields)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	if strings.Contains(string(encodedFields), "private-value") || strings.Contains(string(encodedFields), "secret") {
		t.Fatalf("request log contains raw URL data: %s", encodedFields)
	}
}

func TestRouterBufferedProviders(t *testing.T) {
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Errorf("decode upstream request: %v", err)
		}
		writer.Header().Set("Content-Type", "application/json")

		switch request.URL.Path {
		case "/v1/responses":
			if request.Header.Get("Authorization") != "Bearer client-openai-key" {
				t.Errorf("OpenAI Authorization = %q", request.Header.Get("Authorization"))
			}
			if body["model"] != "openai-target" || body["stream"] != false {
				t.Errorf("OpenAI request = %#v", body)
			}
			_, _ = io.WriteString(writer, responsesResponse())
		case "/v1/messages":
			if request.Header.Get("x-api-key") != "client-anthropic-key" {
				t.Errorf("Anthropic x-api-key = %q", request.Header.Get("x-api-key"))
			}
			if request.Header.Get("Authorization") != "" {
				t.Errorf("Anthropic Authorization must not be forwarded")
			}
			if request.Header.Get("anthropic-version") != upstream.DefaultAnthropicVersion {
				t.Errorf("Anthropic version = %q", request.Header.Get("anthropic-version"))
			}
			if body["model"] != "anthropic-target" || body["stream"] != false {
				t.Errorf("Anthropic request = %#v", body)
			}
			_, _ = io.WriteString(writer, anthropicResponse())
		default:
			http.NotFound(writer, request)
		}
	}))
	defer provider.Close()
	router := newTestRouter(t, provider.URL, testBodyLimit)

	tests := []struct {
		name   string
		model  string
		apiKey string
	}{
		{name: "OpenAI Responses", model: "openai-chat", apiKey: "client-openai-key"},
		{name: "Anthropic Messages", model: "anthropic-chat", apiKey: "client-anthropic-key"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			body := `{"model":"` + test.model + `","messages":[{"role":"user","content":"hello"}],"max_completion_tokens":12}`
			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
			request.Header.Set("Authorization", "Bearer "+test.apiKey)
			response := httptest.NewRecorder()
			router.ServeHTTP(response, request)

			if response.Code != http.StatusOK {
				t.Fatalf("response = %d %s", response.Code, response.Body.String())
			}
			var result map[string]any
			if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
			if result["object"] != "chat.completion" {
				t.Fatalf("response is not a Chat Completion: %#v", result)
			}
			if _, wrapped := result["value"]; wrapped {
				t.Fatalf("response still contains transform envelope: %#v", result)
			}
		})
	}
}

func TestRouterForwardsPromptCacheControlsByProfile(t *testing.T) {
	type capturedRequest struct {
		path string
		body json.RawMessage
	}
	captured := make(chan capturedRequest, 3)
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			t.Errorf("read upstream request: %v", err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}
		captured <- capturedRequest{path: request.URL.Path, body: body}
		writer.Header().Set("Content-Type", "application/json")
		if request.URL.Path == "/v1/messages" {
			_, _ = io.WriteString(writer, anthropicResponse())
			return
		}
		_, _ = io.WriteString(writer, responsesResponse())
	}))
	defer provider.Close()
	profiles, routes, upstreamRoutes := promptCacheTestConfiguration()
	router := newPromptCacheTestRouter(t, provider.URL, transformer.ModeCompatible, profiles, routes, upstreamRoutes, nil)

	tests := []struct {
		name       string
		body       string
		wantPath   string
		assertBody func(*testing.T, json.RawMessage)
	}{
		{
			name: "Anthropic top-level content and tool controls",
			body: `{
				"model":"anthropic-cache",
				"messages":[{"role":"user","content":[{"type":"text","text":"hello","cache_control":{"type":"ephemeral"},"prompt_cache_breakpoint":{"mode":"explicit"}}]}],
				"tools":[{"type":"function","cache_control":{"type":"ephemeral"},"function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}],
				"cache_control":{"type":"ephemeral"},
				"prompt_cache_key":"must-not-leak",
				"max_completion_tokens":12
			}`,
			wantPath:   "/v1/messages",
			assertBody: assertAnthropicPromptCacheRequest,
		},
		{
			name: "OpenAI legacy top-level controls",
			body: `{
				"model":"openai-legacy-cache",
				"messages":[{"role":"user","content":"hello"}],
				"prompt_cache_key":"tenant:legacy",
				"prompt_cache_retention":"in_memory",
				"cache_control":{"type":"ephemeral"}
			}`,
			wantPath:   "/v1/responses",
			assertBody: assertOpenAILegacyPromptCacheRequest,
		},
		{
			name: "OpenAI 5.6 top-level controls and breakpoint",
			body: `{
				"model":"openai-56-cache",
				"messages":[{"role":"user","content":[{"type":"text","text":"hello","prompt_cache_breakpoint":{"mode":"explicit"},"cache_control":{"type":"ephemeral"}}]}],
				"tools":[{"type":"function","cache_control":{"type":"ephemeral"},"function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}],
				"prompt_cache_key":"tenant:56",
				"prompt_cache_options":{"mode":"explicit","ttl":"30m"},
				"cache_control":{"type":"ephemeral"}
			}`,
			wantPath:   "/v1/responses",
			assertBody: assertOpenAI56PromptCacheRequest,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(test.body))
			request.Header.Set("Authorization", "Bearer client-key")
			response := httptest.NewRecorder()
			router.ServeHTTP(response, request)
			if response.Code != http.StatusOK {
				t.Fatalf("response = %d %s", response.Code, response.Body.String())
			}

			upstreamRequest := <-captured
			if upstreamRequest.path != test.wantPath {
				t.Fatalf("upstream path = %q, want %q", upstreamRequest.path, test.wantPath)
			}
			test.assertBody(t, upstreamRequest.body)
		})
	}
}

func TestRouterPromptCacheGenerationMismatchModes(t *testing.T) {
	tests := []struct {
		name       string
		mode       transformer.Mode
		wantStatus int
		wantCalls  int64
	}{
		{name: "strict rejects before upstream", mode: transformer.ModeStrict, wantStatus: http.StatusBadRequest},
		{name: "compatible warns and drops", mode: transformer.ModeCompatible, wantStatus: http.StatusOK, wantCalls: 1},
		{name: "emulate warns and drops", mode: transformer.ModeEmulate, wantStatus: http.StatusOK, wantCalls: 1},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var calls atomic.Int64
			var upstreamBody json.RawMessage
			doer := httpDoerFunc(func(request *http.Request) (*http.Response, error) {
				calls.Add(1)
				body, err := io.ReadAll(request.Body)
				if err != nil {
					return nil, err
				}
				upstreamBody = body
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": []string{"application/json"}},
					Body:       io.NopCloser(strings.NewReader(responsesResponse())),
				}, nil
			})
			profiles, routes, upstreamRoutes := promptCacheTestConfiguration()
			router := newPromptCacheTestRouter(t, "https://provider.example", test.mode, profiles, routes, upstreamRoutes, doer)

			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{
				"model":"openai-legacy-cache",
				"messages":[{"role":"user","content":"hello"}],
				"prompt_cache_options":{"mode":"implicit"}
			}`))
			request.Header.Set("Authorization", "Bearer client-key")
			response := httptest.NewRecorder()
			router.ServeHTTP(response, request)

			if response.Code != test.wantStatus {
				t.Fatalf("response = %d %s, want %d", response.Code, response.Body.String(), test.wantStatus)
			}
			if calls.Load() != test.wantCalls {
				t.Fatalf("upstream calls = %d, want %d", calls.Load(), test.wantCalls)
			}
			if test.mode == transformer.ModeStrict {
				if !strings.Contains(response.Body.String(), string(transformer.DiagnosticCacheControlUnsupported)) {
					t.Fatalf("strict response = %s", response.Body.String())
				}
				return
			}

			object := decodeTestJSONObject(t, upstreamBody)
			assertTestJSONFieldAbsent(t, object, "prompt_cache_options")
			diagnostics := decodeTestDiagnosticsHeader(t, response.Header().Get(diagnosticsTrailer))
			assertTestDiagnostic(t, diagnostics, transformer.SeverityWarning, transformer.DiagnosticCacheControlUnsupported)
		})
	}
}

func TestRouterRejectsCacheDirectiveInvalidPositionBeforeUpstream(t *testing.T) {
	for _, mode := range []transformer.Mode{transformer.ModeStrict, transformer.ModeCompatible, transformer.ModeEmulate} {
		t.Run(string(mode), func(t *testing.T) {
			var calls atomic.Int64
			doer := httpDoerFunc(func(*http.Request) (*http.Response, error) {
				calls.Add(1)
				return nil, errors.New("upstream must not be called")
			})
			profiles, routes, upstreamRoutes := promptCacheTestConfiguration()
			router := newPromptCacheTestRouter(t, "https://provider.example", mode, profiles, routes, upstreamRoutes, doer)

			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{
				"model":"openai-56-cache",
				"messages":[{"role":"user","content":"hello"}],
				"tools":[{"type":"function","function":{"name":"lookup","cache_control":{"type":"ephemeral"}}}]
			}`))
			request.Header.Set("Authorization", "Bearer client-key")
			response := httptest.NewRecorder()
			router.ServeHTTP(response, request)

			if response.Code != http.StatusBadRequest {
				t.Fatalf("response = %d %s", response.Code, response.Body.String())
			}
			if calls.Load() != 0 {
				t.Fatalf("upstream calls = %d, want 0", calls.Load())
			}
			if !strings.Contains(response.Body.String(), string(transformer.DiagnosticInvalidCacheControl)) {
				t.Fatalf("response = %s", response.Body.String())
			}
		})
	}
}

func TestRouterNormalizesBufferedCacheUsage(t *testing.T) {
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		if request.URL.Path == "/v1/messages" {
			_, _ = io.WriteString(writer, anthropicCacheUsageResponse())
			return
		}
		_, _ = io.WriteString(writer, responsesCacheUsageResponse())
	}))
	defer provider.Close()
	router := newTestRouter(t, provider.URL, testBodyLimit)

	tests := []struct {
		model                string
		wantPromptTokens     int64
		wantCompletionTokens int64
		wantTotalTokens      int64
	}{
		{model: "openai-chat", wantPromptTokens: 12, wantCompletionTokens: 3, wantTotalTokens: 15},
		{model: "anthropic-chat", wantPromptTokens: 12, wantCompletionTokens: 2, wantTotalTokens: 14},
	}
	for _, test := range tests {
		t.Run(test.model, func(t *testing.T) {
			body := `{"model":"` + test.model + `","messages":[{"role":"user","content":"hello"}]}`
			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
			request.Header.Set("Authorization", "Bearer client-key")
			response := httptest.NewRecorder()
			router.ServeHTTP(response, request)
			if response.Code != http.StatusOK {
				t.Fatalf("response = %d %s", response.Code, response.Body.String())
			}

			var result struct {
				Usage *chatCacheUsage `json:"usage"`
			}
			if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
			assertChatCacheUsage(t, result.Usage, test.wantPromptTokens, test.wantCompletionTokens, test.wantTotalTokens, 5, 4)
		})
	}
}

func TestRouterDoesNotTurnFailedResponseIntoCompletion(t *testing.T) {
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"resp_failed","status":"failed","output":[],"error":{"code":"server_error","message":"provider failed"}}`)
	}))
	defer provider.Close()
	router := newTestRouter(t, provider.URL, testBodyLimit)

	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"openai-chat","messages":[{"role":"user","content":"hello"}]}`))
	request.Header.Set("Authorization", "Bearer client-key")
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	if response.Code != http.StatusBadGateway || !strings.Contains(response.Body.String(), "provider failed") || !strings.Contains(response.Body.String(), "server_error") {
		t.Fatalf("response = %d %s", response.Code, response.Body.String())
	}
}

func TestRouterPreservesContentFilterFinishReason(t *testing.T) {
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"resp_filtered","created_at":123,"model":"gpt-test","status":"incomplete","output":[],"incomplete_details":{"reason":"content_filter"}}`)
	}))
	defer provider.Close()
	router := newTestRouter(t, provider.URL, testBodyLimit)

	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"openai-chat","messages":[{"role":"user","content":"hello"}]}`))
	request.Header.Set("Authorization", "Bearer client-key")
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"finish_reason":"content_filter"`) {
		t.Fatalf("response = %d %s", response.Code, response.Body.String())
	}
}

func TestRouterPreservesStreamContentFilterFinishReason(t *testing.T) {
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(writer,
			responsesFrame("response.created", `{"type":"response.created","response":{"id":"resp_filtered","created_at":123,"model":"gpt-test","status":"in_progress","output":[]}}`)+
				responsesFrame("response.incomplete", `{"type":"response.incomplete","response":{"id":"resp_filtered","created_at":123,"model":"gpt-test","status":"incomplete","output":[],"incomplete_details":{"reason":"content_filter"}}}`),
		)
	}))
	defer provider.Close()
	router := newTestRouter(t, provider.URL, testBodyLimit)

	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"openai-chat","messages":[{"role":"user","content":"hello"}],"stream":true}`))
	request.Header.Set("Authorization", "Bearer client-key")
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"finish_reason":"content_filter"`) || !strings.HasSuffix(response.Body.String(), "data: [DONE]\n\n") {
		t.Fatalf("response = %d %s", response.Code, response.Body.String())
	}
}

func TestRouterStreamsProviders(t *testing.T) {
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		writer.WriteHeader(http.StatusOK)
		if request.URL.Path == "/v1/responses" {
			_, _ = io.WriteString(writer, responsesTextStream())
			return
		}
		if request.URL.Path == "/v1/messages" {
			_, _ = io.WriteString(writer, anthropicTextStream())
			return
		}
		http.NotFound(writer, request)
	}))
	defer provider.Close()
	router := newTestRouter(t, provider.URL, testBodyLimit)

	for _, model := range []string{"openai-chat", "anthropic-chat"} {
		t.Run(model, func(t *testing.T) {
			body := `{"model":"` + model + `","messages":[{"role":"user","content":"hello"}],"max_completion_tokens":12,"stream":true}`
			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
			request.Header.Set("Authorization", "Bearer client-key")
			response := httptest.NewRecorder()
			router.ServeHTTP(response, request)

			if response.Code != http.StatusOK {
				t.Fatalf("response = %d %s", response.Code, response.Body.String())
			}
			if response.Header().Get("Content-Type") != "text/event-stream" || !response.Flushed {
				t.Fatalf("headers = %#v, flushed = %t", response.Header(), response.Flushed)
			}
			output := response.Body.String()
			if !strings.Contains(output, `"object":"chat.completion.chunk"`) || !strings.Contains(output, `"content":"hello"`) {
				t.Fatalf("stream = %s", output)
			}
			if !strings.HasSuffix(output, "data: [DONE]\n\n") {
				t.Fatalf("stream has no completion marker: %s", output)
			}
		})
	}
}

func TestRouterFlushesBeforeUpstreamCompletes(t *testing.T) {
	firstFrameSent := make(chan struct{})
	releaseUpstream := make(chan struct{})
	firstProviderFrame := responsesFrame("response.created", `{"type":"response.created","response":{"id":"resp_1","created_at":123,"model":"gpt-test","status":"in_progress","output":[]}}`)
	remainingProviderFrames := strings.TrimPrefix(responsesTextStream(), firstProviderFrame)
	released := false
	defer func() {
		if !released {
			close(releaseUpstream)
		}
	}()

	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(writer, firstProviderFrame)
		writer.(http.Flusher).Flush()
		close(firstFrameSent)
		<-releaseUpstream
		_, _ = io.WriteString(writer, remainingProviderFrames)
	}))
	defer provider.Close()

	gateway := httptest.NewServer(newTestRouter(t, provider.URL, testBodyLimit))
	defer gateway.Close()
	request, err := http.NewRequest(http.MethodPost, gateway.URL+"/v1/chat/completions", strings.NewReader(`{"model":"openai-chat","messages":[{"role":"user","content":"hello"}],"stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer client-key")

	type responseResult struct {
		response *http.Response
		err      error
	}
	results := make(chan responseResult, 1)
	go func() {
		response, requestErr := gateway.Client().Do(request)
		results <- responseResult{response: response, err: requestErr}
	}()

	select {
	case <-firstFrameSent:
	case <-time.After(time.Second):
		t.Fatal("upstream did not send its first frame")
	}

	var response *http.Response
	select {
	case result := <-results:
		if result.err != nil {
			t.Fatal(result.err)
		}
		response = result.response
	case <-time.After(time.Second):
		t.Fatal("gateway did not flush before upstream completion")
	}
	defer response.Body.Close()

	reader := bufio.NewReader(response.Body)
	firstLine, err := reader.ReadString('\n')
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(firstLine, "data: ") || !strings.Contains(firstLine, `"role":"assistant"`) {
		t.Fatalf("first line = %q", firstLine)
	}

	close(releaseUpstream)
	released = true
	remaining, err := io.ReadAll(reader)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(remaining), `"content":"hello"`) || !strings.HasSuffix(string(remaining), "data: [DONE]\n\n") {
		t.Fatalf("remaining stream = %s", remaining)
	}
}

func TestRouterCancelsUpstreamWhenClientCancels(t *testing.T) {
	upstreamStarted := make(chan struct{})
	upstreamCanceled := make(chan struct{})
	doer := httpDoerFunc(func(request *http.Request) (*http.Response, error) {
		close(upstreamStarted)
		<-request.Context().Done()
		close(upstreamCanceled)
		return nil, request.Context().Err()
	})
	router := newTestRouterWithDoer(t, "https://provider.example", testBodyLimit, doer)
	ctx, cancel := context.WithCancel(context.Background())
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"openai-chat","messages":[{"role":"user","content":"hello"}]}`)).WithContext(ctx)
	request.Header.Set("Authorization", "Bearer client-key")
	done := make(chan struct{})
	go func() {
		router.ServeHTTP(httptest.NewRecorder(), request)
		close(done)
	}()

	select {
	case <-upstreamStarted:
	case <-time.After(time.Second):
		t.Fatal("upstream request did not start")
	}
	cancel()
	select {
	case <-upstreamCanceled:
	case <-time.After(time.Second):
		t.Fatal("upstream request was not canceled")
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("gateway handler did not stop after cancellation")
	}
}

func TestRouterCancelsBlockedRequestBodyRead(t *testing.T) {
	body := newBlockingBody()
	router := newTestRouter(t, "https://provider.example", testBodyLimit)
	ctx, cancel := context.WithCancel(context.Background())
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil).WithContext(ctx)
	request.Body = body
	request.ContentLength = -1
	request.Header.Set("Authorization", "Bearer client-key")
	done := make(chan struct{})
	go func() {
		router.ServeHTTP(httptest.NewRecorder(), request)
		close(done)
	}()

	waitForSignal(t, body.started, "request body read did not start")
	cancel()
	waitForSignal(t, body.closed, "request body was not closed after cancellation")
	waitForSignal(t, done, "gateway handler did not stop after request cancellation")
}

func TestRouterCancelsBlockedUpstreamStreamRead(t *testing.T) {
	body := newBlockingBody()
	doer := httpDoerFunc(func(_ *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       body,
		}, nil
	})
	router := newTestRouterWithDoer(t, "https://provider.example", testBodyLimit, doer)
	ctx, cancel := context.WithCancel(context.Background())
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"openai-chat","messages":[{"role":"user","content":"hello"}],"stream":true}`)).WithContext(ctx)
	request.Header.Set("Authorization", "Bearer client-key")
	done := make(chan struct{})
	go func() {
		router.ServeHTTP(httptest.NewRecorder(), request)
		close(done)
	}()

	waitForSignal(t, body.started, "upstream stream body read did not start")
	cancel()
	waitForSignal(t, body.closed, "upstream stream body was not closed after cancellation")
	waitForSignal(t, done, "gateway handler did not stop after stream cancellation")
}

func TestRouterEmitsStreamErrorAfterOutputStarts(t *testing.T) {
	doer := httpDoerFunc(func(_ *http.Request) (*http.Response, error) {
		body := &chunkReadCloser{chunks: [][]byte{
			[]byte(responsesFrame("response.created", `{"type":"response.created","response":{"id":"resp_1","created_at":123,"model":"gpt-test","status":"in_progress","output":[]}}`)),
			[]byte(responsesFrame("response.output_text.delta", `{"type":"response.output_text.delta","output_index":"invalid","content_index":0,"delta":"x"}`)),
		}}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       body,
		}, nil
	})
	router := newTestRouterWithDoer(t, "https://provider.example", testBodyLimit, doer)
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"openai-chat","messages":[{"role":"user","content":"hello"}],"stream":true}`))
	request.Header.Set("Authorization", "Bearer client-key")
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)

	output := response.Body.String()
	if response.Code != http.StatusOK || !strings.Contains(output, `"role":"assistant"`) {
		t.Fatalf("partial stream = %d %s", response.Code, output)
	}
	if !strings.Contains(output, `"error":`) || !strings.Contains(output, "invalid_responses_stream_event") {
		t.Fatalf("stream error frame is missing: %s", output)
	}
	if strings.Contains(output, "data: [DONE]") {
		t.Fatalf("failed stream must not emit DONE: %s", output)
	}
}

func TestRouterRejectsNonTerminalStreamFinish(t *testing.T) {
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(writer, strings.Replace(anthropicTextStream(), `"end_turn"`, `"pause_turn"`, 1))
	}))
	defer provider.Close()
	router := newTestRouter(t, provider.URL, testBodyLimit)

	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"anthropic-chat","messages":[{"role":"user","content":"hello"}],"stream":true}`))
	request.Header.Set("Authorization", "Bearer client-key")
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)

	output := response.Body.String()
	if response.Code != http.StatusOK || !strings.Contains(output, "upstream_response_not_terminal") {
		t.Fatalf("response = %d %s", response.Code, output)
	}
	if strings.Contains(output, "data: [DONE]") {
		t.Fatalf("non-terminal stream must not emit DONE: %s", output)
	}
}

func TestRouterStopsReadingAfterTerminalStreamEvent(t *testing.T) {
	body := &chunkReadCloser{chunks: [][]byte{
		[]byte(responsesTextStream()),
		[]byte(responsesFrame("error", `{"type":"error","code":"late_error","message":"must be ignored"}`)),
	}}
	doer := httpDoerFunc(func(_ *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       body,
		}, nil
	})
	router := newTestRouterWithDoer(t, "https://provider.example", testBodyLimit, doer)
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"openai-chat","messages":[{"role":"user","content":"hello"}],"stream":true}`))
	request.Header.Set("Authorization", "Bearer client-key")
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)

	output := response.Body.String()
	if !strings.HasSuffix(output, "data: [DONE]\n\n") || strings.Contains(output, `"error":`) {
		t.Fatalf("terminal stream output = %s", output)
	}
	if len(body.chunks) != 1 {
		t.Fatalf("gateway read %d post-terminal chunks, want 0", 1-len(body.chunks))
	}
}

func TestRouterRespectsStreamUsageOption(t *testing.T) {
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(writer, anthropicTextStream())
	}))
	defer provider.Close()
	router := newTestRouter(t, provider.URL, testBodyLimit)

	tests := []struct {
		name        string
		streamExtra string
		wantUsage   bool
	}{
		{name: "default", wantUsage: false},
		{name: "included", streamExtra: `,"stream_options":{"include_usage":true}`, wantUsage: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			body := `{"model":"anthropic-chat","messages":[{"role":"user","content":"hello"}],"stream":true` + test.streamExtra + `}`
			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
			request.Header.Set("Authorization", "Bearer client-key")
			response := httptest.NewRecorder()
			router.ServeHTTP(response, request)
			if response.Code != http.StatusOK {
				t.Fatalf("response = %d %s", response.Code, response.Body.String())
			}
			hasUsage := strings.Contains(response.Body.String(), `"usage":`)
			if hasUsage != test.wantUsage {
				t.Fatalf("usage present = %t, want %t in %s", hasUsage, test.wantUsage, response.Body.String())
			}
		})
	}
}

func TestRouterStreamsCacheUsageOnlyWhenRequested(t *testing.T) {
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		if request.URL.Path == "/v1/messages" {
			_, _ = io.WriteString(writer, anthropicCacheUsageStream())
			return
		}
		_, _ = io.WriteString(writer, responsesCacheUsageStream())
	}))
	defer provider.Close()
	router := newTestRouter(t, provider.URL, testBodyLimit)

	providers := []struct {
		model                string
		wantCompletionTokens int64
		wantTotalTokens      int64
	}{
		{model: "openai-chat", wantCompletionTokens: 3, wantTotalTokens: 15},
		{model: "anthropic-chat", wantCompletionTokens: 2, wantTotalTokens: 14},
	}
	for _, provider := range providers {
		for _, includeUsage := range []bool{false, true} {
			name := provider.model + "/without-usage"
			streamOptions := ""
			if includeUsage {
				name = provider.model + "/with-usage"
				streamOptions = `,"stream_options":{"include_usage":true}`
			}
			t.Run(name, func(t *testing.T) {
				body := `{"model":"` + provider.model + `","messages":[{"role":"user","content":"hello"}],"stream":true` + streamOptions + `}`
				request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
				request.Header.Set("Authorization", "Bearer client-key")
				response := httptest.NewRecorder()
				router.ServeHTTP(response, request)
				if response.Code != http.StatusOK {
					t.Fatalf("response = %d %s", response.Code, response.Body.String())
				}
				if !strings.HasSuffix(response.Body.String(), "data: [DONE]\n\n") {
					t.Fatalf("stream has no completion marker: %s", response.Body.String())
				}

				usage, found := decodeTestSSEUsage(t, response.Body.String())
				if !includeUsage {
					if found {
						t.Fatalf("usage must be omitted: %#v", usage)
					}
					return
				}
				if !found {
					t.Fatalf("usage frame is missing: %s", response.Body.String())
				}
				assertChatCacheUsage(t, usage, 12, provider.wantCompletionTokens, provider.wantTotalTokens, 5, 4)
			})
		}
	}
}

func TestRouterNormalizesUpstreamError(t *testing.T) {
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		writer.Header().Set("Retry-After", "2")
		writer.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(writer, `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"},"request_id":"req_1"}`)
	}))
	defer provider.Close()
	router := newTestRouter(t, provider.URL, testBodyLimit)

	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"anthropic-chat","messages":[{"role":"user","content":"hello"}],"max_completion_tokens":12}`))
	request.Header.Set("Authorization", "Bearer client-key")
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)

	if response.Code != http.StatusTooManyRequests || response.Header().Get("Retry-After") != "2" {
		t.Fatalf("response = %d headers=%#v", response.Code, response.Header())
	}
	var result apiErrorResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Error.Message != "slow down" || result.Error.Type != "rate_limit_error" || result.Error.Code != "rate_limit_error" {
		t.Fatalf("error = %#v", result.Error)
	}
}

func TestParseUpstreamErrorKeepsOpenAIFieldTypes(t *testing.T) {
	tests := []struct {
		name      string
		body      string
		wantCode  any
		wantParam any
	}{
		{name: "explicit nulls", body: `{"error":{"message":"bad","type":"invalid_request_error","param":null,"code":null}}`, wantCode: nil, wantParam: nil},
		{name: "missing code falls back to type", body: `{"error":{"message":"bad","type":"invalid_request_error"}}`, wantCode: "invalid_request_error", wantParam: nil},
		{name: "invalid shapes are normalized", body: `{"error":{"message":"bad","type":"invalid_request_error","param":{},"code":[]}}`, wantCode: "upstream_error", wantParam: nil},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := parseUpstreamError(http.StatusBadRequest, []byte(test.body))
			if got.Code != test.wantCode || got.Param != test.wantParam {
				t.Fatalf("error = %#v, want code %#v param %#v", got, test.wantCode, test.wantParam)
			}
		})
	}
}

func TestRouterRequiresBearerAuthentication(t *testing.T) {
	var calls atomic.Int64
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		writer.WriteHeader(http.StatusInternalServerError)
	}))
	defer provider.Close()
	router := newTestRouter(t, provider.URL, testBodyLimit)

	for _, authorization := range []string{"", "Basic secret", "Bearer", "Bearer one two"} {
		request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"openai-chat","messages":[{"role":"user","content":"hello"}]}`))
		request.Header.Set("Authorization", authorization)
		response := httptest.NewRecorder()
		router.ServeHTTP(response, request)
		if response.Code != http.StatusUnauthorized || !strings.Contains(response.Body.String(), "invalid_api_key") {
			t.Fatalf("Authorization %q response = %d %s", authorization, response.Code, response.Body.String())
		}
	}

	duplicateRequest := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"openai-chat","messages":[{"role":"user","content":"hello"}]}`))
	duplicateRequest.Header.Add("Authorization", "Bearer first-key")
	duplicateRequest.Header.Add("Authorization", "Bearer second-key")
	duplicateResponse := httptest.NewRecorder()
	router.ServeHTTP(duplicateResponse, duplicateRequest)
	if duplicateResponse.Code != http.StatusUnauthorized || !strings.Contains(duplicateResponse.Body.String(), "invalid_api_key") {
		t.Fatalf("duplicate Authorization response = %d %s", duplicateResponse.Code, duplicateResponse.Body.String())
	}

	if calls.Load() != 0 {
		t.Fatalf("upstream calls = %d, want 0", calls.Load())
	}
}

func TestRouterRejectsInvalidAndUnknownRequestsBeforeUpstream(t *testing.T) {
	var calls atomic.Int64
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		writer.WriteHeader(http.StatusInternalServerError)
	}))
	defer provider.Close()
	router := newTestRouter(t, provider.URL, testBodyLimit)

	tests := []struct {
		name       string
		body       string
		wantStatus int
		wantCode   string
	}{
		{name: "invalid JSON", body: `{`, wantStatus: http.StatusBadRequest, wantCode: "invalid_json"},
		{name: "unknown model", body: `{"model":"missing","messages":[{"role":"user","content":"hello"}]}`, wantStatus: http.StatusNotFound, wantCode: "model_not_found"},
		{name: "multiple candidates", body: `{"model":"openai-chat","messages":[{"role":"user","content":"hello"}],"n":2}`, wantStatus: http.StatusBadRequest, wantCode: "candidate_count_unsupported"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(test.body))
			request.Header.Set("Authorization", "Bearer client-key")
			response := httptest.NewRecorder()
			router.ServeHTTP(response, request)
			if response.Code != test.wantStatus || !strings.Contains(response.Body.String(), test.wantCode) {
				t.Fatalf("response = %d %s", response.Code, response.Body.String())
			}
		})
	}
	if calls.Load() != 0 {
		t.Fatalf("upstream calls = %d, want 0", calls.Load())
	}
}

func TestRouterRejectsOversizedRequestAndInvalidUpstream(t *testing.T) {
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{not-json`)
	}))
	defer provider.Close()

	t.Run("request", func(t *testing.T) {
		router := newTestRouter(t, provider.URL, 8)
		request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"more":"than eight bytes"}`))
		request.Header.Set("Authorization", "Bearer client-key")
		response := httptest.NewRecorder()
		router.ServeHTTP(response, request)
		if response.Code != http.StatusRequestEntityTooLarge || !strings.Contains(response.Body.String(), "request_too_large") {
			t.Fatalf("response = %d %s", response.Code, response.Body.String())
		}
	})

	t.Run("upstream", func(t *testing.T) {
		router := newTestRouter(t, provider.URL, testBodyLimit)
		request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"openai-chat","messages":[{"role":"user","content":"hello"}]}`))
		request.Header.Set("Authorization", "Bearer client-key")
		response := httptest.NewRecorder()
		router.ServeHTTP(response, request)
		if response.Code != http.StatusBadGateway || !strings.Contains(response.Body.String(), "invalid_responses_response") {
			t.Fatalf("response = %d %s", response.Code, response.Body.String())
		}
	})
}

func TestRouterLimitsUpstreamResponses(t *testing.T) {
	t.Run("buffered", func(t *testing.T) {
		doer := httpDoerFunc(func(_ *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader("123456789")),
			}, nil
		})
		router := newTestRouterWithLimits(t, "https://provider.example", testBodyLimit, 8, doer)
		request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"openai-chat","messages":[{"role":"user","content":"hello"}]}`))
		request.Header.Set("Authorization", "Bearer client-key")
		response := httptest.NewRecorder()
		router.ServeHTTP(response, request)
		if response.Code != http.StatusBadGateway || !strings.Contains(response.Body.String(), "upstream_response_too_large") {
			t.Fatalf("response = %d %s", response.Code, response.Body.String())
		}
	})

	t.Run("streaming after first frame", func(t *testing.T) {
		firstFrame := responsesFrame("response.created", `{"type":"response.created","response":{"id":"resp_1","created_at":123,"model":"gpt-test","status":"in_progress","output":[]}}`)
		doer := httpDoerFunc(func(_ *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body:       &chunkReadCloser{chunks: [][]byte{[]byte(firstFrame), []byte("too large")}},
			}, nil
		})
		router := newTestRouterWithLimits(t, "https://provider.example", testBodyLimit, int64(len(firstFrame)+1), doer)
		request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"openai-chat","messages":[{"role":"user","content":"hello"}],"stream":true}`))
		request.Header.Set("Authorization", "Bearer client-key")
		response := httptest.NewRecorder()
		router.ServeHTTP(response, request)
		output := response.Body.String()
		if response.Code != http.StatusOK || !strings.Contains(output, `"role":"assistant"`) || !strings.Contains(output, "upstream_response_too_large") {
			t.Fatalf("response = %d %s", response.Code, output)
		}
		if strings.Contains(output, "data: [DONE]") {
			t.Fatalf("oversized stream must not emit DONE: %s", output)
		}
	})
}

func TestRouterRejectsNonEventStreamResponse(t *testing.T) {
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, responsesResponse())
	}))
	defer provider.Close()
	router := newTestRouter(t, provider.URL, testBodyLimit)

	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"openai-chat","messages":[{"role":"user","content":"hello"}],"stream":true}`))
	request.Header.Set("Authorization", "Bearer client-key")
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	if response.Code != http.StatusBadGateway || !strings.Contains(response.Body.String(), "upstream_invalid_content_type") {
		t.Fatalf("response = %d %s", response.Code, response.Body.String())
	}
}

func TestRouterRejectsNonJSONBufferedResponse(t *testing.T) {
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/plain")
		_, _ = io.WriteString(writer, responsesResponse())
	}))
	defer provider.Close()
	router := newTestRouter(t, provider.URL, testBodyLimit)

	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"openai-chat","messages":[{"role":"user","content":"hello"}]}`))
	request.Header.Set("Authorization", "Bearer client-key")
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	if response.Code != http.StatusBadGateway || !strings.Contains(response.Body.String(), "upstream_invalid_content_type") {
		t.Fatalf("response = %d %s", response.Code, response.Body.String())
	}
}

func TestRouterTimesOutBufferedUpstreamBody(t *testing.T) {
	body := newBlockingBody()
	doer := httpDoerFunc(func(_ *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       body,
		}, nil
	})
	router := newTestRouterWithResponseTimeout(t, "https://provider.example", testBodyLimit, 20*time.Millisecond, doer)
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"openai-chat","messages":[{"role":"user","content":"hello"}]}`))
	request.Header.Set("Authorization", "Bearer client-key")
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)

	if response.Code != http.StatusGatewayTimeout || !strings.Contains(response.Body.String(), "upstream_response_timeout") {
		t.Fatalf("response = %d %s", response.Code, response.Body.String())
	}
	waitForSignal(t, body.started, "upstream body read did not start")
	waitForSignal(t, body.closed, "upstream body was not closed after timeout")
}

func TestRouterDoesNotFollowUpstreamRedirects(t *testing.T) {
	var redirected atomic.Int64
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path == "/redirected" {
			redirected.Add(1)
			writer.WriteHeader(http.StatusOK)
			return
		}
		http.Redirect(writer, request, "/redirected", http.StatusTemporaryRedirect)
	}))
	defer provider.Close()
	router := newTestRouter(t, provider.URL, testBodyLimit)

	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"openai-chat","messages":[{"role":"user","content":"hello"}]}`))
	request.Header.Set("Authorization", "Bearer client-key")
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	if response.Code != http.StatusBadGateway || !strings.Contains(response.Body.String(), "upstream_invalid_status") {
		t.Fatalf("response = %d %s", response.Code, response.Body.String())
	}
	if redirected.Load() != 0 {
		t.Fatalf("redirect target calls = %d, want 0", redirected.Load())
	}
}

func TestDiagnosticsHeaderIsBoundedAndDoesNotExposeSourceValues(t *testing.T) {
	diagnostics := make([]transformer.Diagnostic, 0, 20)
	path := strings.Repeat("😀<&", 100)
	for index := 0; index < 20; index++ {
		diagnostics = append(diagnostics, transformer.Diagnostic{
			Severity:    transformer.SeverityWarning,
			Code:        transformer.DiagnosticCode("warning"),
			Message:     strings.Repeat("😀<&", 100),
			Path:        &path,
			SourceValue: json.RawMessage(`{"secret":"must-not-leak"}`),
		})
	}
	diagnostics = append(diagnostics, transformer.Diagnostic{
		Severity: transformer.SeverityError,
		Code:     transformer.DiagnosticCode("final_error"),
		Message:  "final error",
	})

	header := make(http.Header)
	setDiagnosticsHeader(header, diagnostics)
	encoded := header.Get(diagnosticsTrailer)
	if encoded == "" || len(encoded) > maxDiagnosticsHeader {
		t.Fatalf("diagnostics header length = %d", len(encoded))
	}
	raw, err := base64.RawURLEncoding.DecodeString(encoded)
	if err != nil {
		t.Fatal(err)
	}
	var publicDiagnostics []transformer.Diagnostic
	if err := json.Unmarshal(raw, &publicDiagnostics); err != nil {
		t.Fatal(err)
	}
	if len(publicDiagnostics) > maxGatewayDiagnostics {
		t.Fatalf("diagnostics count = %d", len(publicDiagnostics))
	}
	foundError := false
	for _, diagnostic := range publicDiagnostics {
		if len(diagnostic.SourceValue) != 0 {
			t.Fatalf("source value leaked: %s", diagnostic.SourceValue)
		}
		if diagnostic.Code == "final_error" {
			foundError = true
		}
	}
	if !foundError {
		t.Fatalf("bounded diagnostics dropped the error: %#v", publicDiagnostics)
	}
}

func newTestRouter(t *testing.T, providerURL string, limit int64) http.Handler {
	return newTestRouterWithLimits(t, providerURL, limit, testBodyLimit, nil)
}

func newTestRouterWithDoer(t *testing.T, providerURL string, limit int64, doer upstream.HTTPDoer) http.Handler {
	return newTestRouterWithLimits(t, providerURL, limit, testBodyLimit, doer)
}

func newTestRouterWithLimits(t *testing.T, providerURL string, bodyLimit, responseLimit int64, doer upstream.HTTPDoer) http.Handler {
	return newTestRouterWithConfiguredLimits(t, providerURL, Limits{MaxBodyBytes: bodyLimit, MaxStreamBytes: responseLimit}, doer)
}

func newTestRouterWithResponseTimeout(t *testing.T, providerURL string, limit int64, timeout time.Duration, doer upstream.HTTPDoer) http.Handler {
	return newTestRouterWithConfiguredLimits(t, providerURL, Limits{MaxBodyBytes: limit, MaxStreamBytes: testBodyLimit, ResponseBodyTimeout: timeout}, doer)
}

func newTestRouterWithConfiguredLimits(t *testing.T, providerURL string, limits Limits, doer upstream.HTTPDoer) http.Handler {
	t.Helper()
	service := newTestTransformer(t)
	client, err := upstream.New(upstream.Settings{
		ResponseHeaderTimeout: time.Second,
		Upstreams: map[string]upstream.Config{
			"openai": {
				Endpoint: capabilities.EndpointResponses,
				URL:      providerURL + "/v1/responses",
			},
			"anthropic": {
				Endpoint: capabilities.EndpointMessages,
				URL:      providerURL + "/v1/messages",
			},
		},
		Routes: map[string]string{
			"openai-chat":    "openai",
			"anthropic-chat": "anthropic",
		},
	}, doer)
	if err != nil {
		t.Fatal(err)
	}
	router, err := NewRouter(service, client, limits, zap.NewNop())
	if err != nil {
		t.Fatal(err)
	}
	return router
}

func newPromptCacheTestRouter(
	t *testing.T,
	providerURL string,
	mode transformer.Mode,
	profiles []transformer.CapabilityProfile,
	routes []transformer.ModelRoute,
	upstreamRoutes map[string]string,
	doer upstream.HTTPDoer,
) http.Handler {
	t.Helper()
	defaultMaxTokens := 1024
	service, err := transformer.New(transformer.Config{
		Mode:                   mode,
		DefaultMaxOutputTokens: &defaultMaxTokens,
		Profiles:               profiles,
		Routes:                 routes,
	})
	if err != nil {
		t.Fatal(err)
	}
	client, err := upstream.New(upstream.Settings{
		ResponseHeaderTimeout: time.Second,
		Upstreams: map[string]upstream.Config{
			"openai": {
				Endpoint: capabilities.EndpointResponses,
				URL:      providerURL + "/v1/responses",
			},
			"anthropic": {
				Endpoint: capabilities.EndpointMessages,
				URL:      providerURL + "/v1/messages",
			},
		},
		Routes: upstreamRoutes,
	}, doer)
	if err != nil {
		t.Fatal(err)
	}
	router, err := NewRouter(service, client, Limits{MaxBodyBytes: testBodyLimit, MaxStreamBytes: testBodyLimit}, zap.NewNop())
	if err != nil {
		t.Fatal(err)
	}
	return router
}

func promptCacheTestConfiguration() ([]transformer.CapabilityProfile, []transformer.ModelRoute, map[string]string) {
	profiles := []transformer.CapabilityProfile{
		{
			Provider:    transformer.ProviderAnthropic,
			Endpoint:    transformer.EndpointMessages,
			Model:       "anthropic-cache-target",
			Content:     transformer.ContentCapabilities{Text: true},
			PromptCache: transformer.PromptCacheCapabilities{Mode: transformer.PromptCacheAnthropic},
		},
		{
			Provider: transformer.ProviderOpenAI,
			Endpoint: transformer.EndpointResponses,
			Model:    "openai-legacy-cache-target",
			Content:  transformer.ContentCapabilities{Text: true},
			PromptCache: transformer.PromptCacheCapabilities{
				Mode:                 transformer.PromptCacheOpenAILegacy,
				InMemoryRetention:    true,
				ExtendedRetention24h: true,
			},
		},
		{
			Provider:    transformer.ProviderOpenAI,
			Endpoint:    transformer.EndpointResponses,
			Model:       "openai-56-cache-target",
			Content:     transformer.ContentCapabilities{Text: true},
			PromptCache: transformer.PromptCacheCapabilities{Mode: transformer.PromptCacheOpenAI56},
		},
	}
	routes := []transformer.ModelRoute{
		{Alias: "anthropic-cache", Targets: map[transformer.Endpoint]string{transformer.EndpointMessages: "anthropic-cache-target"}},
		{Alias: "openai-legacy-cache", Targets: map[transformer.Endpoint]string{transformer.EndpointResponses: "openai-legacy-cache-target"}},
		{Alias: "openai-56-cache", Targets: map[transformer.Endpoint]string{transformer.EndpointResponses: "openai-56-cache-target"}},
	}
	upstreamRoutes := map[string]string{
		"anthropic-cache":     "anthropic",
		"openai-legacy-cache": "openai",
		"openai-56-cache":     "openai",
	}
	return profiles, routes, upstreamRoutes
}

func assertAnthropicPromptCacheRequest(t *testing.T, raw json.RawMessage) {
	t.Helper()
	object := decodeTestJSONObject(t, raw)
	assertTestJSONField(t, object, "model", `"anthropic-cache-target"`)
	assertTestJSONField(t, object, "cache_control", `{"type":"ephemeral"}`)
	assertTestJSONFieldAbsent(t, object, "prompt_cache_key")
	assertTestJSONFieldAbsent(t, object, "prompt_cache_options")
	assertTestJSONFieldAbsent(t, object, "prompt_cache_retention")

	messages := decodeTestJSONArray(t, requireTestJSONField(t, object, "messages"))
	message := decodeTestJSONObject(t, requireTestJSONArrayElement(t, messages, 0))
	content := decodeTestJSONArray(t, requireTestJSONField(t, message, "content"))
	part := decodeTestJSONObject(t, requireTestJSONArrayElement(t, content, 0))
	assertTestJSONField(t, part, "cache_control", `{"type":"ephemeral"}`)
	assertTestJSONFieldAbsent(t, part, "prompt_cache_breakpoint")

	tools := decodeTestJSONArray(t, requireTestJSONField(t, object, "tools"))
	tool := decodeTestJSONObject(t, requireTestJSONArrayElement(t, tools, 0))
	assertTestJSONField(t, tool, "cache_control", `{"type":"ephemeral"}`)
}

func assertOpenAILegacyPromptCacheRequest(t *testing.T, raw json.RawMessage) {
	t.Helper()
	object := decodeTestJSONObject(t, raw)
	assertTestJSONField(t, object, "model", `"openai-legacy-cache-target"`)
	assertTestJSONField(t, object, "prompt_cache_key", `"tenant:legacy"`)
	assertTestJSONField(t, object, "prompt_cache_retention", `"in_memory"`)
	assertTestJSONFieldAbsent(t, object, "prompt_cache_options")
	assertTestJSONFieldAbsent(t, object, "cache_control")
}

func assertOpenAI56PromptCacheRequest(t *testing.T, raw json.RawMessage) {
	t.Helper()
	object := decodeTestJSONObject(t, raw)
	assertTestJSONField(t, object, "model", `"openai-56-cache-target"`)
	assertTestJSONField(t, object, "prompt_cache_key", `"tenant:56"`)
	assertTestJSONField(t, object, "prompt_cache_options", `{"mode":"explicit","ttl":"30m"}`)
	assertTestJSONFieldAbsent(t, object, "prompt_cache_retention")
	assertTestJSONFieldAbsent(t, object, "cache_control")

	input := decodeTestJSONArray(t, requireTestJSONField(t, object, "input"))
	message := decodeTestJSONObject(t, requireTestJSONArrayElement(t, input, 0))
	content := decodeTestJSONArray(t, requireTestJSONField(t, message, "content"))
	part := decodeTestJSONObject(t, requireTestJSONArrayElement(t, content, 0))
	assertTestJSONField(t, part, "prompt_cache_breakpoint", `{"mode":"explicit"}`)
	assertTestJSONFieldAbsent(t, part, "cache_control")

	tools := decodeTestJSONArray(t, requireTestJSONField(t, object, "tools"))
	tool := decodeTestJSONObject(t, requireTestJSONArrayElement(t, tools, 0))
	assertTestJSONFieldAbsent(t, tool, "cache_control")
}

func decodeTestJSONObject(t *testing.T, raw json.RawMessage) map[string]json.RawMessage {
	t.Helper()
	var object map[string]json.RawMessage
	if err := json.Unmarshal(raw, &object); err != nil {
		t.Fatalf("decode JSON object %s: %v", raw, err)
	}
	if object == nil {
		t.Fatalf("JSON value is not an object: %s", raw)
	}
	return object
}

func decodeTestJSONArray(t *testing.T, raw json.RawMessage) []json.RawMessage {
	t.Helper()
	var values []json.RawMessage
	if err := json.Unmarshal(raw, &values); err != nil {
		t.Fatalf("decode JSON array %s: %v", raw, err)
	}
	if values == nil {
		t.Fatalf("JSON value is not an array: %s", raw)
	}
	return values
}

func requireTestJSONField(t *testing.T, object map[string]json.RawMessage, name string) json.RawMessage {
	t.Helper()
	raw, exists := object[name]
	if !exists {
		t.Fatalf("JSON field %q is missing in %#v", name, object)
	}
	return raw
}

func requireTestJSONArrayElement(t *testing.T, values []json.RawMessage, index int) json.RawMessage {
	t.Helper()
	if index < 0 || index >= len(values) {
		t.Fatalf("JSON array index %d is missing in %#v", index, values)
	}
	return values[index]
}

func assertTestJSONField(t *testing.T, object map[string]json.RawMessage, name, want string) {
	t.Helper()
	got := normalizeTestJSON(t, requireTestJSONField(t, object, name))
	want = normalizeTestJSON(t, json.RawMessage(want))
	if got != want {
		t.Fatalf("JSON field %q = %s, want %s", name, got, want)
	}
}

func assertTestJSONFieldAbsent(t *testing.T, object map[string]json.RawMessage, name string) {
	t.Helper()
	if raw, exists := object[name]; exists {
		t.Fatalf("JSON field %q leaked with value %s", name, raw)
	}
}

func normalizeTestJSON(t *testing.T, raw json.RawMessage) string {
	t.Helper()
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		t.Fatalf("decode JSON %s: %v", raw, err)
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("encode normalized JSON: %v", err)
	}
	return string(encoded)
}

func decodeTestDiagnosticsHeader(t *testing.T, encoded string) []transformer.Diagnostic {
	t.Helper()
	if encoded == "" {
		t.Fatal("diagnostics header is missing")
	}
	raw, err := base64.RawURLEncoding.DecodeString(encoded)
	if err != nil {
		t.Fatal(err)
	}
	var diagnostics []transformer.Diagnostic
	if err := json.Unmarshal(raw, &diagnostics); err != nil {
		t.Fatal(err)
	}
	return diagnostics
}

func assertTestDiagnostic(t *testing.T, diagnostics []transformer.Diagnostic, severity transformer.Severity, code transformer.DiagnosticCode) {
	t.Helper()
	for _, diagnostic := range diagnostics {
		if diagnostic.Severity == severity && diagnostic.Code == code {
			return
		}
	}
	t.Fatalf("diagnostic %s/%s is missing in %#v", severity, code, diagnostics)
}

type chatCacheUsage struct {
	PromptTokens        int64 `json:"prompt_tokens"`
	CompletionTokens    int64 `json:"completion_tokens"`
	TotalTokens         int64 `json:"total_tokens"`
	PromptTokensDetails *struct {
		CachedTokens     *int64 `json:"cached_tokens"`
		CacheWriteTokens *int64 `json:"cache_write_tokens"`
	} `json:"prompt_tokens_details"`
}

func assertChatCacheUsage(t *testing.T, usage *chatCacheUsage, prompt, completion, total, cached, cacheWrite int64) {
	t.Helper()
	if usage == nil {
		t.Fatal("usage is missing")
	}
	if usage.PromptTokens != prompt || usage.CompletionTokens != completion || usage.TotalTokens != total {
		t.Fatalf("usage = %#v, want prompt=%d completion=%d total=%d", usage, prompt, completion, total)
	}
	if usage.PromptTokensDetails == nil || usage.PromptTokensDetails.CachedTokens == nil || usage.PromptTokensDetails.CacheWriteTokens == nil {
		t.Fatalf("cache usage details are missing: %#v", usage)
	}
	if *usage.PromptTokensDetails.CachedTokens != cached || *usage.PromptTokensDetails.CacheWriteTokens != cacheWrite {
		t.Fatalf("cache usage details = %#v, want cached=%d write=%d", usage.PromptTokensDetails, cached, cacheWrite)
	}
}

func decodeTestSSEUsage(t *testing.T, stream string) (*chatCacheUsage, bool) {
	t.Helper()
	var found *chatCacheUsage
	for _, line := range strings.Split(stream, "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			continue
		}
		var frame struct {
			Usage json.RawMessage `json:"usage"`
		}
		if err := json.Unmarshal([]byte(payload), &frame); err != nil {
			t.Fatalf("decode Chat SSE frame: %v", err)
		}
		trimmed := strings.TrimSpace(string(frame.Usage))
		if trimmed == "" || trimmed == "null" {
			continue
		}
		var usage chatCacheUsage
		if err := json.Unmarshal(frame.Usage, &usage); err != nil {
			t.Fatalf("decode Chat SSE usage: %v", err)
		}
		found = &usage
	}
	return found, found != nil
}

type httpDoerFunc func(*http.Request) (*http.Response, error)

func (function httpDoerFunc) Do(request *http.Request) (*http.Response, error) {
	return function(request)
}

type chunkReadCloser struct {
	chunks [][]byte
}

func (body *chunkReadCloser) Read(target []byte) (int, error) {
	if len(body.chunks) == 0 {
		return 0, io.EOF
	}
	chunk := body.chunks[0]
	body.chunks = body.chunks[1:]
	return copy(target, chunk), nil
}

func (*chunkReadCloser) Close() error {
	return nil
}

type blockingBody struct {
	started   chan struct{}
	closed    chan struct{}
	startOnce sync.Once
	closeOnce sync.Once
}

func newBlockingBody() *blockingBody {
	return &blockingBody{started: make(chan struct{}), closed: make(chan struct{})}
}

func (body *blockingBody) Read([]byte) (int, error) {
	body.startOnce.Do(func() { close(body.started) })
	<-body.closed
	return 0, io.ErrClosedPipe
}

func (body *blockingBody) Close() error {
	body.closeOnce.Do(func() { close(body.closed) })
	return nil
}

func waitForSignal(t *testing.T, signal <-chan struct{}, failure string) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(time.Second):
		t.Fatal(failure)
	}
}

func newTestTransformer(t *testing.T) *transformer.Transformer {
	t.Helper()
	defaultMaxTokens := 1024
	service, err := transformer.New(transformer.Config{
		DefaultMaxOutputTokens: &defaultMaxTokens,
		Profiles: []transformer.CapabilityProfile{
			{
				Provider: transformer.ProviderOpenAI,
				Endpoint: transformer.EndpointResponses,
				Model:    "openai-target",
				Content:  transformer.ContentCapabilities{Text: true},
			},
			{
				Provider: transformer.ProviderAnthropic,
				Endpoint: transformer.EndpointMessages,
				Model:    "anthropic-target",
				Content:  transformer.ContentCapabilities{Text: true},
			},
		},
		Routes: []transformer.ModelRoute{
			{Alias: "openai-chat", Targets: map[transformer.Endpoint]string{transformer.EndpointResponses: "openai-target"}},
			{Alias: "anthropic-chat", Targets: map[transformer.Endpoint]string{transformer.EndpointMessages: "anthropic-target"}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return service
}

func responsesResponse() string {
	return `{"id":"resp_1","created_at":123,"model":"gpt-test","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]}]}]}`
}

func anthropicResponse() string {
	return `{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":3}}`
}

func responsesCacheUsageResponse() string {
	return `{"id":"resp_cache","created_at":123,"model":"gpt-test","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]}]}],"usage":{"input_tokens":12,"output_tokens":3,"total_tokens":15,"input_tokens_details":{"cached_tokens":5,"cache_write_tokens":4}}}`
}

func anthropicCacheUsageResponse() string {
	return `{"id":"msg_cache","type":"message","role":"assistant","model":"claude-test","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":2,"cache_creation_input_tokens":4,"cache_read_input_tokens":5}}`
}

func responsesTextStream() string {
	return responsesFrame("response.created", `{"type":"response.created","response":{"id":"resp_1","created_at":123,"model":"gpt-test","status":"in_progress","output":[]}}`) +
		responsesFrame("response.output_text.delta", `{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"hello"}`) +
		responsesFrame("response.output_text.done", `{"type":"response.output_text.done","item_id":"msg_1","output_index":0,"content_index":0,"text":"hello"}`) +
		responsesFrame("response.completed", `{"type":"response.completed","response":{"id":"resp_1","created_at":123,"model":"gpt-test","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]}]}]}}`)
}

func responsesCacheUsageStream() string {
	return responsesFrame("response.created", `{"type":"response.created","response":{"id":"resp_cache","created_at":123,"model":"gpt-test","status":"in_progress","output":[]}}`) +
		responsesFrame("response.output_text.delta", `{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"hello"}`) +
		responsesFrame("response.output_text.done", `{"type":"response.output_text.done","item_id":"msg_1","output_index":0,"content_index":0,"text":"hello"}`) +
		responsesFrame("response.completed", `{"type":"response.completed","response":{"id":"resp_cache","created_at":123,"model":"gpt-test","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]}]}],"usage":{"input_tokens":12,"output_tokens":3,"total_tokens":15,"input_tokens_details":{"cached_tokens":5,"cache_write_tokens":4}}}}`)
}

func responsesFrame(name, value string) string {
	return "event: " + name + "\ndata: " + value + "\n\n"
}

func anthropicTextStream() string {
	frames := []string{
		`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":2,"output_tokens":0}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}`,
		`{"type":"message_stop"}`,
	}
	var result strings.Builder
	for _, frame := range frames {
		var event struct {
			Type string `json:"type"`
		}
		_ = json.Unmarshal([]byte(frame), &event)
		result.WriteString("event: ")
		result.WriteString(event.Type)
		result.WriteString("\ndata: ")
		result.WriteString(frame)
		result.WriteString("\n\n")
	}
	return result.String()
}

func anthropicCacheUsageStream() string {
	frames := []string{
		`{"type":"message_start","message":{"id":"msg_cache","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":0,"cache_creation_input_tokens":4,"cache_read_input_tokens":5}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}`,
		`{"type":"message_stop"}`,
	}
	var result strings.Builder
	for _, frame := range frames {
		var event struct {
			Type string `json:"type"`
		}
		_ = json.Unmarshal([]byte(frame), &event)
		result.WriteString("event: ")
		result.WriteString(event.Type)
		result.WriteString("\ndata: ")
		result.WriteString(frame)
		result.WriteString("\n\n")
	}
	return result.String()
}
