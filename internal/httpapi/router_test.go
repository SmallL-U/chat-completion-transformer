package httpapi

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"chat-completion-transformer/pkg/transformer"
)

const testBodyLimit = 1 << 20

func TestRouterHealth(t *testing.T) {
	router := newTestRouter(t, newTestTransformer(t, nil), Limits{MaxBodyBytes: testBodyLimit, MaxStreamBytes: testBodyLimit})
	request := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	response := httptest.NewRecorder()

	router.ServeHTTP(response, request)

	if response.Code != http.StatusOK || strings.TrimSpace(response.Body.String()) != `{"status":"ok"}` {
		t.Fatalf("response = %d %s", response.Code, response.Body.String())
	}
}

func TestRouterJSONTransforms(t *testing.T) {
	service := newTestTransformer(t, nil)
	router := newTestRouter(t, service, Limits{MaxBodyBytes: testBodyLimit, MaxStreamBytes: testBodyLimit})
	tests := []struct {
		name        string
		path        string
		body        string
		assertValue func(*testing.T, json.RawMessage)
	}{
		{
			name: "chat request to Responses",
			path: "/v1/transform/chat-completions/to/openai-responses",
			body: `{"model":"general","messages":[{"role":"user","content":"hello"}],"max_completion_tokens":12}`,
			assertValue: func(t *testing.T, raw json.RawMessage) {
				t.Helper()
				assertJSONField(t, raw, "model", "openai-target")
			},
		},
		{
			name: "chat request to Anthropic",
			path: "/v1/transform/chat-completions/to/anthropic-messages",
			body: `{"model":"general","messages":[{"role":"user","content":"hello"}],"max_completion_tokens":12}`,
			assertValue: func(t *testing.T, raw json.RawMessage) {
				t.Helper()
				assertJSONField(t, raw, "model", "anthropic-target")
			},
		},
		{
			name: "Responses response to chat",
			path: "/v1/transform/openai-responses/to/chat-completions",
			body: `{"id":"resp_1","created_at":123,"model":"gpt-test","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]}]}]}`,
			assertValue: func(t *testing.T, raw json.RawMessage) {
				t.Helper()
				assertJSONField(t, raw, "object", "chat.completion")
			},
		},
		{
			name: "Anthropic response to chat",
			path: "/v1/transform/anthropic-messages/to/chat-completions",
			body: `{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":3}}`,
			assertValue: func(t *testing.T, raw json.RawMessage) {
				t.Helper()
				assertJSONField(t, raw, "object", "chat.completion")
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodPost, test.path, strings.NewReader(test.body))
			response := httptest.NewRecorder()
			router.ServeHTTP(response, request)

			if response.Code != http.StatusOK {
				t.Fatalf("response = %d %s", response.Code, response.Body.String())
			}
			var result struct {
				OK          bool                     `json:"ok"`
				Lossless    bool                     `json:"lossless"`
				Value       json.RawMessage          `json:"value"`
				Diagnostics []transformer.Diagnostic `json:"diagnostics"`
			}
			if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
			if !result.OK || result.Value == nil || result.Diagnostics == nil {
				t.Fatalf("result = %#v", result)
			}
			test.assertValue(t, result.Value)
		})
	}
}

func TestRouterRejectsOversizedJSONBody(t *testing.T) {
	router := newTestRouter(t, newTestTransformer(t, nil), Limits{MaxBodyBytes: 8, MaxStreamBytes: testBodyLimit})
	request := httptest.NewRequest(http.MethodPost, "/v1/transform/openai-responses/to/chat-completions", strings.NewReader(`{"more":"than eight bytes"}`))
	response := httptest.NewRecorder()

	router.ServeHTTP(response, request)

	if response.Code != http.StatusRequestEntityTooLarge || !strings.Contains(response.Body.String(), "request_body_too_large") {
		t.Fatalf("response = %d %s", response.Code, response.Body.String())
	}
}

func TestRouterRejectsUnsupportedFullDuplex(t *testing.T) {
	router := newTestRouter(t, newTestTransformer(t, nil), Limits{MaxBodyBytes: testBodyLimit, MaxStreamBytes: testBodyLimit})
	request := httptest.NewRequest(http.MethodPost, "/v1/transform/openai-responses/sse/to/chat-completions", strings.NewReader(responsesTextStream()))
	response := httptest.NewRecorder()

	router.ServeHTTP(response, request)

	if response.Code != http.StatusInternalServerError || !strings.Contains(response.Body.String(), "full_duplex_unavailable") {
		t.Fatalf("response = %d %s", response.Code, response.Body.String())
	}
}

func TestRouterStreamsArbitraryChunksAndFlushes(t *testing.T) {
	service := newTestTransformer(t, nil)
	router := newTestRouter(t, service, Limits{MaxBodyBytes: testBodyLimit, MaxStreamBytes: testBodyLimit})
	tests := []struct {
		name string
		path string
		body string
	}{
		{
			name: "Responses",
			path: "/v1/transform/openai-responses/sse/to/chat-completions",
			body: responsesTextStream(),
		},
		{
			name: "Anthropic",
			path: "/v1/transform/anthropic-messages/sse/to/chat-completions",
			body: anthropicTextStream(),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodPost, test.path, io.NopCloser(&oneByteReader{value: []byte(test.body)}))
			response := newFullDuplexRecorder()
			router.ServeHTTP(response, request)

			result := response.Result()
			if result.StatusCode != http.StatusOK {
				t.Fatalf("response = %d %s", result.StatusCode, response.Body.String())
			}
			if result.Header.Get("Content-Type") != "text/event-stream" || !response.Flushed {
				t.Fatalf("headers = %#v, flushed = %v", result.Header, response.Flushed)
			}
			if !strings.Contains(response.Body.String(), `"content":"hello"`) || !strings.HasSuffix(response.Body.String(), "data: [DONE]\n\n") {
				t.Fatalf("stream = %s", response.Body.String())
			}
			assertDiagnosticsTrailer(t, result.Trailer)
		})
	}
}

func TestRouterStreamsBeforeRequestEOFOverHTTP1(t *testing.T) {
	router := newTestRouter(t, newTestTransformer(t, nil), Limits{MaxBodyBytes: testBodyLimit, MaxStreamBytes: testBodyLimit})
	server := httptest.NewServer(router)
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	bodyReader, bodyWriter := io.Pipe()
	defer bodyReader.Close()
	defer bodyWriter.Close()

	request, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		server.URL+"/v1/transform/openai-responses/sse/to/chat-completions",
		bodyReader,
	)
	if err != nil {
		t.Fatal(err)
	}
	type responseResult struct {
		response *http.Response
		err      error
	}
	responseResults := make(chan responseResult, 1)
	go func() {
		response, requestErr := server.Client().Do(request)
		responseResults <- responseResult{response: response, err: requestErr}
	}()

	stream := responsesTextStream()
	firstFrameEnd := strings.Index(stream, "\n\n") + 2
	if firstFrameEnd < 2 {
		t.Fatal("test stream has no complete first frame")
	}
	if _, err := bodyWriter.Write([]byte(stream[:firstFrameEnd])); err != nil {
		t.Fatal(err)
	}

	var response *http.Response
	select {
	case result := <-responseResults:
		if result.err != nil {
			t.Fatal(result.err)
		}
		response = result.response
	case <-ctx.Done():
		t.Fatal("response headers were not flushed before request EOF")
	}
	defer response.Body.Close()
	if response.ProtoMajor != 1 || response.StatusCode != http.StatusOK {
		t.Fatalf("response = %s %d", response.Proto, response.StatusCode)
	}
	if response.Header.Get("Content-Type") != "text/event-stream" {
		t.Fatalf("Content-Type = %q", response.Header.Get("Content-Type"))
	}

	responseReader := bufio.NewReader(response.Body)
	firstLine, err := responseReader.ReadString('\n')
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(firstLine, "data: ") {
		t.Fatalf("first streamed line = %q", firstLine)
	}

	if _, err := bodyWriter.Write([]byte(stream[firstFrameEnd:])); err != nil {
		t.Fatal(err)
	}
	if err := bodyWriter.Close(); err != nil {
		t.Fatal(err)
	}
	remaining, err := io.ReadAll(responseReader)
	if err != nil {
		t.Fatal(err)
	}
	output := firstLine + string(remaining)
	if !strings.Contains(output, `"content":"hello"`) || !strings.HasSuffix(output, "data: [DONE]\n\n") {
		t.Fatalf("stream = %s", output)
	}
	assertDiagnosticsTrailer(t, response.Trailer)
}

func TestRouterMovesStreamErrorsAfterFirstFrameToTrailer(t *testing.T) {
	router := newTestRouter(t, newTestTransformer(t, nil), Limits{MaxBodyBytes: testBodyLimit, MaxStreamBytes: testBodyLimit})
	body := responsesFrame("response.created", `{"type":"response.created","response":{"id":"resp_1","created_at":1,"model":"gpt-test","status":"in_progress","output":[]}}`) +
		responsesFrame("response.output_text.delta", `{"type":"response.output_text.delta","output_index":"invalid","content_index":0,"delta":"x"}`)
	request := httptest.NewRequest(http.MethodPost, "/v1/transform/openai-responses/sse/to/chat-completions", io.NopCloser(&oneByteReader{value: []byte(body)}))
	response := newFullDuplexRecorder()

	router.ServeHTTP(response, request)

	result := response.Result()
	if result.StatusCode != http.StatusOK || response.Body.Len() == 0 {
		t.Fatalf("response = %d %s", result.StatusCode, response.Body.String())
	}
	diagnostics := assertDiagnosticsTrailer(t, result.Trailer)
	if len(diagnostics) == 0 || diagnostics[len(diagnostics)-1].Severity != transformer.SeverityError {
		t.Fatalf("diagnostics = %#v", diagnostics)
	}
}

func TestRouterPropagatesCancellationToResolver(t *testing.T) {
	resolver := &cancelResolver{started: make(chan struct{}), canceled: make(chan struct{})}
	router := newTestRouter(t, newTestTransformer(t, resolver), Limits{MaxBodyBytes: testBodyLimit, MaxStreamBytes: testBodyLimit})
	ctx, cancel := context.WithCancel(context.Background())
	request := httptest.NewRequest(http.MethodPost, "/v1/transform/chat-completions/to/openai-responses", strings.NewReader(`{
		"model":"general",
		"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]}]
	}`)).WithContext(ctx)
	done := make(chan struct{})
	go func() {
		router.ServeHTTP(newFullDuplexRecorder(), request)
		close(done)
	}()

	waitForSignal(t, resolver.started, "resolver did not start")
	cancel()
	waitForSignal(t, done, "handler did not return after cancellation")
	waitForSignal(t, resolver.canceled, "resolver did not observe request cancellation")
}

func TestRouterCancellationUnblocksStreamRead(t *testing.T) {
	body := newBlockingBody()
	router := newTestRouter(t, newTestTransformer(t, nil), Limits{MaxBodyBytes: testBodyLimit, MaxStreamBytes: testBodyLimit})
	ctx, cancel := context.WithCancel(context.Background())
	request := httptest.NewRequest(http.MethodPost, "/v1/transform/openai-responses/sse/to/chat-completions", body).WithContext(ctx)
	done := make(chan struct{})
	go func() {
		router.ServeHTTP(newFullDuplexRecorder(), request)
		close(done)
	}()

	waitForSignal(t, body.started, "request body read did not start")
	cancel()
	waitForSignal(t, body.closed, "request body was not closed on cancellation")
	waitForSignal(t, done, "handler did not return after cancellation")
}

func newTestRouter(t *testing.T, service *transformer.Transformer, limits Limits) http.Handler {
	t.Helper()
	router, err := NewRouter(service, limits)
	if err != nil {
		t.Fatal(err)
	}
	return router
}

func newTestTransformer(t *testing.T, resolver transformer.AssetResolver) *transformer.Transformer {
	t.Helper()
	service, err := transformer.New(transformer.Config{
		Resolver: resolver,
		Profiles: []transformer.CapabilityProfile{
			{
				Provider: transformer.ProviderOpenAI,
				Endpoint: transformer.EndpointResponses,
				Model:    "openai-target",
				Images:   transformer.ImageCapabilities{URL: true},
				Content:  transformer.ContentCapabilities{Text: true, Image: true},
			},
			{
				Provider: transformer.ProviderAnthropic,
				Endpoint: transformer.EndpointMessages,
				Model:    "anthropic-target",
				Images:   transformer.ImageCapabilities{URL: true},
				Content:  transformer.ContentCapabilities{Text: true, Image: true},
			},
		},
		Routes: []transformer.ModelRoute{{
			Alias: "general",
			Targets: map[transformer.Endpoint]string{
				transformer.EndpointResponses: "openai-target",
				transformer.EndpointMessages:  "anthropic-target",
			},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	return service
}

func assertJSONField(t *testing.T, raw json.RawMessage, key, want string) {
	t.Helper()
	var object map[string]any
	if err := json.Unmarshal(raw, &object); err != nil {
		t.Fatal(err)
	}
	if object[key] != want {
		t.Fatalf("%s = %#v, want %q in %s", key, object[key], want, raw)
	}
}

func assertDiagnosticsTrailer(t *testing.T, trailer http.Header) []transformer.Diagnostic {
	t.Helper()
	value := trailer.Get(diagnosticsTrailer)
	if value == "" {
		t.Fatal("diagnostics trailer is missing")
	}
	decoded, err := base64.RawURLEncoding.DecodeString(value)
	if err != nil {
		t.Fatal(err)
	}
	var diagnostics []transformer.Diagnostic
	if err := json.Unmarshal(decoded, &diagnostics); err != nil {
		t.Fatal(err)
	}
	if diagnostics == nil {
		t.Fatal("diagnostics trailer must contain a JSON array")
	}
	return diagnostics
}

func responsesTextStream() string {
	return responsesFrame("response.created", `{"type":"response.created","response":{"id":"resp_1","created_at":123,"model":"gpt-test","status":"in_progress","output":[]}}`) +
		responsesFrame("response.output_text.delta", `{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"hello"}`) +
		responsesFrame("response.output_text.done", `{"type":"response.output_text.done","item_id":"msg_1","output_index":0,"content_index":0,"text":"hello"}`) +
		responsesFrame("response.completed", `{"type":"response.completed","response":{"id":"resp_1","created_at":123,"model":"gpt-test","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]}]}]}}`)
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

type fullDuplexRecorder struct {
	*httptest.ResponseRecorder
}

func newFullDuplexRecorder() *fullDuplexRecorder {
	return &fullDuplexRecorder{ResponseRecorder: httptest.NewRecorder()}
}

func (*fullDuplexRecorder) EnableFullDuplex() error {
	return nil
}

type oneByteReader struct {
	value []byte
}

func (r *oneByteReader) Read(target []byte) (int, error) {
	if len(r.value) == 0 {
		return 0, io.EOF
	}
	target[0] = r.value[0]
	r.value = r.value[1:]
	return 1, nil
}

type cancelResolver struct {
	started  chan struct{}
	canceled chan struct{}
}

func (r *cancelResolver) ResolveForResponses(ctx context.Context, _ transformer.AssetSource) (transformer.ResolvedAsset, error) {
	close(r.started)
	<-ctx.Done()
	close(r.canceled)
	return transformer.ResolvedAsset{}, ctx.Err()
}

func (r *cancelResolver) ResolveForAnthropic(ctx context.Context, source transformer.AssetSource) (transformer.ResolvedAsset, error) {
	return r.ResolveForResponses(ctx, source)
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

func (b *blockingBody) Read([]byte) (int, error) {
	b.startOnce.Do(func() { close(b.started) })
	<-b.closed
	return 0, io.ErrClosedPipe
}

func (b *blockingBody) Close() error {
	b.closeOnce.Do(func() { close(b.closed) })
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
