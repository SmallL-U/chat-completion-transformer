package upstream

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"time"

	"chat-completion-transformer/internal/capabilities"
)

type recordingDoer struct {
	requests []*http.Request
}

func (d *recordingDoer) Do(request *http.Request) (*http.Response, error) {
	d.requests = append(d.requests, request.Clone(context.Background()))
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     make(http.Header),
		Body:       http.NoBody,
	}, nil
}

type doerFunc func(*http.Request) (*http.Response, error)

func (f doerFunc) Do(request *http.Request) (*http.Response, error) {
	return f(request)
}

func TestSettingsValidateRejectsInvalidSettings(t *testing.T) {
	tests := []struct {
		name     string
		settings Settings
	}{
		{
			name: "zero timeout",
			settings: Settings{
				ResponseHeaderTimeout: 0,
			},
		},
		{
			name: "negative timeout",
			settings: Settings{
				ResponseHeaderTimeout: -time.Second,
			},
		},
		{
			name: "relative URL",
			settings: Settings{
				ResponseHeaderTimeout: time.Second,
				Upstreams: map[string]Config{
					"provider": {
						Endpoint: capabilities.EndpointResponses,
						URL:      "v1/responses",
					},
				},
			},
		},
		{
			name: "unsupported URL scheme",
			settings: Settings{
				ResponseHeaderTimeout: time.Second,
				Upstreams: map[string]Config{
					"provider": {
						Endpoint: capabilities.EndpointResponses,
						URL:      "ftp://provider.example/v1/responses",
					},
				},
			},
		},
		{
			name: "insecure remote URL",
			settings: Settings{
				ResponseHeaderTimeout: time.Second,
				Upstreams: map[string]Config{
					"provider": {
						Endpoint: capabilities.EndpointResponses,
						URL:      "http://provider.example/v1/responses",
					},
				},
			},
		},
		{
			name: "URL user information",
			settings: Settings{
				ResponseHeaderTimeout: time.Second,
				Upstreams: map[string]Config{
					"provider": {
						Endpoint: capabilities.EndpointResponses,
						URL:      "https://user:secret@provider.example/v1/responses",
					},
				},
			},
		},
		{
			name: "URL query",
			settings: Settings{
				ResponseHeaderTimeout: time.Second,
				Upstreams: map[string]Config{
					"provider": {
						Endpoint: capabilities.EndpointResponses,
						URL:      "https://provider.example/v1/responses?api-key=secret",
					},
				},
			},
		},
		{
			name: "URL fragment",
			settings: Settings{
				ResponseHeaderTimeout: time.Second,
				Upstreams: map[string]Config{
					"provider": {
						Endpoint: capabilities.EndpointResponses,
						URL:      "https://provider.example/v1/responses#fragment",
					},
				},
			},
		},
		{
			name: "unsupported endpoint",
			settings: Settings{
				ResponseHeaderTimeout: time.Second,
				Upstreams: map[string]Config{
					"provider": {
						Endpoint: capabilities.EndpointBedrockMessages,
						URL:      "https://provider.example/v1/messages",
					},
				},
			},
		},
		{
			name: "route references unknown upstream",
			settings: Settings{
				ResponseHeaderTimeout: time.Second,
				Upstreams: map[string]Config{
					"provider": {
						Endpoint: capabilities.EndpointResponses,
						URL:      "https://provider.example/v1/responses",
					},
				},
				Routes: map[string]string{"general": "missing"},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := test.settings.Validate(); err == nil {
				t.Fatal("Validate() error = nil, want error")
			}
		})
	}
}

func TestClientDoOpenAIForwardsHeadersWithoutRetainingCredentials(t *testing.T) {
	doer := &recordingDoer{}
	client := newTestClient(t, capabilities.EndpointResponses, doer)
	headers := make(http.Header)
	headers.Set("Authorization", "Bearer request-secret")
	headers.Set("OpenAI-Organization", "org-request")
	headers.Set("OpenAI-Project", "project-request")

	response, err := client.Do(context.Background(), "general", []byte(`{"model":"general"}`), false, headers)
	if err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	_ = response.Body.Close()

	response, err = client.Do(context.Background(), "general", []byte(`{"model":"general"}`), false, nil)
	if err != nil {
		t.Fatalf("second Do() error = %v", err)
	}
	_ = response.Body.Close()

	if len(doer.requests) != 2 {
		t.Fatalf("request count = %d, want 2", len(doer.requests))
	}
	wantHeaders := map[string]string{
		"Authorization":       "Bearer request-secret",
		"OpenAI-Organization": "org-request",
		"OpenAI-Project":      "project-request",
	}
	for name, want := range wantHeaders {
		if got := doer.requests[0].Header.Get(name); got != want {
			t.Errorf("first request %s = %q, want %q", name, got, want)
		}
		if got := doer.requests[1].Header.Get(name); got != "" {
			t.Errorf("second request %s = %q, want empty", name, got)
		}
	}
}

func TestClientDoAnthropicMapsCredentialsAndSetsVersion(t *testing.T) {
	tests := []struct {
		name    string
		headers http.Header
		wantKey string
	}{
		{
			name: "Bearer token",
			headers: http.Header{
				"Authorization": []string{"Bearer bearer-secret"},
			},
			wantKey: "bearer-secret",
		},
		{
			name: "untrusted x-api-key is ignored",
			headers: http.Header{
				"Authorization": []string{"Bearer bearer-secret"},
				"X-Api-Key":     []string{"direct-secret"},
			},
			wantKey: "bearer-secret",
		},
		{
			name: "x-api-key alone is ignored",
			headers: http.Header{
				"X-Api-Key": []string{"direct-secret"},
			},
			wantKey: "",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			doer := &recordingDoer{}
			client := newTestClient(t, capabilities.EndpointMessages, doer)

			response, err := client.Do(context.Background(), "general", []byte(`{"model":"general"}`), false, test.headers)
			if err != nil {
				t.Fatalf("Do() error = %v", err)
			}
			_ = response.Body.Close()

			request := doer.requests[0]
			if got := request.Header.Get("x-api-key"); got != test.wantKey {
				t.Errorf("x-api-key = %q, want %q", got, test.wantKey)
			}
			if got := request.Header.Get("anthropic-version"); got != DefaultAnthropicVersion {
				t.Errorf("anthropic-version = %q, want %q", got, DefaultAnthropicVersion)
			}
			if got := request.Header.Get("Authorization"); got != "" {
				t.Errorf("Authorization = %q, want empty", got)
			}
		})
	}
}

func TestClientDoSetsAcceptForResponseMode(t *testing.T) {
	tests := []struct {
		name   string
		stream bool
		want   string
	}{
		{name: "non-streaming", want: "application/json"},
		{name: "streaming", stream: true, want: "text/event-stream"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			doer := &recordingDoer{}
			client := newTestClient(t, capabilities.EndpointResponses, doer)

			response, err := client.Do(context.Background(), "general", []byte(`{}`), test.stream, nil)
			if err != nil {
				t.Fatalf("Do() error = %v", err)
			}
			_ = response.Body.Close()

			if got := doer.requests[0].Header.Get("Accept"); got != test.want {
				t.Errorf("Accept = %q, want %q", got, test.want)
			}
		})
	}
}

func TestClientDoUnknownAliasReturnsRouteNotFound(t *testing.T) {
	doer := &recordingDoer{}
	client := newTestClient(t, capabilities.EndpointResponses, doer)

	_, err := client.Do(context.Background(), "missing", []byte(`{}`), false, nil)
	if !errors.Is(err, ErrRouteNotFound) {
		t.Fatalf("Do() error = %v, want ErrRouteNotFound", err)
	}
	if len(doer.requests) != 0 {
		t.Fatalf("request count = %d, want 0", len(doer.requests))
	}
}

func TestClientDoCancelsRequestContext(t *testing.T) {
	started := make(chan struct{})
	doer := doerFunc(func(request *http.Request) (*http.Response, error) {
		close(started)
		<-request.Context().Done()
		return nil, request.Context().Err()
	})
	client := newTestClient(t, capabilities.EndpointResponses, doer)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	errCh := make(chan error, 1)
	go func() {
		_, err := client.Do(ctx, "general", []byte(`{}`), false, nil)
		errCh <- err
	}()

	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("upstream request did not start")
	}
	cancel()

	select {
	case err := <-errCh:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Do() error = %v, want context.Canceled", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Do() did not return after context cancellation")
	}
}

func newTestClient(t *testing.T, endpoint capabilities.Endpoint, doer HTTPDoer) *Client {
	t.Helper()

	client, err := New(Settings{
		ResponseHeaderTimeout: time.Second,
		Upstreams: map[string]Config{
			"provider": {
				Endpoint: endpoint,
				URL:      "https://provider.example/v1",
			},
		},
		Routes: map[string]string{"general": "provider"},
	}, doer)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return client
}
