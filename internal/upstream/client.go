// Package upstream routes transformed requests to provider HTTP endpoints.
package upstream

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"chat-completion-transformer/internal/capabilities"
)

const DefaultAnthropicVersion = "2023-06-01"

var ErrRouteNotFound = errors.New("upstream route not found")

// Settings configures named upstreams and model-alias routes.
type Settings struct {
	ResponseHeaderTimeout time.Duration     `json:"response_header_timeout"`
	Upstreams             map[string]Config `json:"upstreams"`
	Routes                map[string]string `json:"routes"`
}

// Config describes one provider HTTP endpoint.
type Config struct {
	Endpoint         capabilities.Endpoint `json:"endpoint"`
	URL              string                `json:"url"`
	AnthropicVersion string                `json:"anthropic_version"`
}

// HTTPDoer is implemented by http.Client.
type HTTPDoer interface {
	Do(*http.Request) (*http.Response, error)
}

// Client resolves model aliases and sends requests to their configured
// provider endpoints.
type Client struct {
	doer      HTTPDoer
	upstreams map[string]resolvedConfig
	routes    map[string]string
}

type resolvedConfig struct {
	endpoint         capabilities.Endpoint
	url              string
	anthropicVersion string
}

// New validates settings and creates an upstream client. When doer is nil, a
// standard HTTP client is created with the configured response-header timeout.
func New(settings Settings, doer HTTPDoer) (*Client, error) {
	if err := settings.Validate(); err != nil {
		return nil, err
	}

	upstreams := make(map[string]resolvedConfig, len(settings.Upstreams))
	for name, config := range settings.Upstreams {
		resolved, err := resolveConfig(name, config)
		if err != nil {
			return nil, err
		}
		upstreams[name] = resolved
	}

	routes := make(map[string]string, len(settings.Routes))
	for alias, upstreamName := range settings.Routes {
		routes[alias] = upstreamName
	}

	if doer == nil {
		doer = newHTTPClient(settings.ResponseHeaderTimeout)
	}

	return &Client{doer: doer, upstreams: upstreams, routes: routes}, nil
}

// Validate checks transport settings without reading credentials. API keys are
// supplied by each incoming Chat Completions request.
func (settings Settings) Validate() error {
	if settings.ResponseHeaderTimeout <= 0 {
		return fmt.Errorf("response_header_timeout must be positive")
	}

	for name, config := range settings.Upstreams {
		if strings.TrimSpace(name) == "" {
			return fmt.Errorf("upstream name is required")
		}
		if _, err := resolveConfig(name, config); err != nil {
			return err
		}
	}

	for alias, upstreamName := range settings.Routes {
		if strings.TrimSpace(alias) == "" {
			return fmt.Errorf("route alias is required")
		}
		if strings.TrimSpace(upstreamName) == "" {
			return fmt.Errorf("route %q requires an upstream", alias)
		}
		if _, exists := settings.Upstreams[upstreamName]; !exists {
			return fmt.Errorf("route %q references unknown upstream %q", alias, upstreamName)
		}
	}

	return nil
}

func resolveConfig(name string, config Config) (resolvedConfig, error) {
	if config.Endpoint != capabilities.EndpointResponses && config.Endpoint != capabilities.EndpointMessages {
		return resolvedConfig{}, fmt.Errorf("upstream %q uses unsupported endpoint %q", name, config.Endpoint)
	}

	rawURL := strings.TrimSpace(config.URL)
	parsedURL, err := url.Parse(rawURL)
	if err != nil || !parsedURL.IsAbs() || parsedURL.Host == "" {
		return resolvedConfig{}, fmt.Errorf("upstream %q URL must be absolute", name)
	}
	if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
		return resolvedConfig{}, fmt.Errorf("upstream %q URL must use http or https", name)
	}
	if parsedURL.User != nil {
		return resolvedConfig{}, fmt.Errorf("upstream %q URL must not contain user information", name)
	}
	if parsedURL.RawQuery != "" || parsedURL.Fragment != "" {
		return resolvedConfig{}, fmt.Errorf("upstream %q URL must not contain a query or fragment", name)
	}
	if parsedURL.Scheme == "http" && !isLoopbackHost(parsedURL.Hostname()) {
		return resolvedConfig{}, fmt.Errorf("upstream %q URL must use https unless it targets loopback", name)
	}

	version := strings.TrimSpace(config.AnthropicVersion)
	if config.Endpoint == capabilities.EndpointMessages && version == "" {
		version = DefaultAnthropicVersion
	}

	return resolvedConfig{
		endpoint:         config.Endpoint,
		url:              rawURL,
		anthropicVersion: version,
	}, nil
}

func isLoopbackHost(host string) bool {
	host = strings.TrimSuffix(strings.ToLower(host), ".")
	if host == "localhost" {
		return true
	}
	address := net.ParseIP(host)
	return address != nil && address.IsLoopback()
}

func newHTTPClient(responseHeaderTimeout time.Duration) *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.ResponseHeaderTimeout = responseHeaderTimeout
	return &http.Client{
		Transport: transport,
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
}

// Endpoint returns the provider endpoint selected for a model alias.
func (c *Client) Endpoint(alias string) (capabilities.Endpoint, error) {
	config, _, err := c.resolve(alias)
	if err != nil {
		return "", err
	}
	return config.endpoint, nil
}

// Do sends a transformed JSON payload to the upstream selected for alias.
func (c *Client) Do(ctx context.Context, alias string, payload []byte, stream bool, clientHeaders http.Header) (*http.Response, error) {
	if ctx == nil {
		return nil, fmt.Errorf("context is required")
	}

	config, upstreamName, err := c.resolve(alias)
	if err != nil {
		return nil, err
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, config.url, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("create request for upstream %q: %w", upstreamName, err)
	}
	request.Header.Set("Content-Type", "application/json")
	if stream {
		request.Header.Set("Accept", "text/event-stream")
	} else {
		request.Header.Set("Accept", "application/json")
	}
	if config.endpoint == capabilities.EndpointResponses {
		copyHeader(request.Header, clientHeaders, "Authorization")
		copyHeader(request.Header, clientHeaders, "OpenAI-Organization")
		copyHeader(request.Header, clientHeaders, "OpenAI-Project")
	} else {
		apiKey := bearerToken(clientHeaders.Get("Authorization"))
		if apiKey != "" {
			request.Header.Set("x-api-key", apiKey)
		}
		request.Header.Set("anthropic-version", config.anthropicVersion)
	}

	response, err := c.doer.Do(request)
	if err != nil {
		return nil, fmt.Errorf("send request to upstream %q: %w", upstreamName, err)
	}
	return response, nil
}

func (c *Client) resolve(alias string) (resolvedConfig, string, error) {
	if c == nil {
		return resolvedConfig{}, "", fmt.Errorf("upstream client is required")
	}

	upstreamName, exists := c.routes[alias]
	if !exists {
		return resolvedConfig{}, "", fmt.Errorf("%w for model alias %q", ErrRouteNotFound, alias)
	}
	config, exists := c.upstreams[upstreamName]
	if !exists {
		return resolvedConfig{}, "", fmt.Errorf("upstream %q is not configured", upstreamName)
	}
	return config, upstreamName, nil
}

func copyHeader(target, source http.Header, name string) {
	value := source.Get(name)
	if value == "" {
		return
	}
	target.Set(name, value)
}

func bearerToken(value string) string {
	parts := strings.Fields(value)
	if len(parts) != 2 || !strings.EqualFold(parts[0], "Bearer") {
		return ""
	}
	return parts[1]
}
