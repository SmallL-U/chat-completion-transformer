package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"chat-completion-transformer/internal/capabilities"
)

const validYAML = `server:
  address: ":9090"
  gin_mode: "release"
  read_header_timeout: "4s"
  idle_timeout: "45s"
  shutdown_timeout: "8s"
  max_body_bytes: 2000000
  max_stream_bytes: 64000000
  max_sse_event_bytes: 4000000
gateway:
  response_header_timeout: "45s"
  upstreams:
    openai-main:
      endpoint: "responses"
      url: "https://api.openai.com/v1/responses"
  routes:
    general: "openai-main"
transformer:
  mode: "compatible"
  instruction_policy: "preserve_messages"
  anthropic_endpoint: "messages"
  default_max_output_tokens: 2048
  profiles:
    - provider: "openai"
      endpoint: "responses"
      model: "gpt-test"
      mid_conversation_system: true
      temperature: true
      top_p: true
      top_k: false
      stop_sequences: true
      metadata: true
      structured_output: true
      strict_tools: true
      parallel_tool_calls: true
      forced_tool_choice_with_thinking: false
      images:
        url: true
        base64: true
        file_id: true
      content:
        text: true
        image: true
        audio: false
        file: true
  routes:
    - alias: "general"
      targets:
        responses: "gpt-test"
`

func TestLoadYAML(t *testing.T) {
	clearConfigEnvironment(t)

	config, err := Load(writeConfig(t, validYAML))
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	if config.Server.Address != ":9090" {
		t.Fatalf("Address = %q, want %q", config.Server.Address, ":9090")
	}
	if config.Server.ReadHeaderTimeout != 4*time.Second {
		t.Fatalf("ReadHeaderTimeout = %v, want 4s", config.Server.ReadHeaderTimeout)
	}
	if config.Server.MaxStreamBytes != 64_000_000 {
		t.Fatalf("MaxStreamBytes = %d, want 64000000", config.Server.MaxStreamBytes)
	}
	if config.Gateway.ResponseHeaderTimeout != 45*time.Second {
		t.Fatalf("ResponseHeaderTimeout = %v, want 45s", config.Gateway.ResponseHeaderTimeout)
	}
	if config.Gateway.Routes["general"] != "openai-main" {
		t.Fatalf("Gateway routes = %+v, want general route", config.Gateway.Routes)
	}
	if config.Transformer.DefaultMaxOutputTokens == nil || *config.Transformer.DefaultMaxOutputTokens != 2048 {
		t.Fatalf("DefaultMaxOutputTokens = %v, want 2048", config.Transformer.DefaultMaxOutputTokens)
	}
	if len(config.Transformer.Profiles) != 1 {
		t.Fatalf("len(Profiles) = %d, want 1", len(config.Transformer.Profiles))
	}
	profile := config.Transformer.Profiles[0]
	if !profile.MidConversationSystem || !profile.Images.FileID || !profile.Content.File {
		t.Fatalf("profile snake_case fields were not decoded: %+v", profile)
	}
	if profile.PromptCache.Mode != capabilities.PromptCacheNone {
		t.Fatalf("PromptCache.Mode = %q, want %q", profile.PromptCache.Mode, capabilities.PromptCacheNone)
	}
	if len(config.Transformer.Routes) != 1 || config.Transformer.Routes[0].Targets[capabilities.EndpointResponses] != "gpt-test" {
		t.Fatalf("Routes = %+v, want responses route", config.Transformer.Routes)
	}
}

func TestLoadPreservesDottedRouteAlias(t *testing.T) {
	clearConfigEnvironment(t)

	const (
		alias        = "gpt-5.6-luna"
		upstreamName = "openmodel.ai"
	)
	configYAML := strings.ReplaceAll(validYAML, "general", alias)
	configYAML = strings.ReplaceAll(configYAML, "gpt-test", alias)
	configYAML = strings.ReplaceAll(configYAML, "openai-main", upstreamName)

	config, err := Load(writeConfig(t, configYAML))
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if config.Gateway.Routes[alias] != upstreamName {
		t.Fatalf("Gateway routes = %+v, want exact alias %q", config.Gateway.Routes, alias)
	}
	if _, exists := config.Gateway.Upstreams[upstreamName]; !exists {
		t.Fatalf("Gateway upstreams = %+v, want exact name %q", config.Gateway.Upstreams, upstreamName)
	}
	if len(config.Transformer.Routes) != 1 || config.Transformer.Routes[0].Alias != alias {
		t.Fatalf("Transformer routes = %+v, want exact alias %q", config.Transformer.Routes, alias)
	}
}

func TestLoadPreservesDottedRouteAliasFromEnvironment(t *testing.T) {
	clearConfigEnvironment(t)

	const (
		alias        = "gpt-5.6-luna"
		upstreamName = "openmodel.ai"
	)
	t.Setenv("CCT_GATEWAY_UPSTREAMS", `{"openmodel.ai":{"endpoint":"responses","url":"https://api.openmodel.ai/v1/responses"}}`)
	t.Setenv("CCT_GATEWAY_ROUTES", `{"gpt-5.6-luna":"openmodel.ai"}`)
	t.Setenv("CCT_TRANSFORMER_ROUTES", `[{"alias":"gpt-5.6-luna","targets":{"responses":"gpt-test"}}]`)

	config, err := Load(writeConfig(t, validYAML))
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if config.Gateway.Routes[alias] != upstreamName {
		t.Fatalf("Gateway routes = %+v, want exact alias %q", config.Gateway.Routes, alias)
	}
	if _, exists := config.Gateway.Upstreams[upstreamName]; !exists {
		t.Fatalf("Gateway upstreams = %+v, want exact name %q", config.Gateway.Upstreams, upstreamName)
	}
	if len(config.Transformer.Routes) != 1 || config.Transformer.Routes[0].Alias != alias {
		t.Fatalf("Transformer routes = %+v, want exact alias %q", config.Transformer.Routes, alias)
	}
}

func TestLoadPromptCacheYAML(t *testing.T) {
	clearConfigEnvironment(t)
	configYAML := strings.Replace(
		validYAML,
		"      content:\n",
		"      prompt_cache:\n        mode: \"openai_5_6\"\n        in_memory_retention: true\n        extended_retention_24h: true\n      content:\n",
		1,
	)

	config, err := Load(writeConfig(t, configYAML))
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	cache := config.Transformer.Profiles[0].PromptCache
	if cache.Mode != capabilities.PromptCacheOpenAI56 || !cache.InMemoryRetention || !cache.ExtendedRetention24h {
		t.Fatalf("PromptCache = %+v, want OpenAI 5.6 mode with both retention flags", cache)
	}
}

func TestRepositoryConfig(t *testing.T) {
	clearConfigEnvironment(t)

	config, err := Load(filepath.Join("..", "..", defaultPath))
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if config.Transformer.DefaultMaxOutputTokens != nil {
		t.Fatalf("DefaultMaxOutputTokens = %v, want nil", config.Transformer.DefaultMaxOutputTokens)
	}
}

func TestLoadScalarEnvironmentOverrides(t *testing.T) {
	clearConfigEnvironment(t)
	t.Setenv("CCT_SERVER_ADDRESS", ":7070")
	t.Setenv("CCT_SERVER_GIN_MODE", "test")
	t.Setenv("CCT_SERVER_READ_HEADER_TIMEOUT", "7s")
	t.Setenv("CCT_SERVER_IDLE_TIMEOUT", "2m")
	t.Setenv("CCT_SERVER_SHUTDOWN_TIMEOUT", "12s")
	t.Setenv("CCT_SERVER_MAX_BODY_BYTES", "3000000")
	t.Setenv("CCT_SERVER_MAX_STREAM_BYTES", "90000000")
	t.Setenv("CCT_SERVER_MAX_SSE_EVENT_BYTES", "5000000")
	t.Setenv("CCT_GATEWAY_RESPONSE_HEADER_TIMEOUT", "90s")
	t.Setenv("CCT_TRANSFORMER_MODE", "strict")
	t.Setenv("CCT_TRANSFORMER_INSTRUCTION_POLICY", "extract_leading")
	t.Setenv("CCT_TRANSFORMER_ANTHROPIC_ENDPOINT", "vertex-messages")
	t.Setenv("CCT_TRANSFORMER_DEFAULT_MAX_OUTPUT_TOKENS", "512")

	config, err := Load(writeConfig(t, validYAML))
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	if config.Server.Address != ":7070" || config.Server.GinMode != "test" {
		t.Fatalf("server identity overrides were not applied: %+v", config.Server)
	}
	if config.Server.ReadHeaderTimeout != 7*time.Second || config.Server.IdleTimeout != 2*time.Minute || config.Server.ShutdownTimeout != 12*time.Second {
		t.Fatalf("server duration overrides were not applied: %+v", config.Server)
	}
	if config.Server.MaxBodyBytes != 3_000_000 || config.Server.MaxStreamBytes != 90_000_000 || config.Server.MaxSSEEventBytes != 5_000_000 {
		t.Fatalf("server size overrides were not applied: %+v", config.Server)
	}
	if config.Gateway.ResponseHeaderTimeout != 90*time.Second {
		t.Fatalf("gateway duration override was not applied: %+v", config.Gateway)
	}
	if config.Transformer.Mode != "strict" || config.Transformer.InstructionPolicy != "extract_leading" || config.Transformer.AnthropicEndpoint != capabilities.EndpointVertexMessages {
		t.Fatalf("transformer scalar overrides were not applied: %+v", config.Transformer)
	}
	if config.Transformer.DefaultMaxOutputTokens == nil || *config.Transformer.DefaultMaxOutputTokens != 512 {
		t.Fatalf("DefaultMaxOutputTokens = %v, want 512", config.Transformer.DefaultMaxOutputTokens)
	}
}

func TestLoadNullableDefaultMaxOutputTokensEnvironmentOverrides(t *testing.T) {
	tests := []struct {
		name  string
		value string
	}{
		{name: "null", value: "null"},
		{name: "empty", value: ""},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			clearConfigEnvironment(t)
			t.Setenv(defaultMaxOutputTokensEnvironment, test.value)

			config, err := Load(writeConfig(t, validYAML))
			if err != nil {
				t.Fatalf("Load() error = %v", err)
			}
			if config.Transformer.DefaultMaxOutputTokens != nil {
				t.Fatalf("DefaultMaxOutputTokens = %v, want nil", config.Transformer.DefaultMaxOutputTokens)
			}
		})
	}
}

func TestLoadComplexEnvironmentOverrides(t *testing.T) {
	clearConfigEnvironment(t)
	t.Setenv("CCT_GATEWAY_UPSTREAMS", `{"anthropic-main":{"endpoint":"messages","url":"https://api.anthropic.com/v1/messages","anthropic_version":"2023-06-01"}}`)
	t.Setenv("CCT_GATEWAY_ROUTES", `{"environment":"anthropic-main"}`)
	t.Setenv("CCT_TRANSFORMER_PROFILES", `[{"provider":"anthropic","endpoint":"messages","model":"claude-env","content":{"text":true}}]`)
	t.Setenv("CCT_TRANSFORMER_ROUTES", `[{"alias":"environment","targets":{"messages":"claude-env"}}]`)

	config, err := Load(writeConfig(t, validYAML))
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	if len(config.Transformer.Profiles) != 1 || config.Transformer.Profiles[0].Model != "claude-env" {
		t.Fatalf("Profiles = %+v, want JSON environment profiles", config.Transformer.Profiles)
	}
	if config.Transformer.Profiles[0].PromptCache.Mode != capabilities.PromptCacheNone {
		t.Fatalf("PromptCache.Mode = %q, want %q", config.Transformer.Profiles[0].PromptCache.Mode, capabilities.PromptCacheNone)
	}
	if len(config.Transformer.Routes) != 1 || config.Transformer.Routes[0].Alias != "environment" {
		t.Fatalf("Routes = %+v, want JSON environment routes", config.Transformer.Routes)
	}
	if config.Gateway.Routes["environment"] != "anthropic-main" {
		t.Fatalf("Gateway routes = %+v, want environment route", config.Gateway.Routes)
	}
}

func TestLoadPromptCacheJSONEnvironment(t *testing.T) {
	clearConfigEnvironment(t)
	t.Setenv("CCT_TRANSFORMER_PROFILES", `[{"provider":"openai","endpoint":"responses","model":"gpt-test","prompt_cache":{"mode":"openai_legacy","in_memory_retention":true,"extended_retention_24h":true},"content":{"text":true}}]`)

	config, err := Load(writeConfig(t, validYAML))
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	cache := config.Transformer.Profiles[0].PromptCache
	if cache.Mode != capabilities.PromptCacheOpenAILegacy || !cache.InMemoryRetention || !cache.ExtendedRetention24h {
		t.Fatalf("PromptCache = %+v, want legacy mode with both retention flags", cache)
	}
}

func TestLoadCollectionEnvironmentStates(t *testing.T) {
	tests := []struct {
		name             string
		setEnvironment   bool
		gatewayValue     string
		transformerValue string
		wantLength       int
	}{
		{name: "unset preserves YAML", wantLength: 1},
		{name: "empty string clears", setEnvironment: true, wantLength: 0},
		{name: "null clears", setEnvironment: true, gatewayValue: `null`, transformerValue: `null`, wantLength: 0},
		{name: "empty JSON clears", setEnvironment: true, gatewayValue: `{}`, transformerValue: `[]`, wantLength: 0},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			clearConfigEnvironment(t)
			if test.setEnvironment {
				t.Setenv("CCT_GATEWAY_UPSTREAMS", test.gatewayValue)
				t.Setenv("CCT_GATEWAY_ROUTES", test.gatewayValue)
				t.Setenv("CCT_TRANSFORMER_PROFILES", test.transformerValue)
				t.Setenv("CCT_TRANSFORMER_ROUTES", test.transformerValue)
			}

			config, err := Load(writeConfig(t, validYAML))
			if err != nil {
				t.Fatalf("Load() error = %v", err)
			}
			if len(config.Gateway.Upstreams) != test.wantLength || len(config.Gateway.Routes) != test.wantLength {
				t.Fatalf("gateway collection lengths = %d/%d, want %d: %+v", len(config.Gateway.Upstreams), len(config.Gateway.Routes), test.wantLength, config.Gateway)
			}
			if len(config.Transformer.Profiles) != test.wantLength || len(config.Transformer.Routes) != test.wantLength {
				t.Fatalf("transformer collection lengths = %d/%d, want %d: %+v", len(config.Transformer.Profiles), len(config.Transformer.Routes), test.wantLength, config.Transformer)
			}
		})
	}
}

func TestLoadRejectsInvalidConfiguration(t *testing.T) {
	t.Run("explicit empty scalar environment", func(t *testing.T) {
		clearConfigEnvironment(t)
		t.Setenv("CCT_SERVER_ADDRESS", "")

		_, err := Load(writeConfig(t, validYAML))
		if err == nil || !strings.Contains(err.Error(), "address is required") {
			t.Fatalf("Load() error = %v, want explicit empty address error", err)
		}
	})

	t.Run("invalid scalar environment", func(t *testing.T) {
		clearConfigEnvironment(t)
		t.Setenv("CCT_SERVER_MAX_BODY_BYTES", "not-a-number")

		_, err := Load(writeConfig(t, validYAML))
		if err == nil || !strings.Contains(err.Error(), "max_body_bytes") {
			t.Fatalf("Load() error = %v, want max_body_bytes decode error", err)
		}
	})

	t.Run("invalid optional integer environment", func(t *testing.T) {
		clearConfigEnvironment(t)
		t.Setenv(defaultMaxOutputTokensEnvironment, "not-a-number")

		_, err := Load(writeConfig(t, validYAML))
		if err == nil || !strings.Contains(err.Error(), defaultMaxOutputTokensEnvironment) {
			t.Fatalf("Load() error = %v, want optional integer environment error", err)
		}
	})

	t.Run("malformed complex environment", func(t *testing.T) {
		clearConfigEnvironment(t)
		t.Setenv("CCT_TRANSFORMER_PROFILES", `[{`)

		_, err := Load(writeConfig(t, validYAML))
		if err == nil || !strings.Contains(err.Error(), "CCT_TRANSFORMER_PROFILES") {
			t.Fatalf("Load() error = %v, want profiles environment error", err)
		}
	})

	t.Run("unknown complex environment field", func(t *testing.T) {
		clearConfigEnvironment(t)
		t.Setenv("CCT_TRANSFORMER_ROUTES", `[{"alias":"general","targets":{"responses":"gpt-test"},"typo":true}]`)

		_, err := Load(writeConfig(t, validYAML))
		if err == nil || !strings.Contains(err.Error(), "unknown field") {
			t.Fatalf("Load() error = %v, want unknown field error", err)
		}
	})

	t.Run("unknown prompt cache JSON environment field", func(t *testing.T) {
		clearConfigEnvironment(t)
		t.Setenv("CCT_TRANSFORMER_PROFILES", `[{"provider":"openai","endpoint":"responses","model":"gpt-test","prompt_cache":{"mode":"openai_5_6","typo":true}}]`)

		_, err := Load(writeConfig(t, validYAML))
		if err == nil || !strings.Contains(err.Error(), "unknown field") || !strings.Contains(err.Error(), "typo") {
			t.Fatalf("Load() error = %v, want unknown nested prompt cache field error", err)
		}
	})

	t.Run("unknown prompt cache YAML field", func(t *testing.T) {
		clearConfigEnvironment(t)
		configYAML := strings.Replace(
			validYAML,
			"      content:\n",
			"      prompt_cache:\n        mode: \"openai_5_6\"\n        typo: true\n      content:\n",
			1,
		)

		_, err := Load(writeConfig(t, configYAML))
		if err == nil || !strings.Contains(err.Error(), "typo") {
			t.Fatalf("Load() error = %v, want unknown nested prompt cache YAML field error", err)
		}
	})

	t.Run("invalid prompt cache profile combinations", func(t *testing.T) {
		tests := []struct {
			name     string
			profiles string
		}{
			{
				name:     "anthropic mode on Responses",
				profiles: `[{"provider":"openai","endpoint":"responses","model":"gpt-test","prompt_cache":{"mode":"anthropic"}}]`,
			},
			{
				name:     "OpenAI mode on Messages",
				profiles: `[{"provider":"anthropic","endpoint":"messages","model":"claude-test","prompt_cache":{"mode":"openai_5_6"}}]`,
			},
			{
				name:     "retention flag outside OpenAI cache mode",
				profiles: `[{"provider":"openai","endpoint":"responses","model":"gpt-test","prompt_cache":{"mode":"none","in_memory_retention":true}}]`,
			},
		}

		for _, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				clearConfigEnvironment(t)
				t.Setenv("CCT_TRANSFORMER_PROFILES", test.profiles)

				_, err := Load(writeConfig(t, validYAML))
				if err == nil || !strings.Contains(err.Error(), "invalid capability profile") {
					t.Fatalf("Load() error = %v, want invalid prompt cache profile error", err)
				}
			})
		}
	})

	t.Run("route without profile", func(t *testing.T) {
		clearConfigEnvironment(t)
		configYAML := strings.Replace(validYAML, `responses: "gpt-test"`, `responses: "missing"`, 1)

		_, err := Load(writeConfig(t, configYAML))
		if err == nil || !strings.Contains(err.Error(), "capability profile not found") {
			t.Fatalf("Load() error = %v, want missing profile error", err)
		}
	})

	t.Run("gateway route without transformer endpoint target", func(t *testing.T) {
		clearConfigEnvironment(t)
		configYAML := strings.Replace(
			validYAML,
			`    - provider: "openai"
      endpoint: "responses"`,
			`    - provider: "anthropic"
      endpoint: "messages"`,
			1,
		)
		configYAML = strings.Replace(configYAML, `        responses: "gpt-test"`, `        messages: "gpt-test"`, 1)

		_, err := Load(writeConfig(t, configYAML))
		if err == nil || !strings.Contains(err.Error(), "has no responses target") {
			t.Fatalf("Load() error = %v, want missing gateway endpoint target error", err)
		}
	})

	t.Run("gateway route references unknown upstream", func(t *testing.T) {
		clearConfigEnvironment(t)
		configYAML := strings.Replace(validYAML, `    general: "openai-main"`, `    general: "missing"`, 1)

		_, err := Load(writeConfig(t, configYAML))
		if err == nil || !strings.Contains(err.Error(), `references unknown upstream "missing"`) {
			t.Fatalf("Load() error = %v, want unknown gateway upstream error", err)
		}
	})

	t.Run("direct messages requires messages transformer endpoint", func(t *testing.T) {
		clearConfigEnvironment(t)
		t.Setenv("CCT_GATEWAY_UPSTREAMS", `{"anthropic":{"endpoint":"messages","url":"https://api.anthropic.com/v1/messages"}}`)
		t.Setenv("CCT_GATEWAY_ROUTES", `{"general":"anthropic"}`)
		t.Setenv("CCT_TRANSFORMER_PROFILES", `[{"provider":"anthropic","endpoint":"messages","model":"claude-test","content":{"text":true}}]`)
		t.Setenv("CCT_TRANSFORMER_ROUTES", `[{"alias":"general","targets":{"messages":"claude-test"}}]`)
		t.Setenv("CCT_TRANSFORMER_ANTHROPIC_ENDPOINT", "vertex-messages")

		_, err := Load(writeConfig(t, validYAML))
		if err == nil || !strings.Contains(err.Error(), "uses direct messages") {
			t.Fatalf("Load() error = %v, want direct Messages endpoint mismatch", err)
		}
	})

	t.Run("zero default output tokens", func(t *testing.T) {
		clearConfigEnvironment(t)
		configYAML := strings.Replace(validYAML, "default_max_output_tokens: 2048", "default_max_output_tokens: 0", 1)

		_, err := Load(writeConfig(t, configYAML))
		if err == nil || !strings.Contains(err.Error(), "default_max_output_tokens") {
			t.Fatalf("Load() error = %v, want default token validation error", err)
		}
	})
}

func writeConfig(t *testing.T, contents string) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), "config.yml")
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	return path
}

func clearConfigEnvironment(t *testing.T) {
	t.Helper()

	environment := []string{
		"CCT_SERVER_ADDRESS",
		"CCT_SERVER_GIN_MODE",
		"CCT_SERVER_READ_HEADER_TIMEOUT",
		"CCT_SERVER_IDLE_TIMEOUT",
		"CCT_SERVER_SHUTDOWN_TIMEOUT",
		"CCT_SERVER_MAX_BODY_BYTES",
		"CCT_SERVER_MAX_STREAM_BYTES",
		"CCT_SERVER_MAX_SSE_EVENT_BYTES",
		"CCT_GATEWAY_RESPONSE_HEADER_TIMEOUT",
		"CCT_GATEWAY_UPSTREAMS",
		"CCT_GATEWAY_ROUTES",
		"CCT_TRANSFORMER_MODE",
		"CCT_TRANSFORMER_INSTRUCTION_POLICY",
		"CCT_TRANSFORMER_ANTHROPIC_ENDPOINT",
		"CCT_TRANSFORMER_DEFAULT_MAX_OUTPUT_TOKENS",
		"CCT_TRANSFORMER_PROFILES",
		"CCT_TRANSFORMER_ROUTES",
	}

	for _, key := range environment {
		value, exists := os.LookupEnv(key)
		if err := os.Unsetenv(key); err != nil {
			t.Fatalf("Unsetenv(%q) error = %v", key, err)
		}
		t.Cleanup(func() {
			if exists {
				_ = os.Setenv(key, value)
				return
			}
			_ = os.Unsetenv(key)
		})
	}
}
