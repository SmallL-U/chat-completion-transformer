package config

import (
	"errors"
	"fmt"
	"strings"

	"chat-completion-transformer/internal/capabilities"
)

const (
	ginModeDebug   = "debug"
	ginModeRelease = "release"
	ginModeTest    = "test"

	modeStrict     = "strict"
	modeCompatible = "compatible"
	modeEmulate    = "emulate"

	instructionPolicyPreserveMessages = "preserve_messages"
	instructionPolicyExtractLeading   = "extract_leading"
)

// Validate checks values that cannot be expressed by configuration decoding.
func (config Config) Validate() error {
	if err := config.Server.validate(); err != nil {
		return fmt.Errorf("server: %w", err)
	}
	if err := config.Transformer.validate(); err != nil {
		return fmt.Errorf("transformer: %w", err)
	}

	return nil
}

func (config ServerConfig) validate() error {
	if strings.TrimSpace(config.Address) == "" {
		return errors.New("address is required")
	}
	if config.GinMode != ginModeDebug && config.GinMode != ginModeRelease && config.GinMode != ginModeTest {
		return fmt.Errorf("unsupported gin_mode %q", config.GinMode)
	}
	if config.ReadHeaderTimeout <= 0 {
		return errors.New("read_header_timeout must be positive")
	}
	if config.IdleTimeout <= 0 {
		return errors.New("idle_timeout must be positive")
	}
	if config.ShutdownTimeout <= 0 {
		return errors.New("shutdown_timeout must be positive")
	}
	if config.MaxBodyBytes <= 0 {
		return errors.New("max_body_bytes must be positive")
	}
	if config.MaxStreamBytes <= 0 {
		return errors.New("max_stream_bytes must be positive")
	}
	if config.MaxSSEEventBytes <= 0 {
		return errors.New("max_sse_event_bytes must be positive")
	}

	return nil
}

func (config TransformerConfig) validate() error {
	if config.Mode != modeStrict && config.Mode != modeCompatible && config.Mode != modeEmulate {
		return fmt.Errorf("unsupported mode %q", config.Mode)
	}
	if config.InstructionPolicy != instructionPolicyPreserveMessages && config.InstructionPolicy != instructionPolicyExtractLeading {
		return fmt.Errorf("unsupported instruction_policy %q", config.InstructionPolicy)
	}
	if config.AnthropicEndpoint != capabilities.EndpointMessages &&
		config.AnthropicEndpoint != capabilities.EndpointBedrockMessages &&
		config.AnthropicEndpoint != capabilities.EndpointVertexMessages {
		return fmt.Errorf("unsupported anthropic_endpoint %q", config.AnthropicEndpoint)
	}
	if config.DefaultMaxOutputTokens != nil && *config.DefaultMaxOutputTokens < 1 {
		return errors.New("default_max_output_tokens must be null or positive")
	}

	registry := capabilities.NewRegistry()
	for index, profile := range config.Profiles {
		if err := registry.RegisterProfile(profile); err != nil {
			return fmt.Errorf("profiles[%d]: %w", index, err)
		}
	}
	for index, route := range config.Routes {
		if err := registry.RegisterRoute(route); err != nil {
			return fmt.Errorf("routes[%d]: %w", index, err)
		}
	}

	return validateRouteProfiles(registry, config.Routes)
}

func validateRouteProfiles(registry *capabilities.Registry, routes []capabilities.ModelRoute) error {
	for routeIndex, route := range routes {
		for endpoint, model := range route.Targets {
			provider, err := providerForEndpoint(endpoint)
			if err != nil {
				return fmt.Errorf("routes[%d]: %w", routeIndex, err)
			}
			if _, err := registry.Profile(provider, endpoint, model); err != nil {
				return fmt.Errorf("routes[%d] target %q: %w", routeIndex, endpoint, err)
			}
		}
	}

	return nil
}

func providerForEndpoint(endpoint capabilities.Endpoint) (capabilities.Provider, error) {
	if endpoint == capabilities.EndpointResponses {
		return capabilities.ProviderOpenAI, nil
	}
	if endpoint == capabilities.EndpointMessages ||
		endpoint == capabilities.EndpointBedrockMessages ||
		endpoint == capabilities.EndpointVertexMessages {
		return capabilities.ProviderAnthropic, nil
	}

	return "", fmt.Errorf("unknown endpoint %q", endpoint)
}
