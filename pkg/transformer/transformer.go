package transformer

import (
	"errors"
	"fmt"

	"chat-completion-transformer/internal/assets"
	"chat-completion-transformer/internal/canonical"
	"chat-completion-transformer/internal/capabilities"
	"chat-completion-transformer/internal/sse"
)

type InstructionPolicy string

const (
	InstructionPolicyPreserveMessages InstructionPolicy = "preserve_messages"
	InstructionPolicyExtractLeading   InstructionPolicy = "extract_leading"
)

type Config struct {
	Mode                   Mode
	InstructionPolicy      InstructionPolicy
	AnthropicEndpoint      Endpoint
	DefaultMaxOutputTokens *int
	MaxSSEEventBytes       int
	Profiles               []CapabilityProfile
	Routes                 []ModelRoute
	Resolver               AssetResolver
}

// Transformer owns immutable defaults plus a concurrency-safe model and
// capability registry. Provider model names are always resolved from aliases.
type Transformer struct {
	mode                   canonical.Mode
	instructionPolicy      InstructionPolicy
	anthropicEndpoint      capabilities.Endpoint
	defaultMaxOutputTokens *int
	maxSSEEventBytes       int
	registry               *capabilities.Registry
	resolver               assets.Resolver
}

func New(config Config) (*Transformer, error) {
	mode := canonical.Mode(config.Mode)
	if mode == "" {
		mode = canonical.ModeCompatible
	}
	if mode != canonical.ModeStrict && mode != canonical.ModeCompatible && mode != canonical.ModeEmulate {
		return nil, fmt.Errorf("unsupported transform mode %q", mode)
	}

	policy := config.InstructionPolicy
	if policy == "" {
		policy = InstructionPolicyPreserveMessages
	}
	if policy != InstructionPolicyPreserveMessages && policy != InstructionPolicyExtractLeading {
		return nil, fmt.Errorf("unsupported instruction policy %q", policy)
	}

	anthropicEndpoint := capabilities.Endpoint(config.AnthropicEndpoint)
	if anthropicEndpoint == "" {
		anthropicEndpoint = capabilities.EndpointMessages
	}
	if anthropicEndpoint != capabilities.EndpointMessages && anthropicEndpoint != capabilities.EndpointBedrockMessages && anthropicEndpoint != capabilities.EndpointVertexMessages {
		return nil, fmt.Errorf("unsupported Anthropic endpoint %q", anthropicEndpoint)
	}
	if config.DefaultMaxOutputTokens != nil && *config.DefaultMaxOutputTokens < 1 {
		return nil, errors.New("default max output tokens must be at least 1")
	}

	registry := capabilities.NewRegistry()
	for _, profile := range config.Profiles {
		if err := registry.RegisterProfile(profile); err != nil {
			return nil, err
		}
	}
	for _, route := range config.Routes {
		if err := registry.RegisterRoute(route); err != nil {
			return nil, err
		}
	}

	resolver := config.Resolver
	if resolver == nil {
		resolver = assets.NativeResolver{}
	}
	maxEventBytes := config.MaxSSEEventBytes
	if maxEventBytes <= 0 {
		maxEventBytes = sse.DefaultMaxEventBytes
	}

	return &Transformer{
		mode:                   mode,
		instructionPolicy:      policy,
		anthropicEndpoint:      anthropicEndpoint,
		defaultMaxOutputTokens: cloneInt(config.DefaultMaxOutputTokens),
		maxSSEEventBytes:       maxEventBytes,
		registry:               registry,
		resolver:               resolver,
	}, nil
}

func (t *Transformer) RegisterProfile(profile CapabilityProfile) error {
	if t == nil || t.registry == nil {
		return errors.New("transformer is not initialized")
	}
	return t.registry.RegisterProfile(profile)
}

func (t *Transformer) RegisterRoute(route ModelRoute) error {
	if t == nil || t.registry == nil {
		return errors.New("transformer is not initialized")
	}
	return t.registry.RegisterRoute(route)
}

func cloneInt(value *int) *int {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}
