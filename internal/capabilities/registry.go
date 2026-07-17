package capabilities

import (
	"errors"
	"fmt"
	"sync"
)

type Provider string

const (
	ProviderOpenAI    Provider = "openai"
	ProviderAnthropic Provider = "anthropic"
)

type Endpoint string

const (
	EndpointResponses       Endpoint = "responses"
	EndpointMessages        Endpoint = "messages"
	EndpointBedrockMessages Endpoint = "bedrock-messages"
	EndpointVertexMessages  Endpoint = "vertex-messages"
)

var (
	ErrInvalidProfile = errors.New("invalid capability profile")
	ErrInvalidRoute   = errors.New("invalid model route")
	ErrProfileMissing = errors.New("capability profile not found")
	ErrRouteMissing   = errors.New("model route not found")
)

type ImageCapabilities struct {
	URL    bool `json:"url"`
	Base64 bool `json:"base64"`
	FileID bool `json:"file_id"`
}

type ContentCapabilities struct {
	Text  bool `json:"text"`
	Image bool `json:"image"`
	Audio bool `json:"audio"`
	File  bool `json:"file"`
}

// Profile describes one concrete protocol endpoint and model combination.
// Profiles are injected by the caller; the transformer never guesses model
// equivalence or applies provider-wide defaults.
type Profile struct {
	Provider                     Provider            `json:"provider"`
	Endpoint                     Endpoint            `json:"endpoint"`
	Model                        string              `json:"model"`
	MidConversationSystem        bool                `json:"mid_conversation_system"`
	Temperature                  bool                `json:"temperature"`
	TopP                         bool                `json:"top_p"`
	TopK                         bool                `json:"top_k"`
	StopSequences                bool                `json:"stop_sequences"`
	Metadata                     bool                `json:"metadata"`
	StructuredOutput             bool                `json:"structured_output"`
	StrictTools                  bool                `json:"strict_tools"`
	ParallelToolCalls            bool                `json:"parallel_tool_calls"`
	ForcedToolChoiceWithThinking bool                `json:"forced_tool_choice_with_thinking"`
	Images                       ImageCapabilities   `json:"images"`
	Files                        ImageCapabilities   `json:"files"`
	Content                      ContentCapabilities `json:"content"`
}

type ModelRoute struct {
	Alias   string              `json:"alias"`
	Targets map[Endpoint]string `json:"targets"`
}

type profileKey struct {
	provider Provider
	endpoint Endpoint
	model    string
}

// Registry is safe for concurrent reads and occasional configuration updates.
type Registry struct {
	mu       sync.RWMutex
	profiles map[profileKey]Profile
	routes   map[string]map[Endpoint]string
}

func NewRegistry() *Registry {
	return &Registry{
		profiles: make(map[profileKey]Profile),
		routes:   make(map[string]map[Endpoint]string),
	}
}

func (r *Registry) RegisterProfile(profile Profile) error {
	if profile.Provider == "" || profile.Endpoint == "" || profile.Model == "" {
		return ErrInvalidProfile
	}
	if !validProviderEndpoint(profile.Provider, profile.Endpoint) {
		return fmt.Errorf("%w: provider %q cannot use endpoint %q", ErrInvalidProfile, profile.Provider, profile.Endpoint)
	}

	key := profileKey{
		provider: profile.Provider,
		endpoint: profile.Endpoint,
		model:    profile.Model,
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	r.profiles[key] = profile
	return nil
}

func (r *Registry) Profile(provider Provider, endpoint Endpoint, model string) (Profile, error) {
	if provider == "" || endpoint == "" || model == "" {
		return Profile{}, ErrInvalidProfile
	}

	key := profileKey{provider: provider, endpoint: endpoint, model: model}
	r.mu.RLock()
	profile, ok := r.profiles[key]
	r.mu.RUnlock()
	if !ok {
		return Profile{}, fmt.Errorf("%w: %s/%s/%s", ErrProfileMissing, provider, endpoint, model)
	}

	return profile, nil
}

func (r *Registry) RegisterRoute(route ModelRoute) error {
	if route.Alias == "" || len(route.Targets) == 0 {
		return fmt.Errorf("%w: route must contain an alias and at least one target", ErrInvalidRoute)
	}

	targets := make(map[Endpoint]string, len(route.Targets))
	for endpoint, model := range route.Targets {
		if endpoint == "" || model == "" {
			return fmt.Errorf("%w: route %q has an empty endpoint or model", ErrInvalidRoute, route.Alias)
		}
		if !validEndpoint(endpoint) {
			return fmt.Errorf("%w: route %q uses unknown endpoint %q", ErrInvalidRoute, route.Alias, endpoint)
		}
		targets[endpoint] = model
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	r.routes[route.Alias] = targets
	return nil
}

func validProviderEndpoint(provider Provider, endpoint Endpoint) bool {
	if provider == ProviderOpenAI {
		return endpoint == EndpointResponses
	}
	if provider != ProviderAnthropic {
		return false
	}
	return endpoint == EndpointMessages || endpoint == EndpointBedrockMessages || endpoint == EndpointVertexMessages
}

func validEndpoint(endpoint Endpoint) bool {
	return endpoint == EndpointResponses || endpoint == EndpointMessages || endpoint == EndpointBedrockMessages || endpoint == EndpointVertexMessages
}

func (r *Registry) Resolve(alias string, endpoint Endpoint) (string, error) {
	if alias == "" || endpoint == "" {
		return "", fmt.Errorf("%w: alias and endpoint are required", ErrRouteMissing)
	}

	r.mu.RLock()
	targets, ok := r.routes[alias]
	if !ok {
		r.mu.RUnlock()
		return "", fmt.Errorf("%w: alias %q", ErrRouteMissing, alias)
	}
	model, ok := targets[endpoint]
	r.mu.RUnlock()
	if !ok {
		return "", fmt.Errorf("%w: alias %q has no %s target", ErrRouteMissing, alias, endpoint)
	}

	return model, nil
}
