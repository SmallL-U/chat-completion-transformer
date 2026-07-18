package capabilities

import (
	"errors"
	"sync"
	"testing"
)

func TestRegistryProfilesAreProtocolModelSpecific(t *testing.T) {
	registry := NewRegistry()
	openAI := Profile{
		Provider: ProviderOpenAI,
		Endpoint: EndpointResponses,
		Model:    "model-a",
		Content:  ContentCapabilities{Text: true},
	}
	if err := registry.RegisterProfile(openAI); err != nil {
		t.Fatal(err)
	}

	profile, err := registry.Profile(ProviderOpenAI, EndpointResponses, "model-a")
	if err != nil {
		t.Fatal(err)
	}
	if profile.Model != "model-a" || !profile.Content.Text {
		t.Fatalf("profile = %#v", profile)
	}

	_, err = registry.Profile(ProviderAnthropic, EndpointMessages, "model-a")
	if !errors.Is(err, ErrProfileMissing) {
		t.Fatalf("error = %v", err)
	}
}

func TestRegistryNeverGuessesRoutes(t *testing.T) {
	registry := NewRegistry()
	if err := registry.RegisterRoute(ModelRoute{
		Alias: "general",
		Targets: map[Endpoint]string{
			EndpointResponses: "openai-model",
		},
	}); err != nil {
		t.Fatal(err)
	}

	model, err := registry.Resolve("general", EndpointResponses)
	if err != nil || model != "openai-model" {
		t.Fatalf("model = %q, error = %v", model, err)
	}

	_, err = registry.Resolve("general", EndpointMessages)
	if !errors.Is(err, ErrRouteMissing) {
		t.Fatalf("error = %v", err)
	}

	_, err = registry.Resolve("gpt-source-model", EndpointMessages)
	if !errors.Is(err, ErrRouteMissing) {
		t.Fatalf("error = %v", err)
	}
}

func TestRegistryConcurrentReads(t *testing.T) {
	registry := NewRegistry()
	if err := registry.RegisterProfile(Profile{
		Provider: ProviderOpenAI,
		Endpoint: EndpointResponses,
		Model:    "model-a",
	}); err != nil {
		t.Fatal(err)
	}

	var wait sync.WaitGroup
	for range 32 {
		wait.Add(1)
		go func() {
			defer wait.Done()
			if _, err := registry.Profile(ProviderOpenAI, EndpointResponses, "model-a"); err != nil {
				t.Errorf("profile: %v", err)
			}
		}()
	}
	wait.Wait()
}

func TestRegistryRejectsMismatchedProviderEndpoint(t *testing.T) {
	registry := NewRegistry()
	err := registry.RegisterProfile(Profile{
		Provider: ProviderOpenAI,
		Endpoint: EndpointMessages,
		Model:    "model-a",
	})
	if !errors.Is(err, ErrInvalidProfile) {
		t.Fatalf("error = %v", err)
	}

	err = registry.RegisterRoute(ModelRoute{
		Alias:   "general",
		Targets: map[Endpoint]string{"unknown": "model-a"},
	})
	if !errors.Is(err, ErrInvalidRoute) {
		t.Fatalf("route error = %v", err)
	}
}

func TestRegistryValidatesPromptCacheProfiles(t *testing.T) {
	tests := []struct {
		name    string
		profile Profile
		wantErr bool
	}{
		{
			name:    "unset normalizes to none",
			profile: Profile{Provider: ProviderOpenAI, Endpoint: EndpointResponses, Model: "openai-none"},
		},
		{
			name: "anthropic direct messages",
			profile: Profile{
				Provider:    ProviderAnthropic,
				Endpoint:    EndpointMessages,
				Model:       "anthropic-cache",
				PromptCache: PromptCacheCapabilities{Mode: PromptCacheAnthropic},
			},
		},
		{
			name: "openai legacy retention flags",
			profile: Profile{
				Provider: ProviderOpenAI,
				Endpoint: EndpointResponses,
				Model:    "openai-legacy",
				PromptCache: PromptCacheCapabilities{
					Mode:                 PromptCacheOpenAILegacy,
					InMemoryRetention:    true,
					ExtendedRetention24h: true,
				},
			},
		},
		{
			name: "openai 5.6",
			profile: Profile{
				Provider:    ProviderOpenAI,
				Endpoint:    EndpointResponses,
				Model:       "openai-56",
				PromptCache: PromptCacheCapabilities{Mode: PromptCacheOpenAI56},
			},
		},
		{
			name: "anthropic mode on responses",
			profile: Profile{
				Provider:    ProviderOpenAI,
				Endpoint:    EndpointResponses,
				Model:       "bad-anthropic",
				PromptCache: PromptCacheCapabilities{Mode: PromptCacheAnthropic},
			},
			wantErr: true,
		},
		{
			name: "anthropic cloud endpoint",
			profile: Profile{
				Provider:    ProviderAnthropic,
				Endpoint:    EndpointBedrockMessages,
				Model:       "bad-bedrock",
				PromptCache: PromptCacheCapabilities{Mode: PromptCacheAnthropic},
			},
			wantErr: true,
		},
		{
			name: "openai mode on messages",
			profile: Profile{
				Provider:    ProviderAnthropic,
				Endpoint:    EndpointMessages,
				Model:       "bad-openai",
				PromptCache: PromptCacheCapabilities{Mode: PromptCacheOpenAI56},
			},
			wantErr: true,
		},
		{
			name: "retention flags on 5.6",
			profile: Profile{
				Provider:    ProviderOpenAI,
				Endpoint:    EndpointResponses,
				Model:       "bad-flags",
				PromptCache: PromptCacheCapabilities{Mode: PromptCacheOpenAI56, InMemoryRetention: true},
			},
			wantErr: true,
		},
		{
			name: "unknown mode",
			profile: Profile{
				Provider:    ProviderOpenAI,
				Endpoint:    EndpointResponses,
				Model:       "bad-mode",
				PromptCache: PromptCacheCapabilities{Mode: "future"},
			},
			wantErr: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			registry := NewRegistry()
			err := registry.RegisterProfile(test.profile)
			if test.wantErr {
				if !errors.Is(err, ErrInvalidProfile) {
					t.Fatalf("error = %v", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			profile, err := registry.Profile(test.profile.Provider, test.profile.Endpoint, test.profile.Model)
			if err != nil {
				t.Fatal(err)
			}
			if test.profile.PromptCache.Mode == PromptCacheUnset && profile.PromptCache.Mode != PromptCacheNone {
				t.Fatalf("prompt cache mode = %q", profile.PromptCache.Mode)
			}
		})
	}
}
