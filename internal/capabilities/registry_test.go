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
