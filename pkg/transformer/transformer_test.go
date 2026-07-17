package transformer

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"
)

func TestNewValidatesConfiguration(t *testing.T) {
	tests := []struct {
		name   string
		config Config
	}{
		{name: "mode", config: Config{Mode: "unknown"}},
		{name: "instruction policy", config: Config{InstructionPolicy: "unknown"}},
		{name: "endpoint", config: Config{AnthropicEndpoint: EndpointResponses}},
		{name: "max tokens", config: Config{DefaultMaxOutputTokens: intPointer(0)}},
		{name: "profile mismatch", config: Config{Profiles: []CapabilityProfile{{Provider: ProviderOpenAI, Endpoint: EndpointMessages, Model: "x"}}}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := New(test.config); err == nil {
				t.Fatal("expected configuration error")
			}
		})
	}
}

func TestNewCopiesDefaultsAndRegistry(t *testing.T) {
	defaultMax := 100
	transformer, err := New(Config{
		DefaultMaxOutputTokens: &defaultMax,
		Profiles: []CapabilityProfile{{
			Provider: ProviderOpenAI,
			Endpoint: EndpointResponses,
			Model:    "target",
		}},
		Routes: []ModelRoute{{
			Alias:   "general",
			Targets: map[Endpoint]string{EndpointResponses: "target"},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	defaultMax = 1
	if transformer.mode != ModeCompatible || transformer.instructionPolicy != InstructionPolicyPreserveMessages {
		t.Fatalf("defaults = %#v", transformer)
	}
	if transformer.defaultMaxOutputTokens == nil || *transformer.defaultMaxOutputTokens != 100 {
		t.Fatalf("max tokens = %v", transformer.defaultMaxOutputTokens)
	}
	model, err := transformer.registry.Resolve("general", EndpointResponses)
	if err != nil || model != "target" {
		t.Fatalf("model = %q, error = %v", model, err)
	}
}

func intPointer(value int) *int {
	return &value
}

func TestProtocolTransformerRoutesChatRequestsThroughCanonicalIR(t *testing.T) {
	defaultMax := 200
	service, err := New(Config{
		DefaultMaxOutputTokens: &defaultMax,
		Profiles: []CapabilityProfile{
			{
				Provider:          ProviderOpenAI,
				Endpoint:          EndpointResponses,
				Model:             "openai-target",
				StructuredOutput:  true,
				StrictTools:       true,
				ParallelToolCalls: true,
				Content:           ContentCapabilities{Text: true},
			},
			{
				Provider:          ProviderAnthropic,
				Endpoint:          EndpointMessages,
				Model:             "anthropic-target",
				StructuredOutput:  true,
				StrictTools:       true,
				ParallelToolCalls: true,
				StopSequences:     true,
				Content:           ContentCapabilities{Text: true},
			},
		},
		Routes: []ModelRoute{{
			Alias: "general",
			Targets: map[Endpoint]string{
				EndpointResponses: "openai-target",
				EndpointMessages:  "anthropic-target",
			},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}

	decoded := service.DecodeChatRequest([]byte(`{
		"model":"general",
		"messages":[{"role":"user","content":"hello"}],
		"max_completion_tokens":80,
		"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}}],
		"response_format":{"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}}}
	}`))
	if !decoded.OK || decoded.Value == nil {
		t.Fatalf("decode = %#v", decoded)
	}

	responses, err := service.EncodeResponsesRequest(context.Background(), *decoded.Value)
	if err != nil || !responses.OK || responses.Value == nil {
		t.Fatalf("Responses result = %#v, error = %v", responses, err)
	}
	var responsesValue map[string]any
	if err := json.Unmarshal(*responses.Value, &responsesValue); err != nil {
		t.Fatal(err)
	}
	if responsesValue["model"] != "openai-target" || responsesValue["max_output_tokens"] != float64(80) {
		t.Fatalf("Responses value = %#v", responsesValue)
	}

	anthropic, err := service.EncodeAnthropicRequest(context.Background(), *decoded.Value)
	if err != nil || !anthropic.OK || anthropic.Value == nil {
		t.Fatalf("Anthropic result = %#v, error = %v", anthropic, err)
	}
	var anthropicValue map[string]any
	if err := json.Unmarshal(*anthropic.Value, &anthropicValue); err != nil {
		t.Fatal(err)
	}
	if anthropicValue["model"] != "anthropic-target" || anthropicValue["max_tokens"] != float64(80) {
		t.Fatalf("Anthropic value = %#v", anthropicValue)
	}
}

func TestProtocolTransformerReportsMissingRoute(t *testing.T) {
	service, err := New(Config{})
	if err != nil {
		t.Fatal(err)
	}
	result, err := service.EncodeResponsesRequest(context.Background(), CanonicalRequest{ModelAlias: "missing"})
	if err != nil {
		t.Fatal(err)
	}
	if result.OK || len(result.Diagnostics) != 1 || result.Diagnostics[0].Code != DiagnosticModelMappingMissing {
		t.Fatalf("result = %#v", result)
	}
}

type blockingResolver struct {
	started  chan struct{}
	canceled chan struct{}
}

func (r *blockingResolver) ResolveForResponses(ctx context.Context, _ AssetSource) (ResolvedAsset, error) {
	close(r.started)
	<-ctx.Done()
	close(r.canceled)
	return ResolvedAsset{}, ctx.Err()
}

func (r *blockingResolver) ResolveForAnthropic(ctx context.Context, source AssetSource) (ResolvedAsset, error) {
	return r.ResolveForResponses(ctx, source)
}

func TestProtocolTransformerPropagatesContextCancellation(t *testing.T) {
	resolver := &blockingResolver{started: make(chan struct{}), canceled: make(chan struct{})}
	service, err := New(Config{
		Resolver: resolver,
		Profiles: []CapabilityProfile{{
			Provider: ProviderOpenAI,
			Endpoint: EndpointResponses,
			Model:    "target",
			Images:   ImageCapabilities{URL: true},
			Content:  ContentCapabilities{Image: true},
		}},
		Routes: []ModelRoute{{Alias: "general", Targets: map[Endpoint]string{EndpointResponses: "target"}}},
	})
	if err != nil {
		t.Fatal(err)
	}

	request := CanonicalRequest{
		ModelAlias: "general",
		Turns: []Turn{{
			Kind: TurnMessage,
			Role: RoleUser,
			Content: []Part{{
				Kind:   PartImage,
				Source: &AssetSource{Kind: AssetSourceURL, URL: "https://example.com/image.png"},
			}},
		}},
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, encodeErr := service.EncodeResponsesRequest(ctx, request)
		done <- encodeErr
	}()

	select {
	case <-resolver.started:
	case <-time.After(time.Second):
		t.Fatal("resolver did not start")
	}
	cancel()

	select {
	case encodeErr := <-done:
		if !errors.Is(encodeErr, context.Canceled) {
			t.Fatalf("error = %v", encodeErr)
		}
	case <-time.After(time.Second):
		t.Fatal("encoding did not stop after cancellation")
	}
	select {
	case <-resolver.canceled:
	default:
		t.Fatal("resolver did not observe cancellation")
	}
}
