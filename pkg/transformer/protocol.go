package transformer

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"chat-completion-transformer/internal/anthropicmessages"
	"chat-completion-transformer/internal/canonical"
	"chat-completion-transformer/internal/capabilities"
	"chat-completion-transformer/internal/chatcompletions"
	"chat-completion-transformer/internal/openairesponses"
)

var ErrNotInitialized = errors.New("transformer is not initialized")

const diagnosticCapabilityProfileMissing canonical.DiagnosticCode = "capability_profile_missing"

// ProtocolTransformer is the complete non-transport protocol conversion API.
// Context-bearing methods return context cancellation as a Go error; expected
// protocol incompatibilities remain structured transformation diagnostics.
type ProtocolTransformer interface {
	DecodeChatRequest([]byte) Result[CanonicalRequest]
	EncodeResponsesRequest(context.Context, CanonicalRequest) (Result[json.RawMessage], error)
	EncodeAnthropicRequest(context.Context, CanonicalRequest) (Result[json.RawMessage], error)
	DecodeResponsesResponse([]byte) Result[CanonicalResponse]
	DecodeAnthropicResponse([]byte) Result[CanonicalResponse]
	EncodeChatResponse(CanonicalResponse) Result[json.RawMessage]
	CreateResponsesStreamDecoder() *ResponsesStreamDecoder
	CreateAnthropicStreamDecoder() *AnthropicStreamDecoder
	CreateChatStreamEncoder(StreamEncoderOptions) *ChatStreamEncoder
}

var _ ProtocolTransformer = (*Transformer)(nil)

func (t *Transformer) DecodeChatRequest(input []byte) Result[CanonicalRequest] {
	if t == nil {
		return canonical.Failure[canonical.Request]([]canonical.Diagnostic{notInitializedDiagnostic()})
	}
	return chatcompletions.DecodeRequest(input)
}

func (t *Transformer) EncodeResponsesRequest(ctx context.Context, request CanonicalRequest) (Result[json.RawMessage], error) {
	if err := validateContext(ctx); err != nil {
		return Result[json.RawMessage]{}, err
	}
	if t == nil || t.registry == nil {
		return Result[json.RawMessage]{}, ErrNotInitialized
	}

	model, profile, failure := t.resolveTarget(request.ModelAlias, capabilities.ProviderOpenAI, capabilities.EndpointResponses)
	if failure != nil {
		return *failure, nil
	}
	result := openairesponses.EncodeRequest(ctx, request, openairesponses.EncodeOptions{
		TargetModel:       model,
		Mode:              t.mode,
		Profile:           profile,
		Resolver:          t.resolver,
		InstructionPolicy: openairesponses.InstructionPolicy(t.instructionPolicy),
	})
	if err := ctx.Err(); err != nil {
		return Result[json.RawMessage]{}, err
	}
	if !result.OK || result.Value == nil {
		return canonical.Failure[json.RawMessage](result.Diagnostics), nil
	}

	encoded, err := json.Marshal(*result.Value)
	if err != nil {
		return Result[json.RawMessage]{}, fmt.Errorf("encode Responses request result: %w", err)
	}
	raw := json.RawMessage(encoded)
	return canonical.Success(raw, result.Diagnostics), nil
}

func (t *Transformer) EncodeAnthropicRequest(ctx context.Context, request CanonicalRequest) (Result[json.RawMessage], error) {
	if err := validateContext(ctx); err != nil {
		return Result[json.RawMessage]{}, err
	}
	if t == nil || t.registry == nil {
		return Result[json.RawMessage]{}, ErrNotInitialized
	}

	model, profile, failure := t.resolveTarget(request.ModelAlias, capabilities.ProviderAnthropic, t.anthropicEndpoint)
	if failure != nil {
		return *failure, nil
	}
	defaultMax := 0
	if t.defaultMaxOutputTokens != nil {
		defaultMax = *t.defaultMaxOutputTokens
	}
	result := anthropicmessages.EncodeRequest(ctx, request, anthropicmessages.RequestEncodeOptions{
		TargetModel:            model,
		Mode:                   t.mode,
		Profile:                profile,
		DefaultMaxOutputTokens: defaultMax,
		Resolver:               t.resolver,
	})
	if err := ctx.Err(); err != nil {
		return Result[json.RawMessage]{}, err
	}
	return result, nil
}

func (t *Transformer) DecodeResponsesResponse(input []byte) Result[CanonicalResponse] {
	if t == nil {
		return canonical.Failure[canonical.Response]([]canonical.Diagnostic{notInitializedDiagnostic()})
	}
	return openairesponses.DecodeResponse(input)
}

func (t *Transformer) DecodeAnthropicResponse(input []byte) Result[CanonicalResponse] {
	if t == nil {
		return canonical.Failure[canonical.Response]([]canonical.Diagnostic{notInitializedDiagnostic()})
	}
	return anthropicmessages.DecodeResponse(input)
}

func (t *Transformer) EncodeChatResponse(response CanonicalResponse) Result[json.RawMessage] {
	if t == nil {
		return canonical.Failure[json.RawMessage]([]canonical.Diagnostic{notInitializedDiagnostic()})
	}
	return chatcompletions.EncodeResponse(response, chatcompletions.ResponseEncodeOptions{Mode: t.mode})
}

func (t *Transformer) resolveTarget(
	alias string,
	provider capabilities.Provider,
	endpoint capabilities.Endpoint,
) (string, capabilities.Profile, *canonical.Result[json.RawMessage]) {
	model, err := t.registry.Resolve(alias, endpoint)
	if err != nil {
		failure := canonical.Failure[json.RawMessage]([]canonical.Diagnostic{
			transformDiagnostic(canonical.DiagnosticModelMappingMissing, err.Error(), "model_alias", alias),
		})
		return "", capabilities.Profile{}, &failure
	}
	profile, err := t.registry.Profile(provider, endpoint, model)
	if err != nil {
		failure := canonical.Failure[json.RawMessage]([]canonical.Diagnostic{
			transformDiagnostic(diagnosticCapabilityProfileMissing, err.Error(), "model_alias", alias),
		})
		return "", capabilities.Profile{}, &failure
	}
	return model, profile, nil
}

func validateContext(ctx context.Context) error {
	if ctx == nil {
		return errors.New("context is required")
	}
	return ctx.Err()
}

func notInitializedDiagnostic() canonical.Diagnostic {
	return transformDiagnostic("not_initialized", ErrNotInitialized.Error(), "", nil)
}

func transformDiagnostic(code canonical.DiagnosticCode, message, path string, source any) canonical.Diagnostic {
	diagnostic := canonical.Diagnostic{Severity: canonical.SeverityError, Code: code, Message: message}
	if path != "" {
		diagnostic.Path = &path
	}
	if source != nil {
		encoded, _ := json.Marshal(source)
		diagnostic.SourceValue = encoded
	}
	return diagnostic
}
