// Package transformer exposes the protocol-neutral transformer API.
package transformer

import (
	"chat-completion-transformer/internal/assets"
	"chat-completion-transformer/internal/canonical"
	"chat-completion-transformer/internal/capabilities"
)

type (
	Object                  = canonical.Object
	Role                    = canonical.Role
	AssetSourceKind         = canonical.AssetSourceKind
	AssetSource             = canonical.AssetSource
	ImageDetail             = canonical.ImageDetail
	PartKind                = canonical.PartKind
	Part                    = canonical.Part
	ToolCall                = canonical.ToolCall
	ToolResult              = canonical.ToolResult
	TurnKind                = canonical.TurnKind
	Turn                    = canonical.Turn
	ToolDefinition          = canonical.ToolDefinition
	ToolChoiceMode          = canonical.ToolChoiceMode
	ToolChoice              = canonical.ToolChoice
	OutputFormatType        = canonical.OutputFormatType
	OutputFormat            = canonical.OutputFormat
	CanonicalRequest        = canonical.Request
	Usage                   = canonical.Usage
	FinishReason            = canonical.FinishReason
	Output                  = canonical.Output
	CanonicalResponse       = canonical.Response
	EventType               = canonical.EventType
	CanonicalEvent          = canonical.Event
	Mode                    = canonical.Mode
	Severity                = canonical.Severity
	DiagnosticCode          = canonical.DiagnosticCode
	Diagnostic              = canonical.Diagnostic
	CapabilityProfile       = capabilities.Profile
	ImageCapabilities       = capabilities.ImageCapabilities
	ContentCapabilities     = capabilities.ContentCapabilities
	PromptCacheMode         = capabilities.PromptCacheMode
	PromptCacheCapabilities = capabilities.PromptCacheCapabilities
	ModelRoute              = capabilities.ModelRoute
	Provider                = capabilities.Provider
	Endpoint                = capabilities.Endpoint
	AssetResolver           = assets.Resolver
	ResolvedAsset           = assets.ResolvedAsset
	NativeAssetResolver     = assets.NativeResolver
)

type Result[T any] = canonical.Result[T]

const (
	RoleSystem    = canonical.RoleSystem
	RoleDeveloper = canonical.RoleDeveloper
	RoleUser      = canonical.RoleUser
	RoleAssistant = canonical.RoleAssistant

	AssetSourceURL    = canonical.AssetSourceURL
	AssetSourceBase64 = canonical.AssetSourceBase64
	AssetSourceFileID = canonical.AssetSourceFileID

	ImageDetailAuto = canonical.ImageDetailAuto
	ImageDetailLow  = canonical.ImageDetailLow
	ImageDetailHigh = canonical.ImageDetailHigh

	PartText    = canonical.PartText
	PartImage   = canonical.PartImage
	PartAudio   = canonical.PartAudio
	PartFile    = canonical.PartFile
	PartRefusal = canonical.PartRefusal
	PartOpaque  = canonical.PartOpaque

	TurnMessage     = canonical.TurnMessage
	TurnToolResults = canonical.TurnToolResults

	ToolChoiceAuto     = canonical.ToolChoiceAuto
	ToolChoiceNone     = canonical.ToolChoiceNone
	ToolChoiceRequired = canonical.ToolChoiceRequired
	ToolChoiceNamed    = canonical.ToolChoiceNamed

	OutputFormatText       = canonical.OutputFormatText
	OutputFormatJSONObject = canonical.OutputFormatJSONObject
	OutputFormatJSONSchema = canonical.OutputFormatJSONSchema

	FinishReasonStop          = canonical.FinishReasonStop
	FinishReasonLength        = canonical.FinishReasonLength
	FinishReasonToolCalls     = canonical.FinishReasonToolCalls
	FinishReasonContentFilter = canonical.FinishReasonContentFilter
	FinishReasonRefusal       = canonical.FinishReasonRefusal
	FinishReasonPause         = canonical.FinishReasonPause
	FinishReasonError         = canonical.FinishReasonError
	FinishReasonUnknown       = canonical.FinishReasonUnknown

	EventResponseStart      = canonical.EventResponseStart
	EventTextDelta          = canonical.EventTextDelta
	EventRefusalDelta       = canonical.EventRefusalDelta
	EventToolCallStart      = canonical.EventToolCallStart
	EventToolArgumentsDelta = canonical.EventToolArgumentsDelta
	EventToolCallEnd        = canonical.EventToolCallEnd
	EventUsage              = canonical.EventUsage
	EventFinish             = canonical.EventFinish
	EventError              = canonical.EventError
	EventOpaque             = canonical.EventOpaque

	ModeStrict     = canonical.ModeStrict
	ModeCompatible = canonical.ModeCompatible
	ModeEmulate    = canonical.ModeEmulate

	SeverityWarning = canonical.SeverityWarning
	SeverityError   = canonical.SeverityError

	DiagnosticUnsupportedContentPart           = canonical.DiagnosticUnsupportedContentPart
	DiagnosticCandidateCountUnsupported        = canonical.DiagnosticCandidateCountUnsupported
	DiagnosticInvalidToolArgumentsJSON         = canonical.DiagnosticInvalidToolArgumentsJSON
	DiagnosticOrphanToolResult                 = canonical.DiagnosticOrphanToolResult
	DiagnosticDuplicateToolCallID              = canonical.DiagnosticDuplicateToolCallID
	DiagnosticDuplicateToolResult              = canonical.DiagnosticDuplicateToolResult
	DiagnosticToolResultNotAdjacent            = canonical.DiagnosticToolResultNotAdjacent
	DiagnosticMissingToolResult                = canonical.DiagnosticMissingToolResult
	DiagnosticRolePriorityCollapsed            = canonical.DiagnosticRolePriorityCollapsed
	DiagnosticMidConversationSystemUnsupported = canonical.DiagnosticMidConversationSystemUnsupported
	DiagnosticSamplingParameterUnsupported     = canonical.DiagnosticSamplingParameterUnsupported
	DiagnosticResponseFormatLossy              = canonical.DiagnosticResponseFormatLossy
	DiagnosticModelMappingMissing              = canonical.DiagnosticModelMappingMissing
	DiagnosticInvalidCacheControl              = canonical.DiagnosticInvalidCacheControl
	DiagnosticCacheControlUnsupported          = canonical.DiagnosticCacheControlUnsupported
	DiagnosticCacheControlProviderMismatch     = canonical.DiagnosticCacheControlProviderMismatch
	DiagnosticCacheBreakpointUnsupported       = canonical.DiagnosticCacheBreakpointUnsupported
	DiagnosticInvalidCacheUsage                = canonical.DiagnosticInvalidCacheUsage
	UsageExtensionAnthropicCacheCreation       = canonical.UsageExtensionAnthropicCacheCreation

	ProviderOpenAI    = capabilities.ProviderOpenAI
	ProviderAnthropic = capabilities.ProviderAnthropic

	PromptCacheUnset        = capabilities.PromptCacheUnset
	PromptCacheNone         = capabilities.PromptCacheNone
	PromptCacheAnthropic    = capabilities.PromptCacheAnthropic
	PromptCacheOpenAILegacy = capabilities.PromptCacheOpenAILegacy
	PromptCacheOpenAI56     = capabilities.PromptCacheOpenAI56

	EndpointResponses       = capabilities.EndpointResponses
	EndpointMessages        = capabilities.EndpointMessages
	EndpointBedrockMessages = capabilities.EndpointBedrockMessages
	EndpointVertexMessages  = capabilities.EndpointVertexMessages
)
