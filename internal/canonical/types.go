package canonical

import "encoding/json"

// Object stores each JSON property as its original encoded value.
type Object map[string]json.RawMessage

// Role is a protocol-independent message role.
type Role string

const (
	RoleSystem    Role = "system"
	RoleDeveloper Role = "developer"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// AssetSourceKind identifies how an asset is addressed.
type AssetSourceKind string

const (
	AssetSourceURL    AssetSourceKind = "url"
	AssetSourceBase64 AssetSourceKind = "base64"
	AssetSourceFileID AssetSourceKind = "file_id"
)

// AssetSource is a URL, inline base64 value, or provider file identifier.
// Fields relevant to Kind are required by the codec constructing the value.
type AssetSource struct {
	Kind      AssetSourceKind `json:"kind"`
	URL       string          `json:"url,omitempty"`
	MediaType string          `json:"media_type,omitempty"`
	Data      string          `json:"data,omitempty"`
	FileID    string          `json:"file_id,omitempty"`
}

// ImageDetail is the requested image fidelity.
type ImageDetail string

const (
	ImageDetailAuto ImageDetail = "auto"
	ImageDetailLow  ImageDetail = "low"
	ImageDetailHigh ImageDetail = "high"
)

// PartKind identifies a canonical content part.
type PartKind string

const (
	PartText    PartKind = "text"
	PartImage   PartKind = "image"
	PartAudio   PartKind = "audio"
	PartFile    PartKind = "file"
	PartRefusal PartKind = "refusal"
	PartOpaque  PartKind = "opaque"
)

// Part is one unit of message or tool-result content.
// Value retains the unmodified JSON for opaque provider content.
type Part struct {
	Kind     PartKind        `json:"kind"`
	Text     string          `json:"text,omitempty"`
	Source   *AssetSource    `json:"source,omitempty"`
	Detail   *ImageDetail    `json:"detail,omitempty"`
	Filename *string         `json:"filename,omitempty"`
	Provider string          `json:"provider,omitempty"`
	Value    json.RawMessage `json:"value,omitempty"`
}

// ToolCall is a model-requested function invocation.
// ArgumentsRaw is authoritative; ArgumentsParsed is populated only when the
// raw value is a valid JSON object.
type ToolCall struct {
	ID              string `json:"id"`
	Name            string `json:"name"`
	ArgumentsRaw    string `json:"arguments_raw"`
	ArgumentsParsed Object `json:"arguments_parsed,omitempty"`
}

// ToolResult is the result associated with one ToolCall.
type ToolResult struct {
	CallID  string `json:"call_id"`
	Content []Part `json:"content"`
	IsError *bool  `json:"is_error,omitempty"`
}

// TurnKind distinguishes ordinary messages from grouped tool results.
type TurnKind string

const (
	TurnMessage     TurnKind = "message"
	TurnToolResults TurnKind = "tool_results"
)

// Turn is a canonical conversation turn. Assistant messages may contain both
// Content and ToolCalls. Consecutive source tool messages belong in one
// TurnToolResults value.
type Turn struct {
	Kind      TurnKind     `json:"kind"`
	Role      Role         `json:"role,omitempty"`
	Content   []Part       `json:"content,omitempty"`
	ToolCalls []ToolCall   `json:"tool_calls,omitempty"`
	Name      *string      `json:"name,omitempty"`
	Results   []ToolResult `json:"results,omitempty"`
}

// ToolDefinition describes a callable function.
type ToolDefinition struct {
	Name        string  `json:"name"`
	Description *string `json:"description,omitempty"`
	InputSchema Object  `json:"input_schema"`
	Strict      *bool   `json:"strict,omitempty"`
}

// ToolChoiceMode controls whether and which tool may be selected.
type ToolChoiceMode string

const (
	ToolChoiceAuto     ToolChoiceMode = "auto"
	ToolChoiceNone     ToolChoiceMode = "none"
	ToolChoiceRequired ToolChoiceMode = "required"
	ToolChoiceNamed    ToolChoiceMode = "named"
)

// ToolChoice is a protocol-independent tool selection policy.
type ToolChoice struct {
	Mode ToolChoiceMode `json:"mode"`
	Name *string        `json:"name,omitempty"`
}

// OutputFormatType identifies the requested output encoding.
type OutputFormatType string

const (
	OutputFormatText       OutputFormatType = "text"
	OutputFormatJSONObject OutputFormatType = "json_object"
	OutputFormatJSONSchema OutputFormatType = "json_schema"
)

// OutputFormat describes plain text or structured model output.
type OutputFormat struct {
	Type        OutputFormatType `json:"type"`
	Name        *string          `json:"name,omitempty"`
	Description *string          `json:"description,omitempty"`
	Schema      Object           `json:"schema,omitempty"`
	Strict      *bool            `json:"strict,omitempty"`
}

// Request is the canonical form of a chat-completion request.
type Request struct {
	ModelAlias         string            `json:"model_alias"`
	Turns              []Turn            `json:"turns"`
	Tools              []ToolDefinition  `json:"tools"`
	ToolChoice         *ToolChoice       `json:"tool_choice,omitempty"`
	ParallelToolCalls  *bool             `json:"parallel_tool_calls,omitempty"`
	MaxOutputTokens    *int              `json:"max_output_tokens,omitempty"`
	Temperature        *float64          `json:"temperature,omitempty"`
	TopP               *float64          `json:"top_p,omitempty"`
	StopSequences      []string          `json:"stop_sequences,omitempty"`
	CandidateCount     *int              `json:"candidate_count,omitempty"`
	OutputFormat       *OutputFormat     `json:"output_format,omitempty"`
	Stream             bool              `json:"stream"`
	StreamIncludeUsage bool              `json:"stream_include_usage"`
	Metadata           map[string]string `json:"metadata,omitempty"`
	Extensions         Object            `json:"extensions,omitempty"`
}

// Usage contains provider-reported token counts. A nil field means the
// provider did not report that count.
type Usage struct {
	InputTokens  *int64 `json:"input_tokens,omitempty"`
	OutputTokens *int64 `json:"output_tokens,omitempty"`
	TotalTokens  *int64 `json:"total_tokens,omitempty"`
	Extensions   Object `json:"extensions,omitempty"`
}

// FinishReason is a normalized model stop reason.
type FinishReason string

const (
	FinishReasonStop          FinishReason = "stop"
	FinishReasonLength        FinishReason = "length"
	FinishReasonToolCalls     FinishReason = "tool_calls"
	FinishReasonContentFilter FinishReason = "content_filter"
	FinishReasonRefusal       FinishReason = "refusal"
	FinishReasonPause         FinishReason = "pause"
	FinishReasonError         FinishReason = "error"
	FinishReasonUnknown       FinishReason = "unknown"
)

// Output is one complete response candidate.
type Output struct {
	Index          int               `json:"index"`
	Content        []Part            `json:"content,omitempty"`
	ToolCalls      []ToolCall        `json:"tool_calls,omitempty"`
	FinishReason   FinishReason      `json:"finish_reason"`
	ProviderReason *string           `json:"provider_reason,omitempty"`
	ProviderItems  []json.RawMessage `json:"provider_items,omitempty"`
	Extensions     Object            `json:"extensions,omitempty"`
}

// Response is the canonical form of a complete provider response.
type Response struct {
	ID         string   `json:"id"`
	Model      *string  `json:"model,omitempty"`
	CreatedAt  *int64   `json:"created_at,omitempty"`
	Outputs    []Output `json:"outputs"`
	Usage      *Usage   `json:"usage,omitempty"`
	Extensions Object   `json:"extensions,omitempty"`
}

// EventType identifies a normalized streaming event.
type EventType string

const (
	EventResponseStart      EventType = "response_start"
	EventTextDelta          EventType = "text_delta"
	EventRefusalDelta       EventType = "refusal_delta"
	EventToolCallStart      EventType = "tool_call_start"
	EventToolArgumentsDelta EventType = "tool_arguments_delta"
	EventToolCallEnd        EventType = "tool_call_end"
	EventUsage              EventType = "usage"
	EventFinish             EventType = "finish"
	EventError              EventType = "error"
	EventOpaque             EventType = "opaque"
)

// Event is a provider-neutral streaming event. Fields required by an event
// variant are selected by Type. Provider and Value retain unknown events.
type Event struct {
	Type           EventType       `json:"type"`
	ID             string          `json:"id,omitempty"`
	Model          *string         `json:"model,omitempty"`
	CreatedAt      *int64          `json:"created_at,omitempty"`
	OutputIndex    *int            `json:"output_index,omitempty"`
	Delta          string          `json:"delta,omitempty"`
	CallID         string          `json:"call_id,omitempty"`
	Name           string          `json:"name,omitempty"`
	Usage          *Usage          `json:"usage,omitempty"`
	Reason         *FinishReason   `json:"reason,omitempty"`
	ProviderReason *string         `json:"provider_reason,omitempty"`
	Error          json.RawMessage `json:"error,omitempty"`
	Provider       *string         `json:"provider,omitempty"`
	Value          json.RawMessage `json:"value,omitempty"`
}
