package canonical

import "encoding/json"

// Mode controls how a lossy transformation is handled.
type Mode string

const (
	ModeStrict     Mode = "strict"
	ModeCompatible Mode = "compatible"
	ModeEmulate    Mode = "emulate"
)

// Severity is the impact of a diagnostic.
type Severity string

const (
	SeverityWarning Severity = "warning"
	SeverityError   Severity = "error"
)

// DiagnosticCode is a stable, machine-readable diagnostic identifier.
type DiagnosticCode string

const (
	DiagnosticUnsupportedContentPart           DiagnosticCode = "unsupported_content_part"
	DiagnosticCandidateCountUnsupported        DiagnosticCode = "candidate_count_unsupported"
	DiagnosticInvalidToolArgumentsJSON         DiagnosticCode = "invalid_tool_arguments_json"
	DiagnosticOrphanToolResult                 DiagnosticCode = "orphan_tool_result"
	DiagnosticDuplicateToolCallID              DiagnosticCode = "duplicate_tool_call_id"
	DiagnosticDuplicateToolResult              DiagnosticCode = "duplicate_tool_result"
	DiagnosticToolResultNotAdjacent            DiagnosticCode = "tool_result_not_adjacent"
	DiagnosticMissingToolResult                DiagnosticCode = "missing_tool_result"
	DiagnosticRolePriorityCollapsed            DiagnosticCode = "role_priority_collapsed"
	DiagnosticMidConversationSystemUnsupported DiagnosticCode = "mid_conversation_system_unsupported"
	DiagnosticSamplingParameterUnsupported     DiagnosticCode = "sampling_parameter_unsupported"
	DiagnosticResponseFormatLossy              DiagnosticCode = "response_format_lossy"
	DiagnosticModelMappingMissing              DiagnosticCode = "model_mapping_missing"
)

// Diagnostic describes a lossy or invalid transformation.
// SourceValue retains the original JSON when one is available.
type Diagnostic struct {
	Severity    Severity        `json:"severity"`
	Code        DiagnosticCode  `json:"code"`
	Message     string          `json:"message"`
	Path        *string         `json:"path,omitempty"`
	SourceValue json.RawMessage `json:"source_value,omitempty"`
}

// Result is the outcome of a protocol transformation.
type Result[T any] struct {
	Value       *T           `json:"value,omitempty"`
	Diagnostics []Diagnostic `json:"diagnostics"`
	Lossless    bool         `json:"lossless"`
	OK          bool         `json:"ok"`
}

// Success returns a successful result. Any diagnostic makes the transform
// lossy; error diagnostics instead produce a failed result.
func Success[T any](value T, diagnostics []Diagnostic) Result[T] {
	if HasErrors(diagnostics) {
		return Failure[T](diagnostics)
	}

	return Result[T]{
		Value:       &value,
		Diagnostics: diagnostics,
		Lossless:    len(diagnostics) == 0,
		OK:          true,
	}
}

// Failure returns a failed transformation result.
func Failure[T any](diagnostics []Diagnostic) Result[T] {
	return Result[T]{
		Diagnostics: diagnostics,
		OK:          false,
	}
}

// HasErrors reports whether diagnostics contains an error.
func HasErrors(diagnostics []Diagnostic) bool {
	for _, diagnostic := range diagnostics {
		if diagnostic.Severity == SeverityError {
			return true
		}
	}

	return false
}
