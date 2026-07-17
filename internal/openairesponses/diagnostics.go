package openairesponses

import (
	"encoding/json"

	"chat-completion-transformer/internal/canonical"
)

const (
	DiagnosticInvalidResponse           canonical.DiagnosticCode = "invalid_responses_response"
	DiagnosticInvalidStreamEvent        canonical.DiagnosticCode = "invalid_responses_stream_event"
	DiagnosticInvalidStreamState        canonical.DiagnosticCode = "invalid_responses_stream_state"
	DiagnosticUnsupportedRequestField   canonical.DiagnosticCode = "unsupported_responses_request_field"
	DiagnosticUnsupportedExtension      canonical.DiagnosticCode = "unsupported_responses_extension"
	DiagnosticAssetResolutionFailed     canonical.DiagnosticCode = "responses_asset_resolution_failed"
	DiagnosticInstructionPolicyFallback canonical.DiagnosticCode = "responses_instruction_policy_fallback"
	DiagnosticInvalidEncodeOptions      canonical.DiagnosticCode = "invalid_responses_encode_options"
	DiagnosticRequestCanceled           canonical.DiagnosticCode = "responses_request_canceled"
	DiagnosticUnsupportedSchema         canonical.DiagnosticCode = "unsupported_responses_schema"
)

func diagnostic(
	severity canonical.Severity,
	code canonical.DiagnosticCode,
	message string,
	path string,
	source any,
) canonical.Diagnostic {
	var pathValue *string
	if path != "" {
		pathValue = &path
	}

	return canonical.Diagnostic{
		Severity:    severity,
		Code:        code,
		Message:     message,
		Path:        pathValue,
		SourceValue: rawValue(source),
	}
}

func lossyDiagnostic(
	mode canonical.Mode,
	code canonical.DiagnosticCode,
	message string,
	path string,
	source any,
) canonical.Diagnostic {
	severity := canonical.SeverityWarning
	if mode == canonical.ModeStrict {
		severity = canonical.SeverityError
	}

	return diagnostic(severity, code, message, path, source)
}

func rawValue(value any) json.RawMessage {
	if value == nil {
		return nil
	}
	if raw, ok := value.(json.RawMessage); ok {
		return cloneRaw(raw)
	}
	if raw, ok := value.([]byte); ok {
		return cloneRaw(raw)
	}

	raw, err := json.Marshal(value)
	if err != nil {
		return nil
	}
	return raw
}

func cloneRaw(raw []byte) json.RawMessage {
	if len(raw) == 0 {
		return nil
	}
	return append(json.RawMessage(nil), raw...)
}
