package canonical

import "testing"

func TestSuccess(t *testing.T) {
	tests := []struct {
		name         string
		diagnostics  []Diagnostic
		wantOK       bool
		wantLossless bool
		wantValue    bool
	}{
		{
			name:         "lossless",
			wantOK:       true,
			wantLossless: true,
			wantValue:    true,
		},
		{
			name: "warning is lossy",
			diagnostics: []Diagnostic{
				{Severity: SeverityWarning, Code: DiagnosticResponseFormatLossy},
			},
			wantOK:    true,
			wantValue: true,
		},
		{
			name: "error becomes failure",
			diagnostics: []Diagnostic{
				{Severity: SeverityError, Code: DiagnosticOrphanToolResult},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := Success("value", test.diagnostics)
			if result.OK != test.wantOK {
				t.Errorf("OK = %v, want %v", result.OK, test.wantOK)
			}

			if result.Lossless != test.wantLossless {
				t.Errorf("Lossless = %v, want %v", result.Lossless, test.wantLossless)
			}

			if (result.Value != nil) != test.wantValue {
				t.Errorf("Value presence = %v, want %v", result.Value != nil, test.wantValue)
			}
		})
	}
}

func TestFailure(t *testing.T) {
	diagnostic := Diagnostic{Severity: SeverityError, Code: DiagnosticModelMappingMissing}
	result := Failure[string]([]Diagnostic{diagnostic})

	if result.OK {
		t.Error("OK = true, want false")
	}

	if result.Lossless {
		t.Error("Lossless = true, want false")
	}

	if result.Value != nil {
		t.Errorf("Value = %#v, want nil", result.Value)
	}

	if !HasErrors(result.Diagnostics) {
		t.Error("HasErrors() = false, want true")
	}
}
