package canonical

import "testing"

func TestValidateToolHistory(t *testing.T) {
	tests := []struct {
		name  string
		turns []Turn
		want  map[DiagnosticCode]int
	}{
		{
			name: "valid single call",
			turns: []Turn{
				messageTurn(RoleUser),
				assistantCallTurn("call_1"),
				toolResultTurn("call_1"),
			},
			want: map[DiagnosticCode]int{},
		},
		{
			name: "valid parallel calls",
			turns: []Turn{
				assistantCallTurn("call_1", "call_2"),
				toolResultTurn("call_1", "call_2"),
			},
			want: map[DiagnosticCode]int{},
		},
		{
			name: "duplicate call ID and result",
			turns: []Turn{
				assistantCallTurn("call_1", "call_1"),
				toolResultTurn("call_1", "call_1"),
			},
			want: map[DiagnosticCode]int{
				DiagnosticDuplicateToolCallID: 1,
				DiagnosticDuplicateToolResult: 1,
			},
		},
		{
			name: "duplicate result",
			turns: []Turn{
				assistantCallTurn("call_1"),
				toolResultTurn("call_1", "call_1"),
			},
			want: map[DiagnosticCode]int{
				DiagnosticDuplicateToolResult: 1,
			},
		},
		{
			name: "orphan result",
			turns: []Turn{
				toolResultTurn("missing"),
			},
			want: map[DiagnosticCode]int{
				DiagnosticOrphanToolResult: 1,
			},
		},
		{
			name: "missing result at end",
			turns: []Turn{
				assistantCallTurn("call_1"),
			},
			want: map[DiagnosticCode]int{
				DiagnosticToolResultNotAdjacent: 1,
				DiagnosticMissingToolResult:     1,
			},
		},
		{
			name: "incomplete parallel results",
			turns: []Turn{
				assistantCallTurn("call_1", "call_2"),
				toolResultTurn("call_1"),
			},
			want: map[DiagnosticCode]int{
				DiagnosticMissingToolResult: 1,
			},
		},
		{
			name: "result is not adjacent",
			turns: []Turn{
				assistantCallTurn("call_1"),
				messageTurn(RoleUser),
				toolResultTurn("call_1"),
			},
			want: map[DiagnosticCode]int{
				DiagnosticToolResultNotAdjacent: 2,
				DiagnosticMissingToolResult:     1,
			},
		},
		{
			name: "wrong result follows call",
			turns: []Turn{
				assistantCallTurn("call_1"),
				toolResultTurn("other"),
			},
			want: map[DiagnosticCode]int{
				DiagnosticOrphanToolResult:  1,
				DiagnosticMissingToolResult: 1,
			},
		},
		{
			name: "old call cannot satisfy new turn",
			turns: []Turn{
				assistantCallTurn("old"),
				toolResultTurn("old"),
				assistantCallTurn("new"),
				toolResultTurn("old"),
			},
			want: map[DiagnosticCode]int{
				DiagnosticDuplicateToolResult:   1,
				DiagnosticToolResultNotAdjacent: 1,
				DiagnosticMissingToolResult:     1,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			diagnostics := ValidateToolHistory(test.turns)
			got := diagnosticCodeCounts(diagnostics)
			if !equalCodeCounts(got, test.want) {
				t.Fatalf("diagnostic counts = %#v, want %#v; diagnostics = %#v", got, test.want, diagnostics)
			}

			for _, diagnostic := range diagnostics {
				if diagnostic.Severity != SeverityError {
					t.Errorf("diagnostic severity = %q, want %q", diagnostic.Severity, SeverityError)
				}

				if diagnostic.Path == nil || *diagnostic.Path == "" {
					t.Errorf("diagnostic path is empty: %#v", diagnostic)
				}
			}
		})
	}
}

func messageTurn(role Role) Turn {
	return Turn{Kind: TurnMessage, Role: role}
}

func assistantCallTurn(callIDs ...string) Turn {
	calls := make([]ToolCall, 0, len(callIDs))
	for _, callID := range callIDs {
		calls = append(calls, ToolCall{ID: callID})
	}

	return Turn{
		Kind:      TurnMessage,
		Role:      RoleAssistant,
		ToolCalls: calls,
	}
}

func toolResultTurn(callIDs ...string) Turn {
	results := make([]ToolResult, 0, len(callIDs))
	for _, callID := range callIDs {
		results = append(results, ToolResult{CallID: callID})
	}

	return Turn{
		Kind:    TurnToolResults,
		Results: results,
	}
}

func diagnosticCodeCounts(diagnostics []Diagnostic) map[DiagnosticCode]int {
	counts := make(map[DiagnosticCode]int)
	for _, diagnostic := range diagnostics {
		counts[diagnostic.Code]++
	}

	return counts
}

func equalCodeCounts(left map[DiagnosticCode]int, right map[DiagnosticCode]int) bool {
	if len(left) != len(right) {
		return false
	}

	for code, count := range left {
		if right[code] != count {
			return false
		}
	}

	return true
}
