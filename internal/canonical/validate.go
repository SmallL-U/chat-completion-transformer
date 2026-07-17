package canonical

import "fmt"

// ValidateToolHistory checks the correlation and adjacency invariants required
// by tool-capable chat protocols. A result is valid only when it appears in the
// single tool-results turn immediately following its assistant call turn.
func ValidateToolHistory(turns []Turn) []Diagnostic {
	diagnostics := make([]Diagnostic, 0)
	seenCalls := make(map[string]int)
	seenResults := make(map[string]int)

	for turnIndex, turn := range turns {
		if turn.Kind == TurnMessage {
			diagnostics = validateCallTurn(turns, turnIndex, seenCalls, diagnostics)
			continue
		}

		if turn.Kind != TurnToolResults {
			continue
		}

		diagnostics = validateResultTurn(turns, turnIndex, seenCalls, seenResults, diagnostics)
	}

	return diagnostics
}

func validateCallTurn(
	turns []Turn,
	turnIndex int,
	seenCalls map[string]int,
	diagnostics []Diagnostic,
) []Diagnostic {
	turn := turns[turnIndex]
	if turn.Role != RoleAssistant || len(turn.ToolCalls) == 0 {
		return diagnostics
	}

	for callIndex, call := range turn.ToolCalls {
		firstTurn, exists := seenCalls[call.ID]
		if !exists {
			seenCalls[call.ID] = turnIndex
			continue
		}

		diagnostics = append(diagnostics, Diagnostic{
			Severity: SeverityError,
			Code:     DiagnosticDuplicateToolCallID,
			Message: fmt.Sprintf(
				"tool call ID %q duplicates a call from turn %d",
				call.ID,
				firstTurn,
			),
			Path: pathPointer("turns.%d.tool_calls.%d.id", turnIndex, callIndex),
		})
	}

	if turnIndex+1 < len(turns) && turns[turnIndex+1].Kind == TurnToolResults {
		return diagnostics
	}

	diagnostics = append(diagnostics, Diagnostic{
		Severity: SeverityError,
		Code:     DiagnosticToolResultNotAdjacent,
		Message:  "tool results must immediately follow an assistant tool-call turn",
		Path:     pathPointer("turns.%d", turnIndex),
	})

	for callIndex, call := range turn.ToolCalls {
		diagnostics = append(diagnostics, missingResultDiagnostic(turnIndex, callIndex, call.ID))
	}

	return diagnostics
}

func validateResultTurn(
	turns []Turn,
	turnIndex int,
	seenCalls map[string]int,
	seenResults map[string]int,
	diagnostics []Diagnostic,
) []Diagnostic {
	turn := turns[turnIndex]
	for resultIndex, result := range turn.Results {
		firstTurn, exists := seenResults[result.CallID]
		if !exists {
			seenResults[result.CallID] = turnIndex
			continue
		}

		diagnostics = append(diagnostics, Diagnostic{
			Severity: SeverityError,
			Code:     DiagnosticDuplicateToolResult,
			Message: fmt.Sprintf(
				"tool result for call ID %q duplicates a result from turn %d",
				result.CallID,
				firstTurn,
			),
			Path: pathPointer("turns.%d.results.%d.call_id", turnIndex, resultIndex),
		})
	}

	if !hasAdjacentCallTurn(turns, turnIndex) {
		return validateUnmatchedResults(turn, turnIndex, seenCalls, diagnostics)
	}

	callTurn := turns[turnIndex-1]
	expected := make(map[string]int, len(callTurn.ToolCalls))
	for _, call := range callTurn.ToolCalls {
		expected[call.ID]++
	}

	matched := make(map[string]int, len(expected))
	for resultIndex, result := range turn.Results {
		if matched[result.CallID] < expected[result.CallID] {
			matched[result.CallID]++
			continue
		}

		if expected[result.CallID] > 0 {
			continue
		}

		diagnostics = append(diagnostics, unmatchedResultDiagnostic(
			turnIndex,
			resultIndex,
			result.CallID,
			seenCalls,
		))
	}

	remainingMatches := make(map[string]int, len(matched))
	for callID, count := range matched {
		remainingMatches[callID] = count
	}

	for callIndex, call := range callTurn.ToolCalls {
		if remainingMatches[call.ID] > 0 {
			remainingMatches[call.ID]--
			continue
		}

		diagnostics = append(diagnostics, missingResultDiagnostic(turnIndex-1, callIndex, call.ID))
	}

	return diagnostics
}

func hasAdjacentCallTurn(turns []Turn, resultTurnIndex int) bool {
	if resultTurnIndex == 0 {
		return false
	}

	previous := turns[resultTurnIndex-1]
	return previous.Kind == TurnMessage &&
		previous.Role == RoleAssistant &&
		len(previous.ToolCalls) > 0
}

func validateUnmatchedResults(
	turn Turn,
	turnIndex int,
	seenCalls map[string]int,
	diagnostics []Diagnostic,
) []Diagnostic {
	for resultIndex, result := range turn.Results {
		diagnostics = append(diagnostics, unmatchedResultDiagnostic(
			turnIndex,
			resultIndex,
			result.CallID,
			seenCalls,
		))
	}

	return diagnostics
}

func unmatchedResultDiagnostic(
	turnIndex int,
	resultIndex int,
	callID string,
	seenCalls map[string]int,
) Diagnostic {
	callTurn, exists := seenCalls[callID]
	if !exists {
		return Diagnostic{
			Severity: SeverityError,
			Code:     DiagnosticOrphanToolResult,
			Message:  fmt.Sprintf("tool result %q has no preceding tool call", callID),
			Path:     pathPointer("turns.%d.results.%d.call_id", turnIndex, resultIndex),
		}
	}

	return Diagnostic{
		Severity: SeverityError,
		Code:     DiagnosticToolResultNotAdjacent,
		Message: fmt.Sprintf(
			"tool result %q is not adjacent to its call in turn %d",
			callID,
			callTurn,
		),
		Path: pathPointer("turns.%d.results.%d.call_id", turnIndex, resultIndex),
	}
}

func missingResultDiagnostic(turnIndex int, callIndex int, callID string) Diagnostic {
	return Diagnostic{
		Severity: SeverityError,
		Code:     DiagnosticMissingToolResult,
		Message:  fmt.Sprintf("tool call %q is missing an adjacent result", callID),
		Path:     pathPointer("turns.%d.tool_calls.%d", turnIndex, callIndex),
	}
}

func pathPointer(format string, values ...any) *string {
	path := fmt.Sprintf(format, values...)
	return &path
}
