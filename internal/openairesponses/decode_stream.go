package openairesponses

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"chat-completion-transformer/internal/canonical"
	"chat-completion-transformer/internal/sse"
)

type streamPhase uint8

const (
	streamInitial streamPhase = iota
	streamStarted
	streamTerminal
)

type contentStreamKey struct {
	outputIndex  int
	contentIndex int
}

type streamedToolCall struct {
	itemID      string
	outputIndex int
	callID      string
	name        string
	arguments   string
	ended       bool
}

// StreamDecoder incrementally converts arbitrary SSE byte chunks into
// canonical events. It keeps all state within one upstream response and must
// not be shared by concurrent streams.
type StreamDecoder struct {
	decoder       *sse.Decoder
	phase         streamPhase
	failed        bool
	closed        bool
	text          map[contentStreamKey]string
	refusals      map[contentStreamKey]string
	textDone      map[contentStreamKey]bool
	refusalDone   map[contentStreamKey]bool
	toolsByItemID map[string]*streamedToolCall
	toolsByOutput map[int]*streamedToolCall
	hadToolCall   bool
	hadRefusal    bool
}

// NewStreamDecoder creates a Responses SSE decoder. Non-positive limits use
// the SSE package's safe default.
func NewStreamDecoder(maxEventBytes int) *StreamDecoder {
	return &StreamDecoder{
		decoder:       sse.NewDecoder(maxEventBytes),
		text:          make(map[contentStreamKey]string),
		refusals:      make(map[contentStreamKey]string),
		textDone:      make(map[contentStreamKey]bool),
		refusalDone:   make(map[contentStreamKey]bool),
		toolsByItemID: make(map[string]*streamedToolCall),
		toolsByOutput: make(map[int]*streamedToolCall),
	}
}

// Feed consumes any network chunk and returns all complete canonical events.
func (d *StreamDecoder) Feed(chunk []byte) canonical.Result[[]canonical.Event] {
	if d == nil || d.decoder == nil {
		return streamFailure("stream decoder is not initialized", nil)
	}
	if d.closed {
		return streamFailure("stream decoder is already closed", nil)
	}
	if d.failed {
		return streamFailure("stream decoder cannot continue after a previous error", nil)
	}

	frames, err := d.decoder.Feed(chunk)
	if err != nil {
		d.failed = true
		return streamFailure(fmt.Sprintf("decode Responses SSE framing: %v", err), chunk)
	}

	events := make([]canonical.Event, 0)
	diagnostics := make([]canonical.Diagnostic, 0)
	for _, frame := range frames {
		frameEvents, frameDiagnostics := d.decodeFrame(frame)
		events = append(events, frameEvents...)
		diagnostics = append(diagnostics, frameDiagnostics...)
	}
	if canonical.HasErrors(diagnostics) {
		d.failed = true
		return canonical.Failure[[]canonical.Event](diagnostics)
	}
	return canonical.Success(events, diagnostics)
}

// Close validates both SSE framing and the Responses lifecycle terminal state.
func (d *StreamDecoder) Close() canonical.Result[[]canonical.Event] {
	if d == nil || d.decoder == nil {
		return streamFailure("stream decoder is not initialized", nil)
	}
	if d.closed {
		return streamFailure("stream decoder is already closed", nil)
	}
	d.closed = true
	if d.failed {
		return streamFailure("stream decoder ended after a previous error", nil)
	}
	if _, err := d.decoder.Close(); err != nil {
		return streamFailure(fmt.Sprintf("close Responses SSE framing: %v", err), nil)
	}
	if d.phase != streamTerminal {
		return streamFailure("Responses SSE stream ended without a terminal event", nil)
	}
	return canonical.Success([]canonical.Event{}, nil)
}

func (d *StreamDecoder) decodeFrame(frame sse.Event) ([]canonical.Event, []canonical.Diagnostic) {
	data := []byte(frame.Data)
	if bytes.Equal(bytes.TrimSpace(data), []byte("[DONE]")) {
		if d.phase == streamTerminal {
			return nil, nil
		}
		return nil, []canonical.Diagnostic{streamStateDiagnostic("received [DONE] before a Responses terminal event", data)}
	}

	object, err := canonical.DecodeObject(data)
	if err != nil {
		return nil, []canonical.Diagnostic{diagnostic(
			canonical.SeverityError,
			DiagnosticInvalidStreamEvent,
			fmt.Sprintf("decode Responses stream event: %v", err),
			"",
			data,
		)}
	}

	typeName, typeDiagnostic := streamEventType(object, frame.Name)
	if typeDiagnostic != nil {
		return nil, []canonical.Diagnostic{*typeDiagnostic}
	}

	switch typeName {
	case "response.created":
		return d.decodeCreated(object, data)
	case "response.queued", "response.in_progress":
		if stateDiagnostic := d.requireActive(typeName, data); stateDiagnostic != nil {
			return nil, []canonical.Diagnostic{*stateDiagnostic}
		}
		return []canonical.Event{opaqueStreamEvent(data)}, nil
	case "response.output_text.delta":
		return d.decodeContentDelta(object, data, canonical.EventTextDelta)
	case "response.output_text.done":
		return d.decodeContentDone(object, data, canonical.EventTextDelta)
	case "response.refusal.delta":
		return d.decodeContentDelta(object, data, canonical.EventRefusalDelta)
	case "response.refusal.done":
		return d.decodeContentDone(object, data, canonical.EventRefusalDelta)
	case "response.output_item.added":
		return d.decodeOutputItem(object, data, false)
	case "response.output_item.done":
		return d.decodeOutputItem(object, data, true)
	case "response.function_call_arguments.delta":
		return d.decodeToolArgumentsDelta(object, data)
	case "response.function_call_arguments.done":
		return d.decodeToolArgumentsDone(object, data)
	case "response.usage", "response.usage.done":
		return d.decodeUsageEvent(object, data)
	case "response.completed", "response.failed", "response.incomplete":
		return d.decodeTerminal(typeName, object, data)
	case "error":
		return d.decodeError(object, data)
	default:
		return []canonical.Event{opaqueStreamEvent(data)}, nil
	}
}

func (d *StreamDecoder) decodeCreated(
	object canonical.Object,
	raw json.RawMessage,
) ([]canonical.Event, []canonical.Diagnostic) {
	if d.phase != streamInitial {
		return nil, []canonical.Diagnostic{streamStateDiagnostic("received response.created more than once or after output", raw)}
	}
	response, err := canonical.DecodeObject(object["response"])
	if err != nil {
		return nil, []canonical.Diagnostic{streamEventDiagnostic("response.created must contain a response object", "response", object["response"])}
	}

	diagnostics := make([]canonical.Diagnostic, 0)
	id := streamRequiredString(response["id"], "response.id", &diagnostics)
	model := streamOptionalString(response["model"], "response.model", &diagnostics)
	createdAt := streamOptionalInt64(response["created_at"], "response.created_at", &diagnostics)
	if canonical.HasErrors(diagnostics) {
		return nil, diagnostics
	}

	d.phase = streamStarted
	event := canonical.Event{Type: canonical.EventResponseStart, ID: id, CreatedAt: createdAt}
	if model != "" {
		event.Model = &model
	}
	return []canonical.Event{event}, diagnostics
}

func (d *StreamDecoder) decodeContentDelta(
	object canonical.Object,
	raw json.RawMessage,
	eventType canonical.EventType,
) ([]canonical.Event, []canonical.Diagnostic) {
	if stateDiagnostic := d.requireActive(string(eventType), raw); stateDiagnostic != nil {
		return nil, []canonical.Diagnostic{*stateDiagnostic}
	}

	diagnostics := make([]canonical.Diagnostic, 0)
	outputIndex := streamRequiredInt(object["output_index"], "output_index", &diagnostics)
	contentIndex := streamRequiredInt(object["content_index"], "content_index", &diagnostics)
	delta := streamRequiredStringValue(object["delta"], "delta", &diagnostics)
	if canonical.HasErrors(diagnostics) {
		return nil, diagnostics
	}

	key := contentStreamKey{outputIndex: outputIndex, contentIndex: contentIndex}
	if eventType == canonical.EventTextDelta {
		d.text[key] += delta
	} else {
		d.refusals[key] += delta
		d.hadRefusal = true
	}
	return []canonical.Event{{Type: eventType, OutputIndex: intPointer(outputIndex), Delta: delta}}, diagnostics
}

func (d *StreamDecoder) decodeContentDone(
	object canonical.Object,
	raw json.RawMessage,
	eventType canonical.EventType,
) ([]canonical.Event, []canonical.Diagnostic) {
	if stateDiagnostic := d.requireActive(string(eventType), raw); stateDiagnostic != nil {
		return nil, []canonical.Diagnostic{*stateDiagnostic}
	}

	diagnostics := make([]canonical.Diagnostic, 0)
	outputIndex := streamRequiredInt(object["output_index"], "output_index", &diagnostics)
	contentIndex := streamRequiredInt(object["content_index"], "content_index", &diagnostics)
	field := "text"
	if eventType == canonical.EventRefusalDelta {
		field = "refusal"
	}
	final := streamRequiredStringValue(object[field], field, &diagnostics)
	if canonical.HasErrors(diagnostics) {
		return nil, diagnostics
	}

	key := contentStreamKey{outputIndex: outputIndex, contentIndex: contentIndex}
	accumulated := d.text[key]
	if eventType == canonical.EventRefusalDelta {
		accumulated = d.refusals[key]
		d.hadRefusal = true
	}
	if !strings.HasPrefix(final, accumulated) {
		return nil, []canonical.Diagnostic{streamStateDiagnostic(
			fmt.Sprintf("%s.done value does not match preceding deltas", field),
			raw,
		)}
	}
	tail := strings.TrimPrefix(final, accumulated)
	if eventType == canonical.EventTextDelta {
		d.text[key] = final
		d.textDone[key] = true
	} else {
		d.refusals[key] = final
		d.refusalDone[key] = true
	}
	if tail == "" {
		return nil, diagnostics
	}
	return []canonical.Event{{Type: eventType, OutputIndex: intPointer(outputIndex), Delta: tail}}, diagnostics
}

func (d *StreamDecoder) decodeOutputItem(
	object canonical.Object,
	raw json.RawMessage,
	done bool,
) ([]canonical.Event, []canonical.Diagnostic) {
	if stateDiagnostic := d.requireActive("response.output_item", raw); stateDiagnostic != nil {
		return nil, []canonical.Diagnostic{*stateDiagnostic}
	}

	diagnostics := make([]canonical.Diagnostic, 0)
	outputIndex := streamRequiredInt(object["output_index"], "output_index", &diagnostics)
	item, err := canonical.DecodeObject(object["item"])
	if err != nil {
		diagnostics = append(diagnostics, streamEventDiagnostic("output item must be an object", "item", object["item"]))
		return nil, diagnostics
	}
	typeName := streamRequiredString(item["type"], "item.type", &diagnostics)
	if canonical.HasErrors(diagnostics) {
		return nil, diagnostics
	}
	if typeName != "function_call" {
		if typeName == "message" {
			return nil, diagnostics
		}
		return []canonical.Event{opaqueStreamEvent(raw)}, diagnostics
	}
	if done {
		return d.finishToolFromItem(outputIndex, item, raw, diagnostics)
	}
	return d.startTool(outputIndex, item, raw, diagnostics)
}

func (d *StreamDecoder) startTool(
	outputIndex int,
	item canonical.Object,
	raw json.RawMessage,
	diagnostics []canonical.Diagnostic,
) ([]canonical.Event, []canonical.Diagnostic) {
	itemID := streamRequiredString(item["id"], "item.id", &diagnostics)
	callID := streamRequiredString(item["call_id"], "item.call_id", &diagnostics)
	name := streamRequiredString(item["name"], "item.name", &diagnostics)
	arguments := streamOptionalString(item["arguments"], "item.arguments", &diagnostics)
	if canonical.HasErrors(diagnostics) {
		return nil, diagnostics
	}
	if _, exists := d.toolsByOutput[outputIndex]; exists {
		return nil, []canonical.Diagnostic{streamStateDiagnostic("output index already has a function call", raw)}
	}
	if _, exists := d.toolsByItemID[itemID]; exists {
		return nil, []canonical.Diagnostic{streamStateDiagnostic("function item ID was added more than once", raw)}
	}

	tool := &streamedToolCall{
		itemID:      itemID,
		outputIndex: outputIndex,
		callID:      callID,
		name:        name,
		arguments:   arguments,
	}
	d.toolsByItemID[itemID] = tool
	d.toolsByOutput[outputIndex] = tool
	d.hadToolCall = true
	events := []canonical.Event{{
		Type:        canonical.EventToolCallStart,
		OutputIndex: intPointer(outputIndex),
		CallID:      callID,
		Name:        name,
	}}
	if arguments != "" {
		events = append(events, canonical.Event{
			Type:        canonical.EventToolArgumentsDelta,
			OutputIndex: intPointer(outputIndex),
			Delta:       arguments,
		})
	}
	return events, diagnostics
}

func (d *StreamDecoder) decodeToolArgumentsDelta(
	object canonical.Object,
	raw json.RawMessage,
) ([]canonical.Event, []canonical.Diagnostic) {
	if stateDiagnostic := d.requireActive("response.function_call_arguments.delta", raw); stateDiagnostic != nil {
		return nil, []canonical.Diagnostic{*stateDiagnostic}
	}

	diagnostics := make([]canonical.Diagnostic, 0)
	outputIndex := streamRequiredInt(object["output_index"], "output_index", &diagnostics)
	itemID := streamRequiredString(object["item_id"], "item_id", &diagnostics)
	delta := streamRequiredStringValue(object["delta"], "delta", &diagnostics)
	if canonical.HasErrors(diagnostics) {
		return nil, diagnostics
	}
	tool, stateDiagnostic := d.findTool(itemID, outputIndex, raw)
	if stateDiagnostic != nil {
		return nil, []canonical.Diagnostic{*stateDiagnostic}
	}
	if tool.ended {
		return nil, []canonical.Diagnostic{streamStateDiagnostic("received function arguments after function call end", raw)}
	}
	tool.arguments += delta
	return []canonical.Event{{
		Type:        canonical.EventToolArgumentsDelta,
		OutputIndex: intPointer(outputIndex),
		Delta:       delta,
	}}, diagnostics
}

func (d *StreamDecoder) decodeToolArgumentsDone(
	object canonical.Object,
	raw json.RawMessage,
) ([]canonical.Event, []canonical.Diagnostic) {
	if stateDiagnostic := d.requireActive("response.function_call_arguments.done", raw); stateDiagnostic != nil {
		return nil, []canonical.Diagnostic{*stateDiagnostic}
	}

	diagnostics := make([]canonical.Diagnostic, 0)
	outputIndex := streamRequiredInt(object["output_index"], "output_index", &diagnostics)
	itemID := streamRequiredString(object["item_id"], "item_id", &diagnostics)
	arguments := streamRequiredStringValue(object["arguments"], "arguments", &diagnostics)
	if canonical.HasErrors(diagnostics) {
		return nil, diagnostics
	}
	tool, stateDiagnostic := d.findTool(itemID, outputIndex, raw)
	if stateDiagnostic != nil {
		return nil, []canonical.Diagnostic{*stateDiagnostic}
	}
	return d.finishTool(tool, arguments, raw, diagnostics)
}

func (d *StreamDecoder) finishToolFromItem(
	outputIndex int,
	item canonical.Object,
	raw json.RawMessage,
	diagnostics []canonical.Diagnostic,
) ([]canonical.Event, []canonical.Diagnostic) {
	itemID := streamRequiredString(item["id"], "item.id", &diagnostics)
	arguments := streamRequiredStringValue(item["arguments"], "item.arguments", &diagnostics)
	if canonical.HasErrors(diagnostics) {
		return nil, diagnostics
	}
	tool, stateDiagnostic := d.findTool(itemID, outputIndex, raw)
	if stateDiagnostic != nil {
		return nil, []canonical.Diagnostic{*stateDiagnostic}
	}
	return d.finishTool(tool, arguments, raw, diagnostics)
}

func (d *StreamDecoder) finishTool(
	tool *streamedToolCall,
	finalArguments string,
	raw json.RawMessage,
	diagnostics []canonical.Diagnostic,
) ([]canonical.Event, []canonical.Diagnostic) {
	if tool.ended {
		if tool.arguments != finalArguments {
			return nil, []canonical.Diagnostic{streamStateDiagnostic("final function arguments changed after function call end", raw)}
		}
		return nil, diagnostics
	}
	if !strings.HasPrefix(finalArguments, tool.arguments) {
		return nil, []canonical.Diagnostic{streamStateDiagnostic("final function arguments do not match preceding deltas", raw)}
	}

	events := make([]canonical.Event, 0, 2)
	if tail := strings.TrimPrefix(finalArguments, tool.arguments); tail != "" {
		events = append(events, canonical.Event{
			Type:        canonical.EventToolArgumentsDelta,
			OutputIndex: intPointer(tool.outputIndex),
			Delta:       tail,
		})
	}
	tool.arguments = finalArguments
	tool.ended = true
	events = append(events, canonical.Event{
		Type:        canonical.EventToolCallEnd,
		OutputIndex: intPointer(tool.outputIndex),
	})
	return events, diagnostics
}

func (d *StreamDecoder) findTool(
	itemID string,
	outputIndex int,
	raw json.RawMessage,
) (*streamedToolCall, *canonical.Diagnostic) {
	tool := d.toolsByItemID[itemID]
	if tool == nil {
		return nil, diagnosticPointer(streamStateDiagnostic("received function arguments before function item was added", raw))
	}
	if tool.outputIndex != outputIndex || d.toolsByOutput[outputIndex] != tool {
		return nil, diagnosticPointer(streamStateDiagnostic("function item ID and output index do not identify the same call", raw))
	}
	return tool, nil
}

func (d *StreamDecoder) decodeUsageEvent(
	object canonical.Object,
	raw json.RawMessage,
) ([]canonical.Event, []canonical.Diagnostic) {
	if stateDiagnostic := d.requireActive("response.usage", raw); stateDiagnostic != nil {
		return nil, []canonical.Diagnostic{*stateDiagnostic}
	}
	diagnostics := make([]canonical.Diagnostic, 0)
	usage := decodeUsage(object["usage"], "usage", &diagnostics)
	diagnostics = asStreamDiagnostics(diagnostics)
	if canonical.HasErrors(diagnostics) {
		return nil, diagnostics
	}
	return []canonical.Event{{Type: canonical.EventUsage, Usage: usage}}, diagnostics
}

func (d *StreamDecoder) decodeTerminal(
	typeName string,
	object canonical.Object,
	raw json.RawMessage,
) ([]canonical.Event, []canonical.Diagnostic) {
	if stateDiagnostic := d.requireActive(typeName, raw); stateDiagnostic != nil {
		return nil, []canonical.Diagnostic{*stateDiagnostic}
	}
	responseRaw := object["response"]
	full := DecodeResponse(responseRaw)
	if !full.OK || full.Value == nil {
		return nil, asStreamDiagnostics(full.Diagnostics)
	}

	events := make([]canonical.Event, 0)
	reconciliationDiagnostics := make([]canonical.Diagnostic, 0)
	if typeName != "response.failed" {
		events, reconciliationDiagnostics = d.reconcileTerminalResponse(responseRaw)
		if canonical.HasErrors(reconciliationDiagnostics) {
			return nil, reconciliationDiagnostics
		}
	}
	events = append(events, d.closeOpenTools()...)
	if full.Value.Usage != nil {
		events = append(events, canonical.Event{Type: canonical.EventUsage, Usage: full.Value.Usage})
	}
	reason := canonical.FinishReasonUnknown
	var providerReason *string
	if len(full.Value.Outputs) > 0 {
		reason = full.Value.Outputs[0].FinishReason
		providerReason = full.Value.Outputs[0].ProviderReason
	}
	if typeName == "response.completed" {
		if d.hadToolCall {
			reason = canonical.FinishReasonToolCalls
		} else if d.hadRefusal {
			reason = canonical.FinishReasonRefusal
		}
	}
	if typeName == "response.failed" {
		reason = canonical.FinishReasonError
		responseObject, _ := canonical.DecodeObject(responseRaw)
		errorRaw := responseObject["error"]
		if !hasNonNullJSON(errorRaw) {
			errorRaw = raw
		}
		events = append(events, canonical.Event{Type: canonical.EventError, Error: cloneRaw(errorRaw)})
	}
	events = append(events, canonical.Event{
		Type:           canonical.EventFinish,
		Reason:         finishReasonPointer(reason),
		ProviderReason: providerReason,
	})
	d.phase = streamTerminal
	diagnostics := append(asStreamDiagnostics(full.Diagnostics), reconciliationDiagnostics...)
	return events, diagnostics
}

func (d *StreamDecoder) reconcileTerminalResponse(responseRaw json.RawMessage) ([]canonical.Event, []canonical.Diagnostic) {
	response, err := canonical.DecodeObject(responseRaw)
	if err != nil {
		return nil, []canonical.Diagnostic{streamStateDiagnostic("terminal response must be an object", responseRaw)}
	}
	var items []json.RawMessage
	if err := json.Unmarshal(response["output"], &items); err != nil {
		return nil, []canonical.Diagnostic{streamStateDiagnostic("terminal response output must be an array", response["output"])}
	}

	events := make([]canonical.Event, 0)
	diagnostics := make([]canonical.Diagnostic, 0)
	seenText := make(map[contentStreamKey]bool)
	seenRefusals := make(map[contentStreamKey]bool)
	seenTools := make(map[int]bool)
	for outputIndex, itemRaw := range items {
		item, err := canonical.DecodeObject(itemRaw)
		if err != nil {
			continue
		}
		var typeName string
		if err := json.Unmarshal(item["type"], &typeName); err != nil {
			continue
		}
		switch typeName {
		case "message":
			itemEvents, itemDiagnostics := d.reconcileTerminalMessage(outputIndex, item, itemRaw, seenText, seenRefusals)
			events = append(events, itemEvents...)
			diagnostics = append(diagnostics, itemDiagnostics...)
		case "function_call":
			seenTools[outputIndex] = true
			itemEvents, itemDiagnostics := d.reconcileTerminalTool(outputIndex, item, itemRaw)
			events = append(events, itemEvents...)
			diagnostics = append(diagnostics, itemDiagnostics...)
		}
	}

	for key := range d.text {
		if !seenText[key] {
			diagnostics = append(diagnostics, streamStateDiagnostic("streamed text is absent from the terminal response", nil))
		}
	}
	for key := range d.refusals {
		if !seenRefusals[key] {
			diagnostics = append(diagnostics, streamStateDiagnostic("streamed refusal is absent from the terminal response", nil))
		}
	}
	for outputIndex := range d.toolsByOutput {
		if !seenTools[outputIndex] {
			diagnostics = append(diagnostics, streamStateDiagnostic("streamed function call is absent from the terminal response", nil))
		}
	}
	return events, diagnostics
}

func (d *StreamDecoder) reconcileTerminalMessage(
	outputIndex int,
	item canonical.Object,
	itemRaw json.RawMessage,
	seenText map[contentStreamKey]bool,
	seenRefusals map[contentStreamKey]bool,
) ([]canonical.Event, []canonical.Diagnostic) {
	var parts []json.RawMessage
	if err := json.Unmarshal(item["content"], &parts); err != nil {
		return nil, []canonical.Diagnostic{streamStateDiagnostic("terminal message content must be an array", itemRaw)}
	}

	events := make([]canonical.Event, 0)
	diagnostics := make([]canonical.Diagnostic, 0)
	for contentIndex, partRaw := range parts {
		part, err := canonical.DecodeObject(partRaw)
		if err != nil {
			continue
		}
		var typeName string
		if err := json.Unmarshal(part["type"], &typeName); err != nil {
			continue
		}
		key := contentStreamKey{outputIndex: outputIndex, contentIndex: contentIndex}
		var field string
		var eventType canonical.EventType
		switch typeName {
		case "output_text":
			field = "text"
			eventType = canonical.EventTextDelta
			seenText[key] = true
		case "refusal":
			field = "refusal"
			eventType = canonical.EventRefusalDelta
			seenRefusals[key] = true
		default:
			continue
		}
		var final string
		if err := json.Unmarshal(part[field], &final); err != nil {
			diagnostics = append(diagnostics, streamStateDiagnostic("terminal content value must be a string", partRaw))
			continue
		}
		partEvents, partDiagnostics := d.reconcileTerminalContent(key, eventType, final, partRaw)
		events = append(events, partEvents...)
		diagnostics = append(diagnostics, partDiagnostics...)
	}
	return events, diagnostics
}

func (d *StreamDecoder) reconcileTerminalContent(
	key contentStreamKey,
	eventType canonical.EventType,
	final string,
	raw json.RawMessage,
) ([]canonical.Event, []canonical.Diagnostic) {
	accumulated := d.text[key]
	done := d.textDone[key]
	if eventType == canonical.EventRefusalDelta {
		accumulated = d.refusals[key]
		done = d.refusalDone[key]
	}
	if done {
		if accumulated == final {
			return nil, nil
		}
		return nil, []canonical.Diagnostic{streamStateDiagnostic("terminal content does not match its done event", raw)}
	}
	if !strings.HasPrefix(final, accumulated) {
		return nil, []canonical.Diagnostic{streamStateDiagnostic("terminal content does not match preceding deltas", raw)}
	}

	tail := strings.TrimPrefix(final, accumulated)
	if eventType == canonical.EventRefusalDelta {
		d.refusals[key] = final
		d.refusalDone[key] = true
	} else {
		d.text[key] = final
		d.textDone[key] = true
	}
	diagnostics := []canonical.Diagnostic{streamStateWarning("terminal response completed content without a preceding done event", raw)}
	if tail == "" {
		return nil, diagnostics
	}
	return []canonical.Event{{Type: eventType, OutputIndex: intPointer(key.outputIndex), Delta: tail}}, diagnostics
}

func (d *StreamDecoder) reconcileTerminalTool(
	outputIndex int,
	item canonical.Object,
	itemRaw json.RawMessage,
) ([]canonical.Event, []canonical.Diagnostic) {
	var itemID, callID, name, arguments string
	if json.Unmarshal(item["id"], &itemID) != nil ||
		json.Unmarshal(item["call_id"], &callID) != nil ||
		json.Unmarshal(item["name"], &name) != nil ||
		json.Unmarshal(item["arguments"], &arguments) != nil {
		return nil, []canonical.Diagnostic{streamStateDiagnostic("terminal function call is invalid", itemRaw)}
	}

	tool := d.toolsByItemID[itemID]
	if tool == nil {
		tool = d.toolsByOutput[outputIndex]
	}
	if tool == nil {
		tool = &streamedToolCall{itemID: itemID, outputIndex: outputIndex, callID: callID, name: name, arguments: arguments, ended: true}
		d.toolsByItemID[itemID] = tool
		d.toolsByOutput[outputIndex] = tool
		d.hadToolCall = true
		events := []canonical.Event{{Type: canonical.EventToolCallStart, OutputIndex: intPointer(outputIndex), CallID: callID, Name: name}}
		if arguments != "" {
			events = append(events, canonical.Event{Type: canonical.EventToolArgumentsDelta, OutputIndex: intPointer(outputIndex), Delta: arguments})
		}
		events = append(events, canonical.Event{Type: canonical.EventToolCallEnd, OutputIndex: intPointer(outputIndex)})
		return events, []canonical.Diagnostic{streamStateWarning("terminal response completed a function call without preceding item events", itemRaw)}
	}
	if tool.itemID != itemID || tool.outputIndex != outputIndex || tool.callID != callID || tool.name != name {
		return nil, []canonical.Diagnostic{streamStateDiagnostic("terminal function call identity does not match preceding events", itemRaw)}
	}
	if tool.ended {
		if tool.arguments == arguments {
			return nil, nil
		}
		return nil, []canonical.Diagnostic{streamStateDiagnostic("terminal function arguments do not match their done event", itemRaw)}
	}

	events, diagnostics := d.finishTool(tool, arguments, itemRaw, nil)
	if canonical.HasErrors(diagnostics) {
		return nil, diagnostics
	}
	diagnostics = append(diagnostics, streamStateWarning("terminal response completed function arguments without a preceding done event", itemRaw))
	return events, diagnostics
}

func (d *StreamDecoder) decodeError(
	object canonical.Object,
	raw json.RawMessage,
) ([]canonical.Event, []canonical.Diagnostic) {
	if d.phase == streamTerminal {
		return nil, []canonical.Diagnostic{streamStateDiagnostic("received error after a terminal event", raw)}
	}
	diagnostics := make([]canonical.Diagnostic, 0)
	providerReason := streamOptionalString(object["code"], "code", &diagnostics)
	if canonical.HasErrors(diagnostics) {
		return nil, diagnostics
	}
	reason := canonical.FinishReasonError
	events := d.closeOpenTools()
	events = append(events,
		canonical.Event{Type: canonical.EventError, Error: cloneRaw(raw)},
		canonical.Event{Type: canonical.EventFinish, Reason: &reason},
	)
	if providerReason != "" {
		events[len(events)-1].ProviderReason = &providerReason
	}
	d.phase = streamTerminal
	return events, diagnostics
}

func (d *StreamDecoder) closeOpenTools() []canonical.Event {
	indexes := make([]int, 0, len(d.toolsByOutput))
	for index, tool := range d.toolsByOutput {
		if !tool.ended {
			indexes = append(indexes, index)
		}
	}
	sort.Ints(indexes)
	events := make([]canonical.Event, 0, len(indexes))
	for _, index := range indexes {
		d.toolsByOutput[index].ended = true
		events = append(events, canonical.Event{Type: canonical.EventToolCallEnd, OutputIndex: intPointer(index)})
	}
	return events
}

func (d *StreamDecoder) requireActive(eventType string, raw json.RawMessage) *canonical.Diagnostic {
	if d.phase == streamStarted {
		return nil
	}
	if d.phase == streamTerminal {
		diagnostic := streamStateDiagnostic(fmt.Sprintf("received %s after a terminal event", eventType), raw)
		return &diagnostic
	}
	diagnostic := streamStateDiagnostic(fmt.Sprintf("received %s before response.created", eventType), raw)
	return &diagnostic
}

func streamEventType(object canonical.Object, eventName string) (string, *canonical.Diagnostic) {
	typeName := ""
	if raw, exists := object["type"]; exists {
		if err := json.Unmarshal(raw, &typeName); err != nil || typeName == "" {
			diagnostic := streamEventDiagnostic("stream event type must be a non-empty string", "type", raw)
			return "", &diagnostic
		}
	}
	if typeName == "" && eventName != "" && eventName != "message" {
		typeName = eventName
	}
	if typeName == "" {
		diagnostic := streamEventDiagnostic("stream event type is required", "type", nil)
		return "", &diagnostic
	}
	if eventName != "" && eventName != "message" && eventName != typeName {
		diagnostic := streamEventDiagnostic("SSE event name does not match the JSON event type", "type", object["type"])
		return "", &diagnostic
	}
	return typeName, nil
}

func streamRequiredString(raw json.RawMessage, path string, diagnostics *[]canonical.Diagnostic) string {
	if !hasJSONValue(raw) {
		*diagnostics = append(*diagnostics, streamEventDiagnostic(path+" is required", path, raw))
		return ""
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		*diagnostics = append(*diagnostics, streamEventDiagnostic(path+" must be a string", path, raw))
		return ""
	}
	if value == "" {
		*diagnostics = append(*diagnostics, streamEventDiagnostic(path+" must not be empty", path, raw))
	}
	return value
}

func streamRequiredStringValue(raw json.RawMessage, path string, diagnostics *[]canonical.Diagnostic) string {
	if !hasJSONValue(raw) {
		*diagnostics = append(*diagnostics, streamEventDiagnostic(path+" is required", path, raw))
		return ""
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		*diagnostics = append(*diagnostics, streamEventDiagnostic(path+" must be a string", path, raw))
	}
	return value
}

func streamOptionalString(raw json.RawMessage, path string, diagnostics *[]canonical.Diagnostic) string {
	if !hasJSONValue(raw) {
		return ""
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		*diagnostics = append(*diagnostics, streamEventDiagnostic(path+" must be a string", path, raw))
	}
	return value
}

func streamRequiredInt(raw json.RawMessage, path string, diagnostics *[]canonical.Diagnostic) int {
	if !hasJSONValue(raw) {
		*diagnostics = append(*diagnostics, streamEventDiagnostic(path+" is required", path, raw))
		return 0
	}
	var value int
	if err := json.Unmarshal(raw, &value); err != nil || value < 0 {
		*diagnostics = append(*diagnostics, streamEventDiagnostic(path+" must be a non-negative integer", path, raw))
	}
	return value
}

func streamOptionalInt64(raw json.RawMessage, path string, diagnostics *[]canonical.Diagnostic) *int64 {
	if !hasJSONValue(raw) {
		return nil
	}
	var value int64
	if err := json.Unmarshal(raw, &value); err != nil || value < 0 {
		*diagnostics = append(*diagnostics, streamEventDiagnostic(path+" must be a non-negative integer", path, raw))
		return nil
	}
	return &value
}

func streamEventDiagnostic(message string, path string, source any) canonical.Diagnostic {
	return diagnostic(canonical.SeverityError, DiagnosticInvalidStreamEvent, message, path, source)
}

func streamStateDiagnostic(message string, source any) canonical.Diagnostic {
	return diagnostic(canonical.SeverityError, DiagnosticInvalidStreamState, message, "", source)
}

func streamStateWarning(message string, source any) canonical.Diagnostic {
	return diagnostic(canonical.SeverityWarning, DiagnosticInvalidStreamState, message, "", source)
}

func streamFailure(message string, source any) canonical.Result[[]canonical.Event] {
	return canonical.Failure[[]canonical.Event]([]canonical.Diagnostic{
		streamStateDiagnostic(message, source),
	})
}

func opaqueStreamEvent(raw json.RawMessage) canonical.Event {
	provider := "openai.responses"
	return canonical.Event{
		Type:     canonical.EventOpaque,
		Provider: &provider,
		Value:    cloneRaw(raw),
	}
}

func asStreamDiagnostics(diagnostics []canonical.Diagnostic) []canonical.Diagnostic {
	for index := range diagnostics {
		if diagnostics[index].Severity == canonical.SeverityError {
			diagnostics[index].Code = DiagnosticInvalidStreamEvent
		}
	}
	return diagnostics
}

func diagnosticPointer(value canonical.Diagnostic) *canonical.Diagnostic {
	return &value
}

func intPointer(value int) *int {
	return &value
}

func finishReasonPointer(value canonical.FinishReason) *canonical.FinishReason {
	return &value
}
