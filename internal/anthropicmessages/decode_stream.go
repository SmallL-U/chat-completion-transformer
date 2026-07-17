package anthropicmessages

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"chat-completion-transformer/internal/canonical"
	"chat-completion-transformer/internal/sse"
)

const (
	diagnosticInvalidStreamJSON canonical.DiagnosticCode = "invalid_anthropic_stream_event"
	diagnosticStreamState       canonical.DiagnosticCode = "invalid_anthropic_stream_state"
	diagnosticStreamTruncated   canonical.DiagnosticCode = "anthropic_stream_truncated"
)

type streamBlock struct {
	kind         string
	initialInput string
	arguments    strings.Builder
}

// StreamDecoder incrementally converts one Anthropic Messages SSE stream into
// canonical events. It must not be shared by concurrent upstream responses.
type StreamDecoder struct {
	frames           *sse.Decoder
	blocks           map[int]*streamBlock
	started          bool
	stopped          bool
	closed           bool
	failed           bool
	usage            canonical.Usage
	pendingReason    *canonical.FinishReason
	providerReason   *string
	finishExtensions canonical.Object
}

func NewStreamDecoder(maxEventBytes int) *StreamDecoder {
	return &StreamDecoder{
		frames: sse.NewDecoder(maxEventBytes),
		blocks: make(map[int]*streamBlock),
	}
}

// Feed accepts arbitrary network chunks and returns all complete canonical
// events decoded from them.
func (d *StreamDecoder) Feed(chunk []byte) canonical.Result[[]canonical.Event] {
	if d.closed {
		return streamFailure(diagnosticStreamState, "received data after stream close", nil)
	}
	if d.failed {
		return streamFailure(diagnosticStreamState, "stream decoder is in a failed state", nil)
	}

	frames, err := d.frames.Feed(chunk)
	if err != nil {
		d.failed = true
		return streamFailure(diagnosticInvalidStreamJSON, fmt.Sprintf("decode Anthropic SSE: %v", err), nil)
	}
	return d.decodeFrames(frames)
}

// Close validates both SSE framing and the Anthropic message lifecycle.
func (d *StreamDecoder) Close() canonical.Result[[]canonical.Event] {
	if d.closed {
		return canonical.Success([]canonical.Event(nil), nil)
	}
	d.closed = true
	if d.failed {
		return streamFailure(diagnosticStreamState, "stream decoder closed after an earlier failure", nil)
	}

	frames, err := d.frames.Close()
	if err != nil {
		return streamFailure(diagnosticStreamTruncated, fmt.Sprintf("close Anthropic SSE: %v", err), nil)
	}
	result := d.decodeFrames(frames)
	if !result.OK {
		return result
	}
	if !d.stopped {
		return streamFailure(diagnosticStreamTruncated, "Anthropic stream closed before message_stop or error", nil)
	}
	return result
}

func (d *StreamDecoder) decodeFrames(frames []sse.Event) canonical.Result[[]canonical.Event] {
	events := make([]canonical.Event, 0)
	diagnostics := make([]canonical.Diagnostic, 0)
	for _, frame := range frames {
		decoded, frameDiagnostics := d.decodeFrame(frame)
		events = append(events, decoded...)
		diagnostics = append(diagnostics, frameDiagnostics...)
		if canonical.HasErrors(frameDiagnostics) {
			d.failed = true
			return canonical.Failure[[]canonical.Event](diagnostics)
		}
	}
	return canonical.Success(events, diagnostics)
}

func (d *StreamDecoder) decodeFrame(frame sse.Event) ([]canonical.Event, []canonical.Diagnostic) {
	raw := json.RawMessage(frame.Data)
	object, err := canonical.DecodeObject(raw)
	if err != nil {
		return nil, []canonical.Diagnostic{makeDiagnostic(canonical.SeverityError, diagnosticInvalidStreamJSON, err.Error(), "", raw)}
	}
	typeName := optionalEventType(takeRaw(object, "type"), frame.Name)
	if typeName == "" {
		return nil, []canonical.Diagnostic{makeDiagnostic(canonical.SeverityError, diagnosticInvalidStreamJSON, "stream event type is required", "type", raw)}
	}
	if d.stopped && typeName != "ping" {
		return nil, []canonical.Diagnostic{makeDiagnostic(canonical.SeverityError, diagnosticStreamState, "received an event after stream termination", "type", quotedRaw(typeName))}
	}

	switch typeName {
	case "message_start":
		return d.decodeMessageStart(object, raw)
	case "content_block_start":
		return d.decodeBlockStart(object, raw)
	case "content_block_delta":
		return d.decodeBlockDelta(object, raw)
	case "content_block_stop":
		return d.decodeBlockStop(object, raw)
	case "message_delta":
		return d.decodeMessageDelta(object, raw)
	case "message_stop":
		return d.decodeMessageStop(raw)
	case "ping":
		return nil, nil
	case "error":
		return d.decodeError(object, raw)
	default:
		return []canonical.Event{opaqueEvent(raw)}, nil
	}
}

func optionalEventType(raw json.RawMessage, fallback string) string {
	if len(raw) > 0 {
		var value string
		if err := json.Unmarshal(raw, &value); err == nil {
			return value
		}
		return ""
	}
	if fallback == "" || fallback == "message" {
		return ""
	}
	return fallback
}

func (d *StreamDecoder) decodeMessageStart(object canonical.Object, raw json.RawMessage) ([]canonical.Event, []canonical.Diagnostic) {
	if d.started {
		return nil, []canonical.Diagnostic{makeDiagnostic(canonical.SeverityError, diagnosticStreamState, "duplicate message_start", "type", raw)}
	}
	messageRaw := takeRaw(object, "message")
	message, err := canonical.DecodeObject(messageRaw)
	if err != nil {
		return nil, []canonical.Diagnostic{makeDiagnostic(canonical.SeverityError, diagnosticInvalidStreamJSON, "message_start.message must be an object", "message", messageRaw)}
	}

	diagnostics := make([]canonical.Diagnostic, 0)
	id := responseRequiredString(takeRaw(message, "id"), "message.id", &diagnostics)
	modelName := responseRequiredString(takeRaw(message, "model"), "message.model", &diagnostics)
	model := &modelName
	d.started = true
	events := []canonical.Event{{Type: canonical.EventResponseStart, ID: id, Model: model}}

	usageRaw, hasUsage := message["usage"]
	delete(message, "usage")
	if hasUsage {
		usage := decodeStreamUsage(usageRaw, "message.usage", &diagnostics)
		if usage != nil {
			d.mergeUsage(*usage)
			copy := d.usage
			events = append(events, canonical.Event{Type: canonical.EventUsage, Usage: &copy})
		}
	}
	if canonical.HasErrors(diagnostics) {
		return nil, diagnostics
	}
	return events, diagnostics
}

func (d *StreamDecoder) decodeBlockStart(object canonical.Object, raw json.RawMessage) ([]canonical.Event, []canonical.Diagnostic) {
	if !d.started {
		return nil, []canonical.Diagnostic{makeDiagnostic(canonical.SeverityError, diagnosticStreamState, "content block started before message_start", "type", raw)}
	}
	index, ok, diagnostic := streamIndex(takeRaw(object, "index"), "index")
	if !ok {
		return nil, []canonical.Diagnostic{diagnostic}
	}
	if _, exists := d.blocks[index]; exists {
		return nil, []canonical.Diagnostic{makeDiagnostic(canonical.SeverityError, diagnosticStreamState, fmt.Sprintf("duplicate content block index %d", index), "index", mustRaw(index))}
	}
	blockRaw := takeRaw(object, "content_block")
	block, err := canonical.DecodeObject(blockRaw)
	if err != nil {
		return nil, []canonical.Diagnostic{makeDiagnostic(canonical.SeverityError, diagnosticInvalidStreamJSON, "content_block must be an object", "content_block", blockRaw)}
	}
	diagnostics := make([]canonical.Diagnostic, 0)
	kind := responseRequiredString(takeRaw(block, "type"), "content_block.type", &diagnostics)
	state := &streamBlock{kind: kind}
	d.blocks[index] = state

	switch kind {
	case "text":
		text := optionalString(takeRaw(block, "text"), "content_block.text", &diagnostics)
		if canonical.HasErrors(diagnostics) || text == "" {
			return nil, diagnostics
		}
		return []canonical.Event{{Type: canonical.EventTextDelta, OutputIndex: intPointer(index), Delta: text}}, diagnostics
	case "tool_use":
		id := responseRequiredString(takeRaw(block, "id"), "content_block.id", &diagnostics)
		name := responseRequiredString(takeRaw(block, "name"), "content_block.name", &diagnostics)
		state.initialInput = string(bytes.TrimSpace(takeRaw(block, "input")))
		if state.initialInput == "" {
			state.initialInput = "{}"
		}
		if canonical.HasErrors(diagnostics) {
			return nil, diagnostics
		}
		return []canonical.Event{{Type: canonical.EventToolCallStart, OutputIndex: intPointer(index), CallID: id, Name: name}}, diagnostics
	default:
		return []canonical.Event{opaqueIndexedEvent(raw, index)}, diagnostics
	}
}

func (d *StreamDecoder) decodeBlockDelta(object canonical.Object, raw json.RawMessage) ([]canonical.Event, []canonical.Diagnostic) {
	if !d.started {
		return nil, []canonical.Diagnostic{makeDiagnostic(canonical.SeverityError, diagnosticStreamState, "content block delta arrived before message_start", "type", raw)}
	}
	index, ok, diagnostic := streamIndex(takeRaw(object, "index"), "index")
	if !ok {
		return nil, []canonical.Diagnostic{diagnostic}
	}
	block, exists := d.blocks[index]
	if !exists {
		return nil, []canonical.Diagnostic{makeDiagnostic(canonical.SeverityError, diagnosticStreamState, fmt.Sprintf("delta for unopened content block %d", index), "index", mustRaw(index))}
	}
	deltaRaw := takeRaw(object, "delta")
	delta, err := canonical.DecodeObject(deltaRaw)
	if err != nil {
		return nil, []canonical.Diagnostic{makeDiagnostic(canonical.SeverityError, diagnosticInvalidStreamJSON, "delta must be an object", "delta", deltaRaw)}
	}
	diagnostics := make([]canonical.Diagnostic, 0)
	typeName := responseRequiredString(takeRaw(delta, "type"), "delta.type", &diagnostics)
	if canonical.HasErrors(diagnostics) {
		return nil, diagnostics
	}

	switch typeName {
	case "text_delta":
		if block.kind != "text" {
			return nil, []canonical.Diagnostic{makeDiagnostic(canonical.SeverityError, diagnosticStreamState, "text delta does not match its content block", "delta.type", quotedRaw(typeName))}
		}
		text := optionalString(takeRaw(delta, "text"), "delta.text", &diagnostics)
		if canonical.HasErrors(diagnostics) {
			return nil, diagnostics
		}
		return []canonical.Event{{Type: canonical.EventTextDelta, OutputIndex: intPointer(index), Delta: text}}, diagnostics
	case "input_json_delta":
		if block.kind != "tool_use" {
			return nil, []canonical.Diagnostic{makeDiagnostic(canonical.SeverityError, diagnosticStreamState, "tool input delta does not match its content block", "delta.type", quotedRaw(typeName))}
		}
		partial := optionalString(takeRaw(delta, "partial_json"), "delta.partial_json", &diagnostics)
		if canonical.HasErrors(diagnostics) {
			return nil, diagnostics
		}
		block.arguments.WriteString(partial)
		return []canonical.Event{{Type: canonical.EventToolArgumentsDelta, OutputIndex: intPointer(index), Delta: partial}}, diagnostics
	default:
		return []canonical.Event{opaqueIndexedEvent(raw, index)}, diagnostics
	}
}

func (d *StreamDecoder) decodeBlockStop(object canonical.Object, raw json.RawMessage) ([]canonical.Event, []canonical.Diagnostic) {
	if !d.started {
		return nil, []canonical.Diagnostic{makeDiagnostic(canonical.SeverityError, diagnosticStreamState, "content block stopped before message_start", "type", raw)}
	}
	index, ok, diagnostic := streamIndex(takeRaw(object, "index"), "index")
	if !ok {
		return nil, []canonical.Diagnostic{diagnostic}
	}
	block, exists := d.blocks[index]
	if !exists {
		return nil, []canonical.Diagnostic{makeDiagnostic(canonical.SeverityError, diagnosticStreamState, fmt.Sprintf("stop for unopened content block %d", index), "index", mustRaw(index))}
	}
	delete(d.blocks, index)
	if block.kind == "text" {
		return nil, nil
	}
	if block.kind != "tool_use" {
		return []canonical.Event{opaqueIndexedEvent(raw, index)}, nil
	}

	arguments := block.arguments.String()
	useInitialInput := arguments == ""
	if arguments == "" {
		arguments = block.initialInput
	}
	if _, err := canonical.DecodeObject([]byte(arguments)); err != nil {
		return nil, []canonical.Diagnostic{makeDiagnostic(
			canonical.SeverityError,
			canonical.DiagnosticInvalidToolArgumentsJSON,
			fmt.Sprintf("completed tool input must be a JSON object: %v", err),
			fmt.Sprintf("content_blocks.%d.input", index),
			quotedRaw(arguments),
		)}
	}
	events := make([]canonical.Event, 0, 2)
	if useInitialInput {
		events = append(events, canonical.Event{Type: canonical.EventToolArgumentsDelta, OutputIndex: intPointer(index), Delta: arguments})
	}
	events = append(events, canonical.Event{Type: canonical.EventToolCallEnd, OutputIndex: intPointer(index)})
	return events, nil
}

func (d *StreamDecoder) decodeMessageDelta(object canonical.Object, raw json.RawMessage) ([]canonical.Event, []canonical.Diagnostic) {
	if !d.started {
		return nil, []canonical.Diagnostic{makeDiagnostic(canonical.SeverityError, diagnosticStreamState, "message_delta arrived before message_start", "type", raw)}
	}
	if len(d.blocks) > 0 {
		return nil, []canonical.Diagnostic{makeDiagnostic(canonical.SeverityError, diagnosticStreamState, "message_delta arrived before all content blocks stopped", "type", raw)}
	}
	diagnostics := make([]canonical.Diagnostic, 0)
	deltaRaw := takeRaw(object, "delta")
	delta, err := canonical.DecodeObject(deltaRaw)
	if err != nil {
		return nil, []canonical.Diagnostic{makeDiagnostic(canonical.SeverityError, diagnosticInvalidStreamJSON, "message_delta.delta must be an object", "delta", deltaRaw)}
	}
	d.decodePendingStop(delta, &diagnostics)

	events := make([]canonical.Event, 0, 1)
	usageRaw, hasUsage := object["usage"]
	delete(object, "usage")
	if hasUsage {
		usage := decodeStreamUsage(usageRaw, "usage", &diagnostics)
		if usage != nil {
			d.mergeUsage(*usage)
			copy := d.usage
			events = append(events, canonical.Event{Type: canonical.EventUsage, Usage: &copy})
		}
	}
	if canonical.HasErrors(diagnostics) {
		return nil, diagnostics
	}
	return events, diagnostics
}

func (d *StreamDecoder) decodePendingStop(delta canonical.Object, diagnostics *[]canonical.Diagnostic) {
	reasonRaw, hasReason := delta["stop_reason"]
	delete(delta, "stop_reason")
	if hasReason && !bytes.Equal(bytes.TrimSpace(reasonRaw), []byte("null")) {
		var reason string
		if err := json.Unmarshal(reasonRaw, &reason); err != nil || reason == "" {
			*diagnostics = append(*diagnostics, makeDiagnostic(canonical.SeverityError, diagnosticInvalidStreamJSON, "stop_reason must be a string or null", "delta.stop_reason", reasonRaw))
		} else {
			mapped, known := mapStopReason(reason)
			d.pendingReason = &mapped
			d.providerReason = &reason
			if !known {
				*diagnostics = append(*diagnostics, makeDiagnostic(canonical.SeverityWarning, diagnosticUnknownStop, fmt.Sprintf("unknown Anthropic stop reason %q was preserved", reason), "delta.stop_reason", reasonRaw))
			}
		}
	}

	for _, name := range []string{"stop_sequence", "stop_details"} {
		raw, exists := delta[name]
		delete(delta, name)
		if !exists {
			continue
		}
		d.setFinishExtension(name, raw)
	}
	if len(delta) == 0 {
		return
	}
	d.setFinishExtension("message_delta_extensions", mustRaw(delta))
}

func (d *StreamDecoder) decodeMessageStop(raw json.RawMessage) ([]canonical.Event, []canonical.Diagnostic) {
	if !d.started {
		return nil, []canonical.Diagnostic{makeDiagnostic(canonical.SeverityError, diagnosticStreamState, "message_stop arrived before message_start", "type", raw)}
	}
	if len(d.blocks) > 0 {
		return nil, []canonical.Diagnostic{makeDiagnostic(canonical.SeverityError, diagnosticStreamState, "message_stop arrived before all content blocks stopped", "type", raw)}
	}
	if d.pendingReason == nil {
		return nil, []canonical.Diagnostic{makeDiagnostic(canonical.SeverityError, diagnosticStreamState, "message_stop arrived before a stop reason", "type", raw)}
	}
	event := canonical.Event{
		Type:           canonical.EventFinish,
		Reason:         d.pendingReason,
		ProviderReason: d.providerReason,
		Value:          mustRaw(d.finishExtensions),
	}
	if len(d.finishExtensions) == 0 {
		event.Value = nil
	}
	d.stopped = true
	return []canonical.Event{event}, nil
}

func (d *StreamDecoder) decodeError(object canonical.Object, raw json.RawMessage) ([]canonical.Event, []canonical.Diagnostic) {
	errorRaw := takeRaw(object, "error")
	if len(errorRaw) == 0 {
		errorRaw = raw
	}
	reason := canonical.FinishReasonError
	providerReason := streamErrorType(errorRaw)
	d.stopped = true
	return []canonical.Event{
		{Type: canonical.EventError, Error: cloneRaw(errorRaw)},
		{Type: canonical.EventFinish, Reason: &reason, ProviderReason: providerReason, Value: cloneRaw(errorRaw)},
	}, nil
}

func streamErrorType(raw json.RawMessage) *string {
	object, err := canonical.DecodeObject(raw)
	if err != nil {
		return nil
	}
	var typeName string
	if err := json.Unmarshal(object["type"], &typeName); err != nil || typeName == "" {
		return nil
	}
	return &typeName
}

func decodeStreamUsage(raw json.RawMessage, path string, diagnostics *[]canonical.Diagnostic) *canonical.Usage {
	object, err := canonical.DecodeObject(raw)
	if err != nil {
		*diagnostics = append(*diagnostics, makeDiagnostic(canonical.SeverityError, diagnosticInvalidStreamJSON, "usage must be an object", path, raw))
		return nil
	}
	usage := canonical.Usage{}
	if inputRaw, exists := object["input_tokens"]; exists {
		delete(object, "input_tokens")
		usage.InputTokens = streamTokenCount(inputRaw, path+".input_tokens", diagnostics)
	}
	if outputRaw, exists := object["output_tokens"]; exists {
		delete(object, "output_tokens")
		usage.OutputTokens = streamTokenCount(outputRaw, path+".output_tokens", diagnostics)
	}
	if usage.InputTokens != nil && usage.OutputTokens != nil {
		total := *usage.InputTokens + *usage.OutputTokens
		usage.TotalTokens = &total
	}
	usage.Extensions = cloneObject(object)
	return &usage
}

func streamTokenCount(raw json.RawMessage, path string, diagnostics *[]canonical.Diagnostic) *int64 {
	var value int64
	if err := json.Unmarshal(raw, &value); err != nil || value < 0 {
		*diagnostics = append(*diagnostics, makeDiagnostic(canonical.SeverityError, diagnosticInvalidStreamJSON, "token count must be a non-negative integer", path, raw))
		return nil
	}
	return &value
}

func (d *StreamDecoder) mergeUsage(next canonical.Usage) {
	if next.InputTokens != nil {
		d.usage.InputTokens = next.InputTokens
	}
	if next.OutputTokens != nil {
		d.usage.OutputTokens = next.OutputTokens
	}
	if len(next.Extensions) > 0 {
		if d.usage.Extensions == nil {
			d.usage.Extensions = make(canonical.Object)
		}
		for key, raw := range next.Extensions {
			d.usage.Extensions[key] = cloneRaw(raw)
		}
	}
	if d.usage.InputTokens != nil && d.usage.OutputTokens != nil {
		total := *d.usage.InputTokens + *d.usage.OutputTokens
		d.usage.TotalTokens = &total
	}
}

func (d *StreamDecoder) setFinishExtension(name string, raw json.RawMessage) {
	if d.finishExtensions == nil {
		d.finishExtensions = make(canonical.Object)
	}
	d.finishExtensions[name] = cloneRaw(raw)
}

func optionalString(raw json.RawMessage, path string, diagnostics *[]canonical.Diagnostic) string {
	if len(raw) == 0 {
		return ""
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		*diagnostics = append(*diagnostics, makeDiagnostic(canonical.SeverityError, diagnosticInvalidStreamJSON, "value must be a string", path, raw))
		return ""
	}
	return value
}

func streamIndex(raw json.RawMessage, path string) (int, bool, canonical.Diagnostic) {
	var value int
	if err := json.Unmarshal(raw, &value); err == nil && value >= 0 {
		return value, true, canonical.Diagnostic{}
	}
	return 0, false, makeDiagnostic(canonical.SeverityError, diagnosticInvalidStreamJSON, "content block index must be a non-negative integer", path, raw)
}

func intPointer(value int) *int {
	return &value
}

func opaqueEvent(raw json.RawMessage) canonical.Event {
	provider := "anthropic.messages"
	return canonical.Event{Type: canonical.EventOpaque, Provider: &provider, Value: cloneRaw(raw)}
}

func opaqueIndexedEvent(raw json.RawMessage, index int) canonical.Event {
	event := opaqueEvent(raw)
	event.OutputIndex = intPointer(index)
	return event
}

func streamFailure(code canonical.DiagnosticCode, message string, source json.RawMessage) canonical.Result[[]canonical.Event] {
	return canonical.Failure[[]canonical.Event]([]canonical.Diagnostic{
		makeDiagnostic(canonical.SeverityError, code, message, "", source),
	})
}
