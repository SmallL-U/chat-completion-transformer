// Package sse implements a small incremental Server-Sent Events decoder.
package sse

import (
	"bytes"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"
)

const DefaultMaxEventBytes = 1 << 20

var (
	ErrEventTooLarge = errors.New("SSE event exceeds the configured size limit")
	ErrUnexpectedEOF = errors.New("SSE stream ended before an event delimiter")
)

// Event is one decoded SSE event. Name defaults to "message".
type Event struct {
	Name  string
	Data  string
	ID    string
	Retry *time.Duration
}

// Decoder accepts arbitrary byte chunks and keeps framing state per stream.
// A Decoder must not be shared by concurrent streams.
type Decoder struct {
	buffer        []byte
	maxEventBytes int
	eventBytes    int
	data          strings.Builder
	hasData       bool
	eventName     string
	lastEventID   string
	retry         *time.Duration
	consumedLine  bool
}

func NewDecoder(maxEventBytes int) *Decoder {
	if maxEventBytes <= 0 {
		maxEventBytes = DefaultMaxEventBytes
	}

	return &Decoder{maxEventBytes: maxEventBytes}
}

// Feed consumes a network chunk and returns every complete event in it.
func (d *Decoder) Feed(chunk []byte) ([]Event, error) {
	if len(chunk) == 0 {
		return nil, nil
	}

	d.buffer = append(d.buffer, chunk...)
	var events []Event
	for {
		line, ok := d.nextLine(false)
		if !ok {
			if len(d.buffer)+d.eventBytes > d.maxEventBytes {
				return nil, ErrEventTooLarge
			}

			return events, nil
		}

		event, err := d.consumeLine(line)
		if err != nil {
			return nil, err
		}
		if event == nil {
			continue
		}

		events = append(events, *event)
	}
}

// Close validates the final frame. Providers are expected to terminate each
// event with a blank line; accepting an unterminated frame would hide a
// truncated upstream response.
func (d *Decoder) Close() ([]Event, error) {
	if len(d.buffer) > 0 {
		line, _ := d.nextLine(true)
		if _, err := d.consumeLine(line); err != nil {
			return nil, err
		}
	}

	if d.hasPendingEvent() {
		return nil, ErrUnexpectedEOF
	}

	return nil, nil
}

func (d *Decoder) nextLine(atEOF bool) ([]byte, bool) {
	for index, value := range d.buffer {
		if value != '\n' && value != '\r' {
			continue
		}

		if value == '\r' && index == len(d.buffer)-1 && !atEOF {
			return nil, false
		}

		line := append([]byte(nil), d.buffer[:index]...)
		consumed := index + 1
		if value == '\r' && consumed < len(d.buffer) && d.buffer[consumed] == '\n' {
			consumed++
		}
		d.buffer = d.buffer[consumed:]
		return line, true
	}

	if !atEOF || len(d.buffer) == 0 {
		return nil, false
	}

	line := append([]byte(nil), d.buffer...)
	d.buffer = nil
	return line, true
}

func (d *Decoder) consumeLine(line []byte) (*Event, error) {
	if !d.consumedLine {
		d.consumedLine = true
		line = bytes.TrimPrefix(line, []byte{0xef, 0xbb, 0xbf})
	}

	if len(line) == 0 {
		return d.dispatch(), nil
	}

	d.eventBytes += len(line)
	if d.eventBytes > d.maxEventBytes {
		return nil, ErrEventTooLarge
	}
	if line[0] == ':' {
		return nil, nil
	}

	field, value := splitField(line)
	switch field {
	case "event":
		d.eventName = value
	case "data":
		if d.hasData {
			d.data.WriteByte('\n')
		}
		d.data.WriteString(value)
		d.hasData = true
	case "id":
		if !strings.ContainsRune(value, '\x00') {
			d.lastEventID = value
		}
	case "retry":
		milliseconds, err := strconv.ParseUint(value, 10, 64)
		if err == nil {
			duration := time.Duration(milliseconds) * time.Millisecond
			d.retry = &duration
		}
	}

	return nil, nil
}

func (d *Decoder) dispatch() *Event {
	if !d.hasData {
		d.resetEvent()
		return nil
	}

	name := d.eventName
	if name == "" {
		name = "message"
	}

	event := &Event{
		Name:  name,
		Data:  d.data.String(),
		ID:    d.lastEventID,
		Retry: d.retry,
	}
	d.resetEvent()
	return event
}

func (d *Decoder) resetEvent() {
	d.eventBytes = 0
	d.data.Reset()
	d.hasData = false
	d.eventName = ""
	d.retry = nil
}

func (d *Decoder) hasPendingEvent() bool {
	return d.hasData || d.eventName != "" || d.retry != nil || d.eventBytes > 0
}

func splitField(line []byte) (string, string) {
	index := bytes.IndexByte(line, ':')
	if index < 0 {
		return string(line), ""
	}

	value := line[index+1:]
	if len(value) > 0 && value[0] == ' ' {
		value = value[1:]
	}

	return string(line[:index]), string(value)
}

func (e Event) String() string {
	return fmt.Sprintf("%s: %s", e.Name, e.Data)
}
