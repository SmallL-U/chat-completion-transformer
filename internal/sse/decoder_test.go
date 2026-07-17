package sse

import (
	"errors"
	"strings"
	"testing"
	"time"
)

func TestDecoderArbitrarySplits(t *testing.T) {
	input := "\ufeffevent: update\r\nid: 7\r\ndata: first\r\ndata: second\r\nretry: 25\r\n\r\n: ping\n\ndata: done\n\n"

	for split := 0; split <= len(input); split++ {
		decoder := NewDecoder(0)
		first, err := decoder.Feed([]byte(input[:split]))
		if err != nil {
			t.Fatalf("split %d first feed: %v", split, err)
		}
		second, err := decoder.Feed([]byte(input[split:]))
		if err != nil {
			t.Fatalf("split %d second feed: %v", split, err)
		}
		events := append(first, second...)
		if _, err := decoder.Close(); err != nil {
			t.Fatalf("split %d close: %v", split, err)
		}
		if len(events) != 2 {
			t.Fatalf("split %d produced %d events", split, len(events))
		}
		if events[0].Name != "update" || events[0].ID != "7" || events[0].Data != "first\nsecond" {
			t.Fatalf("split %d first event = %#v", split, events[0])
		}
		if events[0].Retry == nil || *events[0].Retry != 25*time.Millisecond {
			t.Fatalf("split %d retry = %v", split, events[0].Retry)
		}
		if events[1].Name != "message" || events[1].ID != "7" || events[1].Data != "done" {
			t.Fatalf("split %d second event = %#v", split, events[1])
		}
	}
}

func TestDecoderEmptyData(t *testing.T) {
	decoder := NewDecoder(0)
	events, err := decoder.Feed([]byte("data:\n\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 || events[0].Data != "" {
		t.Fatalf("events = %#v", events)
	}
}

func TestDecoderAcceptsMultipleEventsWhoseCombinedChunkExceedsLimit(t *testing.T) {
	decoder := NewDecoder(12)
	events, err := decoder.Feed([]byte("data: one\n\ndata: two\n\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 2 || events[0].Data != "one" || events[1].Data != "two" {
		t.Fatalf("events = %#v", events)
	}
}

func TestDecoderRejectsOversizeAndTruncatedEvents(t *testing.T) {
	t.Run("oversize", func(t *testing.T) {
		decoder := NewDecoder(8)
		_, err := decoder.Feed([]byte("data: too much"))
		if !errors.Is(err, ErrEventTooLarge) {
			t.Fatalf("error = %v", err)
		}
	})

	t.Run("truncated", func(t *testing.T) {
		decoder := NewDecoder(0)
		if _, err := decoder.Feed([]byte("data: partial\n")); err != nil {
			t.Fatal(err)
		}
		if _, err := decoder.Close(); !errors.Is(err, ErrUnexpectedEOF) {
			t.Fatalf("error = %v", err)
		}
	})
}

func FuzzDecoderChunks(f *testing.F) {
	f.Add("data: hello\n\n", uint8(3))
	f.Fuzz(func(t *testing.T, input string, width uint8) {
		if len(input) > 4<<10 {
			t.Skip()
		}

		decoder := NewDecoder(8 << 10)
		step := int(width) + 1
		for len(input) > 0 {
			end := min(step, len(input))
			_, err := decoder.Feed([]byte(input[:end]))
			if err != nil && !errors.Is(err, ErrEventTooLarge) {
				t.Fatalf("feed: %v", err)
			}
			if err != nil {
				return
			}
			input = input[end:]
		}
		_, err := decoder.Close()
		if err != nil && !errors.Is(err, ErrUnexpectedEOF) && !strings.Contains(err.Error(), "size limit") {
			t.Fatalf("close: %v", err)
		}
	})
}
