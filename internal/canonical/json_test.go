package canonical

import (
	"encoding/json"
	"testing"
)

func TestDecodeObject(t *testing.T) {
	raw := []byte(`{
		"big": 9007199254740993123456789,
		"decimal": 1.2300,
		"nested": { "enabled": true }
	}`)

	object, err := DecodeObject(raw)
	if err != nil {
		t.Fatalf("DecodeObject() error = %v", err)
	}

	assertRawJSON(t, object["big"], `9007199254740993123456789`)
	assertRawJSON(t, object["decimal"], `1.2300`)
	assertRawJSON(t, object["nested"], `{ "enabled": true }`)
}

func TestDecodeObjectRejectsNonObjectAndTrailingValues(t *testing.T) {
	tests := []struct {
		name string
		raw  string
	}{
		{name: "null", raw: `null`},
		{name: "array", raw: `[]`},
		{name: "string", raw: `"value"`},
		{name: "trailing value", raw: `{} {}`},
		{name: "invalid JSON", raw: `{`},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := DecodeObject([]byte(test.raw)); err == nil {
				t.Fatalf("DecodeObject(%q) error = nil", test.raw)
			}
		})
	}
}

func TestOpaqueJSONIsRetained(t *testing.T) {
	raw := json.RawMessage(`{ "precise": 1.2300, "large": 9007199254740993 }`)
	part := Part{Kind: PartOpaque, Value: raw}
	event := Event{Type: EventOpaque, Value: raw}

	assertRawJSON(t, part.Value, string(raw))
	assertRawJSON(t, event.Value, string(raw))
}

func assertRawJSON(t *testing.T, got json.RawMessage, want string) {
	t.Helper()
	if string(got) != want {
		t.Fatalf("raw JSON = %q, want %q", got, want)
	}
}
