package canonical

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// DecodeObject validates a JSON object while preserving every property value
// as json.RawMessage. It rejects null and trailing JSON values.
func DecodeObject(raw []byte) (Object, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()

	var object Object
	if err := decoder.Decode(&object); err != nil {
		return nil, fmt.Errorf("decode JSON object: %w", err)
	}

	if object == nil {
		return nil, errors.New("decode JSON object: expected object")
	}

	var trailing json.RawMessage
	err := decoder.Decode(&trailing)
	if errors.Is(err, io.EOF) {
		return object, nil
	}

	if err != nil {
		return nil, fmt.Errorf("decode trailing JSON: %w", err)
	}

	return nil, errors.New("decode JSON object: unexpected trailing value")
}
