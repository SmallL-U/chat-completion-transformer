package openairesponses

import (
	"encoding/json"
	"testing"
)

var benchmarkDecodedString string

func BenchmarkDecodeStreamJSONString(b *testing.B) {
	raw := []byte(`"response.output_text.delta"`)

	b.Run("encoding_json", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			var value string
			if err := json.Unmarshal(raw, &value); err != nil {
				b.Fatal(err)
			}
			benchmarkDecodedString = value
		}
	})

	b.Run("gjson", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			value, ok := streamJSONString(raw)
			if !ok {
				b.Fatal("string was rejected")
			}
			benchmarkDecodedString = value
		}
	})
}
