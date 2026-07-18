package chatcompletions

import "testing"

var benchmarkSSEFrame []byte

func BenchmarkEncodeTextDeltaFrame(b *testing.B) {
	encoder := &StreamEncoder{
		options: StreamEncodeOptions{Created: 1_700_000_000, IncludeUsage: true},
		id:      "resp_benchmark",
		model:   "gpt-benchmark",
	}
	delta := "hello \"stream\" \\ world\n"

	b.Run("encoding_json", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			benchmarkSSEFrame = encoder.chunkFrame(map[string]any{"content": delta}, nil, nil)
		}
	})

	b.Run("sjson_cached_base", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			benchmarkSSEFrame = encoder.deltaFrame("content", delta)
		}
	})
}
