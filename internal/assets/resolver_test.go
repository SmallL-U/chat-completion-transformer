package assets

import (
	"context"
	"encoding/base64"
	"errors"
	"testing"

	"chat-completion-transformer/internal/canonical"
)

func TestNativeResolver(t *testing.T) {
	data := base64.StdEncoding.EncodeToString([]byte("image"))
	tests := []struct {
		name   string
		source canonical.AssetSource
		kind   canonical.AssetSourceKind
	}{
		{
			name:   "URL",
			source: canonical.AssetSource{Kind: canonical.AssetSourceURL, URL: "https://example.com/image.png"},
			kind:   canonical.AssetSourceURL,
		},
		{
			name: "base64",
			source: canonical.AssetSource{
				Kind:      canonical.AssetSourceBase64,
				MediaType: "image/png",
				Data:      data,
			},
			kind: canonical.AssetSourceBase64,
		},
		{
			name:   "file ID",
			source: canonical.AssetSource{Kind: canonical.AssetSourceFileID, FileID: "file_123"},
			kind:   canonical.AssetSourceFileID,
		},
	}

	resolver := NativeResolver{}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			resolved, err := resolver.ResolveForAnthropic(context.Background(), test.source)
			if err != nil {
				t.Fatal(err)
			}
			if resolved.Kind != test.kind {
				t.Fatalf("kind = %q", resolved.Kind)
			}
		})
	}
}

func TestNativeResolverRejectsUnsafeAndOversizedSources(t *testing.T) {
	resolver := NativeResolver{MaxBase64Bytes: 2}

	_, err := resolver.ResolveForResponses(context.Background(), canonical.AssetSource{
		Kind: canonical.AssetSourceURL,
		URL:  "file:///etc/passwd",
	})
	if !errors.Is(err, ErrUnsupportedSource) {
		t.Fatalf("URL error = %v", err)
	}

	_, err = resolver.ResolveForResponses(context.Background(), canonical.AssetSource{
		Kind:      canonical.AssetSourceBase64,
		MediaType: "image/png",
		Data:      base64.StdEncoding.EncodeToString([]byte("large")),
	})
	if !errors.Is(err, ErrAssetTooLarge) {
		t.Fatalf("base64 error = %v", err)
	}
}

func TestNativeResolverHonorsCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := (NativeResolver{}).ResolveForResponses(ctx, canonical.AssetSource{
		Kind:   canonical.AssetSourceFileID,
		FileID: "file_123",
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v", err)
	}
}
