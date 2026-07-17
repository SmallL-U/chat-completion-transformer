package assets

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"mime"
	"net/url"

	"chat-completion-transformer/internal/canonical"
)

const DefaultMaxBase64Bytes = 25 << 20

var (
	ErrInvalidSource     = errors.New("invalid asset source")
	ErrAssetTooLarge     = errors.New("asset exceeds the configured size limit")
	ErrUnsupportedSource = errors.New("asset source is not supported by the target")
)

// ResolvedAsset is validated provider-ready asset data. Resolvers may return a
// different source kind when a deployment explicitly supports safe fetching or
// file upload; the built-in resolver deliberately performs no network access.
type ResolvedAsset struct {
	Kind      canonical.AssetSourceKind
	URL       string
	MediaType string
	Data      string
	FileID    string
}

type Resolver interface {
	ResolveForResponses(context.Context, canonical.AssetSource) (ResolvedAsset, error)
	ResolveForAnthropic(context.Context, canonical.AssetSource) (ResolvedAsset, error)
}

// NativeResolver validates URL, base64, and file-ID sources that can be sent
// directly to a provider. It never downloads user-provided URLs, avoiding an
// implicit SSRF-prone fetch path in the protocol transformer.
type NativeResolver struct {
	MaxBase64Bytes int
}

func (r NativeResolver) ResolveForResponses(ctx context.Context, source canonical.AssetSource) (ResolvedAsset, error) {
	return r.resolve(ctx, source)
}

func (r NativeResolver) ResolveForAnthropic(ctx context.Context, source canonical.AssetSource) (ResolvedAsset, error) {
	return r.resolve(ctx, source)
}

func (r NativeResolver) resolve(ctx context.Context, source canonical.AssetSource) (ResolvedAsset, error) {
	if err := ctx.Err(); err != nil {
		return ResolvedAsset{}, err
	}

	switch source.Kind {
	case canonical.AssetSourceURL:
		return resolveURL(source)
	case canonical.AssetSourceBase64:
		return r.resolveBase64(source)
	case canonical.AssetSourceFileID:
		return resolveFileID(source)
	default:
		return ResolvedAsset{}, fmt.Errorf("%w: unknown kind %q", ErrInvalidSource, source.Kind)
	}
}

func resolveURL(source canonical.AssetSource) (ResolvedAsset, error) {
	parsed, err := url.Parse(source.URL)
	if err != nil {
		return ResolvedAsset{}, fmt.Errorf("%w: malformed URL", ErrInvalidSource)
	}
	if parsed.Scheme != "https" && parsed.Scheme != "http" {
		return ResolvedAsset{}, fmt.Errorf("%w: URL scheme %q", ErrUnsupportedSource, parsed.Scheme)
	}
	if parsed.Host == "" {
		return ResolvedAsset{}, fmt.Errorf("%w: malformed URL", ErrInvalidSource)
	}

	return ResolvedAsset{Kind: canonical.AssetSourceURL, URL: source.URL}, nil
}

func (r NativeResolver) resolveBase64(source canonical.AssetSource) (ResolvedAsset, error) {
	if source.MediaType == "" || source.Data == "" {
		return ResolvedAsset{}, fmt.Errorf("%w: base64 media type and data are required", ErrInvalidSource)
	}
	mediaType, _, err := mime.ParseMediaType(source.MediaType)
	if err != nil {
		return ResolvedAsset{}, fmt.Errorf("%w: malformed media type: %v", ErrInvalidSource, err)
	}

	maxBytes := r.MaxBase64Bytes
	if maxBytes <= 0 {
		maxBytes = DefaultMaxBase64Bytes
	}
	decodedLength := base64.StdEncoding.DecodedLen(len(source.Data))
	if decodedLength > maxBytes+2 {
		return ResolvedAsset{}, ErrAssetTooLarge
	}
	decoded, err := base64.StdEncoding.DecodeString(source.Data)
	if err != nil {
		return ResolvedAsset{}, fmt.Errorf("%w: malformed base64 data", ErrInvalidSource)
	}
	if len(decoded) > maxBytes {
		return ResolvedAsset{}, ErrAssetTooLarge
	}

	return ResolvedAsset{
		Kind:      canonical.AssetSourceBase64,
		MediaType: mediaType,
		Data:      source.Data,
	}, nil
}

func resolveFileID(source canonical.AssetSource) (ResolvedAsset, error) {
	if source.FileID == "" {
		return ResolvedAsset{}, fmt.Errorf("%w: file ID is required", ErrInvalidSource)
	}

	return ResolvedAsset{Kind: canonical.AssetSourceFileID, FileID: source.FileID}, nil
}
