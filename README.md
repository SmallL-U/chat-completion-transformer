# Chat Completion Transformer

English | [简体中文](README.zh-CN.md)

An OpenAI-compatible Chat Completions gateway and Go protocol-conversion
library. The service accepts Chat Completions requests, calls either OpenAI
Responses or Anthropic Messages, and converts buffered responses and SSE
streams back to the Chat Completions shape.

The gateway only converts protocols. It does not store, manage, replace, or
validate provider API keys.

## HTTP API

The service exposes exactly these two routes:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/healthz` | Liveness check |
| `POST` | `/v1/chat/completions` | OpenAI-compatible Chat Completions gateway |

The old `/v1/transform/*` debugging routes are not part of the public API.
Unknown routes and unsupported methods return an OpenAI-shaped JSON error.

Every Chat Completions request must contain one `Authorization: Bearer <key>`
header. For an OpenAI Responses target, the header is forwarded. For an
Anthropic Messages target, the bearer token is sent as `x-api-key`. The key is
used only for that request and is never read from configuration.

Example:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Authorization: Bearer provider-api-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openai-example",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Set `"stream": true` for Chat Completions SSE. When
`stream_options.include_usage` is true, ordinary chunks contain `usage: null`
and the final usage chunk contains the provider token counts. The gateway
currently supports one completion per request and rejects `n > 1`.

## Configure and run

The service reads `config.yml` from the project root. The repository default is
intentionally route-free, so `/healthz` works immediately while chat requests
need configured upstream and transformer routes.

Each public model alias must appear in both `gateway.routes` and
`transformer.routes`:

```yaml
gateway:
  response_header_timeout: 5m
  upstreams:
    openai:
      endpoint: responses
      url: https://api.openai.com/v1/responses
    anthropic:
      endpoint: messages
      url: https://api.anthropic.com/v1/messages
      anthropic_version: "2023-06-01"
  routes:
    openai-example: openai
    anthropic-example: anthropic

transformer:
  mode: compatible
  instruction_policy: preserve_messages
  anthropic_endpoint: messages
  default_max_output_tokens: 1024
  profiles:
    - provider: openai
      endpoint: responses
      model: gpt-example
      prompt_cache:
        mode: openai_5_6
      temperature: true
      top_p: true
      structured_output: true
      strict_tools: true
      parallel_tool_calls: true
      content: { text: true }
    - provider: anthropic
      endpoint: messages
      model: claude-example
      prompt_cache:
        mode: anthropic
      temperature: true
      top_p: true
      top_k: true
      stop_sequences: true
      structured_output: true
      strict_tools: true
      parallel_tool_calls: true
      content: { text: true }
  routes:
    - alias: openai-example
      targets:
        responses: gpt-example
    - alias: anthropic-example
      targets:
        messages: claude-example
```

Model names are never guessed. The gateway route chooses the provider endpoint;
the transformer route maps the same alias to a provider model with an exact
`(provider, endpoint, model)` capability profile.

Only direct OpenAI Responses and direct Anthropic Messages HTTP upstreams are
supported by the gateway. Remote upstream URLs must use HTTPS. Loopback HTTP is
allowed for local development. URLs containing user information, query strings,
or fragments are rejected, and redirects are not followed.

## Prompt cache control

The gateway maps provider prompt-cache directives; it does not implement a
local cache. It never stores prompts, cache keys, KV state, or provider cache
entries, does not inject cache-write directives, and does not generate a
`prompt_cache_key`. Provider-side automatic caching can still apply when it is
the upstream default.

### Capability profiles

Prompt-cache behavior is selected explicitly on the exact model profile. An
omitted `prompt_cache.mode` is normalized to `none`; model names are never used
to infer an API generation.

| `prompt_cache.mode` | Valid target | Accepted controls |
| --- | --- | --- |
| `none` | Any valid profile | No prompt-cache directive |
| `anthropic` | Direct Anthropic Messages | Top-level and content/tool `cache_control` |
| `openai_legacy` | OpenAI Responses | `prompt_cache_key` and enabled legacy retention values |
| `openai_5_6` | OpenAI Responses | `prompt_cache_key`, `prompt_cache_options`, and input block breakpoints |

Legacy retention values must also be enabled by the exact profile:

```yaml
prompt_cache:
  mode: openai_legacy
  in_memory_retention: true
  extended_retention_24h: true
```

These booleans permit callers to send `prompt_cache_retention: "in_memory"`
and `prompt_cache_retention: "24h"`, respectively; they do not choose or inject
a retention value. `in-memory` and other spellings are not rewritten as
aliases.

Known cache directives are type-checked and gated by the selected profile.
Malformed values and invalid placements are rejected. A valid directive that
the target profile does not support fails in `strict` mode; in `compatible` and
`emulate` modes it is dropped with a transformer warning. Arbitrary extension
fields are never passed through.

### Anthropic Messages

Top-level `cache_control` enables Anthropic automatic prompt caching. Omitting
`ttl` selects the provider's default 5-minute TTL; the explicit values supported
by the gateway are `5m` and `1h`.

```json
{
  "model": "anthropic-example",
  "cache_control": {
    "type": "ephemeral",
    "ttl": "1h"
  },
  "messages": [
    {"role": "user", "content": "Hello"}
  ]
}
```

Explicit breakpoints stay on their original Chat Completions content block or
on the outer `tools[*]` wrapper:

```json
{
  "model": "anthropic-example",
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "Large stable instructions",
          "cache_control": {"type": "ephemeral"}
        }
      ]
    },
    {"role": "user", "content": "Hello"}
  ],
  "tools": [
    {
      "type": "function",
      "cache_control": {"type": "ephemeral", "ttl": "1h"},
      "function": {
        "name": "lookup",
        "description": "Look up a value",
        "parameters": {"type": "object"}
      }
    }
  ]
}
```

`function.cache_control` is not accepted as an alias. Cache control on
Anthropic `tool_use` or `tool_result` wrappers, and on content nested inside a
tool result, is outside the current gateway scope.

### OpenAI Responses

Both OpenAI profile generations accept a caller-supplied, non-empty
`prompt_cache_key`. Reuse a stable key for requests intended to share the same
stable prefix; do not use a unique request ID.

Legacy example:

```json
{
  "model": "openai-legacy-example",
  "prompt_cache_key": "tenant:acme:prompt-v1",
  "prompt_cache_retention": "in_memory",
  "messages": [
    {"role": "user", "content": "Hello"}
  ]
}
```

GPT-5.6+ example with an explicit input breakpoint:

```json
{
  "model": "openai-example",
  "prompt_cache_key": "tenant:acme:prompt-v1",
  "prompt_cache_options": {
    "mode": "explicit",
    "ttl": "30m"
  },
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "Large stable instructions",
          "prompt_cache_breakpoint": {"mode": "explicit"}
        }
      ]
    },
    {"role": "user", "content": "Hello"}
  ]
}
```

GPT-5.6+ breakpoints are accepted only on input text, image, and file blocks.
Legacy profiles reject `prompt_cache_options` and block breakpoints; GPT-5.6+
profiles reject `prompt_cache_retention`.

### Usage and operational notes

When reported by the provider, buffered responses and the final SSE usage chunk
include cache details in the Chat Completions usage object:

```json
{
  "usage": {
    "prompt_tokens": 1200,
    "completion_tokens": 80,
    "total_tokens": 1280,
    "prompt_tokens_details": {
      "cached_tokens": 900,
      "cache_write_tokens": 0
    }
  }
}
```

`prompt_tokens_details` and each nested field are omitted when the provider does
not report them; an explicitly reported zero remains zero. For Anthropic,
`prompt_tokens` includes uncached input, cache-created input, and cache-read
input, so cache use does not under-report the logical prompt size. For streams,
set `stream_options.include_usage` to receive the final usage chunk.

Provider cache writes and longer retention can have different prices from
ordinary input or cache reads. Check current provider pricing before enabling
them broadly. The gateway does not guarantee a cache hit: the provider decides
whether an entry is written, retained, and reused, and usage details are the
observable result.

The following remain deliberately unsupported: cache warm-up through Anthropic
`max_tokens: 0`, route-injected cache policies, automatic breakpoint selection,
automatic key derivation or sharding, Bedrock/Vertex cache behavior, hit-rate or
cost reporting, provider cache deletion/invalidation APIs, and treating
`previous_response_id`, `conversation`, or `store` as prompt-cache controls.

Run the checks and start the server:

```sh
make check
make run
```

The default listener is `:8080`, which binds to all available network
interfaces. For local-only access, set `CCT_SERVER_ADDRESS=127.0.0.1:8080`.
Before exposing the gateway remotely, terminate TLS at a trusted reverse proxy;
otherwise the incoming bearer credential would travel over plaintext HTTP.

Runtime logs are human-readable on stdout and are also written as JSON to
`logs/server.log`. The file rotates at 100 MiB, and the current file plus its
backups are limited to five files in total.

## Environment overrides

Supported scalar values use `CCT_` names with underscores:

```sh
CCT_SERVER_ADDRESS=127.0.0.1:9090
CCT_SERVER_GIN_MODE=release
CCT_SERVER_READ_HEADER_TIMEOUT=5s
CCT_SERVER_IDLE_TIMEOUT=1m
CCT_SERVER_SHUTDOWN_TIMEOUT=10s
CCT_SERVER_MAX_BODY_BYTES=1048576
CCT_SERVER_MAX_STREAM_BYTES=67108864
CCT_SERVER_MAX_SSE_EVENT_BYTES=8388608
CCT_GATEWAY_RESPONSE_HEADER_TIMEOUT=5m
CCT_TRANSFORMER_MODE=compatible
CCT_TRANSFORMER_INSTRUCTION_POLICY=preserve_messages
CCT_TRANSFORMER_ANTHROPIC_ENDPOINT=messages
CCT_TRANSFORMER_DEFAULT_MAX_OUTPUT_TOKENS=1024
```

An unset variable preserves the YAML value. A scalar explicitly set to an empty
string is applied as empty and is therefore distinguishable from an unset
variable; validation rejects empty values where the setting is required. Set
`CCT_TRANSFORMER_DEFAULT_MAX_OUTPUT_TOKENS` to `null` or an empty string to clear
the YAML default.

Map and list settings use strict JSON and replace the complete YAML collection:

```sh
CCT_GATEWAY_UPSTREAMS='{"openai":{"endpoint":"responses","url":"https://api.openai.com/v1/responses"}}'
CCT_GATEWAY_ROUTES='{"openai-example":"openai"}'
CCT_TRANSFORMER_PROFILES='[{"provider":"openai","endpoint":"responses","model":"gpt-example","content":{"text":true}}]'
CCT_TRANSFORMER_ROUTES='[{"alias":"openai-example","targets":{"responses":"gpt-example"}}]'
```

For these collections, an unset variable preserves YAML; an empty string or
`null` clears it. Use `{}` to clear a map and `[]` to clear a list. JSON values
replace rather than merge with the YAML collection.

## Response behavior

Buffered success responses are raw Chat Completions JSON, not a transformer
envelope. HTTP failures use `{ "error": { "message", "type", "param", "code" }
}`. Successful streams contain Chat Completions `data:` frames and end with
`data: [DONE]`; failed or non-terminal streams never emit a false completion
marker.

Lossy-conversion warnings may be returned in the bounded, sanitized
`X-Transformer-Diagnostics` response header. For SSE it is declared as a
trailer. Client disconnects cancel request-body reads, upstream work, and stream
processing.

## Go library

The public conversion API lives in `pkg/transformer`. Protocol-specific codecs
and the canonical intermediate representation are implementation details under
`internal`.

## Development

```sh
make fmt
make test
make test-race
make vet
```
