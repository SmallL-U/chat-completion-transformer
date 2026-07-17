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
      temperature: true
      top_p: true
      structured_output: true
      strict_tools: true
      parallel_tool_calls: true
      content: { text: true }
    - provider: anthropic
      endpoint: messages
      model: claude-example
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

Run the checks and start the server:

```sh
make check
make run
```

The default listener is `127.0.0.1:8080`. If the gateway is exposed remotely,
terminate TLS at a trusted reverse proxy before forwarding requests; otherwise
the incoming bearer credential would travel over plaintext HTTP.

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
