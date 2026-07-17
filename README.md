# Chat Completion Transformer

English | [简体中文](README.zh-CN.md)

Go service and library for translating Chat Completions requests to OpenAI
Responses or Anthropic Messages, and translating provider responses and SSE
streams back to the Chat Completions shape.

The public Go API lives in `pkg/transformer`. Protocol-specific codecs and the
canonical intermediate representation are implementation details under
`internal`.

## Run

The service reads `config.yml` from the project root.

```sh
make check
make run
```

The repository configuration is intentionally route-free. Add capability
profiles and alias routes before sending transformations. A minimal example is:

```yaml
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
    - alias: general
      targets:
        responses: gpt-example
        messages: claude-example
```

Model aliases are never guessed: every target must have both a route and an
exact `(provider, endpoint, model)` capability profile.

## Environment overrides

Every item in `config.yml` can be overridden with a `CCT_` environment
variable. Nested names use underscores, for example:

```sh
CCT_SERVER_ADDRESS=:9090
CCT_SERVER_GIN_MODE=release
CCT_SERVER_READ_HEADER_TIMEOUT=5s
CCT_SERVER_IDLE_TIMEOUT=1m
CCT_SERVER_SHUTDOWN_TIMEOUT=10s
CCT_SERVER_MAX_BODY_BYTES=1048576
CCT_SERVER_MAX_STREAM_BYTES=67108864
CCT_SERVER_MAX_SSE_EVENT_BYTES=8388608
CCT_TRANSFORMER_MODE=compatible
CCT_TRANSFORMER_INSTRUCTION_POLICY=preserve_messages
CCT_TRANSFORMER_ANTHROPIC_ENDPOINT=messages
CCT_TRANSFORMER_DEFAULT_MAX_OUTPUT_TOKENS=1024
```

Set `CCT_TRANSFORMER_DEFAULT_MAX_OUTPUT_TOKENS` to `null` or an empty value to
clear a non-null YAML default.

The two collection values use strict JSON so that one environment variable can
replace the complete YAML item:

```sh
CCT_TRANSFORMER_PROFILES='[{"provider":"openai","endpoint":"responses","model":"gpt-example","content":{"text":true}}]'
CCT_TRANSFORMER_ROUTES='[{"alias":"general","targets":{"responses":"gpt-example"}}]'
```

## HTTP API

| Method | Path | Input | Output |
| --- | --- | --- | --- |
| `GET` | `/healthz` | — | Health JSON |
| `POST` | `/v1/transform/chat-completions/to/openai-responses` | Chat request JSON | Transform result |
| `POST` | `/v1/transform/chat-completions/to/anthropic-messages` | Chat request JSON | Transform result |
| `POST` | `/v1/transform/openai-responses/to/chat-completions` | Responses JSON | Transform result |
| `POST` | `/v1/transform/anthropic-messages/to/chat-completions` | Messages JSON | Transform result |
| `POST` | `/v1/transform/openai-responses/sse/to/chat-completions` | Responses SSE | Chat Completions SSE |
| `POST` | `/v1/transform/anthropic-messages/sse/to/chat-completions` | Messages SSE | Chat Completions SSE |

Buffered routes return `{ "value", "diagnostics", "lossless", "ok" }` and
use `422` when a valid input cannot be represented for the configured target.
Streaming routes flush each generated frame. Diagnostics discovered after the
first frame are returned in the declared `X-Transformer-Diagnostics` trailer as
base64url-encoded JSON.

All transformation and body-reading work is tied to the incoming
`Request.Context()`. Client disconnects cancel asset resolution and stream
processing; process shutdown uses the configured graceful-shutdown deadline.

## Development

```sh
make fmt
make test
make test-race
make vet
```
