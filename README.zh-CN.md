# Chat Completion Transformer

[English](README.md) | 简体中文

这是一个兼容 OpenAI Chat Completions 的网关，同时也是 Go 协议转换库。服务接收
Chat Completions 请求，调用 OpenAI Responses 或 Anthropic Messages，再把普通响应
和 SSE 流转换回 Chat Completions 格式。

网关只负责协议转换，不保存、管理、替换或校验供应商 API key。

## HTTP API

服务只公开以下两个路由：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/healthz` | 存活检查 |
| `POST` | `/v1/chat/completions` | 兼容 OpenAI 的 Chat Completions 网关 |

旧的 `/v1/transform/*` 调试路由不属于公开 API。未知路由和不支持的方法都会返回
OpenAI 格式的 JSON 错误。

每个 Chat Completions 请求必须且只能包含一个
`Authorization: Bearer <key>` 请求头。目标为 OpenAI Responses 时，该请求头会被
透传；目标为 Anthropic Messages 时，Bearer token 会作为 `x-api-key` 发送。key
只用于当前请求，配置文件中不会读取或保存 key。

示例：

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Authorization: Bearer provider-api-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openai-example",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

设置 `"stream": true` 即可使用 Chat Completions SSE。启用
`stream_options.include_usage` 后，普通 chunk 包含 `usage: null`，最后的 usage chunk
包含供应商返回的 token 用量。网关目前每次请求只支持一个候选结果，`n > 1` 会被
拒绝。

## 配置与运行

服务读取项目根目录下的 `config.yml`。仓库默认配置有意不包含路由，因此
`/healthz` 可以直接使用，Chat 请求则需要先配置上游和转换路由。

每个公开模型别名必须同时出现在 `gateway.routes` 和 `transformer.routes` 中：

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

系统不会猜测模型名。Gateway 路由负责选择供应商端点；Transformer 路由把同一个
别名映射到供应商模型，并要求存在精确匹配的 `(provider, endpoint, model)` 能力
配置。

网关只支持直连 OpenAI Responses 和 Anthropic Messages HTTP 端点。远程上游必须
使用 HTTPS；本地开发可以使用回环地址 HTTP。上游 URL 不允许包含用户信息、查询
参数或 fragment，网关也不会跟随重定向。

运行检查并启动服务：

```sh
make check
make run
```

默认监听地址为 `127.0.0.1:8080`。如果需要远程暴露网关，应先由可信反向代理终止
TLS，再把请求转发给本服务；否则传入的 Bearer 凭据会经过明文 HTTP。

## 环境变量覆盖

支持的标量配置使用下划线形式的 `CCT_` 环境变量：

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

未设置环境变量时保留 YAML 值。标量环境变量被显式设为空字符串时，会按空值应用，
因此可以与“未设置”区分；必填配置的空值会被校验拒绝。将
`CCT_TRANSFORMER_DEFAULT_MAX_OUTPUT_TOKENS` 设为 `null` 或空字符串，可以清除 YAML
中的默认值。

Map 和 list 配置使用严格 JSON，并整体替换对应的 YAML 集合：

```sh
CCT_GATEWAY_UPSTREAMS='{"openai":{"endpoint":"responses","url":"https://api.openai.com/v1/responses"}}'
CCT_GATEWAY_ROUTES='{"openai-example":"openai"}'
CCT_TRANSFORMER_PROFILES='[{"provider":"openai","endpoint":"responses","model":"gpt-example","content":{"text":true}}]'
CCT_TRANSFORMER_ROUTES='[{"alias":"openai-example","targets":{"responses":"gpt-example"}}]'
```

对这些集合而言：环境变量未设置时保留 YAML；空字符串或 `null` 会清空集合；Map
使用 `{}` 清空，list 使用 `[]` 清空。JSON 值会整体替换而不是合并 YAML 集合。

## 响应行为

普通成功响应直接返回 Chat Completions JSON，不再包裹 Transformer envelope。HTTP
错误统一使用 `{ "error": { "message", "type", "param", "code" } }`。成功流包含
Chat Completions `data:` 帧，并以 `data: [DONE]` 结束；失败或非终态的流不会发送
虚假的完成标记。

有损转换警告可能通过有大小限制且已脱敏的 `X-Transformer-Diagnostics` 响应头
返回；SSE 中该字段会预先声明为 trailer。客户端断开连接会取消请求体读取、上游
请求和流处理。

## Go 开发库

公开转换 API 位于 `pkg/transformer`。各协议编解码器和 Canonical 中间表示属于
内部实现，位于 `internal`。

## 开发

```sh
make fmt
make test
make test-race
make vet
```
