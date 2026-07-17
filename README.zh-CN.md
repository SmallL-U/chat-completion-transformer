# Chat Completion Transformer

[English](README.md) | 简体中文

这是一个 Go 服务与开发库，用于将 Chat Completions 请求转换为 OpenAI
Responses 或 Anthropic Messages，并将供应商的完整响应和 SSE 流转换回 Chat
Completions 格式。

公开 Go API 位于 `pkg/transformer`。各协议编解码器和 Canonical 中间表示属于
内部实现，位于 `internal`。

## 运行

服务读取项目根目录下的 `config.yml`。

```sh
make check
make run
```

仓库中的默认配置有意不包含模型路由。发送转换请求前，需要添加能力配置和模型
别名路由。以下是最小配置示例：

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

系统不会猜测模型别名：每个目标必须同时具备路由，以及精确匹配的
`(provider, endpoint, model)` 能力配置。

## 环境变量覆盖

`config.yml` 中的每个配置项都可以通过以 `CCT_` 开头的环境变量覆盖。嵌套名称
使用下划线连接，例如：

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

将 `CCT_TRANSFORMER_DEFAULT_MAX_OUTPUT_TOKENS` 设置为 `null` 或空值，可以清除
YAML 中非空的默认值。

两个集合配置使用严格 JSON，因此可以通过一个环境变量替换完整的 YAML 配置项：

```sh
CCT_TRANSFORMER_PROFILES='[{"provider":"openai","endpoint":"responses","model":"gpt-example","content":{"text":true}}]'
CCT_TRANSFORMER_ROUTES='[{"alias":"general","targets":{"responses":"gpt-example"}}]'
```

## HTTP API

| 方法 | 路径 | 输入 | 输出 |
| --- | --- | --- | --- |
| `GET` | `/healthz` | — | 健康状态 JSON |
| `POST` | `/v1/transform/chat-completions/to/openai-responses` | Chat 请求 JSON | 转换结果 |
| `POST` | `/v1/transform/chat-completions/to/anthropic-messages` | Chat 请求 JSON | 转换结果 |
| `POST` | `/v1/transform/openai-responses/to/chat-completions` | Responses JSON | 转换结果 |
| `POST` | `/v1/transform/anthropic-messages/to/chat-completions` | Messages JSON | 转换结果 |
| `POST` | `/v1/transform/openai-responses/sse/to/chat-completions` | Responses SSE | Chat Completions SSE |
| `POST` | `/v1/transform/anthropic-messages/sse/to/chat-completions` | Messages SSE | Chat Completions SSE |

缓冲式路由返回 `{ "value", "diagnostics", "lossless", "ok" }`。当输入有效但
无法转换为当前配置的目标格式时，返回 `422`。流式路由会立即 Flush 每个生成的
帧。首帧之后发现的诊断信息会通过预先声明的 `X-Transformer-Diagnostics`
trailer 返回，其内容为 base64url 编码的 JSON。

所有转换和请求体读取操作都绑定到传入请求的 `Request.Context()`。客户端断开
连接会取消资源解析和流处理；进程关闭则使用配置的优雅关闭超时时间。

## 开发

```sh
make fmt
make test
make test-race
make vet
```
