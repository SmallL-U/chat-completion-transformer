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

系统不会猜测模型名。Gateway 路由负责选择供应商端点；Transformer 路由把同一个
别名映射到供应商模型，并要求存在精确匹配的 `(provider, endpoint, model)` 能力
配置。

网关只支持直连 OpenAI Responses 和 Anthropic Messages HTTP 端点。远程上游必须
使用 HTTPS；本地开发可以使用回环地址 HTTP。上游 URL 不允许包含用户信息、查询
参数或 fragment，网关也不会跟随重定向。

## Prompt 缓存控制

公开的 `POST /v1/chat/completions` 只接受官方 Chat Completions 请求结构。网关只映射
其中的标准 prompt cache 控制，不实现本地缓存，不保存 prompt、KV 或供应商 cache
entry，也不会生成 `prompt_cache_key`。

可接受的缓存控制只有：

- 顶层 `prompt_cache_key: string`；
- 顶层 `prompt_cache_options`，只包含可选的
  `mode: "implicit" | "explicit"` 和 `ttl: "30m"`；
- 顶层已 deprecated 的 `prompt_cache_retention: "in_memory" | "24h"`；
- Chat Completions create schema 明确定义的位置上的
  `prompt_cache_breakpoint: {"mode":"explicit"}`。

Anthropic `cache_control` 是 Messages 上游字段，在 Chat Completions 请求的顶层、message
content、tool 或 function 中都不合法。三种 transformer mode 都会在调用上游前以 HTTP
400 拒绝这类请求。

### Capability profile

Prompt cache 行为必须显式配置在精确的模型 profile 上。省略 `prompt_cache.mode` 时会
归一化为 `none`；系统不会根据模型名推断 API 代际。

| `prompt_cache.mode` | 有效目标 | 接受的控制字段 |
| --- | --- | --- |
| `none` | 任意有效 profile | 无法表达任何 cache control |
| `anthropic` | 直连 Anthropic Messages | 将标准 `prompt_cache_options` 和 content breakpoint 转成 `cache_control` |
| `openai_legacy` | OpenAI Responses | `prompt_cache_key` 和显式启用的 retention 值 |
| `openai_5_6` | OpenAI Responses | `prompt_cache_key`、`prompt_cache_options`、content breakpoint 和显式启用的 retention 值 |

Retention 与 `prompt_cache_options` 相互独立，必须在实际支持它的精确 OpenAI profile 中
逐项启用：

```yaml
prompt_cache:
  mode: openai_5_6
  extended_retention_24h: true
```

`in_memory_retention` 和 `extended_retention_24h` 分别允许调用方发送
`prompt_cache_retention: "in_memory"` 和 `prompt_cache_retention: "24h"`，不会替调用方
选择或注入 retention 值。示例只启用当前较新 OpenAI 模型支持的 `24h`；实际配置只应
启用该上游模型支持的值，`in-memory` 等其他拼写不会被自动改写为别名。

已知缓存字段会经过类型校验和 capability gate。字段结构或位置非法时，请求会被
拒绝；合法字段但目标 profile 不支持时，`strict` 模式失败，`compatible` 和
`emulate` 模式丢弃该字段并返回 Transformer warning。任意 extension 都不会因此获得
通用透传能力。

### Anthropic Messages

客户端仍然发送标准 Chat Completions 请求。例如显式缓存断点应写成：

```json
{
  "model": "anthropic-example",
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
          "text": "较长且稳定的系统指令",
          "prompt_cache_breakpoint": {"mode": "explicit"}
        }
      ]
    },
    {"role": "user", "content": "你好"}
  ]
}
```

Anthropic encoder 只在上游 Messages JSON 中生成原生 `cache_control`：

- `mode: "implicit"` 生成顶层 automatic cache control；
- `mode: "explicit"` 不生成顶层 automatic control；
- 标准 content breakpoint 转成原生 content-block control；
- Chat 默认或显式的 30 分钟最短 TTL 映射为 Anthropic `1h`，因为 `5m` 无法满足
  调用方请求的最短生命周期；
- implicit mode 在 automatic marker 之外写入最新三个显式 marker，explicit mode 写入
  最新四个；
- 没有标准 cache control 的请求不会被主动开启缓存。

`prompt_cache_key` 和 `prompt_cache_retention` 都没有等价的 Anthropic per-request 字段。
strict mode 会拒绝这种保真损失，compatible/emulate mode 会告警并丢弃。Chat tool
definition 没有缓存字段，因此公开 API 不暴露 Anthropic tool `cache_control`。

### OpenAI Responses

两种 OpenAI profile 代际都会转发调用方提供的 string `prompt_cache_key`。Chat schema
没有非空约束，但实际使用时仍应让共享同一稳定前缀的请求复用有意义的稳定 key，不要
使用每次都不同的 request ID。

Legacy 示例：

```json
{
  "model": "openai-legacy-example",
  "prompt_cache_key": "tenant:acme:prompt-v1",
  "prompt_cache_retention": "in_memory",
  "messages": [
    {"role": "user", "content": "你好"}
  ]
}
```

GPT-5.6+ 显式 input 断点示例：

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
          "text": "较长且稳定的系统指令",
          "prompt_cache_breakpoint": {"mode": "explicit"}
        }
      ]
    },
    {"role": "user", "content": "你好"}
  ]
}
```

GPT-5.6+ profile 会转发 `prompt_cache_options` 和可由 Responses 表达的 content
breakpoint；legacy profile 不支持这两个控制。两种 profile 都只转发 capability flag
明确启用的 `prompt_cache_retention` 枚举值；该字段与 `prompt_cache_options` 保持独立。

公开 decoder 只在 Chat Completions create schema 定义的位置接受 breakpoint。如果合法
的 Chat content part 无法由选中的 Responses profile 表达，strict mode 会失败，
compatible/emulate mode 会告警并丢弃该 breakpoint，不会把断点移动到另一个边界。

### Usage 与运行提示

供应商实际报告缓存统计时，普通响应和 SSE 最后的 usage chunk 会使用 Chat
Completions usage 结构返回明细：

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

供应商未报告时，`prompt_tokens_details` 或其中单个字段会被省略；供应商明确报告的
`0` 会保留为 `0`。对 Anthropic 而言，`prompt_tokens` 包含未缓存输入、cache-create
输入和 cache-read 输入，因此启用缓存后不会低报完整逻辑 prompt 大小。流式请求需要
设置 `stream_options.include_usage` 才会收到最后的 usage chunk。

供应商可能对 cache write、较长 retention、普通输入和 cache read 使用不同价格。
大范围开启前请核对供应商当前定价。网关不保证缓存命中：是否写入、保留和复用由
供应商决定，usage 明细只是可观察结果。

以下能力明确延期，当前不宣称支持：通过 Anthropic `max_tokens: 0` 预热缓存、由 route
注入缓存策略、派生新断点、自动派生或分片 key、Bedrock/Vertex 缓存行为、命中率或成本
报表、供应商缓存删除/失效 API，以及把 `previous_response_id`、`conversation` 或
`store` 当作 prompt cache 控制。

运行检查并启动服务：

```sh
make check
make run
```

默认监听地址为 `:8080`，会绑定所有可用网络接口。如果只允许本机访问，请设置
`CCT_SERVER_ADDRESS=127.0.0.1:8080`。远程暴露网关前，应先由可信反向代理终止 TLS；
否则传入的 Bearer 凭据会经过明文 HTTP。

运行日志会以易读文本输出到 stdout，同时以 JSON 写入 `logs/server.log`。单个文件
达到 100 MiB 后轮转，当前文件与历史文件合计最多保留 5 个。

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
