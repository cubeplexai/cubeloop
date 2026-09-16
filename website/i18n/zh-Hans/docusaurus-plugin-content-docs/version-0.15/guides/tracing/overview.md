---
title: Tracing 概览
description: "CubeLoop 基于 OpenTelemetry 的 tracing 系统概览——span、属性、exporter 与语义约定。"
sidebar_position: 1
---

# Tracing 概览

CubeLoop 发出符合
[GenAI Semantic Conventions v1.41](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
的 [OpenTelemetry](https://opentelemetry.io/) span，任何兼容 OTel 的后端
（Jaeger、Tempo、Honeycomb、Datadog、AWS X-Ray、Azure Monitor……）都能直接接收
agent 运行数据，无需额外自定义埋点。

将 `Tracer` 挂载到 `Agent` 后，每次 prompt 都会产生一棵 span 树，可对其进行
聚合、查询，并与服务内其他链路数据关联：

```
trace
└── invoke_agent  14425.8ms  [0x1cd97cdb]         ← 每次 agent.prompt() 一个
    ├── cubeloop.turn  1283.1ms  [0x5cfda93e]        ← 每次 LLM 往返一个
    │   ├── chat deepseek-v4-flash  1208.7ms  tok 6845/68  [0x0d130229]
    │   └── execute_tool subagent  9610.2ms  subagent  [0x38bdd10a]
    │       └── invoke_agent  9601.0ms  [0x8094f99b]   ← subagent 运行，嵌套其中
    │           └── cubeloop.turn  9598.4ms  [0x57c5cfc7]
    │               ├── chat deepseek-v4-flash  1190.3ms  [0x8205ca6b]
    │               └── execute_tool web_search  6500.2ms  web_search  [0xca4e59fc]
    └── cubeloop.turn  491.9ms  ERROR  [0xce25f242]
        └── chat deepseek-v4-flash  427.2ms  ERROR  [0x0bff68ec]
            └── error: Error code: 400 - ... `tool_use` ids were found without
                `tool_result` blocks immediately after: call_01_...
```

每一层都携带标准 `gen_ai.*` 属性——`gen_ai.operation.name`、
`gen_ai.request.model`、`gen_ai.provider.name`、`gen_ai.usage.input_tokens`、
`gen_ai.usage.output_tokens`、`gen_ai.response.finish_reasons`……

## 开箱即用的功能

- **Tracer** —— 构建 SDK `TracerProvider`，为每个 exporter 挂载一个
  `BatchSpanProcessor`，将 CubeLoop 事件流接入 span。
- **Meter** —— 用于 OTel 直方图的同级组件：
  `gen_ai.client.operation.duration`、`gen_ai.client.operation.time_to_first_chunk`、
  `gen_ai.client.token.usage`。
- **JsonlSpanExporter** —— 将每条 span 以 JSON 单行写入
  `./cubeloop-traces/<date>/<trace_id>.jsonl`。文件按 `trace_id` 分片，
  因此一个文件包含整条 trace——运行本身加上所有嵌套 subagent 运行（继承同一
  trace）。适合本地开发与离线调试；可与任何支持 JSONL 的 OTel 查看工具配合，
  也可与 [`cubeloop trace` CLI](./cli) 配合。
- **OTLP** —— 通过 `opentelemetry-exporter-otlp-proto-http`（HTTP）或
  `…-grpc` 带入自己的 exporter，传入 `Tracer(exporters=[…])` 即可。
- **W3C trace context 传播** —— 出站 MCP 调用会自动将当前 `traceparent`
  注入 HTTP 头，让下游已埋点的 MCP 服务器能够续接 trace。
- **`tracer.attached(agent)` / `meter.attached(agent)`** —— 以 RAII 方式
  包裹 attach/detach 的 async context manager，清理只需一个 `async with`
  块，无需显式 `try/finally`。
- **`atexit` flush hook** —— `Tracer(atexit_flush=True)`（默认开启）注册
  进程退出处理程序，在进程正常退出时同步 flush 缓冲 span，即使忘记调用
  `await tracer.shutdown()` 也不会丢失数据。
- **`tracing_context()`** —— 通过 contextvar 作用域块为单次运行设置标签和
  元数据（`cubeloop.tags = ("beta-arm",)`、`cubeloop.metadata.user_id = "u-42"`），
  并发 agent 各自看到独立的值。
- **中间件自有 provider 自动接入 trace** —— 中间件通过
  `Middleware.providers()` 暴露自己持有的 `BaseProvider`，
  `Recorder.attach()` 会自动给这些 provider 接上 listener 注册表，它们的
  `chat` span 就和 agent 主调用落进同一条 trace。`CompactionMiddleware`
  就用这个机制把摘要 LLM 调用呈现为嵌在
  `cubeloop.compaction.summarize` 下的 `chat <summary-model>` —— 详见
  [compaction 指南](../middleware/compaction#tracing)。

## 开销

- 每次 agent 运行一个纯 Python recorder，订阅 agent 的事件流和 provider
  的监听注册——不做 monkey-patching，不新起线程。
- 每层一个 OTel SDK span。`BatchSpanProcessor` 在后台批量导出，不阻塞热路径。
- **默认不记录任何 payload。** `gen_ai.input.messages`、`gen_ai.output.messages`、
  原始请求/响应以及工具参数/结果均需通过 `record_content=True` 显式开启，
  避免意外将 PII 发送到后端。详见[内容记录与脱敏](./content-recording)。

## 各功能使用场景

| 需求 | 使用 |
|---|---|
| 跟踪本地单次 agent 运行并检查 JSONL 文件 | `Tracer` + `JsonlSpanExporter` |
| 发送到 Jaeger / Tempo / Honeycomb / Datadog | `Tracer` + OTLP exporter |
| 在 span 旁边获取延迟与 token 直方图 | `Meter` 与 `Tracer` 并用 |
| 记录 prompt / 模型输出用于评测 | `Tracer(record_content=True)` |
| 在数据离开进程前脱敏 PII | `Tracer(redact=…)` |
| 为运行打上 `user_id` / `session_id` / A-B 分组标签 | `tracing_context(tags=…, metadata=…)` |
| 一行代码完成清理，无需 try/finally | `async with tracer.attached(agent): …` |
| 忘记调用 `shutdown()` 也不丢 span | `Tracer(atexit_flush=True)`（默认） |
| 从上游服务续接 trace | `Tracer(resource=…)` + W3C `traceparent`（MCP 自动，HTTP 手动） |

## 下一步

- [快速开始](./getting-started) —— 安装 extra 并发出第一批 span
- [OTLP 与后端](./otlp) —— 将 CubeLoop 接入 Jaeger、Tempo、Honeycomb……
- [内容记录与脱敏](./content-recording) —— 安全地记录 prompt 与响应
- [Metrics](./metrics) —— 通过 `Meter` 使用直方图
