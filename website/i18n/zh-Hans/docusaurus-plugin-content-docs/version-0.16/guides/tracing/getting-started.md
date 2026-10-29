---
title: 快速开始
description: "在 CubeLoop 中快速上手 OpenTelemetry tracing——安装、配置并导出 span。"
sidebar_position: 2
---

# Tracing 快速开始

## 安装 extra

CubeLoop 将 OpenTelemetry 作为可选依赖：

```bash
pip install "cubeloop[tracing]"
```

这会拉取 `opentelemetry-sdk` 及相关包。若未安装该 extra，导入 `cubeloop.tracing`
时会抛出清晰的错误提示，让你在导入阶段就能发现问题，而不是运行到一半才报错。

## 挂载 Tracer

最简端到端配置——本地 JSONL 导出，符合习惯的 RAII 模式：

```python
import asyncio
from cubeloop import Agent
from cubeloop.providers.anthropic import AnthropicProvider
from cubeloop.tracing import Tracer
from cubeloop.tracing.exporters import JsonlSpanExporter


async def main() -> None:
    agent = Agent(
        model=AnthropicProvider(provider_id="anthropic", api_key="…").model("claude-sonnet-4-6"),
        system_prompt="Be helpful.",
    )

    async with (
        Tracer(
            service_name="my-bot",
            agent_name="assistant",
            exporters=[JsonlSpanExporter(directory="./cubeloop-traces")],
        ) as tracer,
        tracer.attached(agent),
    ):
        await agent.prompt("Say hello.")
        await agent.wait_for_idle()
    # 退出时：自动 detach（关闭所有已取消运行的 span，等待 flush）
    # + tracer shutdown（flush + 关闭 exporter）。无需 try/finally。


asyncio.run(main())
```

如果无法改写成 `async with`（例如将 agent 传递给长生命周期 web handler），
也可以使用显式模式，效果完全等价：

```python
detach = tracer.attach(agent)
try:
    await agent.prompt("…")
finally:
    # 以下两者单独调用均可：
    #   await detach()                # 等待已调度的 flush
    #   await tracer.shutdown()        # flush + 关闭 exporter
    detach()
    await tracer.shutdown()
```

即使完全忘记清理，`Tracer` 默认会注册一个 `atexit` hook，在进程退出时
同步 flush 缓冲 span——传入 `atexit_flush=False` 可关闭此行为，或在开发阶段
把它当作安全网使用。（`SIGKILL` 或 `os._exit` 时不会触发；如需保证必达，
请使用 OTel 的同步 `SimpleSpanProcessor`。）

每次运行产生一个 JSONL 文件，按 `trace_id` 分片：

```text
./cubeloop-traces/
  2026-05-19/
    8e1c9a3f4b2d…d976a.jsonl   ← 一条 trace，一个文件，每行一个 span
```

一条 trace 代表整次运行，包括所有嵌套 subagent 运行（它们继承父级的
`trace_id`，因此写入同一文件）。每个 span 仍携带 `cubeloop.run_id` 属性，
可按单次运行过滤。

用任何支持 OTLP/JSON 的工具，或直接用 `jq` 打开：

```bash
jq -r '"\(.name)  \(.attributes."gen_ai.operation.name" // "")"' \
   cubeloop-traces/2026-05-19/*.jsonl
# invoke_agent  invoke_agent
# cubeloop.turn
# chat claude-sonnet-4-6  chat
```

## Span 层级

单次 prompt 经过一轮 LLM 往返，recorder 产生三个 span：

```text
invoke_agent assistant              [INTERNAL]   gen_ai.operation.name=invoke_agent
└── cubeloop.turn                     [INTERNAL]   cubeloop.turn.index=0
    └── chat <model>                [CLIENT]     gen_ai.operation.name=chat
```

模型调用工具时，每个工具多一层：

```text
invoke_agent assistant
└── cubeloop.turn                     ← turn index 0
    ├── chat <model>                ← 第一轮往返
    └── execute_tool <tool_name>    ← gen_ai.tool.name, gen_ai.tool.call.id
└── cubeloop.turn                     ← turn index 1（工具结果返回后的响应）
    └── chat <model>
```

MCP 工具的 `execute_tool` span 会有一个 CLIENT 子节点：

```text
execute_tool <tool_name>            [INTERNAL]   cubepi 侧封装
└── tools/call <tool_name>          [CLIENT]     gen_ai.operation.name=execute_tool
                                                  mcp.method.name=tools/call
                                                  mcp.session.id=…
                                                  server.address / server.port
```

CLIENT span 会将 W3C `traceparent` 注入出站 HTTP 头，让下游已埋点的
MCP 服务器能够续接 trace。

## 取消、错误与中止

recorder 将取消视为控制信号，而非失败：

- 流式传输中调用 `agent.abort()` → span 以 `cubeloop.aborted=true` 和
  `error.type=cubeloop.aborted` 关闭，**状态为 UNSET**（遵循 OTel 指导原则——
  取消不是错误）。
- provider 抛出异常 → chat/turn/root 以 **状态 ERROR** 关闭，chat span 上有
  `exception` 事件，`error.type` 由异常类派生（`timeout`、`connection_error`、
  完全限定类名……）。
- MCP `tools/call` 返回 `isError=true` → CLIENT span 以 ERROR +
  `error.type=mcp.is_error` 关闭。

无论哪种情况，`detach()` 和 `tracer.shutdown()` 始终会关闭运行遗留的所有
未关闭 span，已取消的运行依然会出现在后端，不会悄悄消失。

## 每个 span 的属性

默认（无需额外开启）：

- `invoke_agent`（根节点）—— `gen_ai.operation.name`、`gen_ai.provider.name`、
  `gen_ai.agent.name`、`cubeloop.run_id`、`cubeloop.agent.system_prompt.sha256`、
  `cubeloop.agent.tools`（名称列表）、`cubeloop.input_messages.count`、
  `cubeloop.output_messages.count`
- `cubeloop.turn` —— `cubeloop.turn.index`、`cubeloop.turn.stop_reason`、
  `cubeloop.turn.tool_calls.count`、`cubeloop.turn.terminated_by_tool`、
  `cubeloop.run_id`
- `chat <model>` —— `gen_ai.operation.name`、`gen_ai.provider.name`、
  `gen_ai.request.model`、`gen_ai.request.max_tokens` / `temperature` /
  `top_p`、`gen_ai.request.stream`、`gen_ai.usage.input_tokens` /
  `output_tokens` / `cache_read_input_tokens` / `cache_creation_input_tokens` /
  `reasoning_output_tokens`、`gen_ai.response.model` /
  `finish_reasons` / `id`、`gen_ai.response.time_to_first_chunk`，以及
  OpenAI 专属字段（`openai.api.type`、service tier、system fingerprint）
- `execute_tool <tool_name>` —— `gen_ai.operation.name=execute_tool`、
  `gen_ai.tool.name`、`gen_ai.tool.call.id`、`gen_ai.tool.description`、
  `gen_ai.tool.type`、`cubeloop.tool.is_error`、`cubeloop.tool.execution_mode`
- `tools/call <tool_name>`（仅 MCP）—— `mcp.method.name`、`mcp.session.id`、
  `mcp.protocol.version`、`server.address`、`server.port`、`gen_ai.tool.name`

可选，通过 `Tracer(record_content=True)` 开启：
`gen_ai.input.messages`、`gen_ai.output.messages`、`gen_ai.system_instructions`、
`gen_ai.tool.definitions`、`gen_ai.tool.call.arguments`、
`gen_ai.tool.call.result`、`cubeloop.llm.raw_request`、
`cubeloop.llm.raw_response`。详见[内容记录与脱敏](./content-recording)。

## 多 agent 单进程

`Tracer` 和 `Meter` 均可安全地在多个 agent 之间共享——多次调用
`attach(agent)` 即可。每次 attach 都有独立的 recorder / metric 状态，
并发 agent 不会共享或相互覆盖 span 或直方图状态；MCP CLIENT span 根据
哪个 agent 的 `execute_tool` span 是父节点来路由到正确的 Tracer。

使用 RAII 方式，叠加多个 agent 只需一个 `async with`：

```python
async with (
    Tracer(...) as tracer,
    tracer.attached(agent_a),
    tracer.attached(agent_b),
):
    await asyncio.gather(agent_a.prompt("…"), agent_b.prompt("…"))
```

## 为单次运行打标签

`cubeloop.tracing.tracing_context` 将 per-run 标签和元数据作用于
`invoke_agent` span，非常适合记录 `user_id`、`session_id`、A/B 测试分组
等需要在后端过滤的字段：

```python
from cubeloop.tracing import tracing_context

async with tracer.attached(agent):
    with tracing_context(tags=["beta-arm"], metadata={"user_id": "u-42"}):
        await agent.prompt("Hello.")
```

span 上的属性：

- `cubeloop.tags = ("beta-arm",)`
- `cubeloop.metadata.user_id = "u-42"`

`cubeloop.metadata.*` 前缀防止用户键与 recorder 自有 schema（如 `cubeloop.run_id`）
冲突。标签和元数据的 contextvar 均为 per-asyncio-task 作用域，并发 agent
各自看到独立的值；嵌套的 `tracing_context` 块会合并（标签追加，元数据键取并集，
内层优先）。

## 下一步

- [OTLP 与后端](./otlp) —— Jaeger、Tempo、Honeycomb、Datadog……
- [内容记录与脱敏](./content-recording)
- [Metrics](./metrics)
