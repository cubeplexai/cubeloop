---
title: OTLP 与后端
description: "将 CubeLoop OpenTelemetry trace 导出到 Jaeger、Tempo、Honeycomb 等兼容 OTLP 的后端。"
sidebar_position: 3
---

# 导出到 OTLP 后端

`cubeloop.tracing.Tracer` 接受任何 `opentelemetry.sdk.trace.export.SpanExporter`，
因此 OpenTelemetry 生态中的所有 exporter 均可直接使用。选择传输协议
（HTTP 或 gRPC），指向你的 collector，将 exporter 传入 Tracer 即可。

## HTTP（OTLP/HTTP）

```bash
pip install "cubeloop[tracing]" opentelemetry-exporter-otlp-proto-http
```

```python
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from cubeloop.tracing import Tracer

tracer = Tracer(
    service_name="my-bot",
    service_version="1.4.2",
    deployment_environment="prod",
    agent_name="assistant",
    exporters=[
        OTLPSpanExporter(
            endpoint="http://otel-collector:4318/v1/traces",
            headers={"x-api-key": "…"},  # 后端专属配置
        ),
    ],
)
```

`service_name`、`service_version`、`deployment_environment` 和 `agent_name`
会作为 OTel Resource 属性（`service.*`、`gen_ai.agent.name`、
`deployment.environment.name`）透传，后端无需额外配置即可对运行进行分组。

## gRPC（OTLP/gRPC）

```bash
pip install "cubeloop[tracing]" opentelemetry-exporter-otlp-proto-grpc
```

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)

exporter = OTLPSpanExporter(endpoint="otel-collector:4317", insecure=True)
tracer = Tracer(service_name="my-bot", exporters=[exporter])
```

## 后端配置示例

以下后端均消费 OTLP，区别仅在于 endpoint 和 auth header。

### Jaeger（>=1.50）

Jaeger 原生支持 OTLP/HTTP，端口 4318：

```python
OTLPSpanExporter(endpoint="http://jaeger:4318/v1/traces")
```

### Grafana Tempo

发送到你的 collector，或直接发送到 Tempo 的 OTLP endpoint：

```python
OTLPSpanExporter(endpoint="http://tempo:4318/v1/traces")
```

### Honeycomb

```python
OTLPSpanExporter(
    endpoint="https://api.honeycomb.io/v1/traces",
    headers={"x-honeycomb-team": HONEYCOMB_API_KEY},
)
```

### Datadog（经由 OTel collector）

在 collector 中配置 Datadog exporter，再将数据发送给它：

```python
OTLPSpanExporter(endpoint="http://otel-collector:4318/v1/traces")
```

Datadog 也直接支持原生 OTLP HTTP——格式相同，URL 不同。

### AWS X-Ray（经由 collector）

OTel collector 内置 AWS X-Ray exporter；与其他 OTLP 目标的配置方式相同。

## 续接上游 trace

`Tracer.attach(agent)` 目前会让每次 agent 运行从自己的 trace 开始——
暂无公开 API 可传入入站 `traceparent`，使 span 嵌套到调用方 HTTP handler
的 trace 中。内部钩子（`Tracer._make_parent_context`）为未来的 `run_scope`
特性预留；在该特性发布之前，agent 运行与周边服务 trace 仅通过 resource
属性（`service.name`、`gen_ai.agent.name`、`cubeloop.run_id`）关联。

如果需要立即将上游 trace 延续到 CubeLoop，可以在调用 `agent.prompt(...)` 之前
手动设置 OTel 当前 span，让 agent 的 span 通过 OTel 的环境上下文继承它——
CubeLoop 不会覆盖已有的活跃父节点。

出站方向，MCP `tools/call` 会自动将 W3C `traceparent` 注入 HTTP 头，
让下游已埋点的 MCP 服务器能够续接 trace 并写入其自己的后端。

## 组合多个 exporter

可以传入多个 exporter，它们将接收所有 span。常见模式——JSONL 用于本地调试，
OTLP 用于生产后端：

```python
tracer = Tracer(
    service_name="my-bot",
    exporters=[
        JsonlSpanExporter(directory="./cubeloop-traces"),
        OTLPSpanExporter(endpoint="https://api.honeycomb.io/v1/traces", headers={…}),
    ],
)
```

## Flush

`Tracer` 底层使用 `BatchSpanProcessor`，span 在后台异步导出。推荐使用
`async with` 形式，它会自动走完所有清理路径：

```python
async with Tracer(...) as tracer, tracer.attached(agent):
    await agent.prompt("...")
# 退出时：
#   - detach() 同步执行：关闭所有已取消运行遗留的 span，
#     然后将 flush 调度为 asyncio.Task 并等待完成。
#   - tracer.shutdown() 再次 flush（幂等）并关闭 exporter。
```

如果你保存了手动 `tracer.attach(agent)` 返回的 `detach`，它会返回已调度
的 flush `Task`。两种有效的手动模式：

```python
# (a) 双保险——最安全，两者都执行。
finally:
    detach()                    # 关闭已取消运行的 span，调度 flush
    await tracer.shutdown()     # 等待 force_flush，然后关闭 SDK

# (b) 等待 detach——不同时关闭时的单次调用。
finally:
    await detach()              # 等待已调度的 flush
```

在非 async 上下文（无正在运行的 event loop）中，`detach()` 返回 `None`——
同步部分已执行，但 flush 需由调用方通过 `await tracer.shutdown()` 完成。

即使以上步骤都遗漏了——例如脚本在到达 `finally` 之前就抛出异常——
`Tracer(atexit_flush=True)`（默认）会注册进程退出处理程序，通过
`BatchSpanProcessor` 同步 flush 缓冲 span。`SIGKILL` / `os._exit` 时不运行；
如需保证必达，请使用 `SimpleSpanProcessor`（每个 span 同步导出）。

### 服务路径的后台 flush

在请求处理路径上——FastAPI 端点、gRPC handler、任何调用方在等待 `async with`
块结果的代码——等待 OTLP flush 完成才返回会让你的 p99 被最慢的 exporter 拖累。
改用独立的 `trace()` 辅助函数并传入 `flush="background"`：

```python
from cubeloop.tracing import trace

@app.post("/chat")
async def chat(req: ChatRequest):
    async with trace(tracer, agent, flush="background"):
        reply = await agent.prompt(req.message)
    # 块立即退出——span 导出作为受监管的后台任务运行，与本次响应解耦。
    return {"reply": reply}
```

`force_flush` 和底层 `BatchSpanProcessor` 的导出均在 worker 线程中运行（脱离
event loop），因此即使 collector 缓慢或积压，也不会阻塞用户可见的响应。
Tracer 持有每个进行中 flush 任务的强引用，防止其在导出过程中被 GC；
`await tracer.shutdown()` 会在关闭 exporter 前等待所有未完成的 flush，
确保 span 不丢失。
