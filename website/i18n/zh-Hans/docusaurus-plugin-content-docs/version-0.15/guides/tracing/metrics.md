---
title: Metrics
description: "在 CubeLoop tracing 中收集 token 用量、延迟和首包时间指标。"
sidebar_position: 5
---

# 使用 `Meter` 采集指标

Span 描述单次运行的形态；直方图描述整个集群的形态。
`cubeloop.tracing.Meter` 与 `Tracer` 对称，发出 OTel GenAI 指标集，让 dashboard
开箱即用。

## 发出的指标

| Instrument | 说明 |
|---|---|
| `gen_ai.client.operation.duration` | 直方图（秒）——在 `chat`、`execute_tool` 和 `invoke_agent` 关闭时记录 |
| `gen_ai.client.operation.time_to_first_chunk` | 直方图（秒）——当 `chat` 至少收到一个内容 chunk 时记录 |
| `gen_ai.client.token.usage` | 直方图（`{token}`）——每次 chat 响应按 `gen_ai.token.type`（`input`、`output`）各记录一次 |

每个数据点携带操作、provider 和请求模型属性，因此失败/已取消的请求
（没有响应体或响应模型落地）仍可按请求内容分组：

- `gen_ai.operation.name` —— `chat` / `execute_tool` / `invoke_agent`
- `gen_ai.provider.name` —— `anthropic`、`openai`、`openai_responses`……
- `gen_ai.request.model` —— 例如 `claude-sonnet-4-6`
- `gen_ai.response.model` —— 例如 `claude-sonnet-4-6`（仅成功时）
- `gen_ai.token.type` —— `input` 或 `output`（仅 token 用量）

## 挂载 Meter

RAII 惯用形式——`async with` 全包，无需手动清理：

```python
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter,
)
from cubeloop.tracing import Tracer, Meter
from cubeloop.tracing.exporters import JsonlSpanExporter

async with (
    Tracer(
        service_name="my-bot",
        agent_name="assistant",
        exporters=[JsonlSpanExporter(directory="./cubeloop-traces")],
    ) as tracer,
    Meter(
        resource=tracer.resource,    # 共享 Resource，使 service.* 与 span 匹配
        exporters=[
            OTLPMetricExporter(endpoint="http://otel-collector:4318/v1/metrics"),
        ],
    ) as meter,
    tracer.attached(agent),
    meter.attached(agent),
):
    await agent.prompt("...")
# 退出顺序自动：先 detach（关闭已取消运行的 span，flush trace pipeline）
# → 再 shutdown 两者（flush + 关闭 exporter）。
```

`Meter.attach()` 与 `Tracer.attach()` 相互独立，可以单独使用任一个；
推荐同时使用并共享一个 `Resource`，让后端将它们视为同一服务。

如果需要非 RAII 的显式形式（例如在长时间运行的服务器生命周期内动态挂载
agent）：

```python
tracer_detach = tracer.attach(agent)
meter_detach = meter.attach(agent)
try:
    ...
finally:
    tracer_detach()       # 关闭已取消运行的 span
    meter_detach()        # 取消订阅 meter 的监听器
    await tracer.shutdown()
    await meter.shutdown()
```

## Bucket 边界

duration 直方图采用 OTel GenAI semconv 推荐的边界值（单位：秒）：

```text
0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60, 120, 300
```

OTel 将这些值作为 `_advisory` 边界暴露；后端可以直接使用或覆盖。

## 单个 Meter 对应多个并发 agent

与 `Tracer` 一样，一个 `Meter` 实例可以安全地挂载到同一进程中的多个 agent。
每次 `attach()` 调用都有独立的内部 `_MeterState`，保存各自的 open-ns
时间戳和属性字典，来自两个 agent 的并发运行永远不会共享或相互覆盖指标状态。

```python
meter = Meter(resource=tracer.resource, exporters=[exporter])
meter.attach(agent_a)
meter.attach(agent_b)
```

两个 agent 各自发出独立的 duration / token / TTFC 观测值，可按
`gen_ai.agent.name`（使用 `Tracer(agent_name=…)` 时在 Resource 层面设置）
或 `gen_ai.request.model` 过滤。

## 关闭

上面的 RAII 形式（`async with … as tracer, … as meter, tracer.attached(agent),
meter.attached(agent):`）会自动处理关闭顺序：先 detach 内层 → 再执行外层
`Tracer/Meter` 的 `__aexit__`，调用 `shutdown()`。

对于手动模式，顺序很重要——`tracer_detach()` 必须在 `tracer.shutdown()`
之前运行，确保正在飞行的取消操作遗留的 span 在同一批次中被关闭和导出：

```python
finally:
    tracer_detach()
    meter_detach()
    await tracer.shutdown()
    await meter.shutdown()
```

`Meter.shutdown()` 等待指标 reader 完成 flush，然后关闭它。
`PeriodicExportingMetricReader` 按固定间隔（默认 60 秒）导出——`shutdown`
是在进程退出前保证最后一个时间窗口数据落地的唯一方式。

## 查询示例（Honeycomb）

过去一小时内按 provider 划分的 p95 chat 延迟：

```text
VISUALIZE  P95(duration_s)
GROUP BY   gen_ai.provider.name
WHERE      gen_ai.operation.name = "chat"
TIME       last 1 hour
```

按模型划分的 token 用量：

```text
VISUALIZE  SUM(token_count)
GROUP BY   gen_ai.request.model, gen_ai.token.type
WHERE      gen_ai.operation.name = "chat"
```

替换为你所用后端的查询 DSL——属性名和聚合方式相同。
