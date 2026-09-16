---
title: Metrics
description: "Collect token usage, latency, and time-to-first-content metrics in CubeLoop tracing."
sidebar_position: 5
---

# Metrics with `Meter`

Spans tell you the shape of one run; histograms tell you the shape of the
fleet. `cubeloop.tracing.Meter` mirrors `Tracer` and emits the OTel GenAI
metric set so dashboards work out of the box.

## What it emits

| Instrument | Description |
|---|---|
| `gen_ai.client.operation.duration` | Histogram (seconds) — recorded for `chat`, `execute_tool`, and `invoke_agent` on close |
| `gen_ai.client.operation.time_to_first_chunk` | Histogram (seconds) — recorded for `chat` when at least one content chunk arrived |
| `gen_ai.client.token.usage` | Histogram (`{token}`) — one observation per `gen_ai.token.type` (`input`, `output`) per chat response |

Each point carries the operation, provider, and request model attributes
so failed / cancelled requests (where no response body / response model
landed) are still groupable by what was asked for:

- `gen_ai.operation.name` — `chat` / `execute_tool` / `invoke_agent`
- `gen_ai.provider.name` — `anthropic`, `openai`, `openai_responses`, …
- `gen_ai.request.model` — e.g. `claude-sonnet-4-6`
- `gen_ai.response.model` — e.g. `claude-sonnet-4-6` (success only)
- `gen_ai.token.type` — `input` or `output` (token usage only)

## Attaching a Meter

The idiomatic RAII form — `async with` everything, no manual cleanup:

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
        resource=tracer.resource,    # share Resource so service.* matches spans
        exporters=[
            OTLPMetricExporter(endpoint="http://otel-collector:4318/v1/metrics"),
        ],
    ) as meter,
    tracer.attached(agent),
    meter.attached(agent),
):
    await agent.prompt("...")
# Exit order auto: detach both (closes any cancelled-run spans, flushes
# the trace pipeline) → shutdown both (flush + close exporters).
```

`Meter.attach()` is independent from `Tracer.attach()`. You can run
either on its own; the recommended setup is both, sharing one
`Resource` so the backend treats them as the same service.

If you need the explicit, non-RAII form (e.g. attaching agents
dynamically over the lifetime of a long-running server):

```python
tracer_detach = tracer.attach(agent)
meter_detach = meter.attach(agent)
try:
    ...
finally:
    tracer_detach()       # closes any cancelled-run spans
    meter_detach()        # unsubscribes the meter's listeners
    await tracer.shutdown()
    await meter.shutdown()
```

## Bucket boundaries

The duration histograms ship with the OTel GenAI semconv's recommended
boundaries (in seconds):

```text
0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60, 120, 300
```

OTel exposes these as the `_advisory` boundaries; backends are free to use
them as-is or override.

## Concurrent agents on one Meter

Like `Tracer`, one `Meter` instance is safe to attach to multiple agents
in the same process. Each `attach()` call gets its own internal
`_MeterState` holding the open-ns timestamps and attribute dicts, so
overlapping runs from two agents never share or overwrite each other's
metric state.

```python
meter = Meter(resource=tracer.resource, exporters=[exporter])
meter.attach(agent_a)
meter.attach(agent_b)
```

Both agents emit independent duration / token / TTFC observations,
filterable by `gen_ai.agent.name` (set at the `Resource` level when
`Tracer(agent_name=…)` is used) or by `gen_ai.request.model`.

## Shutting down

The RAII form (`async with … as tracer, … as meter, tracer.attached(agent),
meter.attached(agent):` from above) handles the shutdown ordering for you:
detach inner first → outer `Tracer/Meter` `__aexit__` runs `shutdown()`.

For the manual pattern, order matters — `tracer_detach()` must run before
`tracer.shutdown()` so any spans an in-flight cancellation left open get
closed and exported in the same flush:

```python
finally:
    tracer_detach()
    meter_detach()
    await tracer.shutdown()
    await meter.shutdown()
```

`Meter.shutdown()` awaits a flush of the metric reader, then closes it.
`PeriodicExportingMetricReader` exports on a fixed interval (60 s by
default) — `shutdown` is the only way to guarantee the final window lands
before the process exits.

## Querying example (Honeycomb)

p95 chat latency by provider over the last hour:

```text
VISUALIZE  P95(duration_s)
GROUP BY   gen_ai.provider.name
WHERE      gen_ai.operation.name = "chat"
TIME       last 1 hour
```

Token usage by model:

```text
VISUALIZE  SUM(token_count)
GROUP BY   gen_ai.request.model, gen_ai.token.type
WHERE      gen_ai.operation.name = "chat"
```

Substitute your backend's query DSL — same attributes, same aggregations.
