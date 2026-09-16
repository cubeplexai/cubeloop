---
title: OTLP & Backends
description: "Export CubeLoop OpenTelemetry traces to OTLP-compatible backends like Jaeger, Tempo, or Honeycomb."
sidebar_position: 3
---

# Exporting to OTLP Backends

`cubeloop.tracing.Tracer` accepts any `opentelemetry.sdk.trace.export.SpanExporter`,
so anything in the OpenTelemetry ecosystem works. Pick the wire format
(HTTP or gRPC), point it at your collector, hand the exporter to the Tracer.

## HTTP (OTLP/HTTP)

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
            headers={"x-api-key": "…"},  # backend-specific
        ),
    ],
)
```

`service_name`, `service_version`, `deployment_environment`, and `agent_name`
flow through as OTel Resource attributes (`service.*`, `gen_ai.agent.name`,
`deployment.environment.name`) so backends can group runs without further
config.

## gRPC (OTLP/gRPC)

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

## Backend recipes

These all consume OTLP — the only thing that differs is the endpoint and
auth header.

### Jaeger (>=1.50)

Jaeger natively accepts OTLP/HTTP on port 4318:

```python
OTLPSpanExporter(endpoint="http://jaeger:4318/v1/traces")
```

### Grafana Tempo

Send to your collector, or directly to Tempo's OTLP endpoint:

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

### Datadog (via the OTel collector)

Configure the collector with the Datadog exporter, then ship to it:

```python
OTLPSpanExporter(endpoint="http://otel-collector:4318/v1/traces")
```

Datadog also accepts native OTLP HTTP directly — same shape, different URL.

### AWS X-Ray (via collector)

The OTel collector includes the AWS X-Ray exporter; treat it like any other
OTLP target.

## Continuing an upstream trace

`Tracer.attach(agent)` currently roots every agent run in its own trace —
there is no public API yet for passing an inbound `traceparent` so that
spans nest under a caller's HTTP handler trace. The internal hook
(`Tracer._make_parent_context`) exists for a future `run_scope` feature;
until that ships, agent runs and the surrounding service trace are linked
only by their resource attributes (`service.name`, `gen_ai.agent.name`,
`cubeloop.run_id`).

If you need the upstream trace to continue into CubeLoop today, the
workaround is to set the OTel current span yourself before calling
`agent.prompt(...)` and let the agent's spans inherit it via OTel's
ambient context — CubeLoop never overrides an active parent.

On the way out, MCP `tools/call` does inject W3C `traceparent` as an HTTP
header automatically, so an instrumented MCP server downstream of the
agent can continue the trace through to its own backend.

## Combining exporters

You can pass multiple exporters and they'll receive every span. Common
pattern — JSONL for local debugging plus OTLP for the production backend:

```python
tracer = Tracer(
    service_name="my-bot",
    exporters=[
        JsonlSpanExporter(directory="./cubeloop-traces"),
        OTLPSpanExporter(endpoint="https://api.honeycomb.io/v1/traces", headers={…}),
    ],
)
```

## Flushing

`Tracer` uses `BatchSpanProcessor` under the hood, so spans are exported in
the background. The recommended pattern — and the one that exercises all
the cleanup paths automatically — is the `async with` form:

```python
async with Tracer(...) as tracer, tracer.attached(agent):
    await agent.prompt("...")
# On exit:
#   - detach() runs synchronously: closes any spans a cancelled run left
#     open, then schedules the flush as an asyncio.Task and awaits it.
#   - tracer.shutdown() flushes again (idempotent) and closes exporters.
```

If you've stored `detach` from a manual `tracer.attach(agent)` call, it
returns the scheduled flush `Task`. Two valid manual patterns:

```python
# (a) Belt-and-braces — safest, both run.
finally:
    detach()                    # closes cancelled-run spans, schedules flush
    await tracer.shutdown()     # awaits force_flush, then shuts the SDK down

# (b) Awaited detach — single call when you're not also shutting down.
finally:
    await detach()              # awaits the scheduled flush
```

Outside an async context (no running loop) `detach()` returns `None` — the
sync part has run, but the flush is the caller's responsibility via
`await tracer.shutdown()`.

If even that gets missed — say a script raises before reaching `finally` —
`Tracer(atexit_flush=True)` (the default) registers a process-exit handler
that sync-flushes buffered spans through `BatchSpanProcessor`. Doesn't run
on `SIGKILL` / `os._exit`; for guaranteed delivery there, use
`SimpleSpanProcessor` (sync export per span).

### Background flush on serving paths

On request-handling paths — FastAPI endpoints, gRPC handlers, any code where
a caller is waiting on the `async with` block — awaiting the OTLP flush before
returning ties your p99 to the slowest exporter. Use the standalone `trace()`
helper with `flush="background"` instead:

```python
from cubeloop.tracing import trace

@app.post("/chat")
async def chat(req: ChatRequest):
    async with trace(tracer, agent, flush="background"):
        reply = await agent.prompt(req.message)
    # Block exits immediately — span export runs as a supervised background
    # task, decoupled from this response.
    return {"reply": reply}
```

Both `force_flush` and the underlying `BatchSpanProcessor` export run in a
worker thread (off the event loop), so a slow or backlogged collector can't
stall a user-visible response. The tracer keeps a strong reference to each
in-flight flush task so it can't be garbage-collected mid-export; `await
tracer.shutdown()` settles any still-pending flushes before closing exporters,
so no spans are lost on clean shutdown.
