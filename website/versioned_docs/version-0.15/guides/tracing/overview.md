---
title: Tracing Overview
description: "Overview of CubeLoop's OpenTelemetry-native tracing system — spans, attributes, exporters, and semantic conventions."
sidebar_position: 1
---

# Tracing Overview

CubeLoop emits [OpenTelemetry](https://opentelemetry.io/) spans that follow the
[GenAI Semantic Conventions v1.41](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
so any OTel-compatible backend (Jaeger, Tempo, Honeycomb, Datadog, AWS X-Ray,
Azure Monitor, …) can ingest agent runs without custom instrumentation.

Attach a `Tracer` to an `Agent` and every prompt produces a tree of spans you
can pivot, query, and join with the rest of your service traces:

```
trace
└── invoke_agent  14425.8ms  [0x1cd97cdb]         ← one per agent.prompt()
    ├── cubeloop.turn  1283.1ms  [0x5cfda93e]        ← one per LLM round-trip
    │   ├── chat deepseek-v4-flash  1208.7ms  tok 6845/68  [0x0d130229]
    │   └── execute_tool subagent  9610.2ms  subagent  [0x38bdd10a]
    │       └── invoke_agent  9601.0ms  [0x8094f99b]   ← subagent run, nested
    │           └── cubeloop.turn  9598.4ms  [0x57c5cfc7]
    │               ├── chat deepseek-v4-flash  1190.3ms  [0x8205ca6b]
    │               └── execute_tool web_search  6500.2ms  web_search  [0xca4e59fc]
    └── cubeloop.turn  491.9ms  ERROR  [0xce25f242]
        └── chat deepseek-v4-flash  427.2ms  ERROR  [0x0bff68ec]
            └── error: Error code: 400 - ... `tool_use` ids were found without
                `tool_result` blocks immediately after: call_01_...
```

Each layer carries standard `gen_ai.*` attributes — `gen_ai.operation.name`,
`gen_ai.request.model`, `gen_ai.provider.name`, `gen_ai.usage.input_tokens`,
`gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons`, …

## What ships out of the box

- **Tracer** — builds an SDK `TracerProvider`, attaches one
  `BatchSpanProcessor` per exporter, wires the CubeLoop event stream into spans.
- **Meter** — sibling for OTel histograms:
  `gen_ai.client.operation.duration`, `gen_ai.client.operation.time_to_first_chunk`,
  `gen_ai.client.token.usage`.
- **JsonlSpanExporter** — write one JSON line per span to
  `./cubeloop-traces/<date>/<trace_id>.jsonl`. Files are sharded by `trace_id`,
  so one file holds a whole trace — the run plus any nested subagent runs
  (which inherit the trace). Useful for local dev and offline debugging; works
  with any OTel viewer that reads JSONL, and with the [`cubeloop trace`
  CLI](./cli).
- **OTLP** — bring your own exporter via `opentelemetry-exporter-otlp-proto-http`
  (HTTP) or `…-grpc` and hand it to `Tracer(exporters=[…])`.
- **W3C trace context propagation** — outgoing MCP calls inject the active
  `traceparent` as an HTTP header so an instrumented MCP server can continue
  the trace.
- **`tracer.attached(agent)` / `meter.attached(agent)`** — async context
  managers that RAII-wrap attach/detach, so cleanup is one `async with`
  block instead of an explicit `try/finally`.
- **`atexit` flush hook** — `Tracer(atexit_flush=True)` (default) registers
  a process-exit handler that sync-flushes any buffered spans, so callers
  who forget `await tracer.shutdown()` still get their spans exported on
  normal exit / Ctrl-C / unhandled exception.
- **`tracing_context()`** — set per-run tags and metadata
  (`cubeloop.tags = ("beta-arm",)`, `cubeloop.metadata.user_id = "u-42"`)
  via a contextvar-scoped block. Concurrent agents each see their own
  values.
- **Middleware-owned providers traced automatically** — middleware that
  exposes extra `BaseProvider` instances via `Middleware.providers()` has
  its listener registry wired by `Recorder.attach()` so those providers'
  `chat` spans land in the same trace as the agent's main call.
  `CompactionMiddleware` uses this to surface its summarizer LLM call as
  a `chat <summary-model>` span nested under
  `cubeloop.compaction.summarize` — see the [compaction
  guide](../middleware/compaction#tracing).

## What it costs

- One pure-Python recorder per agent run subscribing to the agent's event
  stream and the provider's listener registry — no monkey-patching, no extra
  threads.
- One OTel SDK span per layer above. `BatchSpanProcessor` batches export off
  the hot path.
- **No payloads are recorded by default.** `gen_ai.input.messages`,
  `gen_ai.output.messages`, raw request/response, and tool args/results all
  require explicit opt-in via `record_content=True` so you don't accidentally
  ship PII to your backend. See [Content & Redaction](./content-recording).

## When to use each piece

| You want | Use |
|---|---|
| Trace one local agent run and inspect a JSONL file | `Tracer` + `JsonlSpanExporter` |
| Ship to Jaeger / Tempo / Honeycomb / Datadog | `Tracer` + OTLP exporter |
| Latency + token histograms next to the spans | `Meter` alongside `Tracer` |
| Record prompts / model outputs for evaluation | `Tracer(record_content=True)` |
| Redact PII before it leaves the process | `Tracer(redact=…)` |
| Tag runs with `user_id` / `session_id` / A-B arm | `tracing_context(tags=…, metadata=…)` |
| One-liner cleanup, no try/finally | `async with tracer.attached(agent): …` |
| Forget to call `shutdown()` and not lose spans | `Tracer(atexit_flush=True)` (default) |
| Continue a trace from an upstream service | `Tracer(resource=…)` + W3C `traceparent` (auto for MCP, manual for HTTP) |

## Where to go next

- [Getting Started](./getting-started) — install the extra and emit your
  first spans
- [OTLP & Backends](./otlp) — point CubeLoop at Jaeger, Tempo, Honeycomb, …
- [Content Recording & Redaction](./content-recording) — record prompts and
  responses safely
- [Metrics](./metrics) — histograms via `Meter`
