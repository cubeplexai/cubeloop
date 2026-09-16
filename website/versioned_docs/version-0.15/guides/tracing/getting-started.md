---
title: Getting Started
description: "Get started with OpenTelemetry tracing in CubeLoop — install, configure, and export spans."
sidebar_position: 2
---

# Getting Started with Tracing

## Install the extra

CubeLoop keeps OpenTelemetry an optional dependency:

```bash
pip install "cubeloop[tracing]"
```

This pulls in `opentelemetry-sdk` and friends. Without the extra, the
`cubeloop.tracing` import raises a clear error so you find out at import time
rather than mid-run.

## Attach a Tracer

The minimal end-to-end setup — local JSONL export, idiomatic RAII:

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
    # On exit: auto-detach (closes any cancelled-run spans, awaits the
    # flush) + tracer shutdown (flushes + closes exporters). No
    # try/finally needed.


asyncio.run(main())
```

If you can't restructure into an `async with` (e.g. long-lived web
handler that hands the agent around), the explicit pattern still
works and is fully equivalent:

```python
detach = tracer.attach(agent)
try:
    await agent.prompt("…")
finally:
    # Either is enough on its own:
    #   await detach()                # awaits the scheduled flush
    #   await tracer.shutdown()        # flushes + closes exporters
    detach()
    await tracer.shutdown()
```

Even if you forget the cleanup entirely, `Tracer` registers an
`atexit` hook by default that sync-flushes buffered spans at process
exit — pass `atexit_flush=False` to opt out, or rely on it as a
safety net while you're still building. (Doesn't run on `SIGKILL` or
`os._exit`; for guaranteed delivery there, use the synchronous
`SimpleSpanProcessor` from OTel.)

The run produces one JSONL file per trace (sharded by `trace_id`):

```text
./cubeloop-traces/
  2026-05-19/
    8e1c9a3f4b2d…d976a.jsonl   ← one trace, one file, one span per line
```

A trace is the whole OTel tree, including any nested subagent runs (they
inherit the parent's `trace_id`, so they land in the same file). Each span
also carries `cubeloop.run_id` — the **same** id as `agent.prompt(run_id=…)` /
`Message.run_id` (hosts that pass a run id will see it here; otherwise the
agent-minted id). Use it to filter by individual business run inside a
trace that may hold several nested activations.

Open it with any tool that reads OTLP/JSON or with `jq` directly:

```bash
jq -r '"\(.name)  \(.attributes."gen_ai.operation.name" // "")"' \
   cubeloop-traces/2026-05-19/*.jsonl
# invoke_agent  invoke_agent
# cubeloop.turn
# chat claude-sonnet-4-6  chat
```

## Span hierarchy

For a single prompt with one LLM round-trip, the recorder produces three spans:

```text
invoke_agent assistant              [INTERNAL]   gen_ai.operation.name=invoke_agent
└── cubeloop.turn                     [INTERNAL]   cubeloop.turn.index=0
    └── chat <model>                [CLIENT]     gen_ai.operation.name=chat
```

When the model calls a tool, you get an extra layer per tool:

```text
invoke_agent assistant
└── cubeloop.turn                     ← turn index 0
    ├── chat <model>                ← first round trip
    └── execute_tool <tool_name>    ← gen_ai.tool.name, gen_ai.tool.call.id
└── cubeloop.turn                     ← turn index 1 (response after tool result)
    └── chat <model>
```

For MCP tools the `execute_tool` span gets a CLIENT child:

```text
execute_tool <tool_name>            [INTERNAL]   cubeloop-side wrapper
└── tools/call <tool_name>          [CLIENT]     gen_ai.operation.name=execute_tool
                                                  mcp.method.name=tools/call
                                                  mcp.session.id=…
                                                  server.address / server.port
```

The CLIENT span injects W3C `traceparent` into outgoing HTTP headers, so an
instrumented MCP server can continue the trace.

## Cancellation, errors, aborts

The recorder treats cancellation as a control signal, not a failure:

- `agent.abort()` or a cancelled one-shot call → affected spans close with
  `cubeloop.aborted=true` and `error.type=cubeloop.aborted`, **status UNSET**, and no
  exception event (per OTel guidance — cancellation isn't an error).
- `agent.detach()` at a durable HITL prompt → open spans close with
  `cubeloop.run.outcome=suspended` and no aborted/error classification. A later
  `respond()` is a new activation trace; correlate the two with metadata such as
  a host run or conversation ID.
- A provider or one-shot call raising → affected spans close with **status
  ERROR** and `error.type` derived from the exception class (`timeout`,
  `connection_error`, fully-qualified class name, …). With the default
  `record_content=False`, status descriptions are generic and exception events
  contain only `exception.type`; raw exception messages and stack traces are
  included only when content recording is explicitly enabled.
- An MCP transport raising follows the same privacy rule on its CLIENT span. An
  MCP `tools/call` returning `isError=true` closes the CLIENT span with ERROR +
  `error.type=mcp.is_error`; its status description is also generic unless
  content recording is enabled.

In every case, `detach()` and `tracer.shutdown()` close any span the activation
left open, so cancelled and suspended runs remain visible in your backend rather
than silently disappearing.

## What's on each span

Defaults (no opt-in needed):

- `invoke_agent` (root) — `gen_ai.operation.name`, `gen_ai.provider.name`,
  `gen_ai.agent.name`, `cubeloop.run_id` (business run id from
  `prompt`/`resume`/`respond`, not a separate tracer uuid),
  `cubeloop.agent.system_prompt.sha256`, `cubeloop.agent.tools` (names list),
  `cubeloop.input_messages.count`, `cubeloop.output_messages.count`
- `cubeloop.turn` — `cubeloop.turn.index`, `cubeloop.turn.stop_reason`,
  `cubeloop.turn.tool_calls.count`, `cubeloop.turn.terminated_by_tool`,
  `cubeloop.run_id`
- `chat <model>` — `gen_ai.operation.name`, `gen_ai.provider.name`,
  `gen_ai.request.model`, `gen_ai.request.max_tokens` / `temperature` /
  `top_p`, `gen_ai.request.stream`, `gen_ai.usage.input_tokens` /
  `output_tokens` / `cache_read_input_tokens` / `cache_creation_input_tokens` /
  `reasoning_output_tokens`, `gen_ai.response.model` /
  `finish_reasons` / `id`, `gen_ai.response.time_to_first_chunk`, plus
  OpenAI-specific extras (`openai.api.type`, service tier, system fingerprint)
- `execute_tool <tool_name>` — `gen_ai.operation.name=execute_tool`,
  `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.description`,
  `gen_ai.tool.type`, `cubeloop.tool.is_error`, `cubeloop.tool.execution_mode`
- `tools/call <tool_name>` (MCP only) — `mcp.method.name`, `mcp.session.id`,
  `mcp.protocol.version`, `server.address`, `server.port`, `gen_ai.tool.name`

Optional, opt-in via `Tracer(record_content=True)`:
`gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`,
`gen_ai.tool.definitions`, `gen_ai.tool.call.arguments`,
`gen_ai.tool.call.result`, `cubeloop.llm.raw_request`,
`cubeloop.llm.raw_response`, provider/MCP/one-shot exception messages and stack
traces, and stream-log tool-argument/error previews. See
[Content & Redaction](./content-recording).

## Multiple agents, one process

Both `Tracer` and `Meter` are fine to share across agents — call
`attach(agent)` multiple times. Each attach gets its own recorder /
metric state so concurrent agents don't share span or histogram state,
and MCP CLIENT spans route through the right Tracer based on which
agent's `execute_tool` span is the parent.

With the RAII helper, stacking them is one `async with`:

```python
async with (
    Tracer(...) as tracer,
    tracer.attached(agent_a),
    tracer.attached(agent_b),
):
    await asyncio.gather(agent_a.prompt("…"), agent_b.prompt("…"))
```

## Tagging individual runs

`cubeloop.tracing.tracing_context` scopes per-run tags / metadata onto
the `invoke_agent` span — perfect for `user_id`, `session_id`,
A/B-test arm, anything you'd want to filter by in the backend later:

```python
from cubeloop.tracing import tracing_context

async with tracer.attached(agent):
    with tracing_context(tags=["beta-arm"], metadata={"user_id": "u-42"}):
        await agent.prompt("Hello.")
```

Attributes on the span:

- `cubeloop.tags = ("beta-arm",)`
- `cubeloop.metadata.user_id = "u-42"`

The `cubeloop.metadata.*` prefix keeps user keys from clobbering
recorder-owned schema (e.g. `cubeloop.run_id`). Tags and metadata
contextvars are per-asyncio-task, so concurrent agents see
independent values, and nested `tracing_context` blocks merge
(tags concatenate, metadata keys union with inner winning).

## Tracing background LLM calls (`oneshot`)

`attach()` instruments a cubeloop `Agent`. For background tasks that call an LLM
directly — without a full agent loop (no tool use, no multi-turn) — use
`Tracer.oneshot()` instead. It produces the same `invoke_agent` root span and
`chat` child span so the `cubeloop trace` CLI indexes it alongside normal agent
runs.

```python
async with tracer.oneshot(
        model=model,
    operation="consolidate_memory",          # labelled in the trace
    metadata={"conversation_id": conv_id},   # queryable via --meta
) as session:
    text = await session.generate(
        system=SYSTEM_PROMPT,
        messages=[UserMessage(content=[TextContent(text=prompt)])],
        max_output_tokens=1500,
    )
```

The span tree is flat (no `cubeloop.turn` wrapper — there is no loop):

```
invoke_agent  820ms
  └── chat deepseek-v3  815ms  tok 3200/180
```

Filter these traces by operation name in the CLI:

```bash
cubeloop trace ls --meta oneshot_operation=consolidate_memory
cubeloop trace ls --meta conversation_id=conv-123   # alongside the conversation's agent runs
```

The `operation` string is recorded as both `cubeloop.oneshot.operation` (for
dashboards) and `cubeloop.metadata.oneshot_operation` (so `--meta` can reach it,
since the CLI filter only reads `cubeloop.metadata.*` attributes).

## Next

- [OTLP & Backends](./otlp) — Jaeger, Tempo, Honeycomb, Datadog, …
- [Content Recording & Redaction](./content-recording)
- [Metrics](./metrics)
