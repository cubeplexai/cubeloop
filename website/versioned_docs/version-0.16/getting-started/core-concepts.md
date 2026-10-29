---
title: Core Concepts
description: "Learn the six core concepts of CubeLoop: Agent, Tool, Provider, Stream & Events, Middleware, and Checkpointer."
---

# Core Concepts

Six concepts cover everything CubeLoop does. Read this page once, then
the rest of the docs become a lookup table.

## Agent

`Agent` is the stateful façade: you construct it with a provider, a
model, optional tools, and optional middleware/checkpointer. You drive
ordinary applications through three methods:

- `await agent.prompt(message)` — start a new turn from a user message.
- `await agent.resume()` — continue from the last persisted message
  (used with a checkpointer).
- `agent.steer(message)` / `agent.follow_up(message)` — queue a
  message mid-flight or after the current run.

The agent owns an `AgentState` (system prompt, tools, model, message
history, pending tool calls, streaming flag) and a list of subscribers:

```python
unsubscribe = agent.subscribe(my_listener)
# ...
unsubscribe()
```

Subscribers receive every `AgentEvent` the loop emits. They can be
sync or async.

Hosts that need explicit attempt outcomes use the same engine through
`agent.session`:

```python
from cubeloop import PromptExecutionRequest

result = await agent.session.execute(
    PromptExecutionRequest(
        run_id="run-42",
        attempt_id="worker-attempt-1",
        message="Continue the analysis",
    )
)

if result.outcome == "suspended":
    render_form(result.pending_request)
elif result.outcome != "completed":
    record_failure(result.error)
```

An execution session belongs to one `Agent` and runs attempts sequentially.
It is not a distributed lock or a durable worker session. `ExecutionBusy`
rejects overlapping `execute()` calls; applications still own admission,
authorization, leases, and transport status.

## Tool

A tool is an async function the model can call. Decorate it with
`@tool` and CubeLoop generates the input schema from the parameters:

```python
from cubeloop import tool

@tool
async def search(query: str, limit: int = 10) -> str:
    "Search the corpus"
    # do work; declare `signal` / `on_update` in the signature if you need them
    return "…"
```

`@tool` builds an `AgentTool` (name + description + Pydantic parameter
model + async `execute`); the schema is auto-converted to JSON Schema
and passed to the model. Arg parsing, error wrapping, and parallel
execution are handled by the framework. See
[Tool Use](../guides/agents/tool-use) for the longhand `AgentTool(...)`,
`execution_mode`, `on_update` (incremental progress), and `terminate`
(end the turn from a tool).

## Provider

A `Provider` is anything matching this Protocol:

```python
class Provider(Protocol):
    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream: ...

    async def generate(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        reasoning: ReasoningControl | None = None,
    ) -> AssistantMessage: ...
```

`stream()` returns a `MessageStream` — a single async iterator that
yields `StreamEvent`s and exposes the final `AssistantMessage` via
`await stream.result()`. `generate()` consumes one stream and returns
the final message directly; `BaseProvider` implements it for any
provider that implements `stream()`. Built-in providers:

- `AnthropicProvider` — Claude (Messages API, with reasoning, caching,
  tool use).
- `OpenAIProvider` — GPT family (Chat Completions API).
- `OpenAIResponsesProvider` — GPT family (Responses API, server-side
  state).
- `FauxProvider` — deterministic test double (no network).

Write your own by subclassing `BaseProvider` and implementing
`stream()`. See [Providers / Custom](../guides/providers/custom).

## Stream and events

Streams and events are two layers:

- **Provider streams** — `MessageStream` yields *provider* events:
  `start`, `text_start`, `text_delta`, `text_end`, `thinking_*`,
  `toolcall_*`, `done`, `error`. This is the raw token stream.
- **Agent events** — what `agent.subscribe(...)` receives. 14
  types covering the entire loop + HITL: `agent_start`, `agent_end`,
  `turn_start`, `turn_end`, `message_start`, `message_update`,
  `message_end`, `tool_execution_start`, `tool_execution_update`,
  `tool_execution_end`, `hitl_request`, `hitl_answer`,
  `agent_suspended`, `agent_aborted`.

Subscribe to agent events for UI; for low-level token routing dig into
`event.stream_event`. See [Streaming Events](../guides/agents/streaming).

`agent.session.subscribe(...)` adds a host-facing layer. Every event is
wrapped with `run_id`, `attempt_id`, and an attempt-local sequence number.
The session also emits `input_committed` after an injected message has been
appended to the checkpointer, and exactly one `execution_finished` for an
accepted attempt. Mark a consumer `required=True` only when execution cannot
continue safely without it; observer failures never change the execution
outcome.

## Middleware

`Middleware` is a class with up to nine typed hooks:

| Hook | When it runs | Composition rule |
|---|---|---|
| `transform_context` | Before each model call, on the message list | Chained — each receives previous result |
| `convert_to_llm` | Right before serialisation to the provider | Last implementation wins |
| `transform_system_prompt` | Before each model call, on the system prompt | Chained |
| `resolve_tool_call` | Per tool call, before validation — may rewrite the call | First non-`None` wins |
| `before_tool_call` | Per tool call, after arg validation | First `block=True` short-circuits |
| `after_tool_call` | Per tool call, after `execute` | Later override earlier |
| `after_model_response` | After the assistant message lands | Returns a `TurnAction` controlling flow |
| `should_stop_after_turn` | At each turn boundary | Any `True` stops |
| `on_run_end` | Once after all turns complete, before `agent_end` | Messages concatenate; non-empty triggers one extra turn |

Pass middleware as a list to `Agent(middleware=[...])`. See
[Middleware → Composition](../guides/middleware/composition).

## Checkpointer

A `Checkpointer` is anything matching:

```python
class Checkpointer(Protocol):
    async def load(self, thread_id: str) -> CheckpointData | None: ...
    async def append(self, thread_id: str, messages: list[Message]) -> None: ...
    async def save_extra(self, thread_id: str, extra: dict) -> None: ...
```

Bind one to an agent with `Agent(checkpointer=cp, thread_id="…")` and
the loop will append each new message as it lands, restoring history
on the first `prompt()`. Built-in backends: `MemoryCheckpointer`,
`SQLiteCheckpointer`, `PostgresCheckpointer`, `MySQLCheckpointer`.

HITL adds two optional methods for cross-process suspend/resume:
`save_pending_request` / `load_pending_request`. All first-party
backends implement them. See [HITL guide](../guides/hitl/overview).

## HITL (Human-in-the-Loop)

CubeLoop ships a built-in `cubeloop.hitl` module for scenarios where the
agent needs to **pause and wait for a human**:

- **Sandbox confirmation** — a dangerous tool (bash, file write) needs
  approve / deny / edit before running.
- **Mid-run questions** — the agent surfaces a structured form to the user
  and waits for the answer.

```python
from cubeloop.hitl import InMemoryChannel, ConfirmToolCallMiddleware, ask_user_tool

channel = InMemoryChannel()

agent = Agent(
    model=…,
    tools=[bash_tool, ask_user_tool(channel)],
    middleware=[ConfirmToolCallMiddleware(channel, require_confirm={"bash"})],
    channel=channel,
)
```

The channel is an `await`-able coroutine collaborator: tool and
middleware authors write `await channel.ask(...)` or
`await channel.confirm(...)` and the channel handles the suspend. Host
code (your web app / TUI) subscribes to `channel.subscribe()` or polls
`channel.pending`, renders the request to the user, and posts the
answer via `channel.answer(qid, answer)`.

Two channel backends ship in-box:
- **`InMemoryChannel`** — single process (CLI, notebook, tests).
- **`CheckpointedChannel`** — cross-process (web service). The pending
  request is persisted to the checkpointer; a different process can
  answer hours later via `Agent.respond(question_id=, answer=)`.

Full details — the three HITL verbs, both built-in middlewares, the
cross-process suspend/resume protocol, events, trace spans, and error
reference — are in the [HITL guide](../guides/hitl/overview).

## Tracer (optional)

`Tracer` produces OpenTelemetry spans aligned with the
[GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/),
so any OTLP backend (Jaeger, Tempo, Honeycomb, Datadog, AWS X-Ray, …)
can ingest agent runs without custom instrumentation. Install the
extra:

```bash
pip install "cubeloop[tracing]"           # OTel SDK
pip install "cubeloop[tracing-otlp]"      # + OTLP/HTTP exporter
```

then wrap your agent in an `async with`:

```python
from cubeloop.tracing import Tracer
from cubeloop.tracing.exporters import JsonlSpanExporter

async with (
    Tracer(
        service_name="my-bot",
        agent_name="assistant",
        exporters=[JsonlSpanExporter(directory="./cubeloop-traces")],
    ) as tracer,
    tracer.attached(agent),
):
    await agent.prompt("…")
```

Each run emits an `invoke_agent` root span containing one
`cubeloop.turn` per LLM round-trip, plus `chat` (CLIENT) and
`execute_tool` children. By default **no prompt content or model
output is recorded** — opt in with `Tracer(record_content=True)` and
a `redact` callback for PII. Pair with `Meter(...)` for token /
duration / TTFC histograms. Full guide:
[Tracing → Overview](../guides/tracing/overview).

## Putting it together

```
User code
   │
   ▼
┌──────────────────────────────────────────┐
│ Agent                                     │
│  ├─ AgentState (messages, tools, …)       │
│  ├─ Middleware ── compose_middleware()    │
│  ├─ Checkpointer ── append on message_end │
│  └─ run_agent_loop  ◀──── the actual loop │
│       │                                   │
│       ▼                                   │
│  Provider.stream() → MessageStream        │
│       │                                   │
│       └─ events → emit → subscribers      │
└──────────────────────────────────────────┘
```

That diagram is the whole framework. The rest of this site is just
the details.
