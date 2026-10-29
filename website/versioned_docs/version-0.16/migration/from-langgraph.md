---
title: From langgraph
description: "Learn how to migrate your agents from LangGraph to CubeLoop — concept mapping, side-by-side code comparisons, and a step-by-step porting guide for the Pythonic async-native alternative."
---

# Migrating from langgraph

CubeLoop and [langgraph](https://github.com/langchain-ai/langgraph) both
build tool-using LLM agents, but they have different mental models.
This page maps langgraph concepts onto CubeLoop so you can port code
without having to re-learn from scratch.

## Mental-model shift

| langgraph | CubeLoop | Why |
|---|---|---|
| **State graph** with nodes, edges, channels | **Agent loop** that's a plain `while` loop you can read | A linear loop is easier to reason about than a graph; CubeLoop never branches at runtime — control flow lives in middleware |
| **Channels** (typed state slots) | **`AgentContext.extra`** + `AgentState.messages` | A single dict + a single message list cover every state shape we've seen |
| **`StateGraph.add_node(name, fn)`** | A middleware hook or a tool | Functions in langgraph nodes split into two roles in CubeLoop: tool execution (when the model decides) vs. middleware (always-on transforms) |
| **`add_edge(a, b)`** / `add_conditional_edges` | Built-in: tools → next turn → tools → … | The conditional shape (tool calls → re-prompt) is the loop; you don't reify it |
| **`MemorySaver` / `SqliteSaver` / `PostgresSaver`** | `MemoryCheckpointer` / `SQLiteCheckpointer` / `PostgresCheckpointer` | Same idea, append-only schema instead of full snapshots |
| **`config: {"configurable": {"thread_id": …}}`** | `Agent(thread_id=…)` | First-class agent parameter |
| **`stream_mode="messages"` / `"values"` / `"updates"`** | `agent.subscribe(listener)` — one event stream | One pattern, eleven event types |
| **Tools as `@tool` decorated functions** | `AgentTool` with Pydantic params + async execute | Closer to OpenAI/Anthropic native shape |
| **`HumanMessage`, `AIMessage`** | `UserMessage`, `AssistantMessage` | Same role-tagged messages, just renamed |
| **Interrupts via `interrupt_before` / `interrupt_after`** | `agent.steer(...)`, `agent.follow_up(...)`, `agent.abort()` | Imperative control instead of declarative interrupt points |
| **Time travel / fork** at any checkpoint | **`Agent.fork()`** / **`Agent.fork_once()`** at run boundaries | CubeLoop keys off completed `run_id`s; see [Forking](../guides/agents/forking) |
| **`config_schema`** | Constructor parameters on `Agent` | No separate schema layer |

## Side-by-side: a tool-using agent

### langgraph

```python
from typing import TypedDict
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"72°F and sunny in {city}"


llm = ChatAnthropic(model="claude-sonnet-4-6").bind_tools([get_weather])

class State(TypedDict):
    messages: list

def call_model(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

def should_continue(state: State):
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END

graph = StateGraph(State)
graph.add_node("llm", call_model)
graph.add_node("tools", ToolNode([get_weather]))
graph.add_edge("__start__", "llm")
graph.add_conditional_edges("llm", should_continue)
graph.add_edge("tools", "llm")
app = graph.compile()

for chunk in app.stream({"messages": [("user", "Weather in Tokyo?")]}):
    print(chunk)
```

### CubeLoop

```python
import asyncio
from cubeloop import Agent, tool
from cubeloop.providers.anthropic import AnthropicProvider


@tool
async def get_weather(city: str) -> str:
    "Get current weather for a city."
    return f"72°F and sunny in {city}"


provider = AnthropicProvider(provider_id="anthropic", api_key="…")
agent = Agent(
    model=provider.model("claude-sonnet-4-6"),
    tools=[get_weather],
)
agent.subscribe(lambda e, s=None: print(e.type))
asyncio.run(agent.prompt("Weather in Tokyo?"))
```

The `@tool` decorator mirrors langgraph's `@tool`: the input schema comes
from the function signature, the docstring becomes the description, and a
plain `str` return is wrapped for you. (For tools that need a shared params
model or dynamic construction, the longhand `AgentTool(...)` is still there —
see [Tool Use](../guides/agents/tool-use).)

CubeLoop version removes:

- The `StateGraph`, edges, nodes, `END` sentinel, conditional edges.
- The `ToolNode` registry — tools go directly to the `Agent`.
- The `should_continue` function — the loop knows when there are tool
  calls.
- The `State` TypedDict — state lives on the agent.

## Mapping common patterns

### Checkpointing

```python
# langgraph
from langgraph.checkpoint.sqlite import SqliteSaver
graph.compile(checkpointer=SqliteSaver.from_conn_string(":memory:"))

# CubeLoop
from cubeloop.checkpointer import SQLiteCheckpointer
async with SQLiteCheckpointer("agent.db") as cp:
    agent = Agent(..., checkpointer=cp, thread_id="conv-1")
```

CubeLoop's append-only model is O(1) per message, regardless of
conversation length. langgraph saves full snapshots, which scales
linearly with history.

### Streaming

```python
# langgraph
for chunk in app.stream(state, stream_mode="messages"):
    if chunk["event"] == "on_chat_model_stream":
        print(chunk["data"]["chunk"].content, end="")

# CubeLoop
def on_event(event, signal=None):
    if event.type == "message_update" and event.stream_event.type == "text_delta":
        print(event.stream_event.delta, end="")

agent.subscribe(on_event)
await agent.prompt("…")
```

One subscriber, one stream — no mode flag.

### Interrupting / human-in-the-loop

```python
# langgraph
graph.compile(interrupt_before=["tools"])

# CubeLoop
class HumanApproval(Middleware):
    async def before_tool_call(self, ctx, *, signal=None):
        approved = await ask_human(f"Run {ctx.tool_call.name}({ctx.args})?")
        if not approved:
            return BeforeToolCallResult(block=True, reason="rejected")
        return None
```

Imperative interrupts via middleware. You decide per call instead of
configuring graph-level interrupt points.

### Branching

```python
# langgraph
graph.add_conditional_edges("llm", lambda s: "tools" if s["messages"][-1].tool_calls else "summary")
graph.add_node("summary", summarize)
graph.add_edge("summary", END)
```

```python
# CubeLoop
class SummariseAtEnd(Middleware):
    async def should_stop_after_turn(self, ctx) -> bool:
        msg = ctx.message
        if not any(isinstance(c, ToolCall) for c in msg.content):
            # No more tool calls; we're done. Inject a summary turn first.
            ...
            return True
        return False
```

There's no built-in branching primitive; flow control happens through
`should_stop_after_turn` and `after_model_response`.

### Forking

```python
# langgraph — time travel to a saved checkpoint
config = {"configurable": {"thread_id": "t1", "checkpoint_id": "<id>"}}
app.update_state(config, {"messages": [...]})
result = app.invoke(None, config)

# CubeLoop — persistent fork at a completed-run boundary
await agent.fork(
    src_thread_id="conv_123",
    new_thread_id="conv_456",
    after_run_id="R1",
)

# CubeLoop — ephemeral one-shot probe (writes nothing)
result = await agent.fork_once(
    src_thread_id="conv_123",
    message="What if you had said yes?",
    after_run_id="R1",
)
print(result.text)
```

CubeLoop forks at completed-run boundaries rather than arbitrary mid-run
checkpoints. `fork` creates a persistent branch you can continue;
`fork_once` runs a single probe and discards everything. See
[Conversation Forking](../guides/agents/forking).

## What LangGraph still offers beyond CubeLoop

- **Graph-native multi-agent orchestration.** CubeLoop includes
  `SubagentMiddleware` for tool-driven delegation: a parent agent can dispatch
  self-contained tasks to typed child agents with dedicated prompts, models,
  tools, and middleware. What CubeLoop does not yet provide is LangGraph's
  explicit, durable graph model for supervisor routing, conditional branches,
  parallel fan-out/fan-in, and graph-level state management.
- **Visual graph rendering.** No `app.get_graph().draw_mermaid()`
  equivalent. CubeLoop's flow is linear so the picture would be a single
  line anyway.
- **First-party UI for traces.** CubeLoop doesn't render its own trace
  visualizer the way LangSmith / Langfuse do; instead it emits
  vendor-neutral OpenTelemetry — point any OTLP backend
  (LangSmith's OTel endpoint, Langfuse v3, Jaeger, Tempo,
  Honeycomb, Datadog, …) at it via
  `Tracer(exporters=[OTLPSpanExporter(...)])`. See
  [Tracing → OTLP & Backends](../guides/tracing/otlp).

## What CubeLoop does that langgraph doesn't

- **Native OpenTelemetry tracing** — `Tracer` + `Meter` emit OTel
  spans + GenAI-semconv attributes out of the box, ingestible by
  any OTLP backend. See [Tracing → Overview](../guides/tracing/overview).
- **Native async-first** — every entry point is async. No
  `app.invoke` vs. `app.ainvoke` split.
- **Append-only persistence** — O(1) DB writes, JSONB-queryable
  messages.
- **3 core deps** vs. langchain-core + langgraph-sdk + transitives.
- **Streaming-realistic test provider** (`FauxProvider`) ships in the
  box.
- **MCP loaders** for HTTP + stdio transports.

## Porting checklist

1. Replace `StateGraph` construction with a single `Agent(...)` call.
2. Move `@tool`-decorated functions to `AgentTool` instances (Pydantic
   models for params, async execute).
3. Replace `MemorySaver` / `SqliteSaver` / `PostgresSaver` with
   `MemoryCheckpointer` / `SQLiteCheckpointer` / `PostgresCheckpointer`.
4. Replace `stream_mode` callbacks with `agent.subscribe(...)`.
5. Convert custom nodes that did message transforms → `Middleware`
   hooks.
6. Convert `interrupt_before/after` → `before_tool_call` /
   `after_model_response` middleware.
7. If you had a `summary` or `route` node — fold it into
   `after_model_response` with `decision="stop"` or `"loop_to_model"`.

## See also

- [Core Concepts](../getting-started/core-concepts) — the building
  blocks you're mapping to.
- [Middleware → Composition](../guides/middleware/composition) — where
  flow-control logic lives.
- [Checkpointing](../guides/checkpointing/sqlite) — the new
  persistence story.
