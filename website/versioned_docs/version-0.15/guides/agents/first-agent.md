---
title: Building Your First Agent
description: "Create and run your first CubeLoop agent — choose a provider, define a system prompt, and send a message."
---

# Building Your First Agent

This guide is the longer cousin of the [Quick Start](../../getting-started/quick-start).
We'll build a single-tool agent end-to-end, then layer on the things
you'll want next: streaming UI, error handling, and a cancel button.

## Step 1 — set up the provider and model

The provider is the connection to the LLM API; `provider.model("id", ...)`
binds a model id to that provider and produces the value `Agent(model=...)`
expects.

```python
import os
from cubeloop.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(provider_id="anthropic", api_key=os.environ["ANTHROPIC_API_KEY"])
model = provider.model(
    "claude-sonnet-4-6",
    max_tokens=4096,         # response cap
    context_window=200_000,  # hard model limit; defaults are usually fine
    temperature=0.7,
)
```

`provider_id` lives on the provider constructor and is propagated into the
bound model automatically — used by the framework to gate reasoning fields
and tag responses. Building a `Model` by hand and passing both
`provider=` and `model=` to `Agent` is the 0.6 idiom and no longer works
in 0.7.

## Step 2 — declare a tool

A tool is an async function decorated with `@tool`:

```python
from cubeloop import tool


@tool
async def get_weather(city: str) -> str:
    "Get current weather for a city. Returns a short text summary."
    # do real work here — call an HTTP API, query a DB, etc.
    return f"72°F and sunny in {city}"
```

A few details:

- The input schema is generated from the typed parameters and sent to
  the model; the docstring becomes the tool description. Pydantic
  `Field(...)` defaults and metadata are honoured.
- Return a `str` (wrapped as text), a `Content`, a `list` of content,
  or a full `AgentToolResult` when you need `details`/`is_error`.
- Need cancellation or progress streaming? Declare `signal` (an
  `asyncio.Event` set when the user cancels) and/or `on_update(partial)`
  in the signature and CubeLoop injects them — see
  [Tool Use](./tool-use#streaming-tool-progress).
- For a shared params model or dynamic construction, the longhand
  `AgentTool(...)` is equivalent — see [Tool Use](./tool-use).

## Step 3 — assemble the agent

```python
from cubeloop import Agent

agent = Agent(
    model=model,
    system_prompt="You are a concise weather assistant.",
    tools=[get_weather],
)
```

You can pass `tools=[]` (or omit it) for a plain chat agent.

## Step 4 — subscribe to events

`agent.subscribe(listener)` is how you observe the run. The listener
receives every `AgentEvent`:

```python
def on_event(event, signal=None):
    if event.type == "message_update" and event.stream_event.type == "text_delta":
        print(event.stream_event.delta, end="", flush=True)
    elif event.type == "tool_execution_start":
        print(f"\n→ calling {event.tool_name}({event.args})")
    elif event.type == "tool_execution_end":
        print(f"  ✓ done")

agent.subscribe(on_event)
```

You can register multiple listeners and they all receive every event.
Subscribe **before** `prompt()` — events fire as soon as the run
starts.

## Step 5 — prompt and run

```python
import asyncio

async def main():
    await agent.prompt("What's the weather in Tokyo?")

asyncio.run(main())
```

`agent.prompt()` does not return any value. The result lives on
`agent.state.messages` (the full history) and `agent.state.streaming_message`
(the current in-flight message, or `None` between turns).

## Adding error handling

When `provider.stream()` raises, the agent loop still produces an
`AssistantMessage` with `stop_reason="error"` and `error_message`
filled in. The event sequence is:
`message_start` → `message_end` → `turn_end` → `agent_end`.

You can either:

1. Catch in the subscriber, looking for `event.type == "agent_end"` and
   the last message's `stop_reason`:

    ```python
    def on_event(event, signal=None):
        if event.type == "agent_end":
            last = event.messages[-1]
            if getattr(last, "stop_reason", "") == "error":
                print(f"\nerror: {last.error_message}")
    ```

2. Or inspect `agent.state.error_message` after `await
   agent.prompt(...)` returns.

## Adding a cancel button

`agent.abort()` sets the run-level signal. The provider stream
short-circuits to `"aborted"`, in-flight tools see `signal.is_set()
== True`, and the loop emits `agent_end` cleanly.

```python
async def main():
    task = asyncio.create_task(agent.prompt("Search for…"))
    await asyncio.sleep(0.5)
    agent.abort()
    await task              # always completes — never raises
    await agent.wait_for_idle()
```

## Common pitfalls

- **`RuntimeError: Agent is already processing a prompt.`** — You called
  `prompt()` twice without awaiting the first. Use `await
  agent.wait_for_idle()` or queue with `steer()` / `follow_up()`
  instead.
- **No `text_delta` events** — Did you subscribe *before* calling
  `prompt()`? Listeners only see events emitted after registration.
- **Tool not found** — The model invoked a tool whose `name` doesn't
  match any tool in `tools=[...]`. CubeLoop reports this as a tool result
  with `is_error=True` rather than crashing — check the
  `tool_execution_end` event's `result`.
- **Pydantic ValidationError swallowed** — If the model produces
  malformed JSON, CubeLoop captures the validation error and feeds it
  back as a tool error result. The model usually corrects itself on
  the next turn.

## Next

- [Tool Use & Parallel Execution](./tool-use) — multiple tools at once,
  sequential mode, `terminate`, incremental progress.
- [Streaming Events](./streaming) — the full event taxonomy.
- [Multi-turn Conversations](./multi-turn) — keeping state across
  turns, `steer`, `follow_up`, `resume`.
