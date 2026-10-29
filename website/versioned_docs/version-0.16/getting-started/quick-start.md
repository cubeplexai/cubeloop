---
title: Quick Start
description: "Build your first CubeLoop agent in minutes with this step-by-step quick-start guide."
---

# Quick Start

Ship a streaming, tool-using agent in five minutes. We'll build a
weather agent that calls a Python function as a tool, streams Claude's
response token-by-token, and exits cleanly.

## Prerequisites

- Python 3.11+
- `cubeloop` installed (`pip install cubeloop`)
- An `ANTHROPIC_API_KEY` in your environment

## The full script

Save this as `weather_agent.py`:

```python title="weather_agent.py"
import asyncio
import os

from cubeloop import Agent, tool
from cubeloop.providers.anthropic import AnthropicProvider


@tool
async def get_weather(city: str) -> str:
    "Get current weather for a city."
    # In a real app: call an HTTP weather API. Here we hard-code a reply.
    return f"72°F and sunny in {city}"


async def main():
    provider = AnthropicProvider(provider_id="anthropic", api_key=os.environ["ANTHROPIC_API_KEY"])

    agent = Agent(
        model=provider.model("claude-sonnet-4-6"),
        system_prompt="You are a concise weather assistant.",
        tools=[get_weather],
    )

    # Subscribe BEFORE prompt() — that's how you see streaming events.
    def on_event(event, signal=None):
        if event.type == "message_update" and event.stream_event.type == "text_delta":
            print(event.stream_event.delta, end="", flush=True)
        elif event.type == "agent_end":
            print()  # final newline

    agent.subscribe(on_event)
    await agent.prompt("What's the weather in Tokyo?")


asyncio.run(main())
```

Run it:

```bash
python weather_agent.py
```

You should see Claude stream a sentence like
*"The weather in Tokyo is currently 72°F and sunny…"*, with the
tool result threaded through.

## What just happened

CubeLoop ran a loop that looks (conceptually) like this:

1. `agent.prompt("What's the weather in Tokyo?")` enqueued a
   `UserMessage` and called the model.
2. The model decided to invoke `get_weather(city="Tokyo")` — CubeLoop
   parsed the JSON args against the schema `@tool` generated from your
   function signature, called your `async def`, and fed the result back
   as a `ToolResultMessage`.
3. The model produced a final assistant response, streamed back as
   `text_delta` events.
4. The loop emitted `agent_end` and returned.

`agent.subscribe(...)` registered a callback for every event the
runtime emits: `agent_start`, `turn_start`, `message_start`,
`text_delta`, `tool_execution_start`, `tool_execution_end`,
`message_end`, `turn_end`, `agent_end`. The script only cared about
`text_delta`, but you can hang any UI off the rest.

## Sources of confusion (read before you debug)

- **Subscribe before you prompt.** Listeners only receive events
  emitted *after* `subscribe` was called. Calling `prompt` first means
  the early events are gone.
- **`provider.model(...)` binds the model to its provider.** The provider holds credentials and optional `provider_id` metadata; the model id must match a model that provider supports.
- **Tools are decorated async functions.** `@tool` derives the input
  schema from the typed parameters and uses the docstring as the
  description; return a `str` (or an `AgentToolResult`). Need the raw
  `(tool_call_id, params, *, signal, on_update)` form, a shared params
  model, or `on_update` progress? See
  [Tool Use](../guides/agents/tool-use).
- **`agent.prompt()` can only run one prompt at a time.** While it's
  running, use [`agent.steer()`](../guides/agents/multi-turn) to
  inject a follow-up, or `agent.follow_up()` to queue one for after
  the current run.

## Next steps

- [Core Concepts](./core-concepts) — the mental model behind `Agent`,
  `Tool`, `Provider`, `Stream`, `Middleware`, `Checkpointer`.
- [Building Your First Agent](../guides/agents/first-agent) — full
  walkthrough of tools, streaming, and error handling.
- [Tool Use & Parallel Execution](../guides/agents/tool-use) — make
  the agent fan out across multiple tools at once.
- [Recipes → Weather Agent](../recipes/weather-agent) — a slightly
  beefier version of this script with real HTTP calls.
