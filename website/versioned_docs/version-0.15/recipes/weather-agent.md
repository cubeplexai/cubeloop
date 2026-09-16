---
title: Weather Agent
description: "Build a weather agent that fetches live API data with CubeLoop tools."
---

# Recipe: Weather Agent

A complete, runnable agent that calls a real weather API as a tool.
Demonstrates HTTP-calling tools, error handling, streaming UI, and
cancellation.

**Time to run:** 5 minutes.
**Deps:** `cubeloop`, `httpx`, an `ANTHROPIC_API_KEY`.

## The script

```python title="weather_agent.py"
import asyncio
import os

import httpx
from typing import Annotated
from pydantic import Field

from cubeloop import Agent, AgentToolResult, TextContent, tool
from cubeloop.providers.anthropic import AnthropicProvider


# --- The tool -----------------------------------------------------------

@tool
async def get_weather(
    city: Annotated[str, Field(description="The city to look up weather for")],
    units: Annotated[str, Field(pattern="^(metric|imperial)$")] = "metric",
) -> str | AgentToolResult:
    "Get current weather for a city. Returns a short text summary."
    # Free Open-Meteo geocoding + forecast.  No API key required.
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            geo = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "en"},
            )
            geo.raise_for_status()
            results = geo.json().get("results")
            if not results:
                # Returning is_error=True tells the model the call failed.
                return AgentToolResult(
                    content=[TextContent(text=f"Couldn't find a city named {city!r}.")],
                    is_error=True,
                )
            lat, lon = results[0]["latitude"], results[0]["longitude"]

            wx = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current_weather": True,
                    "temperature_unit": "celsius" if units == "metric" else "fahrenheit",
                },
            )
            wx.raise_for_status()
            cw = wx.json()["current_weather"]
            unit = "°C" if units == "metric" else "°F"
            # A plain str is wrapped as a successful text result.
            return f"{cw['temperature']}{unit}, wind {cw['windspeed']} km/h in {city}."
        except httpx.HTTPError as e:
            return AgentToolResult(
                content=[TextContent(text=f"Weather API error: {e}")],
                is_error=True,
            )


# --- The agent ----------------------------------------------------------

async def main():
    provider = AnthropicProvider(provider_id="anthropic", api_key=os.environ["ANTHROPIC_API_KEY"])
    agent = Agent(
        model=provider.model("claude-sonnet-4-6"),
        system_prompt="You are a concise weather assistant. Always use the tool; don't guess.",
        tools=[get_weather],
    )

    def on_event(event, signal=None):
        if event.type == "message_update" and event.stream_event.type == "text_delta":
            print(event.stream_event.delta, end="", flush=True)
        elif event.type == "tool_execution_start":
            print(f"\n[calling {event.tool_name}({event.args})]")
        elif event.type == "agent_end":
            print()

    agent.subscribe(on_event)

    # Wrap prompt() so Ctrl-C cleanly cancels the run.
    task = asyncio.create_task(agent.prompt("Weather in Tokyo and São Paulo, please."))
    try:
        await task
    except KeyboardInterrupt:
        agent.abort()
        await agent.wait_for_idle()


if __name__ == "__main__":
    asyncio.run(main())
```

Run:

```bash
pip install cubeloop httpx
export ANTHROPIC_API_KEY=sk-…
python weather_agent.py
```

Sample output:

```
[calling get_weather({'city': 'Tokyo', 'units': 'metric'})]
[calling get_weather({'city': 'São Paulo', 'units': 'metric'})]
Tokyo is currently 18°C with a wind speed of 12 km/h. São Paulo is 25°C with winds of 9 km/h.
```

## What's going on

- **Two tools in parallel.** The model emits two `get_weather` tool
  calls in the same assistant turn. CubeLoop runs them concurrently —
  the second one doesn't wait for the first.
- **Streaming text + tool events interleave.** The `on_event` filter
  handles both `text_delta` (for the final answer) and
  `tool_execution_start` (for the "thinking" indicator).
- **Errors are tool results, not exceptions.** A bad city or network
  hiccup returns `is_error=True`; the model gets the error message
  and usually retries with a different spelling.
- **Cancellation is clean.** `agent.abort()` propagates through
  `signal` into in-flight tools and the provider stream.

## Extending this recipe

- **Add caching:** memoize the geocoding lookup by `city` —
  Open-Meteo's coordinates are stable.
- **Add retries:** wrap with [`RetryMiddleware`](../guides/middleware/examples#retries-with-backoff)
  to handle transient API errors.
- **Persist conversations:** add a
  [`SQLiteCheckpointer`](../guides/checkpointing/sqlite) so follow-up
  questions ("and in Osaka?") have history.

## Run the example

A self-contained, runnable version of this recipe is in the repository at
[`examples/weather_agent.py`](https://github.com/cubeplexai/cubeloop/blob/main/examples/weather_agent.py).

```bash
git clone https://github.com/cubeplexai/cubeloop && cd cubeloop
uv sync

export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY [+ OPENAI_BASE_URL]
uv run --with httpx python examples/weather_agent.py
```

## See also

- [Building Your First Agent](../guides/agents/first-agent) — the same
  pattern with a hard-coded tool.
- [Tool Use & Parallel Execution](../guides/agents/tool-use) — more on
  parallel tool calls.
