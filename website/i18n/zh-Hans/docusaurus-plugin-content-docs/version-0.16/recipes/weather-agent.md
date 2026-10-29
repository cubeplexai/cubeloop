---
title: 天气 Agent
description: "使用 CubeLoop 工具构建一个调用真实 API 数据的天气 agent。"
---

# Recipe：天气 Agent

一个完整可运行的 agent，将真实的天气 API 作为工具调用。
演示了 HTTP 调用工具、错误处理、流式 UI 和取消操作。

**预计耗时：** 5 分钟。
**依赖：** `cubepi`、`httpx`、`ANTHROPIC_API_KEY`。

## 脚本

```python title="weather_agent.py"
import asyncio
import os

import httpx
from typing import Annotated
from pydantic import Field

from cubeloop import Agent, AgentToolResult, TextContent, tool
from cubeloop.providers.anthropic import AnthropicProvider


# --- 工具 -----------------------------------------------------------------

@tool
async def get_weather(
    city: Annotated[str, Field(description="The city to look up weather for")],
    units: Annotated[str, Field(pattern="^(metric|imperial)$")] = "metric",
) -> str | AgentToolResult:
    "Get current weather for a city. Returns a short text summary."
    # 免费的 Open-Meteo 地理编码 + 预报。无需 API key。
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            geo = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "en"},
            )
            geo.raise_for_status()
            results = geo.json().get("results")
            if not results:
                # 返回 is_error=True 告诉模型这次调用失败了。
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
            # 普通 str 会被包装成一次成功的文本结果。
            return f"{cw['temperature']}{unit}, wind {cw['windspeed']} km/h in {city}."
        except httpx.HTTPError as e:
            return AgentToolResult(
                content=[TextContent(text=f"Weather API error: {e}")],
                is_error=True,
            )


# --- Agent ----------------------------------------------------------------

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

    # 包裹 prompt()，使 Ctrl-C 能干净地取消运行。
    task = asyncio.create_task(agent.prompt("Weather in Tokyo and São Paulo, please."))
    try:
        await task
    except KeyboardInterrupt:
        agent.abort()
        await agent.wait_for_idle()


if __name__ == "__main__":
    asyncio.run(main())
```

运行：

```bash
pip install cubeloop httpx
export ANTHROPIC_API_KEY=sk-…
python weather_agent.py
```

示例输出：

```
[calling get_weather({'city': 'Tokyo', 'units': 'metric'})]
[calling get_weather({'city': 'São Paulo', 'units': 'metric'})]
Tokyo is currently 18°C with a wind speed of 12 km/h. São Paulo is 25°C with winds of 9 km/h.
```

## 运行原理

- **两个工具并行执行。** 模型在同一个 assistant 轮次中发出两次
  `get_weather` 工具调用。CubeLoop 并发运行它们 ——
  第二个不等第一个完成。
- **流式文本和工具事件交错。** `on_event` 过滤器同时处理
  `text_delta`（用于最终答案）和 `tool_execution_start`（用于"思考中"指示器）。
- **错误是工具结果，不是异常。** 城市名错误或网络抖动会返回
  `is_error=True`；模型收到错误消息后通常会用不同拼写重试。
- **取消是干净的。** `agent.abort()` 通过 `signal` 传播到正在运行的
  工具和 provider 流。

## 扩展本 recipe

- **添加缓存：** 按 `city` 记忆化地理编码查询 ——
  Open-Meteo 的坐标是稳定的。
- **添加重试：** 用 [`RetryMiddleware`](../guides/middleware/examples#retries-with-backoff)
  包裹以处理瞬时 API 错误。
- **持久化对话：** 添加
  [`SQLiteCheckpointer`](../guides/checkpointing/sqlite)，使追问
  （"那大阪呢？"）能有历史上下文。

## 另请参见

- [构建你的第一个 Agent](../guides/agents/first-agent) —— 使用硬编码工具的相同模式。
- [工具使用与并行执行](../guides/agents/tool-use) —— 更多关于并行工具调用的内容。

## 运行示例

仓库中有一份完整可运行的代码，位于
[`examples/weather_agent.py`](https://github.com/cubeplexai/cubeloop/blob/main/examples/weather_agent.py)。

```bash
git clone https://github.com/cubeplexai/cubeloop && cd cubeloop
uv sync

export ANTHROPIC_API_KEY=sk-ant-...   # 或 OPENAI_API_KEY [+ OPENAI_BASE_URL]
uv run --with httpx python examples/weather_agent.py
```
