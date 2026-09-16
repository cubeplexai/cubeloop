---
title: 快速开始
description: "在几分钟内构建你的第一个 CubeLoop agent。"
---

# 快速开始

五分钟内跑通一个能流式输出、会调用工具的 Agent。我们写一个天气 Agent：
它把一个 Python 函数当工具调用、把 Claude 的回复逐字符流式打出来,
然后干净退出。

## 前置条件

- Python 3.11+
- 已安装 `cubeloop`（`pip install cubeloop`)
- 环境变量中有 `ANTHROPIC_API_KEY`

## 完整脚本

存为 `weather_agent.py`：

```python title="weather_agent.py"
import asyncio
import os

from cubeloop import Agent, tool
from cubeloop.providers.anthropic import AnthropicProvider


@tool
async def get_weather(city: str) -> str:
    "获取一个城市的当前天气"
    # 真实应用里：调一个 HTTP 天气 API。这里直接返回一段假数据。
    return f"{city} 现在 72°F,晴"


async def main():
    provider = AnthropicProvider(provider_id="anthropic", api_key=os.environ["ANTHROPIC_API_KEY"])

    agent = Agent(
        model=provider.model("claude-sonnet-4-6"),
        system_prompt="你是一个简洁的天气助手。",
        tools=[get_weather],
    )

    # 在 prompt() 之前订阅 —— 这是看到流式事件的关键。
    def on_event(event, signal=None):
        if event.type == "message_update" and event.stream_event.type == "text_delta":
            print(event.stream_event.delta, end="", flush=True)
        elif event.type == "agent_end":
            print()  # 末尾换行

    agent.subscribe(on_event)
    await agent.prompt("东京现在天气怎么样？")


asyncio.run(main())
```

跑起来：

```bash
python weather_agent.py
```

你会看到 Claude 流式输出类似
*"东京当前天气 72°F,晴..."* 的句子,工具结果已经被串进去。

## 刚刚发生了什么

CubeLoop 跑了一个概念上长这样的循环：

1. `agent.prompt("东京现在天气怎么样？")` 把一条 `UserMessage` 入队,
   然后调用模型。
2. 模型决定调用 `get_weather(city="Tokyo")`——CubeLoop 按 `@tool` 从函数
   签名生成的 schema 解析 JSON 参数,调用你的 `async def`,把结果作为
   `ToolResultMessage` 反馈回去。
3. 模型产生最终的 assistant 回复,以 `text_delta` 事件流的形式回来。
4. 循环发出 `agent_end` 然后返回。

`agent.subscribe(...)` 注册了一个回调,接收运行时发出的每一个事件：
`agent_start`、`turn_start`、`message_start`、`text_delta`、
`tool_execution_start`、`tool_execution_end`、`message_end`、
`turn_end`、`agent_end`。这个脚本只关心 `text_delta`,但你可以基于
任意其他事件渲染 UI。

## 调试之前请先读：常见困惑

- **必须在 prompt 之前订阅。** Listener 只会收到 `subscribe` 之后发出
  的事件。先 prompt 再 subscribe,早期事件就丢了。
- **`provider.model(...)` 会把模型绑定到 provider。** provider 保存凭证和可选的 `provider_id` 元数据；模型 id 必须是这个 provider 支持的模型名。
- **工具就是被装饰的 async 函数。** `@tool` 从带类型的参数推断输入
  schema,用 docstring 作为描述;返回 `str`(或 `AgentToolResult`)。
  需要原始的 `(tool_call_id, params, *, signal, on_update)` 写法、共享
  参数模型,或 `on_update` 进度?见
  [工具使用](../guides/agents/tool-use)。
- **`agent.prompt()` 一次只能跑一个 prompt。** 跑的过程中,用
  [`agent.steer()`](../guides/agents/multi-turn) 插入修正,或
  `agent.follow_up()` 把后续消息排队。

## 下一步

- [核心概念](./core-concepts) —— `Agent`、`Tool`、`Provider`、`Stream`、
  `Middleware`、`Checkpointer` 背后的心智模型。
- [构建第一个 Agent](../guides/agents/first-agent) —— 工具、流式、错误
  处理的完整走查。
- [工具使用与并行执行](../guides/agents/tool-use) —— 让 Agent 同时
  扇出多个工具。
- [Recipes → Weather Agent](../recipes/weather-agent) —— 上面这个脚本
  的加强版,真的会发 HTTP 请求。
