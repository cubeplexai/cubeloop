---
title: 第一个 Agent
description: "从零开始创建并运行你的第一个 CubeLoop agent。"
---

# 构建你的第一个 Agent

这是 [快速开始](../../getting-started/quick-start) 的"加长版"。我们
端到端构建一个单工具 Agent,然后逐层加上你后续大概率会想要的东西：
流式 UI、错误处理、取消按钮。

## 第 1 步 —— 配 provider 和 model

provider 是到 LLM API 的连接；用 `provider.model("id", ...)` 把模型 ID 绑定
到 provider 上，产出 `Agent(model=...)` 需要的那个值。

```python
import os
from cubeloop.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(provider_id="anthropic", api_key=os.environ["ANTHROPIC_API_KEY"])
model = provider.model(
    "claude-sonnet-4-6",
    max_tokens=4096,         # 回复 token 上限
    context_window=200_000,  # 模型硬上限；通常默认就够
    temperature=0.7,
)
```

`provider_id` 现在写在 provider 构造函数上，会自动传到 bound model 里——框架
内部用来 clamp 思考等级和打 tag。0.6 那种手工构造 `Model` 再传
`Agent(provider=..., model=...)` 的姿势在 0.7 已经不工作了。

## 第 2 步 —— 声明一个工具

工具就是一个用 `@tool` 装饰的 async 函数:

```python
from cubeloop import tool


@tool
async def get_weather(city: str) -> str:
    "获取一个城市的当前天气,返回简短文字。"
    # 真实工作放在这里 —— 调 HTTP API、查 DB 等。
    return f"{city} 现在 72°F,晴"
```

注意几点：

- 输入 schema 从带类型的参数生成并发给模型;docstring 作为工具描述。
  Pydantic `Field(...)` 的默认值与元数据都会被保留。
- 返回 `str`(自动包成文本)、`Content`、内容列表,或在需要
  `details`/`is_error` 时返回完整的 `AgentToolResult`。
- 需要取消或进度流?在签名里声明 `signal`(用户取消时被 set 的
  `asyncio.Event`)和/或 `on_update(partial)`,CubeLoop 会注入它们 —— 见
  [工具使用](./tool-use)。
- 需要共享参数模型或动态构建?长写法 `AgentTool(...)` 与之等价 —— 见
  [工具使用](./tool-use)。

## 第 3 步 —— 组装 Agent

```python
from cubeloop import Agent

agent = Agent(
    model=model,
    system_prompt="你是一个简洁的天气助手。",
    tools=[get_weather],
)
```

不需要工具时传 `tools=[]`（或干脆省掉)。

## 第 4 步 —— 订阅事件

`agent.subscribe(listener)` 是观察运行过程的入口。Listener 会收到
每一个 `AgentEvent`:

```python
def on_event(event, signal=None):
    if event.type == "message_update" and event.stream_event.type == "text_delta":
        print(event.stream_event.delta, end="", flush=True)
    elif event.type == "tool_execution_start":
        print(f"\n→ 调用 {event.tool_name}({event.args})")
    elif event.type == "tool_execution_end":
        print(f"  ✓ 完成")

agent.subscribe(on_event)
```

可以注册多个 listener,它们都会收到所有事件。**在 `prompt()` 之前
订阅** —— 事件在 run 一开始就会发出。

## 第 5 步 —— prompt 然后跑

```python
import asyncio

async def main():
    await agent.prompt("东京天气怎么样？")

asyncio.run(main())
```

`agent.prompt()` 不返回任何值。结果存在 `agent.state.messages`
（完整历史）和 `agent.state.streaming_message`（当前正在产生的消息,
在两轮之间为 `None`）上。

## 加上错误处理

当 `provider.stream()` 抛异常时,Agent 循环仍然会产出一条
`AssistantMessage`,其 `stop_reason="error"`、`error_message` 已填好。
事件序列是:`message_start` → `message_end` → `turn_end` → `agent_end`。

两种处理方式：

1. 在 subscriber 里捕获 `event.type == "agent_end"`,看最后一条消息
   的 `stop_reason`：

    ```python
    def on_event(event, signal=None):
        if event.type == "agent_end":
            last = event.messages[-1]
            if getattr(last, "stop_reason", "") == "error":
                print(f"\nerror: {last.error_message}")
    ```

2. 或者在 `await agent.prompt(...)` 返回之后,检查
   `agent.state.error_message`。

## 加一个"取消按钮"

`agent.abort()` 会 set run 级别的 signal。Provider 流短路成
`"aborted"`,正在运行的工具看到 `signal.is_set() == True`,
循环干净地发出 `agent_end`。

```python
async def main():
    task = asyncio.create_task(agent.prompt("搜索…"))
    await asyncio.sleep(0.5)
    agent.abort()
    await task              # 一定会 return —— 不会抛异常
    await agent.wait_for_idle()
```

## 常见坑

- **`RuntimeError: Agent is already processing a prompt.`** ——
  上一次 `prompt()` 没 await 完你又调了一次。用 `await
  agent.wait_for_idle()`,或用 `steer()` / `follow_up()` 排队。
- **没有 `text_delta` 事件** —— 是不是在 `prompt()` 之后才订阅的？
  Listener 只看得到注册之后的事件。
- **Tool not found** —— 模型调了一个 `name` 不在 `tools=[...]` 里的
  工具。CubeLoop 把这种情况包装成一个 `is_error=True` 的工具结果,
  不会崩 —— 在 `tool_execution_end` 事件的 `result` 里能看到。
- **Pydantic ValidationError 被吞掉了** —— 如果模型产出格式错误的
  JSON,CubeLoop 会把 validation error 也包成工具的 error result 喂
  回去,模型通常下一轮自动纠正。

## 下一步

- [工具使用与并行执行](./tool-use) —— 同时多个工具、sequential 模式、
  `terminate`、增量进度。
- [流式事件](./streaming) —— 完整的事件分类。
- [多轮会话](./multi-turn) —— 跨轮保留状态、`steer`、`follow_up`、
  `resume`。
