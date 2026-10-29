---
title: 从 langgraph 迁移
description: "将你的 agent 从 LangGraph 迁移到 CubeLoop。"
---

# 从 langgraph 迁移

CubeLoop 和 [langgraph](https://github.com/langchain-ai/langgraph) 都用于构建
使用工具的 LLM agent，但两者的思维模型有所不同。
本页将 langgraph 概念映射到 CubeLoop，让你无需从头学起就能移植代码。

## 思维模型转换

| langgraph | CubeLoop | 原因 |
|---|---|---|
| **状态图**，包含节点、边、channel | **Agent 循环**，就是一个可读的普通 `while` 循环 | 线性循环比图更易于推理；CubeLoop 在运行时从不分支 —— 控制流在 middleware 中 |
| **Channels**（类型化状态槽） | **`AgentContext.extra`** + `AgentState.messages` | 一个 dict 加一个消息列表，覆盖了我们见过的所有状态结构 |
| **`StateGraph.add_node(name, fn)`** | middleware hook 或一个工具 | langgraph 节点中的函数在 CubeLoop 中分为两种角色：工具执行（由模型决定时）和 middleware（始终生效的变换） |
| **`add_edge(a, b)`** / `add_conditional_edges` | 内置：工具 → 下一轮 → 工具 → … | 条件形式（工具调用 → 重新提示）就是循环本身，无需显式构建 |
| **`MemorySaver` / `SqliteSaver` / `PostgresSaver`** | `MemoryCheckpointer` / `SQLiteCheckpointer` / `PostgresCheckpointer` | 思路相同，但采用追加写入 schema，而非完整快照 |
| **`config: {"configurable": {"thread_id": …}}`** | `Agent(thread_id=…)` | 作为 agent 的一等参数 |
| **`stream_mode="messages"` / `"values"` / `"updates"`** | `agent.subscribe(listener)` —— 统一事件流 | 一种模式，十一种事件类型 |
| **以 `@tool` 装饰器修饰的工具函数** | 带有 Pydantic 参数和 async execute 的 `AgentTool` | 更接近 OpenAI/Anthropic 原生形式 |
| **`HumanMessage`、`AIMessage`** | `UserMessage`、`AssistantMessage` | 相同的角色标记消息，只是重命名 |
| **通过 `interrupt_before` / `interrupt_after` 中断** | `agent.steer(...)`、`agent.follow_up(...)`、`agent.abort()` | 命令式控制，而非声明式中断点 |
| **任意检查点的时间旅行 / 分叉** | **`Agent.fork()`** / **`Agent.fork_once()`**，在 run 边界处 | CubeLoop 以已完成的 `run_id` 为基准；参见 [会话分叉](../guides/agents/forking) |
| **`config_schema`** | `Agent` 的构造函数参数 | 没有独立的 schema 层 |

## 并排对比：一个使用工具的 agent

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

`@tool` 装饰器对标 langgraph 的 `@tool`:输入 schema 来自函数签名,
docstring 作为描述,返回的普通 `str` 会自动包装。(若工具需要共享参数模型
或动态构建,完整的 `AgentTool(...)` 写法依然可用 —— 见
[工具使用](../guides/agents/tool-use)。)

CubeLoop 版本去除了：

- `StateGraph`、边、节点、`END` 哨兵、条件边。
- `ToolNode` 注册表 —— 工具直接传给 `Agent`。
- `should_continue` 函数 —— 循环自己知道是否有工具调用。
- `State` TypedDict —— 状态存在于 agent 上。

## 常见模式映射

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

CubeLoop 的追加写入模型每条消息的复杂度为 O(1)，与对话长度无关。
langgraph 保存完整快照，随历史记录线性增长。

### 流式输出

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

一个订阅者，一个流 —— 无需模式标志。

### 中断 / 人机协同

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

通过 middleware 实现命令式中断。你可以针对每次调用做决策，而不是
配置图级别的中断点。

### 分支

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

没有内置的分支原语；流程控制通过 `should_stop_after_turn` 和
`after_model_response` 完成。

### 会话分叉

```python
# langgraph — 时间旅行到某个已保存的检查点
config = {"configurable": {"thread_id": "t1", "checkpoint_id": "<id>"}}
app.update_state(config, {"messages": [...]})
result = app.invoke(None, config)

# CubeLoop — 在已完成的 run 边界处持久化分叉
await agent.fork(
    src_thread_id="conv_123",
    new_thread_id="conv_456",
    after_run_id="R1",
)

# CubeLoop — 临时一次性探针（不写入任何线程）
result = await agent.fork_once(
    src_thread_id="conv_123",
    message="如果你当时回答了'是'呢？",
    after_run_id="R1",
)
print(result.text)
```

CubeLoop 在已完成的 run 边界处分叉，而非任意 mid-run 检查点。`fork`
创建可继续对话的持久化分支；`fork_once` 运行单次探针并丢弃所有内容。
参见[会话分叉](../guides/agents/forking)。

## LangGraph 仍优于 CubeLoop 的能力

- **图原生的多智能体编排。** CubeLoop 提供 `SubagentMiddleware`，支持工具驱动的委派：
  父 Agent 可以将自包含任务分派给具有专属 prompt、model、tools 和 middleware 的
  类型化子 Agent。CubeLoop 目前尚未提供 LangGraph 那种显式、可持久化的图模型，
  用于 supervisor 路由、条件分支、并行 fan-out/fan-in 以及图级状态管理。
- **可视化图渲染。** 没有 `app.get_graph().draw_mermaid()` 的等价物。
  CubeLoop 的流程是线性的，画出来也只是一条直线。
- **原生 trace 可视化 UI。** CubeLoop 不像 LangSmith / Langfuse 那样
  渲染自己的 trace 可视化界面；它改为发射符合厂商中立标准的
  OpenTelemetry —— 通过
  `Tracer(exporters=[OTLPSpanExporter(...)])` 将任意 OTLP 后端
  （LangSmith 的 OTel endpoint、Langfuse v3、Jaeger、Tempo、
  Honeycomb、Datadog 等）接入。参见
  [Tracing → OTLP & Backends](../guides/tracing/otlp)。

## CubeLoop 有而 langgraph 没有的功能

- **原生 OpenTelemetry tracing** —— `Tracer` + `Meter` 开箱即用地
  发射带 GenAI-semconv 属性的 OTel span，可被任意 OTLP 后端采集。
  参见 [Tracing → Overview](../guides/tracing/overview)。
- **原生 async-first** —— 每个入口都是 async，没有
  `app.invoke` vs. `app.ainvoke` 的分裂。
- **追加写入持久化** —— O(1) 数据库写入，消息可用 JSONB 查询。
- **3 个核心依赖** vs. langchain-core + langgraph-sdk + 传递依赖。
- **流式逼真的测试 provider**（`FauxProvider`）随包附带。
- **MCP loaders** 支持 HTTP + stdio 传输。

## 移植检查清单

1. 将 `StateGraph` 构造替换为单个 `Agent(...)` 调用。
2. 将 `@tool` 装饰的函数移至 `AgentTool` 实例（Pydantic 模型作参数，
   async execute）。
3. 将 `MemorySaver` / `SqliteSaver` / `PostgresSaver` 替换为
   `MemoryCheckpointer` / `SQLiteCheckpointer` / `PostgresCheckpointer`。
4. 将 `stream_mode` 回调替换为 `agent.subscribe(...)`。
5. 将做消息变换的自定义节点转换为 `Middleware` hook。
6. 将 `interrupt_before/after` 转换为 `before_tool_call` /
   `after_model_response` middleware。
7. 如果你有 `summary` 或 `route` 节点 —— 将其折叠进
   `after_model_response`，使用 `decision="stop"` 或 `"loop_to_model"`。

## 另请参见

- [核心概念](../getting-started/core-concepts) —— 你正在映射的构建块。
- [Middleware → 组合](../guides/middleware/composition) —— 流程控制逻辑所在之处。
- [Checkpointing](../guides/checkpointing/sqlite) —— 新的持久化方案。
