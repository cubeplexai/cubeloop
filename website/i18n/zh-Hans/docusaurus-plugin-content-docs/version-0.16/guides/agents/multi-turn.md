---
title: 多轮对话
description: "使用 CubeLoop 的有状态 agent 循环和消息历史构建多轮对话 agent。"
---

# 多轮对话

在 CubeLoop 中，一个"轮次"（turn）是：用户输入 → 模型响应（可能包含工具调用）→ 可选的对工具结果的后续模型响应。agent 的 `_messages` 列表在多轮对话中不断增长。本指南介绍如何正确驱动多轮对话流程，以及如何在 agent 处于思考过程中注入输入。

## 基本模式

前一次调用返回后，再次调用 `prompt` 即可：

```python
await agent.prompt("Hi, my name is Sam.")
await agent.prompt("What's my name?")
# → "Your name is Sam."
```

历史消息保存在 `agent.state.messages` 中。CubeLoop 会追加每条用户消息、每条 assistant 消息以及每个工具结果。每次调用时 provider 都会收到完整的消息列表，因此 context window 的大小至关重要（参见下方的[上下文管理](#context-management)）。

## 运行中修正：`agent.steer()`

有时用户希望在模型仍处于当前轮次时进行补充或修正。此时可使用 `steer()`：

```python
import asyncio

async def main():
    task = asyncio.create_task(agent.prompt("Plan a 5-day trip to Kyoto."))
    await asyncio.sleep(2)
    # User changed their mind:
    agent.steer(UserMessage(content=[TextContent(text="Make it 3 days, not 5.")]))
    await task
```

修正消息会被加入队列，循环在轮次之间（实际上是在一批工具调用与下一次模型调用之间）取出该消息。agent 会在下次响应前看到它，消息不会丢失。

`Agent` 上的 `steering_mode` 控制消息的排空方式：

- `"one-at-a-time"`（默认）—— 每个取出点只处理一条队列消息。
- `"all"` —— 一次排空所有队列消息。

需要把排队消息与持久化状态关联起来的服务宿主，可以使用带标识的会话 API：

```python
from cubeloop import InputEnvelope

receipt = agent.session.submit_input(
    InputEnvelope(
        input_id=client_message_id,
        message=UserMessage(content=[TextContent(text=content)]),
        mode="steer",
    )
)
```

`queued` 只表示内存队列接收了这条消息，并不是持久化确认。订阅会话的
`input_committed` 事件；只有消息成功追加后，其 durability 才会是
`checkpoint`。steering 和 follow-up 仍使用彼此独立的队列，并保留各自原有的
取出时机和批处理策略。

## 排队后续消息：`agent.follow_up()`

`follow_up` 用于"当前运行结束后，以此内容开启新的轮次"。这是聊天 UI 的典型模式：用户在 assistant 还在响应时便开始输入。

```python
agent.follow_up(UserMessage(content=[TextContent(text="And what about Osaka?")]))
# When the current prompt() finishes, the loop picks this up
# automatically and starts a new turn.
```

如果调用 `follow_up` 时 agent 处于空闲状态，仍需触发一次运行——大多数应用会在 `prompt()` 返回后调用 `await agent.resume()` 来排空队列。

## `resume()` —— 从上次消息继续 {#resume--continue-from-the-last-message}

`resume()` 是"从上次中断处继续"的入口。有两种用途：

1. **从 checkpointer 加载后。** 状态中有消息但没有进行中的 prompt。`resume()` 会查看最后一条消息并采取行动：
   - assistant → 期待一条排队的 steer/follow_up 以转换为新的用户轮次；否则抛出异常。
   - tool_result → 用工具输出重新调用模型。
   - user → 用该用户消息重新调用模型。
2. **abort 之后。** 一旦 `agent.abort()` 完成清理，即可 resume。

```python
async with SQLiteCheckpointer("conv.db") as cp:
    agent = Agent(model=…, checkpointer=cp, thread_id="conv-1")
    await agent.prompt("hello")    # loads existing history first
    await agent.prompt("how are you?")
```

实例化后的第一次 `prompt()` 会加载已有的 thread。后续调用只是追加。

## 上下文管理 {#context-management}

CubeLoop **不会**代替你截断或摘要上下文。每次轮次都会将完整消息列表发送给模型。几种应对策略：

- **手动截断** —— 实现一个 [`transform_context`](../middleware/hooks#transform_context) middleware，返回一个滑动窗口。
- **摘要 pass** —— 定期注入摘要消息，并通过 `transform_context` 丢弃旧消息。
- **自定义 `convert_to_llm`** —— 在序列化前（最后一个时机）重塑历史，而不修改 `agent.state.messages`。用户可见的历史保持完整，模型看到的则更少。

参见 [Middleware → Examples](../middleware/examples#sliding-window-truncation) 中的完整示例。

## 取消与等待空闲

```python
agent.abort()                # signals the current run to stop
await agent.wait_for_idle()  # awaits the run-cleanup
```

如果 agent 已经处于空闲状态，`wait_for_idle()` 是空操作。可以在任何地方安全调用。

## 从磁盘恢复状态

```python
from cubeloop.checkpointer import SQLiteCheckpointer

async with SQLiteCheckpointer("conv.db") as cp:
    agent = Agent(
        model=model,
        checkpointer=cp,
        thread_id="user-42",
    )
    # First prompt() restores the saved history if any.
    await agent.prompt("continue our chat")
```

checkpoint 中的 `extra` 映射也会被恢复。希望持久化 per-thread 状态的
middleware 应将数据写入 `context.extra`；checkpointer 的 `save_extra` 会在
产生工具结果的轮次结束之后、运行暂停时，以及 `agent_end` 时被调用。宿主应通过 `agent.session.state_context` 访问这个引用稳定的
实时映射；需要在空闲时显式恢复状态，则调用
`await agent.session.load_checkpoint()`。消息和 extra 会一起安装，即使 checkpoint
中的消息列表为空也一样。

## 常见陷阱

- **前一次执行仍在运行或收尾时发起新执行** 会抛出 `ExecutionBusy`。请改用
  `steer()` 或 `follow_up()`，或先调用 `wait_for_idle()`。
- **`resume()` 时最后一条消息是 assistant 且队列为空** 会抛出 `"Cannot continue from message role: assistant"`。请先排队一条后续消息，或改用 `prompt()`。
- **历史无限增长** —— 若没有 `transform_context` middleware，最终会触达 context 限制。请尽早规划截断/摘要策略。
- **多个 agent 使用相同的 `thread_id`** —— 仅追加写入对顺序是安全的，但两个 agent 同时写入同一 thread 会导致消息交错。每个 thread 使用一个 agent 实例，或在应用层进行协调。

## 另请参阅

- [Streaming Events](./streaming) —— steering/follow_up 的精确事件顺序。
- [Checkpointing → SQLite](../checkpointing/sqlite) —— 持久化历史记录。
- [Recipes → Persistent Chat](../../recipes/persistent-chat) —— 带历史重载的完整多轮对话应用。
