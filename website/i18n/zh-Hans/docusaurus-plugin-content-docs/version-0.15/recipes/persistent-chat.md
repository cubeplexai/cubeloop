---
title: 持久化聊天
description: "使用 CubeLoop 和 SQLiteCheckpointer 构建持久化聊天应用。"
---

# Recipe：持久化聊天（SQLite）

一个能在重启后恢复的 REPL 聊天程序。对话历史保存在 SQLite 文件中；
每位用户拥有独立的 `thread_id`。

**预计耗时：** 5 分钟。
**依赖：** `cubeloop[sqlite]`、`ANTHROPIC_API_KEY`。

## 脚本

```python title="chat.py"
import asyncio
import os
import sys

from cubeloop import Agent
from cubeloop.checkpointer import SQLiteCheckpointer
from cubeloop.providers.anthropic import AnthropicProvider


async def main(thread_id: str):
    provider = AnthropicProvider(provider_id="anthropic", api_key=os.environ["ANTHROPIC_API_KEY"])

    async with SQLiteCheckpointer("chat.db") as cp:
        agent = Agent(
            model=provider.model("claude-sonnet-4-6"),
            system_prompt="You are a concise, friendly assistant.",
            checkpointer=cp,
            thread_id=thread_id,
        )

        def on_event(event, signal=None):
            if event.type == "message_update" and event.stream_event.type == "text_delta":
                print(event.stream_event.delta, end="", flush=True)
            elif event.type == "agent_end":
                print()

        agent.subscribe(on_event)

        print(f"chatting on thread {thread_id!r}. Ctrl-D to quit.\n")
        loop = asyncio.get_event_loop()
        while True:
            try:
                user_input = await loop.run_in_executor(None, input, "you> ")
            except EOFError:
                print()
                return
            if not user_input.strip():
                continue
            print("ai > ", end="", flush=True)
            await agent.prompt(user_input)


if __name__ == "__main__":
    asyncio.run(main(thread_id=sys.argv[1] if len(sys.argv) > 1 else "default"))
```

运行：

```bash
pip install "cubeloop[sqlite]"
export ANTHROPIC_API_KEY=sk-…
python chat.py alice
# 聊一会儿，然后按 Ctrl-D。

python chat.py alice
# 历史记录已恢复。问"你刚才告诉你什么了？" —— 模型
# 记得上一次会话的内容。

python chat.py bob
# 不同的 thread，空白起点。
```

## 运行原理

- **每个进程的第一次 `prompt()` 会加载历史记录。** CubeLoop 在第一次
  prompt 开始时检查一次 checkpointer，恢复 `agent.state.messages`，
  然后继续。
- **每个 `message_end` 追加写入数据库。** 没有批处理，没有有损缓冲。
  如果进程在流传输中途崩溃，下次运行会从最后一条已提交的消息继续。
- **system prompt 不会持久化** —— 它是 agent 构造的一部分，而不是
  状态。请将其保留在代码中（或环境配置中）。

## 多用户路由

在 web 服务中，从已认证的用户信息派生 `thread_id`：

```python
agent = Agent(
    model=model,
    checkpointer=cp,
    thread_id=f"user-{user_id}",
)
```

你可以从一个 `SQLiteCheckpointer` 为数千名用户提供服务 ——
各线程通过 `thread_id` 隔离。

## 清理 thread

设计上没有内置的"删除 thread" API —— checkpointer 是追加写入的。
如需清理，直接按 `thread_id` 删除行：

```bash
sqlite3 chat.db "DELETE FROM messages WHERE thread_id='alice'; DELETE FROM thread_extra WHERE thread_id='alice'"
```

或者实现一个小型管理工具，按需执行 SQL。

## 滑动上下文窗口

在长对话后，模型的上下文会变得昂贵。添加一个
[`SlidingWindow`](../guides/middleware/examples#sliding-window-truncation)
middleware：

```python
from cubeloop import Middleware

class SlidingWindow(Middleware):
    def __init__(self, n: int) -> None:
        self.n = n

    async def transform_context(self, messages, *, ctx, signal=None):
        return messages[-self.n:] if len(messages) > self.n else messages


agent = Agent(
    model=model,
    checkpointer=cp,
    thread_id=thread_id,
    middleware=[SlidingWindow(40)],
)
```

数据库保留所有消息；模型只看到最近的 40 条。用户可见的历史记录
（如聊天界面渲染历史对话）保持完整。

## 切换到 Postgres

代码相同，仅替换 checkpointer：

```python
from cubeloop.checkpointer import PostgresCheckpointer

async with PostgresCheckpointer("postgresql://…") as cp:
    agent = Agent(model=…, checkpointer=cp, thread_id=…)
```

Postgres 适合多实例服务或大量并发用户 ——
参见 [Postgres + FastAPI](./postgres-fastapi)。

## 常见陷阱

- **忘记 `async with`** —— 没有它，SQLite 连接永远不会被打开。
  你会得到 `AssertionError`。请务必包裹。
- **多个进程写入同一个 `thread_id`** —— 历史记录会交叉混乱。
  一个 thread 对应一个 agent，或者迁移到 Postgres。
- **`chat.db` 放在 `/tmp`** —— 某些操作系统重启时会清空 `/tmp`。
  用户数据请使用 `~/.local/share/myapp/chat.db` 或类似路径。

## 另请参见

- [多轮对话](../guides/agents/multi-turn) —— `steer`、`follow_up`、`resume`。
- [SQLite Checkpointing](../guides/checkpointing/sqlite) —— 后端详细说明。
- [可恢复长任务](./resumable-tasks) —— 当崩溃发生在工具执行中途，而非轮次之间时。

## 运行示例

仓库中有一份完整可运行的代码，位于
[`examples/persistent_chat.py`](https://github.com/cubeplexai/cubeloop/blob/main/examples/persistent_chat.py)。

```bash
git clone https://github.com/cubeplexai/cubeloop && cd cubeloop
uv sync --extra sqlite

export ANTHROPIC_API_KEY=sk-ant-...   # 或 OPENAI_API_KEY [+ OPENAI_BASE_URL]
uv run python examples/persistent_chat.py alice
# Ctrl-D 退出后重新运行 —— 历史记录会恢复。
uv run python examples/persistent_chat.py bob
# 不同的 thread_id → 全新对话。
```
