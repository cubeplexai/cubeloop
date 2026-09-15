---
title: SQLite 检查点
description: "使用 SQLiteCheckpointer 实现轻量级单进程 agent 状态持久化。"
---

# SQLite 检查点

`SQLiteCheckpointer` 是轻量级持久化后端:单个本地文件、无 server、
追加式消息日志。它是个人电脑、单进程应用、桌面工具,以及任何"单
进程独占会话"场景下的默认选择。

安装 extra:

```bash
pip install "cubeloop[sqlite]"
```

会拉入 `aiosqlite`。

## 基本用法

```python
import asyncio
from cubeloop import Agent
from cubeloop.checkpointer import SQLiteCheckpointer
from cubeloop.providers.anthropic import AnthropicProvider


async def main():
    provider = AnthropicProvider(provider_id="anthropic", api_key="…")
    async with SQLiteCheckpointer("agent.db") as cp:
        agent = Agent(
            model=provider.model("claude-sonnet-4-6"),
            checkpointer=cp,
            thread_id="user-42",
        )
        await agent.prompt("记一下:我最爱的颜色是青色。")

        # 之后再次启动脚本 —— 文件还在。
        await agent.prompt("我说过我最爱什么颜色？")
        # → "你说过是青色。"


asyncio.run(main())
```

两件事必须记住：

1. **`thread_id`** 是会话标识符 —— 一般是用户 id 或 session id。
   两个 Agent 用同一个 `thread_id` 共享同一段历史。
2. **`async with SQLiteCheckpointer(...)`** 是必需的：context manager
   在 `__aenter__` 里打开连接、做一次性建表。不用 context manager
   会抛 `AssertionError`。

## 会持久化什么

`SQLiteCheckpointer` 首次使用时建两张表：

```sql
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    message_json TEXT NOT NULL,
    created_at REAL NOT NULL DEFAULT (julianday('now'))
);

CREATE TABLE thread_extra (
    thread_id TEXT PRIMARY KEY,
    extra_json TEXT NOT NULL DEFAULT '{}'
);
```

- 每条 `UserMessage`、`AssistantMessage`、`ToolResultMessage` 各占
  `messages` 一行。JSON payload 通过 Pydantic `model_dump()` 生成。
- `AgentContext` 的 `extra` dict 在 `agent_end` 时持久化到 `thread_extra`。
  Middleware 需要持久化线程级状态时,往 `context.extra` 写。

这个 schema 是追加式的。CubeLoop 从不 update 或 delete 行。

## HITL 挂起表

当使用 [HITL](../hitl/overview) 模块时,每次 `__aenter__` 会自动创建一个额外的表：

```sql
CREATE TABLE IF NOT EXISTS thread_pending_request (
    thread_id TEXT PRIMARY KEY,
    request_json TEXT NOT NULL,
    created_at REAL NOT NULL DEFAULT (julianday('now'))
);
```

无需手动迁移 —— `CREATE TABLE IF NOT EXISTS` 是零等的。

## CubeLoop 什么时候读

构造 Agent 后的 **第一次** `prompt()`,CubeLoop 会调
`load(thread_id)`。如果线程存在,历史恢复到 `agent.state.messages`,
`extra` 恢复到 agent 私有的 `_extra` dict。

后续 `prompt()` 不再读取 —— 内存里的状态是权威。

这意味着:**不要让一个 `Agent` 实例跨进程共享**。进程 A 的内存状态
会和进程 B 的写入发散。

## 多线程隔离

```python
async with SQLiteCheckpointer("agent.db") as cp:
    alice = Agent(model=…, checkpointer=cp, thread_id="alice")
    bob   = Agent(model=…, checkpointer=cp, thread_id="bob")
    # 每个调用只 load/append 自己的 thread。
```

你可以让多个用户共用一个 checkpointer —— `thread_id` 负责隔离。

## 并发模型

Checkpointer 内部对每次读写都用 `asyncio.Lock`。SQLite 本身允许
多进程写,但 CubeLoop 的假设是单 Agent 实例独占一个 thread。多进程
同时写同一个 `agent.db`:

- 读是安全的。
- 并发写 **不同** thread 是安全的。
- 并发写 **同一** thread 会交错 —— 你会看到两条 assistant 消息背靠背
  之类的怪事。

如果你需要跨进程共享 thread 写入,请用 [Postgres](./postgres),
它对每个 thread 都用 advisory lock。

## 条件式清理 pending request

`clear_pending_request_if_matches(thread_id, question_id=..., run_id=...)`
只有在当前 pending request 的问题 ID 和所属 run ID 都仍然匹配时，才会
删除 pending 状态。SQLite 在一条语句中完成比较和删除；checkpointer 的
写锁会将它与同一进程内的其他写入串行化。

## 文件放哪里

生产环境用绝对路径：

```python
SQLiteCheckpointer("/var/lib/myapp/agent.db")
```

相对路径在 `__aenter__` 时按 `os.getcwd()` 解析。目录必须存在,
请预先创建。

## 备份与查看

文件就是普通 SQLite 数据库:

```bash
# 看一个 thread 的历史
sqlite3 agent.db "SELECT message_json FROM messages WHERE thread_id='user-42' ORDER BY id"

# 备份
cp agent.db agent.db.bak

# VACUUM 回收空间(可选 —— 文件大小随历史线性增长)
sqlite3 agent.db "VACUUM"
```

## 常见坑

- **没用 `async with`** —— `AssertionError: self._db is not None`。
  一定要用 `async with` 包。
- **两个进程写同一个 thread** —— 交错历史。要么用 Postgres,要么在
  应用层协调。
- **未启用 WAL 模式** —— CubeLoop 走默认 journal mode 以确保可移植性。
  对单写多读应用,一次性 `sqlite3 agent.db "PRAGMA journal_mode=WAL"`
  能显著提升并发读。
- **忘传 `thread_id`** —— 不传时,Agent 没有持久化绑定。checkpointer
  会被静默忽略。一定要同时传两个。

## 另请参阅

- [Postgres 检查点](./postgres) —— 多实例部署。
- [自定义后端](./custom) —— 为 Redis、DynamoDB 等实现 Protocol。
- [Recipes → Persistent Chat](../../recipes/persistent-chat) —— 端到端
  的 SQLite 应用。
- [Recipes → 可恢复的长任务](../../recipes/resumable-tasks) —— 工具
  执行中途崩溃也能恢复的 Agent。
