---
title: 会话分叉
description: "在已完成 run 的边界处分叉会话 — 持久化分支、一次性探针，以及确保安全的 HITL 绑定规则。"
---

# 会话分叉

**分叉**在一个已完成 run 的边界处创建会话分支。CubeLoop 提供两种方式：

- **`Agent.fork(...)`** — *持久化*分叉。将源线程中截止指定 `run_id`
  的消息复制到全新线程，后续可以继续与新线程对话。
- **`Agent.fork_once(...)`** — *临时性*一次性探针。在内存中加载快照，
  执行一次 prompt，返回结果，不向源线程写入任何内容。

两个 API 都以 `run_id` 为基准 —— `run_id` 是一次
`prompt() → 最终 assistant 消息` 周期的稳定标识符。一次 prompt 期间产生
的所有消息都携带同一个 `run_id`，因此"run R 之后的边界"是明确且可复现的。

## cubebox 复制按钮 UX

最典型的使用场景是 cubebox 的"分支此回复"按钮：在 UI 中每条 assistant
消息下方都有一个小控件，点击后从该消息上方将会话分叉到全新线程。用户借此
探索"如果 assistant 给出不同回答会怎样？"，而不影响原线程。

在这一流程中：

- host 生成新线程的 id（cubebox 使用 uuid7），
- 调用 `Agent.fork(src, new, after_run_id=...)` 创建分支，
- 在新线程上打开一个新的 agent。

以下功能 —— 显式 `run_id`、`active_run_id`、fork API 以及 HITL 绑定规则
—— 正是该 UX 所需要的。

## `Agent.prompt(run_id=...)` — 接受或生成

`Agent.prompt()` 现在会返回本次调用使用的 `run_id`：

```python
run_id = await agent.prompt("hello")
print(run_id)  # → "8c0b…" — 服务端生成
```

需要控制 id 的 host（cubebox、多机中继，或任何需要在调用完成前就拿到 id
的场景）可以自行传入：

```python
import uuid_extensions   # 或任意 uuid7 来源

my_run_id = uuid_extensions.uuid7str()
run_id = await agent.prompt("hello", run_id=my_run_id)
assert run_id == my_run_id
```

run 进行中，id 可通过 `Agent.state.active_run_id` 获取：

```python
async def watch(agent: Agent) -> None:
    while agent.state.is_streaming:
        print("running:", agent.state.active_run_id)
        await asyncio.sleep(0.1)
```

`active_run_id` 在两次 run 之间为 `None`；在 `prompt()` 入口处被赋值，
正常退出时被清除。若 prompt 中途失败，它保持设置状态，以便 `respond()`
在 HITL 挂起或进程重启后恢复同一 run。

## `Agent.fork(...)` — 持久化分支

```python
await agent.fork(
    src_thread_id="conv_123",
    new_thread_id="conv_456",
    after_run_id="R1",
    metadata={"label": "分支实验"},
)
```

执行步骤：

1. **复制消息。** `conv_123` 中所有 `run_id` 属于截止 `R1`（含）的
   *已完成* run 的消息，会按原样追加到 `conv_456`（分配新 `seq` 编号）。
   `R1` 之后的待处理或已中止 run 不包含在内。
2. **记录血缘关系。** 新线程行写入
   `parent_thread_id = "conv_123"` 以及等于最后一条复制消息源 seq 的
   `forked_at_seq`，便于日后追溯父子关系。
3. **写入分叉元数据。** `metadata` 以 `extra["fork"]` 写入新线程
   （按 `save_extra` 语义合并 —— 已有 key 保留）。

`fork` 返回后，`conv_456` 作为与 `conv_123` 共享截止 R1 历史的新线程
独立存在；之后在任一线程上继续的 prompt 互不影响。源线程不受改动。

`fork` 要求 checkpointer 实现 v4 Protocol 方法（`claim_run`、
`mark_run_complete`、`fork`）；在仅实现 v3 的后端上会抛出 `CheckpointerError`。

## `Agent.fork_once(...)` — 临时性一次探针

```python
result = await agent.fork_once(
    src_thread_id="conv_123",
    message="如果你当时说了'是'会怎样？",
    after_run_id="R1",
)

print(result.text)         # 最终 assistant 文本
print(result.stop_reason)  # "stop" | "max_tokens" | "error" | ...
for m in result.messages:  # 仅本次探针新增的消息（不含历史前缀）
    ...
```

`fork_once` 返回 `ForkOnceResult` dataclass：

```python
@dataclass(frozen=True)
class ForkOnceResult:
    text: str               # 最终 assistant 文本内容合并为一个字符串
    messages: list[Message] # 探针期间产生的消息（不含历史前缀）
    stop_reason: str        # 最终 assistant stop_reason
```

内部实现：构建一个以源线程截止 `R1` 的快照为初始状态的临时 `Agent`，
对其执行一次 prompt，调用返回后丢弃所有内容。探针获得自己的新 `run_id`
（无需传入）。

### 隔离契约

`fork_once` 仅在 **checkpointer 层**提供隔离：

- 探针的任何行为都不会写入源线程、新线程或任何线程 —— 临时 agent 没有
  `thread_id`。
- **工具副作用不隔离。** 若工具会发邮件、调 HTTP API 或写文件，在
  `fork_once` 期间同样会执行。请确保工具在探针中调用是安全的，或对不安全
  的工具避免使用 `fork_once`。
- **禁止使用 HITL。** 如果 agent 上的任何工具或 middleware 携带
  `HitlBinding`，`fork_once` 会立即抛出 `RuntimeError`。请为临时探针
  单独构建 agent（不含 `ask_user_tool` / `ApprovalPolicyMiddleware`）。

## HITL 绑定要求

当配合 **checkpointed** HITL channel 使用 `ask_user_tool` 或
`ApprovalPolicyMiddleware` 时，channel 必须绑定与传给 `prompt()` 的相同
`run_id`：

```python
import uuid
from cubeloop import Agent
from cubeloop.hitl import CheckpointedChannel, ask_user_tool

run_id = uuid.uuid4().hex
channel = CheckpointedChannel(checkpointer=cp, thread_id="conv_123", run_id=run_id)

agent = Agent(
    model=provider.model("claude-sonnet-4-6"),
    checkpointer=cp,
    thread_id="conv_123",
    tools=[ask_user_tool(channel)],
    channel=channel,
)

# run_id 必须传入 —— 与 channel 绑定的 id 一致
result_run_id = await agent.prompt("…", run_id=run_id)
```

`prompt()` 在入口处强制检查：

- 若存在 checkpointed HITL 元素但未传 `run_id`，抛出 `ValueError`（不自动
  生成 —— 必须显式指定）。
- 若传入的 `run_id` 与 channel 绑定的 id 不匹配，抛出 `ValueError` 并在
  消息中同时显示两个 id。

原因：HITL 请求在进程重启后依然持久存在，当你之后调用 `agent.respond(answer)`
时，框架需要知道要在哪个 `run_id` 下恢复。提前绑定可使其确定性。

In-memory（`InMemoryChannel`）HITL 没有此要求 —— 不涉及持久化，无需
恢复契约。

## Schema v3 → v4 迁移

分叉功能需要 v4 schema。各后端的升级路径：

- **Postgres** — 参见 [Postgres → Schema v3→v4](../checkpointing/postgres#schema-v3--v4-migration)。
- **MySQL** — 参见 [MySQL → Schema v3→v4](../checkpointing/mysql#schema-v3--v4-migration)。
- **SQLite** — 在 `__aenter__` 时自动迁移，无需手动操作。
- **Memory** — 无 schema，开箱即用。

## 历史数据行为 {#legacy-data-behaviour}

CubeLoop 能优雅处理此功能上线前的旧消息（`run_id` 列未填充，即
`run_id IS NULL`）：

- **混合线程** — 已有旧消息后又接收了升级后 `prompt()` 的线程，可从任意
  升级后的 `run_id` 处分叉。新线程中旧消息作为前缀携带，其 `run_id`
  在副本中仍为 `NULL`。
- **纯旧消息线程** — 没有升级后 run 的线程无 `run_id` 标记，`after_run_id=`
  无处指向。对此类线程调用 `fork` / `fork_once` 会抛出 `CheckpointerError`。
  要使旧线程可分叉，先发送一次升级后的 `prompt()`，再从其 run id 处分叉。

`Agent.prompt()` 对旧线程始终可用 —— 新 schema 对正常使用完全向后兼容。

## 已知限制：跨进程并发 prompt

若两个进程同时对同一 `thread_id` 驱动 `prompt()`，per-thread 行锁保证消息
行可线性化，但分叉捕获的*语义*快照可能遗漏来自交错的兄弟 run 的上下文。
具体而言：

- 进程 A 启动 run `Ra`，追加 `[user, assistant, ...]`。
- `Ra` 完成前，进程 B 启动 run `Rb`，追加自己的消息并完成。
- `fork(..., after_run_id=Ra)` 可能因交错而同时复制 Rb 的消息。

行数据是正确的；会话切片可能不符合预期。如需严格切片，请避免对同一线程
并发 prompt，或在应用层协调。单进程部署（cubebox 的默认模式）不受影响。

## 另请参阅

- [Postgres 检查点](../checkpointing/postgres) — 支持 v4 schema 的生产后端。
- [MySQL 检查点](../checkpointing/mysql) — 支持 v4 schema 的 MySQL 后端。
- [SQLite 检查点](../checkpointing/sqlite) — 支持自动迁移的单进程后端。
- [自定义后端](../checkpointing/custom) — Protocol 详情，包括分叉支持所需实现的
  `snapshot`、`fork`、`claim_run`、`mark_run_complete`、`load_pending` 方法。
- [HITL 概览](../hitl/overview) — channel/binding 机制。
