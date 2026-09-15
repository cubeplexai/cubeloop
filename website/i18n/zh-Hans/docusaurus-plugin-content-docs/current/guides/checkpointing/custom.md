---
title: 自定义后端
description: "使用 Checkpointer protocol 为 CubeLoop 实现自定义 checkpointer 后端。"
---

# 自定义 Checkpointing 后端

`Checkpointer` protocol 的核心历史记录部分包含三个 async 方法：

```python
class Checkpointer(Protocol):
    async def load(self, thread_id: str) -> CheckpointData | None: ...
    async def append(self, thread_id: str, messages: list[Message]) -> None: ...
    async def save_extra(self, thread_id: str, extra: dict[str, Any]) -> None: ...
```

这足以支持基本的消息和 extra 持久化。完整 protocol 还包括 run、fork 和
HITL 生命周期方法；宿主使用相应功能时也需要实现这些方法。其中
`clear_pending_request_if_matches(thread_id, question_id=..., run_id=...)`
必须以原子方式完成比较和清理，避免过期的恢复流程误删新的 pending request。
完整签名请查看 API 参考。

你可以为 Redis、DynamoDB、S3、文件系统、内存字典实现该 protocol——任何支持追加和列表操作的存储均可。

## agent 何时调用各方法

| 方法 | 调用时机 |
|---|---|
| `load(thread_id)` | agent 构造后第一次调用 `prompt()` 时，调用一次 |
| `append(thread_id, messages)` | 在 `message_end` 内，每次消息完成时调用 |
| `save_extra(thread_id, extra)` | 在 `agent_end` 时，携带当前的 `_extra` 字典调用 |

对未知的 thread，`load` 返回 `None`。若还没有该 thread，`append` 和 `save_extra` 不会做任何有意义的事；请在第一次调用时创建对应的记录行。

## 最小化 Redis 示例

```python
import json
from typing import Any
import redis.asyncio as aredis
from cubeloop.checkpointer.base import CheckpointData
from cubeloop.providers.base import AssistantMessage, Message, ToolResultMessage, UserMessage


_ROLE_TO_CLS: dict[str, type[Message]] = {
    "user": UserMessage,
    "assistant": AssistantMessage,
    "tool_result": ToolResultMessage,
}


class RedisCheckpointer:
    def __init__(self, redis_url: str, prefix: str = "cubeloop:") -> None:
        self._url = redis_url
        self._prefix = prefix
        self._r: aredis.Redis | None = None

    async def __aenter__(self):
        self._r = aredis.from_url(self._url)
        return self

    async def __aexit__(self, *args):
        if self._r is not None:
            await self._r.aclose()
            self._r = None

    def _msgs_key(self, thread_id: str) -> str:
        return f"{self._prefix}msgs:{thread_id}"

    def _extra_key(self, thread_id: str) -> str:
        return f"{self._prefix}extra:{thread_id}"

    async def load(self, thread_id: str) -> CheckpointData | None:
        raw_msgs = await self._r.lrange(self._msgs_key(thread_id), 0, -1)
        raw_extra = await self._r.get(self._extra_key(thread_id))

        if not raw_msgs and raw_extra is None:
            return None

        messages: list[Message] = []
        for item in raw_msgs:
            data = json.loads(item)
            cls = _ROLE_TO_CLS.get(data.get("role"))
            if cls is not None:
                messages.append(cls.model_validate(data))

        extra: dict[str, Any] = json.loads(raw_extra) if raw_extra else {}
        return CheckpointData(messages=messages, extra=extra)

    async def append(self, thread_id: str, messages: list[Message]) -> None:
        if not messages:
            return
        pipe = self._r.pipeline()
        for m in messages:
            pipe.rpush(self._msgs_key(thread_id), json.dumps(m.model_dump(mode="json")))
        await pipe.execute()

    async def save_extra(self, thread_id: str, extra: dict[str, Any]) -> None:
        # Merge-style: load existing, update, write back.
        raw = await self._r.get(self._extra_key(thread_id))
        existing = json.loads(raw) if raw else {}
        existing.update(extra)
        await self._r.set(self._extra_key(thread_id), json.dumps(existing))
```

用法：

```python
async with RedisCheckpointer("redis://localhost:6379") as cp:
    agent = Agent(model=…, checkpointer=cp, thread_id="user-42")
    await agent.prompt("hi")
```

## 需要遵守的不变量

1. **仅追加。** 不要修改过去的消息。agent 假定它追加的历史就是你在 `load` 中返回的内容。
2. **保持顺序。** `load` 按追加顺序返回消息。使用列表、排序键或序列列。
3. **`load` 幂等。** 对同一 thread 两次调用 `load` 应返回相同结果。（CubeLoop 只调用一次，但工具往往也需要调用。）
4. **`extra` 是合并语义。** 先调用 `save_extra({"a": 1})` 后调用 `save_extra({"b": 2})` 应得到 `{"a": 1, "b": 2}`，而非仅 `{"b": 2}`。agent 携带完整字典调用，但 middleware 会分多次写入。
5. **用 `model_validate` 重建消息。** 使用 role 判别符（`UserMessage` / `AssistantMessage` / `ToolResultMessage`）选择正确的类。

## 不使用 async context manager 的自定义后端

`Checkpointer` Protocol 不要求 `__aenter__` / `__aexit__`。内置 checkpointer 使用它是因为需要管理网络资源，但纯内存或本地文件后端可以省略：

```python
class FileCheckpointer:
    def __init__(self, dir_path: str) -> None:
        self._dir = pathlib.Path(dir_path)
        self._dir.mkdir(parents=True, exist_ok=True)

    async def load(self, thread_id: str) -> CheckpointData | None:
        ...
    async def append(self, thread_id: str, messages: list[Message]) -> None:
        ...
    async def save_extra(self, thread_id: str, extra: dict[str, Any]) -> None:
        ...

# Pass directly:
agent = Agent(model=…, checkpointer=FileCheckpointer("/tmp/cp"), thread_id="x")
```

## 测试你的后端

使用 `FauxProvider` 的即插即用测试模式：

```python
from cubeloop import Agent
from cubeloop.providers import FauxProvider, faux_assistant_message

async def test_roundtrip():
    cp = MyCheckpointer(…)
    provider = FauxProvider(provider_id="faux")
    provider.set_responses([faux_assistant_message("hello")])

    agent1 = Agent(model=provider.model("t"),
                   checkpointer=cp, thread_id="t1")
    agent1.subscribe(lambda e, s=None: None)
    await agent1.prompt("hi")

    # Fresh agent, same thread — should restore history.
    provider2 = FauxProvider(provider_id="faux")
    provider2.set_responses([faux_assistant_message("hello again")])
    agent2 = Agent(model=provider2.model("t"),
                   checkpointer=cp, thread_id="t1")
    agent2.subscribe(lambda e, s=None: None)
    await agent2.prompt("hi again")

    assert len(agent2.state.messages) == 4   # 2 user + 2 assistant
```

## 常见陷阱

- **修改返回的 `CheckpointData`** —— 要么在传入时深拷贝，要么在文档中说明 agent 拥有该列表的所有权。CubeLoop 内置实现会进行拷贝。
- **丢失 `metadata`** —— `model_dump(mode="json")` 会保留 `metadata`。若通过 `__dict__` 序列化则会丢失。
- **`save_extra` 合并的竞态问题** —— 读-改-写模式在并发写入时可能丢失数据。若有针对同一 thread 的并发写入者，请使用 SQL `JSONB ||` 或 Redis Lua 脚本。
- **忘记注册 tool result 的 role** —— 容易只映射 `"user"` 和 `"assistant"` 而忘记 `"tool_result"`。三者都需要。

## 另请参阅

- [`Checkpointer` Protocol API](../../api/cubeloop-checkpointer) —— 完整签名。
- [SQLiteCheckpointer 源码](https://github.com/cubeplexai/cubeloop/blob/main/cubeloop/checkpointer/sqlite.py) —— 完整参考实现。
- [PostgresCheckpointer 源码](https://github.com/cubeplexai/cubeloop/blob/main/cubeloop/checkpointer/postgres/checkpointer.py) —— 生产级参考实现。
