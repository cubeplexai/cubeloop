---
title: Custom Backends
description: "Implement a custom checkpointer backend for CubeLoop using the Checkpointer protocol."
---

# Custom Checkpointing Backends

The core history portion of the `Checkpointer` protocol is three async methods:

```python
class Checkpointer(Protocol):
    async def load(self, thread_id: str) -> CheckpointData | None: ...
    async def append(self, thread_id: str, messages: list[Message]) -> None: ...
    async def save_extra(self, thread_id: str, extra: dict[str, Any]) -> None: ...
```

That is enough for basic message and extra persistence. The full protocol also
contains run, fork, and HITL lifecycle methods; implement those when the host
uses the corresponding features. In particular,
`clear_pending_request_if_matches(thread_id, question_id=..., run_id=...)`
must compare and clear atomically so stale resume cleanup cannot erase a newer
pending request. See the API reference for the complete signatures.

You can implement the protocol for Redis, DynamoDB, S3, a filesystem, an
in-memory dict — anything that can append-and-list.

## When the agent calls each method

| Method | Called when |
|---|---|
| `load(thread_id)` | Once, on the first `prompt()` after agent construction |
| `append(thread_id, messages)` | Inside `message_end`, every time a message is finalised |
| `save_extra(thread_id, extra)` | At `agent_end`, with the current `_extra` dict |

`load` returns `None` for unknown threads. `append` and `save_extra`
do nothing useful if you don't already have a thread; create the row
on first call.

## A minimal Redis example

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

Usage:

```python
async with RedisCheckpointer("redis://localhost:6379") as cp:
    agent = Agent(model=…, checkpointer=cp, thread_id="user-42")
    await agent.prompt("hi")
```

## Invariants worth holding

1. **Append-only.** Don't update past messages. The agent assumes the
   history it appended is what you'll return on `load`.
2. **Order preserved.** `load` returns messages in the order they were
   appended. Use a list, a sorted key, or a sequence column.
3. **Idempotent re-`load`.** Calling `load` twice on the same thread
   should yield identical results. (CubeLoop calls it once, but tools
   often need to too.)
4. **`extra` is a merge.** `save_extra({"a": 1})` followed by
   `save_extra({"b": 2})` should leave `{"a": 1, "b": 2}` — not just
   `{"b": 2}`. The agent calls this with the full dict, but middleware
   composes its writes.
5. **Reconstruct messages with `model_validate`.** Use the role
   discriminator (`UserMessage` / `AssistantMessage` /
   `ToolResultMessage`) to pick the right class.

## Custom backend without async context manager

The `Checkpointer` Protocol doesn't require `__aenter__` /
`__aexit__`. Built-in checkpointers use it because they manage
network resources, but a pure in-memory or local-file backend can
skip it:

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

## Testing your backend

Drop-in test pattern using `FauxProvider`:

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

## Common pitfalls

- **Mutating returned `CheckpointData`** — Either deep-copy on the way
  in, or document that the agent owns the list. CubeLoop's built-ins
  copy.
- **Losing `metadata`** — `model_dump(mode="json")` preserves
  `metadata`. If you serialise via `__dict__` you'll drop it.
- **Race on `save_extra` merge** — A read-modify-write pattern can
  lose concurrent writes. Use a SQL `JSONB ||` or a Redis Lua script
  if you have concurrent writers per thread.
- **Forgetting to register the role for tool results** — Easy to map
  `"user"` and `"assistant"` and forget `"tool_result"`. All three are
  needed.

## See also

- [`Checkpointer` Protocol API](../../api/cubeloop-checkpointer) — full
  signature.
- [SQLiteCheckpointer source](https://github.com/cubeplexai/cubeloop/blob/main/cubeloop/checkpointer/sqlite.py)
  — a complete reference implementation.
- [PostgresCheckpointer source](https://github.com/cubeplexai/cubeloop/blob/main/cubeloop/checkpointer/postgres/checkpointer.py)
  — production-grade reference.
