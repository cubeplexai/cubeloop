---
title: SQLite Checkpointing
description: "Use SQLiteCheckpointer for lightweight single-process agent state persistence in CubeLoop."
---

# SQLite Checkpointing

`SQLiteCheckpointer` is the lightweight persistence backend: a single
local file, no server, append-only message log. It's the default
choice for laptops, single-process apps, desktop tools, and any
environment where one Python process owns the conversation.

Install the extra:

```bash
pip install "cubeloop[sqlite]"
```

This pulls in `aiosqlite`.

## Basic usage

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
        await agent.prompt("Remember: my favourite colour is teal.")

        # Restart the script later — the file persists.
        await agent.prompt("What did I say my favourite colour was?")
        # → "You said teal."


asyncio.run(main())
```

Two things to internalise:

1. **`thread_id`** is your conversation identifier — usually a user id
   or a session id. Two agents on the same `thread_id` share the same
   history.
2. **`async with SQLiteCheckpointer(...)`** is required: the context
   manager opens the connection and runs the one-time table creation
   on `__aenter__`. Using it without the context manager will raise
   `AssertionError`.

## What gets persisted

`SQLiteCheckpointer` writes two tables on first use:

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

- Every `UserMessage`, `AssistantMessage`, and `ToolResultMessage`
  becomes one row in `messages`. Pydantic `model_dump()` is used for
  the JSON payload.
- The `extra` dict on `AgentContext` is persisted into `thread_extra`
  after a turn that produced tool results, on HITL suspension
  (`agent_suspended`), and at `agent_end`. Middleware that wants
  thread-scoped state should write into `context.extra`.

This schema is append-only. CubeLoop never updates or deletes rows.

## HITL tables

When the [HITL](../hitl/overview) module is in use, additional tables are
created automatically on `__aenter__`:

```sql
CREATE TABLE IF NOT EXISTS thread_pending_request (
    thread_id TEXT PRIMARY KEY,
    request_json TEXT NOT NULL,
    run_id TEXT,
    created_at REAL NOT NULL DEFAULT (julianday('now'))
);

CREATE TABLE IF NOT EXISTS thread_hitl_answers (
    thread_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    answer_json TEXT NOT NULL,
    answered_at REAL NOT NULL DEFAULT (julianday('now')),
    PRIMARY KEY (thread_id, run_id, question_id)
);
```

`thread_pending_request` stores the single visible HITL request for a
thread. `thread_hitl_answers` stores already-answered prompts so a
parallel approval batch can collect approvals one by one and then
execute the batch only after every gate is satisfied.

No manual migration is needed — `CREATE TABLE IF NOT EXISTS` is
idempotent.

## When CubeLoop reads

On the **first** `prompt()` after instantiation, CubeLoop calls
`load(thread_id)`. If the thread exists, history is restored into
`agent.state.messages` and `extra` is restored into the agent's
private `_extra` dict.

Subsequent `prompt()` calls don't re-read — the in-memory state is
authoritative.

This means: **don't share one `Agent` instance across processes**.
Process A's in-memory state will diverge from Process B's writes.

## Multi-thread isolation

```python
async with SQLiteCheckpointer("agent.db") as cp:
    alice = Agent(model=…, checkpointer=cp, thread_id="alice")
    bob   = Agent(model=…, checkpointer=cp, thread_id="bob")
    # Each call only loads/appends its own thread.
```

You can pool one checkpointer across many users; the `thread_id`
isolates them.

## Concurrency model

The checkpointer uses an `asyncio.Lock` around every read and write.
SQLite itself can be written from multiple processes, but CubeLoop's
expectation is that a single agent instance owns a thread. If you
have multiple processes writing the same `agent.db`:

- Reads are safe.
- Concurrent writes to **different** threads are safe.
- Concurrent writes to the **same** thread will interleave and the
  resulting history won't make sense — you'll see two assistant
  messages back-to-back, etc.

If you need multi-process writers on shared threads, jump to
[Postgres](./postgres) which uses an advisory lock per thread.

## Conditional pending cleanup

`clear_pending_request_if_matches(thread_id, question_id=..., run_id=...)`
deletes pending state only when both stored identifiers still match. SQLite
performs the comparison and delete in one statement, while the checkpointer's
writer lock serializes it with other writes in the same process.

## Forks

`SQLiteCheckpointer` implements the v4 `snapshot` / `fork` /
`claim_run` / `mark_run_complete` / `load_pending` Protocol methods,
so it supports both `Agent.fork(...)` and `Agent.fork_once(...)`. See
the [Conversation Forking](../agents/forking) guide for the user-facing API.

## Schema v3 → v4 migration

Unlike Postgres and MySQL, SQLite's schema is managed by CubeLoop
itself: the v3→v4 upgrade (adding `run_id` to `messages` and
creating a `runs` table) runs automatically on `__aenter__`
the first time a v4 CubeLoop connects to a v3 file. No host action is
required.

Pre-feature messages keep `run_id = NULL` and remain readable; see
[Legacy data behaviour](../agents/forking#legacy-data-behaviour) for the
fork-eligibility rules on mixed threads.

## Where the file lives

Use an absolute path in production:

```python
SQLiteCheckpointer("/var/lib/myapp/agent.db")
```

Relative paths resolve against `os.getcwd()` at the moment of
`__aenter__`. The directory must exist; create it ahead of time.

## Backup and inspection

The file is a normal SQLite database. You can:

```bash
# Inspect history for a thread
sqlite3 agent.db "SELECT message_json FROM messages WHERE thread_id='user-42' ORDER BY id"

# Back up
cp agent.db agent.db.bak

# Vacuum to reclaim space (optional — file grows linearly with history)
sqlite3 agent.db "VACUUM"
```

## Common pitfalls

- **Used without `async with`** — `AssertionError: self._db is not
  None`. Always wrap in `async with`.
- **Two processes writing the same thread** — Interleaved history.
  Use Postgres or coordinate at the application layer.
- **WAL mode not enabled** — CubeLoop uses the default journal mode for
  portability. For a single-writer, many-reader app, enable WAL once
  via `sqlite3 agent.db "PRAGMA journal_mode=WAL"` for better read
  concurrency.
- **Forgetting to set `thread_id`** — Without it, the agent has no
  persistence binding. The checkpointer is silently ignored. Always
  pass both.
- **`CheckpointCorruptionError` on `load()`** — A persisted message row
  failed to deserialize (bad JSON, schema-invalid data, or an unknown
  role). The error's `row_ref` (e.g. `messages.id=42`) locates the bad
  row so you can inspect or repair it with plain SQL; `thread_id` and
  `__cause__` carry the rest of the context. CubeLoop never skips corrupt
  rows silently — dropping a message that carries `tool_calls` would
  leave the transcript in a state every provider rejects.

## See also

- [Postgres Checkpointing](./postgres) — for multi-instance deployments.
- [Custom Backends](./custom) — implement the protocol for Redis,
  DynamoDB, etc.
- [Recipes → Persistent Chat](../../recipes/persistent-chat) —
  end-to-end app with SQLite.
- [Recipes → Resumable Long Tasks](../../recipes/resumable-tasks) —
  agents that survive a process restart mid-tool.
