---
title: Persistent Chat
description: "Build a persistent chat application with CubeLoop and SQLiteCheckpointer."
---

# Recipe: Persistent Chat (SQLite)

A REPL chat that survives restarts. Conversation history is kept in a
SQLite file; each user gets a `thread_id`.

**Time to run:** 5 minutes.
**Deps:** `cubeloop[sqlite]`, an `ANTHROPIC_API_KEY`.

## The script

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

Run:

```bash
pip install "cubeloop[sqlite]"
export ANTHROPIC_API_KEY=sk-…
python chat.py alice
# Have a chat, then Ctrl-D.

python chat.py alice
# History is restored. Ask "what did I just tell you?" — the model
# remembers the previous session.

python chat.py bob
# Different thread, clean slate.
```

## What's going on

- **First `prompt()` per process loads history.** CubeLoop checks the
  checkpointer once at the start of the first prompt, restores
  `agent.state.messages`, and proceeds.
- **Each `message_end` appends to the database.** No batching, no
  lossy buffering. If the process crashes mid-stream, the next run
  picks up at the last committed message.
- **The system prompt isn't persisted** — it's part of agent
  construction, not state. Keep it in your code (or in env config).

## Multi-user routing

In a web service, derive `thread_id` from the authenticated user:

```python
agent = Agent(
    model=model,
    checkpointer=cp,
    thread_id=f"user-{user_id}",
)
```

You can serve thousands of users from one `SQLiteCheckpointer` —
threads are isolated by `thread_id`.

## Pruning a thread

There's no built-in "delete thread" API by design — checkpointers are
append-only. To prune, drop rows by `thread_id` directly:

```bash
sqlite3 chat.db "DELETE FROM messages WHERE thread_id='alice'; DELETE FROM thread_extra WHERE thread_id='alice'"
```

Or implement a small admin tool that does the SQL on demand.

## Sliding the context window

After a long conversation, the model's context gets expensive. Add a
[`SlidingWindow`](../guides/middleware/examples#sliding-window-truncation)
middleware:

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

The DB keeps every message; the model only sees the last 40. The
user-visible history (e.g. for a chat UI rendering past turns) stays
complete.

## Switching to Postgres

Same code, different checkpointer:

```python
from cubeloop.checkpointer import PostgresCheckpointer

async with PostgresCheckpointer("postgresql://…") as cp:
    agent = Agent(model=…, checkpointer=cp, thread_id=…)
```

Postgres is the right choice for multi-instance services or many
concurrent users — see [Postgres + FastAPI](./postgres-fastapi).

## Common pitfalls

- **`async with` forgotten** — Without it, the SQLite connection is
  never opened. You get `AssertionError`. Wrap.
- **Multiple processes writing the same `thread_id`** — Interleaved
  history. One agent per thread, or move to Postgres.
- **`chat.db` in `/tmp`** — Some OSes wipe `/tmp` on reboot. Use
  `~/.local/share/myapp/chat.db` or similar for user data.

## Run the example

A self-contained, runnable version of this recipe is in the repository at
[`examples/persistent_chat.py`](https://github.com/cubeplexai/cubeloop/blob/main/examples/persistent_chat.py).

```bash
git clone https://github.com/cubeplexai/cubeloop && cd cubeloop
uv sync --extra sqlite

export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY [+ OPENAI_BASE_URL]
uv run python examples/persistent_chat.py alice
# Ctrl-D to quit, then re-run — history is restored.
uv run python examples/persistent_chat.py bob
# Different thread_id → clean slate.
```

## See also

- [Multi-turn Conversations](../guides/agents/multi-turn) — `steer`,
  `follow_up`, `resume`.
- [SQLite Checkpointing](../guides/checkpointing/sqlite) — the backend
  in detail.
- [Resumable Long Tasks](./resumable-tasks) — when crashes happen
  mid-tool, not just between turns.
