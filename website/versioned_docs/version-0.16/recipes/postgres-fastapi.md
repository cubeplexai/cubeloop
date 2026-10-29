---
title: Postgres + FastAPI Service
description: "Deploy a FastAPI-backed CubeLoop agent with PostgresCheckpointer for production."
---

# Recipe: Postgres + FastAPI Service

A production-shaped HTTP service that fronts a CubeLoop agent: FastAPI
for routing, server-sent events for streaming, a shared
`PostgresCheckpointer` for persistence, and `thread_id` derived from
the authenticated user.

**Time to run:** 30 minutes.
**Deps:** `cubeloop[postgres]`, `fastapi`, `uvicorn[standard]`,
`sse-starlette`, a running Postgres with the CubeLoop schema applied.

## Schema first

Before the service starts, run the CubeLoop schema migration. The
quickest way for this recipe:

```bash
# Throwaway / recipe: copy the v5 bootstrap from
# examples/checkpointing_postgres.py (cubepi_threads, cubepi_messages +
# 64 partitions, cubepi_runs + 64 partitions, cubepi_hitl_answers,
# cubepi_schema_version = 5).
```

Existing v5 databases need no migration for the CubeLoop package rename.

For a real deployment, generate this via Alembic — see
[Postgres Checkpointing → Bootstrapping via Alembic](../guides/checkpointing/postgres#bootstrapping-via-alembic).

## The service

```python title="service.py"
import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from cubeloop import Agent
from cubeloop.checkpointer import PostgresCheckpointer
from cubeloop.providers.anthropic import AnthropicProvider


# --- App lifecycle ------------------------------------------------------

_provider = AnthropicProvider(provider_id="anthropic", api_key=os.environ["ANTHROPIC_API_KEY"])
_checkpointer: PostgresCheckpointer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _checkpointer
    _checkpointer = await PostgresCheckpointer(
        os.environ["DATABASE_URL"],
        min_pool_size=2,
        max_pool_size=20,
    ).__aenter__()
    yield
    await _checkpointer.__aexit__(None, None, None)


app = FastAPI(lifespan=lifespan)


# --- Auth (stub — replace with your real auth) -------------------------

async def current_user_id() -> str:
    # In production: decode JWT, look up session, etc.
    return "demo-user"


# --- Routes -------------------------------------------------------------

class PromptBody(BaseModel):
    text: str


@app.post("/chat/{conversation_id}/messages")
async def post_message(
    conversation_id: str,
    body: PromptBody,
    user_id: str = Depends(current_user_id),
):
    thread_id = f"{user_id}:{conversation_id}"

    async def event_generator() -> AsyncIterator[dict]:
        agent = Agent(
            model=_provider.model("claude-sonnet-4-6"),
            system_prompt="You are a helpful assistant.",
            checkpointer=_checkpointer,
            thread_id=thread_id,
        )

        queue: asyncio.Queue = asyncio.Queue()
        agent.subscribe(lambda e, s=None: queue.put_nowait(e))

        async def run():
            try:
                await agent.prompt(body.text)
            finally:
                queue.put_nowait(None)   # sentinel

        task = asyncio.create_task(run())

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                # Emit a small subset to the client.
                if event.type == "message_update" and event.stream_event.type == "text_delta":
                    yield {"event": "delta", "data": event.stream_event.delta}
                elif event.type == "tool_execution_start":
                    yield {"event": "tool_start", "data": event.tool_name}
                elif event.type == "agent_end":
                    yield {"event": "done", "data": ""}
        finally:
            await task

    return EventSourceResponse(event_generator())


@app.get("/chat/{conversation_id}/history")
async def get_history(
    conversation_id: str,
    user_id: str = Depends(current_user_id),
):
    thread_id = f"{user_id}:{conversation_id}"
    data = await _checkpointer.load(thread_id)
    if data is None:
        return {"messages": []}
    return {
        "messages": [m.model_dump(mode="json") for m in data.messages],
    }
```

Run:

```bash
pip install "cubeloop[postgres]" fastapi "uvicorn[standard]" sse-starlette
export DATABASE_URL=postgresql://user:pass@localhost/cubeloop
export ANTHROPIC_API_KEY=sk-…
uvicorn service:app --reload --port 8000
```

Test:

```bash
curl -N -X POST http://localhost:8000/chat/conv1/messages \
  -H "content-type: application/json" \
  -d '{"text":"hi"}'
# event: delta
# data: Hello
# event: delta
# data: !
# event: done
```

## Design notes

- **One `PostgresCheckpointer` per process, shared across requests.**
  It holds a connection pool; opening one per request would defeat
  the pool.
- **One `Agent` per request.** Agents own per-conversation state
  (steering queues, listeners). Don't reuse them.
- **`thread_id = f"{user_id}:{conversation_id}"`** — user isolation by
  prefix. The agent reads/writes only its own thread.
- **SSE for streaming.** Each text delta goes to the client as a
  separate event. Tool starts get their own event type — clients can
  render a "thinking" indicator without rebuilding event handling.
- **No load balancer affinity needed.** Because state is in Postgres,
  any service instance can pick up any conversation.

## Concurrency on the same thread

If a user double-clicks send, two `POST` requests arrive simultaneously.
Both create an `Agent` bound to the same `thread_id`. The Postgres
advisory lock serialises their appends, but the **in-memory** states
diverge — the second request's agent might not see the first's
in-progress message in its `agent.state.messages`.

For most chat UIs this is fine (the client controls send timing).
If you need strict ordering, add an application-layer mutex
(`asyncio.Lock` keyed by `thread_id`) or queue.

## Production hardening checklist

- **Auth:** Replace `current_user_id()` with real JWT / session
  validation.
- **Rate limiting:** Add a [`RateLimitMiddleware`](../guides/middleware/examples#rate-limiting)
  to the agent constructor, keyed by `user_id`.
- **Cost tracking:** Subscribe to `agent_end`, sum `usage` on each
  `AssistantMessage`, write to a billing table.
- **Observability:** Use `on_response` to capture `anthropic-*` rate
  headers; export to Prometheus.
- **Backups:** Postgres native — `pg_dump`, point-in-time recovery.
- **Graceful shutdown:** uvicorn's lifespan handler closes the pool;
  add `signal.signal(SIGTERM, ...)` if you have other resources.

## Common pitfalls

- **CubeloopSchemaUninitialized at startup** — Your migrations didn't
  run. Apply the schema first.
- **Connection pool exhaustion** — Default `max_pool_size=10`. Bump
  it if your service has more concurrent agents than that.
- **SSE behind a load balancer** — Some LBs buffer SSE. Disable
  buffering (`X-Accel-Buffering: no` for nginx).
- **Long requests timing out** — Tool-heavy agents can run minutes.
  Set generous proxy timeouts and uvicorn `--timeout-keep-alive 600`.

## Run the example

A self-contained service template for this recipe is in the repository at
[`examples/postgres_fastapi.py`](https://github.com/cubeplexai/cubeloop/blob/main/examples/postgres_fastapi.py).

```bash
git clone https://github.com/cubeplexai/cubeloop && cd cubeloop
uv sync --extra postgres

export DATABASE_URL=postgresql://user:pass@localhost/cubeloop
export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY [+ OPENAI_BASE_URL]

uv run --with fastapi --with "uvicorn[standard]" --with sse-starlette \
  uvicorn examples.postgres_fastapi:app --reload --port 8000

# Test with curl:
curl -N -X POST http://localhost:8000/chat/conv1/messages \
  -H "content-type: application/json" \
  -d '{"text":"hi"}'
```

## See also

- [Postgres Checkpointing](../guides/checkpointing/postgres) — the
  backend in depth.
- [Persistent Chat](./persistent-chat) — the same flow with SQLite.
- [Multi-Provider Failover](./multi-provider-failover) — combine with
  this service for resilience.
