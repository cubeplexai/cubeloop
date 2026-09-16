---
title: Postgres Checkpointing
description: "Use PostgresCheckpointer for production-grade agent state persistence with asyncpg and advisory locks."
---

# Postgres Checkpointing

`PostgresCheckpointer` is the production-grade persistence backend.
It uses `asyncpg` with a connection pool, `msgpack` for payloads, and
a per-thread Postgres advisory lock so multiple processes can write
the same `thread_id` without trampling each other.

Install the extra:

```bash
pip install "cubeloop[postgres]"
```

This pulls in `asyncpg`, `sqlalchemy`, and `msgpack`.

## Basic usage

```python
import asyncio
from cubeloop import Agent
from cubeloop.checkpointer import PostgresCheckpointer
from cubeloop.providers.anthropic import AnthropicProvider


async def main():
    provider = AnthropicProvider(provider_id="anthropic", api_key="…")
    async with PostgresCheckpointer("postgresql://user:pass@host/dbname") as cp:
        agent = Agent(
            model=provider.model("claude-sonnet-4-6"),
            checkpointer=cp,
            thread_id="user-42",
        )
        await agent.prompt("hello")


asyncio.run(main())
```

The DSN is whatever `asyncpg.create_pool(...)` accepts. Pool sizing:

```python
async with PostgresCheckpointer(
    "postgresql://…",
    min_pool_size=2,
    max_pool_size=20,
) as cp:
    …
```

## Schema

The checkpointer expects `cubepi_threads`, `cubepi_messages`,
`cubepi_runs`, `cubepi_hitl_answers`, and `cubepi_schema_version`.
Unlike SQLite, CubeLoop
**does not create these for you** — it verifies on `__aenter__` that
they exist with the expected `schema_version`.

If they're missing, you get `CubeloopSchemaUninitialized`. If the
version doesn't match this CubeLoop release, you get
`CubeloopSchemaMismatch`.

The reason: a production database belongs to the host application's
migration system (Alembic, Atlas, …), not to a third-party library
that might fight your existing migrations.

### Bootstrapping via Alembic

CubeLoop exposes the SQLAlchemy `MetaData` so your migrations can adopt
the schema:

```python
# alembic/env.py
from cubeloop.checkpointer.postgres import cubeloop_metadata, EXPECTED_SCHEMA_VERSION

target_metadata = [my_app_metadata, cubeloop_metadata]
```

Then autogenerate a revision:

```bash
alembic revision --autogenerate -m "add cubeloop checkpointer"
```

Autogenerate emits the `CREATE TABLE`s from `cubeloop_metadata`, but
SQLAlchemy `MetaData` **cannot model two things CubeLoop needs**, so add
them to the generated migration by hand: the 64 hash partitions of
`cubepi_messages` (via `create_message_partitions_op()`), the 64
partitions of `cubepi_runs` (via `create_runs_partitions_op()`), and
the `cubepi_schema_version` row (via `write_schema_version_op()`). Use
the helpers:

```python
# In a migration's upgrade():
from cubeloop.checkpointer.postgres.alembic_helpers import (
    create_message_partitions_op,
    create_runs_partitions_op,
    write_schema_version_op,
)

def upgrade():
    op.create_table(...)                            # auto-generated from cubeloop_metadata
    op.execute(create_message_partitions_op())      # 64 hash partitions of cubepi_messages
    op.execute(create_runs_partitions_op())         # 64 hash partitions of cubepi_runs
    op.execute(write_schema_version_op())           # records historical schema v5
```

Both helpers return a SQL string — you pass them to `op.execute(...)`.
`write_schema_version_op()` is the immutable historical v5 writer. It is
idempotent, deleting other version rows before inserting 5. A future schema
version must ship and use a new version-specific writer; do not reuse this one.

## Data model

```
cubepi_threads
    thread_id (PK)
    parent_thread_id   -- for forks
    forked_at_seq      -- seq number at fork point
    extra              -- JSONB
    created_at / updated_at

cubepi_messages
    thread_id, seq     -- composite PK; partitioned by HASH(thread_id) into 64
    role               -- "user" | "assistant" | "tool"
    metadata           -- JSONB (indexed via GIN)
    payload            -- bytea (msgpack)
    created_at

cubepi_runs
    thread_id, run_id  -- composite PK
    claimed_at / completed_at
    completion_seq

cubepi_hitl_answers
    thread_id, run_id, question_id -- composite PK
    answer                         -- JSONB
    answered_at

cubepi_schema_version
    version (PK)
```

Important properties:

- **`(thread_id, seq)` is the message identity.** `seq` is monotonic
  per thread, allocated under a `pg_advisory_xact_lock(hashtext(thread_id))`.
  Two concurrent writers on the same thread serialize cleanly.
- **`payload` is msgpack-encoded `model.model_dump(mode="json")`.**
  CubeLoop reconstructs the Pydantic model on read.
- **`metadata` is JSONB, queryable.** The full message also has
  `metadata` inside the payload, but the column is the canonical view
  for SQL queries.
- **Tables are partitioned by `HASH(thread_id)` into 64 partitions.**
  Even distribution across partitions, no per-thread bottleneck.

## Concurrency

The advisory lock makes append-on-the-same-thread safe across
processes:

```python
# Process A and Process B both append to thread "user-42" at the same time.
# They serialize through pg_advisory_xact_lock and each gets a consecutive seq.
```

Reads (`load`) take no lock — they're snapshot-consistent within the
transaction.

The pool default of `min=1, max=10` is fine for most apps; bump
`max_pool_size` if you have many concurrent agents.

## `save_extra` semantics

`save_extra` does a JSONB merge, not a replace:

```sql
extra = cubepi_threads.extra || EXCLUDED.extra
```

So writing `{"foo": 1}` then `{"bar": 2}` leaves `{"foo": 1, "bar":
2}`. Middleware can safely write partial dicts without losing prior
keys.

## Conditional pending cleanup

Hosts that coordinate HITL resumes across workers can use
`clear_pending_request_if_matches(thread_id, question_id=..., run_id=...)`. It
clears the pending request and owning run ID only if both stored identifiers
still match. The comparison and clear are one PostgreSQL update, so a newer
request written by another worker is never erased by stale cleanup, even if it
reuses the same question ID in a different run. The same API is available on
all built-in checkpointers.

## Forks

`PostgresCheckpointer` implements the v4 `snapshot` / `fork` /
`claim_run` / `mark_run_complete` / `load_pending` Protocol methods,
so it supports both `Agent.fork(...)` and `Agent.fork_once(...)`. The
`parent_thread_id` and `forked_at_seq` columns on `cubepi_threads`
record fork lineage; `cubepi_runs` (the v4 partitioned table) tracks
per-run claim/completion state.

See the [Conversation Forking](../agents/forking) guide for the user-facing
API and semantics.

## Schema v3 → v4 migration

The fork feature bumps `EXPECTED_SCHEMA_VERSION` from 3 to 4. The
upgrade adds the `run_id` column + index to `cubepi_messages` and
creates a partitioned `cubepi_runs` parent table with its child
partitions. Use the provided alembic helper:

```python
# In a migration's upgrade():
from cubeloop.checkpointer.postgres.alembic_helpers import (
    upgrade_v3_to_v4_op,
    write_schema_version_op,
)

def upgrade():
    op.execute(upgrade_v3_to_v4_op())
    op.execute(write_schema_version_op())  # bumps cubepi_schema_version to 4
```

`upgrade_v3_to_v4_op()` is idempotent under repeated execution
(`IF NOT EXISTS` guards on every DDL statement).

Pre-feature messages keep `run_id = NULL` and remain readable; see
[Legacy data behaviour](../agents/forking#legacy-data-behaviour) for the
fork-eligibility rules on mixed threads.

## Schema v4 -> v5 migration

Durable HITL answer replay bumps `EXPECTED_SCHEMA_VERSION` from 4 to 5.
The upgrade creates `cubepi_hitl_answers`, which stores answered HITL
requests by `(thread_id, run_id, question_id)` until the suspended tool
cycle completes.

```python
# In a migration's upgrade():
from cubeloop.checkpointer.postgres.alembic_helpers import (
    upgrade_v4_to_v5_op,
    write_schema_version_op,
)

def upgrade():
    op.execute(upgrade_v4_to_v5_op())
    op.execute(write_schema_version_op())  # bumps cubepi_schema_version to 5
```

## CubeLoop rename

The package rename does not change the persistence schema: physical names stay
`cubepi_*` and `EXPECTED_SCHEMA_VERSION` stays 5. Existing 0.13.6 databases need
no migration. See the [migration guide](../../migration/from-cubepi).

## Common pitfalls

- **`CubeloopSchemaUninitialized`** — Your DB is empty or your
  migrations didn't run. Apply the host alembic upgrade first.
- **`CubeloopSchemaMismatch`** — You upgraded CubeLoop but didn't generate
  a new migration. Generate one, apply it, and CubeLoop will start.

  :::info Schema v2 (HITL)

  CubeLoop ≥ the HITL feature bumps `EXPECTED_SCHEMA_VERSION` from 1 to 2
  and adds a `pending_request JSONB NULL` column to `cubepi_threads`.
  Your host alembic upgrade must call
  `add_pending_request_column_op()` (from
  `cubeloop.checkpointer.postgres.alembic_helpers`) before bumping the
  schema_version row. See the [HITL guide](../hitl/overview) for the full
  cross-process flow.
  :::
- **Connection pool exhaustion under load** — Default `max_pool_size=10`.
  Bump it if your app's concurrent agent count is higher than that.
- **`asyncpg.exceptions.UndefinedTableError` outside `__aenter__`** —
  Means you're using the checkpointer outside of `async with`. The
  pool isn't connected yet. Wrap in the context manager.
- **Mixing host SQLAlchemy `MetaData`** — CubeLoop ships its own
  `MetaData` instance precisely so it can coexist with your app's
  models without colliding. Don't merge them into your global metadata
  — pass both to Alembic separately.
- **`CheckpointCorruptionError` on `load()`** — A persisted message row
  failed to deserialize (bad msgpack payload, schema-invalid data, or an
  unknown role). The error's `row_ref` (e.g. `cubepi_messages.seq=42`)
  locates the bad row for inspection or surgical repair; `thread_id` and
  `__cause__` carry the rest. CubeLoop never skips corrupt rows silently —
  dropping a message that carries `tool_calls` would leave the
  transcript in a state every provider rejects.

## See also

- [SQLite Checkpointing](./sqlite) — single-process alternative.
- [Custom Backends](./custom) — Protocol details.
- [Recipes → Postgres + FastAPI Service](../../recipes/postgres-fastapi)
  — a deployable HTTP-fronted agent.
- [Package README](https://github.com/cubeplexai/cubeloop/blob/main/cubeloop/checkpointer/postgres/README.md)
  — the full host-integration runbook next to the code.
