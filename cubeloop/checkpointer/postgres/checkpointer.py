"""PostgresCheckpointer — Checkpointer protocol against PostgreSQL.

Append-only message log + per-thread KV (extra). Uses asyncpg pool +
msgpack payload encoding. Schema version verified on context entry.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import asyncpg
import msgpack
from pydantic import TypeAdapter

from cubeloop.checkpointer.base import CheckpointData
from cubeloop.checkpointer.exceptions import (
    CheckpointCorruptionError,
    RunAlreadyClaimedError,
    RunAlreadyCompletedError,
    RunNotClaimedError,
    RunNotCompletedError,
    ThreadAlreadyExistsError,
    ThreadNotFoundError,
)
from cubeloop.checkpointer.postgres.exceptions import (
    CubeloopSchemaMismatch,
    CubeloopSchemaUninitialized,
)
from cubeloop.checkpointer.postgres.models import EXPECTED_SCHEMA_VERSION
from cubeloop.hitl.types import HitlRequest
from cubeloop.providers.base import (
    AssistantMessage,
    Message,
    ToolResultMessage,
    UserMessage,
)
from cubeloop.types import JsonObject, StructuredValue

_STRUCTURED_VALUE_ADAPTER: TypeAdapter[Any] = TypeAdapter(StructuredValue)


def _run_key(run_id: str | None) -> str:
    return run_id or ""


def _schema_mismatch_hint(actual: int, expected: int) -> str:
    """Build the operator-facing hint for a schema-version mismatch.

    Names the actual upgrade_vN_to_vM_op() helper(s) needed to close the gap
    between ``actual`` and ``expected``, rather than hardcoding one version
    transition, so the hint stays correct as EXPECTED_SCHEMA_VERSION grows.
    """
    if actual > expected:
        return "database schema is newer than this CubeLoop release."
    steps = ", ".join(f"upgrade_v{v}_to_v{v + 1}_op()" for v in range(actual, expected))
    return (
        "cubeloop was upgraded but host alembic is behind. "
        f"Generate a new alembic revision that calls {steps} + "
        "write_schema_version_op() (see "
        "cubeloop.checkpointer.postgres.alembic_helpers) and run "
        "`alembic upgrade head` against this database."
    )


_MISSING_VERSION_TABLE_HINT = (
    "data tables exist but schema_version is missing; cannot auto-classify. "
    "Restore cubepi_schema_version from the host's Alembic history; do not "
    "apply fresh CREATE TABLE statements on existing data."
)


def _serialize_structured_value(value: StructuredValue) -> str:
    return _STRUCTURED_VALUE_ADAPTER.dump_json(value).decode("utf-8")


def _deserialize_structured_value(value: Any) -> StructuredValue:
    if isinstance(value, str):  # pragma: no cover - codec-dependent
        return json.loads(value)
    return value


def _role_of(msg: Message) -> str:
    if isinstance(msg, UserMessage):
        return "user"
    if isinstance(msg, AssistantMessage):
        return "assistant"
    if isinstance(msg, ToolResultMessage):
        return "tool"
    raise TypeError(f"unknown Message type: {type(msg).__name__}")


_ROLE_TO_CLS: dict[str, type[Message]] = {
    "user": UserMessage,
    "assistant": AssistantMessage,
    "tool": ToolResultMessage,
}


def _deserialize_row(thread_id: str, r: Any) -> Message:
    """Deserialize one cubepi_messages row; corruption raises typed."""
    try:
        cls = _ROLE_TO_CLS.get(r["role"])
        if cls is None:
            raise ValueError(f"unknown role in DB: {r['role']!r}")
        data = msgpack.unpackb(bytes(r["payload"]), raw=False)
        # The DB metadata column is the source of truth for Message.metadata.
        # (payload also contains it, but column is the canonical view for querying.)
        raw_meta = r["metadata"]
        data["metadata"] = (
            json.loads(raw_meta) if isinstance(raw_meta, str) else (raw_meta or {})
        )
        return cls.model_validate(data)
    except Exception as exc:
        raise CheckpointCorruptionError(
            thread_id=thread_id,
            backend="postgres",
            row_ref=f"cubepi_messages.seq={r['seq']}",
            cause=exc,
        ) from exc


class PostgresCheckpointer:
    """Checkpointer backed by PostgreSQL.

    Usage:
        cp = PostgresCheckpointer(dsn="postgresql://...")
        async with cp:
            await cp.append(thread_id, [msg1, msg2])
            data = await cp.load(thread_id)
            await cp.save_extra(thread_id, {"k": "v"})

    Raises CubepiSchemaUninitialized / CubepiSchemaMismatch at __aenter__
    if the DB schema isn't compatible with this cubepi version.
    """

    def __init__(
        self,
        dsn: str,
        *,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
    ) -> None:
        self._dsn = dsn
        self._min = min_pool_size
        self._max = max_pool_size
        self._pool: asyncpg.Pool | None = None

    async def __aenter__(self) -> "PostgresCheckpointer":
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min,
            max_size=self._max,
        )
        await self._verify_schema()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _verify_schema(self) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            try:
                row = await conn.fetchrow(
                    "SELECT version FROM cubepi_schema_version LIMIT 1"
                )
            except asyncpg.UndefinedTableError:
                has_data = await conn.fetchval(
                    "SELECT to_regclass('cubepi_threads') IS NOT NULL"
                )
                if has_data:
                    raise CubeloopSchemaMismatch(
                        expected=EXPECTED_SCHEMA_VERSION,
                        actual=0,
                        hint=_MISSING_VERSION_TABLE_HINT,
                    )
                raise CubeloopSchemaUninitialized(
                    "cubepi tables not found. Run host application's alembic upgrade."
                )
            if row is None:
                raise CubeloopSchemaUninitialized(
                    "cubepi_schema_version table is empty. Host alembic "
                    "migration must INSERT the current version "
                    "(use write_schema_version_op())."
                )
            if row["version"] != EXPECTED_SCHEMA_VERSION:
                raise CubeloopSchemaMismatch(
                    expected=EXPECTED_SCHEMA_VERSION,
                    actual=row["version"],
                    hint=_schema_mismatch_hint(row["version"], EXPECTED_SCHEMA_VERSION),
                )

    async def load(self, thread_id: str) -> CheckpointData | None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            msg_rows = await conn.fetch(
                "SELECT seq, role, metadata, payload FROM cubepi_messages "
                "WHERE thread_id = $1 ORDER BY seq",
                thread_id,
            )
            extra_row = await conn.fetchrow(
                "SELECT extra, parent_thread_id FROM cubepi_threads "
                "WHERE thread_id = $1",
                thread_id,
            )

        if not msg_rows and extra_row is None:
            return None

        messages: list[Message] = [_deserialize_row(thread_id, r) for r in msg_rows]

        parent_thread_id: str | None = None
        if extra_row is not None:
            raw_extra = extra_row["extra"]
            extra = (
                json.loads(raw_extra)
                if isinstance(raw_extra, str)
                else (raw_extra or {})
            )
            parent_thread_id = extra_row["parent_thread_id"]
        else:
            extra = {}

        return CheckpointData(
            messages=messages, extra=extra, parent_thread_id=parent_thread_id
        )

    async def append(self, thread_id: str, messages: list[Message]) -> None:
        if not messages:
            return
        assert self._pool is not None
        run_ids = {
            rid for m in messages if (rid := getattr(m, "run_id", None)) is not None
        }
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Per-thread advisory lock for monotonic seq allocation
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))",
                    thread_id,
                )
                # Lazy thread row creation
                await conn.execute(
                    "INSERT INTO cubepi_threads (thread_id) "
                    "VALUES ($1) ON CONFLICT DO NOTHING",
                    thread_id,
                )
                # Pre-flight: reject append on any completed run_id.
                if run_ids:
                    done_rows = await conn.fetch(
                        "SELECT run_id FROM cubepi_runs "
                        "WHERE thread_id = $1 AND run_id = ANY($2::text[]) "
                        "AND completed_at IS NOT NULL",
                        thread_id,
                        list(run_ids),
                    )
                    if done_rows:
                        bad = ", ".join(r["run_id"] for r in done_rows)
                        raise RunAlreadyCompletedError(
                            f"append on completed run thread={thread_id} runs={bad}"
                        )
                last_seq = (
                    await conn.fetchval(
                        "SELECT COALESCE(MAX(seq), 0) FROM cubepi_messages "
                        "WHERE thread_id = $1",
                        thread_id,
                    )
                    or 0
                )

                rows = []
                for i, m in enumerate(messages):
                    seq = last_seq + i + 1
                    payload = msgpack.packb(
                        m.model_dump(mode="json"), use_bin_type=True
                    )
                    rows.append(
                        (
                            thread_id,
                            seq,
                            _role_of(m),
                            json.dumps(m.metadata),
                            payload,
                            getattr(m, "run_id", None),
                        )
                    )
                await conn.executemany(
                    "INSERT INTO cubepi_messages "
                    "(thread_id, seq, role, metadata, payload, run_id) "
                    "VALUES ($1, $2, $3, $4, $5, $6)",
                    rows,
                )

    async def claim_run(self, thread_id: str, run_id: str) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))",
                    thread_id,
                )
                # Lazy thread row creation (claim may precede any append).
                await conn.execute(
                    "INSERT INTO cubepi_threads (thread_id) "
                    "VALUES ($1) ON CONFLICT (thread_id) DO NOTHING",
                    thread_id,
                )
                # Pre-check under the advisory lock so a duplicate doesn't
                # raise UniqueViolation (which would abort the txn before
                # we could distinguish in-flight vs completed).
                row = await conn.fetchrow(
                    "SELECT completed_at FROM cubepi_runs "
                    "WHERE thread_id = $1 AND run_id = $2",
                    thread_id,
                    run_id,
                )
                if row is not None:
                    if row["completed_at"] is not None:
                        raise RunAlreadyCompletedError(
                            f"thread={thread_id} run={run_id} already completed"
                        )
                    raise RunAlreadyClaimedError(
                        f"thread={thread_id} run={run_id} in flight"
                    )
                await conn.execute(
                    "INSERT INTO cubepi_runs (thread_id, run_id) VALUES ($1, $2)",
                    thread_id,
                    run_id,
                )

    async def mark_run_complete(self, thread_id: str, run_id: str) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))",
                    thread_id,
                )
                row = await conn.fetchrow(
                    "SELECT completed_at FROM cubepi_runs "
                    "WHERE thread_id = $1 AND run_id = $2",
                    thread_id,
                    run_id,
                )
                if row is None:
                    raise RunNotClaimedError(
                        f"thread={thread_id} run={run_id} has no claim row"
                    )
                if row["completed_at"] is not None:
                    return  # idempotent success
                next_seq = await conn.fetchval(
                    "SELECT COALESCE(MAX(completion_seq), 0) + 1 "
                    "FROM cubepi_runs WHERE thread_id = $1 "
                    "AND completion_seq IS NOT NULL",
                    thread_id,
                )
                await conn.execute(
                    "UPDATE cubepi_runs SET completed_at = now(), "
                    "completion_seq = $3 "
                    "WHERE thread_id = $1 AND run_id = $2",
                    thread_id,
                    run_id,
                    next_seq,
                )

    async def load_pending(
        self, thread_id: str
    ) -> tuple[HitlRequest, str | None] | None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT pending_request, run_id FROM cubepi_threads "
                "WHERE thread_id = $1",
                thread_id,
            )
        if row is None or row["pending_request"] is None:
            return None
        raw = row["pending_request"]
        if isinstance(raw, str):  # pragma: no cover — codec-dependent
            req = HitlRequest.model_validate_json(raw)
        else:
            req = HitlRequest.model_validate(raw)
        return req, row["run_id"]

    async def snapshot(self, thread_id: str, *, after_run_id: str) -> list[Message]:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            # Source thread must exist.
            t_row = await conn.fetchrow(
                "SELECT 1 FROM cubepi_threads WHERE thread_id = $1",
                thread_id,
            )
            if t_row is None:
                raise ThreadNotFoundError(f"thread={thread_id}")
            cutoff = await conn.fetchval(
                "SELECT completion_seq FROM cubepi_runs "
                "WHERE thread_id = $1 AND run_id = $2",
                thread_id,
                after_run_id,
            )
            if cutoff is None:
                raise RunNotCompletedError(
                    f"thread={thread_id} run={after_run_id} not completed"
                )
            rows = await conn.fetch(
                "SELECT seq, role, metadata, payload FROM cubepi_messages "
                "WHERE thread_id = $1 AND ("
                "  run_id IS NULL OR run_id IN ("
                "    SELECT run_id FROM cubepi_runs "
                "    WHERE thread_id = $1 "
                "    AND completion_seq IS NOT NULL "
                "    AND completion_seq <= $2"
                "  )"
                ") ORDER BY seq",
                thread_id,
                cutoff,
            )
        return [_deserialize_row(thread_id, r) for r in rows]

    async def fork(
        self,
        src_thread_id: str,
        new_thread_id: str,
        *,
        after_run_id: str,
        metadata: JsonObject | None = None,
    ) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Serialize concurrent forks/appends on src by taking the
                # src thread's per-thread advisory lock.
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))",
                    src_thread_id,
                )
                # Source must exist (threads row is created lazily by append
                # / claim_run / save_extra / save_pending_request — so its
                # presence is the canonical "thread exists" gate).
                t_row = await conn.fetchrow(
                    "SELECT 1 FROM cubepi_threads WHERE thread_id = $1",
                    src_thread_id,
                )
                if t_row is None:
                    raise ThreadNotFoundError(f"thread={src_thread_id}")
                # Destination must not exist.
                dst_row = await conn.fetchrow(
                    "SELECT 1 FROM cubepi_threads WHERE thread_id = $1",
                    new_thread_id,
                )
                if dst_row is not None:
                    raise ThreadAlreadyExistsError(f"thread={new_thread_id}")
                # Cutoff: after_run_id must be completed on src.
                cutoff = await conn.fetchval(
                    "SELECT completion_seq FROM cubepi_runs "
                    "WHERE thread_id = $1 AND run_id = $2",
                    src_thread_id,
                    after_run_id,
                )
                if cutoff is None:
                    raise RunNotCompletedError(
                        f"thread={src_thread_id} run={after_run_id} not completed"
                    )
                # Build merged extra (carry parent's extra + fork metadata).
                src_extra_row = await conn.fetchrow(
                    "SELECT extra FROM cubepi_threads WHERE thread_id = $1",
                    src_thread_id,
                )
                raw_extra = (
                    src_extra_row["extra"] if src_extra_row is not None else None
                )
                if isinstance(raw_extra, str):
                    base_extra = json.loads(raw_extra)
                else:
                    base_extra = dict(raw_extra) if raw_extra else {}
                if metadata is not None:
                    base_extra["fork"] = json.loads(json.dumps(metadata))
                # INSERT destination threads row first to satisfy the
                # cubepi_messages / cubepi_runs FKs on subsequent copies.
                await conn.execute(
                    "INSERT INTO cubepi_threads "
                    "(thread_id, parent_thread_id, forked_at_seq, extra) "
                    "VALUES ($1, $2, $3, $4::jsonb)",
                    new_thread_id,
                    src_thread_id,
                    cutoff,
                    json.dumps(base_extra),
                )
                # Copy messages whose run is NULL (legacy prefix) or completed
                # at or before cutoff. Preserve seq from the source.
                await conn.execute(
                    "INSERT INTO cubepi_messages "
                    "(thread_id, seq, role, metadata, payload, run_id) "
                    "SELECT $1, seq, role, metadata, payload, run_id "
                    "FROM cubepi_messages "
                    "WHERE thread_id = $2 AND ("
                    "  run_id IS NULL OR run_id IN ("
                    "    SELECT run_id FROM cubepi_runs "
                    "    WHERE thread_id = $2 "
                    "    AND completion_seq IS NOT NULL "
                    "    AND completion_seq <= $3"
                    "  )"
                    ") ORDER BY seq",
                    new_thread_id,
                    src_thread_id,
                    cutoff,
                )
                # Copy completed runs satisfying the cutoff.
                await conn.execute(
                    "INSERT INTO cubepi_runs "
                    "(thread_id, run_id, claimed_at, completed_at, completion_seq) "
                    "SELECT $1, run_id, claimed_at, completed_at, completion_seq "
                    "FROM cubepi_runs "
                    "WHERE thread_id = $2 "
                    "AND completion_seq IS NOT NULL "
                    "AND completion_seq <= $3",
                    new_thread_id,
                    src_thread_id,
                    cutoff,
                )

    async def save_extra(self, thread_id: str, extra: JsonObject) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO cubepi_threads (thread_id, extra, updated_at) "
                "VALUES ($1, $2::jsonb, now()) "
                "ON CONFLICT (thread_id) DO UPDATE "
                "SET extra = cubepi_threads.extra || EXCLUDED.extra, "
                "    updated_at = now()",
                thread_id,
                json.dumps(extra),
            )

    async def save_pending_request(
        self,
        thread_id: str,
        request: HitlRequest | None,
        *,
        run_id: str | None = None,
    ) -> None:
        """Persist a pending HITL request and its owning run_id.

        When ``request is None``, both ``pending_request`` and ``run_id``
        are cleared; the ``run_id`` kwarg is ignored in that case (the
        pending row's run_id is always cleared alongside the pending).

        pending and run_id are set in ONE UPDATE; the surrounding
        transaction also covers the lazy thread INSERT.
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Ensure thread row exists (lazy creation matches save_extra path).
                await conn.execute(
                    "INSERT INTO cubepi_threads (thread_id) "
                    "VALUES ($1) ON CONFLICT DO NOTHING",
                    thread_id,
                )
                if request is None:
                    await conn.execute(
                        "UPDATE cubepi_threads "
                        "SET pending_request = NULL, run_id = NULL, "
                        "updated_at = now() WHERE thread_id = $1",
                        thread_id,
                    )
                else:
                    payload = request.model_dump_json()
                    await conn.execute(
                        "UPDATE cubepi_threads "
                        "SET pending_request = $2::jsonb, run_id = $3, "
                        "updated_at = now() WHERE thread_id = $1",
                        thread_id,
                        payload,
                        run_id,
                    )

    async def load_pending_request(self, thread_id: str) -> HitlRequest | None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT pending_request FROM cubepi_threads WHERE thread_id = $1",
                thread_id,
            )
        if row is None or row["pending_request"] is None:
            return None
        raw = row["pending_request"]
        # asyncpg returns JSONB as already-parsed dict OR str depending on codec config.
        if isinstance(raw, str):  # pragma: no cover — codec-dependent
            return HitlRequest.model_validate_json(raw)
        return HitlRequest.model_validate(raw)

    async def clear_pending_request_if_matches(
        self,
        thread_id: str,
        *,
        question_id: str,
        run_id: str,
    ) -> bool:
        """Clear pending state only while it still names the question and run.

        The predicate and mutation run in one PostgreSQL ``UPDATE``. If a
        concurrent writer has already checkpointed a replacement request,
        PostgreSQL re-evaluates the predicate after acquiring the row lock and
        leaves that newer request intact.
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE cubepi_threads "
                "SET pending_request = NULL, run_id = NULL, updated_at = now() "
                "WHERE thread_id = $1 "
                "AND pending_request->>'question_id' = $2 "
                "AND run_id = $3",
                thread_id,
                question_id,
                run_id,
            )
        return result == "UPDATE 1"

    async def load_pending_run_id(self, thread_id: str) -> str | None:
        """Return the run_id of the currently pending HITL request.

        Filters on ``pending_request IS NOT NULL`` so the result reflects
        a real pending, not a leftover run_id from a cleared row. Returns
        None when: the thread is unknown, has no pending request, or was
        written by a pre-v3 host (legacy rows have run_id NULL).
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT run_id FROM cubepi_threads "
                "WHERE thread_id = $1 AND pending_request IS NOT NULL",
                thread_id,
            )
        return row["run_id"] if row is not None else None

    async def save_hitl_answer(
        self,
        thread_id: str,
        question_id: str,
        answer: StructuredValue,
        *,
        run_id: str | None = None,
    ) -> None:
        assert self._pool is not None
        payload = _serialize_structured_value(answer)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO cubepi_threads (thread_id) "
                    "VALUES ($1) ON CONFLICT DO NOTHING",
                    thread_id,
                )
                await conn.execute(
                    "INSERT INTO cubepi_hitl_answers "
                    "(thread_id, run_id, question_id, answer) "
                    "VALUES ($1, $2, $3, $4::jsonb) "
                    "ON CONFLICT (thread_id, run_id, question_id) DO UPDATE "
                    "SET answer = EXCLUDED.answer, answered_at = now()",
                    thread_id,
                    _run_key(run_id),
                    question_id,
                    payload,
                )

    async def load_hitl_answer(
        self,
        thread_id: str,
        question_id: str,
        *,
        run_id: str | None = None,
    ) -> StructuredValue | None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT answer FROM cubepi_hitl_answers "
                "WHERE thread_id = $1 AND run_id = $2 AND question_id = $3",
                thread_id,
                _run_key(run_id),
                question_id,
            )
        if row is None:
            return None
        return _deserialize_structured_value(row["answer"])

    async def clear_hitl_answers(
        self,
        thread_id: str,
        question_ids: Iterable[str] | None = None,
        *,
        run_id: str | None = None,
    ) -> None:
        assert self._pool is not None
        run_key = _run_key(run_id)
        async with self._pool.acquire() as conn:
            if question_ids is None:
                await conn.execute(
                    "DELETE FROM cubepi_hitl_answers "
                    "WHERE thread_id = $1 AND run_id = $2",
                    thread_id,
                    run_key,
                )
                return
            qids = list(dict.fromkeys(question_ids))
            if not qids:
                return
            await conn.execute(
                "DELETE FROM cubepi_hitl_answers "
                "WHERE thread_id = $1 AND run_id = $2 "
                "AND question_id = ANY($3::text[])",
                thread_id,
                run_key,
                qids,
            )
