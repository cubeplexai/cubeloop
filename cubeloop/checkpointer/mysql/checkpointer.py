"""MySQLCheckpointer — Checkpointer protocol against MySQL.

Append-only message log + per-thread KV (extra). aiomysql pool + msgpack
payloads. Schema version verified on context entry. Mirrors
PostgresCheckpointer; see dev/specs/2026-05-27-mysql-checkpointer.md for the
list of deliberate MySQL divergences.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any
from urllib.parse import unquote, urlparse

import aiomysql
import msgpack
import pymysql
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
from cubeloop.checkpointer.mysql.exceptions import (
    CubeloopSchemaMismatch,
    CubeloopSchemaUninitialized,
)
from cubeloop.checkpointer.mysql.models import EXPECTED_SCHEMA_VERSION
from cubeloop.hitl.types import HitlRequest
from cubeloop.providers.base import (
    AssistantMessage,
    Message,
    ToolResultMessage,
    UserMessage,
)
from cubeloop.types import JsonObject, StructuredValue

_STRUCTURED_VALUE_ADAPTER: TypeAdapter[Any] = TypeAdapter(StructuredValue)

_ER_NO_SUCH_TABLE = 1146
_ER_BAD_FIELD_ERROR = 1054


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


def _deserialize_row(
    thread_id: str, seq: int, role: str, metadata: Any, payload: Any
) -> Message:
    """Deserialize one cubepi_messages row; corruption raises typed."""
    try:
        cls = _ROLE_TO_CLS.get(role)
        if cls is None:
            raise ValueError(f"unknown role in DB: {role!r}")
        data = msgpack.unpackb(bytes(payload), raw=False)
        data["metadata"] = _decode_json(metadata)
        return cls.model_validate(data)
    except Exception as exc:
        raise CheckpointCorruptionError(
            thread_id=thread_id,
            backend="mysql",
            row_ref=f"cubepi_messages.seq={seq}",
            cause=exc,
        ) from exc


def _parse_dsn(dsn: str) -> dict[str, Any]:
    """Parse a mysql:// URL into aiomysql.create_pool kwargs."""
    u = urlparse(dsn)
    db = u.path.lstrip("/")
    return {
        "host": u.hostname or "localhost",
        "port": u.port or 3306,
        "user": unquote(u.username) if u.username else "",
        "password": unquote(u.password) if u.password else "",
        "db": db,
    }


def _decode_json(value: Any) -> dict[str, Any]:
    """aiomysql returns JSON columns as str; tolerate already-parsed dicts."""
    if value is None:
        return {}
    if isinstance(value, str):
        return json.loads(value)
    return value


def _run_key(run_id: str | None) -> str:
    return run_id or ""


def _serialize_structured_value(value: StructuredValue) -> str:
    return _STRUCTURED_VALUE_ADAPTER.dump_json(value).decode("utf-8")


def _decode_json_value(value: Any) -> StructuredValue:
    if isinstance(value, str):  # pragma: no cover - codec-dependent
        return json.loads(value)
    return value


class MySQLCheckpointer:
    """Checkpointer backed by MySQL (8.0.13+, InnoDB).

    Usage:
        cp = MySQLCheckpointer("mysql://user:pw@host:3306/db")
        async with cp:
            await cp.append(thread_id, [msg1, msg2])
            data = await cp.load(thread_id)
            await cp.save_extra(thread_id, {"k": "v"})

    Raises CubepiSchemaUninitialized / CubepiSchemaMismatch at __aenter__ if the
    DB schema isn't compatible with this cubepi version.
    """

    def __init__(
        self,
        dsn: str,
        *,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
    ) -> None:
        self._cfg = _parse_dsn(dsn)
        self._min = min_pool_size
        self._max = max_pool_size
        self._pool: aiomysql.Pool | None = None

    async def __aenter__(self) -> "MySQLCheckpointer":
        self._pool = await aiomysql.create_pool(
            minsize=self._min,
            maxsize=self._max,
            autocommit=True,
            **self._cfg,
        )
        # If verification fails, __aexit__ won't run (the context was never
        # entered), so close the pool here to avoid leaking connections.
        try:
            await self._verify_schema()
        except BaseException:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None
            raise
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._pool is not None:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None

    async def _verify_schema(self) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                row, missing = await self._select_version(cur, "cubepi_schema_version")
                if not missing:
                    if row is None:
                        raise CubeloopSchemaUninitialized(
                            "cubepi_schema_version table is empty. Host "
                            "alembic migration must INSERT the current version "
                            "(use write_schema_version_op())."
                        )
                    actual = row[0]
                    if actual != EXPECTED_SCHEMA_VERSION:
                        if actual > EXPECTED_SCHEMA_VERSION:
                            raise CubeloopSchemaMismatch(
                                expected=EXPECTED_SCHEMA_VERSION,
                                actual=actual,
                                hint="database schema is newer than this CubeLoop release.",
                            )
                        steps = ", ".join(
                            f"upgrade_v{v}_to_v{v + 1}_op()"
                            for v in range(actual, EXPECTED_SCHEMA_VERSION)
                        )
                        raise CubeloopSchemaMismatch(
                            expected=EXPECTED_SCHEMA_VERSION,
                            actual=actual,
                            hint=(
                                "cubeloop was upgraded but host alembic is "
                                f"behind. Generate a new alembic revision that "
                                f"calls {steps} + write_schema_version_op() "
                                "(see cubeloop.checkpointer.mysql.alembic_helpers) "
                                "and run `alembic upgrade head` against this "
                                "database."
                            ),
                        )
                    return

                await cur.execute(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_schema = DATABASE() "
                    "AND table_name = 'cubepi_threads'"
                )
                count_row = await cur.fetchone()
                if count_row and count_row[0]:
                    raise CubeloopSchemaMismatch(
                        expected=EXPECTED_SCHEMA_VERSION,
                        actual=0,
                        hint=(
                            "data tables exist but schema_version is missing; "
                            "cannot auto-classify. Restore cubepi_schema_version "
                            "from the host's Alembic history; do not apply fresh "
                            "CREATE TABLE statements on existing data."
                        ),
                    )
                raise CubeloopSchemaUninitialized(
                    "cubepi tables not found. Run host application's alembic upgrade."
                )

    @staticmethod
    async def _select_version(
        cur: Any, table: str
    ) -> tuple[tuple[Any, ...] | None, bool]:
        """Return (row, missing). missing=True when the table does not exist."""
        try:
            await cur.execute(f"SELECT version FROM {table} LIMIT 1")
            row = await cur.fetchone()
            return row, False
        except pymysql.err.Error as e:
            code = e.args[0] if e.args else None
            if code in (_ER_NO_SUCH_TABLE, _ER_BAD_FIELD_ERROR):
                return None, True
            raise

    async def load(self, thread_id: str) -> CheckpointData | None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT seq, role, metadata, payload FROM cubepi_messages "
                    "WHERE thread_id = %s ORDER BY seq",
                    (thread_id,),
                )
                msg_rows = await cur.fetchall()
                await cur.execute(
                    "SELECT extra, parent_thread_id FROM cubepi_threads "
                    "WHERE thread_id = %s",
                    (thread_id,),
                )
                extra_row = await cur.fetchone()

        if not msg_rows and extra_row is None:
            return None

        messages: list[Message] = [
            _deserialize_row(thread_id, seq, role, metadata, payload)
            for seq, role, metadata, payload in msg_rows
        ]

        parent_thread_id: str | None = None
        if extra_row is not None:
            extra = _decode_json(extra_row[0])
            parent_thread_id = extra_row[1]
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
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    # No-op upsert (idiomatic equivalent of Postgres
                    # ON CONFLICT DO NOTHING); avoids INSERT IGNORE, which would
                    # also swallow unrelated errors and emit a duplicate-key
                    # warning when the row already exists.
                    await cur.execute(
                        "INSERT INTO cubepi_threads (thread_id) VALUES (%s) "
                        "ON DUPLICATE KEY UPDATE thread_id = thread_id",
                        (thread_id,),
                    )
                    await cur.execute(
                        "SELECT thread_id FROM cubepi_threads "
                        "WHERE thread_id = %s FOR UPDATE",
                        (thread_id,),
                    )
                    # Pre-flight: reject append on any completed run_id.
                    if run_ids:
                        placeholders = ", ".join(["%s"] * len(run_ids))
                        await cur.execute(
                            f"SELECT run_id FROM cubepi_runs "
                            f"WHERE thread_id = %s "
                            f"AND run_id IN ({placeholders}) "
                            f"AND completed_at IS NOT NULL",
                            (thread_id, *run_ids),
                        )
                        done_rows = await cur.fetchall()
                        if done_rows:
                            bad = ", ".join(r[0] for r in done_rows)
                            raise RunAlreadyCompletedError(
                                f"append on completed run thread={thread_id} runs={bad}"
                            )
                    await cur.execute(
                        "SELECT COALESCE(MAX(seq), 0) FROM cubepi_messages "
                        "WHERE thread_id = %s",
                        (thread_id,),
                    )
                    (last_seq,) = await cur.fetchone()
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
                    await cur.executemany(
                        "INSERT INTO cubepi_messages "
                        "(thread_id, seq, role, metadata, payload, run_id) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        rows,
                    )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def claim_run(self, thread_id: str, run_id: str) -> None:
        """Atomically claim a run_id on a thread.

        Lazy-creates the cubepi_threads row, takes a per-thread FOR UPDATE
        lock to serialize concurrent claims for the same thread, then
        checks cubepi_runs for an existing row before inserting. The
        pre-check matches the Postgres approach — using INSERT + catching
        IntegrityError would also work, but a pre-SELECT lets us
        distinguish in-flight vs completed cleanly without relying on the
        error-class taxonomy.
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    # Lazy thread row creation (claim may precede any append).
                    await cur.execute(
                        "INSERT INTO cubepi_threads (thread_id) VALUES (%s) "
                        "ON DUPLICATE KEY UPDATE thread_id = thread_id",
                        (thread_id,),
                    )
                    # Per-thread fence: serializes claim_run/append/fork on
                    # the same thread (matches Postgres's pg_advisory_xact_lock).
                    await cur.execute(
                        "SELECT thread_id FROM cubepi_threads "
                        "WHERE thread_id = %s FOR UPDATE",
                        (thread_id,),
                    )
                    await cur.execute(
                        "SELECT completed_at FROM cubepi_runs "
                        "WHERE thread_id = %s AND run_id = %s",
                        (thread_id, run_id),
                    )
                    row = await cur.fetchone()
                    if row is not None:
                        if row[0] is not None:
                            raise RunAlreadyCompletedError(
                                f"thread={thread_id} run={run_id} already completed"
                            )
                        raise RunAlreadyClaimedError(
                            f"thread={thread_id} run={run_id} in flight"
                        )
                    await cur.execute(
                        "INSERT INTO cubepi_runs (thread_id, run_id) VALUES (%s, %s)",
                        (thread_id, run_id),
                    )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def mark_run_complete(self, thread_id: str, run_id: str) -> None:
        """Mark (thread_id, run_id) complete with a monotonic completion_seq.

        Idempotent: a second call on an already-completed row is a no-op.
        Raises ``RunNotClaimedError`` if no claim row exists. The
        completion_seq is the per-thread MAX(completion_seq) + 1, allocated
        under the same per-thread FOR UPDATE fence as claim_run/append.
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT thread_id FROM cubepi_threads "
                        "WHERE thread_id = %s FOR UPDATE",
                        (thread_id,),
                    )
                    await cur.execute(
                        "SELECT completed_at FROM cubepi_runs "
                        "WHERE thread_id = %s AND run_id = %s",
                        (thread_id, run_id),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        raise RunNotClaimedError(
                            f"thread={thread_id} run={run_id} has no claim row"
                        )
                    if row[0] is not None:
                        await conn.commit()
                        return  # idempotent success
                    await cur.execute(
                        "SELECT COALESCE(MAX(completion_seq), 0) + 1 "
                        "FROM cubepi_runs WHERE thread_id = %s "
                        "AND completion_seq IS NOT NULL",
                        (thread_id,),
                    )
                    (next_seq,) = await cur.fetchone()
                    await cur.execute(
                        "UPDATE cubepi_runs SET completed_at = CURRENT_TIMESTAMP, "
                        "completion_seq = %s "
                        "WHERE thread_id = %s AND run_id = %s",
                        (next_seq, thread_id, run_id),
                    )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def load_pending(
        self, thread_id: str
    ) -> tuple[HitlRequest, str | None] | None:
        """Return the thread's pending HITL request + the owning run_id.

        Single SELECT covers both columns (vs. ``load_pending_request`` +
        ``load_pending_run_id``, which incur two round trips and can race
        across a clear/set window). Returns None when no pending exists.
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT pending_request, run_id FROM cubepi_threads "
                    "WHERE thread_id = %s",
                    (thread_id,),
                )
                row = await cur.fetchone()
        if row is None or row[0] is None:
            return None
        raw = row[0]
        if isinstance(raw, str):
            req = HitlRequest.model_validate_json(raw)
        else:
            req = HitlRequest.model_validate(raw)
        return req, row[1]

    async def snapshot(self, thread_id: str, *, after_run_id: str) -> list[Message]:
        """Return the message slice that fork() would copy, without writing.

        Includes legacy messages with NULL ``run_id`` plus messages whose
        ``run_id`` belongs to a completed run at or before
        ``after_run_id``'s completion cutoff. Raises ``ThreadNotFoundError``
        if the source thread row is absent, or ``RunNotCompletedError`` if
        ``after_run_id`` has not yet been completion-marked.
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT 1 FROM cubepi_threads WHERE thread_id = %s",
                    (thread_id,),
                )
                if await cur.fetchone() is None:
                    raise ThreadNotFoundError(f"thread={thread_id}")
                await cur.execute(
                    "SELECT completion_seq FROM cubepi_runs "
                    "WHERE thread_id = %s AND run_id = %s",
                    (thread_id, after_run_id),
                )
                cutoff_row = await cur.fetchone()
                if cutoff_row is None or cutoff_row[0] is None:
                    raise RunNotCompletedError(
                        f"thread={thread_id} run={after_run_id} not completed"
                    )
                cutoff = cutoff_row[0]
                await cur.execute(
                    "SELECT seq, role, metadata, payload FROM cubepi_messages "
                    "WHERE thread_id = %s AND ("
                    "  run_id IS NULL OR run_id IN ("
                    "    SELECT run_id FROM cubepi_runs "
                    "    WHERE thread_id = %s "
                    "    AND completion_seq IS NOT NULL "
                    "    AND completion_seq <= %s"
                    "  )"
                    ") ORDER BY seq",
                    (thread_id, thread_id, cutoff),
                )
                rows = await cur.fetchall()
        return [
            _deserialize_row(thread_id, seq, role, metadata, payload)
            for seq, role, metadata, payload in rows
        ]

    async def fork(
        self,
        src_thread_id: str,
        new_thread_id: str,
        *,
        after_run_id: str,
        metadata: JsonObject | None = None,
    ) -> None:
        """Copy completed prefix of ``src_thread_id`` into ``new_thread_id``.

        Threads-row first: the destination ``cubepi_threads`` row is
        inserted before the message + runs copies so callers see a
        complete view if they probe mid-fork. There is no cubepi_runs ->
        cubepi_threads FK on MySQL (partition limitation), but we keep
        the same ordering for semantic parity with Postgres.

        Per-thread serialization: a ``SELECT … FOR UPDATE`` on the source
        thread row fences concurrent forks/appends — without it, an
        append racing the fork could leak a partial in-flight run into
        the destination.
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    # Per-thread fence on the source thread.
                    await cur.execute(
                        "SELECT thread_id FROM cubepi_threads "
                        "WHERE thread_id = %s FOR UPDATE",
                        (src_thread_id,),
                    )
                    if await cur.fetchone() is None:
                        raise ThreadNotFoundError(f"thread={src_thread_id}")
                    # Destination must not exist.
                    await cur.execute(
                        "SELECT 1 FROM cubepi_threads WHERE thread_id = %s",
                        (new_thread_id,),
                    )
                    if await cur.fetchone() is not None:
                        raise ThreadAlreadyExistsError(f"thread={new_thread_id}")
                    # Cutoff: after_run_id must be completed on src.
                    await cur.execute(
                        "SELECT completion_seq FROM cubepi_runs "
                        "WHERE thread_id = %s AND run_id = %s",
                        (src_thread_id, after_run_id),
                    )
                    cutoff_row = await cur.fetchone()
                    if cutoff_row is None or cutoff_row[0] is None:
                        raise RunNotCompletedError(
                            f"thread={src_thread_id} run={after_run_id} not completed"
                        )
                    cutoff = cutoff_row[0]
                    # Build merged extra (carry parent's extra + fork metadata).
                    await cur.execute(
                        "SELECT extra FROM cubepi_threads WHERE thread_id = %s",
                        (src_thread_id,),
                    )
                    extra_row = await cur.fetchone()
                    base_extra = (
                        _decode_json(extra_row[0]) if extra_row is not None else {}
                    )
                    if metadata is not None:
                        # Round-trip through json to coerce to plain JSON types.
                        base_extra["fork"] = json.loads(json.dumps(metadata))
                    # INSERT destination threads row first (semantic parity
                    # with the Postgres FK-driven ordering, even though MySQL
                    # has no FK on the partitioned messages/runs tables).
                    await cur.execute(
                        "INSERT INTO cubepi_threads "
                        "(thread_id, parent_thread_id, forked_at_seq, extra) "
                        "VALUES (%s, %s, %s, %s)",
                        (
                            new_thread_id,
                            src_thread_id,
                            cutoff,
                            json.dumps(base_extra),
                        ),
                    )
                    # Copy messages: legacy NULL run_id OR completed-at-cutoff.
                    await cur.execute(
                        "INSERT INTO cubepi_messages "
                        "(thread_id, seq, role, metadata, payload, run_id) "
                        "SELECT %s, seq, role, metadata, payload, run_id "
                        "FROM cubepi_messages "
                        "WHERE thread_id = %s AND ("
                        "  run_id IS NULL OR run_id IN ("
                        "    SELECT run_id FROM cubepi_runs "
                        "    WHERE thread_id = %s "
                        "    AND completion_seq IS NOT NULL "
                        "    AND completion_seq <= %s"
                        "  )"
                        ") ORDER BY seq",
                        (new_thread_id, src_thread_id, src_thread_id, cutoff),
                    )
                    # Copy completed runs satisfying the cutoff.
                    await cur.execute(
                        "INSERT INTO cubepi_runs "
                        "(thread_id, run_id, claimed_at, completed_at, "
                        " completion_seq) "
                        "SELECT %s, run_id, claimed_at, completed_at, "
                        "       completion_seq "
                        "FROM cubepi_runs "
                        "WHERE thread_id = %s "
                        "AND completion_seq IS NOT NULL "
                        "AND completion_seq <= %s",
                        (new_thread_id, src_thread_id, cutoff),
                    )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def save_extra(self, thread_id: str, extra: JsonObject) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    # No-op upsert (idiomatic equivalent of Postgres
                    # ON CONFLICT DO NOTHING); avoids INSERT IGNORE, which would
                    # also swallow unrelated errors and emit a duplicate-key
                    # warning when the row already exists.
                    await cur.execute(
                        "INSERT INTO cubepi_threads (thread_id) VALUES (%s) "
                        "ON DUPLICATE KEY UPDATE thread_id = thread_id",
                        (thread_id,),
                    )
                    await cur.execute(
                        "SELECT extra FROM cubepi_threads "
                        "WHERE thread_id = %s FOR UPDATE",
                        (thread_id,),
                    )
                    row = await cur.fetchone()
                    current = _decode_json(row[0]) if row is not None else {}
                    merged = {**current, **extra}
                    await cur.execute(
                        "UPDATE cubepi_threads "
                        "SET extra = %s, updated_at = CURRENT_TIMESTAMP "
                        "WHERE thread_id = %s",
                        (json.dumps(merged), thread_id),
                    )
                await conn.commit()
            except BaseException:  # pragma: no cover - defensive txn rollback
                await conn.rollback()
                raise

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
        explicit transaction also covers the lazy thread INSERT.
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "INSERT INTO cubepi_threads (thread_id) VALUES (%s) "
                        "ON DUPLICATE KEY UPDATE thread_id = thread_id",
                        (thread_id,),
                    )
                    if request is None:
                        await cur.execute(
                            "UPDATE cubepi_threads "
                            "SET pending_request = NULL, run_id = NULL, "
                            "updated_at = CURRENT_TIMESTAMP WHERE thread_id = %s",
                            (thread_id,),
                        )
                    else:
                        payload = request.model_dump_json()
                        await cur.execute(
                            "UPDATE cubepi_threads "
                            "SET pending_request = %s, run_id = %s, "
                            "updated_at = CURRENT_TIMESTAMP WHERE thread_id = %s",
                            (payload, run_id, thread_id),
                        )
                await conn.commit()
            except BaseException:  # pragma: no cover - defensive txn rollback
                await conn.rollback()
                raise

    async def load_pending_request(self, thread_id: str) -> HitlRequest | None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT pending_request FROM cubepi_threads WHERE thread_id = %s",
                    (thread_id,),
                )
                row = await cur.fetchone()
        if row is None or row[0] is None:
            return None
        raw = row[0]
        # aiomysql returns JSON columns as str; tolerate already-parsed dicts (same
        # convention as the existing _parse_json helper in this module).
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
        """Atomically clear pending state when its question and run still match."""
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE cubepi_threads "
                    "SET pending_request = NULL, run_id = NULL, "
                    "updated_at = CURRENT_TIMESTAMP "
                    "WHERE thread_id = %s "
                    "AND JSON_UNQUOTE(JSON_EXTRACT(pending_request, '$.question_id')) = %s "
                    "AND run_id = %s",
                    (thread_id, question_id, run_id),
                )
                return cur.rowcount == 1

    async def load_pending_run_id(self, thread_id: str) -> str | None:
        """Return the run_id of the currently pending HITL request.

        Filters on ``pending_request IS NOT NULL`` so the result reflects
        a real pending, not a leftover run_id from a cleared row. Returns
        None when: the thread is unknown, has no pending request, or was
        written by a pre-v3 host (legacy rows have run_id NULL).
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT run_id FROM cubepi_threads "
                    "WHERE thread_id = %s AND pending_request IS NOT NULL",
                    (thread_id,),
                )
                row = await cur.fetchone()
        return row[0] if row is not None else None

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
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "INSERT INTO cubepi_threads (thread_id) VALUES (%s) "
                        "ON DUPLICATE KEY UPDATE thread_id = thread_id",
                        (thread_id,),
                    )
                    await cur.execute(
                        "INSERT INTO cubepi_hitl_answers "
                        "(thread_id, run_id, question_id, answer) "
                        "VALUES (%s, %s, %s, %s) "
                        "ON DUPLICATE KEY UPDATE "
                        "answer = VALUES(answer), "
                        "answered_at = CURRENT_TIMESTAMP",
                        (thread_id, _run_key(run_id), question_id, payload),
                    )
                await conn.commit()
            except BaseException:  # pragma: no cover - defensive txn rollback
                await conn.rollback()
                raise

    async def load_hitl_answer(
        self,
        thread_id: str,
        question_id: str,
        *,
        run_id: str | None = None,
    ) -> StructuredValue | None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT answer FROM cubepi_hitl_answers "
                    "WHERE thread_id = %s AND run_id = %s AND question_id = %s",
                    (thread_id, _run_key(run_id), question_id),
                )
                row = await cur.fetchone()
        if row is None:
            return None
        return _decode_json_value(row[0])

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
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    if question_ids is None:
                        await cur.execute(
                            "DELETE FROM cubepi_hitl_answers "
                            "WHERE thread_id = %s AND run_id = %s",
                            (thread_id, run_key),
                        )
                    else:
                        qids = list(dict.fromkeys(question_ids))
                        if qids:
                            placeholders = ",".join("%s" for _ in qids)
                            await cur.execute(
                                "DELETE FROM cubepi_hitl_answers "
                                "WHERE thread_id = %s AND run_id = %s "
                                f"AND question_id IN ({placeholders})",
                                (thread_id, run_key, *qids),
                            )
                await conn.commit()
            except BaseException:  # pragma: no cover - defensive txn rollback
                await conn.rollback()
                raise
