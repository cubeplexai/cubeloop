from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, cast

import aiosqlite
from pydantic import TypeAdapter

from cubeloop.checkpointer.base import CheckpointData
from cubeloop.checkpointer.exceptions import (
    CheckpointCorruptionError,
    CheckpointerLockTimeoutError,
    RunAlreadyClaimedError,
    RunAlreadyCompletedError,
    RunNotClaimedError,
    RunNotCompletedError,
    ThreadAlreadyExistsError,
    ThreadNotFoundError,
)
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


def _serialize_structured_value(value: StructuredValue) -> str:
    return _STRUCTURED_VALUE_ADAPTER.dump_json(value).decode("utf-8")


def _deserialize_structured_value(value: str) -> StructuredValue:
    return cast(StructuredValue, json.loads(value))


@asynccontextmanager
async def _writer_txn(db: aiosqlite.Connection) -> AsyncIterator[None]:
    """Wrap a writer transaction in BEGIN IMMEDIATE and surface
    SQLITE_BUSY as CheckpointerLockTimeoutError."""
    try:
        await db.execute("BEGIN IMMEDIATE")
    except aiosqlite.OperationalError as exc:
        if "lock" in str(exc).lower() or "busy" in str(exc).lower():
            raise CheckpointerLockTimeoutError(str(exc)) from exc
        raise
    try:
        yield
    except BaseException:
        await db.rollback()
        raise
    else:
        await db.commit()


class SQLiteCheckpointer:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> SQLiteCheckpointer:
        self._db = await aiosqlite.connect(self._db_path)
        # Set busy_timeout to 5s — gives writer contention a chance to wait
        # rather than immediately failing with SQLITE_BUSY.
        await self._db.execute("PRAGMA busy_timeout = 5000")
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS messages ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  thread_id TEXT NOT NULL,"
            "  message_json TEXT NOT NULL,"
            "  created_at REAL NOT NULL DEFAULT (julianday('now'))"
            ")"
        )
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS thread_extra ("
            "  thread_id TEXT PRIMARY KEY,"
            "  extra_json TEXT NOT NULL DEFAULT '{}'"
            ")"
        )
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS thread_pending_request ("
            "  thread_id TEXT PRIMARY KEY,"
            "  request_json TEXT NOT NULL,"
            "  run_id TEXT,"
            "  created_at REAL NOT NULL DEFAULT (julianday('now'))"
            ")"
        )
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS thread_hitl_answers ("
            "  thread_id TEXT NOT NULL,"
            "  run_id TEXT NOT NULL,"
            "  question_id TEXT NOT NULL,"
            "  answer_json TEXT NOT NULL,"
            "  answered_at REAL NOT NULL DEFAULT (julianday('now')),"
            "  PRIMARY KEY (thread_id, run_id, question_id)"
            ")"
        )
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS runs ("
            "  thread_id TEXT NOT NULL,"
            "  run_id TEXT NOT NULL,"
            "  claimed_at REAL NOT NULL DEFAULT (julianday('now')),"
            "  completed_at REAL,"
            "  completion_seq INTEGER,"
            "  PRIMARY KEY (thread_id, run_id)"
            ")"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS ix_runs_thread_completion "
            "ON runs (thread_id, completion_seq)"
        )
        # One-shot migration: older DBs (created before run_id existed) have
        # the table without the run_id column. SQLite has no schema_version
        # gate, so we ALTER inline when it's missing.
        cur = await self._db.execute("PRAGMA table_info(thread_pending_request)")
        cols = {row[1] for row in await cur.fetchall()}
        if "run_id" not in cols:
            await self._db.execute(
                "ALTER TABLE thread_pending_request ADD COLUMN run_id TEXT"
            )
        # Same one-shot ALTER pattern for the new run_id column on messages.
        cur = await self._db.execute("PRAGMA table_info(messages)")
        cols = {row[1] for row in await cur.fetchall()}
        if "run_id" not in cols:
            await self._db.execute("ALTER TABLE messages ADD COLUMN run_id TEXT")
        # And for parent_thread_id on thread_extra.
        cur = await self._db.execute("PRAGMA table_info(thread_extra)")
        cols = {row[1] for row in await cur.fetchall()}
        if "parent_thread_id" not in cols:
            await self._db.execute(
                "ALTER TABLE thread_extra ADD COLUMN parent_thread_id TEXT"
            )
        await self._db.commit()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def load(self, thread_id: str) -> CheckpointData | None:
        assert self._db is not None
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT id, message_json FROM messages WHERE thread_id = ? ORDER BY id",
                (thread_id,),
            )
            rows = await cursor.fetchall()

            extra_cursor = await self._db.execute(
                "SELECT extra_json, parent_thread_id FROM thread_extra "
                "WHERE thread_id = ?",
                (thread_id,),
            )
            extra_row = await extra_cursor.fetchone()

            if not rows and not extra_row:
                return None

            messages = []
            for row in rows:
                messages.append(_deserialize_row(thread_id, row[0], row[1]))

            extra = json.loads(extra_row[0]) if extra_row else {}
            parent = extra_row[1] if extra_row else None
            return CheckpointData(
                messages=messages, extra=extra, parent_thread_id=parent
            )

    async def append(self, thread_id: str, messages: list[Message]) -> None:
        assert self._db is not None
        run_ids = {
            rid for m in messages if (rid := getattr(m, "run_id", None)) is not None
        }
        async with self._lock, _writer_txn(self._db):
            if run_ids:
                placeholders = ",".join("?" for _ in run_ids)
                cur = await self._db.execute(
                    f"SELECT run_id FROM runs WHERE thread_id = ? "
                    f"AND run_id IN ({placeholders}) "
                    f"AND completed_at IS NOT NULL",
                    (thread_id, *run_ids),
                )
                done = await cur.fetchall()
                if done:
                    bad = ", ".join(r[0] for r in done)
                    raise RunAlreadyCompletedError(
                        f"append on completed run thread={thread_id} runs={bad}"
                    )
            for msg in messages:
                msg_json = _serialize_message(msg)
                await self._db.execute(
                    "INSERT INTO messages (thread_id, message_json, run_id) "
                    "VALUES (?, ?, ?)",
                    (thread_id, msg_json, getattr(msg, "run_id", None)),
                )

    async def save_extra(self, thread_id: str, extra: JsonObject) -> None:
        assert self._db is not None
        async with self._lock, _writer_txn(self._db):
            existing_cursor = await self._db.execute(
                "SELECT extra_json FROM thread_extra WHERE thread_id = ?",
                (thread_id,),
            )
            existing_row = await existing_cursor.fetchone()
            if existing_row:
                existing_extra = json.loads(existing_row[0])
                existing_extra.update(extra)
                await self._db.execute(
                    "UPDATE thread_extra SET extra_json = ? WHERE thread_id = ?",
                    (json.dumps(existing_extra), thread_id),
                )
            else:
                await self._db.execute(
                    "INSERT INTO thread_extra (thread_id, extra_json) VALUES (?, ?)",
                    (thread_id, json.dumps(extra)),
                )

    async def save_pending_request(
        self,
        thread_id: str,
        request: HitlRequest | None,
        *,
        run_id: str | None = None,
    ) -> None:
        assert self._db is not None
        async with self._lock, _writer_txn(self._db):
            if request is None:
                # Clearing pending implicitly clears the associated run_id.
                await self._db.execute(
                    "DELETE FROM thread_pending_request WHERE thread_id = ?",
                    (thread_id,),
                )
            else:
                payload = request.model_dump_json()
                # Single statement writes pending + run_id atomically.
                await self._db.execute(
                    "INSERT OR REPLACE INTO thread_pending_request "
                    "(thread_id, request_json, run_id) VALUES (?, ?, ?)",
                    (thread_id, payload, run_id),
                )

    async def claim_run(self, thread_id: str, run_id: str) -> None:
        assert self._db is not None
        async with self._lock, _writer_txn(self._db):
            cur = await self._db.execute(
                "SELECT completed_at FROM runs WHERE thread_id = ? AND run_id = ?",
                (thread_id, run_id),
            )
            row = await cur.fetchone()
            if row is not None:
                completed_at = row[0]
                if completed_at is None:
                    raise RunAlreadyClaimedError(
                        f"thread={thread_id} run={run_id} in flight"
                    )
                raise RunAlreadyCompletedError(
                    f"thread={thread_id} run={run_id} already completed"
                )
            await self._db.execute(
                "INSERT INTO runs (thread_id, run_id) VALUES (?, ?)",
                (thread_id, run_id),
            )

    async def mark_run_complete(self, thread_id: str, run_id: str) -> None:
        assert self._db is not None
        async with self._lock, _writer_txn(self._db):
            cur = await self._db.execute(
                "SELECT completed_at FROM runs WHERE thread_id = ? AND run_id = ?",
                (thread_id, run_id),
            )
            row = await cur.fetchone()
            if row is None:
                raise RunNotClaimedError(
                    f"thread={thread_id} run={run_id} has no claim row"
                )
            if row[0] is not None:
                return  # idempotent success
            cur = await self._db.execute(
                "SELECT COALESCE(MAX(completion_seq), 0) + 1 FROM runs "
                "WHERE thread_id = ? AND completion_seq IS NOT NULL",
                (thread_id,),
            )
            seq_row = await cur.fetchone()
            assert seq_row is not None
            next_seq = seq_row[0]
            await self._db.execute(
                "UPDATE runs SET completed_at = julianday('now'), "
                "completion_seq = ? WHERE thread_id = ? AND run_id = ?",
                (next_seq, thread_id, run_id),
            )

    async def load_pending(
        self, thread_id: str
    ) -> tuple[HitlRequest, str | None] | None:
        assert self._db is not None
        async with self._lock:
            cur = await self._db.execute(
                "SELECT request_json, run_id FROM thread_pending_request "
                "WHERE thread_id = ?",
                (thread_id,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            return HitlRequest.model_validate_json(row[0]), row[1]

    async def load_pending_request(self, thread_id: str) -> HitlRequest | None:
        assert self._db is not None
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT request_json FROM thread_pending_request WHERE thread_id = ?",
                (thread_id,),
            )
            row = await cursor.fetchone()
            return HitlRequest.model_validate_json(row[0]) if row else None

    async def clear_pending_request_if_matches(
        self,
        thread_id: str,
        *,
        question_id: str,
        run_id: str,
    ) -> bool:
        assert self._db is not None
        async with self._lock, _writer_txn(self._db):
            cursor = await self._db.execute(
                "DELETE FROM thread_pending_request "
                "WHERE thread_id = ? AND json_extract(request_json, '$.question_id') = ? "
                "AND run_id = ?",
                (thread_id, question_id, run_id),
            )
            return cursor.rowcount == 1

    async def load_pending_run_id(self, thread_id: str) -> str | None:
        assert self._db is not None
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT run_id FROM thread_pending_request WHERE thread_id = ?",
                (thread_id,),
            )
            row = await cursor.fetchone()
        return row[0] if row else None

    async def save_hitl_answer(
        self,
        thread_id: str,
        question_id: str,
        answer: StructuredValue,
        *,
        run_id: str | None = None,
    ) -> None:
        assert self._db is not None
        payload = _serialize_structured_value(answer)
        async with self._lock, _writer_txn(self._db):
            await self._db.execute(
                "INSERT OR REPLACE INTO thread_hitl_answers "
                "(thread_id, run_id, question_id, answer_json) "
                "VALUES (?, ?, ?, ?)",
                (thread_id, _run_key(run_id), question_id, payload),
            )

    async def load_hitl_answer(
        self,
        thread_id: str,
        question_id: str,
        *,
        run_id: str | None = None,
    ) -> StructuredValue | None:
        assert self._db is not None
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT answer_json FROM thread_hitl_answers "
                "WHERE thread_id = ? AND run_id = ? AND question_id = ?",
                (thread_id, _run_key(run_id), question_id),
            )
            row = await cursor.fetchone()
        return _deserialize_structured_value(row[0]) if row else None

    async def clear_hitl_answers(
        self,
        thread_id: str,
        question_ids: Iterable[str] | None = None,
        *,
        run_id: str | None = None,
    ) -> None:
        assert self._db is not None
        run_key = _run_key(run_id)
        async with self._lock, _writer_txn(self._db):
            if question_ids is None:
                await self._db.execute(
                    "DELETE FROM thread_hitl_answers "
                    "WHERE thread_id = ? AND run_id = ?",
                    (thread_id, run_key),
                )
                return
            qids = list(dict.fromkeys(question_ids))
            if not qids:
                return
            placeholders = ",".join("?" for _ in qids)
            await self._db.execute(
                "DELETE FROM thread_hitl_answers "
                f"WHERE thread_id = ? AND run_id = ? "
                f"AND question_id IN ({placeholders})",
                (thread_id, run_key, *qids),
            )

    async def snapshot(self, thread_id: str, *, after_run_id: str) -> list[Message]:
        assert self._db is not None
        async with self._lock:
            cur = await self._db.execute(
                "SELECT completion_seq FROM runs WHERE thread_id = ? AND run_id = ?",
                (thread_id, after_run_id),
            )
            row = await cur.fetchone()
            if row is None or row[0] is None:
                raise RunNotCompletedError(
                    f"thread={thread_id} run={after_run_id} not completed"
                )
            cutoff = row[0]
            cur = await self._db.execute(
                "SELECT id, message_json FROM messages WHERE thread_id = ? "
                "AND (run_id IS NULL OR run_id IN ("
                "  SELECT run_id FROM runs WHERE thread_id = ? "
                "  AND completion_seq IS NOT NULL "
                "  AND completion_seq <= ?"
                ")) ORDER BY id",
                (thread_id, thread_id, cutoff),
            )
            rows = await cur.fetchall()
            return [_deserialize_row(thread_id, r[0], r[1]) for r in rows]

    async def fork(
        self,
        src_thread_id: str,
        new_thread_id: str,
        *,
        after_run_id: str,
        metadata: JsonObject | None = None,
    ) -> None:
        assert self._db is not None
        async with self._lock, _writer_txn(self._db):
            # Source existence: messages OR thread_extra OR runs.
            cur = await self._db.execute(
                "SELECT 1 FROM messages WHERE thread_id = ? LIMIT 1",
                (src_thread_id,),
            )
            src_has_msg = await cur.fetchone() is not None
            cur = await self._db.execute(
                "SELECT 1 FROM thread_extra WHERE thread_id = ?",
                (src_thread_id,),
            )
            src_has_extra = await cur.fetchone() is not None
            cur = await self._db.execute(
                "SELECT 1 FROM runs WHERE thread_id = ? LIMIT 1",
                (src_thread_id,),
            )
            src_has_runs = await cur.fetchone() is not None
            if not (src_has_msg or src_has_extra or src_has_runs):
                raise ThreadNotFoundError(f"thread={src_thread_id}")
            # Destination collision.
            cur = await self._db.execute(
                "SELECT 1 FROM messages WHERE thread_id = ? LIMIT 1",
                (new_thread_id,),
            )
            if await cur.fetchone():
                raise ThreadAlreadyExistsError(f"thread={new_thread_id}")
            cur = await self._db.execute(
                "SELECT 1 FROM thread_extra WHERE thread_id = ?",
                (new_thread_id,),
            )
            if await cur.fetchone():
                raise ThreadAlreadyExistsError(f"thread={new_thread_id}")
            cur = await self._db.execute(
                "SELECT 1 FROM runs WHERE thread_id = ? LIMIT 1",
                (new_thread_id,),
            )
            if await cur.fetchone():
                raise ThreadAlreadyExistsError(f"thread={new_thread_id}")
            # Cutoff.
            cur = await self._db.execute(
                "SELECT completion_seq FROM runs WHERE thread_id = ? AND run_id = ?",
                (src_thread_id, after_run_id),
            )
            row = await cur.fetchone()
            if row is None or row[0] is None:
                raise RunNotCompletedError(
                    f"thread={src_thread_id} run={after_run_id} not completed"
                )
            cutoff = row[0]
            # Copy messages.
            await self._db.execute(
                "INSERT INTO messages (thread_id, run_id, message_json) "
                "SELECT ?, run_id, message_json FROM messages "
                "WHERE thread_id = ? AND ("
                "  run_id IS NULL OR run_id IN ("
                "    SELECT run_id FROM runs WHERE thread_id = ? "
                "    AND completion_seq IS NOT NULL "
                "    AND completion_seq <= ?"
                "  )"
                ") ORDER BY id",
                (new_thread_id, src_thread_id, src_thread_id, cutoff),
            )
            # Copy runs.
            await self._db.execute(
                "INSERT INTO runs (thread_id, run_id, claimed_at, "
                "completed_at, completion_seq) "
                "SELECT ?, run_id, claimed_at, completed_at, completion_seq "
                "FROM runs WHERE thread_id = ? "
                "AND completion_seq IS NOT NULL AND completion_seq <= ?",
                (new_thread_id, src_thread_id, cutoff),
            )
            # Build merged extra.
            cur = await self._db.execute(
                "SELECT extra_json FROM thread_extra WHERE thread_id = ?",
                (src_thread_id,),
            )
            row = await cur.fetchone()
            merged_extra = json.loads(row[0]) if row else {}
            if metadata is not None:
                merged_extra["fork"] = json.loads(json.dumps(metadata))
            await self._db.execute(
                "INSERT INTO thread_extra (thread_id, extra_json, "
                "parent_thread_id) VALUES (?, ?, ?)",
                (new_thread_id, json.dumps(merged_extra), src_thread_id),
            )


def _serialize_message(msg: Any) -> str:
    if hasattr(msg, "model_dump"):
        return json.dumps(msg.model_dump())
    return json.dumps(msg)


def _deserialize_message(data: dict[str, Any]) -> Message:
    role = data.get("role")
    if role == "user":
        return UserMessage.model_validate(data)
    elif role == "assistant":
        return AssistantMessage.model_validate(data)
    elif role == "tool_result":
        return ToolResultMessage.model_validate(data)
    # Unknown role is corruption, same as the postgres/mysql backends —
    # passing the raw dict through would fail far from the cause.
    raise ValueError(f"unknown role in DB: {role!r}")


def _deserialize_row(thread_id: str, row_id: int, message_json: str) -> Message:
    try:
        return _deserialize_message(json.loads(message_json))
    except Exception as exc:
        raise CheckpointCorruptionError(
            thread_id=thread_id,
            backend="sqlite",
            row_ref=f"messages.id={row_id}",
            cause=exc,
        ) from exc
