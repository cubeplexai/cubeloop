from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from cubeloop.hitl.types import HitlRequest
from cubeloop.providers.base import Message
from cubeloop.types import JsonObject, StructuredValue


@dataclass
class CheckpointData:
    messages: list[Message] = field(default_factory=list)
    extra: JsonObject = field(default_factory=dict)
    parent_thread_id: str | None = None


@runtime_checkable
class Checkpointer(Protocol):
    async def load(self, thread_id: str) -> CheckpointData | None: ...
    async def append(self, thread_id: str, messages: list[Message]) -> None: ...
    async def save_extra(self, thread_id: str, extra: JsonObject) -> None: ...

    async def save_pending_request(
        self,
        thread_id: str,
        request: HitlRequest | None,
        *,
        run_id: str | None = None,
    ) -> None:
        """Persist (or clear, if request is None) the pending HITL request for a thread.

        First-party implementations (Memory, SQLite, Postgres, MySQL) all implement this.
        HITL-requiring features (Agent.respond, CheckpointedChannel) use
        ``getattr(checkpointer, "save_pending_request", None)`` for graceful degradation.
        """
        ...

    async def load_pending_request(self, thread_id: str) -> HitlRequest | None:
        """Load the persisted pending HITL request for a thread, or None.

        Returns a ``HitlRequest`` instance or ``None``.
        """
        ...

    async def clear_pending_request_if_matches(
        self,
        thread_id: str,
        *,
        question_id: str,
        run_id: str,
    ) -> bool:
        """Atomically clear pending state when its question and run still match.

        Returns True when the matching pending request was cleared and False
        when the thread has no pending request or either identifier changed.
        """
        ...

    async def snapshot(self, thread_id: str, *, after_run_id: str) -> list[Message]:
        """Return messages of completed runs of `thread_id` up through
        and including `after_run_id`, in source seq order. Raises
        ThreadNotFoundError or RunNotCompletedError."""
        ...

    async def fork(
        self,
        src_thread_id: str,
        new_thread_id: str,
        *,
        after_run_id: str,
        metadata: JsonObject | None = None,
    ) -> None:
        """Atomically physical-copy messages of completed runs up
        through `after_run_id` from src to new. See spec §3.2 / §3.4."""
        ...

    async def claim_run(
        self,
        thread_id: str,
        run_id: str,
    ) -> None:
        """Insert cubepi_runs row with claimed_at=now, completed_at=NULL.
        Lazily creates the threads row if needed. Raises
        RunAlreadyClaimedError or RunAlreadyCompletedError on PK
        conflict (distinguished by completed_at IS NULL/NOT NULL)."""
        ...

    async def mark_run_complete(
        self,
        thread_id: str,
        run_id: str,
    ) -> None:
        """Allocate next per-thread completion_seq; UPDATE the run row.
        Idempotent on already-completed rows (does NOT raise
        RunAlreadyCompletedError). Raises RunNotClaimedError when no
        row exists."""
        ...

    async def load_pending(
        self,
        thread_id: str,
    ) -> tuple[HitlRequest, str | None] | None:
        """Read (HitlRequest, run_id) atomically from the pending row,
        or None when no pending request exists."""
        ...

    async def save_hitl_answer(
        self,
        thread_id: str,
        question_id: str,
        answer: StructuredValue,
        *,
        run_id: str | None = None,
    ) -> None:
        """Persist an answered HITL request for replay during resume."""
        pass

    async def load_hitl_answer(
        self,
        thread_id: str,
        question_id: str,
        *,
        run_id: str | None = None,
    ) -> StructuredValue | None:
        """Load a persisted HITL answer for replay, or None."""
        pass

    async def clear_hitl_answers(
        self,
        thread_id: str,
        question_ids: Iterable[str] | None = None,
        *,
        run_id: str | None = None,
    ) -> None:
        """Clear persisted HITL answers for a run or selected question ids."""
        pass
