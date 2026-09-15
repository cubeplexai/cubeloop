from __future__ import annotations

import asyncio
import copy
import time
from collections.abc import Iterable
from dataclasses import dataclass

from cubeloop.checkpointer.base import CheckpointData
from cubeloop.checkpointer.exceptions import (
    RunAlreadyClaimedError,
    RunAlreadyCompletedError,
    RunNotClaimedError,
    RunNotCompletedError,
    ThreadAlreadyExistsError,
    ThreadNotFoundError,
)
from cubeloop.hitl.types import HitlRequest
from cubeloop.providers.base import Message
from cubeloop.types import JsonObject, StructuredValue


@dataclass
class _RunState:
    claimed_at: float
    completed_at: float | None = None
    completion_seq: int | None = None


def _legible_message_copy(m: Message) -> Message:
    return m.model_copy(deep=True)


def _run_key(run_id: str | None) -> str:
    return run_id or ""


class MemoryCheckpointer:
    def __init__(self) -> None:
        self._store: dict[str, CheckpointData] = {}
        self._pending: dict[str, HitlRequest] = {}
        self._pending_run_id: dict[str, str | None] = {}
        self._hitl_answers: dict[tuple[str, str, str], StructuredValue] = {}
        self._runs: dict[str, dict[str, _RunState]] = {}
        self._lock = asyncio.Lock()

    async def load(self, thread_id: str) -> CheckpointData | None:
        return self._store.get(thread_id)

    async def append(self, thread_id: str, messages: list[Message]) -> None:
        async with self._lock:
            for m in messages:
                if m.run_id is None:
                    continue
                rs = self._runs.get(thread_id, {}).get(m.run_id)
                if rs is not None and rs.completed_at is not None:
                    raise RunAlreadyCompletedError(
                        f"append on completed run thread={thread_id} run={m.run_id}"
                    )
            if thread_id not in self._store:
                self._store[thread_id] = CheckpointData()
            self._store[thread_id].messages.extend(messages)

    async def save_extra(self, thread_id: str, extra: JsonObject) -> None:
        async with self._lock:
            if thread_id not in self._store:
                self._store[thread_id] = CheckpointData()
            self._store[thread_id].extra.update(extra)

    async def save_pending_request(
        self,
        thread_id: str,
        request: HitlRequest | None,
        *,
        run_id: str | None = None,
    ) -> None:
        async with self._lock:
            if request is None:
                self._pending.pop(thread_id, None)
                self._pending_run_id.pop(thread_id, None)
            else:
                self._pending[thread_id] = request
                self._pending_run_id[thread_id] = run_id

    async def load_pending_request(self, thread_id: str) -> HitlRequest | None:
        return self._pending.get(thread_id)

    async def clear_pending_request_if_matches(
        self,
        thread_id: str,
        *,
        question_id: str,
        run_id: str,
    ) -> bool:
        async with self._lock:
            request = self._pending.get(thread_id)
            if (
                request is None
                or request.question_id != question_id
                or self._pending_run_id.get(thread_id) != run_id
            ):
                return False
            self._pending.pop(thread_id, None)
            self._pending_run_id.pop(thread_id, None)
            return True

    async def load_pending_run_id(self, thread_id: str) -> str | None:
        return self._pending_run_id.get(thread_id)

    async def load_pending(
        self, thread_id: str
    ) -> tuple[HitlRequest, str | None] | None:
        req = self._pending.get(thread_id)
        if req is None:
            return None
        return req, self._pending_run_id.get(thread_id)

    async def save_hitl_answer(
        self,
        thread_id: str,
        question_id: str,
        answer: StructuredValue,
        *,
        run_id: str | None = None,
    ) -> None:
        key = (thread_id, _run_key(run_id), question_id)
        async with self._lock:
            self._hitl_answers[key] = copy.deepcopy(answer)

    async def load_hitl_answer(
        self,
        thread_id: str,
        question_id: str,
        *,
        run_id: str | None = None,
    ) -> StructuredValue | None:
        key = (thread_id, _run_key(run_id), question_id)
        async with self._lock:
            answer = self._hitl_answers.get(key)
            return copy.deepcopy(answer)

    async def clear_hitl_answers(
        self,
        thread_id: str,
        question_ids: Iterable[str] | None = None,
        *,
        run_id: str | None = None,
    ) -> None:
        run_key = _run_key(run_id)
        async with self._lock:
            if question_ids is None:
                keys = [
                    key
                    for key in self._hitl_answers
                    if key[0] == thread_id and key[1] == run_key
                ]
            else:
                keys = [
                    (thread_id, run_key, question_id)
                    for question_id in set(question_ids)
                ]
            for key in keys:
                self._hitl_answers.pop(key, None)

    async def claim_run(self, thread_id: str, run_id: str) -> None:
        async with self._lock:
            runs = self._runs.setdefault(thread_id, {})
            existing = runs.get(run_id)
            if existing is not None:
                if existing.completed_at is None:
                    raise RunAlreadyClaimedError(
                        f"thread={thread_id} run={run_id} in flight"
                    )
                raise RunAlreadyCompletedError(
                    f"thread={thread_id} run={run_id} already completed"
                )
            runs[run_id] = _RunState(claimed_at=time.time())

    async def mark_run_complete(self, thread_id: str, run_id: str) -> None:
        async with self._lock:
            runs = self._runs.get(thread_id) or {}
            state = runs.get(run_id)
            if state is None:
                raise RunNotClaimedError(
                    f"thread={thread_id} run={run_id} has no claim row"
                )
            if state.completed_at is not None:
                return  # idempotent
            existing_seqs = [
                s.completion_seq for s in runs.values() if s.completion_seq is not None
            ]
            next_seq = max(existing_seqs, default=0) + 1
            state.completed_at = time.time()
            state.completion_seq = next_seq

    async def snapshot(self, thread_id: str, *, after_run_id: str) -> list[Message]:
        async with self._lock:
            data = self._store.get(thread_id)
            if data is None and thread_id not in self._runs:
                raise ThreadNotFoundError(f"thread={thread_id}")
            runs = self._runs.get(thread_id, {})
            cutoff_state = runs.get(after_run_id)
            if cutoff_state is None or cutoff_state.completion_seq is None:
                raise RunNotCompletedError(
                    f"thread={thread_id} run={after_run_id} not completed"
                )
            cutoff = cutoff_state.completion_seq
            selected: list[Message] = []
            for m in data.messages if data else []:
                if m.run_id is None:
                    selected.append(_legible_message_copy(m))
                    continue
                rs = runs.get(m.run_id)
                if rs is None or rs.completion_seq is None:
                    continue
                if rs.completion_seq <= cutoff:
                    selected.append(_legible_message_copy(m))
            return selected

    async def fork(
        self,
        src_thread_id: str,
        new_thread_id: str,
        *,
        after_run_id: str,
        metadata: JsonObject | None = None,
    ) -> None:
        async with self._lock:
            if new_thread_id in self._store or new_thread_id in self._runs:
                raise ThreadAlreadyExistsError(f"thread={new_thread_id}")
            src_data = self._store.get(src_thread_id)
            src_runs = self._runs.get(src_thread_id, {})
            if src_data is None and not src_runs:
                raise ThreadNotFoundError(f"thread={src_thread_id}")
            cutoff_state = src_runs.get(after_run_id)
            if cutoff_state is None or cutoff_state.completion_seq is None:
                raise RunNotCompletedError(
                    f"thread={src_thread_id} run={after_run_id} not completed"
                )
            cutoff = cutoff_state.completion_seq
            # Select messages.
            new_messages: list[Message] = []
            for m in src_data.messages if src_data else []:
                if m.run_id is None:
                    new_messages.append(_legible_message_copy(m))
                    continue
                rs = src_runs.get(m.run_id)
                if rs is None or rs.completion_seq is None:
                    continue
                if rs.completion_seq <= cutoff:
                    new_messages.append(_legible_message_copy(m))
            # Deep copy extra and merge metadata.
            base_extra = copy.deepcopy(src_data.extra if src_data else {})
            if metadata is not None:
                base_extra["fork"] = copy.deepcopy(metadata)
            # Carry completed runs satisfying cutoff.
            new_runs: dict[str, _RunState] = {}
            for rid, state in src_runs.items():
                if state.completion_seq is None:
                    continue
                if state.completion_seq <= cutoff:
                    new_runs[rid] = _RunState(
                        claimed_at=state.claimed_at,
                        completed_at=state.completed_at,
                        completion_seq=state.completion_seq,
                    )
            self._store[new_thread_id] = CheckpointData(
                messages=new_messages,
                extra=base_extra,
                parent_thread_id=src_thread_id,
            )
            self._runs[new_thread_id] = new_runs
