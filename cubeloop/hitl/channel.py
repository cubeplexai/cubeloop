from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import time
from typing import Any, AsyncIterator, Protocol, TypeAlias
from typing import cast

from cubeloop.checkpointer.base import Checkpointer
from cubeloop.hitl.exceptions import (
    HitlAborted,
    HitlCancelled,
    HitlConcurrencyError,
    HitlStaleAnswer,
    HitlTimedOut,
)
from cubeloop.hitl.types import (
    ApproveAnswer,
    ApproveRequest,
    AskRequest,
    ConfirmRequest,
    HitlPayload,
    HitlRequest,
    Question,
)
from cubeloop.types import JsonObject, StructuredValue


# Sentinel default for the per-call `timeout` kwarg on confirm/approve/ask.
# Distinguishes "caller omitted timeout — use the channel's default_timeout"
# from "caller explicitly passed timeout=None — wait indefinitely."
# Without this sentinel, those two cases would both look like `timeout is None`
# and there would be no way to override default_timeout for a single call.
class _UseDefaultTimeout:
    pass


_USE_DEFAULT_TIMEOUT = _UseDefaultTimeout()
TimeoutArg: TypeAlias = float | None | _UseDefaultTimeout

# ContextVar used by CheckpointedChannel to enforce the "do not ask HITL
# from inside a custom tool body" durability guard. Set by
# cubeloop.agent.tools._execute_prepared for non-builtin tools; the guard fires
# in CheckpointedChannel._on_pending_set when the var is True and
# allow_inside_custom_tool=False.
_in_custom_tool_var: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_in_custom_tool_var",
    default=False,
)


class HitlChannel(Protocol):
    # ---- agent side ----
    async def confirm(
        self,
        prompt: str,
        *,
        details: JsonObject | None = None,
        tool_call_id: str | None = None,
        timeout: float | None = None,
        signal: asyncio.Event | None = None,
    ) -> bool: ...

    async def approve(
        self,
        tool_name: str,
        tool_call_id: str,
        args: JsonObject,
        *,
        details: JsonObject | None = None,
        timeout: float | None = None,
        signal: asyncio.Event | None = None,
    ) -> ApproveAnswer: ...

    async def ask(
        self,
        questions: list[Question],
        *,
        timeout: float | None = None,
        signal: asyncio.Event | None = None,
    ) -> dict[str, str | list[str]]: ...

    @property
    def pending(self) -> HitlRequest | None: ...

    @property
    def _future(self) -> asyncio.Future[StructuredValue] | None: ...

    def subscribe(self) -> AsyncIterator[HitlRequest]: ...

    async def answer(self, question_id: str, answer: StructuredValue) -> None: ...

    async def cancel(self, question_id: str, reason: str = "cancelled") -> None: ...

    def attach_resume_answer(
        self, question_id: str, answer: StructuredValue
    ) -> None: ...


class _BaseChannel:
    """Shared state machine for InMemoryChannel and CheckpointedChannel.

    Maintains the single-pending invariant, the awaiting future, the
    resume-answer slot, subscriber queues, and the optional emit
    callback wired by the Agent at construction.
    """

    def __init__(
        self,
        *,
        default_timeout: float | None = None,
        thread_id: str | None = None,
    ) -> None:
        self._default_timeout = default_timeout
        self._thread_id = thread_id
        self._pending: HitlRequest | None = None
        self._future: asyncio.Future[StructuredValue] | None = None
        self._resume_slot: tuple[str, StructuredValue] | None = None
        self._subscribers: list[asyncio.Queue[HitlRequest]] = []
        self._emit = None  # set by Agent._bind_channel
        # Per-content-hash sequence counter for question_id derivation.
        # Two consecutive ask()/confirm() calls with identical payload get
        # distinct qids ("hash.0", "hash.1", …) — prevents a stale answer
        # from a prior retry of the same prompt matching a new pending
        # request. On resume, a fresh channel starts the counter at 0 and
        # re-emits the same qid sequence the original run produced, so
        # _resume_slot still matches when the tool body re-asks in the
        # same order.
        self._qid_seq: dict[str, int] = {}

    @property
    def pending(self) -> HitlRequest | None:
        return self._pending

    def attach_resume_answer(self, question_id: str, answer: StructuredValue) -> None:
        self._resume_slot = (question_id, answer)
        # A resume replays the tool body / middleware hook from the top. The
        # per-content qid counter (_qid_seq) is per-channel-instance, so on a
        # same-process resume it has already advanced past the persisted
        # question_id; left as-is, the replay would derive hash.N while the
        # persisted slot holds hash.M (M < N) and _await_answer's resume
        # short-circuit would miss — re-asking an already-answered prompt or
        # hanging. Reset it so the replay reproduces the exact qid sequence
        # the original run produced (hash.0, hash.1, …). Cross-process resume
        # already started from an empty counter; this makes same-process
        # resume behave identically.
        self._qid_seq = {}

    def _bind_emit(self, emit) -> None:
        self._emit = emit

    async def _load_answer(self, question_id: str) -> StructuredValue | None:
        return None

    async def _save_answer(self, question_id: str, answer: StructuredValue) -> None:
        return None

    def _next_qid(self, kind: str, payload_repr: StructuredValue) -> str:
        """Derive the next question_id for an ask/confirm payload.

        Combines a content hash (so resume after detach can match on
        payload content) with a per-content monotonic counter (so two
        retries of the *same* prompt get distinct ids — a stale answer
        for retry N cannot match retry N+1's pending request).

        On resume, a fresh channel instance starts with an empty counter
        and walks `hash.0`, `hash.1`, …, in the same order the original
        run produced — so `attach_resume_answer(persisted_qid, ans)` still
        matches when the tool body re-asks in the same order.
        """
        content_hash = _derive_question_id(kind, payload_repr)
        seq = self._qid_seq.get(content_hash, 0)
        self._qid_seq[content_hash] = seq + 1
        return f"{content_hash}.{seq}"

    def subscribe(self) -> AsyncIterator[HitlRequest]:
        queue: asyncio.Queue[HitlRequest] = asyncio.Queue()
        self._subscribers.append(queue)

        async def gen():
            try:
                while True:
                    yield await queue.get()
            finally:
                if queue in self._subscribers:
                    self._subscribers.remove(queue)

        return gen()

    async def _await_answer(
        self,
        payload: HitlPayload,
        timeout: TimeoutArg,
        signal: asyncio.Event | None,
        question_id: str,
    ) -> StructuredValue:
        from cubeloop.hitl._trace import hitl_span

        kind = payload.kind
        effective_timeout: float | None = (
            self._default_timeout
            if isinstance(timeout, _UseDefaultTimeout)
            else timeout
        )

        attrs: dict[str, Any] = {
            "question_id": question_id,
            "timeout_seconds": effective_timeout,
        }
        if isinstance(payload, ApproveRequest):
            attrs["tool_call_id"] = payload.tool_call_id
            attrs["tool_name"] = payload.tool_name
        # Only CheckpointedChannel carries _run_id; _BaseChannel/InMemoryChannel
        # don't. hitl_span skips None-valued attrs, so leaving this out for the
        # base channel is correct.
        run_id = getattr(self, "_run_id", None)
        if run_id is not None:
            attrs["run_id"] = run_id

        outcome = "unknown"
        from_resume = False
        t0 = time.monotonic()
        with hitl_span(kind, **attrs) as span:
            try:
                # Resume short-circuit — return immediately, do NOT set _pending.
                if (
                    self._resume_slot is not None
                    and self._resume_slot[0] == question_id
                ):
                    _, ans = self._resume_slot
                    self._resume_slot = None
                    ans = _normalize_answer(payload, ans)
                    from_resume = True
                    outcome = _outcome_from_answer(kind, ans)
                    if self._emit is not None:
                        from cubeloop.agent.types import HitlAnswerEvent

                        await self._emit_event(
                            HitlAnswerEvent(question_id=question_id, answer=ans)
                        )
                    return ans

                persisted_answer = await self._load_answer(question_id)
                if persisted_answer is not None:
                    ans = _normalize_answer(payload, persisted_answer)
                    from_resume = True
                    outcome = _outcome_from_answer(kind, ans)
                    if self._emit is not None:
                        from cubeloop.agent.types import HitlAnswerEvent

                        await self._emit_event(
                            HitlAnswerEvent(question_id=question_id, answer=ans)
                        )
                    return ans

                if self._pending is not None:
                    raise HitlConcurrencyError(
                        f"channel busy: already pending {self._pending.question_id}"
                    )

                req = HitlRequest(
                    question_id=question_id,
                    thread_id=self._thread_id,
                    payload=payload,
                    created_at=time.time(),
                    timeout_seconds=effective_timeout,
                )
                self._pending = req
                self._future = asyncio.get_running_loop().create_future()

                exc_caught: BaseException | None = None
                signal_task: asyncio.Task[bool] | None = None
                # CRITICAL: _on_pending_set is called INSIDE the try/finally so that
                # if it raises (e.g. CheckpointedChannel's HitlDurabilityNotGuaranteed
                # guard), the finally still clears _pending/_future. Otherwise the
                # channel would be permanently wedged.
                try:
                    await self._on_pending_set(req)
                    if signal is None and effective_timeout is None:
                        result = await self._future
                        outcome = _outcome_from_answer(kind, result)
                        return result
                    tasks: list[
                        asyncio.Future[StructuredValue] | asyncio.Future[bool]
                    ] = [self._future]
                    if signal is not None:
                        signal_task = cast(
                            asyncio.Task[bool],
                            asyncio.ensure_future(signal.wait()),
                        )
                        tasks.append(signal_task)
                    done, pending_tasks = await asyncio.wait(
                        tasks,
                        timeout=effective_timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    # Clean up pending tasks (the loser of the race + any signal task
                    # we never want to leave hanging).
                    for p in pending_tasks:
                        if p is not self._future:
                            p.cancel()
                    if not done:
                        assert effective_timeout is not None
                        raise HitlTimedOut(effective_timeout)
                    # Use task identity, not signal.is_set() — Agent.abort() leaves the
                    # signal sticky-set, so signal.is_set() would be true even on the
                    # happy path after a prior abort. The race winner is what matters.
                    if (
                        signal_task is not None
                        and signal_task in done
                        and self._future not in done
                    ):
                        raise HitlAborted("agent signal fired during HITL pending")
                    result = self._future.result()
                    outcome = _outcome_from_answer(kind, result)
                    return result
                except BaseException as exc:
                    exc_caught = exc
                    raise
                finally:
                    # Cancel any still-pending signal task to avoid leaks.
                    if signal_task is not None and not signal_task.done():
                        signal_task.cancel()
                    # Keep the local _pending/_future slot occupied WHILE
                    # _on_pending_cleared runs — CheckpointedChannel's hook
                    # may do an `await save_pending_request(thread_id, None)`,
                    # and during that await another caller could otherwise
                    # see _pending == None, persist a new pending, then have
                    # it wiped by this prior request's late clear. Holding
                    # the local slot forces concurrent calls to bounce on
                    # HitlConcurrencyError instead.
                    try:
                        await self._on_pending_cleared(req, exc=exc_caught)
                    finally:
                        self._pending = None
                        self._future = None
            except BaseException as exc:
                outcome = _outcome_from_exception(exc)
                raise
            finally:
                # `detached` is a parallel boolean alongside `outcome` —
                # convenient for filtering spans by "this was a cross-process
                # suspend" without re-parsing the outcome string.
                span.set_attribute("hitl.from_resume", from_resume)
                span.set_attribute("hitl.outcome", outcome)
                span.set_attribute("hitl.detached", outcome == "detached")
                span.set_attribute("hitl.duration_seconds", time.monotonic() - t0)

    async def _on_pending_set(self, req: HitlRequest) -> None:
        # Broadcast to subscribers.
        for q in list(self._subscribers):
            q.put_nowait(req)
        # Emit event — guard avoids importing not-yet-defined event types.
        if self._emit is not None:  # pragma: no cover — integration tested
            from cubeloop.agent.types import HitlRequestEvent  # avoid circular

            await self._emit_event(HitlRequestEvent(request=req))

    async def _on_pending_cleared(
        self,
        req: HitlRequest,
        *,
        exc: BaseException | None = None,
    ) -> None:
        # No-op in InMemory; CheckpointedChannel overrides to clear DB row
        # ONLY when the unwind cause is not HitlDetached (which signals a
        # cross-process suspend that must keep persisted pending).
        pass

    async def _emit_event(self, event: StructuredValue) -> None:
        if self._emit is None:  # pragma: no cover — caller already guards this
            return
        res = self._emit(event)
        if asyncio.iscoroutine(res):
            await res

    async def answer(self, question_id: str, answer: StructuredValue) -> None:
        if self._pending is None or self._pending.question_id != question_id:
            raise HitlStaleAnswer(
                f"answer for {question_id}; pending is "
                f"{self._pending.question_id if self._pending else 'None'}"
            )
        await self._save_answer(question_id, answer)
        if self._future is not None and not self._future.done():
            self._future.set_result(answer)
        if self._emit is not None:  # pragma: no cover — integration tested
            from cubeloop.agent.types import HitlAnswerEvent  # avoid circular

            await self._emit_event(
                HitlAnswerEvent(question_id=question_id, answer=answer)
            )

    async def cancel(self, question_id: str, reason: str = "cancelled") -> None:
        if self._pending is None or self._pending.question_id != question_id:
            raise HitlStaleAnswer(
                f"cancel for {question_id}; pending is "
                f"{self._pending.question_id if self._pending else 'None'}"
            )
        if self._future is not None and not self._future.done():
            self._future.set_exception(HitlCancelled(reason))
        if self._emit is not None:  # pragma: no cover — integration tested
            from cubeloop.agent.types import HitlAnswerEvent  # avoid circular

            await self._emit_event(
                HitlAnswerEvent(question_id=question_id, answer=None, cancelled=True)
            )

    # ---- agent-side verbs ----

    async def confirm(
        self,
        prompt: str,
        *,
        details: JsonObject | None = None,
        tool_call_id: str | None = None,
        timeout: TimeoutArg = _USE_DEFAULT_TIMEOUT,
        signal: asyncio.Event | None = None,
    ) -> bool:
        # question_id derives from the prompt content (or tool_call_id, when
        # caller supplies one) — NOT from _resume_slot. If a tool body asked
        # Q_A then Q_B and detached with Q_B pending, respond() pre-loads
        # Q_B's qid+answer; the tool's rerun of confirm(Q_A) derives its own
        # qid and falls through to a fresh await instead of consuming Q_B's
        # slot. See `_derive_question_id` docstring for rationale.
        qid = tool_call_id or self._next_qid(
            "confirm", cast(StructuredValue, {"prompt": prompt, "details": details})
        )
        return cast(
            bool,
            await self._await_answer(
                ConfirmRequest(prompt=prompt, details=details),
                timeout=timeout,
                signal=signal,
                question_id=qid,
            ),
        )

    async def approve(
        self,
        tool_name: str,
        tool_call_id: str,
        args: JsonObject,
        *,
        details: JsonObject | None = None,
        timeout: TimeoutArg = _USE_DEFAULT_TIMEOUT,
        signal: asyncio.Event | None = None,
    ) -> ApproveAnswer:
        return cast(
            ApproveAnswer,
            await self._await_answer(
                ApproveRequest(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    args=args,
                    details=details,
                ),
                timeout=timeout,
                signal=signal,
                question_id=tool_call_id,
            ),
        )

    async def ask(
        self,
        questions: list[Question],
        *,
        timeout: TimeoutArg = _USE_DEFAULT_TIMEOUT,
        signal: asyncio.Event | None = None,
    ) -> dict[str, str | list[str]]:
        # question_id derives from the questions content — NOT from
        # _resume_slot. If a tool body asked Q_A then Q_B and detached with
        # Q_B pending, respond() pre-loads Q_B's qid+answer; the tool's
        # rerun of ask(Q_A) derives its own qid and falls through to a
        # fresh await instead of consuming Q_B's slot. See
        # `_derive_question_id` docstring for rationale and limitations.
        qid = self._next_qid(
            "ask",
            cast(StructuredValue, [q.model_dump(mode="json") for q in questions]),
        )
        return cast(
            dict[str, str | list[str]],
            await self._await_answer(
                AskRequest(questions=questions),
                timeout=timeout,
                signal=signal,
                question_id=qid,
            ),
        )


def _derive_question_id(kind: str, payload_repr: StructuredValue) -> str:
    """Content hash for a HITL request payload.

    Used as the prefix of the final qid (see `_BaseChannel._next_qid`):
    the hash gives resume-after-detach the ability to match by payload
    content, while a per-content sequence counter appended afterwards
    keeps retries of the same prompt distinct from each other.
    """
    data = json.dumps(
        {"kind": kind, "payload": payload_repr},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:32]


def _outcome_from_answer(kind: str, ans: StructuredValue) -> str:
    if kind == "approve" and isinstance(ans, ApproveAnswer):
        return {"approve": "approved", "deny": "denied", "edit": "edited"}.get(
            ans.decision, "answered"
        )
    return "answered"


def _normalize_answer(payload: HitlPayload, ans: StructuredValue) -> StructuredValue:
    if isinstance(payload, ApproveRequest) and not isinstance(ans, ApproveAnswer):
        return ApproveAnswer.model_validate(ans)
    return ans


def _outcome_from_exception(exc: BaseException) -> str:
    from cubeloop.hitl.exceptions import (
        HitlAborted,
        HitlCancelled,
        HitlDetached,
        HitlTimedOut,
    )

    if isinstance(exc, HitlCancelled):
        return "cancelled"
    if isinstance(exc, HitlTimedOut):
        return "timed_out"
    if isinstance(exc, HitlAborted):
        return "aborted"
    if isinstance(exc, HitlDetached):
        return "detached"
    return "error"


class InMemoryChannel(_BaseChannel):
    """In-process HITL channel; no persistence."""

    def __init__(
        self,
        *,
        default_timeout: float | None = None,
        thread_id: str | None = None,
    ) -> None:
        super().__init__(default_timeout=default_timeout, thread_id=thread_id)
        self._answer_ledger: dict[str, StructuredValue] = {}

    async def _load_answer(self, question_id: str) -> StructuredValue | None:
        return self._answer_ledger.get(question_id)

    async def _save_answer(self, question_id: str, answer: StructuredValue) -> None:
        self._answer_ledger[question_id] = answer


class CheckpointedChannel(_BaseChannel):
    """Cross-process HITL channel — persists the pending request via a
    checkpointer so a separate process can call ``Agent.respond()`` after
    detach.

    The checkpointer MUST implement ``save_pending_request`` and
    ``load_pending_request`` (first-party checkpointers do; third-party
    Protocol-only impls may not).
    """

    def __init__(
        self,
        *,
        checkpointer: Checkpointer,
        thread_id: str,
        run_id: str | None = None,
        default_timeout: float | None = None,
        allow_inside_custom_tool: bool = False,
    ) -> None:
        # Validate the checkpointer has the HITL methods early — better to
        # fail at construction than at first ask/approve/confirm (codex pass 3).
        if not (
            hasattr(checkpointer, "save_pending_request")
            and hasattr(checkpointer, "load_pending_request")
            and hasattr(checkpointer, "save_hitl_answer")
            and hasattr(checkpointer, "load_hitl_answer")
            and hasattr(checkpointer, "clear_hitl_answers")
        ):
            from cubeloop.hitl.exceptions import HitlError

            raise HitlError(
                "CheckpointedChannel requires a checkpointer with "
                "save_pending_request, load_pending_request, "
                "save_hitl_answer, load_hitl_answer, and "
                "clear_hitl_answers methods. "
                "First-party checkpointers (Memory/SQLite/Postgres/MySQL) "
                "implement these; third-party Protocol-only impls may not."
            )
        super().__init__(default_timeout=default_timeout, thread_id=thread_id)
        self._checkpointer = checkpointer
        # Optional host-side run identifier (e.g. cubebox run_id) persisted
        # alongside the pending request via save_pending_request. Lets a host
        # recover "which run owns this paused HITL" after worker crash in a
        # single atomic write — no two-step race between pending JSON and
        # run_id columns.
        self._run_id = run_id
        self._allow_inside_custom_tool = allow_inside_custom_tool

    async def _load_answer(self, question_id: str) -> StructuredValue | None:
        assert self._thread_id is not None
        return await self._checkpointer.load_hitl_answer(
            self._thread_id, question_id, run_id=self._run_id
        )

    async def _save_answer(self, question_id: str, answer: StructuredValue) -> None:
        assert self._thread_id is not None
        await self._checkpointer.save_hitl_answer(
            self._thread_id, question_id, answer, run_id=self._run_id
        )

    async def _on_pending_set(self, req: HitlRequest) -> None:
        if _in_custom_tool_var.get() and not self._allow_inside_custom_tool:
            from cubeloop.hitl.exceptions import HitlDurabilityNotGuaranteed

            raise HitlDurabilityNotGuaranteed(
                "CheckpointedChannel called from inside a custom tool body. "
                "Use ApprovalPolicyMiddleware or ask_user_tool, or pass "
                "allow_inside_custom_tool=True to opt in."
            )
        # Only pass run_id when the caller actually set one. Third-party
        # checkpointers that still implement the v2 contract
        # `save_pending_request(thread_id, request)` would raise TypeError
        # if we always passed the new kwarg. With this guard, legacy
        # backends keep working as long as no host opts into run_id
        # persistence on them.
        if self._run_id is not None:
            assert self._thread_id is not None
            await self._checkpointer.save_pending_request(
                self._thread_id, req, run_id=self._run_id
            )
        else:
            assert self._thread_id is not None
            await self._checkpointer.save_pending_request(self._thread_id, req)
        await super()._on_pending_set(req)

    async def _on_pending_cleared(
        self,
        req: HitlRequest,
        *,
        exc: BaseException | None = None,
    ) -> None:
        from cubeloop.hitl.exceptions import HitlDetached

        if isinstance(exc, HitlDetached):
            return
        # request=None clears pending AND run_id atomically.
        assert self._thread_id is not None
        await self._checkpointer.save_pending_request(self._thread_id, None)
