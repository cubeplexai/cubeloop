from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from cubeloop.errors import (
    ContentFiltered,
    ContextLengthExceeded,
    ProviderBadRequest,
    ProviderError,
    ProviderUnavailable,
    RateLimited,
    error_from_stream_fields,
)
from cubeloop.providers.base import (
    AssistantMessage,
    BoundModel,
    Message,
    MessageStream,
    Model,
    Provider,
    ReasoningControl,
    StreamEvent,
    StreamOptions,
    ToolChoice,
    ToolDefinition,
    Usage,
    chain_providers,  # re-exported for back-compat; canonical home is base.py
)


_log = logging.getLogger("cubeloop.providers.fallback")


DEFAULT_RETRY_ERRORS: frozenset[type[ProviderError]] = frozenset(
    {RateLimited, ProviderUnavailable}
)

DEFAULT_TRIGGER_ERRORS: frozenset[type[ProviderError]] = frozenset(
    {
        RateLimited,
        ProviderUnavailable,
        ContextLengthExceeded,
        ProviderBadRequest,
    }
)

# ContentFiltered / ProviderAuthFailed stay fail-closed unless the caller
# opts in. ModelNotFound inherits ProviderBadRequest so it is included.


_FallbackOnFailover = Callable[
    [BoundModel, BoundModel | None, BaseException | str], Awaitable[None] | None
]
_FallbackOnRetry = Callable[
    [BoundModel, BaseException, int, float], Awaitable[None] | None
]


class _FallbackRunState:
    """Per-task sticky index. Isolated across concurrent agent runs."""

    __slots__ = ("active_index",)

    def __init__(self) -> None:
        self.active_index = 0


_run_state: contextvars.ContextVar[_FallbackRunState | None] = contextvars.ContextVar(
    "cubepi_fallback_run_state", default=None
)
_attempt_index: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "cubeloop_fallback_attempt_index", default=None
)


def begin_fallback_run() -> contextvars.Token[_FallbackRunState | None]:
    """Reset sticky state for a new agent run / user turn. Returns a reset token."""

    return _run_state.set(_FallbackRunState())


def end_fallback_run(token: contextvars.Token[_FallbackRunState | None]) -> None:
    _run_state.reset(token)


def reset_active() -> None:
    """Point the current run back at ``chain[0]`` (no-op if no run state)."""

    state = _run_state.get()
    if state is not None:
        state.active_index = 0


def get_active_index() -> int:
    """Return the current sticky index, or 0 if no run is open."""

    state = _run_state.get()
    return 0 if state is None else state.active_index


def get_attempt_index() -> int | None:
    """Return the fallback leg currently entering its bound model."""
    return _attempt_index.get()


def set_active_index(index: int) -> None:
    """Restore a previously snapshotted sticky index into the current run."""

    state = _run_state.get()
    if state is not None:
        state.active_index = max(0, index)


def _current_state() -> _FallbackRunState | None:
    return _run_state.get()


def _is_trigger(err: BaseException, trigger: tuple[type[ProviderError], ...]) -> bool:
    # ContentFiltered is a ProviderBadRequest subclass but stays fail-closed
    # unless the caller explicitly lists it in trigger_errors.
    if isinstance(err, ContentFiltered):
        return ContentFiltered in trigger
    return isinstance(err, trigger)


def _is_retryable(err: BaseException, retry: tuple[type[ProviderError], ...]) -> bool:
    return isinstance(err, retry)


def _context_too_small(err: BaseException, bound: BoundModel) -> bool:
    if not isinstance(err, ContextLengthExceeded):
        return False
    need = err.tokens_in
    window = bound.spec.context_window or None
    if need is None or not window:
        return False
    return int(window) < int(need)


def _typed_from_event(event: StreamEvent, bound: BoundModel) -> ProviderError:
    return error_from_stream_fields(
        error_message=event.error_message,
        error_type=event.error_type,
        error_code=event.error_code,
        status_code=event.status_code,
        retry_after=event.retry_after,
        provider_id=event.provider_id or bound.spec.provider_id,
        model_id=event.model_id or bound.spec.id,
        tokens_in=event.tokens_in,
        context_window=event.context_window,
    )


def _typed_from_message(msg: AssistantMessage, bound: BoundModel) -> ProviderError:
    return error_from_stream_fields(
        error_message=msg.error_message,
        error_type=msg.error_type,
        error_code=msg.error_code,
        status_code=msg.status_code,
        retry_after=msg.retry_after,
        provider_id=msg.provider_id or bound.spec.provider_id,
        model_id=msg.model_id or bound.spec.id,
        tokens_in=msg.tokens_in,
        context_window=msg.context_window,
    )


def _window_cannot_fit(bound: BoundModel, tokens_in: int | None) -> bool:
    window = bound.spec.context_window or None
    if tokens_in is None or not window:
        return False
    return int(window) < int(tokens_in)


def _needed_tokens(err: BaseException | None) -> int | None:
    if isinstance(err, ContextLengthExceeded):
        return err.tokens_in
    return None


@dataclass(frozen=True)
class FailoverAttempt:
    model_id: str
    provider_id: str
    error: BaseException
    duration_ms: float
    attempt: int
    phase: str  # "retry" | "failover"


@dataclass(frozen=True)
class FallbackBoundModel:
    """Ordered chain of BoundModels — retry the active leg, then hop.

    chain[0] is the user-selected primary. Transient errors
    (``DEFAULT_RETRY_ERRORS``) are retried on the same model before the next
    chain entry is tried. Residual ``ProviderBadRequest`` / ``ModelNotFound``
    hop without same-model retry. Mid-stream errors (after the first
    non-error event) are forwarded as-is.

    Sticky active leg: after a successful response from chain[i], later
    ``stream`` / ``generate`` calls in the same agent run (or same task, when
    used standalone) start at *i*. A new ``Agent.prompt`` resets to chain[0]
    via :func:`begin_fallback_run`.

    provider and spec proxy chain[0] so tracing/billing code that reads
    agent._model.provider / agent._model.spec continues to work unchanged.
    """

    chain: tuple[BoundModel, ...]
    trigger_errors: frozenset[type[ProviderError]] = DEFAULT_TRIGGER_ERRORS
    retry_errors: frozenset[type[ProviderError]] = DEFAULT_RETRY_ERRORS
    max_retries_per_model: int = 3
    max_retry_after: float = 5.0
    retry_backoff: float = 0.0
    sticky_within_run: bool = True
    on_failover: _FallbackOnFailover | None = None
    on_retry: _FallbackOnRetry | None = None

    def __post_init__(self) -> None:
        if not self.chain:
            raise ValueError(
                "FallbackBoundModel.chain must contain at least one BoundModel"
            )
        if self.max_retries_per_model < 0:
            raise ValueError("max_retries_per_model must be >= 0")

    @property
    def provider(self) -> Provider:
        return self.chain[0].provider

    @property
    def spec(self) -> Model:
        return self.chain[0].spec

    def reset_active(self) -> None:
        """Reset the current-run sticky pointer to ``chain[0]``."""

        reset_active()

    async def _notify_failover(
        self,
        failed: BoundModel,
        next_model: BoundModel | None,
        error: BaseException | str,
        attempt: int,
    ) -> None:
        failed_label = f"{failed.spec.provider_id}/{failed.spec.id}"
        next_label = (
            f"{next_model.spec.provider_id}/{next_model.spec.id}"
            if next_model
            else "none (exhausted)"
        )
        _log.warning(
            "cubeloop.providers.fallback: failover triggered  "
            "failed=%s  →  next=%s  reason=%s  attempt=%s/%s",
            failed_label,
            next_label,
            error,
            attempt,
            len(self.chain),
        )
        if self.on_failover is not None:
            try:
                result = self.on_failover(failed, next_model, error)
                if inspect.isawaitable(result):
                    await result
            except Exception as cb_exc:  # noqa: BLE001
                _log.warning(
                    "cubeloop.providers.fallback: on_failover callback raised; swallowed: %s",
                    cb_exc,
                )

    async def _notify_retry(
        self,
        failed: BoundModel,
        error: BaseException,
        attempt: int,
        wait_s: float,
    ) -> None:
        _log.info(
            "cubeloop.providers.fallback: same-model retry  model=%s/%s  "
            "attempt=%s/%s  wait=%.2fs  reason=%s",
            failed.spec.provider_id,
            failed.spec.id,
            attempt,
            self.max_retries_per_model,
            wait_s,
            error,
        )
        if self.on_retry is not None:
            try:
                result = self.on_retry(failed, error, attempt, wait_s)
                if inspect.isawaitable(result):
                    await result
            except Exception as cb_exc:  # noqa: BLE001
                _log.warning(
                    "cubeloop.providers.fallback: on_retry callback raised; swallowed: %s",
                    cb_exc,
                )

    def _wait_s(self, err: BaseException, retry_i: int) -> float:
        wait = self.retry_backoff * (2 ** max(0, retry_i - 1))
        if isinstance(err, RateLimited) and err.retry_after is not None:
            wait = max(wait, float(err.retry_after))
        return min(wait, self.max_retry_after)

    def _start_index(self) -> int:
        if not self.sticky_within_run:
            return 0
        state = _current_state()
        if state is None:
            return 0
        idx = state.active_index
        if idx < 0 or idx >= len(self.chain):
            return 0
        return idx

    def _stick(self, index: int) -> None:
        if not self.sticky_within_run:
            return
        state = _current_state()
        if state is not None:
            state.active_index = index

    def _exhaust(self, failures: list[FailoverAttempt]) -> ProviderUnavailable:
        last = failures[-1].error if failures else "no providers in chain"
        last_typed = next(
            (f.error for f in reversed(failures) if isinstance(f.error, ProviderError)),
            None,
        )
        by_leg: dict[tuple[str, str], BaseException] = {}
        for item in failures:
            by_leg[(item.provider_id, item.model_id)] = item.error
        exc = ProviderUnavailable(
            f"all providers exhausted; last error: {last!r}",
            errors=list(by_leg.values()),
        )
        if last_typed is not None:
            exc.__cause__ = last_typed
        return exc

    def _record(
        self,
        failures: list[FailoverAttempt],
        bound: BoundModel,
        err: BaseException,
        duration_ms: float,
        attempt: int,
        phase: str,
    ) -> None:
        failures.append(
            FailoverAttempt(
                model_id=bound.spec.id,
                provider_id=bound.spec.provider_id,
                error=err,
                duration_ms=duration_ms,
                attempt=attempt,
                phase=phase,
            )
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream:
        trigger = tuple(self.trigger_errors)
        retry = tuple(self.retry_errors)
        failures: list[FailoverAttempt] = []
        start = self._start_index()
        needed: int | None = None

        for index in range(start, len(self.chain)):
            bound = self.chain[index]
            next_bound = self.chain[index + 1] if index + 1 < len(self.chain) else None
            same_retries = 0

            if _window_cannot_fit(bound, needed):
                skip = ContextLengthExceeded(
                    "skipped: context_window cannot fit prior tokens_in",
                    provider=bound.spec.provider_id,
                    model=bound.spec.id,
                    tokens_in=needed,
                    context_window=bound.spec.context_window,
                )
                self._record(failures, bound, skip, 0.0, 0, "failover")
                await self._notify_failover(bound, next_bound, skip, index + 1)
                continue

            while True:
                t0 = time.monotonic()
                err: BaseException | None = None
                first: StreamEvent | None = None
                try:
                    attempt_token = _attempt_index.set(index)
                    try:
                        inner = await bound.stream(
                            messages,
                            system_prompt=system_prompt,
                            tools=tools,
                            tool_choice=tool_choice,
                            options=options,
                        )
                    finally:
                        _attempt_index.reset(attempt_token)
                except Exception as exc:
                    err = exc
                else:
                    iterator = inner.__aiter__()
                    try:
                        first = await iterator.__anext__()
                    except StopAsyncIteration:
                        err = ProviderUnavailable(
                            "stream ended before producing any events"
                        )
                    else:
                        if first.type == "error":
                            err = _typed_from_event(first, bound)
                        else:
                            self._stick(index)
                            outer = MessageStream()

                            async def _forward(
                                first_ev: StreamEvent = first,
                                src: Any = iterator,
                                src_stream: MessageStream = inner,
                                out: MessageStream = outer,
                            ) -> None:
                                try:
                                    out.push(first_ev)
                                    async for ev in src:
                                        out.push(ev)
                                    out.set_result(await src_stream.result())
                                except BaseException as fwd_exc:  # noqa: BLE001
                                    err_msg = AssistantMessage(
                                        content=[],
                                        stop_reason="error",
                                        error_message=str(fwd_exc),
                                        usage=Usage(),
                                        timestamp=time.time(),
                                    )
                                    out.push(
                                        StreamEvent(
                                            type="error",
                                            error_message=str(fwd_exc),
                                        )
                                    )
                                    out.set_result(err_msg)
                                    if not isinstance(fwd_exc, Exception):
                                        raise

                            outer.attach_task(asyncio.create_task(_forward()))
                            return outer

                assert err is not None
                needed = _needed_tokens(err) or needed
                duration_ms = (time.monotonic() - t0) * 1000
                hop = await self._decide(
                    bound=bound,
                    next_bound=next_bound,
                    err=err,
                    duration_ms=duration_ms,
                    same_retries=same_retries,
                    failures=failures,
                    trigger=trigger,
                    retry=retry,
                    chain_pos=index + 1,
                )
                if hop == "retry":
                    same_retries += 1
                    continue
                if hop == "failover":
                    break
                raise err

        raise self._exhaust(failures)

    async def generate(
        self,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: StreamOptions | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        reasoning: ReasoningControl | None = None,
    ) -> AssistantMessage:
        trigger = tuple(self.trigger_errors)
        retry = tuple(self.retry_errors)
        failures: list[FailoverAttempt] = []
        start = self._start_index()
        needed: int | None = None

        for index in range(start, len(self.chain)):
            bound = self.chain[index]
            next_bound = self.chain[index + 1] if index + 1 < len(self.chain) else None
            same_retries = 0

            if _window_cannot_fit(bound, needed):
                skip = ContextLengthExceeded(
                    "skipped: context_window cannot fit prior tokens_in",
                    provider=bound.spec.provider_id,
                    model=bound.spec.id,
                    tokens_in=needed,
                    context_window=bound.spec.context_window,
                )
                self._record(failures, bound, skip, 0.0, 0, "failover")
                await self._notify_failover(bound, next_bound, skip, index + 1)
                continue

            while True:
                t0 = time.monotonic()
                err: BaseException | None = None
                try:
                    result = await bound.generate(
                        messages,
                        system_prompt=system_prompt,
                        tools=tools,
                        tool_choice=tool_choice,
                        options=options,
                        max_output_tokens=max_output_tokens,
                        temperature=temperature,
                        reasoning=reasoning,
                    )
                except Exception as exc:
                    err = exc
                else:
                    if result.stop_reason == "error":
                        err = _typed_from_message(result, bound)
                    else:
                        self._stick(index)
                        return result

                assert err is not None
                needed = _needed_tokens(err) or needed
                duration_ms = (time.monotonic() - t0) * 1000
                hop = await self._decide(
                    bound=bound,
                    next_bound=next_bound,
                    err=err,
                    duration_ms=duration_ms,
                    same_retries=same_retries,
                    failures=failures,
                    trigger=trigger,
                    retry=retry,
                    chain_pos=index + 1,
                )
                if hop == "retry":
                    same_retries += 1
                    continue
                if hop == "failover":
                    break
                raise err

        raise self._exhaust(failures)

    async def _decide(
        self,
        *,
        bound: BoundModel,
        next_bound: BoundModel | None,
        err: BaseException,
        duration_ms: float,
        same_retries: int,
        failures: list[FailoverAttempt],
        trigger: tuple[type[ProviderError], ...],
        retry: tuple[type[ProviderError], ...],
        chain_pos: int,
    ) -> str:
        if _context_too_small(err, bound):
            self._record(failures, bound, err, duration_ms, same_retries, "failover")
            await self._notify_failover(bound, next_bound, err, chain_pos)
            return "failover"

        if _is_retryable(err, retry) and same_retries < self.max_retries_per_model:
            wait_s = self._wait_s(err, same_retries + 1)
            self._record(failures, bound, err, duration_ms, same_retries + 1, "retry")
            await self._notify_retry(bound, err, same_retries + 1, wait_s)
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            return "retry"

        if _is_trigger(err, trigger):
            self._record(failures, bound, err, duration_ms, same_retries, "failover")
            await self._notify_failover(bound, next_bound, err, chain_pos)
            return "failover"

        return "raise"


# ``chain_providers`` lives in :mod:`cubeloop.providers.base` so the tracing /
# meter modules can import it without pulling in this fallback module. The
# import above re-exports it under ``cubeloop.providers.fallback.chain_providers``
# for back-compat with existing call sites.
__all__ = [
    "DEFAULT_RETRY_ERRORS",
    "DEFAULT_TRIGGER_ERRORS",
    "FailoverAttempt",
    "FallbackBoundModel",
    "begin_fallback_run",
    "end_fallback_run",
    "reset_active",
    "chain_providers",
]
