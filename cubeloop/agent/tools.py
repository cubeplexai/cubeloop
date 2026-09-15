from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel
from pydantic import ValidationError

from cubeloop.hitl.exceptions import HitlControlException
from cubeloop.session.turn_execution_context import ToolExecutionBinding

from cubeloop.agent.types import (
    AfterToolCallContext,
    AgentContext,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
    MessageEndEvent,
    MessageStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
)
from cubeloop.utils import emit_event
from cubeloop.providers.base import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from cubeloop.types import JsonObject, StructuredObject, StructuredValue


@dataclass
class ToolCallBatch:
    messages: list[ToolResultMessage]
    terminate: bool


@dataclass
class _PreparedToolCall:
    tool_call: ToolCall
    tool: AgentTool | ToolExecutionBinding
    args: BaseModel | JsonObject
    hitl_trace: StructuredObject | None = None


@dataclass
class _ImmediateOutcome:
    result: AgentToolResult
    is_error: bool
    blocked_by_hook: bool = False
    block_reason: str | None = None
    hitl_trace: StructuredObject | None = None


@dataclass
class _FinalizedOutcome:
    tool_call: ToolCall
    result: AgentToolResult
    is_error: bool
    blocked_by_hook: bool = False
    block_reason: str | None = None
    hitl_trace: StructuredObject | None = None


def _error_result(message: str) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=message)])


def _format_validation_error(exc: ValidationError, tool_name: str) -> str:
    """Render a pydantic ValidationError as text the LLM can act on.

    The default ``str(exc)`` leaks pydantic-internal model names and a
    documentation URL — the model can read it but cannot reliably extract
    the fix from it. This produces one short line per error, naming the
    field path, the constraint, and (for discriminated unions) the allowed
    discriminator values, so the model can correct the call without
    another round-trip.
    """
    lines = [f"Invalid arguments for tool '{tool_name}':"]
    for err in exc.errors():
        loc_parts = [str(p) for p in err.get("loc", ())]
        loc = ".".join(loc_parts) if loc_parts else "<root>"
        etype = err.get("type", "")
        msg = err.get("msg", "")
        ctx = err.get("ctx") or {}

        if etype == "union_tag_not_found":
            # pydantic stores discriminator wrapped in single quotes, e.g.
            # "'operation'" — strip for readability. expected_tags is not
            # populated for this error type; the model has the JSON Schema
            # to enumerate allowed values.
            disc = str(ctx.get("discriminator", "?")).strip("'\"")
            lines.append(f"- {loc}: missing required discriminator key '{disc}'")
        elif etype == "union_tag_invalid":
            disc = str(ctx.get("discriminator", "?")).strip("'\"")
            tag = ctx.get("tag", "")
            allowed = ctx.get("expected_tags") or ""
            lines.append(
                f"- {loc}: discriminator '{disc}'={tag!r} is not one of: {allowed}"
            )
        elif etype == "missing":
            lines.append(f"- {loc}: field required")
        elif etype == "literal_error":
            expected = ctx.get("expected", "")
            lines.append(f"- {loc}: must be one of {expected}")
        elif etype == "extra_forbidden":
            lines.append(f"- {loc}: unexpected field")
        else:
            lines.append(f"- {loc}: {msg}")
    return "\n".join(lines)


def _merge_hitl_details(
    base: StructuredValue, hitl: StructuredObject | None
) -> StructuredValue:
    if hitl is None:
        return base
    if base is None:
        return {"hitl": hitl}
    if isinstance(base, dict):  # pragma: no cover — E2E tested
        merged = dict(base)
        merged["hitl"] = hitl
        return merged
    return {"_non_dict_details": base, "hitl": hitl}


def _make_tool_result_message(
    finalized: _FinalizedOutcome, *, run_id: str | None
) -> ToolResultMessage:
    details = _merge_hitl_details(finalized.result.details, finalized.hitl_trace)
    return ToolResultMessage(
        tool_call_id=finalized.tool_call.id,
        tool_name=finalized.tool_call.name,
        content=finalized.result.content,
        details=details,
        is_error=finalized.is_error,
        timestamp=time.time(),
        run_id=run_id,
    )


async def _resolve_tool_call(
    tool_call: ToolCall,
    context: AgentContext,
    resolve_tool_call: Callable | None,
    signal: asyncio.Event | None,
) -> tuple[ToolCall, bool, _ImmediateOutcome | None]:
    """Give the resolve hook a chance to rewrite the call before anything
    else sees it — events, validation, before/after hooks, and tracing all
    operate on the rewritten call.

    Returns ``(call, was_resolved, resolver_error)``; ``resolver_error`` is
    an immediate error outcome (with the original call for attribution)
    when the resolver raised."""
    if not resolve_tool_call:
        return tool_call, False, None
    try:
        rewritten = await resolve_tool_call(tool_call, context=context, signal=signal)
    except HitlControlException:
        raise
    except Exception as exc:
        return (
            tool_call,
            False,
            _ImmediateOutcome(result=_error_result(str(exc)), is_error=True),
        )
    if rewritten is None:
        return tool_call, False, None
    if rewritten.id != tool_call.id:
        # Enforce the resolver contract: the result message is keyed by the
        # original id on the wire, so a resolver that invents a new id would
        # desynchronize provider-side tool_result correlation.
        rewritten = ToolCall(
            id=tool_call.id, name=rewritten.name, arguments=rewritten.arguments
        )
    turn_context = context.turn_execution_context
    if turn_context is not None and turn_context.binding_for(rewritten.name) is None:
        resolved_tool = next(
            (tool for tool in context.tools or [] if tool.name == rewritten.name),
            None,
        )
        if resolved_tool is not None:
            extended = turn_context.extend(resolved_tool)
            context.turn_execution_context = extended
            if context.on_turn_context is not None:
                callback_result = context.on_turn_context(extended)
                if inspect.isawaitable(callback_result):
                    _ = await callback_result
    return rewritten, True, None


async def _prepare_tool_call(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCall,
    before_tool_call: Callable | None,
    signal: asyncio.Event | None,
    *,
    resolved: bool = False,
) -> _PreparedToolCall | _ImmediateOutcome:
    turn_context = context.turn_execution_context
    binding = (
        turn_context.binding_for(tool_call.name) if turn_context is not None else None
    )
    tool: AgentTool | ToolExecutionBinding | None = binding
    if tool is None and context.tools:
        for t in context.tools:
            if t.name == tool_call.name:
                tool = t
                break

    if tool is None:
        return _ImmediateOutcome(
            result=_error_result(f"Tool {tool_call.name} not found"),
            is_error=True,
        )

    try:
        validated_args = tool.parameters.model_validate(tool_call.arguments)
    except ValidationError as exc:
        message = _format_validation_error(exc, tool.name)
        if resolved:
            # The model never saw a resolved tool's schema in the tools
            # block — append it so a bad call self-corrects in one round
            # trip instead of a blind retry.
            schema = json.dumps(
                tool.parameters.model_json_schema(),
                sort_keys=True,
                ensure_ascii=False,
            )
            message = f"{message}\n\nFull schema for '{tool.name}':\n{schema}"
        return _ImmediateOutcome(
            result=_error_result(message),
            is_error=True,
        )
    except Exception as exc:  # pragma: no cover — defensive
        return _ImmediateOutcome(result=_error_result(str(exc)), is_error=True)

    if before_tool_call:
        try:
            before_ctx = BeforeToolCallContext(
                assistant_message=assistant_message,
                tool_call=tool_call,
                args=validated_args,
                context=context,
            )
            before_result = await before_tool_call(before_ctx, signal=signal)
        except HitlControlException:
            raise
        except Exception as exc:
            return _ImmediateOutcome(result=_error_result(str(exc)), is_error=True)

        if before_result and before_result.block:
            return _ImmediateOutcome(
                result=_error_result(
                    before_result.reason or "Tool execution was blocked"
                ),
                is_error=True,
                blocked_by_hook=True,
                block_reason=before_result.deny_reason or before_result.reason,
                hitl_trace=before_result.hitl_trace,
            )

        if before_result and before_result.edited_args is not None:
            try:
                validated_args = tool.parameters.model_validate(
                    before_result.edited_args
                )
            except ValidationError as exc:  # pragma: no cover — defensive
                return _ImmediateOutcome(
                    result=_error_result(_format_validation_error(exc, tool.name)),
                    is_error=True,
                )

        hitl_trace_carry = before_result.hitl_trace if before_result else None
    else:
        hitl_trace_carry = None

    return _PreparedToolCall(
        tool_call=tool_call,
        tool=tool,
        args=validated_args,
        hitl_trace=hitl_trace_carry,
    )


async def _execute_prepared(
    prepared: _PreparedToolCall,
    signal: asyncio.Event | None,
    emit_fn: Callable,
) -> tuple[AgentToolResult, bool]:
    from cubeloop.hitl.channel import _in_custom_tool_var

    is_builtin = prepared.tool.hitl_builtin
    token = None if is_builtin else _in_custom_tool_var.set(True)
    try:
        try:
            result = await prepared.tool.execute(
                prepared.tool_call.id,
                prepared.args,
                signal=signal,
                on_update=lambda partial: emit_event(
                    emit_fn,
                    ToolExecutionUpdateEvent(
                        tool_call_id=prepared.tool_call.id,
                        tool_name=prepared.tool_call.name,
                        args=prepared.tool_call.arguments,
                        partial_result=partial,
                    ),
                ),
            )
            # Honor an explicit is_error set by the tool body: the pipeline
            # tracks error state as a separate bool, so a tool returning
            # AgentToolResult(is_error=True) without raising must surface here
            # (otherwise the model sees a successful result).
            return result, bool(result.is_error)
        except (
            HitlControlException,
            asyncio.CancelledError,
            KeyboardInterrupt,
            SystemExit,
        ):
            raise
        except BaseException as exc:
            # BaseException, not Exception: a tool body raising a bare
            # BaseException subclass must degrade to an error result too —
            # anything that escapes here detonates the whole batch.
            return _error_result(str(exc)), True
    finally:
        if token is not None:
            _in_custom_tool_var.reset(token)


async def _finalize(
    context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: _PreparedToolCall,
    result: AgentToolResult,
    is_error: bool,
    after_tool_call: Callable | None,
    signal: asyncio.Event | None,
) -> _FinalizedOutcome:
    if after_tool_call:
        try:
            after_ctx = AfterToolCallContext(
                assistant_message=assistant_message,
                tool_call=prepared.tool_call,
                args=prepared.args,
                result=result,
                is_error=is_error,
                context=context,
            )
            after_result = await after_tool_call(after_ctx, signal=signal)
            if after_result:
                result = AgentToolResult(
                    content=(
                        after_result.content
                        if after_result.content is not None
                        else result.content
                    ),
                    details=(
                        after_result.details
                        if after_result.details is not None
                        else result.details
                    ),
                    terminate=(
                        after_result.terminate
                        if after_result.terminate is not None
                        else result.terminate
                    ),
                )
                is_error = (
                    after_result.is_error
                    if after_result.is_error is not None
                    else is_error
                )
        except (
            HitlControlException,
            asyncio.CancelledError,
            KeyboardInterrupt,
            SystemExit,
        ):
            raise
        except BaseException as exc:
            result = _error_result(str(exc))
            is_error = True

    return _FinalizedOutcome(
        tool_call=prepared.tool_call,
        result=result,
        is_error=is_error,
        hitl_trace=prepared.hitl_trace,
    )


def _should_terminate(finalized: list[_FinalizedOutcome]) -> bool:
    return len(finalized) > 0 and all(f.result.terminate is True for f in finalized)


def _execution_mode_for(context: AgentContext, tool_name: str) -> str | None:
    turn_context = context.turn_execution_context
    if turn_context is not None:
        binding = turn_context.binding_for(tool_name)
        if binding is not None:
            return binding.execution_mode
    if context.tools:
        tool = next((tool for tool in context.tools if tool.name == tool_name), None)
        if tool is not None:
            return tool.execution_mode
    return None


async def execute_tool_calls(
    context: AgentContext,
    assistant_message: AssistantMessage,
    *,
    tool_execution: str = "parallel",
    before_tool_call: Callable | None = None,
    resolve_tool_call: Callable | None = None,
    after_tool_call: Callable | None = None,
    signal: asyncio.Event | None = None,
    emit: Callable,
) -> ToolCallBatch:
    tool_calls = [c for c in assistant_message.content if isinstance(c, ToolCall)]

    has_sequential_raw = any(
        _execution_mode_for(context, call.name) == "sequential" for call in tool_calls
    )

    if tool_execution == "sequential" or has_sequential_raw:
        # Already sequential by raw names: resolve lazily, one call at a
        # time, so a later dispatcher call's side effects (e.g. a deferred
        # group loader) never run before an earlier sequential tool that
        # may set up the state they depend on.
        return await _execute_sequential(
            context,
            assistant_message,
            tool_calls,
            before_tool_call,
            resolve_tool_call,
            after_tool_call,
            signal,
            emit,
        )

    # Would-be-parallel batch: no ordering guarantee exists, so resolution
    # may run eagerly. It must — the execution-mode decision has to see the
    # REAL tool names: a dispatcher call targeting a sequential tool routes
    # the whole batch through the sequential executor.
    resolutions: list[tuple[ToolCall, bool, _ImmediateOutcome | None]] = []
    for tc in tool_calls:
        resolutions.append(
            await _resolve_tool_call(tc, context, resolve_tool_call, signal)
        )

    has_sequential_resolved = any(
        _execution_mode_for(context, call.name) == "sequential"
        for call, _, _ in resolutions
    )

    if has_sequential_resolved:
        return await _execute_sequential(
            context,
            assistant_message,
            tool_calls,
            before_tool_call,
            resolve_tool_call,
            after_tool_call,
            signal,
            emit,
            preresolved=resolutions,
        )
    return await _execute_parallel(
        context,
        assistant_message,
        resolutions,
        before_tool_call,
        after_tool_call,
        signal,
        emit,
    )


async def _execute_sequential(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    before_tool_call: Callable | None,
    resolve_tool_call: Callable | None,
    after_tool_call: Callable | None,
    signal: asyncio.Event | None,
    emit_fn: Callable,
    *,
    preresolved: list[tuple[ToolCall, bool, _ImmediateOutcome | None]] | None = None,
) -> ToolCallBatch:
    finalized_list: list[_FinalizedOutcome] = []
    messages: list[ToolResultMessage] = []

    try:
        for idx, tc in enumerate(tool_calls):
            if preresolved is not None:
                rtc, was_resolved, resolver_error = preresolved[idx]
            else:
                # Lazy: resolve only when this call's turn comes, after every
                # earlier call in the batch has fully executed.
                rtc, was_resolved, resolver_error = await _resolve_tool_call(
                    tc, context, resolve_tool_call, signal
                )

            await emit_event(
                emit_fn,
                ToolExecutionStartEvent(
                    tool_call_id=rtc.id, tool_name=rtc.name, args=rtc.arguments
                ),
            )

            preparation = resolver_error or await _prepare_tool_call(
                context,
                assistant_message,
                rtc,
                before_tool_call,
                signal,
                resolved=was_resolved,
            )

            if isinstance(preparation, _ImmediateOutcome):
                finalized = _FinalizedOutcome(
                    tool_call=rtc,
                    result=preparation.result,
                    is_error=preparation.is_error,
                    blocked_by_hook=preparation.blocked_by_hook,
                    block_reason=preparation.block_reason,
                    hitl_trace=preparation.hitl_trace,
                )
            else:
                result, is_error = await _execute_prepared(preparation, signal, emit_fn)
                finalized = await _finalize(
                    context,
                    assistant_message,
                    preparation,
                    result,
                    is_error,
                    after_tool_call,
                    signal,
                )

            await emit_event(
                emit_fn,
                ToolExecutionEndEvent(
                    tool_call_id=rtc.id,
                    tool_name=rtc.name,
                    result=finalized.result,
                    is_error=finalized.is_error,
                    terminate=bool(finalized.result.terminate),
                    blocked_by_hook=finalized.blocked_by_hook,
                    block_reason=finalized.block_reason,
                ),
            )
            tool_msg = _make_tool_result_message(
                finalized, run_id=assistant_message.run_id
            )
            await emit_event(emit_fn, MessageStartEvent(message=tool_msg))
            await emit_event(emit_fn, MessageEndEvent(message=tool_msg))
            finalized_list.append(finalized)
            messages.append(tool_msg)

    except HitlControlException as exc:
        # Completed calls' results were already emitted (and
        # checkpointed) per-iteration; carry them on the exception so
        # the stateless loop entry points can return them to callers
        # that persist the loop's return value across the suspend.
        exc.partial_tool_results = tuple(messages)
        raise

    return ToolCallBatch(messages=messages, terminate=_should_terminate(finalized_list))


async def _execute_parallel(
    context: AgentContext,
    assistant_message: AssistantMessage,
    resolutions: list[tuple[ToolCall, bool, _ImmediateOutcome | None]],
    before_tool_call: Callable | None,
    after_tool_call: Callable | None,
    signal: asyncio.Event | None,
    emit_fn: Callable,
) -> ToolCallBatch:
    # Two-phase: prepare ALL tool calls (which may raise HitlDetached or other
    # HitlControlException from before_tool_call) BEFORE scheduling any
    # asyncio.create_task or emitting any ToolExecutionStartEvent. Otherwise:
    # (a) a later prepare detach would leak already-started tool tasks
    #     (side effects happen but ToolResultMessage is never emitted/
    #     checkpointed, so the resumed run duplicates the side effects);
    # (b) Start events emitted in the prepare loop would have no matching
    #     End event when prepare later raised, leaving state.pending_tool_calls
    #     and trace spans permanently open.
    entries: list[_FinalizedOutcome | _PreparedToolCall] = []
    for rtc, was_resolved, resolver_error in resolutions:
        preparation = resolver_error or await _prepare_tool_call(
            context,
            assistant_message,
            rtc,
            before_tool_call,
            signal,
            resolved=was_resolved,
        )

        if isinstance(preparation, _ImmediateOutcome):
            # Immediate outcomes get a paired Start+End right here (the
            # "execution" was the prepare step itself — e.g. blocked by hook
            # or unknown tool). Pairing keeps the event stream balanced
            # even though no real tool body runs.
            await emit_event(
                emit_fn,
                ToolExecutionStartEvent(
                    tool_call_id=rtc.id, tool_name=rtc.name, args=rtc.arguments
                ),
            )
            finalized = _FinalizedOutcome(
                tool_call=rtc,
                result=preparation.result,
                is_error=preparation.is_error,
                blocked_by_hook=preparation.blocked_by_hook,
                block_reason=preparation.block_reason,
                hitl_trace=preparation.hitl_trace,
            )
            await emit_event(
                emit_fn,
                ToolExecutionEndEvent(
                    tool_call_id=rtc.id,
                    tool_name=rtc.name,
                    result=finalized.result,
                    is_error=finalized.is_error,
                    terminate=bool(finalized.result.terminate),
                    blocked_by_hook=finalized.blocked_by_hook,
                    block_reason=finalized.block_reason,
                ),
            )
            entries.append(finalized)
        else:
            entries.append(preparation)

    async def _run(prep: _PreparedToolCall) -> _FinalizedOutcome:
        # Start event lives inside _run so it is emitted only for tools
        # that actually get scheduled. If a later prepare raises before
        # we get here, no Start is leaked.
        await emit_event(
            emit_fn,
            ToolExecutionStartEvent(
                tool_call_id=prep.tool_call.id,
                tool_name=prep.tool_call.name,
                args=prep.tool_call.arguments,
            ),
        )
        result, is_error = await _execute_prepared(prep, signal, emit_fn)
        fin = await _finalize(
            context,
            assistant_message,
            prep,
            result,
            is_error,
            after_tool_call,
            signal,
        )
        await emit_event(
            emit_fn,
            ToolExecutionEndEvent(
                tool_call_id=prep.tool_call.id,
                tool_name=prep.tool_call.name,
                result=fin.result,
                is_error=fin.is_error,
                terminate=bool(fin.result.terminate),
                blocked_by_hook=fin.blocked_by_hook,
                block_reason=fin.block_reason,
            ),
        )
        return fin

    # Now that every prepare has succeeded, fan out the executions.
    # `scheduled` stays index-aligned with `entries` so a failed task's
    # _PreparedToolCall (tool_call id/name, hitl_trace) is recoverable when
    # synthesizing its error outcome.
    scheduled: list[_FinalizedOutcome | asyncio.Task] = [
        entry
        if isinstance(entry, _FinalizedOutcome)
        else asyncio.create_task(_run(entry))
        for entry in entries
    ]
    tasks = [s for s in scheduled if isinstance(s, asyncio.Task)]

    # Settle EVERY task before processing any outcome: one failing tool must
    # never drop its siblings' results (they were computed, and their side
    # effects happened — losing the ToolResultMessage would leave dangling
    # tool_calls in the checkpoint and duplicate the side effects on resume).
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        # Outer cancel: gather already propagated it to the children.
        # Settle them, salvage the results of tools that completed before
        # the cancel landed, then re-raise. The Agent layer backfills the
        # genuinely-unanswered ids (_complete_cancelled_tool_calls); without
        # the salvage it would stamp "[cancelled]" over real, side-effecting
        # completions. Best-effort: never mask the CancelledError.
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
            salvaged = [
                s if isinstance(s, _FinalizedOutcome) else s.result()
                for s in scheduled
                if isinstance(s, _FinalizedOutcome)
                or (s.done() and not s.cancelled() and s.exception() is None)
            ]
            await _emit_tool_result_messages(
                salvaged, emit_fn, run_id=assistant_message.run_id
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            # Salvage is best-effort by contract: a failure while emitting
            # already-completed results must never mask the CancelledError
            # being re-raised below (mirrors _complete_cancelled_tool_calls).
            pass
        raise

    finalized_list: list[_FinalizedOutcome] = []
    control_exc: BaseException | None = None
    framework_exc: BaseException | None = None
    for entry, slot in zip(entries, scheduled):
        if isinstance(slot, _FinalizedOutcome):
            finalized_list.append(slot)
            continue
        if slot.cancelled():
            # Task.exception() would re-raise the CancelledError; outer
            # cancellation re-raised above, so this is a tool self-cancel.
            exc: BaseException | None = asyncio.CancelledError()
        else:
            exc = slot.exception()
        if exc is None:
            finalized_list.append(slot.result())
            continue
        assert isinstance(entry, _PreparedToolCall)
        if isinstance(exc, (HitlControlException, KeyboardInterrupt, SystemExit)):
            # Must propagate (suspend/interpreter contracts) — but only
            # after every sibling result below has been emitted. The call
            # itself deliberately stays unanswered, with no End event:
            # same shape as a sequential detach; the HITL resume/abort
            # paths backfill it. First raiser (batch order) wins.
            if control_exc is None:
                control_exc = exc
            continue
        if not isinstance(exc, asyncio.CancelledError):
            # Tool and hook failures are converted to error results at the
            # source (_execute_prepared/_finalize), so anything else landing
            # here is framework-side — typically emit_fn raising while
            # processing a Start/End event. Emitter failures propagate at
            # every other emit_event call site; synthesizing a bogus
            # tool_result here would let the run continue with event
            # processing broken. Deferred until every task has settled.
            if framework_exc is None:
                framework_exc = exc
            continue
        # Per-task isolation: a stray CancelledError (tool self-cancel with
        # no outer cancel — a tool bug) degrades to an error result for
        # THIS call only.
        synthesized = _FinalizedOutcome(
            tool_call=entry.tool_call,
            result=_error_result("[Tool execution cancelled]"),
            is_error=True,
            hitl_trace=entry.hitl_trace,
        )
        # Pair the Start emitted inside _run — an unpaired Start leaves
        # state.pending_tool_calls and trace spans open.
        await emit_event(
            emit_fn,
            ToolExecutionEndEvent(
                tool_call_id=entry.tool_call.id,
                tool_name=entry.tool_call.name,
                result=synthesized.result,
                is_error=True,
                terminate=False,
            ),
        )
        finalized_list.append(synthesized)

    if framework_exc is not None:
        # Event processing is broken, so emitting sibling results below
        # would fail too. Every task has already settled (no leaks) —
        # propagate, taking precedence over a suspend: durably suspending
        # a run whose event pipeline is failing is not safe.
        raise framework_exc

    if control_exc is not None:
        # Persist the siblings BEFORE propagating the suspend (each
        # MessageEndEvent checkpoints immediately). Deliberately NOT
        # best-effort: suspending after a sibling's persistence failed
        # would durably record a batch whose completed work is missing —
        # resume would re-run tools whose side effects already happened.
        # A persistence failure propagates instead (consistent with every
        # other MessageEndEvent site): the run fails rather than suspends.
        emitted = await _emit_tool_result_messages(
            finalized_list, emit_fn, run_id=assistant_message.run_id
        )
        if isinstance(control_exc, HitlControlException):
            # Stateless loop entry points append these to the message
            # lists they return, so callers persisting the return value
            # keep the completed siblings' results across the suspend.
            control_exc.partial_tool_results = tuple(emitted)
        raise control_exc

    messages = await _emit_tool_result_messages(
        finalized_list, emit_fn, run_id=assistant_message.run_id
    )
    return ToolCallBatch(messages=messages, terminate=_should_terminate(finalized_list))


async def _emit_tool_result_messages(
    finalized_list: list[_FinalizedOutcome],
    emit_fn: Callable,
    *,
    run_id: str | None,
) -> list[ToolResultMessage]:
    messages: list[ToolResultMessage] = []
    for finalized in finalized_list:
        tool_msg = _make_tool_result_message(finalized, run_id=run_id)
        await emit_event(emit_fn, MessageStartEvent(message=tool_msg))
        await emit_event(emit_fn, MessageEndEvent(message=tool_msg))
        messages.append(tool_msg)
    return messages
