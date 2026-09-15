from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Callable

from cubeloop.agent.tools import execute_tool_calls
from cubeloop.agent.types import (
    AgentContext,
    AgentEndEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ShouldStopAfterTurnContext,
    TurnEndEvent,
    TurnStartEvent,
)
from cubeloop.checkpointer.base import Checkpointer
from cubeloop.hitl.exceptions import HitlAborted, HitlDetached
from cubeloop.utils import emit_event
from cubeloop.providers.base import (
    AssistantMessage,
    BoundModel,
    Message,
    StreamOptions,
    ToolCall,
    ToolResultMessage,
)

if TYPE_CHECKING:
    from cubeloop.providers.fallback import FallbackBoundModel

_MAX_ON_RUN_END_CYCLES = 100


async def run_agent_loop(
    *,
    prompts: list[Message],
    context: AgentContext,
    model: BoundModel | FallbackBoundModel,
    convert_to_llm: Callable,
    emit: Callable,
    transform_context: Callable | None = None,
    transform_system_prompt: Callable | None = None,
    after_model_response: Callable | None = None,
    before_tool_call: Callable | None = None,
    resolve_tool_call: Callable | None = None,
    after_tool_call: Callable | None = None,
    should_stop_after_turn: Callable | None = None,
    get_steering_messages: Callable | None = None,
    get_follow_up_messages: Callable | None = None,
    on_run_end: Callable | None = None,
    stream_options: StreamOptions | None = None,
    tool_execution: str = "parallel",
    system_prompt: str | None = None,
    set_outcome: Callable[[str], None] | None = None,
) -> list[Message]:
    new_messages: list[Message] = list(prompts)
    current_context = AgentContext(
        system_prompt=system_prompt
        if system_prompt is not None
        else context.system_prompt,
        messages=list(context.messages) + list(prompts),
        tools=context.tools,
        extra=context.extra,
        run_id=context.run_id,
        attempt_id=context.attempt_id,
        on_turn_context=context.on_turn_context,
        set_input_admission=context.set_input_admission,
    )

    await emit_event(emit, AgentStartEvent())
    await emit_event(emit, TurnStartEvent())
    for prompt in prompts:
        await emit_event(emit, MessageStartEvent(message=prompt))
        await emit_event(emit, MessageEndEvent(message=prompt))

    await _run_loop(
        current_context=current_context,
        new_messages=new_messages,
        model=model,
        convert_to_llm=convert_to_llm,
        transform_context=transform_context,
        transform_system_prompt=transform_system_prompt,
        after_model_response=after_model_response,
        before_tool_call=before_tool_call,
        resolve_tool_call=resolve_tool_call,
        after_tool_call=after_tool_call,
        should_stop_after_turn=should_stop_after_turn,
        get_steering_messages=get_steering_messages,
        get_follow_up_messages=get_follow_up_messages,
        on_run_end=on_run_end,
        stream_options=stream_options,
        tool_execution=tool_execution,
        emit=emit,
        set_outcome=set_outcome,
    )
    return new_messages


async def run_agent_loop_continue(
    *,
    context: AgentContext,
    model: BoundModel | FallbackBoundModel,
    convert_to_llm: Callable,
    emit: Callable,
    transform_context: Callable | None = None,
    transform_system_prompt: Callable | None = None,
    after_model_response: Callable | None = None,
    before_tool_call: Callable | None = None,
    resolve_tool_call: Callable | None = None,
    after_tool_call: Callable | None = None,
    should_stop_after_turn: Callable | None = None,
    get_steering_messages: Callable | None = None,
    get_follow_up_messages: Callable | None = None,
    on_run_end: Callable | None = None,
    stream_options: StreamOptions | None = None,
    tool_execution: str = "parallel",
    system_prompt: str | None = None,
    set_outcome: Callable[[str], None] | None = None,
) -> list[Message]:
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")

    new_messages: list[Message] = []
    current_context = AgentContext(
        system_prompt=system_prompt
        if system_prompt is not None
        else context.system_prompt,
        messages=list(context.messages),
        tools=context.tools,
        extra=context.extra,
        run_id=context.run_id,
        attempt_id=context.attempt_id,
        on_turn_context=context.on_turn_context,
        set_input_admission=context.set_input_admission,
    )

    await emit_event(emit, AgentStartEvent())
    await emit_event(emit, TurnStartEvent())

    await _run_loop(
        current_context=current_context,
        new_messages=new_messages,
        model=model,
        convert_to_llm=convert_to_llm,
        transform_context=transform_context,
        transform_system_prompt=transform_system_prompt,
        after_model_response=after_model_response,
        before_tool_call=before_tool_call,
        resolve_tool_call=resolve_tool_call,
        after_tool_call=after_tool_call,
        should_stop_after_turn=should_stop_after_turn,
        get_steering_messages=get_steering_messages,
        get_follow_up_messages=get_follow_up_messages,
        on_run_end=on_run_end,
        stream_options=stream_options,
        tool_execution=tool_execution,
        emit=emit,
        set_outcome=set_outcome,
    )
    return new_messages


async def run_agent_loop_resume(
    *,
    context: AgentContext,
    model: BoundModel | FallbackBoundModel,
    convert_to_llm: Callable,
    emit: Callable,
    transform_context: Callable | None = None,
    transform_system_prompt: Callable | None = None,
    after_model_response: Callable | None = None,
    before_tool_call: Callable | None = None,
    resolve_tool_call: Callable | None = None,
    after_tool_call: Callable | None = None,
    should_stop_after_turn: Callable | None = None,
    get_steering_messages: Callable | None = None,
    get_follow_up_messages: Callable | None = None,
    on_run_end: Callable | None = None,
    stream_options: StreamOptions | None = None,
    tool_execution: str = "parallel",
    system_prompt: str | None = None,
    checkpointer: Checkpointer | None = None,
    thread_id: str | None = None,
    set_outcome: Callable[[str], None] | None = None,
) -> list[Message]:
    new_messages: list[Message] = []

    try:
        return await _run_agent_loop_resume_body(
            context=context,
            model=model,
            convert_to_llm=convert_to_llm,
            emit=emit,
            transform_context=transform_context,
            transform_system_prompt=transform_system_prompt,
            after_model_response=after_model_response,
            before_tool_call=before_tool_call,
            resolve_tool_call=resolve_tool_call,
            after_tool_call=after_tool_call,
            should_stop_after_turn=should_stop_after_turn,
            get_steering_messages=get_steering_messages,
            get_follow_up_messages=get_follow_up_messages,
            on_run_end=on_run_end,
            stream_options=stream_options,
            tool_execution=tool_execution,
            system_prompt=system_prompt,
            checkpointer=checkpointer,
            thread_id=thread_id,
            new_messages=new_messages,
            set_outcome=set_outcome,
        )
    except HitlDetached as exc:  # pragma: no cover — E2E tested
        # The Agent caller (Agent.detach) emitted AgentSuspendedEvent already.
        # Loop exits silently — assistant message and pending state remain
        # intact for the next respond() call. Sibling tool_results that
        # completed before the detach were emitted by the executor; append
        # them so callers persisting the returned messages keep them.
        for msg in exc.partial_tool_results:
            context.messages.append(msg)
            new_messages.append(msg)
        if set_outcome is not None:
            set_outcome("suspended")
        return new_messages
    except HitlAborted as exc:  # pragma: no cover — E2E tested
        # The Agent caller (Agent.abort_pending) emitted AgentAbortedEvent
        # already. Loop exits silently — synthetic deny + terminal aborted
        # assistant message already appended by abort_pending.
        for msg in exc.partial_tool_results:
            context.messages.append(msg)
            new_messages.append(msg)
        if set_outcome is not None:
            set_outcome("abandoned")
        return new_messages


async def _run_agent_loop_resume_body(  # pragma: no cover — E2E tested
    *,
    context,
    model,
    convert_to_llm,
    emit,
    transform_context,
    transform_system_prompt,
    after_model_response,
    before_tool_call,
    resolve_tool_call,
    after_tool_call,
    should_stop_after_turn,
    get_steering_messages,
    get_follow_up_messages,
    on_run_end: Callable | None,
    stream_options,
    tool_execution,
    system_prompt,
    checkpointer,
    thread_id,
    new_messages: list,
    set_outcome: Callable[[str], None] | None = None,
) -> list[Message]:
    from cubeloop.hitl.exceptions import HitlInconsistentState

    # Sanity check
    if not context.messages:
        raise HitlInconsistentState("resume called with empty message history")

    # Locate the suspended AssistantMessage by scanning backwards for the most
    # recent assistant turn that still has unresolved tool_calls. We cannot
    # require the tail to be that assistant message — a previous crashed
    # resume may have left some ToolResultMessage(s) at the tail before
    # crashing, and crash-recovery idempotency depends on us picking the
    # right assistant turn and skipping already-resolved tool_calls.
    asst_pos = -1
    last: AssistantMessage | None = None
    for i in range(len(context.messages) - 1, -1, -1):
        msg = context.messages[i]
        if not isinstance(msg, AssistantMessage):
            continue
        tcs = [c for c in msg.content if isinstance(c, ToolCall)]
        if not tcs:
            continue
        already = {
            m.tool_call_id
            for m in context.messages[i + 1 :]
            if isinstance(m, ToolResultMessage)
        }
        if any(tc.id not in already for tc in tcs):
            asst_pos = i
            last = msg
            break

    if last is None or asst_pos < 0:
        raise HitlInconsistentState(
            "resume could not locate an AssistantMessage with unresolved tool_calls"
        )

    unresolved = [c for c in last.content if isinstance(c, ToolCall)]
    # Idempotency: skip tool_calls already resolved by a prior crashed resume.
    already_resolved = {
        m.tool_call_id
        for m in context.messages[asst_pos + 1 :]
        if isinstance(m, ToolResultMessage)
    }
    remaining = [tc for tc in unresolved if tc.id not in already_resolved]

    await emit_event(emit, AgentStartEvent())
    await emit_event(emit, TurnStartEvent())

    current_context = context
    batch_tool_results: list[ToolResultMessage] = []
    terminated_by_tool = False

    if remaining:
        # Build a fresh assistant message containing only the remaining
        # tool_calls so execute_tool_calls processes those exact entries.
        # We do NOT mutate `last` itself — model_copy gives an independent
        # AssistantMessage that execute_tool_calls can read from.
        remaining_ids = {tc.id for tc in remaining}
        partial_msg = last.model_copy(
            update={
                "content": [
                    c
                    for c in last.content
                    if not isinstance(c, ToolCall) or c.id in remaining_ids
                ],
            }
        )
        batch = await execute_tool_calls(
            current_context,
            partial_msg,
            tool_execution=tool_execution,
            before_tool_call=before_tool_call,
            resolve_tool_call=resolve_tool_call,
            after_tool_call=after_tool_call,
            signal=(stream_options or StreamOptions()).signal,
            emit=emit,
        )
        batch_tool_results = list(batch.messages)
        terminated_by_tool = batch.terminate

        for r in batch_tool_results:
            current_context.messages.append(r)
            new_messages.append(r)

    # Clear pending_request from checkpointer NOW — after tool_results are
    # appended (and have been checkpointed by the Agent layer's MessageEndEvent
    # handler). The pending_request column / row is the cross-process witness;
    # holding it until here preserves crash-recovery idempotency (see spec §5.2).
    if checkpointer is not None and thread_id is not None:
        save_pending = getattr(checkpointer, "save_pending_request", None)
        if save_pending is not None:
            await save_pending(thread_id, None)
        clear_answers = getattr(checkpointer, "clear_hitl_answers", None)
        if clear_answers is not None:
            await clear_answers(thread_id, run_id=last.run_id)

    # Emit TurnEndEvent with the ACTUAL tool_results so listeners get the
    # right payload (codex BLOCKING: previous draft emitted []).
    await emit_event(emit, TurnEndEvent(message=last, tool_results=batch_tool_results))

    # Drain steering AFTER tool_results, BEFORE termination check — preserves
    # the Anthropic adjacency invariant (no user/system message wedged between
    # tool_use and tool_result) AND matches existing _run_loop ordering, which
    # drains steering before its terminal AgentEndEvent.
    if get_steering_messages:
        steering = await get_steering_messages() or []
        for msg in steering:
            await emit_event(emit, MessageStartEvent(message=msg))
            await emit_event(emit, MessageEndEvent(message=msg))
            current_context.messages.append(msg)
            new_messages.append(msg)

    resumed_follow_ups: list[Message] = []
    if get_follow_up_messages:
        resumed_follow_ups = await get_follow_up_messages() or []
        for msg in resumed_follow_ups:
            await emit_event(emit, MessageStartEvent(message=msg))
            await emit_event(emit, MessageEndEvent(message=msg))
            current_context.messages.append(msg)
            new_messages.append(msg)

    # Honor should_stop_after_turn (codex BLOCKING: previous draft skipped this).
    should_stop = False
    if should_stop_after_turn:
        stop_ctx = ShouldStopAfterTurnContext(
            message=last,
            tool_results=batch_tool_results,
            context=current_context,
            new_messages=new_messages,
        )
        should_stop = await should_stop_after_turn(stop_ctx)
        if should_stop and not resumed_follow_ups:
            if current_context.set_input_admission is not None:
                current_context.set_input_admission(False)
            await emit_event(emit, AgentEndEvent(messages=new_messages))
            if set_outcome is not None:
                set_outcome("complete")
            return new_messages

    # Terminate-by-tool semantics (codex BLOCKING: previous draft ignored).
    if terminated_by_tool and not resumed_follow_ups:
        if current_context.set_input_admission is not None:
            current_context.set_input_admission(False)
        await emit_event(emit, AgentEndEvent(messages=new_messages))
        if set_outcome is not None:
            set_outcome("complete")
        return new_messages

    # Fall through to the normal loop for the next model turn.
    await _run_loop(
        current_context=current_context,
        new_messages=new_messages,
        model=model,
        convert_to_llm=convert_to_llm,
        transform_context=transform_context,
        transform_system_prompt=transform_system_prompt,
        after_model_response=after_model_response,
        before_tool_call=before_tool_call,
        resolve_tool_call=resolve_tool_call,
        after_tool_call=after_tool_call,
        should_stop_after_turn=should_stop_after_turn,
        get_steering_messages=get_steering_messages,
        get_follow_up_messages=get_follow_up_messages,
        on_run_end=on_run_end,
        stream_options=stream_options,
        tool_execution=tool_execution,
        emit=emit,
        set_outcome=set_outcome,
    )
    return new_messages


async def _run_loop(
    *,
    current_context: AgentContext,
    new_messages: list[Message],
    model: BoundModel | FallbackBoundModel,
    convert_to_llm: Callable,
    transform_context: Callable | None,
    transform_system_prompt: Callable | None,
    after_model_response: Callable | None,
    before_tool_call: Callable | None,
    resolve_tool_call: Callable | None,
    after_tool_call: Callable | None,
    should_stop_after_turn: Callable | None,
    get_steering_messages: Callable | None,
    get_follow_up_messages: Callable | None,
    on_run_end: Callable | None,
    stream_options: StreamOptions | None,
    tool_execution: str,
    emit: Callable,
    set_outcome: Callable[[str], None] | None = None,
) -> None:
    try:
        await _run_loop_inner(
            current_context=current_context,
            new_messages=new_messages,
            model=model,
            convert_to_llm=convert_to_llm,
            transform_context=transform_context,
            transform_system_prompt=transform_system_prompt,
            after_model_response=after_model_response,
            before_tool_call=before_tool_call,
            resolve_tool_call=resolve_tool_call,
            after_tool_call=after_tool_call,
            should_stop_after_turn=should_stop_after_turn,
            get_steering_messages=get_steering_messages,
            get_follow_up_messages=get_follow_up_messages,
            on_run_end=on_run_end,
            stream_options=stream_options,
            tool_execution=tool_execution,
            emit=emit,
            set_outcome=set_outcome,
        )
    except HitlDetached as exc:
        # AgentSuspendedEvent already emitted by Agent.detach() — exit silently
        # so AgentEndEvent doesn't double-signal termination. Assistant message
        # and pending state remain intact for the next respond() call.
        # Sibling tool_results that completed before the detach were emitted
        # (and checkpointed) by the executor; append them so callers reading
        # current_context/new_messages keep them too.
        for msg in exc.partial_tool_results:
            current_context.messages.append(msg)
            new_messages.append(msg)
        if set_outcome is not None:
            set_outcome("suspended")
        return
    except HitlAborted as exc:
        # AgentAbortedEvent already emitted by Agent.abort_pending() — exit
        # silently. Synthetic deny + terminal aborted assistant message are
        # already appended; conversation is closed.
        for msg in exc.partial_tool_results:
            current_context.messages.append(msg)
            new_messages.append(msg)
        if set_outcome is not None:
            set_outcome("abandoned")
        return


async def _run_loop_inner(
    *,
    current_context: AgentContext,
    new_messages: list[Message],
    model: BoundModel | FallbackBoundModel,
    convert_to_llm: Callable,
    transform_context: Callable | None,
    transform_system_prompt: Callable | None,
    after_model_response: Callable | None,
    before_tool_call: Callable | None,
    resolve_tool_call: Callable | None,
    after_tool_call: Callable | None,
    should_stop_after_turn: Callable | None,
    get_steering_messages: Callable | None,
    get_follow_up_messages: Callable | None,
    on_run_end: Callable | None,
    stream_options: StreamOptions | None,
    tool_execution: str,
    emit: Callable,
    set_outcome: Callable[[str], None] | None = None,
) -> None:
    opts = stream_options or StreamOptions()
    first_turn = True

    # Poll for steering messages at start (user may have typed while waiting)
    if get_steering_messages:
        pending = await get_steering_messages() or []
        if pending:
            for msg in pending:
                await emit_event(emit, MessageStartEvent(message=msg))
                await emit_event(emit, MessageEndEvent(message=msg))
                current_context.messages.append(msg)
                new_messages.append(msg)

    on_run_end_cycles = 0

    while True:
        has_more_tool_calls = True

        while has_more_tool_calls:
            if not first_turn:
                await emit_event(emit, TurnStartEvent())
            else:
                first_turn = False

            message = await _stream_assistant_response(
                current_context,
                model,
                convert_to_llm,
                transform_context,
                transform_system_prompt,
                opts,
                emit,
            )
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                # Emit message_end before turn/agent end events.
                await emit_event(emit, MessageEndEvent(message=message))
                await emit_event(emit, TurnEndEvent(message=message, tool_results=[]))
                if current_context.set_input_admission is not None:
                    current_context.set_input_admission(False)
                await emit_event(emit, AgentEndEvent(messages=new_messages))
                if set_outcome is not None:
                    set_outcome("abandoned")
                return

            # Apply after_model_response hook if configured.
            # The hook returns a TurnAction that can mutate the response,
            # inject messages, or change control flow.
            # message_end is deferred until here so that a mutated response
            # is what gets persisted via Agent._process_event.
            skip_tool_execution = False
            # Messages injected by after_model_response that must be appended
            # AFTER this turn's tool_results, not immediately (see below).
            deferred_inject_messages: list[Message] = []
            if after_model_response is not None:
                turn_action = await after_model_response(
                    message,
                    current_context,
                    signal=opts.signal,
                )
                if turn_action is not None:
                    if turn_action.response is not None:
                        # Replace both the local variable and the copy in
                        # context.messages that _stream_assistant_response appended.
                        message = turn_action.response
                        new_messages[-1] = message
                        if current_context.messages and isinstance(
                            current_context.messages[-1], AssistantMessage
                        ):
                            current_context.messages[-1] = message
                    # Emit message_end now — message reflects any hook mutation.
                    await emit_event(emit, MessageEndEvent(message=message))
                    if turn_action.inject_messages:
                        # When this turn still has tool_calls to execute (the
                        # "natural" decision falls through to tool handling
                        # below), the tool_results MUST be appended before any
                        # injected messages. Otherwise an injected (e.g. user)
                        # message lands between an assistant tool_use and its
                        # tool_result, which strict Anthropic-style endpoints
                        # reject ("tool_use ... without tool_result blocks
                        # immediately after"). Defer such injects until after
                        # the tool_results are appended.
                        _will_execute_tools = turn_action.decision == "natural" and any(
                            isinstance(c, ToolCall) for c in message.content
                        )
                        if _will_execute_tools:
                            deferred_inject_messages = list(turn_action.inject_messages)
                        else:
                            for inj in turn_action.inject_messages:
                                await emit_event(emit, MessageStartEvent(message=inj))
                                await emit_event(emit, MessageEndEvent(message=inj))
                                current_context.messages.append(inj)
                                new_messages.append(inj)
                    if turn_action.decision == "stop":
                        await emit_event(
                            emit, TurnEndEvent(message=message, tool_results=[])
                        )
                        # Break inner while so outer while runs on_run_end before AgentEndEvent.
                        has_more_tool_calls = False
                        break
                    if turn_action.decision == "loop_to_model":
                        # Skip tool execution; re-invoke the model on next iteration.
                        # inject_messages are already appended to context above.
                        await emit_event(
                            emit, TurnEndEvent(message=message, tool_results=[])
                        )
                        skip_tool_execution = True
                    # decision == "natural" falls through to normal tool handling
                else:
                    # Hook returned None — emit message_end with unmodified message.
                    await emit_event(emit, MessageEndEvent(message=message))
            else:
                # No hook configured — emit message_end directly.
                await emit_event(emit, MessageEndEvent(message=message))

            if skip_tool_execution:
                has_more_tool_calls = True
                continue

            tool_calls = [c for c in message.content if isinstance(c, ToolCall)]
            tool_results: list[ToolResultMessage] = []
            has_more_tool_calls = False

            if tool_calls:
                batch = await execute_tool_calls(
                    current_context,
                    message,
                    tool_execution=tool_execution,
                    before_tool_call=before_tool_call,
                    resolve_tool_call=resolve_tool_call,
                    after_tool_call=after_tool_call,
                    signal=opts.signal,
                    emit=emit,
                )
                tool_results = batch.messages
                has_more_tool_calls = not batch.terminate

                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            # Append messages deferred from after_model_response so they land
            # AFTER the tool_results, preserving tool_use/tool_result adjacency.
            for inj in deferred_inject_messages:
                await emit_event(emit, MessageStartEvent(message=inj))
                await emit_event(emit, MessageEndEvent(message=inj))
                current_context.messages.append(inj)
                new_messages.append(inj)

            await emit_event(
                emit, TurnEndEvent(message=message, tool_results=tool_results)
            )

            if should_stop_after_turn:
                stop_ctx = ShouldStopAfterTurnContext(
                    message=message,
                    tool_results=tool_results,
                    context=current_context,
                    new_messages=new_messages,
                )
                if await should_stop_after_turn(stop_ctx):
                    # Break inner while so outer while runs on_run_end before AgentEndEvent.
                    has_more_tool_calls = False
                    break

            # Check for steering messages after tool execution
            if get_steering_messages and has_more_tool_calls:
                steering = await get_steering_messages() or []
                if steering:
                    for msg in steering:
                        await emit_event(emit, MessageStartEvent(message=msg))
                        await emit_event(emit, MessageEndEvent(message=msg))
                        current_context.messages.append(msg)
                        new_messages.append(msg)

        # The in-loop drain only fires when more tool calls remain, so a steer
        # that arrives while the model is finishing a tool-less turn would be
        # dropped. Drain it here and re-invoke the model so "steer anytime"
        # works during a final text turn too.
        if get_steering_messages:
            steering = await get_steering_messages() or []
            if steering:
                for msg in steering:
                    await emit_event(emit, MessageStartEvent(message=msg))
                    await emit_event(emit, MessageEndEvent(message=msg))
                    current_context.messages.append(msg)
                    new_messages.append(msg)
                first_turn = False
                continue

        # After inner loop completes, check for follow-up messages
        if get_follow_up_messages:
            follow_ups = await get_follow_up_messages() or []
            if follow_ups:
                for msg in follow_ups:
                    await emit_event(emit, MessageStartEvent(message=msg))
                    await emit_event(emit, MessageEndEvent(message=msg))
                    current_context.messages.append(msg)
                    new_messages.append(msg)
                first_turn = False
                continue

        if on_run_end:
            if current_context.set_input_admission is not None:
                current_context.set_input_admission(False)
            inject = await on_run_end(current_context, signal=opts.signal)
            if inject:
                on_run_end_cycles += 1
                if on_run_end_cycles > _MAX_ON_RUN_END_CYCLES:
                    break
                if current_context.set_input_admission is not None:
                    current_context.set_input_admission(True)
                for msg in inject:
                    await emit_event(emit, MessageStartEvent(message=msg))
                    await emit_event(emit, MessageEndEvent(message=msg))
                    current_context.messages.append(msg)
                    new_messages.append(msg)
                first_turn = False
                continue

        else:
            if current_context.set_input_admission is not None:
                current_context.set_input_admission(False)

        break

    if current_context.set_input_admission is not None:
        current_context.set_input_admission(False)
    await emit_event(emit, AgentEndEvent(messages=new_messages))
    if set_outcome is not None:
        set_outcome("complete")


async def _stream_assistant_response(
    context: AgentContext,
    model: BoundModel | FallbackBoundModel,
    convert_to_llm: Callable,
    transform_context: Callable | None,
    transform_system_prompt: Callable | None,
    options: StreamOptions,
    emit: Callable,
) -> AssistantMessage:
    messages = context.messages
    if transform_context:
        messages = await transform_context(messages, ctx=context, signal=options.signal)

    llm_messages = convert_to_llm(messages, ctx=context)
    if asyncio.iscoroutine(llm_messages):
        llm_messages = await llm_messages

    tools_defs = None
    if context.tools:
        visible = [t for t in context.tools if t.expose_to_model]
        if visible:
            tools_defs = [t.to_definition() for t in visible]

    sp = context.system_prompt
    if transform_system_prompt:
        sp = await transform_system_prompt(sp, ctx=context, signal=options.signal)

    turn_id = uuid.uuid4().hex
    previous_on_model_attempt = options.on_model_attempt

    async def _capture_model_attempt(model_spec) -> None:
        from cubeloop.providers.fallback import get_attempt_index
        from cubeloop.session.turn_execution_context import TurnExecutionContext

        existing = context.turn_execution_context
        fallback_index = get_attempt_index()
        model_attempt_id = (
            f"fallback:{fallback_index}" if fallback_index is not None else None
        )
        if (
            existing is None
            or existing.turn_id != turn_id
            or not existing.matches_model(model_spec, model_attempt_id)
        ):
            captured = TurnExecutionContext.capture(
                turn_id=turn_id,
                run_id=context.run_id or "",
                attempt_id=context.attempt_id or "",
                model=model_spec,
                reasoning=options.reasoning,
                system_prompt=sp,
                messages=llm_messages,
                tools=context.tools,
                policy_revision=(
                    str(context.extra["policy_revision"])
                    if "policy_revision" in context.extra
                    else None
                ),
                model_attempt_id=model_attempt_id,
            )
            context.turn_id = turn_id
            context.turn_execution_context = captured
            if context.on_turn_context is not None:
                callback_result = context.on_turn_context(captured)
                if asyncio.iscoroutine(callback_result):
                    await callback_result
        if previous_on_model_attempt is not None:
            callback_result = previous_on_model_attempt(model_spec)
            if asyncio.iscoroutine(callback_result):
                await callback_result

    options = options.model_copy(update={"on_model_attempt": _capture_model_attempt})

    stream = await model.stream(
        llm_messages,
        system_prompt=sp,
        tools=tools_defs,
        options=options,
    )

    partial_message: AssistantMessage | None = None
    added_partial = False

    async for event in stream:
        if event.type == "start":
            partial_message = event.partial
            if partial_message:
                context.messages.append(partial_message)
                added_partial = True
                await emit_event(
                    emit,
                    MessageStartEvent(message=partial_message.model_copy(deep=True)),
                )

        elif event.type in (
            "text_start",
            "text_delta",
            "text_end",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        ):
            if partial_message and event.partial:
                partial_message = event.partial
                context.messages[-1] = partial_message
                await emit_event(
                    emit,
                    MessageUpdateEvent(
                        message=partial_message.model_copy(deep=True),
                        stream_event=event,
                    ),
                )

        elif event.type in ("done", "error"):
            final_message = await stream.result()
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
            if not added_partial:
                await emit_event(emit, MessageStartEvent(message=final_message))
            # message_end is intentionally NOT emitted here; the caller (_run_loop)
            # emits it after running after_model_response so the persisted message
            # reflects any hook mutation.
            return final_message

    # Fallback: stream ended without done/error event
    final_message = await stream.result()
    if added_partial:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
        await emit_event(emit, MessageStartEvent(message=final_message))
    # message_end deferred to caller — see comment above.
    return final_message
