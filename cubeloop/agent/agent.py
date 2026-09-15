from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

from cubeloop.agent._outcome import RunOutcome
from cubeloop.agent._tool_cycle import ToolCycleViolation, check_tool_cycle
from cubeloop.agent.loop import (
    run_agent_loop,
    run_agent_loop_continue,
    run_agent_loop_resume,
)
from cubeloop.checkpointer.base import Checkpointer
from cubeloop.checkpointer.exceptions import CompletionMarkerFailedError
from cubeloop.hitl import HitlError, HitlRequest
from cubeloop.hitl.channel import HitlChannel
from cubeloop.hitl.exceptions import HitlDetached
from cubeloop.middleware.base import Middleware, compose_middleware
from cubeloop.agent.types import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentTool,
    AgentToolResult,
    ForkOnceResult,
    MessageEndEvent,
    MessageStartEvent,
    TurnEndEvent,
)
from cubeloop.providers.base import (
    AssistantMessage,
    BoundModel,
    Message,
    Model,
    OnPayloadCallback,
    OnResponseCallback,
    ReasoningControl,
    StreamOptions,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from cubeloop.types import JsonObject, StructuredValue

TMessage = TypeVar("TMessage")


def _consume_control_exception(future: asyncio.Future[StructuredValue]) -> None:
    """Mark a detach control exception retrieved without changing awaiters."""
    if not future.cancelled():
        future.exception()


if TYPE_CHECKING:
    from cubeloop.deferred.types import DeferredStrategy, DeferredToolGroup
    from cubeloop.providers.fallback import FallbackBoundModel


def _deny_in_fork(tool: AgentTool) -> AgentTool:
    """Return a copy of *tool* whose execute always returns an error.

    Used by fork_once to keep middleware tools in the schema (prompt-cache
    parity) while preventing execution that would mutate parent state.
    """
    from dataclasses import replace

    async def _denied(
        tool_call_id: str,
        args: object,
        *,
        signal: asyncio.Event | None = None,
        on_update: Callable | None = None,
    ) -> AgentToolResult:
        return AgentToolResult(
            content=[
                TextContent(text=f"{tool.name} is not available in a forked agent.")
            ],
            is_error=True,
        )

    return replace(tool, execute=_denied)


def _default_convert_to_llm(
    messages: list[Message], *, ctx: AgentContext
) -> list[Message]:
    del ctx
    return list(messages)


class _MessageQueue:
    def __init__(self, mode: str = "one-at-a-time") -> None:
        self.mode = mode
        self._messages: list[Message] = []

    def enqueue(self, message: Message) -> None:
        self._messages.append(message)

    def has_items(self) -> bool:
        return len(self._messages) > 0

    def drain(self) -> list[Message]:
        if self.mode == "all":
            drained = self._messages[:]
            self._messages = []
            return drained
        if not self._messages:
            return []
        first = self._messages[0]
        self._messages = self._messages[1:]
        return [first]

    def remove(self, steer_id: str) -> bool:
        kept = [
            m
            for m in self._messages
            if getattr(m, "metadata", {}).get("steer_id") != steer_id
        ]
        removed = len(kept) != len(self._messages)
        self._messages = kept
        return removed

    def clear(self) -> None:
        self._messages = []


@dataclass
class AgentState:
    system_prompt: str = ""
    model: Model = field(
        default_factory=lambda: Model(id="unknown", provider_id="unknown")
    )
    reasoning: ReasoningControl = field(default_factory=ReasoningControl)
    is_streaming: bool = False
    streaming_message: Message | None = None
    error_message: str | None = None
    active_run_id: str | None = None
    last_outcome: RunOutcome | None = None
    extra: JsonObject = field(default_factory=dict)
    _tools: list[AgentTool] = field(default_factory=list)
    _messages: list[Message] = field(default_factory=list)
    _pending_tool_calls: set[str] = field(default_factory=set)

    @property
    def tools(self) -> list[AgentTool]:
        return list(self._tools)

    @tools.setter
    def tools(self, value: list[AgentTool]) -> None:
        self._tools = list(value)

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    @messages.setter
    def messages(self, value: list[Message]) -> None:
        self._messages = list(value)

    @property
    def pending_tool_calls(self) -> set[str]:
        return set(self._pending_tool_calls)

    @pending_tool_calls.setter
    def pending_tool_calls(self, value: set[str]) -> None:
        self._pending_tool_calls = set(value)


class Agent(Generic[TMessage]):
    def __init__(
        self,
        *,
        model: BoundModel | FallbackBoundModel,
        system_prompt: str = "",
        tools: list[AgentTool] | None = None,
        reasoning: ReasoningControl | None = None,
        convert_to_llm: Callable[..., list[Message]] | None = None,
        transform_context: Callable | None = None,
        transform_system_prompt: Callable | None = None,
        after_model_response: Callable | None = None,
        before_tool_call: Callable | None = None,
        resolve_tool_call: Callable | None = None,
        after_tool_call: Callable | None = None,
        should_stop_after_turn: Callable | None = None,
        on_run_end: Callable | None = None,
        on_payload: OnPayloadCallback | None = None,
        on_response: OnResponseCallback | None = None,
        steering_mode: str = "one-at-a-time",
        follow_up_mode: str = "one-at-a-time",
        tool_execution: str = "parallel",
        checkpointer: Checkpointer | None = None,
        thread_id: str | None = None,
        middleware: list[Middleware] | None = None,
        deferred_tool_groups: list[DeferredToolGroup] | None = None,
        deferred_tool_strategy: DeferredStrategy = "dispatch",
        channel: HitlChannel | None = None,
        messages: Sequence[Message] | None = None,
    ) -> None:
        self._model = model
        self._state = AgentState(
            system_prompt=system_prompt,
            model=model.spec,
            reasoning=reasoning or ReasoningControl(),
        )
        if messages is not None:
            if thread_id is not None and checkpointer is not None:
                raise ValueError(
                    "Agent(messages=...) cannot be combined with "
                    "thread_id + checkpointer (pre-seed conflicts with lazy "
                    "load). Construct an ephemeral Agent without those for "
                    "fork_once-style usage."
                )
            seeded = [m.model_copy(deep=True) for m in messages]
            self._state.messages = list(seeded)
        if tools:
            self._state.tools = tools
        if deferred_tool_groups:
            from cubeloop.deferred.middleware import DeferredToolsMiddleware

            deferred_mw = DeferredToolsMiddleware(
                groups=deferred_tool_groups,
                extra_ref=lambda: self._extra,
                strategy=deferred_tool_strategy,
                on_tools_expanded=lambda new: self._state._tools.extend(
                    t for t in new if t.name not in {e.name for e in self._state._tools}
                ),
            )
            middleware = [*(middleware or []), deferred_mw]
        middleware = middleware or []
        # Retain the list so observers (e.g. cubeloop.tracing.Recorder) can
        # walk it after construction — they need to reach attributes like
        # ``Middleware.providers()`` that aren't captured by the composed
        # hook callables below.
        self._middleware: list[Middleware] = list(middleware)
        middleware_tools: list[AgentTool] = []
        for mw in middleware:
            middleware_tools.extend(getattr(mw, "tools", []) or [])
        if middleware_tools:
            self._state.tools = [*self._state.tools, *middleware_tools]
        # Compose middleware hooks, then let explicit callables override.
        _mw_hooks = compose_middleware(middleware)
        self.convert_to_llm = (
            convert_to_llm or _mw_hooks.get("convert_to_llm") or _default_convert_to_llm
        )
        self.transform_context = transform_context or _mw_hooks.get("transform_context")
        self.transform_system_prompt = transform_system_prompt or _mw_hooks.get(
            "transform_system_prompt"
        )
        self.after_model_response = after_model_response or _mw_hooks.get(
            "after_model_response"
        )
        self.before_tool_call = before_tool_call or _mw_hooks.get("before_tool_call")
        # Unlike the other hooks, an explicit resolve_tool_call COMPOSES with
        # the middleware chain instead of replacing it (first-non-None, with
        # the explicit callable as chain head). Replacing would silently
        # disable deferred dispatch for any Agent(deferred_tool_groups=...,
        # resolve_tool_call=...) — the auto-created middleware is internal
        # wiring the caller never sees and cannot re-compose by hand.
        _mw_resolver = _mw_hooks.get("resolve_tool_call")
        self.resolve_tool_call: Callable | None
        if resolve_tool_call is not None and _mw_resolver is not None:
            _explicit_resolver = resolve_tool_call

            async def _composed_resolver(tool_call, *, context, signal=None):
                result = await _explicit_resolver(
                    tool_call, context=context, signal=signal
                )
                if result is not None:
                    return result
                return await _mw_resolver(tool_call, context=context, signal=signal)

            self.resolve_tool_call = _composed_resolver
        else:
            self.resolve_tool_call = resolve_tool_call or _mw_resolver
        self.after_tool_call = after_tool_call or _mw_hooks.get("after_tool_call")
        self.should_stop_after_turn = should_stop_after_turn or _mw_hooks.get(
            "should_stop_after_turn"
        )
        self.on_run_end = on_run_end or _mw_hooks.get("on_run_end")
        self.on_payload = on_payload
        self.on_response = on_response
        self.tool_execution = tool_execution
        self.checkpointer = checkpointer
        self.thread_id = thread_id
        self._run_aware = (
            self.checkpointer is not None
            and hasattr(self.checkpointer, "claim_run")
            and hasattr(self.checkpointer, "mark_run_complete")
        )
        self._channel = channel
        self._pending_suspension_event: HitlRequest | None = None
        # _bind_emit is a _BaseChannel internal, not part of the HitlChannel
        # protocol. Third-party channels that only implement the public
        # protocol won't have it — skip the wiring instead of crashing.
        if channel is not None and hasattr(channel, "_bind_emit"):
            channel._bind_emit(lambda e: self._process_event(e))
        self._run_lock = asyncio.Lock()

        # Share the same dict so agent.state.extra reflects mutations made via
        # AgentContext.extra during middleware hooks.
        self._extra: JsonObject = self._state.extra

        self._steering_queue = _MessageQueue(steering_mode)
        self._follow_up_queue = _MessageQueue(follow_up_mode)
        self._listeners: list[Callable] = []
        self._active_signal: asyncio.Event | None = None
        self._active_done: asyncio.Event | None = None
        self._fallback_active_index: int = 0
        self._checkpoint_loaded = messages is not None
        from cubeloop.session import ExecutionSession

        self.session = ExecutionSession(self)

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def channel(self) -> HitlChannel | None:
        return self._channel

    @property
    def in_flight_hitl_request(self) -> HitlRequest | None:
        if self._channel is None:
            raise HitlError("agent has no channel bound; pass channel= to Agent()")
        return self._channel.pending

    def subscribe(self, listener: Callable) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: (
            self._listeners.remove(listener) if listener in self._listeners else None
        )

    def steer(self, message: Message) -> None:
        self._steering_queue.enqueue(message)

    def cancel_steer(self, steer_id: str) -> bool:
        """Remove a not-yet-drained steering message by its steer_id.

        Returns True if a queued message was removed; False if it was already
        drained or never queued (best-effort cancel).
        """
        return self._steering_queue.remove(steer_id)

    def follow_up(self, message: Message) -> None:
        self._follow_up_queue.enqueue(message)

    def abort(self) -> None:
        if self._active_signal:
            self._active_signal.set()

    async def wait_for_idle(self) -> None:
        await self.session.wait_for_idle()

    def reset(self) -> None:
        self._state._messages = []
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state._pending_tool_calls = set()
        self._state.error_message = None
        self._steering_queue.clear()
        self._follow_up_queue.clear()
        self.session._reset_inputs()
        self._checkpoint_loaded = False

    def _outcome_sink(self) -> Callable[[str], None]:
        def _sink(value: str) -> None:
            # Cast through Any: loop callsites pass plain str literals from the
            # RunOutcome alphabet ("complete" / "suspended" / "abandoned").
            self._state.last_outcome = value  # type: ignore[assignment]

        return _sink

    async def _dispatch_outcome(self, outcome: RunOutcome | None, run_id: str) -> None:
        if outcome == "complete":
            run_messages = [m for m in self._state.messages if m.run_id == run_id]
            try:
                check_tool_cycle(run_messages)
            except ToolCycleViolation:
                outcome = "incomplete"
        if outcome != "complete":
            return
        if not (self._run_aware and self.thread_id):
            return
        assert self.checkpointer is not None  # narrowed by _run_aware
        try:
            await self.checkpointer.mark_run_complete(self.thread_id, run_id)
        except Exception as exc:
            raise CompletionMarkerFailedError(
                thread_id=self.thread_id,
                run_id=run_id,
                cause=exc,
            ) from exc

    async def _emit_suspended_event(self, outcome: RunOutcome) -> None:
        """Publish suspension only after the run transition is committed."""
        if outcome != "suspended":
            self._pending_suspension_event = None
            return
        pending = self._pending_suspension_event
        if pending is None:
            return
        self._pending_suspension_event = None
        from cubeloop.agent.types import AgentSuspendedEvent

        await self._process_event(AgentSuspendedEvent(pending_request=pending))

    def _validate_hitl_bindings(self, run_id: str | None, *, caller: str) -> None:
        """Reject HITL-bound tools/middleware that disagree with `run_id`.

        Called from both `prompt(run_id=...)` and `respond()` (where the
        run_id comes from the recovered pending row). This guarantees that
        any checkpointed HITL channel/middleware on the Agent is bound to
        the same run_id as the run that's about to drive the loop —
        otherwise a second HITL pause would persist its pending row under
        the wrong run_id and the next resume would stamp messages or try
        to complete an unclaimed run.

        Two callers, two None-semantics:
        - `prompt(run_id=None)` with bindings → "generate-mode rejected":
          the caller explicitly asked for an unbound run; the bindings
          already carry one, so we must reject.
        - `respond()` with `recovered_run_id=None` → legacy pending row
          (pre-v4 save_pending_request, or forged). Skip the binding
          check; the legacy fallback in respond() will skip dispatch
          anyway, and the host's channel binding may correctly identify
          the original run.
        """
        bound: set[str] = set()
        for elem in (*self._state.tools, *self._middleware):
            binding = getattr(elem, "hitl", None)
            if binding is None or not binding.checkpointed:
                continue
            if binding.run_id is None:
                raise ValueError(
                    f"Checkpointed HITL element {elem!r} has no run_id bound; "
                    "construct CheckpointedChannel(run_id=...) before passing "
                    "it to ask_user_tool/HITL middleware"
                )
            bound.add(binding.run_id)
        if not bound:
            return
        if run_id is None:
            if caller == "respond":
                # Legacy pending — no run_id to validate against.
                return
            raise ValueError(
                f"Agent has checkpointed HITL elements bound to "
                f"run_ids {sorted(bound)!r}; {caller}(run_id=...) "
                "must be explicitly supplied (generate-mode rejected)"
            )
        if any(b != run_id for b in bound):
            raise ValueError(
                f"{caller}() run_id={run_id!r} does not match "
                f"HITL-bound run_ids {sorted(bound)!r}"
            )

    def _validate_input_run_ids(
        self,
        message: str | Message | list[Message],
        effective_run_id: str,
    ) -> None:
        if isinstance(message, str):
            return
        if isinstance(message, list):
            candidates: list[Message] = message
        else:
            candidates = [message]
        for m in candidates:
            if getattr(m, "run_id", None) is not None and m.run_id != effective_run_id:
                raise ValueError(
                    f"message.run_id={m.run_id!r} does not match "
                    f"prompt(run_id={effective_run_id!r})"
                )

    async def prompt(
        self,
        message: str | Message | list[Message],
        *,
        run_id: str | None = None,
    ) -> str:
        from cubeloop.session import PromptExecutionRequest

        self.session._assert_idle()
        self._validate_hitl_bindings(run_id, caller="prompt")
        effective_run_id = run_id or uuid.uuid4().hex
        result = await self.session.execute(
            PromptExecutionRequest(
                run_id=effective_run_id,
                attempt_id=uuid.uuid4().hex,
                message=message,
            )
        )
        if result.error is not None and result.error.cause is not None:
            raise result.error.cause
        return effective_run_id

    async def _execute_prompt(
        self,
        message: str | Message | list[Message],
        *,
        run_id: str | None = None,
    ) -> str:
        # Fail-fast guard: if the run-lock is already held OR a stream is in
        # progress, raise immediately instead of queueing on the lock. This
        # makes two concurrent cold prompt() calls fail-fast deterministically
        # (lock.locked() is the atomic source of truth — checking is_streaming
        # alone races against the to-be-set flag inside the lock body).
        if self._run_lock.locked() or self._state.is_streaming:
            raise RuntimeError(
                "Agent is already processing a prompt. "
                "Use steer() or follow_up() to queue messages."
            )
        self._validate_hitl_bindings(run_id, caller="prompt")
        effective_run_id = run_id or uuid.uuid4().hex
        # Reject mismatched caller-supplied Message.run_id BEFORE any state
        # mutation so the supplied run_id remains reusable (claim_run must
        # not be called for a rejected prompt).
        self._validate_input_run_ids(message, effective_run_id)
        try:
            async with self._run_lock:
                # Re-check under the lock in case streaming flipped during acquire.
                if self._state.is_streaming:  # pragma: no cover — defensive re-check
                    raise RuntimeError(
                        "Agent is already processing a prompt. "
                        "Use steer() or follow_up() to queue messages."
                    )
                # Claim the run UNDER the lock: this serializes concurrent cold
                # prompt() calls against the checkpointer. With the fail-fast
                # `.locked()` guard above, the second concurrent caller raises
                # immediately instead of awaiting claim_run + queueing on the
                # lock.
                if self._run_aware and self.thread_id is not None:
                    assert self.checkpointer is not None  # narrowed by _run_aware
                    await self.checkpointer.claim_run(self.thread_id, effective_run_id)
                self._state.active_run_id = effective_run_id
                self._state.last_outcome = None

                if isinstance(message, str):
                    messages: list[Message] = [
                        UserMessage(
                            content=[TextContent(text=message)],
                            timestamp=time.time(),
                            run_id=effective_run_id,
                        )
                    ]
                elif isinstance(message, list):
                    messages = message
                else:
                    messages = [message]

                # Restore history and extra from checkpointer if this is first prompt
                if self.checkpointer and self.thread_id and not self._checkpoint_loaded:
                    data = await self.checkpointer.load(self.thread_id)
                    if data:
                        self._state._messages = list(data.messages)
                        self._extra.clear()
                        self._extra.update(data.extra)
                    self._checkpoint_loaded = True

                await self._run_prompt(messages)
        except BaseException:
            # Spec §3.7: leave active_run_id SET on failure.
            self._pending_suspension_event = None
            raise
        else:
            outcome: RunOutcome = self._state.last_outcome or "abandoned"
            # If _dispatch_outcome raises (CompletionMarkerFailedError),
            # propagate UP through this else: — active_run_id stays SET because
            # the clear line below is unreachable on the exception path.
            await self._dispatch_outcome(outcome, effective_run_id)
            self._state.active_run_id = None
            await self._emit_suspended_event(outcome)
            return effective_run_id

    async def fork(
        self,
        src_thread_id: str,
        new_thread_id: str,
        *,
        after_run_id: str,
        metadata: JsonObject | None = None,
    ) -> None:
        if self.checkpointer is None:
            raise RuntimeError("fork requires a checkpointer")
        if not self._run_aware:
            from cubeloop.checkpointer.exceptions import CheckpointerError

            raise CheckpointerError(
                "backend does not support fork; missing claim_run / mark_run_complete"
            )
        await self.checkpointer.fork(
            src_thread_id,
            new_thread_id,
            after_run_id=after_run_id,
            metadata=metadata,
        )

    async def fork_once(
        self,
        src_thread_id: str,
        message: str | Message | list[Message],
        *,
        after_run_id: str,
    ) -> ForkOnceResult:
        if self.checkpointer is None:
            raise RuntimeError("fork_once requires a checkpointer")
        # Degraded-mode guard
        if not self._run_aware or not hasattr(self.checkpointer, "snapshot"):
            from cubeloop.checkpointer.exceptions import CheckpointerError

            raise CheckpointerError(
                "backend does not support fork_once; missing snapshot / "
                "claim_run / mark_run_complete"
            )
        # HITL pre-flight
        hitl_offenders = [
            elem
            for elem in (*self._state.tools, *self._middleware)
            if getattr(elem, "hitl", None) is not None
        ]
        if hitl_offenders:
            names = ", ".join(
                getattr(e, "name", type(e).__name__) for e in hitl_offenders
            )
            raise RuntimeError(
                f"fork_once() does not support HITL. Found HITL-bearing "
                f"tools/middleware: {names}. Construct a different Agent "
                "without these for ephemeral probes."
            )
        snapshot = await self.checkpointer.snapshot(
            src_thread_id, after_run_id=after_run_id
        )
        # Build transient agent. Forward the parent's resolved execution
        # options + middleware-composed hooks so the probe behaves like the
        # parent would (tool_execution, reasoning, transform/response hooks,
        # on_payload/on_response, queue modes). Passing the resolved hooks
        # as explicit args means we pass middleware=[] to avoid composing
        # middleware twice — the parent's composed result already lives on
        # self.transform_context etc.
        #
        # Keep middleware-owned tools in the schema (prompt-cache parity)
        # but deny their execution at runtime — their execute callbacks
        # close over the parent's state and would corrupt it.
        _mw_tool_ids = {
            id(t) for mw in self._middleware for t in getattr(mw, "tools", []) or []
        }
        fork_tools = [
            _deny_in_fork(t) if id(t) in _mw_tool_ids else t for t in self._state.tools
        ]
        child: Agent = Agent(
            model=self._model,
            system_prompt=self._state.system_prompt,
            tools=fork_tools,
            reasoning=self._state.reasoning,
            convert_to_llm=self.convert_to_llm,
            transform_context=self.transform_context,
            transform_system_prompt=self.transform_system_prompt,
            after_model_response=self.after_model_response,
            before_tool_call=self.before_tool_call,
            # Without the resolver, dispatch-mode forks could never invoke
            # loaded deferred tools (the dispatcher's execute is fork-denied
            # and hidden tools are not in the payload).
            resolve_tool_call=self.resolve_tool_call,
            after_tool_call=self.after_tool_call,
            should_stop_after_turn=self.should_stop_after_turn,
            on_run_end=self.on_run_end,
            on_payload=self.on_payload,
            on_response=self.on_response,
            tool_execution=self.tool_execution,
            middleware=[],
            messages=snapshot,
        )
        pre_len = len(child.state.messages)
        fresh_run_id = uuid.uuid4().hex
        # Tracing placeholder — replaced by real OTel span in Task 34
        with self._fork_once_span(
            src_thread_id=src_thread_id, after_run_id=after_run_id
        ):
            await child.prompt(message, run_id=fresh_run_id)
        new_messages = child.state.messages[pre_len:]
        # Extract final assistant text and stop_reason
        final_text = ""
        stop_reason = "stop"
        for m in reversed(new_messages):
            if isinstance(m, AssistantMessage):
                final_text = "".join(
                    c.text for c in m.content if isinstance(c, TextContent)
                )
                stop_reason = m.stop_reason
                break
        return ForkOnceResult(
            text=final_text, messages=new_messages, stop_reason=stop_reason
        )

    def _fork_once_span(self, *, src_thread_id: str, after_run_id: str):
        """Emit a `cubeloop.agent.fork_once` OTel span around the probe run.

        The opentelemetry dependency is part of the optional `tracing` extra;
        when it isn't installed we fall back to a no-op nullcontext so the
        agent still works.
        """
        try:
            from opentelemetry import trace
        except ImportError:
            from contextlib import nullcontext

            return nullcontext()
        tracer = trace.get_tracer("cubeloop.agent")
        return tracer.start_as_current_span(
            "cubeloop.agent.fork_once",
            attributes={
                "cubeloop.fork.src_thread_id": src_thread_id,
                "cubeloop.fork.after_run_id": after_run_id,
            },
        )

    async def resume(self, *, run_id: str | None = None) -> str:
        from cubeloop.session import ContinueExecutionRequest

        effective_run_id = run_id or uuid.uuid4().hex
        result = await self.session.execute(
            ContinueExecutionRequest(
                run_id=effective_run_id,
                attempt_id=uuid.uuid4().hex,
            )
        )
        if result.error is not None and result.error.cause is not None:
            raise result.error.cause
        return effective_run_id

    async def _execute_continue(self, *, run_id: str | None = None) -> str:
        # Same fail-fast pattern as prompt(): lock.locked() is the atomic gate.
        if self._run_lock.locked() or self._state.is_streaming:
            raise RuntimeError(
                "Agent is already processing. Wait for completion before continuing."
            )
        self._validate_hitl_bindings(run_id, caller="resume")
        effective_run_id = run_id or uuid.uuid4().hex
        try:
            async with self._run_lock:
                if self._state.is_streaming:  # pragma: no cover — defensive re-check
                    raise RuntimeError(
                        "Agent is already processing. "
                        "Wait for completion before continuing."
                    )

                if not self._state._messages:
                    raise RuntimeError("No messages to continue from")

                last = self._state._messages[-1]
                # Validate preconditions BEFORE claim_run so a precondition
                # failure does not leave a claimed run dangling on the
                # checkpointer.
                if isinstance(last, AssistantMessage) and not (
                    self._steering_queue.has_items()
                    or self._follow_up_queue.has_items()
                ):
                    raise RuntimeError("Cannot continue from message role: assistant")

                # Claim + stamp under the lock (same invariants as prompt()):
                # without this, messages emitted by the resumed turn would
                # land with run_id=None and fork queries (which treat NULL
                # as legacy / always-copy) would happily copy them into a
                # later fork even if this resumed work is still in flight or
                # later abandoned.
                if self._run_aware and self.thread_id is not None:
                    assert self.checkpointer is not None  # narrowed by _run_aware
                    await self.checkpointer.claim_run(self.thread_id, effective_run_id)
                self._state.active_run_id = effective_run_id
                self._state.last_outcome = None

                if isinstance(last, AssistantMessage):
                    steering = self._steering_queue.drain()
                    if steering:
                        await self._run_prompt(steering)
                    else:
                        follow_ups = self._follow_up_queue.drain()
                        await self._run_prompt(follow_ups)
                else:
                    await self._run_continuation()
        except BaseException:
            # Spec §3.7 parity: leave active_run_id SET on failure so callers
            # can observe which run failed.
            self._pending_suspension_event = None
            raise
        else:
            outcome: RunOutcome = self._state.last_outcome or "abandoned"
            await self._dispatch_outcome(outcome, effective_run_id)
            self._state.active_run_id = None
            await self._emit_suspended_event(outcome)
            return effective_run_id

    def _build_stream_options(self, signal: asyncio.Event) -> StreamOptions:
        return StreamOptions(
            reasoning=self._state.reasoning,
            signal=signal,
            on_payload=self.on_payload,
            on_response=self.on_response,
        )

    async def _run_prompt(self, messages: list[Message]) -> None:
        from cubeloop.providers.fallback import begin_fallback_run, end_fallback_run

        token = begin_fallback_run()
        try:
            await self._run_prompt_body(messages)
        finally:
            end_fallback_run(token)

    async def _run_prompt_body(self, messages: list[Message]) -> None:
        sink = self._outcome_sink()
        await self._run_with_lifecycle(
            lambda signal: run_agent_loop(
                prompts=messages,
                context=self._create_context_snapshot(),
                model=self._model,
                convert_to_llm=self.convert_to_llm,
                transform_context=self.transform_context,
                transform_system_prompt=self.transform_system_prompt,
                after_model_response=self.after_model_response,
                before_tool_call=self.before_tool_call,
                resolve_tool_call=self.resolve_tool_call,
                after_tool_call=self.after_tool_call,
                should_stop_after_turn=self.should_stop_after_turn,
                on_run_end=self.on_run_end,
                get_steering_messages=self._make_async_drain(self._steering_queue),
                get_follow_up_messages=self._make_async_drain(self._follow_up_queue),
                stream_options=self._build_stream_options(signal),
                tool_execution=self.tool_execution,
                emit=lambda e: self._process_event(e),
                set_outcome=sink,
            )
        )

    async def _run_continuation(self) -> None:
        from cubeloop.providers.fallback import begin_fallback_run, end_fallback_run

        token = begin_fallback_run()
        try:
            await self._run_continuation_body()
        finally:
            end_fallback_run(token)

    async def _run_continuation_body(self) -> None:
        sink = self._outcome_sink()
        await self._run_with_lifecycle(
            lambda signal: run_agent_loop_continue(
                context=self._create_context_snapshot(),
                model=self._model,
                convert_to_llm=self.convert_to_llm,
                transform_context=self.transform_context,
                transform_system_prompt=self.transform_system_prompt,
                after_model_response=self.after_model_response,
                before_tool_call=self.before_tool_call,
                resolve_tool_call=self.resolve_tool_call,
                after_tool_call=self.after_tool_call,
                should_stop_after_turn=self.should_stop_after_turn,
                on_run_end=self.on_run_end,
                get_steering_messages=self._make_async_drain(self._steering_queue),
                get_follow_up_messages=self._make_async_drain(self._follow_up_queue),
                stream_options=self._build_stream_options(signal),
                tool_execution=self.tool_execution,
                emit=lambda e: self._process_event(e),
                set_outcome=sink,
            )
        )

    @staticmethod
    def _make_async_drain(queue: _MessageQueue) -> Callable:
        async def _drain() -> list[Message]:
            return queue.drain()

        return _drain

    def _create_context_snapshot(self) -> AgentContext:
        return AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state._messages),
            tools=list(self._state._tools),
            extra=self._extra,
            run_id=self._state.active_run_id,
            attempt_id=self.session.active_attempt_id,
            turn_execution_context=self.session._resume_turn_context,
            on_turn_context=self.session._capture_turn_context,
            set_input_admission=self.session._set_input_admission,
        )

    async def detach(self) -> None:
        if self._channel is None:
            raise HitlError("agent has no channel bound")
        pending = self._channel.pending
        if (
            pending is None
            or self._channel._future is None
            or self._channel._future.done()
        ):
            return  # nothing to detach
        # Snapshot the payload before the channel clears its in-memory pending slot,
        # then commit the control-flow transition. The owning prompt/resume task
        # publishes AgentSuspendedEvent only after it records the suspended outcome
        # and clears active_run_id, so observers cannot report an uncommitted pause.
        self._pending_suspension_event = pending
        future = self._channel._future
        future.add_done_callback(_consume_control_exception)
        future.set_exception(HitlDetached())

    async def load_pending_hitl_request(self) -> HitlRequest | None:
        if self.checkpointer is None or self.thread_id is None:
            return None
        load_pending = getattr(self.checkpointer, "load_pending_request", None)
        if load_pending is None:
            return None  # checkpointer doesn't support HITL — graceful None
        return await load_pending(self.thread_id)

    async def respond(
        self, *, question_id: str | None = None, answer: StructuredValue
    ) -> None:
        from cubeloop.session import RespondExecutionRequest

        self.session._assert_idle()
        recovered_run_id: str | None = None
        if self.checkpointer is not None and self.thread_id is not None:
            load_pending = getattr(self.checkpointer, "load_pending", None)
            if load_pending is not None:
                loaded = await load_pending(self.thread_id)
                if loaded is not None:
                    recovered_run_id = loaded[1]
        result = await self.session.execute(
            RespondExecutionRequest(
                run_id=recovered_run_id or uuid.uuid4().hex,
                attempt_id=uuid.uuid4().hex,
                question_id=question_id,
                answer=answer,
            )
        )
        if result.error is not None and result.error.cause is not None:
            raise result.error.cause

    async def _execute_respond(
        self,
        *,
        question_id: str | None = None,
        answer: StructuredValue,
        expected_run_id: str | None = None,
    ) -> None:
        from cubeloop.hitl.exceptions import (
            HitlNoPendingRequest,
            HitlStaleAnswer,
        )

        if self._channel is None:
            raise HitlError("agent has no channel bound")
        if not (self.thread_id and self.checkpointer):
            raise RuntimeError("respond() requires thread_id + checkpointer")

        load_pending = getattr(self.checkpointer, "load_pending", None)
        if load_pending is None:
            raise HitlError(
                "respond() requires a checkpointer that implements "
                "load_pending (added in checkpointer v4)"
            )
        save_hitl_answer = getattr(self.checkpointer, "save_hitl_answer", None)
        if save_hitl_answer is None:
            raise HitlError(
                "respond() requires a checkpointer that implements save_hitl_answer"
            )

        async with self._run_lock:
            if not self._checkpoint_loaded:
                data = await self.checkpointer.load(self.thread_id)
                if data:
                    self._state._messages = list(data.messages or [])
                    self._extra.clear()
                    self._extra.update(data.extra or {})
                self._checkpoint_loaded = True

            loaded = await load_pending(self.thread_id)
            if loaded is None:
                raise HitlNoPendingRequest("no pending request on this thread")
            pending, recovered_run_id = loaded
            if (
                recovered_run_id is not None
                and expected_run_id is not None
                and recovered_run_id != expected_run_id
            ):
                raise ValueError(
                    f"respond run_id={expected_run_id!r} does not match "
                    f"pending run_id={recovered_run_id!r}"
                )
            if question_id is None:
                question_id = pending.question_id
            if question_id != pending.question_id:
                raise HitlStaleAnswer(
                    f"answer for {question_id}, pending is {pending.question_id}"
                )

            # Validate HITL bindings on this Agent against the recovered
            # run_id. Without this, a host that built CheckpointedChannel
            # with a different run_id would persist a second pending row
            # under the wrong run, then the next resume would stamp
            # messages / try to complete an unclaimed run.
            self._validate_hitl_bindings(recovered_run_id, caller="respond")

            # Thread the recovered run_id into agent state so the resume loop
            # stamps appended messages and _dispatch_outcome can mark the
            # run complete. respond() does NOT call claim_run — the original
            # prompt() already claimed it (single-claim invariant per spec).
            if recovered_run_id is not None:
                self._state.active_run_id = recovered_run_id
            self._state.last_outcome = None

            await save_hitl_answer(
                self.thread_id,
                question_id,
                answer,
                run_id=recovered_run_id,
            )
            self._channel.attach_resume_answer(question_id, answer)
            try:
                await self._run_hitl_resume()
            except BaseException:
                # Spec §3.7: leave active_run_id SET on raise.
                self._pending_suspension_event = None
                raise
            else:
                # Legacy guard: pending persisted without run_id (older
                # save_pending_request callers) cannot drive dispatch.
                outcome: RunOutcome = self._state.last_outcome or "abandoned"
                if recovered_run_id is not None:
                    await self._dispatch_outcome(outcome, recovered_run_id)
                self._state.active_run_id = None
                await self._emit_suspended_event(outcome)

    async def abort_pending(
        self, reason: str = "aborted by host"
    ) -> None:  # pragma: no cover — E2E tested
        """Abort a pending HITL request and CLOSE the conversation.

        Per spec §5.2 "abort closes the conversation" — no new model call.
        Two-phase: Phase 1 (no lock) interrupts any in-flight HITL await via
        the agent signal; Phase 2 (with lock) appends synthetic deny
        tool_results + a terminal stop_reason="aborted" assistant message
        and emits AgentAbortedEvent.
        """
        from cubeloop.agent.types import AgentAbortedEvent

        if self._channel is None:
            raise HitlError("agent has no channel bound")
        if not (self.thread_id and self.checkpointer):
            raise RuntimeError("abort_pending() requires thread_id + checkpointer")

        # ============= Phase 1: interrupt any in-flight HITL await =============
        # If prompt() is currently suspended in channel.{ask,confirm,approve},
        # set the agent signal. _BaseChannel._await_answer races signal vs
        # future and raises HitlAborted when signal wins. HitlAborted
        # propagates through tool/middleware (HitlControlException is re-raised
        # by the selective handler in _execute_prepared) up to _run_loop's
        # outer silent catch. The HITL channel's finally calls
        # _on_pending_cleared(exc=HitlAborted) which clears persisted pending
        # (HitlAborted != HitlDetached).
        #
        # CRITICAL: do NOT acquire _run_lock here — prompt() holds it.
        in_flight = self._channel.pending
        if in_flight is not None:
            if self._active_signal is not None:
                self._active_signal.set()
            else:
                # Edge: channel has pending but agent has no active signal
                # (e.g. respond() race window). Cancel directly.
                await self._channel.cancel(in_flight.question_id, reason=reason)

        # ============= Phase 2: append synthetic deny + close conversation =====
        async with self._run_lock:
            save_pending = getattr(self.checkpointer, "save_pending_request", None)
            if save_pending is None:
                raise HitlError(
                    "abort_pending() requires a checkpointer that implements "
                    "save_pending_request"
                )

            # Reload messages from checkpoint to see whatever prompt() persisted.
            data = await self.checkpointer.load(self.thread_id)
            self._state._messages = list(data.messages or []) if data else []
            if not self._state._messages:
                # Nothing to close (no in-flight, no persisted history).
                await self._process_event(AgentAbortedEvent(reason=reason))
                return

            # Scan BACKWARDS for the most recent AssistantMessage that has
            # tool_calls still unresolved. We cannot just look at the tail —
            # an in-flight execute may have partially appended ToolResultMessage(s)
            # before signal-abort fired, leaving the tail as a ToolResultMessage
            # but the originating assistant turn still needing synthetic deny
            # for the OTHER tool_calls in the same batch.
            asst_pos = -1
            last_assistant = None
            for i in range(len(self._state._messages) - 1, -1, -1):
                msg = self._state._messages[i]
                if not isinstance(msg, AssistantMessage):
                    continue
                tcs = [c for c in msg.content if isinstance(c, ToolCall)]
                if not tcs:
                    continue
                already = {
                    m.tool_call_id
                    for m in self._state._messages[i + 1 :]
                    if isinstance(m, ToolResultMessage)
                }
                if any(tc.id not in already for tc in tcs):
                    asst_pos = i
                    last_assistant = msg
                    break

            if last_assistant is None or asst_pos < 0:
                # No unresolved assistant turn — conversation already closed
                # by some other path. Still clear pending + emit for observability.
                await save_pending(self.thread_id, None)
                clear_answers = getattr(self.checkpointer, "clear_hitl_answers", None)
                if clear_answers is not None:
                    await clear_answers(
                        self.thread_id, run_id=self._state.active_run_id
                    )
                await self._process_event(AgentAbortedEvent(reason=reason))
                return

            last = last_assistant
            tool_call_ids = [c.id for c in last.content if isinstance(c, ToolCall)]
            already_resolved = {
                m.tool_call_id
                for m in self._state._messages[asst_pos + 1 :]
                if isinstance(m, ToolResultMessage)
            }
            unresolved = [
                tc_id for tc_id in tool_call_ids if tc_id not in already_resolved
            ]

            # Stamp synthetic cleanup messages with the originating
            # assistant's run_id so a later fork after an EARLIER completed
            # run cannot copy aborted-run artifacts as "legacy" history
            # (fork queries treat run_id IS NULL as always-copy).
            owning_run_id = last.run_id

            # Synthesize deny tool_result for each unresolved tool_call.
            for tc_id in unresolved:
                tc = next(
                    c for c in last.content if isinstance(c, ToolCall) and c.id == tc_id
                )
                synthetic = ToolResultMessage(
                    tool_call_id=tc_id,
                    tool_name=tc.name,
                    content=[TextContent(text=f"aborted: {reason}")],
                    details={"hitl": {"decision": "aborted", "reason": reason}},
                    is_error=True,
                    timestamp=time.time(),
                    run_id=owning_run_id,
                )
                self._state._messages.append(synthetic)
                await self.checkpointer.append(self.thread_id, [synthetic])

            # Append terminal aborted assistant only if we actually appended
            # synthetic denials — otherwise the conversation already closed.
            if unresolved:
                term = AssistantMessage(
                    content=[TextContent(text=f"Conversation aborted: {reason}")],
                    stop_reason="aborted",
                    usage=Usage(),
                    timestamp=time.time(),
                    run_id=owning_run_id,
                )
                self._state._messages.append(term)
                await self.checkpointer.append(self.thread_id, [term])

            # Defensive clear (Phase 1's _on_pending_cleared usually did this,
            # but cross-process abort_pending may bypass Phase 1 entirely).
            await save_pending(self.thread_id, None)
            clear_answers = getattr(self.checkpointer, "clear_hitl_answers", None)
            if clear_answers is not None:
                await clear_answers(self.thread_id, run_id=owning_run_id)
            await self._process_event(AgentAbortedEvent(reason=reason))

    async def _run_hitl_resume(self) -> None:
        from cubeloop.providers.fallback import (
            begin_fallback_run,
            end_fallback_run,
            get_active_index,
            set_active_index,
        )

        token = begin_fallback_run()
        set_active_index(self._fallback_active_index)
        try:
            await self._run_hitl_resume_body()
        finally:
            self._fallback_active_index = get_active_index()
            end_fallback_run(token)

    async def _run_hitl_resume_body(self) -> None:
        sink = self._outcome_sink()
        await self._run_with_lifecycle(
            lambda signal: run_agent_loop_resume(
                context=self._create_context_snapshot(),
                model=self._model,
                convert_to_llm=self.convert_to_llm,
                transform_context=self.transform_context,
                transform_system_prompt=self.transform_system_prompt,
                after_model_response=self.after_model_response,
                before_tool_call=self.before_tool_call,
                resolve_tool_call=self.resolve_tool_call,
                after_tool_call=self.after_tool_call,
                should_stop_after_turn=self.should_stop_after_turn,
                on_run_end=self.on_run_end,
                get_steering_messages=self._make_async_drain(self._steering_queue),
                get_follow_up_messages=self._make_async_drain(self._follow_up_queue),
                stream_options=self._build_stream_options(signal),
                tool_execution=self.tool_execution,
                emit=lambda e: self._process_event(e),
                checkpointer=self.checkpointer,
                thread_id=self.thread_id,
                set_outcome=sink,
            )
        )

    async def _run_with_lifecycle(self, executor: Callable) -> None:
        signal = asyncio.Event()
        done = asyncio.Event()
        self._active_signal = signal
        self._active_done = done
        self._state.is_streaming = True
        self._state.streaming_message = None
        self._state.error_message = None
        if self.session._cancel_requested:
            signal.set()

        try:
            await executor(signal)
        except asyncio.CancelledError:
            # A cancel can land after the assistant message (carrying
            # tool_calls) was checkpointed but before the tool_results were.
            # That leaves the persisted history with orphan tool_calls, which
            # every provider rejects on the next turn. Backfill synthetic
            # tool_results so the thread stays resumable, then re-raise.
            await self._complete_cancelled_tool_calls()
            raise
        except Exception as error:
            await self._handle_run_failure(error, signal.is_set())
        finally:
            self._state.is_streaming = False
            self._state.streaming_message = None
            self._state._pending_tool_calls = set()
            self._active_signal = None
            done.set()
            self._active_done = None

    async def _handle_run_failure(self, error: Exception, aborted: bool) -> None:
        failure_message = AssistantMessage(
            content=[TextContent(text="")],
            stop_reason="aborted" if aborted else "error",
            error_message=str(error),
            usage=Usage(),
            timestamp=time.time(),
        )
        await self._process_event(MessageStartEvent(message=failure_message))
        await self._process_event(MessageEndEvent(message=failure_message))
        await self._process_event(
            TurnEndEvent(message=failure_message, tool_results=[])
        )
        await self._process_event(AgentEndEvent(messages=[failure_message]))

    async def _complete_cancelled_tool_calls(self) -> None:
        """Backfill tool_results for tool_calls left dangling by a cancel.

        Best-effort: a checkpoint failure here must never mask the
        CancelledError that triggered cleanup. Only the most recent assistant
        message can have un-answered tool_calls (the loop appends tool_results
        immediately after), so we scan back to it and synthesize a result for
        every tool_call id that has no ToolResultMessage yet.
        """
        try:
            last_idx = -1
            for i in range(len(self._state._messages) - 1, -1, -1):
                if isinstance(self._state._messages[i], AssistantMessage):
                    last_idx = i
                    break
            if last_idx == -1:
                return
            last_assistant = self._state._messages[last_idx]
            assert isinstance(last_assistant, AssistantMessage)

            # Only results from this turn count as answered — tool_call ids
            # are not globally unique, so scanning all history could treat a
            # reused id from an earlier turn as already answered and skip the
            # backfill, leaving the thread wedged.
            answered = {
                m.tool_call_id
                for m in self._state._messages[last_idx + 1 :]
                if isinstance(m, ToolResultMessage)
            }
            # Stamp synthetic results with the originating assistant's
            # run_id so a fork after an earlier completed run cannot copy
            # them as "legacy" history (fork treats run_id IS NULL as
            # always-copy; an uncompleted run's tool_results would be
            # orphaned in the forked transcript).
            owning_run_id = last_assistant.run_id
            synthetic: list[Message] = []
            for block in last_assistant.content:
                if not isinstance(block, ToolCall) or block.id in answered:
                    continue
                synthetic.append(
                    ToolResultMessage(
                        tool_call_id=block.id,
                        tool_name=block.name,
                        content=[
                            TextContent(text="[Tool execution cancelled by user]")
                        ],
                        is_error=True,
                        timestamp=time.time(),
                        run_id=owning_run_id,
                    )
                )

            if not synthetic:
                return
            self._state._messages.extend(synthetic)
            if self.checkpointer and self.thread_id:
                await self.checkpointer.append(self.thread_id, synthetic)
        except asyncio.CancelledError:  # pragma: no cover - re-raise the trigger
            raise
        except Exception:  # pragma: no cover - cleanup must never mask the cancel
            pass

    async def _process_event(self, event: AgentEvent) -> None:
        if event.type == "message_start":
            self._state.streaming_message = event.message
        elif event.type == "message_update":
            self._state.streaming_message = event.message
        elif event.type == "message_end":
            msg = event.message
            active = self._state.active_run_id
            if active is not None:
                if msg.run_id is None:
                    if isinstance(msg, AssistantMessage):
                        # The loop retains this provider-owned object for tool
                        # execution and the next model call. Stamp it in place
                        # so tool results can inherit the owning turn's run ID.
                        msg.run_id = active
                    else:
                        msg = msg.model_copy(update={"run_id": active})
                        event = event.model_copy(update={"message": msg})
                elif msg.run_id != active:
                    raise ValueError(
                        f"message.run_id={msg.run_id!r} does not match "
                        f"active run_id={active!r}"
                    )
            self._state.streaming_message = None
            self._state._messages.append(msg)
            if self.checkpointer and self.thread_id:
                await self.checkpointer.append(self.thread_id, [msg])
        elif event.type == "tool_execution_start":
            self._state._pending_tool_calls = self._state._pending_tool_calls | {
                event.tool_call_id
            }
        elif event.type == "tool_execution_end":
            self._state._pending_tool_calls = self._state._pending_tool_calls - {
                event.tool_call_id
            }
        elif event.type == "turn_end":
            msg = event.message
            if isinstance(msg, AssistantMessage) and msg.error_message:
                self._state.error_message = msg.error_message
        elif event.type == "agent_end":
            self._state.streaming_message = None
            if self.checkpointer and self.thread_id:
                await self.checkpointer.save_extra(self.thread_id, self._extra)

        await self.session._publish_agent_event(event)
        await self._emit_to_listeners(event)

    async def _emit_to_listeners(self, event: AgentEvent) -> None:
        cancellation: asyncio.CancelledError | None = None
        first_error: Exception | None = None
        for listener in tuple(self._listeners):
            try:
                result = listener(event, self._active_signal)
                if asyncio.iscoroutine(result):
                    await result
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
            except Exception as exc:
                # One observer must not hide an event from later observers.
                # Preserve existing error propagation after every listener had
                # the chance to see the same committed state transition.
                if first_error is None:
                    first_error = exc
        if cancellation is not None:
            raise cancellation
        if first_error is not None:
            raise first_error
