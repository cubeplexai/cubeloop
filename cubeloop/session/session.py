from __future__ import annotations

import asyncio
import copy
import inspect
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from cubeloop.agent._tool_cycle import ToolCycleViolation, check_tool_cycle
from cubeloop.agent.types import AgentEndEvent, AgentSuspendedEvent, MessageEndEvent
from cubeloop.checkpointer.base import CheckpointData
from cubeloop.hitl.types import HitlRequest
from cubeloop.session.events import (
    ExecutionEventEnvelope,
    ExecutionFinished,
    InputCommitted,
    SessionEvent,
)
from cubeloop.session.input import (
    InputDurability,
    InputEnvelope,
    InputReceipt,
    InputStatus,
)
from cubeloop.session.types import (
    ContinueExecutionRequest,
    DeliveryError,
    ExecutionBusy,
    ExecutionError,
    ExecutionOutcome,
    ExecutionRequest,
    ExecutionResult,
    PromptExecutionRequest,
    RespondExecutionRequest,
)
from cubeloop.types import JsonObject

if TYPE_CHECKING:
    from cubeloop.agent.agent import Agent
    from cubeloop.providers.base import Message
    from cubeloop.session.turn_execution_context import TurnExecutionContext


@dataclass
class _Consumer:
    listener: Callable[[ExecutionEventEnvelope], Any]
    name: str
    required: bool
    capacity: int
    delivery_timeout: float
    queue: asyncio.Queue[_DeliveryItem]
    task: asyncio.Task[None] | None = None
    closed: bool = False


@dataclass
class _DeliveryItem:
    envelope: ExecutionEventEnvelope
    acknowledged: asyncio.Future[None]


class _RequiredConsumerError(RuntimeError):
    pass


class _ConsumerClosedError(RuntimeError):
    pass


_INPUT_DEDUP_CAPACITY = 4096


class ExecutionSession:
    """Public lifecycle for sequential execution attempts owned by one Agent."""

    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        self._active_attempt_id: str | None = None
        self._active_run_id: str | None = None
        self._cancel_requested = False
        self._accepting_cancel = False
        self._accepting_input = False
        self._idle = asyncio.Event()
        self._idle.set()
        self._seq = 0
        self._publish_lock = asyncio.Lock()
        self._required_delivery_failure: tuple[_Consumer, DeliveryError] | None = None
        self._consumers: list[_Consumer] = []
        self._input_status: dict[str, InputStatus] = {}
        self._input_durability: dict[str, InputDurability] = {}
        self._queued_inputs: dict[str, Message] = {}
        self._delivery_errors: list[DeliveryError] = []
        self._checkpoint_write_failed = False
        self._turn_contexts: list[TurnExecutionContext] = []
        self._resume_turn_context: TurnExecutionContext | None = None
        self._pending_request: HitlRequest | None = None

    @property
    def active_attempt_id(self) -> str | None:
        return self._active_attempt_id

    @property
    def turn_contexts(self) -> tuple[TurnExecutionContext, ...]:
        return tuple(self._turn_contexts)

    def _capture_turn_context(self, context: TurnExecutionContext) -> None:
        for index in range(len(self._turn_contexts) - 1, -1, -1):
            existing = self._turn_contexts[index]
            if (
                existing.turn_id == context.turn_id
                and existing.model == context.model
                and existing.model_attempt_id == context.model_attempt_id
            ):
                self._turn_contexts[index] = context
                return
        self._turn_contexts.append(context)

    @property
    def state_context(self) -> JsonObject:
        """The stable, live mapping persisted as checkpoint extra state."""
        return self._agent._extra

    async def load_checkpoint(self) -> CheckpointData | None:
        """Atomically install checkpoint messages and extra while idle."""
        if self._active_attempt_id is not None or self._agent._run_lock.locked():
            raise ExecutionBusy("cannot load a checkpoint during execution")
        if self._agent.checkpointer is None or self._agent.thread_id is None:
            return None

        async with self._agent._run_lock:
            data = await self._agent.checkpointer.load(self._agent.thread_id)
            if data is None:
                return None
            installed = copy.deepcopy(data)
            exposed = copy.deepcopy(data)
            self._agent._state._messages = list(installed.messages)
            self._agent._extra.clear()
            self._agent._extra.update(installed.extra)
            self._agent._checkpoint_loaded = True
            self._drop_memory_input_receipts()
            self._checkpoint_write_failed = False
            return exposed

    def request_cancel(self) -> None:
        if not self._accepting_cancel:
            return
        self._cancel_requested = True
        self._agent.abort()

    async def wait_for_idle(self) -> None:
        await self._idle.wait()

    async def request_detach(self) -> None:
        if self._agent.channel is None or self._agent.channel.pending is None:
            raise RuntimeError("execution can detach only at a pending HITL safe point")
        await self._agent.detach()

    def subscribe(
        self,
        listener: Callable[[ExecutionEventEnvelope], Any],
        *,
        required: bool = False,
        name: str | None = None,
        capacity: int = 256,
        delivery_timeout: float = 1.0,
    ) -> Callable[[], None]:
        if capacity < 1:
            raise ValueError("consumer capacity must be positive")
        if delivery_timeout <= 0:
            raise ValueError("consumer delivery_timeout must be positive")
        consumer_name = name or getattr(listener, "__name__", "session-consumer")
        consumer = _Consumer(
            listener=listener,
            name=str(consumer_name),
            required=required,
            capacity=capacity,
            delivery_timeout=delivery_timeout,
            queue=asyncio.Queue(maxsize=capacity),
        )
        self._consumers.append(consumer)

        def unsubscribe() -> None:
            consumer.closed = True
            if consumer.task is not None:
                consumer.task.cancel()
            if consumer in self._consumers:
                self._consumers.remove(consumer)
            if (
                self._required_delivery_failure is not None
                and self._required_delivery_failure[0] is consumer
            ):
                self._required_delivery_failure = None

        return unsubscribe

    def submit_input(self, envelope: InputEnvelope) -> InputReceipt:
        existing = self._input_status.get(envelope.input_id)
        if existing is not None:
            durability = self._input_durability.get(envelope.input_id)
            return InputReceipt(
                input_id=envelope.input_id,
                status=existing,
                durability=durability,
            )
        if not self._accepting_input:
            return InputReceipt(input_id=envelope.input_id, status="closed")
        if (
            envelope.message.run_id is not None
            and envelope.message.run_id != self._active_run_id
        ):
            return InputReceipt(input_id=envelope.input_id, status="closed")
        if len(self._input_status) >= _INPUT_DEDUP_CAPACITY:
            terminal_id = next(
                (
                    input_id
                    for input_id, status in self._input_status.items()
                    if status != "queued"
                ),
                None,
            )
            if terminal_id is None:
                return InputReceipt(input_id=envelope.input_id, status="closed")
            del self._input_status[terminal_id]
            self._input_durability.pop(terminal_id, None)

        message = envelope.message.model_copy(deep=True)
        metadata = dict(message.metadata)
        metadata["input_id"] = envelope.input_id
        metadata["input_mode"] = envelope.mode
        metadata["steer_id"] = envelope.input_id
        message = message.model_copy(
            update={"metadata": metadata, "run_id": self._active_run_id}
        )
        if envelope.mode == "steer":
            self._agent.steer(message)
        else:
            self._agent.follow_up(message)
        self._input_status[envelope.input_id] = "queued"
        self._queued_inputs[envelope.input_id] = message
        return InputReceipt(input_id=envelope.input_id, status="queued")

    def cancel_input(self, input_id: str) -> InputReceipt:
        status = self._input_status.get(input_id)
        if status == "committed":
            return InputReceipt(
                input_id=input_id,
                status="committed",
                durability=self._input_durability.get(input_id, "memory"),
            )
        if status == "cancelled":
            return InputReceipt(input_id=input_id, status="cancelled")
        if status != "queued":
            return InputReceipt(input_id=input_id, status="closed")

        removed = self._agent._steering_queue.remove(input_id)
        removed = self._agent._follow_up_queue.remove(input_id) or removed
        if not removed:
            return InputReceipt(input_id=input_id, status="closed")
        self._input_status[input_id] = "cancelled"
        self._queued_inputs.pop(input_id, None)
        return InputReceipt(input_id=input_id, status="cancelled")

    def _reset_inputs(self) -> None:
        self._input_status.clear()
        self._input_durability.clear()
        self._queued_inputs.clear()

    def _drop_memory_input_receipts(self) -> None:
        for input_id, durability in tuple(self._input_durability.items()):
            if durability != "memory":
                continue
            self._input_status.pop(input_id, None)
            self._input_durability.pop(input_id, None)
            self._queued_inputs.pop(input_id, None)

    def _mark_checkpoint_write_failure(self) -> None:
        self._checkpoint_write_failed = True

    def _reconcile_consumed_inputs(self) -> list[str]:
        committed: list[str] = []
        for input_id, status in tuple(self._input_status.items()):
            if status != "queued":
                continue
            queued_message = self._queued_inputs.get(input_id)
            if queued_message is None or not any(
                message is queued_message for message in self._agent.state.messages
            ):
                continue
            self._input_status[input_id] = "committed"
            self._input_durability[input_id] = "memory"
            self._queued_inputs.pop(input_id, None)
            committed.append(input_id)
        return committed

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute one attempt and return settled lifecycle facts."""
        self._assert_idle()

        self._idle.clear()
        self._active_attempt_id = request.attempt_id
        self._active_run_id = request.run_id
        self._cancel_requested = False
        self._accepting_cancel = True
        self._accepting_input = True
        self._seq = 0
        self._delivery_errors = []
        self._reconcile_consumed_inputs()
        for input_id, status in tuple(self._input_status.items()):
            if status != "queued":
                continue
            self._agent._steering_queue.remove(input_id)
            self._agent._follow_up_queue.remove(input_id)
            del self._input_status[input_id]
            self._input_durability.pop(input_id, None)
            self._queued_inputs.pop(input_id, None)
        self._resume_turn_context = None
        if isinstance(request, RespondExecutionRequest):
            self._resume_turn_context = next(
                (
                    context
                    for context in reversed(self._turn_contexts)
                    if context.run_id == request.run_id
                ),
                None,
            )
        self._turn_contexts.clear()
        if self._resume_turn_context is not None:
            self._turn_contexts.append(self._resume_turn_context)
        self._pending_request = None
        caught: BaseException | None = None
        try:
            if isinstance(request, PromptExecutionRequest):
                payload = request.message
                if isinstance(payload, list):
                    payload = [message.model_copy(deep=True) for message in payload]
                elif not isinstance(payload, str):
                    payload = payload.model_copy(deep=True)
                await self._agent._execute_prompt(payload, run_id=request.run_id)
            elif isinstance(request, RespondExecutionRequest):
                answer = copy.deepcopy(request.answer)
                await self._agent._execute_respond(
                    question_id=request.question_id,
                    answer=answer,
                    expected_run_id=request.run_id,
                )
            elif isinstance(request, ContinueExecutionRequest):
                await self._agent._execute_continue(run_id=request.run_id)
            else:  # pragma: no cover - closed typed union
                raise TypeError(f"unsupported execution request: {type(request)!r}")
        except asyncio.CancelledError as exc:
            caught = exc
        except Exception as exc:
            caught = exc
        except BaseException:
            self._release_attempt()
            raise
        finally:
            self._accepting_input = False
            self._accepting_cancel = False

        try:
            for input_id in self._reconcile_consumed_inputs():
                await self._publish(
                    InputCommitted(input_id=input_id, durability="memory"),
                    terminal=False,
                )
        except asyncio.CancelledError as exc:
            if caught is None:
                caught = exc
        except Exception as exc:
            if caught is None:
                caught = exc
        except BaseException:
            self._release_attempt()
            raise

        try:
            try:
                result = await self._settle(request, caught)
            except asyncio.CancelledError as exc:
                result = self._cancelled_result(request, exc)
            except Exception as exc:
                result = ExecutionResult(
                    run_id=request.run_id,
                    attempt_id=request.attempt_id,
                    outcome="failed",
                    error=ExecutionError(
                        kind="finalization",
                        message=str(exc),
                        cause=exc,
                    ),
                    checkpoint_committed=False,
                    history_consistent=False,
                    delivery_errors=tuple(self._delivery_errors),
                )
            publish_task = asyncio.create_task(
                self._publish(ExecutionFinished(result=result), terminal=True)
            )
            while True:
                try:
                    final_errors = await asyncio.shield(publish_task)
                    break
                except asyncio.CancelledError:
                    if publish_task.done():
                        final_errors = publish_task.result()
                        break
            if final_errors:
                result = replace(
                    result,
                    delivery_errors=tuple([*result.delivery_errors, *final_errors]),
                )
            return result
        finally:
            self._release_attempt()

    def _release_attempt(self) -> None:
        self._active_attempt_id = None
        self._active_run_id = None
        self._resume_turn_context = None
        self._idle.set()

    def _assert_idle(self) -> None:
        if self._active_attempt_id is not None or self._agent._run_lock.locked():
            raise ExecutionBusy("execution session is already processing an attempt")

    def _set_input_admission(self, accepting: bool) -> None:
        self._accepting_input = accepting

    def _cancelled_result(
        self,
        request: ExecutionRequest,
        cause: asyncio.CancelledError,
    ) -> ExecutionResult:
        return ExecutionResult(
            run_id=request.run_id,
            attempt_id=request.attempt_id,
            outcome="cancelled",
            error=ExecutionError(
                kind="cancelled",
                message="execution cancelled",
                cause=cause,
            ),
            checkpoint_committed=False,
            history_consistent=False,
            delivery_errors=tuple(self._delivery_errors),
        )

    async def _publish_agent_event(self, event: SessionEvent) -> None:
        if self._active_attempt_id is None:
            return
        if isinstance(event, (AgentEndEvent, AgentSuspendedEvent)):
            self._accepting_input = False
            self._accepting_cancel = False
        if isinstance(event, AgentSuspendedEvent):
            self._pending_request = event.pending_request.model_copy(deep=True)
        committed_input_id: str | None = None
        if isinstance(event, MessageEndEvent):
            input_id = event.message.metadata.get("input_id")
            if (
                isinstance(input_id, str)
                and self._input_status.get(input_id) == "queued"
                and self._queued_inputs.get(input_id) is event.message
            ):
                self._input_status[input_id] = "committed"
                self._input_durability[input_id] = (
                    "checkpoint" if self._has_durable_checkpoint() else "memory"
                )
                self._queued_inputs.pop(input_id, None)
                committed_input_id = input_id
        await self._publish(event, terminal=False)
        if committed_input_id is None:
            return
        durability: InputDurability = (
            "checkpoint" if self._has_durable_checkpoint() else "memory"
        )
        await self._publish(
            InputCommitted(input_id=committed_input_id, durability=durability),
            terminal=False,
        )

    async def _publish(
        self,
        event: SessionEvent,
        *,
        terminal: bool,
    ) -> list[DeliveryError]:
        async with self._publish_lock:
            return await self._publish_serialized(event, terminal=terminal)

    async def _publish_serialized(
        self,
        event: SessionEvent,
        *,
        terminal: bool,
    ) -> list[DeliveryError]:
        if not terminal and self._required_delivery_failure is not None:
            _, failure = self._required_delivery_failure
            raise _RequiredConsumerError(
                f"required consumer {failure.consumer} failed: {failure.message}"
            )
        self._seq += 1
        envelope = ExecutionEventEnvelope(
            run_id=self._active_run_id or "",
            attempt_id=self._active_attempt_id or "",
            seq=self._seq,
            event=event,
        )
        errors: list[DeliveryError] = []
        required_failure: tuple[_Consumer, DeliveryError] | None = None
        for consumer in tuple(self._consumers):
            if consumer.closed:
                continue
            try:
                if consumer.task is None:
                    consumer.task = asyncio.create_task(self._consume(consumer))
                acknowledged = asyncio.get_running_loop().create_future()
                item = _DeliveryItem(
                    envelope=self._snapshot_envelope(envelope),
                    acknowledged=acknowledged,
                )
                await asyncio.wait_for(
                    consumer.queue.put(item),
                    timeout=consumer.delivery_timeout,
                )
                await asyncio.wait_for(
                    asyncio.shield(acknowledged),
                    timeout=consumer.delivery_timeout,
                )
            except TimeoutError:
                acknowledged.cancel()
                diagnostic = DeliveryError(
                    consumer=consumer.name,
                    seq=envelope.seq,
                    reason="timeout",
                    message="consumer delivery timed out",
                )
                errors.append(diagnostic)
                self._close_consumer(consumer)
                if consumer.required and not terminal and required_failure is None:
                    required_failure = (consumer, diagnostic)
            except Exception as exc:
                diagnostic = DeliveryError(
                    consumer=consumer.name,
                    seq=envelope.seq,
                    reason="error",
                    message=str(exc),
                )
                errors.append(diagnostic)
                self._close_consumer(consumer)
                if consumer.required and not terminal and required_failure is None:
                    required_failure = (consumer, diagnostic)
        self._delivery_errors.extend(errors)
        if required_failure is not None:
            failed_consumer, diagnostic = required_failure
            self._required_delivery_failure = required_failure
            raise _RequiredConsumerError(
                f"required consumer {failed_consumer.name} failed: {diagnostic.message}"
            )
        return errors

    @staticmethod
    def _snapshot_envelope(
        envelope: ExecutionEventEnvelope,
    ) -> ExecutionEventEnvelope:
        event = envelope.event
        if isinstance(event, BaseModel):
            event = event.model_copy(deep=True)
        elif isinstance(event, ExecutionFinished):
            pending = event.result.pending_request
            error = event.result.error
            result = replace(
                event.result,
                pending_request=(
                    pending.model_copy(deep=True) if pending is not None else None
                ),
                error=replace(error, cause=None) if error is not None else None,
            )
            event = ExecutionFinished(result=result)
        return replace(envelope, event=event)

    async def _consume(self, consumer: _Consumer) -> None:
        while True:
            item = await consumer.queue.get()
            try:
                value = consumer.listener(item.envelope)
                if inspect.isawaitable(value):
                    await value
            except asyncio.CancelledError:
                if not item.acknowledged.done():
                    item.acknowledged.set_exception(
                        _ConsumerClosedError("consumer task was cancelled")
                    )
                raise
            except BaseException as exc:
                if not item.acknowledged.done():
                    item.acknowledged.set_exception(exc)
            else:
                if not item.acknowledged.done():
                    item.acknowledged.set_result(None)
            finally:
                consumer.queue.task_done()

    @staticmethod
    def _close_consumer(consumer: _Consumer) -> None:
        consumer.closed = True
        if consumer.task is not None:
            consumer.task.cancel()

    def _has_durable_checkpoint(self) -> bool:
        return (
            self._agent.checkpointer is not None and self._agent.thread_id is not None
        )

    async def _settle(
        self,
        request: ExecutionRequest,
        caught: BaseException | None,
    ) -> ExecutionResult:
        tool_cycle_consistent = True
        run_messages = [
            message
            for message in self._agent.state.messages
            if message.run_id == request.run_id
        ]
        try:
            check_tool_cycle(run_messages)
        except ToolCycleViolation:
            tool_cycle_consistent = False
        history_consistent = tool_cycle_consistent and not self._checkpoint_write_failed

        pending = None
        pending_is_durable = False
        if self._agent.checkpointer is not None and self._agent.thread_id is not None:
            load_pending = getattr(self._agent.checkpointer, "load_pending", None)
            if load_pending is not None:
                loaded = await load_pending(self._agent.thread_id)
                if loaded is not None and loaded[1] == request.run_id:
                    pending = loaded[0].model_copy(deep=True)
                    pending_is_durable = True
        if pending is None:
            pending = self._pending_request

        private_outcome = self._agent.state.last_outcome
        error: ExecutionError | None = None
        if isinstance(caught, asyncio.CancelledError):
            outcome: ExecutionOutcome = "cancelled"
            error = ExecutionError(
                kind="cancelled",
                message="execution cancelled",
                cause=caught,
            )
        elif caught is not None:
            outcome = "failed"
            error = ExecutionError(
                kind="execution",
                message=str(caught),
                cause=caught,
            )
        elif (
            history_consistent
            and private_outcome == "complete"
            and not self._agent.state.error_message
        ):
            outcome = "completed"
        elif self._cancel_requested:
            outcome = "cancelled"
            error = ExecutionError(
                kind="cancelled",
                message="execution cancelled",
            )
        elif private_outcome == "suspended":
            outcome = "suspended"
        elif not tool_cycle_consistent or private_outcome == "incomplete":
            outcome = "incomplete"
            error = ExecutionError(
                kind="inconsistent",
                message="execution history has incomplete tool-call results",
            )
        else:
            outcome = "failed"
            message = (
                self._agent.state.error_message or "execution ended without a cause"
            )
            error = ExecutionError(kind="execution", message=message)

        checkpoint_committed = self._has_durable_checkpoint() and (
            (outcome == "completed" and self._agent._run_aware)
            or (outcome == "suspended" and pending_is_durable)
        )
        return ExecutionResult(
            run_id=request.run_id,
            attempt_id=request.attempt_id,
            outcome=outcome,
            pending_request=pending if outcome == "suspended" else None,
            error=error,
            checkpoint_committed=checkpoint_committed,
            history_consistent=history_consistent,
            delivery_errors=tuple(self._delivery_errors),
        )
