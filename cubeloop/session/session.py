from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from cubeloop.agent._tool_cycle import ToolCycleViolation, check_tool_cycle
from cubeloop.agent.types import MessageEndEvent
from cubeloop.checkpointer.base import CheckpointData
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


_INPUT_DEDUP_CAPACITY = 4096


class ExecutionSession:
    """Public lifecycle for sequential execution attempts owned by one Agent."""

    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        self._active_attempt_id: str | None = None
        self._active_run_id: str | None = None
        self._cancel_requested = False
        self._seq = 0
        self._consumers: list[_Consumer] = []
        self._input_status: dict[str, InputStatus] = {}
        self._delivery_errors: list[DeliveryError] = []
        self._turn_contexts: list[TurnExecutionContext] = []

    @property
    def active_attempt_id(self) -> str | None:
        return self._active_attempt_id

    @property
    def turn_contexts(self) -> tuple[TurnExecutionContext, ...]:
        return tuple(self._turn_contexts)

    def _capture_turn_context(self, context: TurnExecutionContext) -> None:
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
            messages = list(data.messages)
            extra = dict(data.extra)
            self._agent._state._messages = messages
            self._agent._extra.clear()
            self._agent._extra.update(extra)
            self._agent._checkpoint_loaded = True
            return data

    def request_cancel(self) -> None:
        self._cancel_requested = True
        self._agent.abort()

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

        return unsubscribe

    def submit_input(self, envelope: InputEnvelope) -> InputReceipt:
        existing = self._input_status.get(envelope.input_id)
        if existing is not None:
            return InputReceipt(input_id=envelope.input_id, status=existing)
        if self._active_attempt_id is None:
            return InputReceipt(input_id=envelope.input_id, status="closed")
        if len(self._input_status) >= _INPUT_DEDUP_CAPACITY:
            return InputReceipt(input_id=envelope.input_id, status="closed")

        metadata = dict(envelope.message.metadata)
        metadata["input_id"] = envelope.input_id
        metadata["input_mode"] = envelope.mode
        metadata.setdefault("steer_id", envelope.input_id)
        message = envelope.message.model_copy(update={"metadata": metadata})
        if envelope.mode == "steer":
            self._agent.steer(message)
        else:
            self._agent.follow_up(message)
        self._input_status[envelope.input_id] = "queued"
        return InputReceipt(input_id=envelope.input_id, status="queued")

    def cancel_input(self, input_id: str) -> InputReceipt:
        status = self._input_status.get(input_id)
        if status == "committed":
            durability: InputDurability = (
                "checkpoint" if self._has_durable_checkpoint() else "memory"
            )
            return InputReceipt(
                input_id=input_id,
                status="committed",
                durability=durability,
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
        return InputReceipt(input_id=input_id, status="cancelled")

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute one attempt and return settled lifecycle facts."""
        if self._active_attempt_id is not None or self._agent._run_lock.locked():
            raise ExecutionBusy("execution session is already processing an attempt")

        self._active_attempt_id = request.attempt_id
        self._active_run_id = request.run_id
        self._cancel_requested = False
        self._seq = 0
        self._delivery_errors = []
        self._input_status.clear()
        self._turn_contexts.clear()
        caught: BaseException | None = None
        try:
            if isinstance(request, PromptExecutionRequest):
                await self._agent._execute_prompt(
                    request.message, run_id=request.run_id
                )
            elif isinstance(request, RespondExecutionRequest):
                await self._agent._execute_respond(
                    question_id=request.question_id,
                    answer=request.answer,
                )
            elif isinstance(request, ContinueExecutionRequest):
                await self._agent._execute_continue(run_id=request.run_id)
            else:  # pragma: no cover - closed typed union
                raise TypeError(f"unsupported execution request: {type(request)!r}")
        except asyncio.CancelledError as exc:
            caught = exc
        except BaseException as exc:
            caught = exc

        try:
            result = await self._settle(request, caught)
            final_errors = await self._publish(
                ExecutionFinished(result=result), terminal=True
            )
            if final_errors:
                result = replace(
                    result,
                    delivery_errors=tuple([*result.delivery_errors, *final_errors]),
                )
            return result
        finally:
            self._active_attempt_id = None
            self._active_run_id = None

    async def _publish_agent_event(self, event: SessionEvent) -> None:
        if self._active_attempt_id is None:
            return
        await self._publish(event, terminal=False)
        if not isinstance(event, MessageEndEvent):
            return
        message = event.message
        input_id = message.metadata.get("input_id")
        if not isinstance(input_id, str):
            return
        if self._input_status.get(input_id) != "queued":
            return
        self._input_status[input_id] = "committed"
        durability: InputDurability = (
            "checkpoint" if self._has_durable_checkpoint() else "memory"
        )
        await self._publish(
            InputCommitted(input_id=input_id, durability=durability),
            terminal=False,
        )

    async def _publish(
        self,
        event: SessionEvent,
        *,
        terminal: bool,
    ) -> list[DeliveryError]:
        self._seq += 1
        envelope = ExecutionEventEnvelope(
            run_id=self._active_run_id or "",
            attempt_id=self._active_attempt_id or "",
            seq=self._seq,
            event=event,
        )
        errors: list[DeliveryError] = []
        required_failure: DeliveryError | None = None
        for consumer in tuple(self._consumers):
            if consumer.closed:
                continue
            try:
                if consumer.task is None:
                    consumer.task = asyncio.create_task(self._consume(consumer))
                acknowledged = asyncio.get_running_loop().create_future()
                item = _DeliveryItem(envelope=envelope, acknowledged=acknowledged)
                await asyncio.wait_for(
                    consumer.queue.put(item),
                    timeout=consumer.delivery_timeout,
                )
                await asyncio.wait_for(
                    asyncio.shield(acknowledged),
                    timeout=consumer.delivery_timeout,
                )
            except TimeoutError:
                diagnostic = DeliveryError(
                    consumer=consumer.name,
                    seq=envelope.seq,
                    reason="timeout",
                    message="consumer delivery timed out",
                )
                errors.append(diagnostic)
                self._close_consumer(consumer)
                if consumer.required and not terminal and required_failure is None:
                    required_failure = diagnostic
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
                    required_failure = diagnostic
        self._delivery_errors.extend(errors)
        if required_failure is not None:
            raise _RequiredConsumerError(
                f"required consumer {required_failure.consumer} failed: "
                f"{required_failure.message}"
            )
        return errors

    async def _consume(self, consumer: _Consumer) -> None:
        while True:
            item = await consumer.queue.get()
            try:
                value = consumer.listener(item.envelope)
                if inspect.isawaitable(value):
                    await value
            except asyncio.CancelledError:
                if not item.acknowledged.done():
                    item.acknowledged.cancel()
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
        history_consistent = True
        run_messages = [
            message
            for message in self._agent.state.messages
            if message.run_id == request.run_id
        ]
        try:
            check_tool_cycle(run_messages)
        except ToolCycleViolation:
            history_consistent = False

        pending = None
        if self._agent.checkpointer is not None and self._agent.thread_id is not None:
            load_pending = getattr(self._agent.checkpointer, "load_pending", None)
            if load_pending is not None:
                loaded = await load_pending(self._agent.thread_id)
                pending = loaded[0] if loaded is not None else None

        private_outcome = self._agent.state.last_outcome
        error: ExecutionError | None = None
        if isinstance(caught, asyncio.CancelledError) or (
            private_outcome == "abandoned" and self._cancel_requested
        ):
            outcome: ExecutionOutcome = "cancelled"
            error = ExecutionError(
                kind="cancelled",
                message="execution cancelled",
                cause=caught,
            )
        elif caught is not None:
            outcome = "failed"
            error = ExecutionError(
                kind="finalization",
                message=str(caught),
                cause=caught,
            )
        elif not history_consistent or private_outcome == "incomplete":
            outcome = "incomplete"
            error = ExecutionError(
                kind="inconsistent",
                message="execution history has incomplete tool-call results",
            )
        elif private_outcome == "suspended":
            outcome = "suspended"
        elif private_outcome == "complete" and not self._agent.state.error_message:
            outcome = "completed"
        else:
            outcome = "failed"
            message = (
                self._agent.state.error_message or "execution ended without a cause"
            )
            error = ExecutionError(kind="execution", message=message)

        checkpoint_committed = outcome in {"completed", "suspended"}
        return ExecutionResult(
            run_id=request.run_id,
            attempt_id=request.attempt_id,
            outcome=outcome,
            pending_request=pending,
            error=error,
            checkpoint_committed=checkpoint_committed,
            history_consistent=history_consistent,
            delivery_errors=tuple(self._delivery_errors),
        )
