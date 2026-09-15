from __future__ import annotations

import asyncio

import pytest

from cubeloop import Agent
from cubeloop.checkpointer.base import CheckpointData
from cubeloop.checkpointer.memory import MemoryCheckpointer
from cubeloop.hitl.channel import CheckpointedChannel
from cubeloop.hitl.types import ConfirmRequest, HitlRequest
from cubeloop.providers.base import AssistantMessage, TextContent, UserMessage
from cubeloop.providers.faux import FauxProvider
from cubeloop.session import (
    ExecutionBusy,
    PromptExecutionRequest,
    RespondExecutionRequest,
)


def _provider(*responses: AssistantMessage) -> FauxProvider:
    provider = FauxProvider(provider_id="faux")
    provider.set_responses(list(responses))
    return provider


def _answer(text: str = "done") -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        stop_reason="end_turn",
    )


@pytest.mark.asyncio
async def test_execute_returns_explicit_completed_result() -> None:
    agent = Agent(model=_provider(_answer()).model("faux-model"))

    result = await agent.session.execute(
        PromptExecutionRequest(run_id="run-1", attempt_id="attempt-1", message="hi")
    )

    assert result.run_id == "run-1"
    assert result.attempt_id == "attempt-1"
    assert result.outcome == "completed"
    assert result.error is None
    assert result.history_consistent is True
    assert result.checkpoint_committed is False


@pytest.mark.asyncio
async def test_completed_checkpointed_attempt_reports_committed() -> None:
    agent = Agent(
        model=_provider(_answer()).model("faux-model"),
        checkpointer=MemoryCheckpointer(),
        thread_id="thread-1",
    )

    result = await agent.session.execute(
        PromptExecutionRequest(run_id="run-1", attempt_id="attempt-1", message="hi")
    )

    assert result.outcome == "completed"
    assert result.checkpoint_committed is True


@pytest.mark.asyncio
async def test_settlement_failure_returns_result_and_terminal_event() -> None:
    class BrokenPendingCheckpointer(MemoryCheckpointer):
        async def load_pending(self, thread_id: str):
            del thread_id
            raise RuntimeError("pending storage unavailable")

    agent = Agent(
        model=_provider(_answer()).model("faux-model"),
        checkpointer=BrokenPendingCheckpointer(),
        thread_id="thread-1",
    )
    events = []
    agent.session.subscribe(lambda event: events.append(event))

    result = await agent.session.execute(
        PromptExecutionRequest(run_id="run-1", attempt_id="attempt-1", message="hi")
    )

    assert result.outcome == "failed"
    assert result.error is not None
    assert result.error.kind == "finalization"
    assert "pending storage unavailable" in result.error.message
    assert result.checkpoint_committed is False
    finished = [event for event in events if event.event.type == "execution_finished"]
    assert len(finished) == 1
    assert finished[0].event.result == result


@pytest.mark.asyncio
async def test_cancellation_during_settlement_still_publishes_terminal_event() -> None:
    entered = asyncio.Event()

    class SlowPendingCheckpointer(MemoryCheckpointer):
        async def load_pending(self, thread_id: str):
            del thread_id
            entered.set()
            await asyncio.Future()

    agent = Agent(
        model=_provider(_answer()).model("faux-model"),
        checkpointer=SlowPendingCheckpointer(),
        thread_id="thread-1",
    )
    events = []
    agent.session.subscribe(lambda event: events.append(event))
    task = asyncio.create_task(
        agent.session.execute(
            PromptExecutionRequest(run_id="run-1", attempt_id="attempt-1", message="hi")
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)

    task.cancel()
    result = await task

    assert result.outcome == "cancelled"
    finished = [event for event in events if event.event.type == "execution_finished"]
    assert len(finished) == 1
    assert finished[0].event.result == result


@pytest.mark.asyncio
async def test_cancellation_during_terminal_delivery_does_not_drop_terminal() -> None:
    terminal_entered = asyncio.Event()
    release_terminal = asyncio.Event()
    events = []

    async def listener(envelope):
        if envelope.event.type == "execution_finished":
            terminal_entered.set()
            await release_terminal.wait()
        events.append(envelope)

    agent = Agent(model=_provider(_answer()).model("faux-model"))
    agent.session.subscribe(listener)
    task = asyncio.create_task(
        agent.session.execute(
            PromptExecutionRequest(run_id="run-1", attempt_id="attempt-1", message="hi")
        )
    )
    await asyncio.wait_for(terminal_entered.wait(), timeout=1)

    task.cancel()
    release_terminal.set()
    result = await task

    assert result.outcome == "completed"
    assert (
        len([event for event in events if event.event.type == "execution_finished"])
        == 1
    )


@pytest.mark.asyncio
async def test_respond_rejects_request_run_mismatch_before_resume() -> None:
    checkpointer = MemoryCheckpointer()
    pending = HitlRequest(
        question_id="question-1",
        thread_id="thread-1",
        payload=ConfirmRequest(prompt="continue?"),
        created_at=0,
    )
    await checkpointer.save_pending_request("thread-1", pending, run_id="run-real")
    channel = CheckpointedChannel(
        checkpointer=checkpointer,
        thread_id="thread-1",
        run_id="run-real",
    )
    agent = Agent(
        model=_provider(_answer()).model("faux-model"),
        checkpointer=checkpointer,
        thread_id="thread-1",
        channel=channel,
    )

    result = await agent.session.execute(
        RespondExecutionRequest(
            run_id="run-stale",
            attempt_id="attempt-1",
            question_id="question-1",
            answer=True,
        )
    )

    assert result.outcome == "failed"
    assert result.error is not None
    assert "does not match" in result.error.message
    assert await checkpointer.load_pending("thread-1") == (pending, "run-real")


@pytest.mark.asyncio
async def test_agent_prompt_and_session_share_one_admission_gate() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_response(messages, model):
        del messages, model
        entered.set()
        await release.wait()
        return _answer()

    provider = _provider()
    provider.set_responses([slow_response])
    agent = Agent(model=provider.model("faux-model"))
    first = asyncio.create_task(agent.prompt("first", run_id="run-1"))
    await asyncio.wait_for(entered.wait(), timeout=1)

    with pytest.raises(ExecutionBusy):
        await agent.session.execute(
            PromptExecutionRequest(
                run_id="run-2",
                attempt_id="attempt-2",
                message="second",
            )
        )

    release.set()
    assert await first == "run-1"
    assert [m.run_id for m in agent.state.messages] == ["run-1", "run-1"]


@pytest.mark.asyncio
async def test_agent_respond_checks_admission_before_loading_pending() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class CountingCheckpointer(MemoryCheckpointer):
        load_pending_calls = 0

        async def load_pending(self, thread_id: str):
            self.load_pending_calls += 1
            return await super().load_pending(thread_id)

    async def slow_response(messages, model):
        del messages, model
        entered.set()
        await release.wait()
        return _answer()

    provider = _provider()
    provider.set_responses([slow_response])
    checkpointer = CountingCheckpointer()
    agent = Agent(
        model=provider.model("faux-model"),
        checkpointer=checkpointer,
        thread_id="thread-1",
    )
    active = asyncio.create_task(agent.prompt("hi", run_id="run-1"))
    await asyncio.wait_for(entered.wait(), timeout=1)
    calls_before = checkpointer.load_pending_calls

    with pytest.raises(ExecutionBusy):
        await agent.respond(answer=True)

    assert checkpointer.load_pending_calls == calls_before
    release.set()
    assert await active == "run-1"


@pytest.mark.asyncio
async def test_load_checkpoint_restores_extra_in_place_even_without_messages() -> None:
    checkpointer = MemoryCheckpointer()
    await checkpointer.save_extra("thread-1", {"todo": ["ship"]})
    agent = Agent(
        model=_provider(_answer()).model("faux-model"),
        checkpointer=checkpointer,
        thread_id="thread-1",
    )
    live_context = agent.session.state_context

    loaded = await agent.session.load_checkpoint()

    assert loaded is not None
    assert loaded.messages == []
    assert live_context is agent.session.state_context
    assert live_context == {"todo": ["ship"]}
    live_context["memory"] = "pinned"
    await agent.prompt("hi", run_id="run-1")
    persisted = await checkpointer.load("thread-1")
    assert persisted is not None
    assert persisted.extra == {"todo": ["ship"], "memory": "pinned"}


@pytest.mark.asyncio
async def test_load_checkpoint_failure_does_not_partially_replace_live_state() -> None:
    class BrokenCheckpointer(MemoryCheckpointer):
        async def load(self, thread_id: str) -> CheckpointData | None:
            del thread_id
            raise RuntimeError("cannot load")

    agent = Agent(
        model=_provider(_answer()).model("faux-model"),
        checkpointer=BrokenCheckpointer(),
        thread_id="thread-1",
    )
    agent._state._messages = [UserMessage(content=[TextContent(text="existing")])]
    agent.session.state_context["existing"] = True

    with pytest.raises(RuntimeError, match="cannot load"):
        await agent.session.load_checkpoint()

    assert agent.session.state_context == {"existing": True}
    assert len(agent.state.messages) == 1


@pytest.mark.asyncio
async def test_load_checkpoint_handles_unconfigured_and_empty_storage() -> None:
    unconfigured = Agent(model=_provider(_answer()).model("faux-model"))
    assert await unconfigured.session.load_checkpoint() is None

    empty = Agent(
        model=_provider(_answer()).model("faux-model"),
        checkpointer=MemoryCheckpointer(),
        thread_id="thread-1",
    )
    assert await empty.session.load_checkpoint() is None


@pytest.mark.asyncio
async def test_load_checkpoint_rejects_while_execution_is_active() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_response(messages, model):
        del messages, model
        entered.set()
        await release.wait()
        return _answer()

    provider = _provider()
    provider.set_responses([slow_response])
    agent = Agent(model=provider.model("faux-model"))
    task = asyncio.create_task(agent.prompt("hi", run_id="run-1"))
    await asyncio.wait_for(entered.wait(), timeout=1)

    with pytest.raises(ExecutionBusy, match="during execution"):
        await agent.session.load_checkpoint()

    release.set()
    await task


@pytest.mark.asyncio
async def test_subscribe_validates_limits_and_unsubscribe_stops_delivery() -> None:
    agent = Agent(model=_provider(_answer()).model("faux-model"))

    with pytest.raises(ValueError, match="capacity"):
        agent.session.subscribe(lambda event: None, capacity=0)
    with pytest.raises(ValueError, match="delivery_timeout"):
        agent.session.subscribe(lambda event: None, delivery_timeout=0)

    events = []
    unsubscribe = agent.session.subscribe(lambda event: events.append(event))
    await agent.prompt("first", run_id="run-1")
    assert events

    unsubscribe()
    provider = _provider(_answer())
    agent.model = provider.model("faux-model")
    await agent.prompt("second", run_id="run-2")
    assert {event.run_id for event in events} == {"run-1"}


@pytest.mark.asyncio
async def test_cancelled_attempt_returns_cancelled_result() -> None:
    entered = asyncio.Event()

    async def never_finishes(messages, model):
        del messages, model
        entered.set()
        await asyncio.Future()

    provider = _provider()
    provider.set_responses([never_finishes])
    agent = Agent(model=provider.model("faux-model"))
    task = asyncio.create_task(
        agent.session.execute(
            PromptExecutionRequest(
                run_id="run-1",
                attempt_id="attempt-1",
                message="hi",
            )
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)

    agent.session.request_cancel()
    task.cancel()
    result = await task

    assert result.outcome == "cancelled"
    assert result.error is not None
    assert result.error.kind == "cancelled"


@pytest.mark.asyncio
async def test_detach_rejects_outside_hitl_safe_point() -> None:
    agent = Agent(model=_provider(_answer()).model("faux-model"))

    with pytest.raises(RuntimeError, match="HITL safe point"):
        await agent.session.request_detach()
