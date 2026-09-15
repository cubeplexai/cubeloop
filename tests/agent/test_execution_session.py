from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from cubeloop import Agent
from cubeloop.agent.types import (
    AfterToolCallResult,
    AgentTool,
    AgentToolResult,
)
from cubeloop.checkpointer.base import CheckpointData
from cubeloop.checkpointer.memory import MemoryCheckpointer
from cubeloop.hitl.ask_user import ask_user_tool
from cubeloop.hitl.channel import CheckpointedChannel, InMemoryChannel
from cubeloop.hitl.types import ConfirmRequest, HitlRequest
from cubeloop.providers.base import AssistantMessage, TextContent, ToolCall, UserMessage
from cubeloop.providers.faux import FauxProvider
from cubeloop.session import (
    ExecutionBusy,
    InputEnvelope,
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
async def test_prompt_snapshots_mutable_payload_before_checkpoint_await() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class SlowClaimCheckpointer(MemoryCheckpointer):
        async def claim_run(self, thread_id: str, run_id: str) -> None:
            entered.set()
            await release.wait()
            await super().claim_run(thread_id, run_id)

    checkpointer = SlowClaimCheckpointer()
    agent = Agent(
        model=_provider(_answer()).model("faux-model"),
        checkpointer=checkpointer,
        thread_id="thread-1",
    )
    message = UserMessage(
        content=[TextContent(text="original")],
        metadata={"nested": {"value": 1}},
        run_id="run-1",
    )
    payload = [message]
    task = asyncio.create_task(
        agent.session.execute(
            PromptExecutionRequest(
                run_id="run-1", attempt_id="attempt-1", message=payload
            )
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)

    message.content[0].text = "mutated"
    message.metadata["nested"]["value"] = 2
    message.run_id = "other-run"
    payload.append(UserMessage(content=[TextContent(text="late")]))
    release.set()
    result = await task

    assert result.outcome == "completed"
    user_messages = [item for item in agent.state.messages if item.role == "user"]
    assert len(user_messages) == 1
    assert user_messages[0].content[0].text == "original"
    assert user_messages[0].metadata["nested"]["value"] == 1
    assert user_messages[0].run_id == "run-1"


@pytest.mark.asyncio
async def test_respond_snapshots_mutable_answer_before_checkpoint_await() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class SlowPendingCheckpointer(MemoryCheckpointer):
        block_pending = False

        async def load_pending(self, thread_id: str):
            if self.block_pending:
                entered.set()
                await release.wait()
            return await super().load_pending(thread_id)

    checkpointer = SlowPendingCheckpointer()
    channel = CheckpointedChannel(
        checkpointer=checkpointer,
        thread_id="thread-1",
        run_id="run-1",
    )
    observed: list[str] = []

    async def final_response(messages, model):
        del model
        observed.append(messages[-1].content[0].text)
        return _answer()

    provider = _provider()
    provider.set_responses(
        [
            AssistantMessage(
                content=[
                    ToolCall(
                        id="ask-1",
                        name="ask_user",
                        arguments={
                            "questions": [{"key": "answer", "prompt": "Continue?"}]
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
            final_response,
        ]
    )
    agent = Agent(
        model=provider.model("faux-model"),
        tools=[ask_user_tool(channel)],
        channel=channel,
        checkpointer=checkpointer,
        thread_id="thread-1",
    )
    first = asyncio.create_task(agent.prompt("hi", run_id="run-1"))
    for _ in range(100):
        if channel.pending is not None:
            break
        await asyncio.sleep(0.01)
    assert channel.pending is not None
    question_id = channel.pending.question_id
    await agent.session.request_detach()
    _ = await first
    checkpointer.block_pending = True
    answer = {"answer": "original"}
    resumed = asyncio.create_task(agent.respond(question_id=question_id, answer=answer))
    await asyncio.wait_for(entered.wait(), timeout=1)

    answer["answer"] = "mutated"
    release.set()
    _ = await resumed

    assert len(observed) == 1
    assert "original" in observed[0]
    assert "mutated" not in observed[0]


@pytest.mark.asyncio
async def test_hitl_resume_uses_tool_binding_captured_before_detach() -> None:
    checkpointer = MemoryCheckpointer()
    channel = CheckpointedChannel(
        checkpointer=checkpointer,
        thread_id="thread-1",
        run_id="run-1",
    )
    tool = ask_user_tool(channel)
    provider = _provider(
        AssistantMessage(
            content=[
                ToolCall(
                    id="ask-1",
                    name="ask_user",
                    arguments={"questions": [{"key": "answer", "prompt": "Continue?"}]},
                )
            ],
            stop_reason="tool_use",
        ),
        _answer(),
    )
    replacement_calls = 0

    async def replacement_execute(tool_call_id, args, *, signal=None, on_update=None):
        nonlocal replacement_calls
        del tool_call_id, args, signal, on_update
        replacement_calls += 1
        return AgentToolResult(content=[TextContent(text="replacement")])

    agent = Agent(
        model=provider.model("faux-model"),
        tools=[tool],
        channel=channel,
        checkpointer=checkpointer,
        thread_id="thread-1",
    )
    first = asyncio.create_task(
        agent.session.execute(
            PromptExecutionRequest(run_id="run-1", attempt_id="attempt-1", message="hi")
        )
    )
    for _ in range(100):
        if channel.pending is not None:
            break
        await asyncio.sleep(0.01)
    assert channel.pending is not None
    question_id = channel.pending.question_id
    await agent.session.request_detach()
    assert (await first).outcome == "suspended"
    tool.execute = replacement_execute

    stale = await agent.session.execute(
        RespondExecutionRequest(
            run_id="run-1",
            attempt_id="attempt-stale",
            question_id="stale-question",
            answer={"answer": "no"},
        )
    )
    assert stale.outcome == "failed"

    resumed = await agent.session.execute(
        RespondExecutionRequest(
            run_id="run-1",
            attempt_id="attempt-2",
            question_id=question_id,
            answer={"answer": "yes"},
        )
    )

    assert resumed.outcome == "completed"
    assert replacement_calls == 0


@pytest.mark.asyncio
async def test_terminating_hitl_resume_drains_accepted_follow_up() -> None:
    checkpointer = MemoryCheckpointer()
    channel = CheckpointedChannel(
        checkpointer=checkpointer,
        thread_id="thread-1",
        run_id="run-1",
    )
    resume_entered = asyncio.Event()
    release_resume = asyncio.Event()
    stop_entered = asyncio.Event()
    release_stop = asyncio.Event()

    async def terminate_after_resume(context, signal=None):
        del context, signal
        resume_entered.set()
        await release_resume.wait()
        return AfterToolCallResult(terminate=True)

    async def slow_stop(context):
        del context
        stop_entered.set()
        await release_stop.wait()
        return True

    provider = _provider(
        AssistantMessage(
            content=[
                ToolCall(
                    id="ask-1",
                    name="ask_user",
                    arguments={"questions": [{"key": "answer", "prompt": "Continue?"}]},
                )
            ],
            stop_reason="tool_use",
        ),
        _answer("follow-up handled"),
    )
    agent = Agent(
        model=provider.model("faux-model"),
        tools=[ask_user_tool(channel)],
        channel=channel,
        checkpointer=checkpointer,
        thread_id="thread-1",
        after_tool_call=terminate_after_resume,
        should_stop_after_turn=slow_stop,
    )
    first = asyncio.create_task(
        agent.session.execute(
            PromptExecutionRequest(run_id="run-1", attempt_id="attempt-1", message="hi")
        )
    )
    for _ in range(100):
        if channel.pending is not None:
            break
        await asyncio.sleep(0.01)
    assert channel.pending is not None
    question_id = channel.pending.question_id
    await agent.session.request_detach()
    assert (await first).outcome == "suspended"
    resumed = asyncio.create_task(
        agent.session.execute(
            RespondExecutionRequest(
                run_id="run-1",
                attempt_id="attempt-2",
                question_id=question_id,
                answer={"answer": "yes"},
            )
        )
    )
    await asyncio.wait_for(resume_entered.wait(), timeout=1)
    envelope = InputEnvelope(
        input_id="follow-up-1",
        message=UserMessage(content=[TextContent(text="one more thing")]),
        mode="follow_up",
    )
    assert agent.session.submit_input(envelope).status == "queued"
    release_resume.set()
    await asyncio.wait_for(stop_entered.wait(), timeout=1)
    late = agent.session.submit_input(
        InputEnvelope(
            input_id="too-late",
            message=UserMessage(content=[TextContent(text="missed drain")]),
            mode="follow_up",
        )
    )
    assert late.status == "closed"
    release_stop.set()
    result = await resumed

    assert result.outcome == "completed"
    receipt = agent.session.cancel_input("follow-up-1")
    assert receipt.status == "committed"
    assert receipt.durability == "checkpoint"
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_checkpoint_message_write_failure_marks_history_inconsistent() -> None:
    class Args(BaseModel):
        value: str

    class FailingToolResultCheckpointer(MemoryCheckpointer):
        async def append(self, thread_id, messages):
            if any(message.role == "tool_result" for message in messages):
                raise RuntimeError("tool result write failed")
            await super().append(thread_id, messages)

    async def execute(tool_call_id, args, *, signal=None, on_update=None):
        del tool_call_id, args, signal, on_update
        return AgentToolResult(content=[TextContent(text="done")])

    tool = AgentTool(name="work", description="work", parameters=Args, execute=execute)
    provider = _provider(
        AssistantMessage(
            content=[ToolCall(id="call-1", name="work", arguments={"value": "x"})],
            stop_reason="tool_use",
        ),
        _answer("next attempt"),
    )
    agent = Agent(
        model=provider.model("faux-model"),
        tools=[tool],
        checkpointer=FailingToolResultCheckpointer(),
        thread_id="thread-1",
    )

    result = await agent.session.execute(
        PromptExecutionRequest(run_id="run-1", attempt_id="attempt-1", message="hi")
    )

    assert result.outcome == "failed"
    assert result.history_consistent is False

    retry = await agent.session.execute(
        PromptExecutionRequest(run_id="run-2", attempt_id="attempt-2", message="retry")
    )
    assert retry.outcome == "failed"
    assert retry.history_consistent is False
    assert retry.checkpoint_committed is False


@pytest.mark.asyncio
async def test_degraded_checkpointer_does_not_report_completed_commitment() -> None:
    class DegradedCheckpointer:
        async def load(self, thread_id):
            return None

        async def append(self, thread_id, messages):
            return None

        async def save_extra(self, thread_id, extra):
            return None

    agent = Agent(
        model=_provider(_answer()).model("faux-model"),
        checkpointer=DegradedCheckpointer(),
        thread_id="thread-1",
    )

    result = await agent.session.execute(
        PromptExecutionRequest(run_id="run-1", attempt_id="attempt-1", message="hi")
    )

    assert result.outcome == "completed"
    assert result.checkpoint_committed is False


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
async def test_agent_prompt_checks_admission_before_hitl_binding() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_response(messages, model):
        del messages, model
        entered.set()
        await release.wait()
        return _answer()

    provider = _provider()
    provider.set_responses([slow_response])
    checkpointer = MemoryCheckpointer()
    channel = CheckpointedChannel(
        checkpointer=checkpointer,
        thread_id="thread-1",
        run_id="run-1",
    )
    agent = Agent(
        model=provider.model("faux-model"),
        tools=[ask_user_tool(channel)],
        channel=channel,
        checkpointer=checkpointer,
        thread_id="thread-1",
    )
    active = asyncio.create_task(agent.prompt("first", run_id="run-1"))
    await asyncio.wait_for(entered.wait(), timeout=1)

    with pytest.raises(ExecutionBusy):
        await agent.prompt("second")

    release.set()
    assert await active == "run-1"


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
async def test_wait_for_idle_includes_terminal_session_delivery() -> None:
    terminal_entered = asyncio.Event()
    release_terminal = asyncio.Event()

    async def listener(envelope):
        if envelope.event.type == "execution_finished":
            terminal_entered.set()
            await release_terminal.wait()

    agent = Agent(model=_provider(_answer()).model("faux-model"))
    agent.session.subscribe(listener)
    execution = asyncio.create_task(agent.prompt("hi", run_id="run-1"))
    await asyncio.wait_for(terminal_entered.wait(), timeout=1)
    idle = asyncio.create_task(agent.wait_for_idle())
    await asyncio.sleep(0)

    assert not idle.done()
    release_terminal.set()
    assert await execution == "run-1"
    await idle


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
    agent.session._input_status["memory-input"] = "committed"
    agent.session._input_durability["memory-input"] = "memory"

    loaded = await agent.session.load_checkpoint()

    assert loaded is not None
    assert loaded.messages == []
    assert live_context is agent.session.state_context
    assert live_context == {"todo": ["ship"]}
    assert "memory-input" not in agent.session._input_status
    assert "memory-input" not in agent.session._input_durability
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


@pytest.mark.asyncio
async def test_in_memory_detach_returns_pending_request() -> None:
    channel = InMemoryChannel(thread_id="thread-1")
    checkpointer = MemoryCheckpointer()
    stale = HitlRequest(
        question_id="stale-question",
        thread_id="thread-1",
        payload=ConfirmRequest(prompt="stale"),
        created_at=0,
    )
    await checkpointer.save_pending_request("thread-1", stale, run_id="old-run")
    provider = _provider(
        AssistantMessage(
            content=[
                ToolCall(
                    id="ask-1",
                    name="ask_user",
                    arguments={"questions": [{"key": "answer", "prompt": "Continue?"}]},
                )
            ],
            stop_reason="tool_use",
        )
    )
    agent = Agent(
        model=provider.model("faux-model"),
        tools=[ask_user_tool(channel)],
        channel=channel,
        checkpointer=checkpointer,
        thread_id="thread-1",
    )
    execution = asyncio.create_task(
        agent.session.execute(
            PromptExecutionRequest(run_id="run-1", attempt_id="attempt-1", message="hi")
        )
    )
    for _ in range(100):
        if channel.pending is not None:
            break
        await asyncio.sleep(0.01)
    assert channel.pending is not None
    question_id = channel.pending.question_id

    await agent.session.request_detach()
    result = await execution

    assert result.outcome == "suspended"
    assert result.checkpoint_committed is False
    assert result.pending_request is not None
    assert result.pending_request.question_id == question_id


@pytest.mark.asyncio
async def test_loaded_pending_request_is_copied_before_result_exposure() -> None:
    checkpointer = MemoryCheckpointer()
    pending = HitlRequest(
        question_id="question-1",
        thread_id="thread-1",
        payload=ConfirmRequest(prompt="original"),
        created_at=0,
    )
    await checkpointer.save_pending_request("thread-1", pending, run_id="run-1")
    agent = Agent(
        model=_provider(_answer()).model("faux-model"),
        checkpointer=checkpointer,
        thread_id="thread-1",
    )
    agent._state.last_outcome = "suspended"

    result = await agent.session._settle(
        PromptExecutionRequest(run_id="run-1", attempt_id="attempt-1", message="hi"),
        None,
    )
    assert result.pending_request is not None
    result.pending_request.payload.prompt = "mutated"

    loaded = await checkpointer.load_pending("thread-1")
    assert loaded is not None
    assert loaded[0].payload.prompt == "original"


@pytest.mark.asyncio
async def test_completed_attempt_does_not_expose_pending_from_another_run() -> None:
    checkpointer = MemoryCheckpointer()
    stale = HitlRequest(
        question_id="stale-question",
        thread_id="thread-1",
        payload=ConfirmRequest(prompt="stale"),
        created_at=0,
    )
    await checkpointer.save_pending_request("thread-1", stale, run_id="old-run")
    agent = Agent(
        model=_provider(_answer()).model("faux-model"),
        checkpointer=checkpointer,
        thread_id="thread-1",
    )

    result = await agent.session.execute(
        PromptExecutionRequest(run_id="run-1", attempt_id="attempt-1", message="hi")
    )

    assert result.outcome == "completed"
    assert result.pending_request is None
    assert result.checkpoint_committed is True


@pytest.mark.asyncio
async def test_cancel_requested_during_startup_reaches_new_run_signal() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class SlowClaimCheckpointer(MemoryCheckpointer):
        async def claim_run(self, thread_id: str, run_id: str) -> None:
            entered.set()
            await release.wait()
            await super().claim_run(thread_id, run_id)

    checkpointer = SlowClaimCheckpointer()
    agent = Agent(
        model=_provider(_answer()).model("faux-model"),
        checkpointer=checkpointer,
        thread_id="thread-1",
    )
    task = asyncio.create_task(
        agent.session.execute(
            PromptExecutionRequest(run_id="run-1", attempt_id="attempt-1", message="hi")
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)

    agent.session.request_cancel()
    release.set()
    result = await task

    assert result.outcome == "cancelled"


@pytest.mark.asyncio
async def test_committed_completion_takes_precedence_over_cancel_request() -> None:
    agent = Agent(
        model=_provider(_answer()).model("faux-model"),
        checkpointer=MemoryCheckpointer(),
        thread_id="thread-1",
    )
    agent._state.last_outcome = "complete"
    agent.session._cancel_requested = True

    result = await agent.session._settle(
        PromptExecutionRequest(run_id="run-1", attempt_id="attempt-1", message="hi"),
        None,
    )

    assert result.outcome == "completed"
    assert result.checkpoint_committed is True


@pytest.mark.asyncio
async def test_process_control_exception_releases_session_ownership() -> None:
    class ProcessExit(BaseException):
        pass

    async def raises_process_exit(context, signal=None):
        del context, signal
        raise ProcessExit("stop host")

    provider = _provider(_answer())
    agent = Agent(model=provider.model("faux-model"), on_run_end=raises_process_exit)

    with pytest.raises(ProcessExit, match="stop host"):
        await agent.session.execute(
            PromptExecutionRequest(run_id="run-1", attempt_id="attempt-1", message="hi")
        )

    assert agent.session.active_attempt_id is None
    await asyncio.wait_for(agent.wait_for_idle(), timeout=1)


@pytest.mark.asyncio
async def test_reconciled_input_process_control_releases_session_ownership() -> None:
    class ProcessExit(BaseException):
        pass

    entered = asyncio.Event()
    release = asyncio.Event()
    agent = Agent(model=_provider().model("faux-model"))

    async def consume_without_publishing(message, *, run_id=None):
        del message, run_id
        entered.set()
        await release.wait()
        accepted = agent._follow_up_queue.drain()
        agent._state._messages.extend(accepted)
        agent._state.last_outcome = "complete"
        return "run-1"

    agent._execute_prompt = consume_without_publishing  # type: ignore[method-assign]

    def stop_host(envelope):
        if envelope.event.type == "input_committed":
            raise ProcessExit("stop host")

    agent.session.subscribe(stop_host)
    task = asyncio.create_task(
        agent.session.execute(
            PromptExecutionRequest(run_id="run-1", attempt_id="attempt-1", message="hi")
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    assert (
        agent.session.submit_input(
            InputEnvelope(
                input_id="input-1",
                message=UserMessage(content=[TextContent(text="accepted")]),
                mode="follow_up",
            )
        ).status
        == "queued"
    )
    release.set()

    with pytest.raises(ProcessExit, match="stop host"):
        _ = await task
    assert agent.session.active_attempt_id is None
    await asyncio.wait_for(agent.wait_for_idle(), timeout=1)
