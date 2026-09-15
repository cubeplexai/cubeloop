from __future__ import annotations

import asyncio

import pytest

from cubeloop import Agent
from cubeloop.checkpointer.memory import MemoryCheckpointer
from cubeloop.providers.base import AssistantMessage, TextContent, UserMessage
from cubeloop.providers.faux import FauxProvider
from cubeloop.session import InputEnvelope, PromptExecutionRequest


@pytest.mark.asyncio
async def test_input_committed_only_after_checkpoint_append() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def first_response(messages, model):
        del messages, model
        entered.set()
        await release.wait()
        return AssistantMessage(
            content=[TextContent(text="first")], stop_reason="end_turn"
        )

    provider = FauxProvider(provider_id="faux")
    provider.set_responses(
        [
            first_response,
            AssistantMessage(
                content=[TextContent(text="after steer")], stop_reason="end_turn"
            ),
        ]
    )
    checkpointer = MemoryCheckpointer()
    agent = Agent(
        model=provider.model("faux-model"),
        checkpointer=checkpointer,
        thread_id="thread-1",
    )
    events = []
    agent.session.subscribe(lambda event: events.append(event), name="test-observer")
    task = asyncio.create_task(
        agent.session.execute(
            PromptExecutionRequest(
                run_id="run-1", attempt_id="attempt-1", message="start"
            )
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)

    receipt = agent.session.submit_input(
        InputEnvelope(
            input_id="steer-1",
            message=UserMessage(content=[TextContent(text="change course")]),
            mode="steer",
        )
    )
    assert receipt.status == "queued"
    assert not [event for event in events if event.event.type == "input_committed"]

    release.set()
    result = await task
    assert result.outcome == "completed"
    committed = [event for event in events if event.event.type == "input_committed"]
    assert len(committed) == 1
    assert committed[0].event.input_id == "steer-1"
    assert committed[0].event.durability == "checkpoint"
    persisted = await checkpointer.load("thread-1")
    assert persisted is not None
    assert any(
        message.metadata.get("input_id") == "steer-1" for message in persisted.messages
    )


@pytest.mark.asyncio
async def test_duplicate_and_cancelled_queued_input_is_not_injected() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def response(messages, model):
        del messages, model
        entered.set()
        await release.wait()
        return AssistantMessage(
            content=[TextContent(text="done")], stop_reason="end_turn"
        )

    provider = FauxProvider(provider_id="faux")
    provider.set_responses([response])
    agent = Agent(model=provider.model("faux-model"))
    task = asyncio.create_task(
        agent.session.execute(
            PromptExecutionRequest(
                run_id="run-1", attempt_id="attempt-1", message="start"
            )
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    envelope = InputEnvelope(
        input_id="steer-1",
        message=UserMessage(content=[TextContent(text="cancel me")]),
        mode="steer",
    )

    assert agent.session.submit_input(envelope).status == "queued"
    assert agent.session.submit_input(envelope).status == "queued"
    assert agent.session.cancel_input("steer-1").status == "cancelled"

    release.set()
    await task
    assert not any(
        message.metadata.get("input_id") == "steer-1"
        for message in agent.state.messages
    )
    assert agent.session.cancel_input("steer-1").status == "cancelled"


def test_input_to_idle_session_is_closed() -> None:
    provider = FauxProvider(provider_id="faux")
    agent = Agent(model=provider.model("faux-model"))

    receipt = agent.session.submit_input(
        InputEnvelope(
            input_id="late",
            message=UserMessage(content=[TextContent(text="too late")]),
            mode="follow_up",
        )
    )

    assert receipt.status == "closed"


@pytest.mark.asyncio
async def test_follow_up_receipt_becomes_committed_in_memory() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def first_response(messages, model):
        del messages, model
        entered.set()
        await release.wait()
        return AssistantMessage(
            content=[TextContent(text="first")], stop_reason="end_turn"
        )

    provider = FauxProvider(provider_id="faux")
    provider.set_responses(
        [
            first_response,
            AssistantMessage(
                content=[TextContent(text="done")], stop_reason="end_turn"
            ),
        ]
    )
    agent = Agent(model=provider.model("faux-model"))
    task = asyncio.create_task(
        agent.session.execute(
            PromptExecutionRequest(
                run_id="run-1", attempt_id="attempt-1", message="start"
            )
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)

    receipt = agent.session.submit_input(
        InputEnvelope(
            input_id="follow-up-1",
            message=UserMessage(content=[TextContent(text="one more thing")]),
            mode="follow_up",
        )
    )
    assert receipt.status == "queued"
    assert agent.session.cancel_input("unknown").status == "closed"

    release.set()
    await task
    committed = agent.session.cancel_input("follow-up-1")
    assert committed.status == "committed"
    assert committed.durability == "memory"
    duplicate = agent.session.submit_input(
        InputEnvelope(
            input_id="follow-up-1",
            message=UserMessage(content=[TextContent(text="duplicate")]),
            mode="follow_up",
        )
    )
    assert duplicate.status == "committed"
    assert duplicate.durability == "memory"


@pytest.mark.asyncio
async def test_session_input_id_overrides_caller_steering_key() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def response(messages, model):
        del messages, model
        entered.set()
        await release.wait()
        return AssistantMessage(
            content=[TextContent(text="done")], stop_reason="end_turn"
        )

    provider = FauxProvider(provider_id="faux")
    provider.set_responses([response])
    agent = Agent(model=provider.model("faux-model"))
    task = asyncio.create_task(
        agent.session.execute(
            PromptExecutionRequest(
                run_id="run-1", attempt_id="attempt-1", message="start"
            )
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)

    receipt = agent.session.submit_input(
        InputEnvelope(
            input_id="owned-id",
            message=UserMessage(
                content=[TextContent(text="cancel")],
                metadata={"steer_id": "caller-id"},
            ),
            mode="steer",
        )
    )
    assert receipt.status == "queued"
    assert agent.session.cancel_input("owned-id").status == "cancelled"

    release.set()
    await task
    assert not any(
        message.metadata.get("input_id") == "owned-id"
        for message in agent.state.messages
    )


@pytest.mark.asyncio
async def test_input_admission_closes_before_final_on_run_end_hook() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def on_run_end(context, signal=None):
        del context, signal
        entered.set()
        await release.wait()

    provider = FauxProvider(provider_id="faux")
    provider.set_responses(
        [AssistantMessage(content=[TextContent(text="done")], stop_reason="end_turn")]
    )
    agent = Agent(model=provider.model("faux-model"), on_run_end=on_run_end)
    task = asyncio.create_task(agent.prompt("hi", run_id="run-1"))
    await asyncio.wait_for(entered.wait(), timeout=1)

    receipt = agent.session.submit_input(
        InputEnvelope(
            input_id="too-late",
            message=UserMessage(content=[TextContent(text="late")]),
            mode="follow_up",
        )
    )

    assert receipt.status == "closed"
    release.set()
    assert await task == "run-1"


@pytest.mark.asyncio
async def test_committed_input_id_remains_deduplicated_across_attempts() -> None:
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()
    release_second = asyncio.Event()

    async def first_response(messages, model):
        del messages, model
        first_entered.set()
        await release_first.wait()
        return AssistantMessage(
            content=[TextContent(text="first")], stop_reason="end_turn"
        )

    async def second_attempt(messages, model):
        del messages, model
        second_entered.set()
        await release_second.wait()
        return AssistantMessage(
            content=[TextContent(text="second")], stop_reason="end_turn"
        )

    provider = FauxProvider(provider_id="faux")
    provider.set_responses(
        [
            first_response,
            AssistantMessage(
                content=[TextContent(text="after input")], stop_reason="end_turn"
            ),
            second_attempt,
        ]
    )
    agent = Agent(model=provider.model("faux-model"))
    first = asyncio.create_task(agent.prompt("first", run_id="run-1"))
    await asyncio.wait_for(first_entered.wait(), timeout=1)
    envelope = InputEnvelope(
        input_id="stable-id",
        message=UserMessage(content=[TextContent(text="once")]),
        mode="follow_up",
    )
    assert agent.session.submit_input(envelope).status == "queued"
    release_first.set()
    assert await first == "run-1"

    second = asyncio.create_task(agent.prompt("second", run_id="run-2"))
    await asyncio.wait_for(second_entered.wait(), timeout=1)
    duplicate = agent.session.submit_input(envelope)

    assert duplicate.status == "committed"
    assert duplicate.durability == "memory"
    release_second.set()
    assert await second == "run-2"
    assert (
        sum(
            message.metadata.get("input_id") == "stable-id"
            for message in agent.state.messages
        )
        == 1
    )


@pytest.mark.asyncio
async def test_input_with_mismatched_run_id_is_rejected_before_queueing() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def response(messages, model):
        del messages, model
        entered.set()
        await release.wait()
        return AssistantMessage(
            content=[TextContent(text="done")], stop_reason="end_turn"
        )

    provider = FauxProvider(provider_id="faux")
    provider.set_responses([response])
    agent = Agent(model=provider.model("faux-model"))
    task = asyncio.create_task(agent.prompt("hi", run_id="run-1"))
    await asyncio.wait_for(entered.wait(), timeout=1)

    receipt = agent.session.submit_input(
        InputEnvelope(
            input_id="wrong-run",
            message=UserMessage(
                content=[TextContent(text="wrong")], run_id="run-other"
            ),
            mode="steer",
        )
    )

    assert receipt.status == "closed"
    release.set()
    await task
