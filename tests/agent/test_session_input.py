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
