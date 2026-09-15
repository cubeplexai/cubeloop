from __future__ import annotations

import asyncio

import pytest

from cubeloop import Agent
from cubeloop.agent.types import AgentStartEvent
from cubeloop.providers.base import UserMessage
from cubeloop.providers.base import AssistantMessage, TextContent
from cubeloop.providers.faux import FauxProvider
from cubeloop.session import InputEnvelope, PromptExecutionRequest


def _agent() -> Agent:
    provider = FauxProvider(provider_id="faux")
    provider.set_responses(
        [AssistantMessage(content=[TextContent(text="done")], stop_reason="end_turn")]
    )
    return Agent(model=provider.model("faux-model"))


@pytest.mark.asyncio
async def test_attempt_events_have_monotonic_sequence_and_one_finished() -> None:
    agent = _agent()
    events = []
    agent.session.subscribe(lambda event: events.append(event), name="observer")

    result = await agent.session.execute(
        PromptExecutionRequest(run_id="run-1", attempt_id="attempt-1", message="hi")
    )

    assert [event.seq for event in events] == list(range(1, len(events) + 1))
    assert {event.run_id for event in events} == {"run-1"}
    assert {event.attempt_id for event in events} == {"attempt-1"}
    finished = [event for event in events if event.event.type == "execution_finished"]
    assert len(finished) == 1
    assert finished[0].event.result == result


@pytest.mark.asyncio
async def test_required_consumer_failure_fails_execution() -> None:
    agent = _agent()

    async def broken(event):
        if event.event.type == "message_end":
            raise RuntimeError("publisher closed")

    agent.session.subscribe(broken, required=True, name="host-publisher")

    result = await agent.session.execute(
        PromptExecutionRequest(run_id="run-1", attempt_id="attempt-1", message="hi")
    )

    assert result.outcome == "failed"
    assert result.error is not None
    assert result.error.kind == "execution"
    assert "host-publisher" in result.error.message


@pytest.mark.asyncio
async def test_persisted_input_is_committed_before_required_delivery_failure() -> None:
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
    provider.set_responses([first_response])
    agent = Agent(model=provider.model("faux-model"))

    async def fail_on_input(envelope):
        event = envelope.event
        if (
            event.type == "message_end"
            and event.message.metadata.get("input_id") == "input-1"
        ):
            raise RuntimeError("host unavailable")

    agent.session.subscribe(fail_on_input, required=True, name="host")
    task = asyncio.create_task(
        agent.session.execute(
            PromptExecutionRequest(run_id="run-1", attempt_id="attempt-1", message="hi")
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    envelope = InputEnvelope(
        input_id="input-1",
        message=UserMessage(content=[TextContent(text="persist once")]),
        mode="follow_up",
    )
    assert agent.session.submit_input(envelope).status == "queued"
    release.set()

    result = await task

    assert result.outcome == "failed"
    assert agent.session.submit_input(envelope).status == "committed"


@pytest.mark.asyncio
async def test_consumers_receive_isolated_agent_event_snapshots() -> None:
    agent = _agent()
    observed = []

    def mutate_first(envelope):
        event = envelope.event
        if event.type == "message_start" and event.message.role == "user":
            event.message.content[0].text = "mutated"

    def observe_second(envelope):
        event = envelope.event
        if event.type == "message_start" and event.message.role == "user":
            observed.append(event.message.content[0].text)

    agent.session.subscribe(mutate_first, name="mutator")
    agent.session.subscribe(observe_second, name="observer")

    await agent.prompt("original", run_id="run-1")

    assert observed == ["original"]
    assert agent.state.messages[0].content[0].text == "original"


@pytest.mark.asyncio
async def test_final_notification_timeout_preserves_settled_result() -> None:
    agent = _agent()

    async def stalls_on_finished(event):
        if event.event.type == "execution_finished":
            await asyncio.Future()

    agent.session.subscribe(
        stalls_on_finished,
        required=True,
        name="stalled-host",
        delivery_timeout=0.01,
    )

    result = await agent.session.execute(
        PromptExecutionRequest(run_id="run-1", attempt_id="attempt-1", message="hi")
    )

    assert result.outcome == "completed"
    assert len(result.delivery_errors) == 1
    assert result.delivery_errors[0].consumer == "stalled-host"
    assert result.delivery_errors[0].reason == "timeout"


@pytest.mark.asyncio
async def test_input_submitted_from_terminal_listener_is_closed() -> None:
    agent = _agent()
    receipts = []

    def terminal_listener(envelope):
        if envelope.event.type == "execution_finished":
            receipts.append(
                agent.session.submit_input(
                    InputEnvelope(
                        input_id="late",
                        message=UserMessage(content=[TextContent(text="too late")]),
                        mode="follow_up",
                    )
                )
            )

    agent.session.subscribe(terminal_listener)

    result = await agent.session.execute(
        PromptExecutionRequest(run_id="run-1", attempt_id="attempt-1", message="hi")
    )

    assert result.outcome == "completed"
    assert len(receipts) == 1
    assert receipts[0].status == "closed"


@pytest.mark.asyncio
async def test_required_consumer_failure_stops_concurrent_publishers() -> None:
    agent = _agent()

    async def broken(envelope):
        del envelope
        raise RuntimeError("publisher closed")

    agent.session.subscribe(broken, required=True, name="required")
    first = asyncio.create_task(
        agent.session._publish(AgentStartEvent(), terminal=False)
    )
    await asyncio.sleep(0)
    second = asyncio.create_task(
        agent.session._publish(AgentStartEvent(), terminal=False)
    )

    results = await asyncio.gather(first, second, return_exceptions=True)

    assert all(isinstance(result, RuntimeError) for result in results)
    assert all("required consumer" in str(result) for result in results)


@pytest.mark.asyncio
async def test_unsubscribing_running_observer_records_delivery_error() -> None:
    agent = _agent()
    entered = asyncio.Event()

    async def waiting(envelope):
        del envelope
        entered.set()
        await asyncio.Future()

    unsubscribe = agent.session.subscribe(waiting, name="observer")
    task = asyncio.create_task(
        agent.session.execute(
            PromptExecutionRequest(run_id="run-1", attempt_id="attempt-1", message="hi")
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    unsubscribe()

    result = await task

    assert result.outcome == "completed"
    assert len(result.delivery_errors) == 1
    assert result.delivery_errors[0].consumer == "observer"
    assert result.delivery_errors[0].reason == "error"
