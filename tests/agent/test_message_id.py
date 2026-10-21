"""Stable transcript ids are assigned once, at append time."""

import asyncio

import pytest

from cubeloop.agent.agent import Agent
from cubeloop.agent.types import AgentContext
from cubeloop.checkpointer.memory import MemoryCheckpointer
from cubeloop.middleware.base import Middleware
from cubeloop.middleware.compaction.state import message_ref
from cubeloop.providers.base import AssistantMessage, Message, TextContent, UserMessage
from cubeloop.providers.faux import FauxProvider


class _CaptureIds(Middleware):
    """Record the ids the loop is about to send to the model."""

    def __init__(self) -> None:
        self.ids: list[str] = []

    async def transform_context(
        self,
        messages: list[Message],
        *,
        ctx: AgentContext,
        signal: asyncio.Event | None = None,
    ) -> list[Message]:
        del ctx, signal
        self.ids = [message.id for message in messages]
        return messages


def _agent(cp: MemoryCheckpointer, middleware: list[Middleware] | None = None) -> Agent:
    provider = FauxProvider()
    provider.set_responses(
        [AssistantMessage(content=[TextContent(text="ok")], stop_reason="end_turn")]
    )
    return Agent(
        model=provider.model("faux-model"),
        checkpointer=cp,
        thread_id="t",
        middleware=middleware,
    )


def test_missing_id_loads_as_empty_string() -> None:
    msg = UserMessage.model_validate(
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    )
    assert msg.id == ""


@pytest.mark.asyncio
async def test_prompt_shares_one_id_across_the_loop_state_and_checkpoint() -> None:
    """The session copies the caller's message before the run.

    ``run_id`` stamping then copies a user message again. Both copies must
    carry the id assigned at append time. The caller's own object is left
    untouched.
    """
    cp = MemoryCheckpointer()
    capture = _CaptureIds()
    agent = _agent(cp, middleware=[capture])
    message = UserMessage(content=[TextContent(text="hello")])

    await agent.prompt(message, run_id="R1")

    assert message.id == ""
    stored_id = agent.state.messages[0].id
    assert stored_id
    assert capture.ids[0] == stored_id
    loaded = await cp.load("t")
    assert loaded is not None
    assert loaded.messages[0].id == stored_id
    assert message_ref(loaded.messages[0]) == f"id:{stored_id}"
    assert agent.state.messages[1].id
    assert agent.state.messages[1].id != stored_id


@pytest.mark.asyncio
async def test_prompt_keeps_a_host_supplied_id() -> None:
    cp = MemoryCheckpointer()
    agent = _agent(cp)
    message = UserMessage(content=[TextContent(text="hello")], id="host-1")

    await agent.prompt(message, run_id="R1")

    assert message.id == "host-1"
    assert agent.state.messages[0].id == "host-1"
    loaded = await cp.load("t")
    assert loaded is not None
    assert loaded.messages[0].id == "host-1"


@pytest.mark.asyncio
async def test_loaded_message_without_id_is_not_backfilled() -> None:
    cp = MemoryCheckpointer()
    old = UserMessage(content=[TextContent(text="old")])
    await cp.append("t", [old])
    agent = _agent(cp)

    await agent.prompt("next", run_id="R2")

    assert agent.state.messages[0].id == ""
    assert agent.state.messages[1].id
    assert message_ref(agent.state.messages[0]).startswith("sha256:")
    assert message_ref(agent.state.messages[1]) == f"id:{agent.state.messages[1].id}"
