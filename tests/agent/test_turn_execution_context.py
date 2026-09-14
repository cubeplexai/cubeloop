from __future__ import annotations

import pytest
from pydantic import BaseModel

from cubeloop import Agent
from cubeloop.agent.types import AgentTool, AgentToolResult
from cubeloop.providers.base import (
    AssistantMessage,
    TextContent,
    ToolCall,
    UserMessage,
)
from cubeloop.providers.faux import FauxProvider
from cubeloop.providers.fallback import FallbackBoundModel


class _Args(BaseModel):
    value: str


@pytest.mark.asyncio
async def test_context_captures_transformed_request_as_immutable_view() -> None:
    provider = FauxProvider(provider_id="faux")
    provider.set_responses(
        [AssistantMessage(content=[TextContent(text="done")], stop_reason="end_turn")]
    )

    async def transform(messages, *, ctx, signal=None):
        del ctx, signal
        return [
            *messages,
            UserMessage(
                content=[TextContent(text="snapshot")], metadata={"nested": {"v": 1}}
            ),
        ]

    agent = Agent(
        model=provider.model("faux-model"),
        system_prompt="stable",
        transform_context=transform,
    )

    await agent.prompt("hi", run_id="run-1")

    context = agent.session.turn_contexts[-1]
    assert context.run_id == "run-1"
    assert context.attempt_id
    assert context.model.id == "faux-model"
    assert context.system_prompt == "stable"
    assert context.messages[-1].content[0].text == "snapshot"
    transformed = context.messages[-1]
    with pytest.raises(TypeError):
        transformed.metadata["nested"] = {"v": 2}  # type: ignore[index]
    assert agent.state.messages[0].metadata == {}


@pytest.mark.asyncio
async def test_tool_call_uses_binding_captured_for_the_model_request() -> None:
    calls: list[str] = []

    async def old_execute(tool_call_id, args, *, signal=None, on_update=None):
        del tool_call_id, args, signal, on_update
        calls.append("old")
        return AgentToolResult(content=[TextContent(text="old")])

    async def new_execute(tool_call_id, args, *, signal=None, on_update=None):
        del tool_call_id, args, signal, on_update
        calls.append("new")
        return AgentToolResult(content=[TextContent(text="new")])

    old_tool = AgentTool(
        name="work", description="work", parameters=_Args, execute=old_execute
    )
    new_tool = AgentTool(
        name="work", description="work", parameters=_Args, execute=new_execute
    )
    provider = FauxProvider(provider_id="faux")
    provider.set_responses(
        [
            AssistantMessage(
                content=[ToolCall(id="call-1", name="work", arguments={"value": "x"})],
                stop_reason="tool_use",
            ),
            AssistantMessage(
                content=[TextContent(text="done")], stop_reason="end_turn"
            ),
        ]
    )

    async def replace_after_response(message, context, signal=None):
        del message, signal
        context.tools = [new_tool]

    agent = Agent(
        model=provider.model("faux-model"),
        tools=[old_tool],
        after_model_response=replace_after_response,
    )

    await agent.prompt("hi", run_id="run-1")

    assert calls == ["old"]
    assert len(agent.session.turn_contexts) == 2
    assert (
        agent.session.turn_contexts[0].turn_id != agent.session.turn_contexts[1].turn_id
    )


@pytest.mark.asyncio
async def test_retry_reuses_context_and_fallback_captures_new_model_view() -> None:
    primary = FauxProvider(provider_id="primary").model("primary-model")
    fallback_provider = FauxProvider(provider_id="fallback")
    fallback_provider.set_responses(
        [AssistantMessage(content=[TextContent(text="done")], stop_reason="end_turn")]
    )
    model = FallbackBoundModel(
        chain=(primary, fallback_provider.model("fallback-model")),
        max_retries_per_model=1,
    )
    agent = Agent(model=model)

    await agent.prompt("hi", run_id="run-1")

    contexts = agent.session.turn_contexts
    assert [(ctx.model.provider_id, ctx.model.id) for ctx in contexts] == [
        ("primary", "primary-model"),
        ("fallback", "fallback-model"),
    ]
    assert contexts[0].turn_id == contexts[1].turn_id
