from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from cubeloop import Agent
from dataclasses import FrozenInstanceError

from cubeloop.agent.tools import execute_tool_calls
from cubeloop.agent.types import AgentContext, AgentTool, AgentToolResult
from cubeloop.providers.base import (
    AssistantMessage,
    ReasoningControl,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from cubeloop.providers.faux import FauxProvider
from cubeloop.providers.fallback import FallbackBoundModel
from cubeloop.session import TurnExecutionContext


class _Args(BaseModel):
    value: str


class _ReplacementArgs(BaseModel):
    replacement: str


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
    with pytest.raises(AttributeError, match="missing"):
        _ = transformed.content[0].missing
    with pytest.raises(FrozenInstanceError):
        context.model.id = "changed"
    with pytest.raises(FrozenInstanceError):
        context.reasoning.mode = "on"


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
        old_tool.parameters = _ReplacementArgs
        old_tool.execute = new_execute
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


@pytest.mark.asyncio
async def test_fallback_legs_with_identical_model_specs_capture_separately() -> None:
    primary = FauxProvider(provider_id="same").model("same-model")
    fallback_provider = FauxProvider(provider_id="same")
    fallback_provider.set_responses(
        [AssistantMessage(content=[TextContent(text="done")], stop_reason="end_turn")]
    )
    agent = Agent(
        model=FallbackBoundModel(
            chain=(primary, fallback_provider.model("same-model")),
            max_retries_per_model=0,
        )
    )

    await agent.prompt("hi", run_id="run-1")

    contexts = agent.session.turn_contexts
    assert len(contexts) == 2
    assert [context.model_attempt_id for context in contexts] == [
        "fallback:0",
        "fallback:1",
    ]
    assert contexts[0].turn_id == contexts[1].turn_id


def test_turn_context_extension_is_immutable_and_rejects_rebinding() -> None:
    async def execute(tool_call_id, args, *, signal=None, on_update=None):
        del tool_call_id, args, signal, on_update
        return AgentToolResult(content=[])

    first = AgentTool(
        name="work", description="work", parameters=_Args, execute=execute
    )
    replacement = AgentTool(
        name="work", description="replacement", parameters=_Args, execute=execute
    )
    provider = FauxProvider(provider_id="faux")
    context = TurnExecutionContext.capture(
        turn_id="turn-1",
        run_id="run-1",
        attempt_id="attempt-1",
        model=provider.model("faux-model").spec,
        reasoning=ReasoningControl(),
        system_prompt="",
        messages=[],
        tools=[],
    )

    extended = context.extend(first)
    assert context.tools == ()
    assert extended.binding_for("work") is not None
    assert extended.extend(first) is extended
    with pytest.raises(ValueError, match="already binds"):
        extended.extend(replacement)
    with pytest.raises(FrozenInstanceError):
        extended.tools[0].definition.description = "changed"


def test_message_view_preserves_tool_result_request_fields() -> None:
    provider = FauxProvider(provider_id="faux")
    context = TurnExecutionContext.capture(
        turn_id="turn-1",
        run_id="run-1",
        attempt_id="attempt-1",
        model=provider.model("faux-model").spec,
        reasoning=ReasoningControl(),
        system_prompt="",
        messages=[
            ToolResultMessage(
                tool_call_id="call-1",
                tool_name="work",
                content=[TextContent(text="failed")],
                details={"code": "denied"},
                is_error=True,
            )
        ],
        tools=[],
    )

    message = context.messages[0]
    assert message.tool_call_id == "call-1"
    assert message.tool_name == "work"
    assert message.is_error is True
    assert message.details["code"] == "denied"


@pytest.mark.asyncio
async def test_resolved_tool_extension_updates_public_session_context() -> None:
    async def execute(tool_call_id, args, *, signal=None, on_update=None):
        del tool_call_id, args, signal, on_update
        return AgentToolResult(content=[TextContent(text="done")])

    tool = AgentTool(name="work", description="work", parameters=_Args, execute=execute)
    provider = FauxProvider(provider_id="faux")
    agent = Agent(model=provider.model("faux-model"))
    captured = TurnExecutionContext.capture(
        turn_id="turn-1",
        run_id="run-1",
        attempt_id="attempt-1",
        model=provider.model("faux-model").spec,
        reasoning=ReasoningControl(),
        system_prompt="",
        messages=[],
        tools=[],
    )
    agent.session._capture_turn_context(captured)
    context = AgentContext(
        system_prompt="",
        messages=[],
        tools=[tool],
        turn_execution_context=captured,
        on_turn_context=agent.session._capture_turn_context,
    )

    async def resolve(call, *, context, signal=None):
        del context, signal
        return call.model_copy(update={"name": "work"})

    await execute_tool_calls(
        context,
        AssistantMessage(
            content=[ToolCall(id="call-1", name="alias", arguments={"value": "x"})],
            stop_reason="tool_use",
        ),
        resolve_tool_call=resolve,
        emit=lambda event: None,
    )

    assert len(agent.session.turn_contexts) == 1
    assert agent.session.turn_contexts[0].binding_for("work") is not None


@pytest.mark.asyncio
async def test_tool_batch_uses_captured_execution_mode() -> None:
    order = []

    async def execute(tool_call_id, args, *, signal=None, on_update=None):
        del tool_call_id, signal, on_update
        order.append(f"start:{args.value}")
        await asyncio.sleep(0)
        order.append(f"end:{args.value}")
        return AgentToolResult(content=[TextContent(text="done")])

    tool = AgentTool(
        name="work",
        description="work",
        parameters=_Args,
        execute=execute,
        execution_mode="sequential",
    )
    provider = FauxProvider(provider_id="faux")
    captured = TurnExecutionContext.capture(
        turn_id="turn-1",
        run_id="run-1",
        attempt_id="attempt-1",
        model=provider.model("faux-model").spec,
        reasoning=ReasoningControl(),
        system_prompt="",
        messages=[],
        tools=[tool],
    )
    context = AgentContext(
        system_prompt="",
        messages=[],
        tools=[tool],
        turn_execution_context=captured,
    )
    tool.execution_mode = "parallel"

    await execute_tool_calls(
        context,
        AssistantMessage(
            content=[
                ToolCall(id="call-1", name="work", arguments={"value": "a"}),
                ToolCall(id="call-2", name="work", arguments={"value": "b"}),
            ],
            stop_reason="tool_use",
        ),
        tool_execution="parallel",
        emit=lambda event: None,
    )

    assert order == ["start:a", "end:a", "start:b", "end:b"]
