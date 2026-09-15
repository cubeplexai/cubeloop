from __future__ import annotations

from typing import Any

import pytest

from cubeloop.providers.base import (
    AssistantMessage,
    BaseProvider,
    Message,
    Model,
    ReasoningControl,
    StreamOptions,
    TextContent,
    ToolDefinition,
    UserMessage,
)
from cubeloop.providers.faux import FauxProvider, faux_assistant_message


class _RecordingProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(provider_id="rec")
        self.generate_calls: list[dict[str, Any]] = []

    async def generate(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        tool_choice: Any = None,
        options: StreamOptions | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        reasoning: ReasoningControl | None = None,
    ) -> AssistantMessage:
        self.generate_calls.append(
            {
                "model": model,
                "messages": messages,
                "system_prompt": system_prompt,
                "tools": tools,
                "options": options,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
                "reasoning": reasoning,
            }
        )
        return AssistantMessage(
            content=[TextContent(text="ok")],
            provider_id=model.provider_id,
            model_id=model.id,
        )


@pytest.mark.asyncio
async def test_bound_model_generate_forwards_to_provider() -> None:
    provider = _RecordingProvider()
    bound = provider.model("model-x", temperature=0.5)
    messages: list[Message] = [UserMessage(content=[TextContent(text="hi")])]

    response = await bound.generate(
        messages=messages,
        system_prompt="be brief",
        max_output_tokens=64,
        temperature=0.0,
        reasoning=ReasoningControl(mode="off", effort="minimal"),
    )

    assert isinstance(response, AssistantMessage)
    assert response.provider_id == "rec"
    assert response.model_id == "model-x"

    assert len(provider.generate_calls) == 1
    call = provider.generate_calls[0]
    assert call["model"] is bound.spec
    assert call["messages"] is messages
    assert call["system_prompt"] == "be brief"
    assert call["max_output_tokens"] == 64
    assert call["temperature"] == 0.0
    assert call["reasoning"] == ReasoningControl(mode="off", effort="minimal")


@pytest.mark.asyncio
async def test_bound_model_generate_invokes_model_attempt_callback() -> None:
    provider = _RecordingProvider()
    bound = provider.model("model-x")
    attempts: list[Model] = []

    async def capture(model: Model) -> None:
        attempts.append(model)

    await bound.generate(
        messages=[UserMessage(content=[TextContent(text="hi")])],
        options=StreamOptions(on_model_attempt=capture),
    )

    assert attempts == [bound.spec]


@pytest.mark.asyncio
async def test_bound_model_stream_forwards_to_provider() -> None:
    provider = FauxProvider(provider_id="faux")
    provider.set_responses([faux_assistant_message("hello")])
    bound = provider.model("faux-1")

    stream = await bound.stream(
        messages=[UserMessage(content=[TextContent(text="hi")])],
        system_prompt="be brief",
    )

    events: list[str] = []
    async for event in stream:
        events.append(event.type)
        if event.type in ("done", "error"):
            break
    result = await stream.result()

    assert "start" in events
    assert "done" in events
    assert result.model_id == "faux-1"
    assert result.provider_id == "faux"


class _MangledNamesProvider(BaseProvider):
    """A custom provider that follows the Provider protocol shape but uses
    different parameter names for ``model`` and ``messages``. Pre-BoundModel,
    cubepi/agent/loop.py called ``provider.stream(model, messages, ...)``
    positionally, so this kind of provider worked. ``BoundModel.stream`` /
    ``BoundModel.generate`` must keep forwarding those two args positionally
    so this still works.
    """

    def __init__(self) -> None:
        super().__init__(provider_id="mangled")
        self.captured_model: Model | None = None
        self.captured_messages: list[Message] | None = None

    async def generate(
        self,
        spec_in,  # NOT ``model=``
        msg_in,  # NOT ``messages=``
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        tool_choice: Any = None,
        options: StreamOptions | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        reasoning: ReasoningControl | None = None,
    ) -> AssistantMessage:
        self.captured_model = spec_in
        self.captured_messages = msg_in
        return AssistantMessage(
            content=[TextContent(text="ok")],
            provider_id=spec_in.provider_id,
            model_id=spec_in.id,
        )


@pytest.mark.asyncio
async def test_bound_model_generate_forwards_positionally_for_mangled_provider() -> (
    None
):
    provider = _MangledNamesProvider()
    bound = provider.model("model-x")
    messages: list[Message] = [UserMessage(content=[TextContent(text="hi")])]

    response = await bound.generate(messages=messages, system_prompt="x")

    assert response.provider_id == "mangled"
    assert response.model_id == "model-x"
    assert provider.captured_model is bound.spec
    assert provider.captured_messages is messages
