import asyncio
from typing import Any

import pytest

from cubeloop.providers.base import (
    AssistantMessage,
    BaseProvider,
    ImageContent,
    Message,
    MessageStream,
    Model,
    ModelCost,
    ReasoningControl,
    StreamEvent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    StreamOptions,
    Usage,
    UserMessage,
    format_provider_error,
    is_empty_completion,
)


class TestFormatProviderError:
    @staticmethod
    def _model() -> Model:
        return Model(id="gpt-4o", provider_id="openai")

    def test_includes_provider_model_and_exception(self):
        msg = format_provider_error(RuntimeError("boom"), self._model())
        assert "openai/gpt-4o" in msg
        assert "RuntimeError" in msg
        assert "boom" in msg

    def test_includes_base_url_when_given(self):
        msg = format_provider_error(
            RuntimeError("boom"),
            self._model(),
            base_url="https://api.deepseek.com/anthropic",
        )
        assert "https://api.deepseek.com/anthropic" in msg

    def test_surfaces_underlying_cause_chain(self):
        # Mirrors openai's APIConnectionError("Connection error.") wrapping the
        # real transport failure in __cause__ — the part users actually need.
        try:
            try:
                raise OSError("Cannot connect to proxy 192.168.1.111:7892")
            except OSError as root:
                raise RuntimeError("Connection error.") from root
        except RuntimeError as exc:
            msg = format_provider_error(exc, self._model())
        assert "Connection error." in msg
        assert "Cannot connect to proxy 192.168.1.111:7892" in msg
        assert "OSError" in msg

    def test_surfaces_implicit_context_cause(self):
        # Exceptions raised during handling without `from` keep the original in
        # __context__; that must still be surfaced.
        try:
            try:
                raise OSError("network down")
            except OSError:
                raise RuntimeError("wrapper")
        except RuntimeError as exc:
            msg = format_provider_error(exc, self._model())
        assert "network down" in msg


class TestMessageTypes:
    def test_text_content_defaults(self):
        tc = TextContent()
        assert tc.type == "text"
        assert tc.text == ""

    def test_text_content_with_value(self):
        tc = TextContent(text="hello")
        assert tc.text == "hello"

    def test_image_content(self):
        ic = ImageContent(source="base64data", media_type="image/png")
        assert ic.type == "image"
        assert ic.source == "base64data"
        assert ic.media_type == "image/png"

    def test_thinking_content(self):
        tc = ThinkingContent(thinking="step by step")
        assert tc.type == "thinking"
        assert tc.thinking == "step by step"

    def test_tool_call(self):
        tc = ToolCall(id="tc-1", name="search", arguments={"query": "hello"})
        assert tc.type == "tool_call"
        assert tc.id == "tc-1"
        assert tc.name == "search"
        assert tc.arguments == {"query": "hello"}

    def test_user_message(self):
        msg = UserMessage(content=[TextContent(text="hi")])
        assert msg.role == "user"
        assert msg.timestamp is None

    def test_assistant_message_defaults(self):
        msg = AssistantMessage(content=[TextContent(text="hello")])
        assert msg.role == "assistant"
        assert msg.stop_reason == "stop"
        assert msg.error_message is None
        assert msg.usage is None

    def test_assistant_message_with_tool_calls(self):
        msg = AssistantMessage(
            content=[
                TextContent(text="Let me search."),
                ToolCall(id="tc-1", name="search", arguments={"q": "test"}),
            ],
            stop_reason="tool_use",
        )
        assert len(msg.content) == 2
        assert msg.content[1].type == "tool_call"

    def test_tool_result_message(self):
        msg = ToolResultMessage(
            tool_call_id="tc-1",
            tool_name="search",
            content=[TextContent(text="result")],
        )
        assert msg.role == "tool_result"
        assert msg.is_error is False

    def test_model_defaults(self):
        m = Model(id="gpt-4o", provider_id="openai")
        assert m.context_window == 200_000
        assert m.max_tokens == 8192
        assert m.reasoning is False
        assert m.cost is None

    def test_usage(self):
        u = Usage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0

    def test_model_cost(self):
        c = ModelCost(input=3.0, output=15.0)
        assert c.cache_read == 0

    def test_tool_definition(self):
        td = ToolDefinition(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        assert td.name == "search"


class TestStreamEvent:
    def test_content_index_default_none(self):
        event = StreamEvent(type="text_delta", delta="hi")
        assert event.content_index is None

    def test_content_index_set(self):
        event = StreamEvent(type="text_start", content_index=0)
        assert event.content_index == 0


class TestMessageStream:
    async def test_stream_iteration_and_result(self):
        stream = MessageStream()
        msg = AssistantMessage(content=[TextContent(text="hello")])

        async def produce():
            await asyncio.sleep(0)
            stream.push(StreamEvent(type="text_delta", delta="hello"))
            stream.push(StreamEvent(type="done"))
            stream.set_result(msg)

        asyncio.create_task(produce())

        events = []
        async for event in stream:
            events.append(event)

        assert len(events) == 2
        assert events[0].type == "text_delta"
        assert events[1].type == "done"

        result = await stream.result()
        assert result.content[0].text == "hello"

    async def test_stream_error_event(self):
        stream = MessageStream()
        error_msg = AssistantMessage(
            content=[],
            stop_reason="error",
            error_message="API error",
        )

        async def produce():
            await asyncio.sleep(0)
            stream.push(StreamEvent(type="error", error_message="API error"))
            stream.set_result(error_msg)

        asyncio.create_task(produce())

        events = []
        async for event in stream:
            events.append(event)

        result = await stream.result()
        assert result.stop_reason == "error"

    async def test_stream_subsequent_anext_after_exhaustion(self):
        stream = MessageStream()
        stream.push(StreamEvent(type="done"))
        stream.set_result(AssistantMessage(content=[]))

        events = [ev async for ev in stream]
        assert len(events) == 1
        assert events[0].type == "done"

        # Subsequent anext calls must continue to raise StopAsyncIteration without hanging
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()


class TestIsEmptyCompletion:
    def test_empty_content_with_stop(self):
        msg = AssistantMessage(content=[], stop_reason="stop")
        assert is_empty_completion(msg) is True

    def test_whitespace_text_content_with_stop(self):
        msg = AssistantMessage(
            content=[TextContent(text="   \n\t  ")],
            stop_reason="stop",
        )
        assert is_empty_completion(msg) is True

    def test_thinking_content_with_text_returns_false(self):
        msg = AssistantMessage(
            content=[ThinkingContent(thinking="analyzing query...")],
            stop_reason="stop",
        )
        assert is_empty_completion(msg) is False

    def test_thinking_content_empty_or_whitespace_returns_true(self):
        msg = AssistantMessage(
            content=[ThinkingContent(thinking="   \n ")],
            stop_reason="stop",
        )
        assert is_empty_completion(msg) is True

    def test_empty_thinking_and_empty_text_with_stop(self):
        msg = AssistantMessage(
            content=[
                ThinkingContent(thinking=""),
                TextContent(text=""),
            ],
            stop_reason="stop",
        )
        assert is_empty_completion(msg) is True

    def test_non_empty_text_returns_false(self):
        msg = AssistantMessage(
            content=[TextContent(text="Hello world")],
            stop_reason="stop",
        )
        assert is_empty_completion(msg) is False

    def test_thinking_and_non_empty_text_returns_false(self):
        msg = AssistantMessage(
            content=[
                ThinkingContent(thinking="analyzing..."),
                TextContent(text="Result"),
            ],
            stop_reason="stop",
        )
        assert is_empty_completion(msg) is False

    def test_tool_call_returns_false(self):
        msg = AssistantMessage(
            content=[ToolCall(id="tc-1", name="search", arguments={})],
            stop_reason="stop",
        )
        assert is_empty_completion(msg) is False

    def test_non_stop_stop_reason_returns_false(self):
        assert (
            is_empty_completion(AssistantMessage(content=[], stop_reason="tool_use"))
            is False
        )
        assert (
            is_empty_completion(AssistantMessage(content=[], stop_reason="length"))
            is False
        )
        assert (
            is_empty_completion(AssistantMessage(content=[], stop_reason="error"))
            is False
        )
        assert (
            is_empty_completion(AssistantMessage(content=[], stop_reason="aborted"))
            is False
        )

    def test_image_content_returns_false(self):
        msg = AssistantMessage(
            content=[ImageContent(data="abc", mime_type="image/png")],
            stop_reason="stop",
        )
        assert is_empty_completion(msg) is False


class TestMessageStreamTaskTracking:
    async def test_attach_task_stores_reference(self):
        ms = MessageStream()

        async def dummy():
            ms.push(StreamEvent(type="done"))
            ms.set_result(AssistantMessage(content=[]))

        task = asyncio.create_task(dummy())
        ms.attach_task(task)
        assert ms._producer_task is task
        await task

    async def test_result_propagates_task_exception(self):
        ms = MessageStream()

        async def failing():
            raise RuntimeError("producer died before pushing error")

        task = asyncio.create_task(failing())
        ms.attach_task(task)
        with pytest.raises(RuntimeError, match="producer died"):
            await ms.result()


class TestBaseProviderGenerate:
    async def test_generate_consumes_stream_and_returns_assistant_message(self):
        provider = _RecordingProvider(
            AssistantMessage(content=[TextContent(text="ok")])
        )

        result = await provider.generate(
            Model(id="gpt-4o", provider_id="openai"),
            [UserMessage(content=[TextContent(text="hi")])],
            system_prompt="system",
        )

        assert result.content[0].text == "ok"
        assert provider.call_count == 1

    async def test_generate_applies_common_per_call_overrides(self):
        provider = _RecordingProvider(AssistantMessage(content=[]))
        base_model = Model(id="gpt-4o", provider_id="openai", max_tokens=128)
        base_options = StreamOptions(reasoning=ReasoningControl(mode="auto"))
        reasoning = ReasoningControl(mode="on", effort="high")

        await provider.generate(
            base_model,
            [],
            options=base_options,
            max_output_tokens=512,
            temperature=0.0,
            reasoning=reasoning,
        )

        assert provider.seen_model is not None
        assert provider.seen_model.max_tokens == 512
        assert provider.seen_model.temperature == 0.0
        assert base_model.max_tokens == 128
        assert provider.seen_options is not None
        assert provider.seen_options.reasoning == reasoning
        assert base_options.reasoning == ReasoningControl(mode="auto")


class TestAssistantMessageMetadata:
    def test_default_metadata_fields(self):
        msg = AssistantMessage(content=[])
        assert msg.provider_id == ""
        assert msg.model_id == ""
        assert msg.response_id is None

    def test_metadata_fields_set(self):
        msg = AssistantMessage(
            content=[],
            provider_id="anthropic",
            model_id="claude-sonnet-4-20250514",
            response_id="msg_abc123",
        )
        assert msg.provider_id == "anthropic"
        assert msg.model_id == "claude-sonnet-4-20250514"
        assert msg.response_id == "msg_abc123"


class _RecordingProvider(BaseProvider):
    def __init__(self, result: AssistantMessage) -> None:
        super().__init__()
        self._result = result
        self.call_count = 0
        self.seen_model: Model | None = None
        self.seen_options: StreamOptions | None = None

    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        tool_choice: Any = None,
        options: StreamOptions | None = None,
    ) -> MessageStream:
        del messages, system_prompt, tools, tool_choice
        self.call_count += 1
        self.seen_model = model
        self.seen_options = options
        stream = MessageStream()
        stream.push(StreamEvent(type="done"))
        stream.set_result(self._result)
        return stream
