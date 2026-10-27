"""Tests for OpenAIResponsesProvider."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cubeloop.providers.base import (
    AssistantMessage,
    ImageContent,
    Model,
    ReasoningControl,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    UserMessage,
)
from cubeloop.providers.openai_responses import OpenAIResponsesProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _model(*, reasoning: bool = False, max_tokens: int = 4096) -> Model:
    return Model(
        id="o3" if reasoning else "gpt-4.1",
        provider_id="openai",
        api="openai-responses",
        reasoning=reasoning,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Message conversion tests (_build_input / _convert_tool)
# ---------------------------------------------------------------------------


class TestBuildInput:
    def test_user_message(self):
        msgs = [UserMessage(content=[TextContent(text="hello")])]
        result = OpenAIResponsesProvider._build_input(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == [{"type": "input_text", "text": "hello"}]

    def test_assistant_text_message(self):
        msgs = [AssistantMessage(content=[TextContent(text="hi there")])]
        result = OpenAIResponsesProvider._build_input(msgs)
        assert len(result) == 1
        assert result[0]["type"] == "message"
        assert result[0]["role"] == "assistant"
        assert result[0]["content"][0]["type"] == "output_text"
        assert result[0]["content"][0]["text"] == "hi there"
        assert result[0]["status"] == "completed"

    def test_assistant_tool_call(self):
        msgs = [
            AssistantMessage(
                content=[
                    ToolCall(
                        id="call_123|fc_456",
                        name="search",
                        arguments={"q": "test"},
                    )
                ],
                stop_reason="tool_use",
            )
        ]
        result = OpenAIResponsesProvider._build_input(msgs)
        assert len(result) == 1
        assert result[0]["type"] == "function_call"
        assert result[0]["call_id"] == "call_123"
        assert result[0]["id"] == "fc_456"
        assert result[0]["name"] == "search"
        assert result[0]["arguments"] == '{"q": "test"}'

    def test_assistant_tool_call_simple_id(self):
        """Tool call id without pipe separator should omit item id."""
        msgs = [
            AssistantMessage(
                content=[ToolCall(id="call_123", name="search", arguments={"q": "x"})],
            )
        ]
        result = OpenAIResponsesProvider._build_input(msgs)
        assert result[0]["call_id"] == "call_123"
        assert "id" not in result[0]

    def test_tool_result_message(self):
        msgs = [
            ToolResultMessage(
                tool_call_id="call_123|fc_456",
                tool_name="search",
                content=[TextContent(text="found it")],
            )
        ]
        result = OpenAIResponsesProvider._build_input(msgs)
        assert len(result) == 1
        assert result[0]["type"] == "function_call_output"
        assert result[0]["call_id"] == "call_123"
        assert result[0]["output"] == "found it"

    def test_thinking_content_skipped(self):
        """ThinkingContent should be skipped in input conversion."""
        msgs = [
            AssistantMessage(
                content=[
                    ThinkingContent(thinking="let me think..."),
                    TextContent(text="answer"),
                ]
            )
        ]
        result = OpenAIResponsesProvider._build_input(msgs)
        assert len(result) == 1
        assert result[0]["type"] == "message"

    def test_mixed_conversation(self):
        msgs = [
            UserMessage(content=[TextContent(text="hello")]),
            AssistantMessage(
                content=[
                    TextContent(text="hi"),
                    ToolCall(id="c1|fc_1", name="search", arguments={"q": "test"}),
                ],
                stop_reason="tool_use",
            ),
            ToolResultMessage(
                tool_call_id="c1|fc_1",
                tool_name="search",
                content=[TextContent(text="result")],
            ),
        ]
        result = OpenAIResponsesProvider._build_input(msgs)
        assert len(result) == 4  # user + message + function_call + function_call_output
        assert result[0]["role"] == "user"
        assert result[1]["type"] == "message"
        assert result[2]["type"] == "function_call"
        assert result[3]["type"] == "function_call_output"


class TestOpenAIResponsesImageConversion:
    def test_user_message_with_image(self):
        msg = UserMessage(
            content=[
                TextContent(text="Describe this"),
                ImageContent(source="imgdata", media_type="image/jpeg"),
            ]
        )
        result = OpenAIResponsesProvider._build_input([msg])
        assert len(result) == 1
        assert result[0]["role"] == "user"
        content = result[0]["content"]
        assert len(content) == 2
        assert content[0] == {"type": "input_text", "text": "Describe this"}
        assert content[1] == {
            "type": "input_image",
            "image_url": "data:image/jpeg;base64,imgdata",
        }


class TestConvertTool:
    def test_convert_tool_definition(self):
        td = ToolDefinition(
            name="search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        )
        result = OpenAIResponsesProvider._convert_tool(td)
        assert result["type"] == "function"
        assert result["name"] == "search"
        assert result["description"] == "Search the web"
        assert result["parameters"]["type"] == "object"


class TestMapStopReason:
    def test_completed_no_tools(self):
        partial = AssistantMessage(content=[TextContent(text="hello")])
        assert OpenAIResponsesProvider._map_stop_reason("completed", partial) == "stop"

    def test_completed_with_tools(self):
        partial = AssistantMessage(content=[ToolCall(id="x", name="y", arguments={})])
        assert (
            OpenAIResponsesProvider._map_stop_reason("completed", partial) == "tool_use"
        )

    def test_incomplete(self):
        partial = AssistantMessage(content=[])
        assert (
            OpenAIResponsesProvider._map_stop_reason("incomplete", partial) == "length"
        )

    def test_failed(self):
        partial = AssistantMessage(content=[])
        assert OpenAIResponsesProvider._map_stop_reason("failed", partial) == "error"

    def test_none_status(self):
        partial = AssistantMessage(content=[])
        assert OpenAIResponsesProvider._map_stop_reason(None, partial) == "stop"


# ---------------------------------------------------------------------------
# Streaming tests (with mocked OpenAI client)
# ---------------------------------------------------------------------------


def _make_event(type: str, **kwargs) -> SimpleNamespace:
    """Create a mock streaming event."""
    event = SimpleNamespace(type=type, **kwargs)
    return event


def _make_output_item(type: str, **kwargs) -> SimpleNamespace:
    return SimpleNamespace(type=type, **kwargs)


async def _async_iter(events):
    """Create an async iterator from a list of events."""
    for event in events:
        yield event


class TestStreamText:
    """Test streaming text output."""

    @pytest.mark.asyncio
    async def test_stream_simple_text(self):
        events = [
            _make_event(
                "response.output_item.added",
                item=_make_output_item("message", id="msg_1", content=[]),
            ),
            _make_event(
                "response.content_part.added",
                part=SimpleNamespace(type="output_text", text=""),
            ),
            _make_event("response.output_text.delta", delta="Hello "),
            _make_event("response.output_text.delta", delta="world!"),
            _make_event(
                "response.output_item.done",
                item=_make_output_item(
                    "message",
                    id="msg_1",
                    content=[SimpleNamespace(type="output_text", text="Hello world!")],
                    status="completed",
                ),
            ),
            _make_event(
                "response.completed",
                response=SimpleNamespace(
                    id="resp_1",
                    status="completed",
                    usage=SimpleNamespace(
                        input_tokens=10,
                        output_tokens=5,
                        total_tokens=15,
                        input_tokens_details=SimpleNamespace(cached_tokens=2),
                    ),
                    service_tier=None,
                ),
            ),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.responses = MagicMock()
            mock_client.responses.create = AsyncMock(return_value=_async_iter(events))

            provider = OpenAIResponsesProvider(api_key="test-key")
            provider._client = mock_client

            model = _model()
            ms = await provider.stream(
                model, [UserMessage(content=[TextContent(text="hi")])]
            )

            stream_events = []
            async for event in ms:
                stream_events.append(event)

            result = await ms.result()

            # Check event sequence
            event_types = [e.type for e in stream_events]
            assert "start" in event_types
            assert "text_start" in event_types
            assert "text_delta" in event_types
            assert "text_end" in event_types
            assert "done" in event_types

            # Check final message
            assert result.stop_reason == "stop"
            assert len(result.content) == 1
            assert isinstance(result.content[0], TextContent)
            assert result.content[0].text == "Hello world!"

            # Check usage
            assert result.usage is not None
            assert result.usage.input_tokens == 8  # 10 - 2 cached
            assert result.usage.output_tokens == 5
            assert result.usage.cache_read_tokens == 2


class TestStreamThinking:
    """Test streaming reasoning/thinking output."""

    @pytest.mark.asyncio
    async def test_stream_with_reasoning(self):
        events = [
            _make_event(
                "response.output_item.added",
                item=_make_output_item("reasoning", id="rs_1", summary=[]),
            ),
            _make_event("response.reasoning_summary_text.delta", delta="Thinking..."),
            _make_event(
                "response.output_item.done",
                item=_make_output_item(
                    "reasoning",
                    id="rs_1",
                    summary=[SimpleNamespace(text="Thinking...")],
                ),
            ),
            _make_event(
                "response.output_item.added",
                item=_make_output_item("message", id="msg_1", content=[]),
            ),
            _make_event("response.output_text.delta", delta="Answer"),
            _make_event(
                "response.output_item.done",
                item=_make_output_item(
                    "message",
                    id="msg_1",
                    content=[SimpleNamespace(type="output_text", text="Answer")],
                ),
            ),
            _make_event(
                "response.completed",
                response=SimpleNamespace(
                    id="resp_1",
                    status="completed",
                    usage=SimpleNamespace(
                        input_tokens=10,
                        output_tokens=20,
                        total_tokens=30,
                        input_tokens_details=SimpleNamespace(cached_tokens=0),
                    ),
                    service_tier=None,
                ),
            ),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.responses = MagicMock()
            mock_client.responses.create = AsyncMock(return_value=_async_iter(events))

            provider = OpenAIResponsesProvider(api_key="test-key")
            provider._client = mock_client

            model = _model(reasoning=True)
            ms = await provider.stream(
                model,
                [UserMessage(content=[TextContent(text="think about this")])],
                options=StreamOptions(
                    reasoning=ReasoningControl(
                        mode="on",
                        effort="medium",
                        summary="auto",
                    )
                ),
            )

            stream_events = []
            async for event in ms:
                stream_events.append(event)

            result = await ms.result()

            event_types = [e.type for e in stream_events]
            assert "thinking_start" in event_types
            assert "thinking_delta" in event_types
            assert "thinking_end" in event_types
            assert "text_start" in event_types
            assert "text_delta" in event_types
            assert "text_end" in event_types

            # Check content blocks
            assert len(result.content) == 2
            assert isinstance(result.content[0], ThinkingContent)
            assert isinstance(result.content[1], TextContent)
            assert result.content[1].text == "Answer"

            # Verify reasoning params were sent
            call_kwargs = mock_client.responses.create.call_args
            assert call_kwargs.kwargs.get("reasoning") is not None or (
                len(call_kwargs.args) > 0
                and isinstance(call_kwargs.args[0], dict)
                and "reasoning" in call_kwargs.args[0]
            )


class TestStreamToolUse:
    """Test streaming tool use."""

    @pytest.mark.asyncio
    async def test_stream_tool_call(self):
        events = [
            _make_event(
                "response.output_item.added",
                item=_make_output_item(
                    "function_call",
                    id="fc_1",
                    call_id="call_abc",
                    name="search",
                    arguments="",
                ),
            ),
            _make_event("response.function_call_arguments.delta", delta='{"q": '),
            _make_event("response.function_call_arguments.delta", delta='"test"}'),
            _make_event(
                "response.function_call_arguments.done",
                arguments='{"q": "test"}',
            ),
            _make_event(
                "response.output_item.done",
                item=_make_output_item(
                    "function_call",
                    id="fc_1",
                    call_id="call_abc",
                    name="search",
                    arguments='{"q": "test"}',
                ),
            ),
            _make_event(
                "response.completed",
                response=SimpleNamespace(
                    id="resp_1",
                    status="completed",
                    usage=SimpleNamespace(
                        input_tokens=10,
                        output_tokens=15,
                        total_tokens=25,
                        input_tokens_details=SimpleNamespace(cached_tokens=0),
                    ),
                    service_tier=None,
                ),
            ),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.responses = MagicMock()
            mock_client.responses.create = AsyncMock(return_value=_async_iter(events))

            provider = OpenAIResponsesProvider(api_key="test-key")
            provider._client = mock_client

            tools = [
                ToolDefinition(
                    name="search",
                    description="Search",
                    parameters={
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                )
            ]

            model = _model()
            ms = await provider.stream(
                model,
                [UserMessage(content=[TextContent(text="search for test")])],
                tools=tools,
            )

            stream_events = []
            async for event in ms:
                stream_events.append(event)

            result = await ms.result()

            event_types = [e.type for e in stream_events]
            assert "toolcall_start" in event_types
            assert "toolcall_delta" in event_types
            assert "toolcall_end" in event_types

            assert result.stop_reason == "tool_use"
            assert len(result.content) == 1
            assert isinstance(result.content[0], ToolCall)
            assert result.content[0].name == "search"
            assert result.content[0].arguments == {"q": "test"}
            assert result.content[0].id == "call_abc|fc_1"


class TestStreamParallelToolCalls:
    """Test parallel tool calls with interleaved events."""

    @pytest.mark.asyncio
    async def test_parallel_tool_calls(self):
        events = [
            _make_event(
                "response.output_item.added",
                item=_make_output_item(
                    "function_call",
                    id="fc_1",
                    call_id="call_a",
                    name="search",
                    arguments="",
                ),
            ),
            _make_event(
                "response.output_item.added",
                item=_make_output_item(
                    "function_call",
                    id="fc_2",
                    call_id="call_b",
                    name="fetch",
                    arguments="",
                ),
            ),
            _make_event("response.function_call_arguments.delta", delta='{"q": "x"}'),
            _make_event(
                "response.function_call_arguments.done",
                arguments='{"q": "x"}',
            ),
            _make_event(
                "response.output_item.done",
                item=_make_output_item(
                    "function_call",
                    id="fc_1",
                    call_id="call_a",
                    name="search",
                    arguments='{"q": "x"}',
                ),
            ),
            _make_event("response.function_call_arguments.delta", delta='{"url": "y"}'),
            _make_event(
                "response.function_call_arguments.done",
                arguments='{"url": "y"}',
            ),
            _make_event(
                "response.output_item.done",
                item=_make_output_item(
                    "function_call",
                    id="fc_2",
                    call_id="call_b",
                    name="fetch",
                    arguments='{"url": "y"}',
                ),
            ),
            _make_event(
                "response.completed",
                response=SimpleNamespace(
                    id="resp_1",
                    status="completed",
                    usage=SimpleNamespace(
                        input_tokens=10,
                        output_tokens=20,
                        total_tokens=30,
                        input_tokens_details=SimpleNamespace(cached_tokens=0),
                    ),
                    service_tier=None,
                ),
            ),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.responses = MagicMock()
            mock_client.responses.create = AsyncMock(return_value=_async_iter(events))

            provider = OpenAIResponsesProvider(api_key="test-key")
            provider._client = mock_client

            model = _model()
            ms = await provider.stream(
                model,
                [UserMessage(content=[TextContent(text="do both")])],
            )

            stream_events = []
            async for event in ms:
                stream_events.append(event)

            result = await ms.result()

            assert result.stop_reason == "tool_use"
            assert len(result.content) == 2
            assert isinstance(result.content[0], ToolCall)
            assert isinstance(result.content[1], ToolCall)
            assert result.content[0].name == "search"
            assert result.content[0].arguments == {"q": "x"}
            assert result.content[0].id == "call_a|fc_1"
            assert result.content[1].name == "fetch"
            assert result.content[1].arguments == {"url": "y"}
            assert result.content[1].id == "call_b|fc_2"


class TestStreamAbort:
    """Test abort handling."""

    @pytest.mark.asyncio
    async def test_abort_signal(self):
        signal = asyncio.Event()
        signal.set()  # Already aborted

        events = [
            _make_event(
                "response.output_item.added",
                item=_make_output_item("message", id="msg_1", content=[]),
            ),
            _make_event("response.output_text.delta", delta="Hello"),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.responses = MagicMock()
            mock_client.responses.create = AsyncMock(return_value=_async_iter(events))

            provider = OpenAIResponsesProvider(api_key="test-key")
            provider._client = mock_client

            model = _model()
            ms = await provider.stream(
                model,
                [UserMessage(content=[TextContent(text="hi")])],
                options=StreamOptions(signal=signal),
            )

            stream_events = []
            async for event in ms:
                stream_events.append(event)

            result = await ms.result()
            assert result.stop_reason == "aborted"


class TestStreamError:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_api_error(self):
        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.responses = MagicMock()
            mock_client.responses.create = AsyncMock(
                side_effect=RuntimeError("API down")
            )

            provider = OpenAIResponsesProvider(api_key="test-key")
            provider._client = mock_client

            model = _model()
            ms = await provider.stream(
                model,
                [UserMessage(content=[TextContent(text="hi")])],
            )

            stream_events = []
            async for event in ms:
                stream_events.append(event)

            result = await ms.result()
            assert result.stop_reason == "error"
            assert "API down" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_stream_error_event(self):
        events = [
            _make_event("error", message="rate limit exceeded", code="rate_limit"),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.responses = MagicMock()
            mock_client.responses.create = AsyncMock(return_value=_async_iter(events))

            provider = OpenAIResponsesProvider(api_key="test-key")
            provider._client = mock_client

            model = _model()
            ms = await provider.stream(
                model,
                [UserMessage(content=[TextContent(text="hi")])],
            )

            stream_events = []
            async for event in ms:
                stream_events.append(event)

            result = await ms.result()
            assert result.stop_reason == "error"
            assert "rate limit exceeded" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_response_failed_event(self):
        events = [
            _make_event(
                "response.failed",
                response=SimpleNamespace(
                    error=SimpleNamespace(code="server_error", message="internal"),
                    incomplete_details=None,
                ),
            ),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.responses = MagicMock()
            mock_client.responses.create = AsyncMock(return_value=_async_iter(events))

            provider = OpenAIResponsesProvider(api_key="test-key")
            provider._client = mock_client

            model = _model()
            ms = await provider.stream(
                model,
                [UserMessage(content=[TextContent(text="hi")])],
            )

            stream_events = []
            async for event in ms:
                stream_events.append(event)

            result = await ms.result()
            assert result.stop_reason == "error"
            assert "server_error" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_empty_completion_emits_error(self):
        events = [
            _make_event(
                "response.completed",
                response=SimpleNamespace(
                    status="completed",
                    usage=None,
                    id="resp_empty",
                    error=None,
                    incomplete_details=None,
                ),
            ),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.responses = MagicMock()
            mock_client.responses.create = AsyncMock(return_value=_async_iter(events))

            provider = OpenAIResponsesProvider(api_key="test-key")
            provider._client = mock_client

            model = _model()
            ms = await provider.stream(
                model,
                [UserMessage(content=[TextContent(text="hi")])],
            )

            stream_events = [e async for e in ms]
            result = await ms.result()

            assert result.stop_reason == "error"
            assert result.error_type == "EmptyCompletion"
            assert any(
                e.type == "error" and e.error_type == "EmptyCompletion"
                for e in stream_events
            )


class TestReasoningParams:
    """Test that reasoning parameters are correctly configured."""

    @pytest.mark.asyncio
    async def test_responses_default_profile_writes_reasoning_object(self):
        events = [
            _make_event(
                "response.completed",
                response=SimpleNamespace(
                    id="resp_1",
                    status="completed",
                    usage=SimpleNamespace(
                        input_tokens=5,
                        output_tokens=5,
                        total_tokens=10,
                        input_tokens_details=SimpleNamespace(cached_tokens=0),
                    ),
                    service_tier=None,
                ),
            ),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.responses = MagicMock()
            mock_client.responses.create = AsyncMock(return_value=_async_iter(events))

            provider = OpenAIResponsesProvider(api_key="test-key")
            provider._client = mock_client

            model = _model(reasoning=True)
            ms = await provider.stream(
                model,
                [UserMessage(content=[TextContent(text="think")])],
                options=StreamOptions(
                    reasoning=ReasoningControl(
                        mode="on",
                        effort="high",
                        summary="auto",
                    )
                ),
            )

            async for _ in ms:
                pass
            await ms.result()

            call_kwargs = mock_client.responses.create.call_args[1]
            assert call_kwargs["reasoning"] == {
                "effort": "high",
                "summary": "auto",
            }
            assert call_kwargs["include"] == ["reasoning.encrypted_content"]

    @pytest.mark.asyncio
    async def test_reasoning_effort_sent(self):
        """ReasoningControl writes Responses reasoning params."""
        events = [
            _make_event(
                "response.completed",
                response=SimpleNamespace(
                    id="resp_1",
                    status="completed",
                    usage=SimpleNamespace(
                        input_tokens=5,
                        output_tokens=5,
                        total_tokens=10,
                        input_tokens_details=SimpleNamespace(cached_tokens=0),
                    ),
                    service_tier=None,
                ),
            ),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.responses = MagicMock()
            mock_client.responses.create = AsyncMock(return_value=_async_iter(events))

            provider = OpenAIResponsesProvider(api_key="test-key")
            provider._client = mock_client

            model = _model(reasoning=True)
            ms = await provider.stream(
                model,
                [UserMessage(content=[TextContent(text="think")])],
                options=StreamOptions(
                    reasoning=ReasoningControl(
                        mode="on",
                        effort="high",
                        summary="auto",
                    )
                ),
            )

            # Consume the stream to ensure the background task has run
            async for _ in ms:
                pass
            await ms.result()

            call_kwargs = mock_client.responses.create.call_args[1]
            assert call_kwargs["reasoning"] == {"effort": "high", "summary": "auto"}
            assert "reasoning.encrypted_content" in call_kwargs["include"]

    @pytest.mark.asyncio
    async def test_no_reasoning_when_off(self):
        """mode=off maps to "none" effort for reasoning-only Responses models."""
        events = [
            _make_event(
                "response.completed",
                response=SimpleNamespace(
                    id="resp_1",
                    status="completed",
                    usage=SimpleNamespace(
                        input_tokens=5,
                        output_tokens=5,
                        total_tokens=10,
                        input_tokens_details=SimpleNamespace(cached_tokens=0),
                    ),
                    service_tier=None,
                ),
            ),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.responses = MagicMock()
            mock_client.responses.create = AsyncMock(return_value=_async_iter(events))

            provider = OpenAIResponsesProvider(api_key="test-key")
            provider._client = mock_client

            model = _model(reasoning=True)
            ms = await provider.stream(
                model,
                [UserMessage(content=[TextContent(text="hi")])],
                options=StreamOptions(
                    reasoning=ReasoningControl(
                        mode="off",
                        effort="minimal",
                        summary="none",
                    )
                ),
            )
            async for _ in ms:
                pass
            await ms.result()

            call_kwargs = mock_client.responses.create.call_args[1]
            assert call_kwargs["reasoning"] == {"effort": "none"}
            assert "include" not in call_kwargs

    @pytest.mark.asyncio
    async def test_system_prompt_as_developer_for_reasoning(self):
        """For reasoning models, system prompt should use 'developer' role."""
        events = [
            _make_event(
                "response.completed",
                response=SimpleNamespace(
                    id="resp_1",
                    status="completed",
                    usage=SimpleNamespace(
                        input_tokens=5,
                        output_tokens=5,
                        total_tokens=10,
                        input_tokens_details=SimpleNamespace(cached_tokens=0),
                    ),
                    service_tier=None,
                ),
            ),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.responses = MagicMock()
            mock_client.responses.create = AsyncMock(return_value=_async_iter(events))

            provider = OpenAIResponsesProvider(api_key="test-key")
            provider._client = mock_client

            model = _model(reasoning=True)
            ms = await provider.stream(
                model,
                [UserMessage(content=[TextContent(text="hi")])],
                system_prompt="You are helpful.",
            )
            async for _ in ms:
                pass
            await ms.result()

            call_kwargs = mock_client.responses.create.call_args[1]
            api_input = call_kwargs["input"]
            # First item should be the system/developer prompt
            assert api_input[0]["role"] == "developer"
            assert api_input[0]["content"] == "You are helpful."

    @pytest.mark.asyncio
    async def test_system_prompt_as_system_for_non_reasoning(self):
        """For non-reasoning models, system prompt should use 'system' role."""
        events = [
            _make_event(
                "response.completed",
                response=SimpleNamespace(
                    id="resp_1",
                    status="completed",
                    usage=SimpleNamespace(
                        input_tokens=5,
                        output_tokens=5,
                        total_tokens=10,
                        input_tokens_details=SimpleNamespace(cached_tokens=0),
                    ),
                    service_tier=None,
                ),
            ),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.responses = MagicMock()
            mock_client.responses.create = AsyncMock(return_value=_async_iter(events))

            provider = OpenAIResponsesProvider(api_key="test-key")
            provider._client = mock_client

            model = _model(reasoning=False)
            ms = await provider.stream(
                model,
                [UserMessage(content=[TextContent(text="hi")])],
                system_prompt="You are helpful.",
            )
            async for _ in ms:
                pass
            await ms.result()

            call_kwargs = mock_client.responses.create.call_args[1]
            api_input = call_kwargs["input"]
            assert api_input[0]["role"] == "system"


class TestRefusalHandling:
    """Test that refusal deltas are handled correctly."""

    @pytest.mark.asyncio
    async def test_refusal_delta(self):
        events = [
            _make_event(
                "response.output_item.added",
                item=_make_output_item("message", id="msg_1", content=[]),
            ),
            _make_event("response.refusal.delta", delta="I cannot help with that."),
            _make_event(
                "response.output_item.done",
                item=_make_output_item(
                    "message",
                    id="msg_1",
                    content=[
                        SimpleNamespace(
                            type="refusal", refusal="I cannot help with that."
                        )
                    ],
                ),
            ),
            _make_event(
                "response.completed",
                response=SimpleNamespace(
                    id="resp_1",
                    status="completed",
                    usage=SimpleNamespace(
                        input_tokens=5,
                        output_tokens=10,
                        total_tokens=15,
                        input_tokens_details=SimpleNamespace(cached_tokens=0),
                    ),
                    service_tier=None,
                ),
            ),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.responses = MagicMock()
            mock_client.responses.create = AsyncMock(return_value=_async_iter(events))

            provider = OpenAIResponsesProvider(api_key="test-key")
            provider._client = mock_client

            model = _model()
            ms = await provider.stream(
                model,
                [UserMessage(content=[TextContent(text="help")])],
            )

            stream_events = []
            async for event in ms:
                stream_events.append(event)

            result = await ms.result()
            assert len(result.content) == 1
            assert isinstance(result.content[0], TextContent)
            assert "cannot help" in result.content[0].text
