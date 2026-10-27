from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cubeloop.providers.openai import OpenAIProvider
from cubeloop.providers.base import (
    AssistantMessage,
    ImageContent,
    Model,
    StreamOptions,
    TextContent,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    UserMessage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model() -> Model:
    return Model(id="gpt-4o", provider_id="openai", api="openai")


def _make_chunk(
    content=None,
    tool_calls=None,
    finish_reason=None,
    id=None,
):
    """Build a mock OpenAI chat-completion chunk."""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(id=id, choices=[choice])


def _make_empty_chunk(id=None):
    """Build a chunk with empty choices list (e.g. role-only chunk)."""
    return SimpleNamespace(id=id, choices=[])


def _tc_delta(index, id=None, name=None, arguments=None):
    """Build a tool-call delta fragment."""
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=id, function=fn)


async def _async_iter(chunks):
    for chunk in chunks:
        yield chunk


class TestOpenAIMessageConversion:
    def test_convert_user_message(self):
        msg = UserMessage(content=[TextContent(text="hello")])
        result = OpenAIProvider._convert_message(msg)
        assert result["role"] == "user"
        assert result["content"] == "hello"

    def test_convert_assistant_message(self):
        msg = AssistantMessage(content=[TextContent(text="hi")])
        result = OpenAIProvider._convert_message(msg)
        assert result["role"] == "assistant"
        assert result["content"] == "hi"

    def test_convert_assistant_with_tool_calls(self):
        msg = AssistantMessage(
            content=[ToolCall(id="tc-1", name="search", arguments={"q": "test"})],
            stop_reason="tool_use",
        )
        result = OpenAIProvider._convert_message(msg)
        assert result["role"] == "assistant"
        assert result["content"] is None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "tc-1"
        assert result["tool_calls"][0]["function"]["name"] == "search"

    def test_convert_tool_result(self):
        msg = ToolResultMessage(
            tool_call_id="tc-1",
            tool_name="search",
            content=[TextContent(text="result")],
        )
        result = OpenAIProvider._convert_message(msg)
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "tc-1"
        assert result["content"] == "result"


class TestOpenAIImageConversion:
    def test_user_message_with_image(self):
        msg = UserMessage(
            content=[
                TextContent(text="What's in this image?"),
                ImageContent(source="base64data", media_type="image/png"),
            ]
        )
        result = OpenAIProvider._convert_message(msg)
        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 2
        assert result["content"][0] == {"type": "text", "text": "What's in this image?"}
        assert result["content"][1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,base64data"},
        }

    def test_user_message_text_only_stays_simple(self):
        msg = UserMessage(content=[TextContent(text="hello")])
        result = OpenAIProvider._convert_message(msg)
        assert result["role"] == "user"
        assert result["content"] == "hello"


class TestOpenAIToolConversion:
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
        result = OpenAIProvider._convert_tool(td)
        assert result["type"] == "function"
        assert result["function"]["name"] == "search"
        assert result["function"]["parameters"]["type"] == "object"


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


class TestOpenAIStreamText:
    """Simple text streaming: start -> text_start -> text_delta -> text_end -> done."""

    @pytest.mark.asyncio
    async def test_stream_simple_text(self):
        chunks = [
            _make_chunk(id="chatcmpl-abc", content="Hello"),
            _make_chunk(content=" world"),
            _make_chunk(finish_reason="stop"),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_async_iter(chunks)
            )

            provider = OpenAIProvider(api_key="test-key")
            provider._client = mock_client

            model = _model()
            ms = await provider.stream(
                model, [UserMessage(content=[TextContent(text="hi")])]
            )

            events = []
            async for event in ms:
                events.append(event)

            result = await ms.result()

            types = [e.type for e in events]
            assert "start" in types
            assert "text_start" in types
            assert "text_delta" in types
            assert "text_end" in types
            assert "done" in types

            assert result.stop_reason == "stop"
            assert len(result.content) == 1
            assert isinstance(result.content[0], TextContent)
            assert result.content[0].text == "Hello world"

    @pytest.mark.asyncio
    async def test_text_delta_events_carry_fragments(self):
        """Each text_delta event should carry the individual fragment."""
        chunks = [
            _make_chunk(content="A"),
            _make_chunk(content="B"),
            _make_chunk(content="C"),
            _make_chunk(finish_reason="stop"),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_async_iter(chunks)
            )

            provider = OpenAIProvider(api_key="test-key")
            provider._client = mock_client

            ms = await provider.stream(
                _model(), [UserMessage(content=[TextContent(text="go")])]
            )

            deltas = []
            async for event in ms:
                if event.type == "text_delta":
                    deltas.append(event.delta)

            result = await ms.result()
            assert deltas == ["A", "B", "C"]
            assert result.content[0].text == "ABC"


class TestOpenAIStreamToolCall:
    """Tool-call streaming: toolcall_start -> toolcall_delta -> toolcall_end."""

    @pytest.mark.asyncio
    async def test_stream_tool_call(self):
        chunks = [
            # First chunk introduces the tool call
            _make_chunk(
                tool_calls=[_tc_delta(0, id="call_1", name="search", arguments="")]
            ),
            # Argument fragments
            _make_chunk(tool_calls=[_tc_delta(0, arguments='{"q":')]),
            _make_chunk(tool_calls=[_tc_delta(0, arguments=' "test"}')]),
            # Finish
            _make_chunk(finish_reason="tool_calls"),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_async_iter(chunks)
            )

            provider = OpenAIProvider(api_key="test-key")
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

            ms = await provider.stream(
                _model(),
                [UserMessage(content=[TextContent(text="search")])],
                tools=tools,
            )

            events = []
            async for event in ms:
                events.append(event)

            result = await ms.result()

            types = [e.type for e in events]
            assert "toolcall_start" in types
            assert "toolcall_delta" in types
            assert "toolcall_end" in types

            assert result.stop_reason == "tool_use"
            assert len(result.content) >= 1
            tc = [c for c in result.content if isinstance(c, ToolCall)][0]
            assert tc.name == "search"
            assert tc.id == "call_1"
            assert tc.arguments == {"q": "test"}


class TestOpenAIStreamTextThenTool:
    """Text followed by tool call: text_end must fire before toolcall_start."""

    @pytest.mark.asyncio
    async def test_text_end_before_toolcall_start(self):
        chunks = [
            _make_chunk(content="Let me search."),
            _make_chunk(
                tool_calls=[_tc_delta(0, id="call_2", name="search", arguments="")]
            ),
            _make_chunk(tool_calls=[_tc_delta(0, arguments='{"q": "x"}')]),
            _make_chunk(finish_reason="tool_calls"),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_async_iter(chunks)
            )

            provider = OpenAIProvider(api_key="test-key")
            provider._client = mock_client

            ms = await provider.stream(
                _model(), [UserMessage(content=[TextContent(text="go")])]
            )

            types = []
            async for event in ms:
                types.append(event.type)

            result = await ms.result()

            # text_end must appear before toolcall_start
            assert "text_end" in types
            assert "toolcall_start" in types
            text_end_idx = types.index("text_end")
            tc_start_idx = types.index("toolcall_start")
            assert text_end_idx < tc_start_idx

            # Both text and tool call in content
            assert any(isinstance(c, TextContent) for c in result.content)
            assert any(isinstance(c, ToolCall) for c in result.content)


class TestOpenAIStreamKwargsBuilding:
    """Verify kwargs construction: system_prompt, tools, on_response hook."""

    @pytest.mark.asyncio
    async def test_system_prompt_prepended(self):
        chunks = [_make_chunk(finish_reason="stop")]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_async_iter(chunks)
            )

            provider = OpenAIProvider(api_key="test-key")
            provider._client = mock_client

            ms = await provider.stream(
                _model(),
                [UserMessage(content=[TextContent(text="hi")])],
                system_prompt="You are helpful.",
            )
            async for _ in ms:
                pass
            await ms.result()

            call_kwargs = mock_client.chat.completions.create.call_args[1]
            msgs = call_kwargs["messages"]
            assert msgs[0] == {"role": "system", "content": "You are helpful."}
            assert msgs[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_tools_in_kwargs(self):
        chunks = [_make_chunk(finish_reason="stop")]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_async_iter(chunks)
            )

            provider = OpenAIProvider(api_key="test-key")
            provider._client = mock_client

            tools = [
                ToolDefinition(
                    name="calc",
                    description="Calculator",
                    parameters={"type": "object", "properties": {}},
                )
            ]

            ms = await provider.stream(
                _model(),
                [UserMessage(content=[TextContent(text="1+1")])],
                tools=tools,
            )
            async for _ in ms:
                pass
            await ms.result()

            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert "tools" in call_kwargs
            assert call_kwargs["tools"][0]["function"]["name"] == "calc"

    @pytest.mark.asyncio
    async def test_on_response_hook_called(self):
        """When response has a .response attr, invoke_on_response should fire."""
        chunks = [_make_chunk(finish_reason="stop")]

        mock_http = SimpleNamespace(status_code=200, headers={"x-req-id": "abc"})

        class MockStreamResponse:
            """Async iterable with a .response attribute."""

            def __init__(self, items, response):
                self._items = items
                self.response = response

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._items:
                    raise StopAsyncIteration
                return self._items.pop(0)

        mock_response = MockStreamResponse(list(chunks), mock_http)

        on_response_calls = []

        async def on_response(resp, model):
            on_response_calls.append((resp.status, resp.headers))

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            provider = OpenAIProvider(api_key="test-key")
            provider._client = mock_client

            ms = await provider.stream(
                _model(),
                [UserMessage(content=[TextContent(text="hi")])],
                options=StreamOptions(on_response=on_response),
            )
            async for _ in ms:
                pass
            await ms.result()

            assert len(on_response_calls) == 1
            assert on_response_calls[0][0] == 200
            assert on_response_calls[0][1]["x-req-id"] == "abc"

    @pytest.mark.asyncio
    async def test_no_on_response_without_http_response(self):
        """When response has no .response attr, on_response should NOT fire."""
        chunks = [_make_chunk(finish_reason="stop")]

        on_response_calls = []

        async def on_response(resp, model):
            on_response_calls.append(resp)

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_async_iter(chunks)
            )

            provider = OpenAIProvider(api_key="test-key")
            provider._client = mock_client

            ms = await provider.stream(
                _model(),
                [UserMessage(content=[TextContent(text="hi")])],
                options=StreamOptions(on_response=on_response),
            )
            async for _ in ms:
                pass
            await ms.result()

            # _async_iter has no .response attribute, so hook should not fire
            assert len(on_response_calls) == 0


class TestOpenAIStreamAbort:
    """Abort signal mid-stream."""

    @pytest.mark.asyncio
    async def test_abort_before_stream_starts(self):
        signal = asyncio.Event()
        signal.set()  # Already aborted

        chunks = [
            _make_chunk(id="chatcmpl-abort", content="Hello"),
            _make_chunk(content=" world"),
            _make_chunk(finish_reason="stop"),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_async_iter(chunks)
            )

            provider = OpenAIProvider(api_key="test-key")
            provider._client = mock_client

            ms = await provider.stream(
                _model(),
                [UserMessage(content=[TextContent(text="hi")])],
                options=StreamOptions(signal=signal),
            )

            events = []
            async for event in ms:
                events.append(event)

            result = await ms.result()
            assert result.stop_reason == "aborted"
            assert "aborted" in (result.error_message or "")

            types = [e.type for e in events]
            assert "error" in types


class TestOpenAIStreamError:
    """Error handling in _produce."""

    @pytest.mark.asyncio
    async def test_api_error(self):
        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=RuntimeError("API down")
            )

            provider = OpenAIProvider(api_key="test-key")
            provider._client = mock_client

            ms = await provider.stream(
                _model(),
                [UserMessage(content=[TextContent(text="hi")])],
            )

            events = []
            async for event in ms:
                events.append(event)

            result = await ms.result()
            assert result.stop_reason == "error"
            assert "API down" in (result.error_message or "")

            types = [e.type for e in events]
            assert "error" in types

    @pytest.mark.asyncio
    async def test_error_result_carries_provider_and_model(self):
        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=RuntimeError("API down")
            )

            provider = OpenAIProvider(api_key="test-key")
            provider._client = mock_client

            ms = await provider.stream(
                _model(),
                [UserMessage(content=[TextContent(text="hi")])],
            )
            error_events = [e async for e in ms if e.type == "error"]
            result = await ms.result()

            assert result.provider_id == "openai"
            assert result.model_id == "gpt-4o"
            assert "openai/gpt-4o" in (result.error_message or "")
            assert error_events and "openai/gpt-4o" in (
                error_events[0].error_message or ""
            )

    @pytest.mark.asyncio
    async def test_error_message_surfaces_underlying_cause(self):
        try:
            raise OSError("Cannot connect to proxy 192.168.1.111:7892")
        except OSError as root:
            api_err = RuntimeError("Connection error.")
            api_err.__cause__ = root

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=api_err)

            provider = OpenAIProvider(api_key="test-key")
            provider._client = mock_client

            ms = await provider.stream(
                _model(),
                [UserMessage(content=[TextContent(text="hi")])],
            )
            async for _ in ms:
                pass
            result = await ms.result()

            assert "Connection error." in (result.error_message or "")
            assert "Cannot connect to proxy 192.168.1.111:7892" in (
                result.error_message or ""
            )

    @pytest.mark.asyncio
    async def test_base_exception_reraised(self):
        """BaseException subclass (non-Exception) should set error result and re-raise."""

        class _CustomInterrupt(BaseException):
            pass

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=_CustomInterrupt("interrupted")
            )

            provider = OpenAIProvider(api_key="test-key")
            provider._client = mock_client

            ms = await provider.stream(
                _model(),
                [UserMessage(content=[TextContent(text="hi")])],
            )

            # Consume the stream events
            events = []
            async for event in ms:
                events.append(event)

            # The result should still be set (error)
            result = await ms.result()
            assert result.stop_reason == "error"
            assert "interrupted" in (result.error_message or "")

            # The underlying task should have re-raised _CustomInterrupt
            assert ms._producer_task is not None
            with pytest.raises(_CustomInterrupt):
                ms._producer_task.result()

    @pytest.mark.asyncio
    async def test_empty_completion_emits_error(self):
        """When OpenAI returns stop with no text and no tool calls, emit EmptyCompletion error."""
        chunks = [
            _make_chunk(finish_reason="stop", content=""),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_async_iter(chunks)
            )

            provider = OpenAIProvider(api_key="test-key")
            provider._client = mock_client

            ms = await provider.stream(
                _model(),
                [UserMessage(content=[TextContent(text="hi")])],
            )

            events = [e async for e in ms]
            result = await ms.result()

            assert result.stop_reason == "error"
            assert result.error_type == "EmptyCompletion"
            assert any(
                e.type == "error" and e.error_type == "EmptyCompletion" for e in events
            )

    @pytest.mark.asyncio
    async def test_empty_choices_skipped(self):
        """Chunks with no choices should be skipped gracefully."""
        chunks = [
            _make_empty_chunk(id="chatcmpl-empty"),  # no choices
            _make_chunk(content="Hello"),
            _make_chunk(finish_reason="stop"),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_async_iter(chunks)
            )

            provider = OpenAIProvider(api_key="test-key")
            provider._client = mock_client

            ms = await provider.stream(
                _model(),
                [UserMessage(content=[TextContent(text="hi")])],
            )

            events = []
            async for event in ms:
                events.append(event)

            result = await ms.result()
            assert result.stop_reason == "stop"
            assert result.content[0].text == "Hello"


class TestOpenAIStreamResponseId:
    """Response ID capture from first chunk."""

    @pytest.mark.asyncio
    async def test_response_id_captured(self):
        chunks = [
            _make_chunk(id="chatcmpl-123", content="Hi"),
            _make_chunk(id="chatcmpl-123", content="!"),
            _make_chunk(finish_reason="stop"),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_async_iter(chunks)
            )

            provider = OpenAIProvider(api_key="test-key")
            provider._client = mock_client

            ms = await provider.stream(
                _model(),
                [UserMessage(content=[TextContent(text="hi")])],
            )
            async for _ in ms:
                pass

            result = await ms.result()
            assert result.response_id == "chatcmpl-123"


class TestOpenAIStreamNoFinishReason:
    """Stream ends without explicit finish_reason."""

    @pytest.mark.asyncio
    async def test_fallback_done_without_finish_reason(self):
        """When all chunks have finish_reason=None and iterator ends,
        the provider should still produce done + set_result."""
        chunks = [
            _make_chunk(content="partial"),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_async_iter(chunks)
            )

            provider = OpenAIProvider(api_key="test-key")
            provider._client = mock_client

            ms = await provider.stream(
                _model(),
                [UserMessage(content=[TextContent(text="hi")])],
            )

            events = []
            async for event in ms:
                events.append(event)

            result = await ms.result()
            types = [e.type for e in events]
            assert "done" in types
            # Result is returned even without a finish_reason
            assert result is not None
            assert result.content[0].text == "partial"


class TestOpenAIConvertMessageEdgeCases:
    """Edge cases in _convert_message and constructor."""

    def test_unknown_message_type_fallback(self):
        """An unrecognized message type should produce a fallback dict."""

        class FakeMessage:
            pass

        result = OpenAIProvider._convert_message(FakeMessage())
        assert result == {"role": "user", "content": ""}

    def test_base_url_constructor(self):
        """base_url kwarg should be passed to AsyncOpenAI."""
        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            OpenAIProvider(api_key="key-123", base_url="https://custom.api/v1")
            mock_openai.assert_called_once_with(
                api_key="key-123", base_url="https://custom.api/v1"
            )

    def test_no_base_url(self):
        """Without base_url, only api_key should be passed."""
        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            OpenAIProvider(api_key="key-only")
            mock_openai.assert_called_once_with(api_key="key-only")

    def test_no_api_key(self):
        """Without api_key, no kwargs should be passed."""
        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            OpenAIProvider()
            mock_openai.assert_called_once_with()


class TestOpenAIStreamFinishReasonLength:
    """finish_reason='length' maps to stop_reason='length'."""

    @pytest.mark.asyncio
    async def test_length_stop_reason(self):
        chunks = [
            _make_chunk(content="truncated text"),
            _make_chunk(finish_reason="length"),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_async_iter(chunks)
            )

            provider = OpenAIProvider(api_key="test-key")
            provider._client = mock_client

            ms = await provider.stream(
                _model(),
                [UserMessage(content=[TextContent(text="write a long essay")])],
            )
            async for _ in ms:
                pass

            result = await ms.result()
            assert result.stop_reason == "length"


class TestOpenAIStreamOnPayload:
    """Test on_payload hook integration."""

    @pytest.mark.asyncio
    async def test_on_payload_modifies_kwargs(self):
        chunks = [_make_chunk(finish_reason="stop")]

        async def on_payload(payload, model):
            payload["temperature"] = 0.5
            return payload

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_async_iter(chunks)
            )

            provider = OpenAIProvider(api_key="test-key")
            provider._client = mock_client

            ms = await provider.stream(
                _model(),
                [UserMessage(content=[TextContent(text="hi")])],
                options=StreamOptions(on_payload=on_payload),
            )
            async for _ in ms:
                pass
            await ms.result()

            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["temperature"] == 0.5


class TestOpenAIStreamMultipleToolCalls:
    """Multiple tool calls in a single response."""

    @pytest.mark.asyncio
    async def test_two_tool_calls(self):
        chunks = [
            # First tool call
            _make_chunk(
                tool_calls=[_tc_delta(0, id="call_a", name="search", arguments="")]
            ),
            _make_chunk(tool_calls=[_tc_delta(0, arguments='{"q": "a"}')]),
            # Second tool call
            _make_chunk(
                tool_calls=[_tc_delta(1, id="call_b", name="fetch", arguments="")]
            ),
            _make_chunk(tool_calls=[_tc_delta(1, arguments='{"url": "b"}')]),
            # Finish
            _make_chunk(finish_reason="tool_calls"),
        ]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_async_iter(chunks)
            )

            provider = OpenAIProvider(api_key="test-key")
            provider._client = mock_client

            ms = await provider.stream(
                _model(),
                [UserMessage(content=[TextContent(text="do both")])],
            )

            events = []
            async for event in ms:
                events.append(event)

            result = await ms.result()

            types = [e.type for e in events]
            assert types.count("toolcall_start") == 2
            assert types.count("toolcall_end") == 2

            tool_calls = [c for c in result.content if isinstance(c, ToolCall)]
            assert len(tool_calls) == 2
            assert tool_calls[0].name == "search"
            assert tool_calls[0].arguments == {"q": "a"}
            assert tool_calls[1].name == "fetch"
            assert tool_calls[1].arguments == {"url": "b"}


# ---------------------------------------------------------------------------
# Typed error wrapping — provider integration
# ---------------------------------------------------------------------------


class _FakeSdkExc(Exception):
    """Minimal fake SDK exception with configurable status_code."""

    __module__ = "openai"  # so the __module__ guard in classify_and_raise allows it

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code


class TestOpenAITypedErrorWrapping:
    """Verify that raw SDK exceptions are re-raised as typed ProviderError subclasses.

    The provider catches raw SDK exceptions, classifies them via classify_and_raise,
    and then the outer BaseException handler converts them to error AssistantMessages.
    The typed exception is available via the response listener's ``exc`` argument.
    """

    @pytest.mark.asyncio
    async def test_rate_limit_429_delivers_rate_limited_to_listener(self) -> None:
        from cubeloop.errors import RateLimited

        exc = _FakeSdkExc("Rate limit exceeded", status_code=429)
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=exc)

        provider = OpenAIProvider(api_key="test-key")
        provider._client = mock_client

        captured_exc: list[BaseException | None] = []

        def on_response(body, model, e):  # type: ignore[misc]
            captured_exc.append(e)

        provider.subscribe_response(on_response)

        ms = await provider.stream(
            _model(), [UserMessage(content=[TextContent(text="hi")])]
        )
        async for _ in ms:
            pass
        result = await ms.result()

        assert result.stop_reason == "error"
        assert len(captured_exc) == 1
        assert isinstance(captured_exc[0], RateLimited)

    @pytest.mark.asyncio
    async def test_context_length_message_delivers_context_length_exceeded(
        self,
    ) -> None:
        from cubeloop.errors import ContextLengthExceeded

        exc = _FakeSdkExc(
            "This model's maximum context length is 128000 tokens.", status_code=400
        )
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=exc)

        provider = OpenAIProvider(api_key="test-key")
        provider._client = mock_client

        captured_exc: list[BaseException | None] = []

        def on_response(body, model, e):  # type: ignore[misc]
            captured_exc.append(e)

        provider.subscribe_response(on_response)

        ms = await provider.stream(
            _model(), [UserMessage(content=[TextContent(text="hi")])]
        )
        async for _ in ms:
            pass
        await ms.result()

        assert len(captured_exc) == 1
        assert isinstance(captured_exc[0], ContextLengthExceeded)
