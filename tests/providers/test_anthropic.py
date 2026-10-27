from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cubeloop.providers.anthropic import AnthropicProvider, CacheRetention
from cubeloop.providers.base import (
    ImageContent,
    Model,
    ReasoningControl,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolDefinition,
    UserMessage,
    AssistantMessage,
    ToolResultMessage,
)


class TestAnthropicMessageConversion:
    def test_convert_user_message(self):
        msg = UserMessage(content=[TextContent(text="hello")])
        result = AnthropicProvider._convert_message(msg)
        assert result["role"] == "user"
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "hello"

    def test_convert_assistant_message(self):
        msg = AssistantMessage(content=[TextContent(text="hi")])
        result = AnthropicProvider._convert_message(msg)
        assert result["role"] == "assistant"

    def test_convert_assistant_with_tool_call(self):
        msg = AssistantMessage(
            content=[ToolCall(id="tc-1", name="search", arguments={"q": "test"})],
            stop_reason="tool_use",
        )
        result = AnthropicProvider._convert_message(msg)
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["id"] == "tc-1"
        assert result["content"][0]["name"] == "search"
        assert result["content"][0]["input"] == {"q": "test"}

    def test_convert_tool_result(self):
        msg = ToolResultMessage(
            tool_call_id="tc-1",
            tool_name="search",
            content=[TextContent(text="result")],
        )
        result = AnthropicProvider._convert_message(msg)
        assert result["role"] == "user"
        assert result["content"][0]["type"] == "tool_result"
        assert result["content"][0]["tool_use_id"] == "tc-1"

    def test_convert_assistant_with_empty_content_uses_placeholder(self):
        # A previously-failed turn is persisted as AssistantMessage(content=[],
        # stop_reason="error"). Anthropic rejects messages with no content
        # blocks ("messages.N: all messages must have non-empty content"), so
        # replaying such a history must synthesize at least one block.
        msg = AssistantMessage(content=[], stop_reason="error", error_message="boom")
        result = AnthropicProvider._convert_message(msg)
        assert result["role"] == "assistant"
        assert len(result["content"]) >= 1
        assert all(
            block.get("text") or block.get("input") for block in result["content"]
        )

    def test_strip_trailing_empty_assistants_drops_persisted_error_tail(self):
        # Trailing empty error assistant must be dropped before both the API
        # build and the cache-policy query: Anthropic treats the last assistant
        # as a prefill, and leaving the dropped index visible to the policy
        # would point cache_control at a non-existent message.
        messages = [
            UserMessage(content=[TextContent(text="ask")]),
            AssistantMessage(content=[], stop_reason="error", error_message="boom"),
            AssistantMessage(content=[], stop_reason="error"),
        ]
        stripped = AnthropicProvider._strip_trailing_empty_assistants(messages)
        assert len(stripped) == 1
        assert isinstance(stripped[0], UserMessage)

    def test_strip_trailing_empty_assistants_preserves_real_tail(self):
        messages = [
            UserMessage(content=[TextContent(text="ask")]),
            AssistantMessage(content=[TextContent(text="reply")]),
        ]
        stripped = AnthropicProvider._strip_trailing_empty_assistants(messages)
        assert stripped == messages

    def test_build_api_messages_fills_non_trailing_empty_assistant(self):
        # Mid-history empty error assistant must be filled with a placeholder
        # to preserve the user→assistant→user alternation Anthropic requires.
        messages = [
            UserMessage(content=[TextContent(text="first")]),
            AssistantMessage(content=[], stop_reason="error"),
            UserMessage(content=[TextContent(text="retry")]),
        ]
        api_messages, _ = AnthropicProvider._build_api_messages(messages)
        assert [m["role"] for m in api_messages] == ["user", "assistant", "user"]
        assert len(api_messages[1]["content"]) >= 1


class TestAnthropicToolConversion:
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
        result = AnthropicProvider._convert_tool(td)
        assert result["name"] == "search"
        assert result["description"] == "Search the web"
        assert result["input_schema"]["type"] == "object"


# ---------------------------------------------------------------------------
# Prompt caching tests
# ---------------------------------------------------------------------------


def _make_provider(retention: CacheRetention = "short") -> AnthropicProvider:
    """Create a provider without hitting the network (api_key is unused in tests)."""
    return AnthropicProvider(api_key="test-key", cache_retention=retention)


class TestCacheRetention:
    def test_default_retention_is_short(self):
        provider = AnthropicProvider(api_key="test-key")
        assert provider._cache_retention == "short"

    def test_retention_none_returns_no_cache_control(self):
        provider = _make_provider("none")
        assert provider._get_cache_control() is None

    def test_retention_short_returns_ephemeral(self):
        provider = _make_provider("short")
        cc = provider._get_cache_control()
        assert cc == {"type": "ephemeral"}

    def test_retention_long_returns_ephemeral_with_ttl(self):
        provider = _make_provider("long")
        cc = provider._get_cache_control()
        assert cc == {"type": "ephemeral", "ttl": "1h"}


class TestCacheControlOnMessages:
    """Verify cache_control markers are placed on the last message content block."""

    CACHE_CONTROL = {"type": "ephemeral"}

    def test_cache_control_on_last_user_message_text(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "first"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "reply"}]},
            {"role": "user", "content": [{"type": "text", "text": "second"}]},
        ]
        AnthropicProvider._apply_message_cache_control(msgs, self.CACHE_CONTROL)

        # Only the last message's last block should have cache_control
        assert msgs[-1]["content"][-1]["cache_control"] == self.CACHE_CONTROL
        assert "cache_control" not in msgs[0]["content"][0]
        assert "cache_control" not in msgs[1]["content"][0]

    def test_cache_control_on_last_tool_result(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tc-1",
                        "content": [{"type": "text", "text": "ok"}],
                        "is_error": False,
                    }
                ],
            }
        ]
        AnthropicProvider._apply_message_cache_control(msgs, self.CACHE_CONTROL)
        assert msgs[0]["content"][-1]["cache_control"] == self.CACHE_CONTROL

    def test_cache_control_on_multi_block_message(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "text", "text": "part two"},
                ],
            }
        ]
        AnthropicProvider._apply_message_cache_control(msgs, self.CACHE_CONTROL)
        # Only the last block gets the marker
        assert "cache_control" not in msgs[0]["content"][0]
        assert msgs[0]["content"][1]["cache_control"] == self.CACHE_CONTROL

    def test_cache_control_converts_bare_string_content(self):
        msgs = [{"role": "user", "content": "bare string"}]
        AnthropicProvider._apply_message_cache_control(msgs, self.CACHE_CONTROL)
        # Should have been converted to a list with a text block
        assert isinstance(msgs[0]["content"], list)
        assert msgs[0]["content"][0]["type"] == "text"
        assert msgs[0]["content"][0]["text"] == "bare string"
        assert msgs[0]["content"][0]["cache_control"] == self.CACHE_CONTROL

    def test_empty_messages_is_noop(self):
        msgs: list[dict] = []
        # Should not raise
        AnthropicProvider._apply_message_cache_control(msgs, self.CACHE_CONTROL)
        assert msgs == []

    def test_no_cache_control_when_retention_none(self):
        provider = _make_provider("none")
        assert provider._get_cache_control() is None


class TestCacheControlOnSystemPrompt:
    """Verify that the stream method builds system prompt blocks with cache_control."""

    def test_system_prompt_has_cache_control(self):
        """The system prompt should be sent as a content block with cache_control."""
        provider = _make_provider("short")
        cache_control = provider._get_cache_control()
        # Simulate what stream() does for system_prompt
        system_prompt = "You are a helpful assistant."
        system_block = {
            "type": "text",
            "text": system_prompt,
            **({"cache_control": cache_control} if cache_control else {}),
        }
        assert system_block["cache_control"] == {"type": "ephemeral"}

    def test_system_prompt_no_cache_when_retention_none(self):
        provider = _make_provider("none")
        cache_control = provider._get_cache_control()
        system_prompt = "You are a helpful assistant."
        system_block = {
            "type": "text",
            "text": system_prompt,
            **({"cache_control": cache_control} if cache_control else {}),
        }
        assert "cache_control" not in system_block


class TestCacheControlOnTools:
    """Verify cache_control is applied to the last tool definition."""

    CACHE_CONTROL = {"type": "ephemeral"}

    def _make_tools(self, count: int) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name=f"tool_{i}",
                description=f"Tool {i}",
                parameters={
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                },
            )
            for i in range(count)
        ]

    def test_cache_control_on_last_tool_only(self):
        tools = self._make_tools(3)
        api_tools = [AnthropicProvider._convert_tool(t) for t in tools]
        # Apply cache_control the same way stream() does
        if api_tools:
            api_tools[-1]["cache_control"] = self.CACHE_CONTROL

        assert "cache_control" not in api_tools[0]
        assert "cache_control" not in api_tools[1]
        assert api_tools[2]["cache_control"] == self.CACHE_CONTROL

    def test_single_tool_gets_cache_control(self):
        tools = self._make_tools(1)
        api_tools = [AnthropicProvider._convert_tool(t) for t in tools]
        if api_tools:
            api_tools[-1]["cache_control"] = self.CACHE_CONTROL

        assert api_tools[0]["cache_control"] == self.CACHE_CONTROL

    def test_no_cache_control_when_retention_none(self):
        provider = _make_provider("none")
        cache_control = provider._get_cache_control()
        tools = self._make_tools(2)
        api_tools = [AnthropicProvider._convert_tool(t) for t in tools]
        if cache_control and api_tools:
            api_tools[-1]["cache_control"] = cache_control

        # With retention="none", no tool should have cache_control
        for tool in api_tools:
            assert "cache_control" not in tool


# ---------------------------------------------------------------------------
# Streaming tests (with mocked Anthropic client)
# ---------------------------------------------------------------------------


def _anthropic_model(*, reasoning: bool = False, max_tokens: int = 8192) -> Model:
    return Model(
        id="claude-sonnet-4-20250514" if not reasoning else "claude-sonnet-4-20250514",
        provider_id="anthropic",
        api="anthropic",
        reasoning=reasoning,
        max_tokens=max_tokens,
    )


def _make_event(type: str, **kwargs) -> SimpleNamespace:
    """Create a mock Anthropic streaming event."""
    return SimpleNamespace(type=type, **kwargs)


def _make_content_block(type: str, **kwargs) -> SimpleNamespace:
    """Create a mock content block (for content_block_start events)."""
    return SimpleNamespace(type=type, **kwargs)


def _make_final_message(
    *,
    id: str = "msg_123",
    content: list | None = None,
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 20,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> SimpleNamespace:
    """Create a mock Anthropic Message returned by get_final_message()."""
    return SimpleNamespace(
        id=id,
        content=content or [],
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
        ),
    )


class _MockAnthropicStream:
    """Mock for the async context manager returned by client.messages.stream().

    When used as ``async with client.messages.stream(**kw) as stream:``, the
    ``__aenter__`` returns *self*.  The instance is async-iterable over a
    pre-defined list of SDK events, and exposes ``.response`` and
    ``.get_final_message()`` like the real SDK stream.
    """

    def __init__(
        self,
        events: list,
        final_message: SimpleNamespace | None = None,
        status_code: int = 200,
        headers: dict | None = None,
    ) -> None:
        self._events = events
        self._final_message = final_message or _make_final_message()
        self.response = SimpleNamespace(
            status_code=status_code,
            headers=headers or {"x-request-id": "req-abc"},
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self._iter_events()

    async def _iter_events(self):
        for ev in self._events:
            yield ev

    async def get_final_message(self):
        return self._final_message


def _inject_mock_stream(
    provider: AnthropicProvider, mock_stream: _MockAnthropicStream
) -> MagicMock:
    """Replace provider._client with a mock whose messages.stream returns *mock_stream*."""
    mock_client = MagicMock()
    mock_client.messages = MagicMock()
    mock_client.messages.stream = MagicMock(return_value=mock_stream)
    provider._client = mock_client
    return mock_client


class TestAnthropicStreamText:
    """Test text streaming end-to-end."""

    @pytest.mark.asyncio
    async def test_stream_simple_text(self):
        events = [
            _make_event(
                "content_block_start",
                index=0,
                content_block=_make_content_block("text"),
            ),
            _make_event(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(text="Hello "),
            ),
            _make_event(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(text="world!"),
            ),
            _make_event("content_block_stop", index=0),
        ]
        final = _make_final_message(
            content=[SimpleNamespace(type="text", text="Hello world!")],
            stop_reason="end_turn",
            input_tokens=10,
            output_tokens=5,
        )
        mock_stream = _MockAnthropicStream(events, final)

        provider = _make_provider("none")
        _inject_mock_stream(provider, mock_stream)

        model = _anthropic_model()
        ms = await provider.stream(
            model, [UserMessage(content=[TextContent(text="hi")])]
        )

        stream_events = []
        async for event in ms:
            stream_events.append(event)

        result = await ms.result()

        event_types = [e.type for e in stream_events]
        assert "start" in event_types
        assert "text_start" in event_types
        assert "text_delta" in event_types
        assert "text_end" in event_types
        assert "done" in event_types

        # Check that the text deltas accumulated correctly in partial
        text_deltas = [e for e in stream_events if e.type == "text_delta"]
        assert len(text_deltas) == 2
        assert text_deltas[0].delta == "Hello "
        assert text_deltas[1].delta == "world!"

        # Check final message
        assert result.stop_reason == "stop"
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "Hello world!"

        # Check usage
        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

        # Check metadata
        assert result.response_id == "msg_123"
        assert result.provider_id == "anthropic"
        assert result.model_id == model.id


class TestAnthropicStreamThinking:
    """Test thinking + text streaming."""

    @pytest.mark.asyncio
    async def test_stream_thinking_then_text(self):
        events = [
            # Thinking block
            _make_event(
                "content_block_start",
                index=0,
                content_block=_make_content_block("thinking"),
            ),
            _make_event(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(thinking="Let me think..."),
            ),
            _make_event("content_block_stop", index=0),
            # Text block
            _make_event(
                "content_block_start",
                index=1,
                content_block=_make_content_block("text"),
            ),
            _make_event(
                "content_block_delta",
                index=1,
                delta=SimpleNamespace(text="Answer"),
            ),
            _make_event("content_block_stop", index=1),
        ]
        final = _make_final_message(
            content=[
                SimpleNamespace(type="thinking", thinking="Let me think..."),
                SimpleNamespace(type="text", text="Answer"),
            ],
            stop_reason="end_turn",
        )
        mock_stream = _MockAnthropicStream(events, final)

        provider = _make_provider("none")
        _inject_mock_stream(provider, mock_stream)

        model = _anthropic_model(reasoning=True)
        ms = await provider.stream(
            model,
            [UserMessage(content=[TextContent(text="think about this")])],
            options=StreamOptions(
                reasoning=ReasoningControl(mode="on", effort="medium")
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
        assert result.content[0].thinking == "Let me think..."
        assert isinstance(result.content[1], TextContent)
        assert result.content[1].text == "Answer"


class TestAnthropicStreamToolCall:
    """Test tool use streaming."""

    @pytest.mark.asyncio
    async def test_stream_tool_call(self):
        events = [
            _make_event(
                "content_block_start",
                index=0,
                content_block=_make_content_block(
                    "tool_use", id="toolu_abc", name="search"
                ),
            ),
            _make_event(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(partial_json='{"q": '),
            ),
            _make_event(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(partial_json='"test"}'),
            ),
            _make_event("content_block_stop", index=0),
        ]
        final = _make_final_message(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="toolu_abc",
                    name="search",
                    input={"q": "test"},
                ),
            ],
            stop_reason="tool_use",
        )
        mock_stream = _MockAnthropicStream(events, final)

        provider = _make_provider("none")
        _inject_mock_stream(provider, mock_stream)

        tools = [
            ToolDefinition(
                name="search",
                description="Search the web",
                parameters={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            )
        ]

        model = _anthropic_model()
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
        assert result.content[0].id == "toolu_abc"
        assert result.content[0].arguments == {"q": "test"}

    @pytest.mark.asyncio
    async def test_toolcall_end_partial_carries_parsed_arguments(self):
        """The streaming toolcall_end event must expose the parsed arguments.

        Live consumers (the SSE tool_call event → frontend tool cards showing
        the web_search query / web_fetch url) read the toolcall_end partial. If
        args are only assembled in the final message, the card stays blank until
        reload.
        """
        events = [
            _make_event(
                "content_block_start",
                index=0,
                content_block=_make_content_block(
                    "tool_use", id="toolu_abc", name="search"
                ),
            ),
            _make_event(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(partial_json='{"q": '),
            ),
            _make_event(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(partial_json='"test"}'),
            ),
            _make_event("content_block_stop", index=0),
        ]
        final = _make_final_message(
            content=[
                SimpleNamespace(
                    type="tool_use", id="toolu_abc", name="search", input={"q": "test"}
                )
            ],
            stop_reason="tool_use",
        )
        provider = _make_provider("none")
        _inject_mock_stream(provider, _MockAnthropicStream(events, final))

        ms = await provider.stream(
            _anthropic_model(),
            [UserMessage(content=[TextContent(text="search for test")])],
        )
        stream_events = []
        async for event in ms:
            stream_events.append(event)
        await ms.result()

        end_events = [e for e in stream_events if e.type == "toolcall_end"]
        assert end_events, "expected a toolcall_end event"
        partial = end_events[-1].partial
        tool_blocks = [b for b in partial.content if isinstance(b, ToolCall)]
        assert tool_blocks, "toolcall_end partial should contain the ToolCall block"
        assert tool_blocks[-1].arguments == {"q": "test"}

    @pytest.mark.asyncio
    async def test_toolcall_end_partial_tolerates_malformed_args_json(self):
        """Incomplete/garbled partial_json falls back to {} without raising;
        the final message still carries authoritative args."""
        events = [
            _make_event(
                "content_block_start",
                index=0,
                content_block=_make_content_block(
                    "tool_use", id="toolu_x", name="search"
                ),
            ),
            _make_event(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(
                    partial_json='{"q": "te'
                ),  # truncated, invalid JSON
            ),
            _make_event("content_block_stop", index=0),
        ]
        final = _make_final_message(
            content=[
                SimpleNamespace(
                    type="tool_use", id="toolu_x", name="search", input={"q": "te"}
                )
            ],
            stop_reason="tool_use",
        )
        provider = _make_provider("none")
        _inject_mock_stream(provider, _MockAnthropicStream(events, final))

        ms = await provider.stream(
            _anthropic_model(),
            [UserMessage(content=[TextContent(text="x")])],
        )
        stream_events = []
        async for event in ms:
            stream_events.append(event)
        await ms.result()

        end_events = [e for e in stream_events if e.type == "toolcall_end"]
        assert end_events
        tool_blocks = [
            b for b in end_events[-1].partial.content if isinstance(b, ToolCall)
        ]
        assert tool_blocks[-1].arguments == {}


class TestAnthropicStreamKwargsBuilding:
    """Verify kwargs sent to the Anthropic SDK."""

    @pytest.mark.asyncio
    async def test_system_prompt_in_kwargs(self):
        """When system_prompt is provided, kwargs['system'] should be set with cache_control."""
        events = []
        final = _make_final_message(content=[])
        mock_stream = _MockAnthropicStream(events, final)

        provider = _make_provider("short")
        mock_client = _inject_mock_stream(provider, mock_stream)

        model = _anthropic_model()
        ms = await provider.stream(
            model,
            [UserMessage(content=[TextContent(text="hi")])],
            system_prompt="You are helpful.",
        )
        async for _ in ms:
            pass
        await ms.result()

        call_kwargs = mock_client.messages.stream.call_args[1]
        assert "system" in call_kwargs
        system_blocks = call_kwargs["system"]
        assert len(system_blocks) == 1
        assert system_blocks[0]["type"] == "text"
        assert system_blocks[0]["text"] == "You are helpful."
        assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_system_prompt_no_cache_with_retention_none(self):
        """With cache_retention='none', system prompt should not have cache_control."""
        events = []
        final = _make_final_message(content=[])
        mock_stream = _MockAnthropicStream(events, final)

        provider = _make_provider("none")
        mock_client = _inject_mock_stream(provider, mock_stream)

        model = _anthropic_model()
        ms = await provider.stream(
            model,
            [UserMessage(content=[TextContent(text="hi")])],
            system_prompt="You are helpful.",
        )
        async for _ in ms:
            pass
        await ms.result()

        call_kwargs = mock_client.messages.stream.call_args[1]
        system_blocks = call_kwargs["system"]
        assert "cache_control" not in system_blocks[0]

    @pytest.mark.asyncio
    async def test_tools_in_kwargs_with_cache_control(self):
        """When tools are provided, the last tool should get cache_control."""
        events = []
        final = _make_final_message(content=[])
        mock_stream = _MockAnthropicStream(events, final)

        provider = _make_provider("short")
        mock_client = _inject_mock_stream(provider, mock_stream)

        tools = [
            ToolDefinition(
                name="search",
                description="Search",
                parameters={"type": "object", "properties": {}},
            ),
            ToolDefinition(
                name="fetch",
                description="Fetch",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        model = _anthropic_model()
        ms = await provider.stream(
            model,
            [UserMessage(content=[TextContent(text="hi")])],
            tools=tools,
        )
        async for _ in ms:
            pass
        await ms.result()

        call_kwargs = mock_client.messages.stream.call_args[1]
        api_tools = call_kwargs["tools"]
        assert len(api_tools) == 2
        assert "cache_control" not in api_tools[0]
        assert api_tools[1]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_thinking_in_kwargs(self):
        """When thinking is enabled, kwargs['thinking'] should be set."""
        events = []
        final = _make_final_message(content=[])
        mock_stream = _MockAnthropicStream(events, final)

        provider = _make_provider("none")
        mock_client = _inject_mock_stream(provider, mock_stream)

        model = _anthropic_model(reasoning=True)
        ms = await provider.stream(
            model,
            [UserMessage(content=[TextContent(text="think")])],
            options=StreamOptions(reasoning=ReasoningControl(mode="on", effort="high")),
        )
        async for _ in ms:
            pass
        await ms.result()

        call_kwargs = mock_client.messages.stream.call_args[1]
        assert "thinking" in call_kwargs
        assert call_kwargs["thinking"]["type"] == "enabled"
        assert call_kwargs["thinking"]["budget_tokens"] > 0

    @pytest.mark.asyncio
    async def test_thinking_disabled_when_off(self):
        """When reasoning is off, Anthropic receives explicit disabled thinking."""
        events = []
        final = _make_final_message(content=[])
        mock_stream = _MockAnthropicStream(events, final)

        provider = _make_provider("none")
        mock_client = _inject_mock_stream(provider, mock_stream)

        model = _anthropic_model()
        ms = await provider.stream(
            model,
            [UserMessage(content=[TextContent(text="hi")])],
            options=StreamOptions(reasoning=ReasoningControl(mode="off")),
        )
        async for _ in ms:
            pass
        await ms.result()

        call_kwargs = mock_client.messages.stream.call_args[1]
        assert call_kwargs["thinking"] == {"type": "disabled"}

    @pytest.mark.asyncio
    async def test_on_response_callback_invoked(self):
        """The on_response callback should receive HTTP metadata from the stream."""
        events = []
        final = _make_final_message(content=[])
        mock_stream = _MockAnthropicStream(
            events, final, status_code=200, headers={"x-request-id": "req-xyz"}
        )

        provider = _make_provider("none")
        _inject_mock_stream(provider, mock_stream)

        captured = {}

        async def on_response(resp, model):
            captured["status"] = resp.status
            captured["headers"] = resp.headers

        model = _anthropic_model()
        ms = await provider.stream(
            model,
            [UserMessage(content=[TextContent(text="hi")])],
            options=StreamOptions(on_response=on_response),
        )
        async for _ in ms:
            pass
        await ms.result()

        assert captured["status"] == 200
        assert captured["headers"]["x-request-id"] == "req-xyz"

    @pytest.mark.asyncio
    async def test_trailing_empty_assistant_dropped_and_cache_marker_preserved(self):
        """Stripping the trailing empty error assistant must keep cache_control
        on the new last message, not silently drop it because the policy was
        queried against the original (un-trimmed) message list."""
        events = []
        final = _make_final_message(content=[])
        mock_stream = _MockAnthropicStream(events, final)

        provider = _make_provider("short")
        mock_client = _inject_mock_stream(provider, mock_stream)

        history = [
            UserMessage(content=[TextContent(text="ask")]),
            AssistantMessage(content=[TextContent(text="reply")]),
            AssistantMessage(content=[], stop_reason="error", error_message="boom"),
        ]
        ms = await provider.stream(_anthropic_model(), history)
        async for _ in ms:
            pass
        await ms.result()

        sent = mock_client.messages.stream.call_args[1]["messages"]
        assert len(sent) == 2  # trailing empty error assistant dropped
        # Default cache policy marks the now-last message; cache_control must
        # still reach the wire.
        last_block = sent[-1]["content"][-1]
        assert last_block.get("cache_control") == {"type": "ephemeral"}


class TestAnthropicParallelToolResults:
    """Parallel tool calls: all tool_results must land in ONE user message.

    Anthropic rejects a request where an assistant turn emits N parallel
    tool_use blocks but the immediately-following user message carries fewer
    than N tool_result blocks. cubepi stores one ToolResultMessage per call,
    so the provider must coalesce consecutive ones.
    """

    @staticmethod
    def _history():
        return [
            UserMessage(content=[TextContent(text="weather?")]),
            AssistantMessage(
                content=[
                    ToolCall(id="call_0", name="fetch", arguments={"u": "a"}),
                    ToolCall(id="call_1", name="fetch", arguments={"u": "b"}),
                ],
                stop_reason="tool_use",
            ),
            ToolResultMessage(
                tool_call_id="call_0",
                tool_name="fetch",
                content=[TextContent(text="result a")],
            ),
            ToolResultMessage(
                tool_call_id="call_1",
                tool_name="fetch",
                content=[TextContent(text="result b")],
            ),
        ]

    @pytest.mark.asyncio
    async def test_consecutive_tool_results_merged_into_one_user_message(self):
        mock_stream = _MockAnthropicStream([], _make_final_message(content=[]))
        provider = _make_provider("none")
        mock_client = _inject_mock_stream(provider, mock_stream)

        ms = await provider.stream(_anthropic_model(), self._history())
        async for _ in ms:
            pass
        await ms.result()

        messages = mock_client.messages.stream.call_args[1]["messages"]
        # user, assistant(2 tool_use), user(2 tool_result) — NOT 4 messages.
        assert len(messages) == 3
        tool_results = messages[2]["content"]
        assert messages[2]["role"] == "user"
        assert [b["tool_use_id"] for b in tool_results] == ["call_0", "call_1"]
        # Every tool_use id has a matching tool_result in the next message.
        tool_use_ids = {b["id"] for b in messages[1]["content"]}
        result_ids = {b["tool_use_id"] for b in tool_results}
        assert tool_use_ids == result_ids

    @pytest.mark.asyncio
    async def test_cache_breakpoint_lands_on_merged_message(self):
        # Default policy marks the last message; after merge it must still be
        # the (single) tool-result user message that gets the breakpoint.
        mock_stream = _MockAnthropicStream([], _make_final_message(content=[]))
        provider = _make_provider("short")
        mock_client = _inject_mock_stream(provider, mock_stream)

        ms = await provider.stream(_anthropic_model(), self._history())
        async for _ in ms:
            pass
        await ms.result()

        messages = mock_client.messages.stream.call_args[1]["messages"]
        assert len(messages) == 3
        # Breakpoint on the last content block of the merged tool-result message.
        assert messages[2]["content"][-1]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_breakpoint_on_interior_result_keeps_its_block(self):
        # A custom policy that targets the FIRST result of a parallel run must
        # mark THAT result's block, not slide to the last block of the merge.
        class _FirstResult:
            def mark_system(self) -> bool:
                return False

            def mark_last_tool(self) -> bool:
                return False

            def message_breakpoint_indices(self, messages):
                return [2]  # the first ToolResultMessage in _history()

        mock_stream = _MockAnthropicStream([], _make_final_message(content=[]))
        provider = AnthropicProvider(
            api_key="x", cache_retention="short", cache_policy=_FirstResult()
        )
        mock_client = _inject_mock_stream(provider, mock_stream)

        ms = await provider.stream(_anthropic_model(), self._history())
        async for _ in ms:
            pass
        await ms.result()

        results = mock_client.messages.stream.call_args[1]["messages"][2]["content"]
        # call_0 (first result) carries the breakpoint; call_1 does not.
        assert results[0]["tool_use_id"] == "call_0"
        assert results[0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in results[1]

    @pytest.mark.asyncio
    async def test_single_tool_result_not_affected(self):
        history = [
            UserMessage(content=[TextContent(text="time?")]),
            AssistantMessage(
                content=[ToolCall(id="call_0", name="now", arguments={})],
                stop_reason="tool_use",
            ),
            ToolResultMessage(
                tool_call_id="call_0",
                tool_name="now",
                content=[TextContent(text="noon")],
            ),
        ]
        mock_stream = _MockAnthropicStream([], _make_final_message(content=[]))
        provider = _make_provider("none")
        mock_client = _inject_mock_stream(provider, mock_stream)

        ms = await provider.stream(_anthropic_model(), history)
        async for _ in ms:
            pass
        await ms.result()

        messages = mock_client.messages.stream.call_args[1]["messages"]
        assert len(messages) == 3
        assert messages[2]["content"][0]["tool_use_id"] == "call_0"

    @pytest.mark.asyncio
    async def test_separate_turns_not_merged(self):
        # Two single-tool turns separated by an assistant message must stay
        # as two distinct user messages — merging only spans a contiguous run.
        history = [
            UserMessage(content=[TextContent(text="q")]),
            AssistantMessage(
                content=[ToolCall(id="c0", name="t", arguments={})],
                stop_reason="tool_use",
            ),
            ToolResultMessage(
                tool_call_id="c0",
                tool_name="t",
                content=[TextContent(text="r0")],
            ),
            AssistantMessage(
                content=[ToolCall(id="c1", name="t", arguments={})],
                stop_reason="tool_use",
            ),
            ToolResultMessage(
                tool_call_id="c1",
                tool_name="t",
                content=[TextContent(text="r1")],
            ),
        ]
        mock_stream = _MockAnthropicStream([], _make_final_message(content=[]))
        provider = _make_provider("none")
        mock_client = _inject_mock_stream(provider, mock_stream)

        ms = await provider.stream(_anthropic_model(), history)
        async for _ in ms:
            pass
        await ms.result()

        messages = mock_client.messages.stream.call_args[1]["messages"]
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant", "user", "assistant", "user"]


class TestAnthropicStreamAbort:
    """Test abort signal handling."""

    @pytest.mark.asyncio
    async def test_abort_signal(self):
        signal = asyncio.Event()
        signal.set()  # Already aborted

        events = [
            _make_event(
                "content_block_start",
                index=0,
                content_block=_make_content_block("text"),
            ),
            _make_event(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(text="Hello"),
            ),
        ]
        final = _make_final_message(content=[])
        mock_stream = _MockAnthropicStream(events, final)

        provider = _make_provider("none")
        _inject_mock_stream(provider, mock_stream)

        model = _anthropic_model()
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
        assert result.error_message == "Request was aborted"

        event_types = [e.type for e in stream_events]
        assert "error" in event_types


class TestAnthropicStreamError:
    """Test error handling in the streaming loop."""

    @pytest.mark.asyncio
    async def test_exception_produces_error_result(self):
        """When the SDK raises Exception, we get an error result."""
        provider = _make_provider("none")
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.stream = MagicMock(side_effect=RuntimeError("API down"))
        provider._client = mock_client

        model = _anthropic_model()
        ms = await provider.stream(
            model, [UserMessage(content=[TextContent(text="hi")])]
        )

        stream_events = []
        async for event in ms:
            stream_events.append(event)

        result = await ms.result()
        assert result.stop_reason == "error"
        assert "API down" in (result.error_message or "")

        event_types = [e.type for e in stream_events]
        assert "error" in event_types

    @pytest.mark.asyncio
    async def test_error_result_carries_provider_and_model(self):
        provider = _make_provider("none")
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.stream = MagicMock(side_effect=RuntimeError("API down"))
        provider._client = mock_client

        model = _anthropic_model()
        ms = await provider.stream(
            model, [UserMessage(content=[TextContent(text="hi")])]
        )
        async for _ in ms:
            pass
        result = await ms.result()

        assert result.provider_id == "anthropic"
        assert result.model_id == model.id
        assert f"anthropic/{model.id}" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_base_exception_reraises(self):
        """When the SDK raises BaseException (not Exception), it should be re-raised on the task."""

        class _FatalError(BaseException):
            """Custom BaseException that won't disrupt the test runner."""

        provider = _make_provider("none")
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.stream = MagicMock(side_effect=_FatalError("fatal"))
        provider._client = mock_client

        model = _anthropic_model()
        ms = await provider.stream(
            model, [UserMessage(content=[TextContent(text="hi")])]
        )

        stream_events = []
        async for event in ms:
            stream_events.append(event)

        result = await ms.result()
        assert result.stop_reason == "error"
        assert "fatal" in (result.error_message or "")

        # The BaseException re-raise path (line 157) causes the task to fail.
        # We verify the task completed with the re-raised exception.
        task = ms._producer_task
        assert task is not None
        assert task.done()
        with pytest.raises(_FatalError):
            task.result()

    @pytest.mark.asyncio
    async def test_empty_completion_emits_error(self):
        """When Anthropic returns end_turn with no content, emit EmptyCompletion error."""
        events: list = []
        final = _make_final_message(
            content=[],
            stop_reason="end_turn",
            input_tokens=10,
            output_tokens=0,
        )
        mock_stream = _MockAnthropicStream(events, final)

        provider = _make_provider("none")
        _inject_mock_stream(provider, mock_stream)

        model = _anthropic_model()
        ms = await provider.stream(
            model, [UserMessage(content=[TextContent(text="hi")])]
        )

        stream_events = [e async for e in ms]
        result = await ms.result()

        assert result.stop_reason == "error"
        assert result.error_type == "EmptyCompletion"
        assert any(
            e.type == "error" and e.error_type == "EmptyCompletion"
            for e in stream_events
        )


class TestAnthropicConvertResponse:
    """Test _convert_response with all block types."""

    def test_text_block(self):
        response = _make_final_message(
            id="msg_resp",
            content=[SimpleNamespace(type="text", text="Hello")],
            stop_reason="end_turn",
            input_tokens=5,
            output_tokens=3,
            cache_read_input_tokens=2,
            cache_creation_input_tokens=1,
        )
        model = _anthropic_model()
        result = AnthropicProvider._convert_response(response, model)

        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "Hello"
        assert result.stop_reason == "stop"
        assert result.usage.input_tokens == 5
        assert result.usage.output_tokens == 3
        assert result.usage.cache_read_tokens == 2
        assert result.usage.cache_write_tokens == 1
        assert result.response_id == "msg_resp"
        assert result.provider_id == "anthropic"
        assert result.model_id == model.id

    def test_thinking_block(self):
        response = _make_final_message(
            content=[SimpleNamespace(type="thinking", thinking="reasoning...")],
            stop_reason="end_turn",
        )
        model = _anthropic_model()
        result = AnthropicProvider._convert_response(response, model)

        assert len(result.content) == 1
        assert isinstance(result.content[0], ThinkingContent)
        assert result.content[0].thinking == "reasoning..."

    def test_tool_use_block(self):
        response = _make_final_message(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="toolu_xyz",
                    name="search",
                    input={"q": "test"},
                ),
            ],
            stop_reason="tool_use",
        )
        model = _anthropic_model()
        result = AnthropicProvider._convert_response(response, model)

        assert len(result.content) == 1
        assert isinstance(result.content[0], ToolCall)
        assert result.content[0].id == "toolu_xyz"
        assert result.content[0].name == "search"
        assert result.content[0].arguments == {"q": "test"}
        assert result.stop_reason == "tool_use"

    def test_max_tokens_stop_reason(self):
        response = _make_final_message(
            content=[SimpleNamespace(type="text", text="truncated")],
            stop_reason="max_tokens",
        )
        model = _anthropic_model()
        result = AnthropicProvider._convert_response(response, model)
        assert result.stop_reason == "length"

    def test_unknown_stop_reason_passes_through(self):
        response = _make_final_message(
            content=[SimpleNamespace(type="text", text="x")],
            stop_reason="some_new_reason",
        )
        model = _anthropic_model()
        result = AnthropicProvider._convert_response(response, model)
        assert result.stop_reason == "some_new_reason"

    def test_mixed_content_blocks(self):
        response = _make_final_message(
            content=[
                SimpleNamespace(type="thinking", thinking="hmm"),
                SimpleNamespace(type="text", text="answer"),
                SimpleNamespace(
                    type="tool_use", id="t1", name="run", input={"cmd": "ls"}
                ),
            ],
            stop_reason="tool_use",
        )
        model = _anthropic_model()
        result = AnthropicProvider._convert_response(response, model)

        assert len(result.content) == 3
        assert isinstance(result.content[0], ThinkingContent)
        assert isinstance(result.content[1], TextContent)
        assert isinstance(result.content[2], ToolCall)

    def test_cache_tokens_default_to_zero(self):
        """When cache token attributes are missing, they should default to 0."""
        response = SimpleNamespace(
            id="msg_1",
            content=[SimpleNamespace(type="text", text="hi")],
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=5,
                output_tokens=3,
                # No cache_read_input_tokens or cache_creation_input_tokens
            ),
        )
        model = _anthropic_model()
        result = AnthropicProvider._convert_response(response, model)
        assert result.usage.cache_read_tokens == 0
        assert result.usage.cache_write_tokens == 0


class TestAnthropicConvertMessageExtended:
    """Test _convert_message for ImageContent, ThinkingContent, and unknown types."""

    def test_image_content(self):
        msg = UserMessage(
            content=[
                TextContent(text="Describe this"),
                ImageContent(source="base64data", media_type="image/png"),
            ]
        )
        result = AnthropicProvider._convert_message(msg)

        assert result["role"] == "user"
        assert len(result["content"]) == 2
        assert result["content"][0] == {"type": "text", "text": "Describe this"}
        assert result["content"][1] == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "base64data",
            },
        }

    def test_thinking_content_in_assistant(self):
        msg = AssistantMessage(
            content=[
                ThinkingContent(thinking="let me think"),
                TextContent(text="answer"),
            ]
        )
        result = AnthropicProvider._convert_message(msg)

        assert result["role"] == "assistant"
        assert len(result["content"]) == 2
        assert result["content"][0] == {"type": "thinking", "thinking": "let me think"}
        assert result["content"][1] == {"type": "text", "text": "answer"}

    def test_unknown_message_type_fallback(self):
        """A message type that is not User/Assistant/ToolResult should produce empty user content."""

        # Create a mock message that is none of the known types
        class UnknownMessage:
            role = "unknown"
            content = []

        result = AnthropicProvider._convert_message(UnknownMessage())  # type: ignore[arg-type]
        assert result == {"role": "user", "content": []}


class TestAnthropicApplyMessageCacheControlEdge:
    """Test _apply_message_cache_control edge case: empty content (line 186)."""

    CACHE_CONTROL = {"type": "ephemeral"}

    def test_empty_content_list_is_noop(self):
        """Message with empty content list should be skipped (line 186)."""
        msgs = [{"role": "user", "content": []}]
        AnthropicProvider._apply_message_cache_control(msgs, self.CACHE_CONTROL)
        # Content should still be empty, no crash
        assert msgs[0]["content"] == []

    def test_none_content_is_noop(self):
        """Message with no 'content' key should be skipped."""
        msgs = [{"role": "user"}]
        AnthropicProvider._apply_message_cache_control(msgs, self.CACHE_CONTROL)
        assert "content" not in msgs[0] or msgs[0].get("content") is None

    def test_content_is_falsy_but_present(self):
        """Message with content=None should be skipped."""
        msgs = [{"role": "user", "content": None}]
        AnthropicProvider._apply_message_cache_control(msgs, self.CACHE_CONTROL)
        assert msgs[0]["content"] is None


class TestAnthropicBaseUrl:
    """Constructor base_url branch (line 73)."""

    def test_base_url_forwarded_to_async_anthropic(self):
        from unittest.mock import patch as _patch

        with _patch("anthropic.AsyncAnthropic") as mock_anthropic:
            mock_anthropic.return_value = MagicMock()
            AnthropicProvider(api_key="x", base_url="https://proxy.example/anthropic")
            assert mock_anthropic.call_args.kwargs.get("base_url") == (
                "https://proxy.example/anthropic"
            )

    def test_no_base_url_omits_kwarg(self):
        from unittest.mock import patch as _patch

        with _patch("anthropic.AsyncAnthropic") as mock_anthropic:
            mock_anthropic.return_value = MagicMock()
            AnthropicProvider(api_key="x")
            assert "base_url" not in mock_anthropic.call_args.kwargs


# ---------------------------------------------------------------------------
# Typed error wrapping — provider integration
# ---------------------------------------------------------------------------


class _FakeAnthropicExc(Exception):
    """Minimal fake Anthropic SDK exception with configurable status_code."""

    __module__ = "anthropic"  # so the __module__ guard in classify_and_raise allows it

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code


class TestAnthropicTypedErrorWrapping:
    """Verify that raw SDK exceptions are re-raised as typed ProviderError subclasses.

    The provider wraps the entire async with stream block. When the SDK raises,
    classify_and_raise converts it to a typed ProviderError. The outer
    BaseException handler converts it to an error AssistantMessage and passes
    the typed exception to response listeners.
    """

    @pytest.mark.asyncio
    async def test_rate_limit_429_delivers_rate_limited_to_listener(self) -> None:
        from cubeloop.errors import RateLimited

        exc = _FakeAnthropicExc("Rate limit exceeded", status_code=429)
        provider = _make_provider("none")
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.stream = MagicMock(side_effect=exc)
        provider._client = mock_client

        captured_exc: list[BaseException | None] = []

        def on_response(body, model, e):  # type: ignore[misc]
            captured_exc.append(e)

        provider.subscribe_response(on_response)

        ms = await provider.stream(
            _anthropic_model(), [UserMessage(content=[TextContent(text="hi")])]
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

        exc = _FakeAnthropicExc(
            "This model's maximum context length is 200000 tokens.", status_code=400
        )
        provider = _make_provider("none")
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.stream = MagicMock(side_effect=exc)
        provider._client = mock_client

        captured_exc: list[BaseException | None] = []

        def on_response(body, model, e):  # type: ignore[misc]
            captured_exc.append(e)

        provider.subscribe_response(on_response)

        ms = await provider.stream(
            _anthropic_model(), [UserMessage(content=[TextContent(text="hi")])]
        )
        async for _ in ms:
            pass
        await ms.result()

        assert len(captured_exc) == 1
        assert isinstance(captured_exc[0], ContextLengthExceeded)
