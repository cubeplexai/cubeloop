from __future__ import annotations

import pytest

from cubeloop.agent.types import (
    AfterToolCallContext,
    AgentContext,
    AgentToolResult,
)
from cubeloop.middleware import ToolResultLimitMiddleware
from cubeloop.middleware.tool_result_limit import (
    DEFAULT_MAX_TOOL_RESULT_CHARS,
    truncate_tool_result_content,
)
from cubeloop.providers.base import (
    AssistantMessage,
    Content,
    ImageContent,
    TextContent,
    ToolCall,
)


def _ctx(
    text: str | list[Content],
    *,
    name: str = "execute",
) -> AfterToolCallContext:
    content: list[Content] = (
        [TextContent(text=text)] if isinstance(text, str) else list(text)
    )
    return AfterToolCallContext(
        assistant_message=AssistantMessage(content=[]),
        tool_call=ToolCall(id="t1", name=name, arguments={}),
        args={},
        result=AgentToolResult(content=content),
        is_error=False,
        context=AgentContext(system_prompt="", messages=[]),
    )


class TestTruncateToolResultContent:
    def test_under_limit_returns_none(self) -> None:
        content = [TextContent(text="hello")]
        assert truncate_tool_result_content(content, max_chars=10) is None

    def test_exact_limit_returns_none(self) -> None:
        content = [TextContent(text="a" * 20)]
        assert truncate_tool_result_content(content, max_chars=20) is None

    def test_over_limit_keeps_prefix_and_appends_notice(self) -> None:
        content = [TextContent(text="abcdefghij")]
        out = truncate_tool_result_content(content, max_chars=4)
        assert out is not None
        assert len(out) == 1
        assert isinstance(out[0], TextContent)
        assert out[0].text.startswith("abcd")
        assert "[truncated:" in out[0].text
        assert "10 characters" in out[0].text
        assert "showing first 4" in out[0].text

    def test_truncates_across_text_blocks(self) -> None:
        content = [
            TextContent(text="aaa"),
            TextContent(text="bbbbbb"),
        ]
        out = truncate_tool_result_content(content, max_chars=5)
        assert out is not None
        texts = [b.text for b in out if isinstance(b, TextContent)]
        assert texts[0] == "aaa"
        assert texts[1].startswith("bb")
        assert "[truncated:" in texts[-1]

    def test_preserves_image_blocks(self) -> None:
        content = [
            ImageContent(source="data:image/png;base64,xx", media_type="image/png"),
            TextContent(text="x" * 50),
        ]
        out = truncate_tool_result_content(content, max_chars=4)
        assert out is not None
        assert isinstance(out[0], ImageContent)
        assert isinstance(out[1], TextContent)
        assert out[1].text.startswith("xxxx")
        assert "[truncated:" in out[1].text

    def test_drops_text_blocks_after_budget_is_exhausted(self) -> None:
        content = [
            TextContent(text="aaaa"),
            TextContent(text="dropped"),
        ]
        out = truncate_tool_result_content(content, max_chars=4)
        assert out is not None
        texts = [b.text for b in out if isinstance(b, TextContent)]
        assert len(texts) == 1
        assert texts[0].startswith("aaaa")
        assert "dropped" not in texts[0]
        assert "[truncated:" in texts[0]

    def test_appends_notice_when_last_kept_block_is_not_text(self) -> None:
        content = [
            TextContent(text="abcdefghij"),
            ImageContent(source="data:image/png;base64,xx", media_type="image/png"),
        ]
        out = truncate_tool_result_content(content, max_chars=4)
        assert out is not None
        assert isinstance(out[0], TextContent)
        assert out[0].text == "abcd"
        assert isinstance(out[1], ImageContent)
        assert isinstance(out[2], TextContent)
        assert out[2].text.startswith("[truncated:")


class TestToolResultLimitMiddleware:
    def test_rejects_non_positive_max_chars(self) -> None:
        with pytest.raises(ValueError, match="max_chars"):
            ToolResultLimitMiddleware(max_chars=0)

    async def test_under_limit_is_noop(self) -> None:
        mw = ToolResultLimitMiddleware(max_chars=20)
        assert await mw.after_tool_call(_ctx("short")) is None

    async def test_over_limit_rewrites_content(self) -> None:
        mw = ToolResultLimitMiddleware(max_chars=8)
        result = await mw.after_tool_call(_ctx("0123456789abcdef"))
        assert result is not None
        assert result.content is not None
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text.startswith("01234567")
        assert "[truncated:" in result.content[0].text
        assert result.details is None

    async def test_excluded_tool_passes_through(self) -> None:
        mw = ToolResultLimitMiddleware(
            max_chars=4,
            exclude_tool_names={"load_skill"},
        )
        huge = "x" * 100
        assert await mw.after_tool_call(_ctx(huge, name="load_skill")) is None
        rewritten = await mw.after_tool_call(_ctx(huge, name="execute"))
        assert rewritten is not None

    async def test_default_cap_matches_exported_constant(self) -> None:
        mw = ToolResultLimitMiddleware()
        assert mw.max_chars == DEFAULT_MAX_TOOL_RESULT_CHARS
        body = "y" * (DEFAULT_MAX_TOOL_RESULT_CHARS + 50)
        result = await mw.after_tool_call(_ctx(body))
        assert result is not None
        assert result.content is not None
        text = result.content[0].text
        assert text.startswith("y" * DEFAULT_MAX_TOOL_RESULT_CHARS)
        assert "[truncated:" in text
