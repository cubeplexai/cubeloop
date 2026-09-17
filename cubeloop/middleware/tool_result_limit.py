"""Cap tool-result text before CubeLoop emits ``ToolExecutionEndEvent``.

``after_tool_call`` runs after the tool returns and before the end event is
published, so a truncated result is what the model, the checkpointer, and
host event consumers all see. Hosts whose event bus has a byte budget
(CubePlex projects at 1 MiB) stay inside that budget without failing the
required consumer.

Non-text blocks (images) are preserved. Tools listed in
``exclude_tool_names`` pass through unchanged.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection

from cubeloop.agent.types import AfterToolCallContext, AfterToolCallResult
from cubeloop.middleware.base import Middleware
from cubeloop.providers.base import Content, TextContent

DEFAULT_MAX_TOOL_RESULT_CHARS = 20_000

_TRUNCATION_NOTICE = (
    "\n\n[truncated: tool result was {total} characters; "
    "showing first {kept}. Narrow the tool call and retry.]"
)


def _text_char_count(content: list[Content]) -> int:
    return sum(len(block.text) for block in content if isinstance(block, TextContent))


def truncate_tool_result_content(
    content: list[Content],
    *,
    max_chars: int,
) -> list[Content] | None:
    """Return truncated content, or ``None`` when no truncation is needed."""
    total = _text_char_count(content)
    if total <= max_chars:
        return None

    remaining = max_chars
    truncated: list[Content] = []
    for block in content:
        if not isinstance(block, TextContent):
            truncated.append(block)
            continue
        if remaining <= 0:
            continue
        if len(block.text) <= remaining:
            truncated.append(block)
            remaining -= len(block.text)
            continue
        truncated.append(TextContent(text=block.text[:remaining]))
        remaining = 0

    notice = _TRUNCATION_NOTICE.format(total=total, kept=max_chars)
    if truncated and isinstance(truncated[-1], TextContent):
        truncated[-1] = TextContent(text=truncated[-1].text + notice)
    else:
        truncated.append(TextContent(text=notice.lstrip()))
    return truncated


class ToolResultLimitMiddleware(Middleware):
    """Rewrite oversized tool results in ``after_tool_call``.

    Place this after other ``after_tool_call`` middleware so the truncated
    content is what later composition and ``ToolExecutionEndEvent`` see.
    Under the default last-non-None composer, a later middleware that
    returns a result without ``content`` will drop this rewrite — put this
    last, or use a merging composer.
    """

    def __init__(
        self,
        *,
        max_chars: int = DEFAULT_MAX_TOOL_RESULT_CHARS,
        exclude_tool_names: Collection[str] = (),
    ) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be >= 1")
        self.max_chars = max_chars
        self.exclude_tool_names = frozenset(exclude_tool_names)

    async def after_tool_call(
        self,
        ctx: AfterToolCallContext,
        *,
        signal: asyncio.Event | None = None,
    ) -> AfterToolCallResult | None:
        del signal
        if ctx.tool_call.name in self.exclude_tool_names:
            return None
        truncated = truncate_tool_result_content(
            ctx.result.content,
            max_chars=self.max_chars,
        )
        if truncated is None:
            return None
        return AfterToolCallResult(content=truncated)
