"""compact_history: one-shot compaction outside an agent turn."""

from __future__ import annotations

from typing import Any

from cubeloop.middleware.compaction import compact_history
from cubeloop.middleware.compaction.state import CompactionState
from cubeloop.providers.base import (
    AssistantMessage,
    BoundModel,
    Message,
    Model,
    ReasoningControl,
    StreamOptions,
    TextContent,
    ToolDefinition,
    ToolResultMessage,
    UserMessage,
    synthetic_user_message,
)


def _user(text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)])


class _FakeSummaryProvider:
    def __init__(
        self, *, reply: str = "summary text", raises: Exception | None = None
    ) -> None:
        self.reply = reply
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append({"messages": messages, "system_prompt": system_prompt})
        if self.raises is not None:
            raise self.raises
        return AssistantMessage(content=[TextContent(text=self.reply)])


def _model(provider: _FakeSummaryProvider) -> BoundModel:
    return BoundModel(
        provider=provider,
        spec=Model(id="summary-model", provider_id="faux"),
    )


def _transcript(provider: _FakeSummaryProvider) -> str:
    assert provider.calls
    blocks = provider.calls[0]["messages"][0].content
    return "\n".join(block.text for block in blocks if isinstance(block, TextContent))


async def test_force_compacts_below_the_token_threshold() -> None:
    provider = _FakeSummaryProvider(reply="llm handoff")
    messages = [_user(f"turn {i}") for i in range(6)]

    result = await compact_history(
        messages,
        model=_model(provider),
        keep_tail_tokens=1,
        min_compact_messages=2,
        max_tokens_before_compact=10_000_000,
        force=True,
        fallback_on_error=False,
    )

    assert result.compacted is True
    assert result.reason is None
    assert result.used_fallback is False
    assert result.state is not None
    assert result.state.summary == "llm handoff"
    assert result.state.is_fallback is False
    assert result.boundary == 5
    assert len(provider.calls) == 1


async def test_threshold_skips_the_model_when_not_forced() -> None:
    provider = _FakeSummaryProvider()
    messages = [_user(f"turn {i}") for i in range(6)]

    result = await compact_history(
        messages,
        model=_model(provider),
        keep_tail_tokens=1,
        min_compact_messages=2,
        max_tokens_before_compact=10_000_000,
        force=False,
    )

    assert result.compacted is False
    assert result.reason == "below_threshold"
    assert result.boundary == 0
    assert provider.calls == []


async def test_summarizer_failure_does_not_fallback_when_disabled() -> None:
    provider = _FakeSummaryProvider(raises=RuntimeError("model down"))
    messages = [_user(f"turn {i}") for i in range(6)]

    result = await compact_history(
        messages,
        model=_model(provider),
        keep_tail_tokens=1,
        min_compact_messages=2,
        force=True,
        fallback_on_error=False,
    )

    assert result.compacted is False
    assert result.reason == "summarizer_failed"
    assert result.state is None
    assert result.used_fallback is False


async def test_summarizer_failure_uses_fallback_by_default() -> None:
    provider = _FakeSummaryProvider(raises=RuntimeError("model down"))
    messages = [_user(f"turn {i}") for i in range(6)]

    result = await compact_history(
        messages,
        model=_model(provider),
        keep_tail_tokens=1,
        min_compact_messages=2,
        force=True,
    )

    assert result.compacted is True
    assert result.used_fallback is True
    assert result.state is not None
    assert result.state.is_fallback is True
    assert "LLM summariser unavailable" in result.state.summary


async def test_empty_and_short_histories_do_not_call_the_model() -> None:
    provider = _FakeSummaryProvider()

    empty = await compact_history([], model=_model(provider), force=True)
    assert empty.reason == "empty"
    assert empty.compacted is False

    short = await compact_history(
        [_user("hi"), _user("there")],
        model=_model(provider),
        force=True,
    )
    assert short.reason == "too_short"
    assert provider.calls == []


async def test_no_boundary_when_the_tail_covers_the_history() -> None:
    provider = _FakeSummaryProvider()
    messages = [_user(f"turn {i}") for i in range(6)]

    result = await compact_history(
        messages,
        model=_model(provider),
        keep_tail_tokens=10**9,
        min_compact_messages=2,
        force=True,
    )

    assert result.reason == "no_boundary"
    assert result.compacted is False
    assert provider.calls == []


async def test_trailing_synthetic_user_does_not_consume_the_tail() -> None:
    provider = _FakeSummaryProvider()
    messages = [_user(f"turn {i}") for i in range(4)]
    messages.append(_user("TAIL_REAL"))
    messages.append(synthetic_user_message("control", source="compaction"))

    result = await compact_history(
        messages,
        model=_model(provider),
        keep_tail_tokens=1,
        min_compact_messages=2,
        force=True,
        fallback_on_error=False,
    )

    assert result.compacted is True
    assert result.boundary == 4
    assert "TAIL_REAL" not in _transcript(provider)
    assert "control" not in _transcript(provider)


async def test_pruned_tool_result_is_not_sent_verbatim() -> None:
    provider = _FakeSummaryProvider()
    long_text = "SECRET_TOOL_BODY " + ("x" * 400)
    messages: list[Message] = [
        _user("please read"),
        ToolResultMessage(
            tool_call_id="call-1",
            tool_name="read_file",
            content=[TextContent(text=long_text)],
        ),
        _user("next"),
        _user("tail"),
    ]

    result = await compact_history(
        messages,
        model=_model(provider),
        keep_tail_tokens=1,
        min_compact_messages=2,
        force=True,
        fallback_on_error=False,
    )

    assert result.compacted is True
    transcript = _transcript(provider)
    assert "SECRET_TOOL_BODY" not in transcript
    assert "[read_file]" in transcript


async def test_compressor_text_is_stored_on_the_state() -> None:
    provider = _FakeSummaryProvider()
    messages: list[Message] = [
        _user("please read"),
        ToolResultMessage(
            tool_call_id="call-1",
            tool_name="read_file",
            content=[TextContent(text="x" * 400)],
        ),
        _user("next"),
        _user("tail"),
    ]

    def _keep(message: ToolResultMessage) -> str | None:
        del message
        return "KEEP THIS FACT"

    result = await compact_history(
        messages,
        model=_model(provider),
        keep_tail_tokens=1,
        min_compact_messages=2,
        force=True,
        tool_result_compressor=_keep,
    )

    assert result.state is not None
    assert result.state.preserved_tool_results[0].text == "KEEP THIS FACT"
    assert "KEEP THIS FACT" not in _transcript(provider)


async def test_stale_boundary_is_dropped_before_summarizing(caplog: Any) -> None:
    provider = _FakeSummaryProvider(reply="fresh")
    messages = [_user(f"turn {i}") for i in range(6)]
    stale = CompactionState(summary="old", summarized_message_refs=["sha256:nope"])

    result = await compact_history(
        messages,
        model=_model(provider),
        state=stale,
        boundary=1,
        keep_tail_tokens=1,
        min_compact_messages=2,
        force=True,
        fallback_on_error=False,
    )

    assert result.compacted is True
    assert result.boundary == 5
    assert result.state is not None
    assert result.state.summary == "fresh"
    assert "dropping compaction state" in caplog.text


async def test_missing_state_with_a_boundary_restarts_from_the_prefix() -> None:
    provider = _FakeSummaryProvider(reply="from start")
    messages = [_user(f"turn {i}") for i in range(6)]

    result = await compact_history(
        messages,
        model=_model(provider),
        state=None,
        boundary=3,
        keep_tail_tokens=1,
        min_compact_messages=2,
        force=True,
        fallback_on_error=False,
    )

    assert result.compacted is True
    assert result.state is not None
    assert result.boundary == 5
    assert len(result.state.summarized_message_refs) == result.boundary
    assert "turn 0" in _transcript(provider)


async def test_state_with_zero_boundary_does_not_keep_the_old_refs() -> None:
    provider = _FakeSummaryProvider(reply="fresh")
    messages = [_user(f"turn {i}") for i in range(6)]
    orphan = CompactionState(
        summary="OLD_SUMMARY_TEXT",
        summarized_message_refs=["sha256:old-a", "sha256:old-b"],
    )

    result = await compact_history(
        messages,
        model=_model(provider),
        state=orphan,
        boundary=0,
        keep_tail_tokens=1,
        min_compact_messages=2,
        force=True,
        fallback_on_error=False,
    )

    assert result.compacted is True
    assert result.state is not None
    assert len(result.state.summarized_message_refs) == result.boundary
    assert "OLD_SUMMARY_TEXT" not in provider.calls[0]["system_prompt"]
