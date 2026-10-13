from __future__ import annotations

import asyncio
import logging
from typing import Any

from cubeloop.agent.types import AgentContext
from cubeloop.middleware.compaction import (
    CompactionMiddleware,
    CompactionState,
    ToolResultCompressor,
    _load_state,
)
from cubeloop.middleware.compaction.state import message_ref, message_refs
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
    Usage,
    UserMessage,
)


def _user(text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)])


def _assistant(text: str) -> AssistantMessage:
    return AssistantMessage(content=[TextContent(text=text)])


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
        self.calls.append(
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
        if self.raises is not None:
            raise self.raises
        return AssistantMessage(content=[TextContent(text=self.reply)])


def _make_middleware(
    provider: _FakeSummaryProvider,
    *,
    max_tokens_before: int = 1000,
    keep_tail_tokens: int = 8,
) -> CompactionMiddleware:
    """Tiny tail-token budget (~2 small test messages) approximates the
    old ``keep_recent_messages=2`` behaviour for these scenarios."""
    return CompactionMiddleware(
        summary_model=BoundModel(
            provider=provider,
            spec=Model(id="summary-model", provider_id="faux"),
        ),
        max_tokens_before_compact=max_tokens_before,
        keep_tail_tokens=keep_tail_tokens,
        max_summary_tokens=512,
        min_compact_messages=2,
    )


async def test_under_threshold_returns_existing_compressed_view() -> None:
    provider = _FakeSummaryProvider()
    middleware = _make_middleware(provider, max_tokens_before=100_000)
    messages: list[Message] = [
        _user("old"),
        _assistant("old reply"),
        _user("recent"),
        _assistant("recent reply"),
    ]
    ctx = AgentContext(
        system_prompt="",
        messages=messages,
        extra={
            "compaction": CompactionState(
                summary="old summary",
                summarized_message_refs=message_refs(messages[:2]),
            ).model_dump(),
            "compaction_until_msg_index": 2,
        },
    )

    result = await middleware.transform_context(messages, ctx=ctx)

    assert len(result) == 3
    assert isinstance(result[0], UserMessage)
    assert "old summary" in result[0].content[0].text
    assert result[1:] == messages[2:]
    assert provider.calls == []


def test_load_state_accepts_state_and_ignores_unknown_values() -> None:
    state = CompactionState(summary="cached")

    assert _load_state(state) is state
    assert _load_state("not-state") is None


def test_message_ref_prefers_explicit_message_id() -> None:
    class _MessageWithId:
        id = "msg-123"

        def model_dump(self, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            return {"content": "unused"}

    assert message_ref(_MessageWithId()) == "id:msg-123"  # type: ignore[arg-type]


async def test_over_threshold_writes_json_safe_state_to_ctx_extra() -> None:
    provider = _FakeSummaryProvider(reply="New summary")
    middleware = _make_middleware(provider, max_tokens_before=1)
    signal = asyncio.Event()
    messages: list[Message] = [
        _user("turn 1"),
        _assistant("reply 1"),
        _user("turn 2"),
        _assistant("reply 2"),
        _user("turn 3"),
        _assistant("reply 3"),
    ]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})

    result = await middleware.transform_context(messages, ctx=ctx, signal=signal)

    assert isinstance(ctx.extra["compaction"], dict)
    state = CompactionState.model_validate(ctx.extra["compaction"])
    assert state.summary == "New summary"
    assert ctx.extra["compaction_until_msg_index"] > 0
    assert isinstance(result[0], UserMessage)
    assert provider.calls[0]["options"].signal is signal


async def test_over_threshold_without_safe_boundary_returns_compressed_view() -> None:
    provider = _FakeSummaryProvider()
    middleware = _make_middleware(provider, max_tokens_before=1)
    messages: list[Message] = [
        _user("turn 1"),
        _assistant("reply 1"),
    ]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})

    result = await middleware.transform_context(messages, ctx=ctx)

    assert result == messages
    assert provider.calls == []


async def test_summarizer_failure_writes_fallback_state() -> None:
    """When the LLM raises, build_fallback_summary() runs so the agent
    still gets a compressed view; failures counter increments."""
    provider = _FakeSummaryProvider(raises=RuntimeError("LLM unavailable"))
    middleware = _make_middleware(provider, max_tokens_before=1)
    messages: list[Message] = [
        _user("turn 1"),
        _assistant("reply 1"),
        _user("turn 2"),
        _assistant("reply 2"),
        _user("turn 3"),
        _assistant("reply 3"),
    ]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})

    result = await middleware.transform_context(messages, ctx=ctx)

    # Fallback state was written despite the LLM failure.
    assert "compaction" in ctx.extra
    state = CompactionState.model_validate(ctx.extra["compaction"])
    assert state.is_fallback is True
    # Failure counter incremented.
    assert ctx.extra["compaction_failures"] == 1
    # Result is compressed (summary + tail), not the raw message list.
    assert len(result) < len(messages)


async def test_stale_boundary_larger_than_history_is_ignored() -> None:
    provider = _FakeSummaryProvider()
    middleware = _make_middleware(provider, max_tokens_before=100_000)
    messages: list[Message] = [_user("new question")]
    ctx = AgentContext(
        system_prompt="",
        messages=messages,
        extra={
            "compaction": CompactionState(summary="old summary").model_dump(),
            "compaction_until_msg_index": 10,
        },
    )

    result = await middleware.transform_context(messages, ctx=ctx)

    assert result == messages
    assert "compaction" not in ctx.extra
    assert "compaction_until_msg_index" not in ctx.extra


async def test_stale_boundary_from_replaced_history_is_ignored() -> None:
    provider = _FakeSummaryProvider()
    middleware = _make_middleware(provider, max_tokens_before=100_000)
    old_messages: list[Message] = [
        _user("old turn 1"),
        _assistant("old reply 1"),
        _user("old turn 2"),
        _assistant("old reply 2"),
    ]
    new_messages: list[Message] = [
        _user("new turn 1"),
        _assistant("new reply 1"),
        _user("new turn 2"),
        _assistant("new reply 2"),
    ]
    ctx = AgentContext(
        system_prompt="",
        messages=new_messages,
        extra={
            "compaction": CompactionState(
                summary="old summary",
                summarized_message_refs=message_refs(old_messages[:2]),
            ).model_dump(),
            "compaction_until_msg_index": 2,
        },
    )

    result = await middleware.transform_context(new_messages, ctx=ctx)

    assert result == new_messages
    assert "compaction" not in ctx.extra
    assert "compaction_until_msg_index" not in ctx.extra


async def test_stale_boundary_with_mismatched_ref_count_is_ignored() -> None:
    provider = _FakeSummaryProvider()
    middleware = _make_middleware(provider, max_tokens_before=100_000)
    messages: list[Message] = [
        _user("turn 1"),
        _assistant("reply 1"),
        _user("turn 2"),
        _assistant("reply 2"),
    ]
    ctx = AgentContext(
        system_prompt="",
        messages=messages,
        extra={
            "compaction": CompactionState(
                summary="old summary",
                summarized_message_refs=message_refs(messages[:1]),
            ).model_dump(),
            "compaction_until_msg_index": 2,
        },
    )

    result = await middleware.transform_context(messages, ctx=ctx)

    assert result == messages
    assert "compaction" not in ctx.extra
    assert "compaction_until_msg_index" not in ctx.extra


async def test_malformed_persisted_state_is_cleared() -> None:
    provider = _FakeSummaryProvider()
    middleware = _make_middleware(provider, max_tokens_before=100_000)
    messages: list[Message] = [_user("new question")]
    ctx = AgentContext(
        system_prompt="",
        messages=messages,
        extra={
            "compaction": {},
            "compaction_until_msg_index": 1,
        },
    )

    result = await middleware.transform_context(messages, ctx=ctx)

    assert result == messages
    assert "compaction" not in ctx.extra
    assert "compaction_until_msg_index" not in ctx.extra


# --- Task 6: circuit breaker, anti-thrashing, refs survive pruning ---


async def test_circuit_breaker_opens_after_three_failures() -> None:
    """After 3 LLM failures the breaker opens; 4th call uses fallback (no LLM).

    Messages grow each turn so safe_boundary keeps advancing — without that,
    once a fallback state is persisted, subsequent calls early-return at
    ``new_boundary <= boundary`` and the LLM is never called again.
    """
    provider = _FakeSummaryProvider(raises=RuntimeError("down"))
    middleware = _make_middleware(provider, max_tokens_before=1)
    ctx = AgentContext(system_prompt="", messages=[], extra={})

    messages: list[Message] = []
    for i in range(3):
        messages = [
            *messages,
            _user(f"turn {i * 2}"),
            _assistant(f"reply {i * 2}"),
            _user(f"turn {i * 2 + 1}"),
            _assistant(f"reply {i * 2 + 1}"),
        ]
        await middleware.transform_context(messages, ctx=ctx)
        assert ctx.extra["compaction_failures"] == i + 1
        assert "compaction" in ctx.extra

    calls_before_breaker = len(provider.calls)

    # Turn 4: breaker open → LLM NOT called, fallback still runs.
    messages = [*messages, _user("more"), _assistant("more reply")]
    result = await middleware.transform_context(messages, ctx=ctx)
    assert len(provider.calls) == calls_before_breaker  # no new LLM call
    assert ctx.extra["compaction_failures"] == 3  # frozen at MAX_FAILURES
    assert "compaction" in ctx.extra
    assert len(result) < len(messages)  # still compressed via fallback


async def test_circuit_breaker_resets_on_llm_success() -> None:
    provider = _FakeSummaryProvider(reply="real summary")
    middleware = _make_middleware(provider, max_tokens_before=1)
    messages: list[Message] = [
        _user("turn 1"),
        _assistant("reply 1"),
        _user("turn 2"),
        _assistant("reply 2"),
        _user("turn 3"),
        _assistant("reply 3"),
    ]
    ctx = AgentContext(
        system_prompt="",
        messages=messages,
        extra={"compaction_failures": 2},  # pre-seed: 2 prior failures
    )

    await middleware.transform_context(messages, ctx=ctx)
    assert ctx.extra["compaction_failures"] == 0  # reset after success


async def test_anti_thrashing_skips_compaction_when_guard_tripped() -> None:
    """Guard fires when low_savings_count >= 2, raw history not over 1.5×
    threshold, and new_boundary advance < 8 messages."""
    provider = _FakeSummaryProvider(reply="x")
    # Threshold chosen so tokens_now (~18) EXCEEDS threshold (so the
    # under-threshold fast-path doesn't fire) AND raw_tokens (~18) stays
    # under 1.5 * threshold (so emergency override doesn't bypass the guard).
    middleware = _make_middleware(provider, max_tokens_before=15)
    messages: list[Message] = [
        _user("turn 1"),
        _assistant("reply 1"),
        _user("turn 2"),
        _assistant("reply 2"),
        _user("turn 3"),
        _assistant("reply 3"),
    ]
    ctx = AgentContext(
        system_prompt="",
        messages=messages,
        extra={"compaction_low_savings_count": 2},  # guard tripped
    )

    calls_before = len(provider.calls)
    await middleware.transform_context(messages, ctx=ctx)
    # Neither LLM nor fallback ran — early skip.
    assert len(provider.calls) == calls_before
    assert "compaction" not in ctx.extra


async def test_anti_thrashing_emergency_override_when_raw_history_too_large() -> None:
    """When raw history >= 1.5 × threshold, the guard is overridden."""
    provider = _FakeSummaryProvider(reply="short summary")
    middleware = _make_middleware(provider, max_tokens_before=10)
    # Build a large enough history that raw tokens exceed 1.5 × 10 = 15.
    messages: list[Message] = [
        _user("x" * 100),
        _assistant("y" * 100),
        _user("z" * 100),
        _assistant("w" * 100),
        _user("a" * 100),
        _assistant("b" * 100),
    ]
    ctx = AgentContext(
        system_prompt="",
        messages=messages,
        extra={"compaction_low_savings_count": 2},  # guard tripped
    )

    calls_before = len(provider.calls)
    await middleware.transform_context(messages, ctx=ctx)
    # Emergency override fired despite the guard.
    assert len(provider.calls) > calls_before
    assert "compaction" in ctx.extra


async def test_anti_thrashing_emergency_override_when_real_tokens_too_large() -> None:
    """Cache-aware emergency: char-based raw_tokens stays under 1.5×, but the
    real fill (dominated by cache_read) is over it, so the guard is overridden.
    The old raw-only check would have let the guard keep skipping."""
    provider = _FakeSummaryProvider(reply="short summary")
    middleware = _make_middleware(provider, max_tokens_before=200)
    messages: list[Message] = [
        _user("turn 1"),
        _assistant("reply 1"),
        _user("turn 2"),
        _assistant("reply 2"),
        _user("turn 3"),
        AssistantMessage(
            content=[TextContent(text="reply 3")],
            usage=Usage(cache_read_tokens=500),  # real fill lives in the cache
        ),
    ]
    ctx = AgentContext(
        system_prompt="",
        messages=messages,
        extra={"compaction_low_savings_count": 2},  # guard tripped
    )

    # raw_tokens (~19) is far under 1.5 * 200 = 300, so the raw-only emergency
    # check does NOT fire; the real estimate (>=500, cache-aware) does.
    calls_before = len(provider.calls)
    await middleware.transform_context(messages, ctx=ctx)
    assert len(provider.calls) > calls_before
    assert "compaction" in ctx.extra


async def test_pruned_tool_results_do_not_break_state_refs() -> None:
    """Refs persisted in CompactionState come from ORIGINAL messages even
    when the transcript was built from pre-pruned content. Otherwise the
    next turn would see ref mismatch and clear the state, looping forever."""
    from cubeloop.providers.base import ToolCall, ToolResultMessage

    provider = _FakeSummaryProvider(reply="real summary")
    middleware = _make_middleware(provider, max_tokens_before=1)
    big_result = "tool output line\n" * 200  # > 120 chars → pruner targets it
    messages: list[Message] = [
        _user("audit q"),
        AssistantMessage(
            content=[ToolCall(id="c1", name="audit_query", arguments={"q": "X"})]
        ),
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="audit_query",
            content=[TextContent(text=big_result)],
        ),
        _user("what next?"),
        _assistant("fix it"),
        _user("ok"),
    ]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})

    await middleware.transform_context(messages, ctx=ctx)
    boundary_after_first = ctx.extra["compaction_until_msg_index"]

    # Second turn with same messages: state must validate, not be cleared.
    await middleware.transform_context(messages, ctx=ctx)
    assert ctx.extra.get("compaction_until_msg_index", 0) >= boundary_after_first
    state = CompactionState.model_validate(ctx.extra["compaction"])
    assert state.is_fallback is False  # real LLM summary


def test_keep_recent_messages_no_longer_accepted() -> None:
    """Breaking change — old parameter name raises TypeError."""
    import pytest

    provider = _FakeSummaryProvider()
    with pytest.raises(TypeError):
        CompactionMiddleware(
            summary_model=BoundModel(
                provider=provider,
                spec=Model(id="m", provider_id="faux"),
            ),
            max_tokens_before_compact=100,
            keep_recent_messages=8,  # type: ignore[call-arg]
        )


# --- Task 7: filter-safe prefix, pruner toggle, prompt override ---


def test_summary_prefix_includes_non_instruction_disclaimer() -> None:
    from cubeloop.middleware.compaction import SUMMARY_PREFIX

    text = SUMMARY_PREFIX.lower()
    assert "do not treat" in text or "not instructions" in text
    assert "reference" in text


async def test_prune_tool_outputs_disabled_keeps_full_result_content() -> None:
    """When prune_tool_outputs=False, original tool result content survives.

    Audit-chain agents (finance, compliance) pass False so historical tool
    results stay full-fidelity across compactions.
    """
    from cubeloop.providers.base import ToolCall, ToolResultMessage

    provider = _FakeSummaryProvider(reply="summary")
    middleware = CompactionMiddleware(
        summary_model=BoundModel(
            provider=provider,
            spec=Model(id="m", provider_id="faux"),
        ),
        max_tokens_before_compact=20,
        keep_tail_tokens=8,
        min_compact_messages=2,
        prune_tool_outputs=False,
    )
    big_text = "important audit detail " * 200
    messages: list[Message] = [
        _user("audit q"),
        AssistantMessage(
            content=[ToolCall(id="c1", name="audit_query", arguments={"q": "X"})]
        ),
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="audit_query",
            content=[TextContent(text=big_text)],
        ),
        _user("next?"),
        _assistant("ok"),
        _user("confirm"),
    ]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})
    await middleware.transform_context(messages, ctx=ctx)

    # The original message list is never mutated regardless of toggle.
    assert messages[2].content[0].text == big_text


async def test_summary_prompt_constructor_argument_passthrough() -> None:
    provider = _FakeSummaryProvider(reply="x")
    middleware = CompactionMiddleware(
        summary_model=BoundModel(
            provider=provider,
            spec=Model(id="m", provider_id="faux"),
        ),
        max_tokens_before_compact=1,
        keep_tail_tokens=8,
        min_compact_messages=2,
        summary_prompt="CUSTOM PROMPT BODY",
    )
    messages: list[Message] = [
        _user("turn 1"),
        _assistant("reply 1"),
        _user("turn 2"),
        _assistant("reply 2"),
        _user("turn 3"),
        _assistant("reply 3"),
    ]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})
    await middleware.transform_context(messages, ctx=ctx)

    assert len(provider.calls) == 1
    # No prior summary, so the prompt should be the override verbatim.
    assert provider.calls[0]["system_prompt"] == "CUSTOM PROMPT BODY"


# --- coverage: extra_llm_calls ---


def test_extra_llm_calls_returns_summary_model() -> None:
    provider = _FakeSummaryProvider()
    bound = BoundModel(
        provider=provider,
        spec=Model(id="m", provider_id="faux"),
    )
    middleware = CompactionMiddleware(
        summary_model=bound,
        max_tokens_before_compact=100,
        keep_tail_tokens=8,
    )
    assert middleware.extra_llm_calls() == (bound,)


# --- codex review: effective tail budget clamp ---


async def test_keep_tail_tokens_clamped_below_threshold() -> None:
    """If keep_tail_tokens >= max_tokens_before_compact, the tail must NOT
    swallow the entire history; otherwise compaction can never trigger.
    The middleware clamps the effective tail to half the threshold."""
    provider = _FakeSummaryProvider(reply="summary")
    middleware = CompactionMiddleware(
        summary_model=BoundModel(
            provider=provider,
            spec=Model(id="m", provider_id="faux"),
        ),
        max_tokens_before_compact=10,
        keep_tail_tokens=200,  # configured larger than threshold AND than raw
        min_compact_messages=2,
    )
    messages: list[Message] = [
        _user("turn 1"),
        _assistant("reply 1"),
        _user("turn 2"),
        _assistant("reply 2"),
        _user("turn 3"),
        _assistant("reply 3"),
    ]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})
    await middleware.transform_context(messages, ctx=ctx)

    # Without the clamp, the whole 18-token history would fit in the
    # 200-token tail budget and safe_boundary would return None → no
    # compaction. With the clamp, the effective tail is min(200, 10//2)=5
    # tokens, so compaction fires and writes state.
    assert "compaction" in ctx.extra
    assert len(provider.calls) == 1


# --- codex review: half-open circuit breaker ---


async def test_circuit_breaker_half_open_retries_llm_after_fallback_runs() -> None:
    """After _MAX_FAILURES failures the breaker opens. After
    _HALF_OPEN_AFTER_FALLBACK_RUNS fallback-only runs, the breaker goes
    half-open and the LLM is attempted again. If it succeeds, the breaker
    fully resets."""
    from cubeloop.middleware.compaction import _HALF_OPEN_AFTER_FALLBACK_RUNS

    provider = _FakeSummaryProvider(raises=RuntimeError("down"))
    middleware = _make_middleware(provider, max_tokens_before=1)
    ctx = AgentContext(system_prompt="", messages=[], extra={})

    # Drive the breaker open with 3 LLM failures across growing histories.
    messages: list[Message] = []
    for i in range(3):
        messages = [
            *messages,
            _user(f"turn {i * 2}"),
            _assistant(f"reply {i * 2}"),
            _user(f"turn {i * 2 + 1}"),
            _assistant(f"reply {i * 2 + 1}"),
        ]
        await middleware.transform_context(messages, ctx=ctx)
    assert ctx.extra["compaction_failures"] == 3
    calls_when_open = len(provider.calls)

    # Run fallback-only turns until the breaker should go half-open.
    for i in range(_HALF_OPEN_AFTER_FALLBACK_RUNS):
        messages = [
            *messages,
            _user(f"fbr {i} q"),
            _assistant(f"fbr {i} a"),
        ]
        await middleware.transform_context(messages, ctx=ctx)
    # No LLM call attempted during fallback-only phase.
    assert len(provider.calls) == calls_when_open
    assert ctx.extra["compaction_fallback_runs"] == _HALF_OPEN_AFTER_FALLBACK_RUNS

    # Swap to a working LLM and run one more turn — half-open path triggers,
    # LLM is attempted, succeeds, breaker fully resets.
    provider.raises = None
    provider.reply = "real summary"
    messages = [*messages, _user("recovered q"), _assistant("recovered a")]
    await middleware.transform_context(messages, ctx=ctx)

    assert len(provider.calls) == calls_when_open + 1
    assert ctx.extra["compaction_failures"] == 0
    assert ctx.extra["compaction_fallback_runs"] == 0


async def test_half_open_failure_re_opens_breaker() -> None:
    """Half-open retry that fails snaps the breaker back to MAX_FAILURES."""
    from cubeloop.middleware.compaction import _HALF_OPEN_AFTER_FALLBACK_RUNS

    provider = _FakeSummaryProvider(raises=RuntimeError("down"))
    middleware = _make_middleware(provider, max_tokens_before=1)
    ctx = AgentContext(system_prompt="", messages=[], extra={})

    messages: list[Message] = []
    # 3 failures → breaker opens.
    for i in range(3):
        messages = [
            *messages,
            _user(f"t{i}a"),
            _assistant(f"r{i}a"),
            _user(f"t{i}b"),
            _assistant(f"r{i}b"),
        ]
        await middleware.transform_context(messages, ctx=ctx)
    # Fallback-only runs until half-open is ready.
    for i in range(_HALF_OPEN_AFTER_FALLBACK_RUNS):
        messages = [*messages, _user(f"f{i}"), _assistant(f"a{i}")]
        await middleware.transform_context(messages, ctx=ctx)

    # Half-open turn: LLM still failing.
    calls_before_retry = len(provider.calls)
    messages = [*messages, _user("retry q"), _assistant("retry a")]
    await middleware.transform_context(messages, ctx=ctx)

    # LLM was attempted (half-open allowed one try) and failed.
    assert len(provider.calls) == calls_before_retry + 1
    # Breaker re-opens (failures back to MAX, not MAX+1).
    assert ctx.extra["compaction_failures"] == 3


# --- coverage: _load_int with bad type ---


async def test_corrupt_failure_counter_treated_as_zero() -> None:
    """Non-numeric values in ctx.extra are treated as 0 — defensive parsing.

    Covers both the ``isinstance`` reject path (wrong type) and the
    ``ValueError`` catch path (string that can't be parsed as int).
    """
    provider = _FakeSummaryProvider()
    middleware = _make_middleware(provider, max_tokens_before=1)
    messages: list[Message] = [
        _user("turn 1"),
        _assistant("reply 1"),
        _user("turn 2"),
        _assistant("reply 2"),
        _user("turn 3"),
        _assistant("reply 3"),
    ]
    ctx = AgentContext(
        system_prompt="",
        messages=messages,
        extra={
            "compaction_failures": "not-an-int",  # ValueError in int()
            "compaction_low_savings_count": ["bad", "shape"],  # isinstance reject
            "compaction_fallback_runs": None,  # isinstance reject
        },
    )
    await middleware.transform_context(messages, ctx=ctx)
    # Compaction proceeded as if all counters were 0.
    assert "compaction" in ctx.extra
    assert ctx.extra["compaction_failures"] == 0  # success path


async def test_state_invalidation_clears_guard_counters() -> None:
    """When the persisted summary is invalidated (history replaced), the
    breaker / anti-thrash counters must also reset — otherwise a fresh
    conversation would skip the LLM on its first compaction because the
    previous conversation hit MAX_FAILURES."""
    provider = _FakeSummaryProvider(reply="summary")
    middleware = _make_middleware(provider, max_tokens_before=1)
    # New messages — refs in ctx.extra won't match, so state is invalidated.
    messages: list[Message] = [
        _user("brand new 1"),
        _assistant("reply 1"),
        _user("brand new 2"),
        _assistant("reply 2"),
        _user("brand new 3"),
        _assistant("reply 3"),
    ]
    ctx = AgentContext(
        system_prompt="",
        messages=messages,
        extra={
            "compaction": CompactionState(
                summary="stale from previous conversation",
                summarized_message_refs=["sha256:does-not-match"],
            ).model_dump(),
            "compaction_until_msg_index": 1,
            "compaction_failures": 3,  # stale breaker — would gate LLM
            "compaction_low_savings_count": 2,  # stale guard — would skip
            "compaction_fallback_runs": 99,
        },
    )

    await middleware.transform_context(messages, ctx=ctx)

    # The LLM was called (breaker counter cleared); fresh state written.
    assert len(provider.calls) == 1
    assert ctx.extra["compaction_failures"] == 0
    # The new summary is real, not a fallback.
    state = CompactionState.model_validate(ctx.extra["compaction"])
    assert state.is_fallback is False
    # Stale guard counters were cleared BEFORE the turn; the new turn may
    # legitimately set them to small values. The point is that the STALE
    # values (2 and 99) did not carry through.
    assert ctx.extra.get("compaction_low_savings_count", 0) < 2
    assert ctx.extra.get("compaction_fallback_runs", 0) < 99


# --- codex round 3: tail clamp only fires when tail would swallow history ---


async def test_keep_tail_tokens_below_threshold_honoured_verbatim() -> None:
    """A configured tail smaller than the threshold must NOT be silently
    reduced. Codex P2: the original ``// 2`` clamp was over-eager."""
    provider = _FakeSummaryProvider(reply="summary")
    middleware = CompactionMiddleware(
        summary_model=BoundModel(
            provider=provider,
            spec=Model(id="m", provider_id="faux"),
        ),
        max_tokens_before_compact=100,
        keep_tail_tokens=80,  # below threshold — must be honoured as-is
        min_compact_messages=2,
    )
    # Build messages that, combined with a tail of 80 tokens, leave enough
    # prefix to summarise. With keep_tail clamped to 50 (the OLD bug), the
    # tail would protect more recent history than the caller wanted.
    messages: list[Message] = [_user("x" * 60) for _ in range(8)]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})
    await middleware.transform_context(messages, ctx=ctx)
    # State written → boundary captured.
    boundary = ctx.extra.get("compaction_until_msg_index")
    assert boundary is not None and boundary > 0

    # Compute what tail_start should be with the un-clamped 80-token budget.
    # _user("x"*60) → 30 tokens each (60 chars / 2). tail budget 80 →
    # last 2 messages fit (60 tokens, third would push to 90 > 80).
    # Boundary should land at ≤ 6 (last 2 in tail).
    # With the old `// 2` clamp (50 tokens), only 1 message fits; boundary
    # would land at 7 — much more aggressive.
    assert boundary <= 6


async def test_keep_tail_tokens_above_threshold_still_clamped() -> None:
    """Safety net is still active for the truly-broken configuration."""
    provider = _FakeSummaryProvider(reply="summary")
    middleware = CompactionMiddleware(
        summary_model=BoundModel(
            provider=provider,
            spec=Model(id="m", provider_id="faux"),
        ),
        max_tokens_before_compact=10,
        keep_tail_tokens=200,  # above threshold — clamp engages
        min_compact_messages=2,
    )
    messages: list[Message] = [
        _user("turn 1"),
        _assistant("reply 1"),
        _user("turn 2"),
        _assistant("reply 2"),
        _user("turn 3"),
        _assistant("reply 3"),
    ]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})
    await middleware.transform_context(messages, ctx=ctx)
    # Clamp let compaction fire.
    assert "compaction" in ctx.extra
    assert len(provider.calls) == 1


async def test_under_threshold_does_not_silently_prune_tool_outputs() -> None:
    """Codex round 5 P2: when the un-pruned history is already under the
    compaction threshold, the middleware must return the ORIGINAL messages,
    not the pruned view. Otherwise old tool outputs are silently replaced
    with one-liner placeholders on every turn, with no state recording the
    loss."""
    from cubeloop.providers.base import ToolCall, ToolResultMessage

    provider = _FakeSummaryProvider()
    # Big threshold so the un-pruned history stays under it.
    middleware = _make_middleware(provider, max_tokens_before=100_000)
    big_text = "important detail " * 200  # >> _PRUNE_KEEP_CHARS
    messages: list[Message] = [
        _user("q"),
        AssistantMessage(
            content=[ToolCall(id="c1", name="read_file", arguments={"p": "x"})]
        ),
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="read_file",
            content=[TextContent(text=big_text)],
        ),
        _user("more"),
        _assistant("done"),
        _user("another"),
    ]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})

    result = await middleware.transform_context(messages, ctx=ctx)

    # No compaction happened.
    assert provider.calls == []
    assert "compaction" not in ctx.extra
    # The original tool result content is intact in the returned view.
    tool_result_msg = next(m for m in result if isinstance(m, ToolResultMessage))
    assert tool_result_msg.content[0].text == big_text


async def test_no_safe_boundary_does_not_silently_prune_tool_outputs() -> None:
    """Codex round 6 P2: when over threshold but safe_boundary returns None
    (no valid split point), the middleware must return original messages
    rather than the pruned view."""
    from cubeloop.providers.base import ToolCall, ToolResultMessage

    provider = _FakeSummaryProvider()
    # Threshold low enough that the conversation exceeds it.
    middleware = _make_middleware(provider, max_tokens_before=10, keep_tail_tokens=1000)
    big_text = "x" * 5000  # > _PRUNE_KEEP_CHARS
    # Only 2 messages — too short for safe_boundary to find a valid split.
    messages: list[Message] = [
        AssistantMessage(
            content=[ToolCall(id="c1", name="read_file", arguments={"p": "x"})]
        ),
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="read_file",
            content=[TextContent(text=big_text)],
        ),
    ]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})

    result = await middleware.transform_context(messages, ctx=ctx)

    # No compaction (no valid boundary), no LLM call.
    assert provider.calls == []
    assert "compaction" not in ctx.extra
    # The big tool result content must NOT have been silently pruned.
    tool_result_msg = next(m for m in result if isinstance(m, ToolResultMessage))
    assert tool_result_msg.content[0].text == big_text


async def test_anti_thrash_guard_skip_does_not_silently_prune() -> None:
    """Codex round 6 P2: when the anti-thrash guard fires, the returned view
    must come from the un-pruned messages so the main model still sees the
    full tool outputs."""
    from cubeloop.providers.base import ToolCall, ToolResultMessage

    provider = _FakeSummaryProvider()
    big_text = "x" * 5000  # 2500 tokens via approx_tokens
    # Threshold chosen so raw_tokens (~2515) > threshold (compaction
    # triggers) but raw < 1.5 × threshold (emergency override skipped) and
    # the boundary advance stays under _ANTI_THRASH_NEW_MSGS=8.
    middleware = _make_middleware(provider, max_tokens_before=2000)
    messages: list[Message] = [
        _user("turn 1"),
        _assistant("reply 1"),
        _user("turn 2"),
        AssistantMessage(
            content=[ToolCall(id="c1", name="read_file", arguments={"p": "x"})]
        ),
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="read_file",
            content=[TextContent(text=big_text)],
        ),
        _user("turn 3"),
    ]
    ctx = AgentContext(
        system_prompt="",
        messages=messages,
        extra={"compaction_low_savings_count": 2},  # guard tripped
    )

    result = await middleware.transform_context(messages, ctx=ctx)

    # Guard fired — no LLM call, no state written.
    assert provider.calls == []
    assert "compaction" not in ctx.extra
    # The original big tool result is intact in the returned view.
    tool_result_msg = next(m for m in result if isinstance(m, ToolResultMessage))
    assert tool_result_msg.content[0].text == big_text


# --- tool_result_compressor integration tests ---


def _make_middleware_with_compressor(
    provider: _FakeSummaryProvider,
    compressor: ToolResultCompressor,
    *,
    max_tokens_before: int = 20,
    keep_tail_tokens: int = 8,
) -> CompactionMiddleware:
    return CompactionMiddleware(
        summary_model=BoundModel(
            provider=provider,
            spec=Model(id="summary-model", provider_id="faux"),
        ),
        max_tokens_before_compact=max_tokens_before,
        keep_tail_tokens=keep_tail_tokens,
        max_summary_tokens=512,
        min_compact_messages=2,
        tool_result_compressor=compressor,
    )


async def test_compressor_preserved_text_appended_to_summary() -> None:
    """Preserved tool results appear in the summary message text."""
    from cubeloop.providers.base import ToolCall, ToolResultMessage

    provider = _FakeSummaryProvider(reply="conversation summary")

    def compressor(msg: ToolResultMessage) -> str | None:
        if msg.tool_name == "chip_metrics":
            return "AAPL: $185.32"
        return None

    middleware = _make_middleware_with_compressor(provider, compressor)
    big = "x" * 5000
    messages: list[Message] = [
        _user("show me AAPL"),
        AssistantMessage(
            content=[ToolCall(id="c1", name="chip_metrics", arguments={})]
        ),
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="chip_metrics",
            content=[TextContent(text=big)],
        ),
        _user("and MSFT?"),
        _assistant("looking up"),
        _user("thanks"),
    ]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})

    result = await middleware.transform_context(messages, ctx=ctx)

    summary_msg = result[0]
    assert isinstance(summary_msg, UserMessage)
    assert "conversation summary" in summary_msg.content[0].text
    assert "AAPL: $185.32" in summary_msg.content[0].text
    assert "chip_metrics" in summary_msg.content[0].text


async def test_compressor_preserved_excluded_from_summarizer_input() -> None:
    """Preserved messages are not fed to the summarizer LLM."""
    from cubeloop.providers.base import ToolCall, ToolResultMessage

    provider = _FakeSummaryProvider(reply="summary")

    def compressor(msg: ToolResultMessage) -> str | None:
        if msg.tool_name == "chip_metrics":
            return "kept"
        return None

    middleware = _make_middleware_with_compressor(provider, compressor)
    big = "x" * 5000
    messages: list[Message] = [
        _user("query"),
        AssistantMessage(
            content=[ToolCall(id="c1", name="chip_metrics", arguments={})]
        ),
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="chip_metrics",
            content=[TextContent(text=big)],
        ),
        _user("next"),
        _assistant("ok"),
        _user("go"),
    ]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})

    await middleware.transform_context(messages, ctx=ctx)

    assert len(provider.calls) == 1
    transcript_text = provider.calls[0]["messages"][0].content[0].text
    # The preserved tool result should not appear in the summarizer transcript.
    assert big not in transcript_text


async def test_compressor_none_return_uses_default_pruning() -> None:
    """When compressor returns None for all messages, behavior is identical
    to not having a compressor."""
    from cubeloop.providers.base import ToolCall, ToolResultMessage

    provider = _FakeSummaryProvider(reply="summary")

    def compressor(msg: ToolResultMessage) -> str | None:
        return None

    middleware = _make_middleware_with_compressor(provider, compressor)
    big = "x" * 5000
    messages: list[Message] = [
        _user("q"),
        AssistantMessage(content=[ToolCall(id="c1", name="bash", arguments={})]),
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="bash",
            content=[TextContent(text=big)],
        ),
        _user("next"),
        _assistant("ok"),
        _user("go"),
    ]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})

    result = await middleware.transform_context(messages, ctx=ctx)

    summary_msg = result[0]
    assert isinstance(summary_msg, UserMessage)
    # No preserved section in the summary
    assert "Preserved tool results" not in summary_msg.content[0].text


async def test_compressor_preserved_persists_across_compaction_rounds() -> None:
    """Preserved results from earlier rounds survive in subsequent compressed views."""
    from cubeloop.providers.base import ToolCall, ToolResultMessage

    provider = _FakeSummaryProvider(reply="round 1 summary")

    def compressor(msg: ToolResultMessage) -> str | None:
        if msg.tool_name == "chip_metrics":
            return "preserved data"
        return None

    middleware = _make_middleware_with_compressor(
        provider, compressor, max_tokens_before=20, keep_tail_tokens=4
    )

    # Round 1
    messages: list[Message] = [
        _user("q1"),
        AssistantMessage(
            content=[ToolCall(id="c1", name="chip_metrics", arguments={})]
        ),
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="chip_metrics",
            content=[TextContent(text="x" * 5000)],
        ),
        _user("q2"),
        _assistant("r2"),
        _user("q3"),
    ]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})
    await middleware.transform_context(messages, ctx=ctx)

    # Verify preserved results are in state
    state = CompactionState.model_validate(ctx.extra["compaction"])
    assert len(state.preserved_tool_results) == 1
    assert state.preserved_tool_results[0].text == "preserved data"
    assert state.preserved_tool_results[0].tool_name == "chip_metrics"

    # Round 2 — under threshold, returns compressed view with preserved data
    provider2 = _FakeSummaryProvider(reply="round 2 summary")
    middleware2 = _make_middleware_with_compressor(
        provider2, compressor, max_tokens_before=100_000, keep_tail_tokens=4
    )
    result = await middleware2.transform_context(messages, ctx=ctx)
    summary_msg = result[0]
    assert "preserved data" in summary_msg.content[0].text


async def test_trailing_synthetic_control_does_not_age_current_tool_results() -> None:
    """A call-local control must not make same-turn tool evidence look old.

    This is the production failure shape: one assistant proposes two tools, both
    return large results, and an earlier transform appends a small synthetic
    user control. The ordinary four-message history has no legal boundary at
    ``min_compact_messages=4``; the control must not manufacture one.
    """
    from cubeloop.providers.base import (
        ToolCall,
        ToolResultMessage,
        synthetic_user_message,
    )

    provider = _FakeSummaryProvider(reply="summary must not run")
    middleware = CompactionMiddleware(
        summary_model=BoundModel(
            provider=provider,
            spec=Model(id="summary-model", provider_id="faux"),
        ),
        max_tokens_before_compact=55_000,
        keep_tail_tokens=8_000,
        max_summary_tokens=1_024,
        min_compact_messages=4,
    )
    report_a = "CURRENT_REPORT_A_SENTINEL|" + "A" * 106_277
    report_b = "CURRENT_REPORT_B_SENTINEL|" + "B" * 38_796
    control = synthetic_user_message(
        '{"candidate_schema_version":1,"required_action":"evaluate current evidence"}',
        source="freshness_candidate",
    )
    messages: list[Message] = [
        _user("Compare the latest two reports."),
        AssistantMessage(
            content=[
                ToolCall(id="c1", name="report_a", arguments={}),
                ToolCall(id="c2", name="report_b", arguments={}),
            ]
        ),
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="report_a",
            content=[TextContent(text=report_a)],
        ),
        ToolResultMessage(
            tool_call_id="c2",
            tool_name="report_b",
            content=[TextContent(text=report_b)],
        ),
        control,
    ]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})

    result = await middleware.transform_context(messages, ctx=ctx)

    assert result == messages
    assert result[-1] == control
    assert report_a in result[2].content[0].text
    assert report_b in result[3].content[0].text
    assert provider.calls == []
    assert "compaction" not in ctx.extra


async def test_trailing_controls_count_toward_compaction_threshold() -> None:
    """Controls stay in the size calculation even though they do not set the tail."""
    from cubeloop.middleware.compaction.tokens import real_context_estimate
    from cubeloop.providers.base import synthetic_user_message

    provider = _FakeSummaryProvider(reply="combined view crossed threshold")
    middleware = CompactionMiddleware(
        summary_model=BoundModel(
            provider=provider,
            spec=Model(id="summary-model", provider_id="faux"),
        ),
        max_tokens_before_compact=20,
        keep_tail_tokens=4,
        max_summary_tokens=128,
        min_compact_messages=2,
    )
    history: list[Message] = [
        _user("u1"),
        _assistant("a1"),
        _user("u2"),
        _assistant("a2"),
        _user("u3"),
        _assistant("a3"),
    ]
    control = synthetic_user_message("C" * 100, source="large_call_control")
    messages = [*history, control]
    assert real_context_estimate(history) < 20
    assert real_context_estimate(messages) >= 20
    ctx = AgentContext(system_prompt="", messages=messages, extra={})

    result = await middleware.transform_context(messages, ctx=ctx)

    assert len(provider.calls) == 1
    assert "compaction" in ctx.extra
    assert result[-1] == control


async def test_old_prefix_compacts_without_pruning_current_multi_tool_evidence() -> (
    None
):
    """Older turns may compact while a current multi-tool turn stays raw."""
    from cubeloop.providers.base import (
        ToolCall,
        ToolResultMessage,
        synthetic_user_message,
    )

    provider = _FakeSummaryProvider(reply="old prefix summary")
    middleware = CompactionMiddleware(
        summary_model=BoundModel(
            provider=provider,
            spec=Model(id="summary-model", provider_id="faux"),
        ),
        max_tokens_before_compact=8_000,
        keep_tail_tokens=6_000,
        max_summary_tokens=512,
        min_compact_messages=4,
    )
    report_a = "RAW_CURRENT_A|" + "A" * 5_000
    report_b = "RAW_CURRENT_B|" + "B" * 5_000
    control_a = synthetic_user_message("policy A", source="policy_a")
    control_b = synthetic_user_message("policy B", source="policy_b")
    messages: list[Message] = [
        _user("old user 1 " * 700),
        _assistant("old answer 1 " * 700),
        _user("old user 2 " * 700),
        _assistant("old answer 2 " * 700),
        _user("current request"),
        AssistantMessage(
            content=[
                ToolCall(id="c1", name="report_a", arguments={}),
                ToolCall(id="c2", name="report_b", arguments={}),
            ]
        ),
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="report_a",
            content=[TextContent(text=report_a)],
        ),
        ToolResultMessage(
            tool_call_id="c2",
            tool_name="report_b",
            content=[TextContent(text=report_b)],
        ),
        control_a,
        control_b,
    ]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})

    result = await middleware.transform_context(messages, ctx=ctx)

    assert len(provider.calls) == 1
    assert isinstance(result[0], UserMessage)
    assert result[0].metadata.get("synthetic_source") == "compaction_summary"
    assert result[-2:] == [control_a, control_b]
    result_a = next(
        message
        for message in result
        if isinstance(message, ToolResultMessage) and message.tool_call_id == "c1"
    )
    result_b = next(
        message
        for message in result
        if isinstance(message, ToolResultMessage) and message.tool_call_id == "c2"
    )
    assert result_a.content[0].text == report_a
    assert result_b.content[0].text == report_b


def test_split_only_detaches_trailing_synthetic_user_messages() -> None:
    """Plain user tails and synthetic tool results remain ordinary history."""
    from cubeloop.middleware.compaction import (
        _split_trailing_synthetic_user_controls,
    )
    from cubeloop.providers.base import (
        ToolCall,
        ToolResultMessage,
        synthetic_user_message,
    )

    synthetic_result = ToolResultMessage(
        tool_call_id="c1",
        tool_name="audit",
        content=[TextContent(text="synthetic cleanup result")],
        metadata={"synthetic": True},
    )
    tool_history: list[Message] = [
        _user("q"),
        AssistantMessage(content=[ToolCall(id="c1", name="audit", arguments={})]),
        synthetic_result,
    ]

    history, controls = _split_trailing_synthetic_user_controls(tool_history)
    assert history == tool_history
    assert controls == []

    plain_tail = [*tool_history, _user("real user tail")]
    history, controls = _split_trailing_synthetic_user_controls(plain_tail)
    assert history == plain_tail
    assert controls == []

    control_a = synthetic_user_message("a", source="a")
    control_b = synthetic_user_message("b", source="b")
    history, controls = _split_trailing_synthetic_user_controls(
        [*plain_tail, control_a, control_b]
    )
    assert history == plain_tail
    assert controls == [control_a, control_b]


async def test_compaction_state_refs_survive_trailing_control_becoming_internal() -> (
    None
):
    """Suffix splitting does not change persisted prefix indices or refs."""
    from cubeloop.providers.base import synthetic_user_message

    provider = _FakeSummaryProvider(reply="stable summary")
    compacting = CompactionMiddleware(
        summary_model=BoundModel(
            provider=provider,
            spec=Model(id="summary-model", provider_id="faux"),
        ),
        max_tokens_before_compact=1,
        keep_tail_tokens=4,
        max_summary_tokens=128,
        min_compact_messages=2,
    )
    control = synthetic_user_message("continue", source="continuation")
    messages: list[Message] = [
        _user("turn 1"),
        _assistant("reply 1"),
        _user("turn 2"),
        _assistant("reply 2"),
        _user("turn 3"),
        _assistant("reply 3"),
        control,
    ]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})

    first = await compacting.transform_context(messages, ctx=ctx)
    boundary = ctx.extra["compaction_until_msg_index"]
    persisted_state = ctx.extra["compaction"]
    assert first[-1] == control

    under_threshold = CompactionMiddleware(
        summary_model=BoundModel(
            provider=_FakeSummaryProvider(),
            spec=Model(id="summary-model", provider_id="faux"),
        ),
        max_tokens_before_compact=1_000_000,
        keep_tail_tokens=4,
        min_compact_messages=2,
    )
    repeated = await under_threshold.transform_context(messages, ctx=ctx)
    assert repeated[0].metadata.get("synthetic_source") == "compaction_summary"
    assert repeated[-1] == control
    assert ctx.extra["compaction_until_msg_index"] == boundary
    assert ctx.extra["compaction"] == persisted_state

    with_response = [*messages, _assistant("response after control")]
    resumed = await under_threshold.transform_context(with_response, ctx=ctx)
    assert resumed[0].metadata.get("synthetic_source") == "compaction_summary"
    assert control in resumed
    assert resumed[-1] == with_response[-1]
    assert ctx.extra["compaction_until_msg_index"] == boundary
    assert ctx.extra["compaction"] == persisted_state


# --- #233: a dropped summary must not take the trigger down with it ---


def test_message_ref_ignores_volatile_bookkeeping() -> None:
    """Identity is the message's content, not its bookkeeping.

    ``run_id`` / ``timestamp`` / ``metadata`` are rewritten after the message is
    stored — the framework re-stamps ``run_id`` on messages it re-touches (the
    injected ``todo_stale_reminder`` nudges did this every turn) and hosts write
    per-turn snapshots into ``UserMessage.metadata``. Hashing them made refs
    drift, which is what dropped the state in #233.
    """
    baseline = UserMessage(content=[TextContent(text="hello")], run_id="run-1")
    restamped = UserMessage(
        content=[TextContent(text="hello")],
        run_id="run-2",
        timestamp=1_231.5,
    )
    snapshot_a = UserMessage(
        content=[TextContent(text="hello")],
        metadata={"memory_snapshot": {"captured_at": "2026-10-13T11:43:58Z"}},
    )
    snapshot_b = UserMessage(
        content=[TextContent(text="hello")],
        metadata={"memory_snapshot": {"captured_at": "2026-10-13T12:13:44Z"}},
    )

    assert message_ref(baseline) == message_ref(restamped)
    assert message_ref(snapshot_a) == message_ref(snapshot_b)

    # Content and tool linkage are identity and must still be hashed.
    assert message_ref(baseline) != message_ref(
        UserMessage(content=[TextContent(text="hello!")], run_id="run-1")
    )
    result = [
        ToolResultMessage(
            tool_call_id="c1", tool_name="t", content=[TextContent(text="x")]
        ),
        ToolResultMessage(
            tool_call_id="c2", tool_name="t", content=[TextContent(text="x")]
        ),
    ]
    assert message_ref(result[0]) != message_ref(result[1])


def _stale_state_for(messages: list[Message]) -> CompactionState:
    """A state whose refs describe a *different* history — the shape of #233."""
    return CompactionState(
        summary="stale summary from a history that no longer matches",
        summarized_message_refs=["sha256:does-not-match"],
    )


async def test_over_limit_raw_history_is_compacted_after_state_invalidation() -> None:
    """The trigger must size the raw history it is about to send.

    Regression (#233): ``real_context_estimate`` anchors on the most recent
    usage-bearing assistant and adds only what follows it, so it reports the
    size of the summary-era prompt — not the raw history that replaces it once
    the state is dropped. On the real conversation it returned 57,278 for both
    the compacted view and the 901-message raw history (actual send:
    >= 256,001 tokens). The check saw 57,278 < 80,281 and returned the
    over-limit history without compacting.
    """
    provider = _FakeSummaryProvider(reply="summary")
    middleware = CompactionMiddleware(
        summary_model=BoundModel(
            provider=provider,
            spec=Model(id="summary-model", provider_id="faux"),
        ),
        # The production values from #233.
        max_tokens_before_compact=80_281,
        keep_tail_tokens=8,
        min_compact_messages=2,
    )
    messages: list[Message] = [
        _user("turn 1"),
        _assistant("reply 1"),
        # 200_000 chars ≈ 100_000 approx tokens: on its own well over the
        # 80_281 threshold. It sits BEFORE the anchor, which is exactly why the
        # anchor-based estimate cannot see it.
        _user("x" * 200_000),
        AssistantMessage(
            content=[TextContent(text="ok")],
            usage=Usage(input_tokens=57_000),
        ),
        _user("continue"),
        _assistant("reply"),
    ]
    ctx = AgentContext(
        system_prompt="",
        messages=messages,
        extra={
            "compaction": _stale_state_for(messages).model_dump(),
            "compaction_until_msg_index": 3,
        },
    )

    result = await middleware.transform_context(messages, ctx=ctx)

    # The state was dropped, so compaction had to run on the raw history.
    assert len(provider.calls) == 1
    assert ctx.extra["compaction_until_msg_index"] > 0
    assert len(result) < len(messages)
    assert result[0].metadata.get("synthetic_source") == "compaction_summary"
    # The over-limit prefix is gone; only the protected tail survived.
    assert not any(
        isinstance(block, TextContent) and len(block.text) > 100_000
        for message in result
        if isinstance(message, (UserMessage, AssistantMessage))
        for block in message.content
        if isinstance(block, TextContent)
    )


async def test_char_floor_does_not_re_trigger_on_a_valid_compacted_view() -> None:
    """The floor covers only the dropped-state fallback.

    With a live summary the view is already compacted: the raw history behind it
    must not drive the trigger, or every turn would re-summarise. Guards against
    the fix for #233 being applied unconditionally.
    """
    provider = _FakeSummaryProvider(reply="summary")
    middleware = _make_middleware(
        provider, max_tokens_before=80_281, keep_tail_tokens=8
    )
    messages: list[Message] = [
        # A raw prefix that is far over the threshold, but already summarised.
        _user("x" * 200_000),
        _assistant("reply 1"),
        _user("turn 2"),
        _assistant("reply 2"),
        # The live tail — small, and with no usage of its own.
        _user("continue"),
        _assistant("reply 3"),
    ]
    boundary = 4
    ctx = AgentContext(
        system_prompt="",
        messages=messages,
        extra={
            "compaction": CompactionState(
                summary="live summary",
                summarized_message_refs=message_refs(messages[:boundary]),
            ).model_dump(),
            "compaction_until_msg_index": boundary,
        },
    )

    result = await middleware.transform_context(messages, ctx=ctx)

    assert provider.calls == []
    assert len(result) == 3
    assert result[0].metadata.get("synthetic_source") == "compaction_summary"
    assert "live summary" in result[0].content[0].text
    assert result[1:] == messages[boundary:]
    assert ctx.extra["compaction_until_msg_index"] == boundary


async def test_dropping_compaction_state_is_logged(caplog: Any) -> None:
    """A dropped summary means a raw-history send; it must not be silent."""
    provider = _FakeSummaryProvider(reply="summary")
    middleware = _make_middleware(provider, max_tokens_before=1_000_000)
    messages: list[Message] = [
        _user("brand new 1"),
        _assistant("reply 1"),
        _user("brand new 2"),
        _assistant("reply 2"),
    ]
    ctx = AgentContext(
        system_prompt="",
        messages=messages,
        extra={
            "compaction": _stale_state_for(messages).model_dump(),
            "compaction_until_msg_index": 2,
        },
    )

    with caplog.at_level(logging.WARNING):
        await middleware.transform_context(messages, ctx=ctx)

    assert any(
        "dropping compaction state" in record.getMessage() for record in caplog.records
    )
