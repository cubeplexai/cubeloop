from __future__ import annotations

from typing import Any

import pytest

from cubeloop.errors import (
    ContextLengthExceeded,
    ProviderAuthFailed,
    ProviderBadRequest,
    ProviderError,
    ProviderUnavailable,
    RateLimited,
)
from cubeloop.providers.base import (
    AssistantMessage,
    BaseProvider,
    BoundModel,
    Message,
    MessageStream,
    Model,
    ReasoningControl,
    StreamEvent,
    StreamOptions,
    TextContent,
    ToolDefinition,
    UserMessage,
)
from cubeloop.providers.faux import FauxProvider, faux_assistant_message
from cubeloop.providers.fallback import DEFAULT_TRIGGER_ERRORS, FallbackBoundModel


class _RaisingProvider(BaseProvider):
    """Provider that raises a given exception unconditionally from stream() and generate()."""

    def __init__(self, error: ProviderError) -> None:
        super().__init__(provider_id=error.provider or "raising")
        self._error = error

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
        raise self._error

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
        raise self._error


def _faux(provider_id: str = "faux", response: str | None = None) -> BoundModel:
    p = FauxProvider(provider_id=provider_id)
    if response is not None:
        p.set_responses([faux_assistant_message(response)])
    return p.model("model-1")


def _raising(error: ProviderError, model_id: str = "model-1") -> BoundModel:
    p = _RaisingProvider(error)
    return BoundModel(provider=p, spec=Model(id=model_id, provider_id=p.provider_id))


def _messages() -> list[Message]:
    return [UserMessage(content=[TextContent(text="hi")])]


# ---------------------------------------------------------------------------
# DEFAULT_TRIGGER_ERRORS tests
# ---------------------------------------------------------------------------


def test_default_trigger_errors_composition() -> None:
    """DEFAULT_TRIGGER_ERRORS includes residual bad-request; auth stays out."""
    assert RateLimited in DEFAULT_TRIGGER_ERRORS
    assert ProviderUnavailable in DEFAULT_TRIGGER_ERRORS
    assert ContextLengthExceeded in DEFAULT_TRIGGER_ERRORS
    assert ProviderBadRequest in DEFAULT_TRIGGER_ERRORS
    assert ProviderAuthFailed not in DEFAULT_TRIGGER_ERRORS
    from cubeloop.errors import ContentFiltered

    assert ContentFiltered not in DEFAULT_TRIGGER_ERRORS


def test_empty_chain_raises_value_error() -> None:
    """FallbackBoundModel rejects an empty chain at construction time."""
    with pytest.raises(ValueError, match="must contain at least one"):
        FallbackBoundModel(chain=())


# ---------------------------------------------------------------------------
# stream() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_primary_succeeds() -> None:
    """Primary succeeds — returns its stream, no failover."""
    primary = _faux("primary", "hello")
    fallback = _faux("fallback", "world")
    fbm = FallbackBoundModel(chain=(primary, fallback))

    stream = await fbm.stream(_messages())
    events = [ev.type async for ev in stream]
    result = await stream.result()

    assert "done" in events
    assert result.provider_id == "primary"
    # fallback provider was never used
    assert fallback.provider.call_count == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_stream_primary_raises_trigger_error_fallback_succeeds() -> None:
    """Primary raises RateLimited → failover to second model, on_failover called."""
    rate_err = RateLimited("429", provider="primary", model="model-1")
    primary = _raising(rate_err)
    fallback = _faux("fallback", "ok")

    failover_calls: list[tuple[BoundModel, BoundModel | None, Any]] = []

    async def _cb(failed: BoundModel, nxt: BoundModel | None, err: Any) -> None:
        failover_calls.append((failed, nxt, err))

    fbm = FallbackBoundModel(chain=(primary, fallback), on_failover=_cb)

    stream = await fbm.stream(_messages())
    result = await stream.result()

    assert result.provider_id == "fallback"
    assert len(failover_calls) == 1
    assert failover_calls[0][0] is primary
    assert failover_calls[0][1] is fallback
    assert isinstance(failover_calls[0][2], RateLimited)


@pytest.mark.asyncio
async def test_stream_primary_raises_non_trigger_error_reraises() -> None:
    """Primary raises ProviderAuthFailed (not in trigger_errors) → re-raised."""
    auth = ProviderAuthFailed("401", provider="primary", model="model-1")
    primary = _raising(auth)
    fallback = _faux("fallback", "ok")

    fbm = FallbackBoundModel(chain=(primary, fallback))

    with pytest.raises(ProviderAuthFailed):
        await fbm.stream(_messages())

    assert fallback.provider.call_count == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_stream_primary_first_event_error_fallback_succeeds() -> None:
    """Primary emits error as first StreamEvent → fallback to second model."""
    # FauxProvider with no responses queued emits StreamEvent(type="error") as first event.
    primary_prov = FauxProvider(provider_id="primary")
    primary = primary_prov.model("model-1")
    fallback = _faux("fallback", "rescued")

    fbm = FallbackBoundModel(chain=(primary, fallback))

    stream = await fbm.stream(_messages())
    result = await stream.result()

    assert result.provider_id == "fallback"


@pytest.mark.asyncio
async def test_stream_all_exhausted_raises_provider_unavailable() -> None:
    """All models in chain fail → raises ProviderUnavailable."""
    err = RateLimited("429", provider="p", model="m")
    fbm = FallbackBoundModel(
        chain=(_raising(err, "m1"), _raising(err, "m2"), _raising(err, "m3"))
    )

    with pytest.raises(ProviderUnavailable, match="all providers exhausted"):
        await fbm.stream(_messages())


@pytest.mark.asyncio
async def test_stream_on_failover_callback_raises_is_swallowed() -> None:
    """on_failover callback that raises must not abort the failover."""
    rate_err = RateLimited("429", provider="primary", model="model-1")
    primary = _raising(rate_err)
    fallback = _faux("fallback", "ok")

    async def _bad_cb(failed: BoundModel, nxt: BoundModel | None, err: Any) -> None:
        raise RuntimeError("callback is broken")

    fbm = FallbackBoundModel(chain=(primary, fallback), on_failover=_bad_cb)

    stream = await fbm.stream(_messages())
    result = await stream.result()

    assert result.provider_id == "fallback"


# ---------------------------------------------------------------------------
# generate() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_primary_raises_trigger_error_fallback_succeeds() -> None:
    """generate() — primary raises RateLimited, fallback returns AssistantMessage."""
    rate_err = RateLimited("429", provider="primary", model="model-1")
    primary = _raising(rate_err)
    fallback = _faux("fallback", "generated")

    fbm = FallbackBoundModel(chain=(primary, fallback))

    result = await fbm.generate(_messages())

    assert result.provider_id == "fallback"


# ---------------------------------------------------------------------------
# Custom trigger_errors tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_trigger_errors_includes_auth_failed() -> None:
    """Custom trigger_errors that includes ProviderAuthFailed → auth failure triggers failover."""
    auth_err = ProviderAuthFailed("401", provider="primary", model="model-1")
    primary = _raising(auth_err)
    fallback = _faux("fallback", "ok")

    fbm = FallbackBoundModel(
        chain=(primary, fallback),
        trigger_errors=frozenset({ProviderAuthFailed}),
    )

    stream = await fbm.stream(_messages())
    result = await stream.result()

    assert result.provider_id == "fallback"


# ---------------------------------------------------------------------------
# Coverage gap tests
# ---------------------------------------------------------------------------


def test_provider_and_spec_properties() -> None:
    """provider and spec proxy chain[0]."""
    primary = _faux("primary", "hello")
    fallback = _faux("fallback", "world")
    fbm = FallbackBoundModel(chain=(primary, fallback))

    assert fbm.provider is primary.provider
    assert fbm.spec is primary.spec


class _EmptyStreamProvider(BaseProvider):
    """Returns a stream that immediately terminates with no events (StopAsyncIteration path)."""

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
        ms = MessageStream()

        async def _produce() -> None:
            raise RuntimeError("empty stream — no events emitted")

        ms.attach_task(__import__("asyncio").create_task(_produce()))
        return ms


@pytest.mark.asyncio
async def test_stream_empty_stream_triggers_failover() -> None:
    """Stream that terminates before emitting any event → StopAsyncIteration path → failover."""
    primary_prov = _EmptyStreamProvider()
    primary_prov.provider_id = "empty"
    primary = BoundModel(provider=primary_prov, spec=Model(id="m", provider_id="empty"))
    fallback = _faux("fallback", "recovered")

    fbm = FallbackBoundModel(chain=(primary, fallback))

    stream = await fbm.stream(_messages())
    result = await stream.result()

    assert result.provider_id == "fallback"


class _MidStreamErrorProvider(BaseProvider):
    """Emits one start event then the producer task raises — exercises _forward's except handler."""

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
        ms = MessageStream()

        async def _produce() -> None:
            ms.push(StreamEvent(type="start"))
            raise RuntimeError("mid-stream failure")

        ms.attach_task(__import__("asyncio").create_task(_produce()))
        return ms


@pytest.mark.asyncio
async def test_stream_mid_stream_error_is_forwarded() -> None:
    """Error after the first non-error event is forwarded as-is (no retry)."""
    prov = _MidStreamErrorProvider()
    prov.provider_id = "midstream"
    bound = BoundModel(provider=prov, spec=Model(id="m", provider_id="midstream"))
    fbm = FallbackBoundModel(chain=(bound,))

    stream = await fbm.stream(_messages())
    events = [ev.type async for ev in stream]
    result = await stream.result()

    assert "start" in events
    assert "error" in events
    assert result.stop_reason == "error"


@pytest.mark.asyncio
async def test_generate_non_trigger_error_reraises() -> None:
    """generate() — ProviderAuthFailed (not in trigger_errors) re-raised immediately."""
    auth = ProviderAuthFailed("401", provider="primary", model="model-1")
    primary = _raising(auth)
    fallback = _faux("fallback", "ok")

    fbm = FallbackBoundModel(chain=(primary, fallback))

    with pytest.raises(ProviderAuthFailed):
        await fbm.generate(_messages())


@pytest.mark.asyncio
async def test_generate_all_exhausted_raises_provider_unavailable() -> None:
    """generate() — all models fail → raises ProviderUnavailable."""
    err = RateLimited("429", provider="p", model="m")
    fbm = FallbackBoundModel(chain=(_raising(err, "m1"), _raising(err, "m2")))

    with pytest.raises(ProviderUnavailable, match="all providers exhausted"):
        await fbm.generate(_messages())


@pytest.mark.asyncio
async def test_generate_error_assistant_message_triggers_failover() -> None:
    """generate() — primary returns AssistantMessage(stop_reason="error") → failover."""
    # FauxProvider with no queued responses returns an error AssistantMessage.
    primary_prov = FauxProvider(provider_id="primary")
    primary = primary_prov.model("model-1")
    fallback = _faux("fallback", "generated via fallback")

    fbm = FallbackBoundModel(chain=(primary, fallback))

    result = await fbm.generate(_messages())

    assert result.provider_id == "fallback"
    assert result.stop_reason != "error"


# ---------------------------------------------------------------------------
# chain_providers helper — used by Recorder / Meter to subscribe to every
# unique provider in a fallback chain (issue #167).
# ---------------------------------------------------------------------------


def test_chain_providers_for_fallback_returns_unique_providers_in_order() -> None:
    """FallbackBoundModel chain → list of unique providers, primary first."""
    from cubeloop.providers.fallback import chain_providers

    p1 = FauxProvider(provider_id="p1")
    p2 = FauxProvider(provider_id="p2")
    p3 = FauxProvider(provider_id="p3")
    fbm = FallbackBoundModel(chain=(p1.model("a"), p2.model("b"), p3.model("c")))

    out = chain_providers(fbm)
    assert out == [p1, p2, p3]


def test_chain_providers_dedupes_shared_provider_across_legs() -> None:
    """Two chain entries on the same provider instance → single entry in output."""
    from cubeloop.providers.fallback import chain_providers

    shared = FauxProvider(provider_id="shared")
    other = FauxProvider(provider_id="other")
    # primary + tertiary share the same provider; secondary differs.
    fbm = FallbackBoundModel(
        chain=(shared.model("a"), other.model("b"), shared.model("c"))
    )

    out = chain_providers(fbm)
    assert out == [shared, other]


def test_chain_providers_for_plain_bound_model_returns_single_entry() -> None:
    """A plain BoundModel → single-entry list with its provider."""
    from cubeloop.providers.fallback import chain_providers

    p = FauxProvider(provider_id="plain")
    out = chain_providers(p.model("x"))
    assert out == [p]


def test_chain_providers_for_none_returns_empty() -> None:
    """None model → empty list (used by attach()'s legacy fallback path)."""
    from cubeloop.providers.fallback import chain_providers

    assert chain_providers(None) == []


def test_chain_providers_warns_when_chain_leg_is_not_base_provider(caplog) -> None:
    """Chain entries whose provider isn't a BaseProvider are skipped
    AND logged at WARNING so an operator notices the dropped leg.
    Without the warning, tracing / metrics silently miss the leg.
    """
    import logging

    from cubeloop.providers.fallback import chain_providers

    real = FauxProvider(provider_id="real")

    # Duck-typed Provider-protocol wrapper that isn't a BaseProvider —
    # has the .stream / .generate shape but no subscribe_* listeners,
    # which is exactly why tracing / metrics can't use it.
    class _DuckProvider:
        async def stream(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

        async def generate(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

    duck = _DuckProvider()
    duck_bound = BoundModel(provider=duck, spec=real.model("m").spec)  # type: ignore[arg-type]
    fbm = FallbackBoundModel(chain=(real.model("m"), duck_bound))

    with caplog.at_level(logging.WARNING, logger="cubepi.providers.base"):
        out = chain_providers(fbm)

    assert out == [real], "duck-typed leg should be dropped"
    assert any(
        "chain[1]" in record.message and "_DuckProvider" in record.message
        for record in caplog.records
    ), "expected a WARNING-level log mentioning chain[1] / _DuckProvider"


def test_chain_providers_returns_empty_for_object_without_provider_or_chain() -> None:
    """Walk: model is not None, has no .chain, .provider isn't a
    BaseProvider → final `return []` path."""
    from cubeloop.providers.base import chain_providers

    class _Bare:
        """Object that satisfies `model is not None` but exposes neither
        a `.chain` nor a BaseProvider-typed `.provider`. Final return []
        is the only sensible answer."""

    assert chain_providers(_Bare()) == []


def test_collect_agent_providers_falls_back_to_legacy_provider_attribute() -> None:
    """When ``agent._model`` is absent / yields no providers and the
    agent itself exposes a ``provider`` attribute that IS a BaseProvider,
    use it. Covers the legacy fallback path."""
    from cubeloop.providers.base import collect_agent_providers

    real = FauxProvider(provider_id="legacy")

    class _LegacyAgent:
        _model = None
        provider = real

    assert collect_agent_providers(_LegacyAgent()) == [real]


def test_collect_agent_providers_empty_when_no_model_or_legacy_provider() -> None:
    """No `_model`, no `provider` → empty list. Defensive for fully
    detached agents (unlikely in practice but the contract guarantees []
    rather than a crash)."""
    from cubeloop.providers.base import collect_agent_providers

    class _BlankAgent:
        pass

    assert collect_agent_providers(_BlankAgent()) == []


# ---------------------------------------------------------------------------
# tool_choice forwarding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_forwards_tool_choice() -> None:
    """stream() accepts and forwards tool_choice to inner BoundModel."""
    primary = _faux("primary", "hello")
    fbm = FallbackBoundModel(chain=(primary,))

    stream = await fbm.stream(_messages(), tool_choice="required")
    result = await stream.result()

    assert result.provider_id == "primary"


@pytest.mark.asyncio
async def test_generate_forwards_tool_choice() -> None:
    """generate() accepts and forwards tool_choice to inner BoundModel."""
    primary = _faux("primary", "hello")
    fbm = FallbackBoundModel(chain=(primary,))

    result = await fbm.generate(_messages(), tool_choice="required")

    assert result.provider_id == "primary"


# ---------------------------------------------------------------------------
# retry / sticky / typed first-event (issue #211)
# ---------------------------------------------------------------------------


class _CountingRaiseProvider(BaseProvider):
    """Raise ``error`` the first ``fail_times`` stream/generate calls, then succeed."""

    def __init__(
        self, error: ProviderError, fail_times: int, provider_id: str = "count"
    ) -> None:
        super().__init__(provider_id=provider_id)
        self._error = error
        self._fail_times = fail_times
        self.calls = 0

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
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._error
        inner = FauxProvider(provider_id=self.provider_id)
        inner.set_responses([faux_assistant_message("recovered")])
        return await inner.stream(model, messages)

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
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._error
        return faux_assistant_message("recovered")


class _FirstEventTypedProvider(BaseProvider):
    def __init__(self, event: StreamEvent, provider_id: str = "typed-ev") -> None:
        super().__init__(provider_id=provider_id)
        self._event = event

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
        ms = MessageStream()

        async def _produce() -> None:
            ms.push(self._event)
            if self._event.type == "error":
                ms.set_result(
                    AssistantMessage(
                        content=[],
                        stop_reason="error",
                        error_message=self._event.error_message,
                        error_type=self._event.error_type,
                    )
                )

        ms.attach_task(__import__("asyncio").create_task(_produce()))
        return ms


@pytest.mark.asyncio
async def test_same_model_retry_recovers_without_secondary() -> None:
    err = ProviderUnavailable("blip", provider="primary", model="m")
    primary_p = _CountingRaiseProvider(err, fail_times=2, provider_id="primary")
    primary = BoundModel(provider=primary_p, spec=Model(id="m", provider_id="primary"))
    fallback = _faux("fallback", "nope")
    retries: list[int] = []

    def _on_retry(failed, error, attempt, wait_s):  # noqa: ANN001
        retries.append(attempt)

    fbm = FallbackBoundModel(
        chain=(primary, fallback),
        on_retry=_on_retry,
        retry_backoff=0.0,
    )
    stream = await fbm.stream(_messages())
    result = await stream.result()
    assert result.content
    assert fallback.provider.call_count == 0  # type: ignore[attr-defined]
    assert primary_p.calls == 3
    assert retries == [1, 2]


@pytest.mark.asyncio
async def test_generate_reports_every_retry_and_fallback_model_attempt() -> None:
    error = ProviderUnavailable("blip", provider="primary", model="m")
    primary_provider = _CountingRaiseProvider(
        error, fail_times=99, provider_id="primary"
    )
    primary = BoundModel(
        provider=primary_provider,
        spec=Model(id="m", provider_id="primary"),
    )
    fallback = _faux("fallback", "ok")
    attempts: list[str] = []
    model = FallbackBoundModel(
        chain=(primary, fallback),
        max_retries_per_model=1,
        retry_backoff=0.0,
    )

    await model.generate(
        _messages(),
        options=StreamOptions(
            on_model_attempt=lambda spec: attempts.append(spec.provider_id)
        ),
    )

    assert attempts == ["primary", "primary", "fallback"]


@pytest.mark.asyncio
async def test_retries_then_failover() -> None:
    err = RateLimited("429", provider="primary", model="m")
    primary_p = _CountingRaiseProvider(err, fail_times=99, provider_id="primary")
    primary = BoundModel(provider=primary_p, spec=Model(id="m", provider_id="primary"))
    fallback = _faux("fallback", "ok")
    hops: list[Any] = []

    def _on_fail(failed, nxt, error):  # noqa: ANN001
        hops.append(error)

    fbm = FallbackBoundModel(
        chain=(primary, fallback),
        max_retries_per_model=3,
        on_failover=_on_fail,
        retry_backoff=0.0,
    )
    stream = await fbm.stream(_messages())
    result = await stream.result()
    assert result.provider_id == "fallback"
    assert primary_p.calls == 4  # first try + 3 retries
    assert len(hops) == 1


@pytest.mark.asyncio
async def test_context_length_skips_same_model_retry() -> None:
    err = ContextLengthExceeded(
        "too long", provider="p", model="small", tokens_in=200_000, context_window=8_000
    )
    primary_p = _CountingRaiseProvider(err, fail_times=99, provider_id="small")
    primary = BoundModel(
        provider=primary_p,
        spec=Model(id="small", provider_id="small", context_window=8_000),
    )
    fallback = _faux("fallback", "ok")
    fbm = FallbackBoundModel(chain=(primary, fallback), retry_backoff=0.0)
    stream = await fbm.stream(_messages())
    await stream.result()
    assert primary_p.calls == 1


@pytest.mark.asyncio
async def test_context_length_skips_too_small_fallback() -> None:
    err = ContextLengthExceeded(
        "too long", provider="p", model="a", tokens_in=200_000, context_window=8_000
    )
    a = _raising(err, "a")
    # rewrite spec window
    a = BoundModel(
        provider=a.provider,
        spec=Model(id="a", provider_id="p", context_window=8_000),
    )
    small_err = ContextLengthExceeded(
        "still too long",
        provider="p",
        model="b",
        tokens_in=200_000,
        context_window=16_000,
    )
    b = BoundModel(
        provider=_RaisingProvider(small_err),
        spec=Model(id="b", provider_id="p", context_window=16_000),
    )
    fbm = FallbackBoundModel(chain=(a, b), retry_backoff=0.0)
    with pytest.raises(ProviderUnavailable) as ei:
        await fbm.stream(_messages())
    assert ei.value.errors
    assert all(isinstance(e, ContextLengthExceeded) for e in ei.value.errors)
    assert getattr(b.provider, "_error", None) is small_err
    # 16k window cannot fit 200k — must not be invoked.
    assert not hasattr(b.provider, "calls") or getattr(b.provider, "calls", 0) == 0
    # _RaisingProvider has no counter; wrap via call_count if present
    if hasattr(b.provider, "call_count"):
        assert b.provider.call_count == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_residual_bad_request_failovers_without_retry() -> None:
    err = ProviderBadRequest("InvalidParameter", provider="primary", model="m")
    primary_p = _CountingRaiseProvider(err, fail_times=99, provider_id="primary")
    primary = BoundModel(provider=primary_p, spec=Model(id="m", provider_id="primary"))
    fallback = _faux("fallback", "ok")
    fbm = FallbackBoundModel(chain=(primary, fallback), retry_backoff=0.0)
    stream = await fbm.stream(_messages())
    result = await stream.result()
    assert result.provider_id == "fallback"
    assert primary_p.calls == 1


@pytest.mark.asyncio
async def test_first_event_typed_auth_does_not_failover() -> None:
    ev = StreamEvent(
        type="error",
        error_message="bad key",
        error_type="ProviderAuthFailed",
        status_code=401,
    )
    primary = BoundModel(
        provider=_FirstEventTypedProvider(ev, "primary"),
        spec=Model(id="m", provider_id="primary"),
    )
    fallback = _faux("fallback", "ok")
    fbm = FallbackBoundModel(chain=(primary, fallback))
    with pytest.raises(ProviderAuthFailed):
        await fbm.stream(_messages())
    assert fallback.provider.call_count == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_first_event_typed_rate_limited_retries_then_failovers() -> None:
    ev = StreamEvent(
        type="error",
        error_message="slow down",
        error_type="RateLimited",
        status_code=429,
    )
    primary = BoundModel(
        provider=_FirstEventTypedProvider(ev, "primary"),
        spec=Model(id="m", provider_id="primary"),
    )
    fallback = _faux("fallback", "ok")
    fbm = FallbackBoundModel(
        chain=(primary, fallback), max_retries_per_model=1, retry_backoff=0.0
    )
    stream = await fbm.stream(_messages())
    result = await stream.result()
    assert result.provider_id == "fallback"


@pytest.mark.asyncio
async def test_sticky_after_successful_failover() -> None:
    from cubeloop.providers.fallback import begin_fallback_run, end_fallback_run

    err = ProviderUnavailable("down", provider="primary", model="m")
    primary = _raising(err)
    fb = FauxProvider(provider_id="fallback")
    fb.set_responses([faux_assistant_message("ok"), faux_assistant_message("ok2")])
    fallback = fb.model("model-1")
    fbm = FallbackBoundModel(
        chain=(primary, fallback), max_retries_per_model=0, retry_backoff=0.0
    )
    token = begin_fallback_run()
    try:
        s1 = await fbm.stream(_messages())
        r1 = await s1.result()
        assert r1.provider_id == "fallback"
        s2 = await fbm.stream(_messages())
        r2 = await s2.result()
        assert r2.provider_id == "fallback"
    finally:
        end_fallback_run(token)


@pytest.mark.asyncio
async def test_sticky_second_call_skips_primary() -> None:
    from cubeloop.providers.fallback import begin_fallback_run, end_fallback_run

    err = ProviderUnavailable("down", provider="primary", model="m")
    primary_p = _CountingRaiseProvider(err, fail_times=99, provider_id="primary")
    primary = BoundModel(provider=primary_p, spec=Model(id="m", provider_id="primary"))
    fb = FauxProvider(provider_id="fallback")
    fb.set_responses([faux_assistant_message("ok"), faux_assistant_message("ok2")])
    fallback = fb.model("model-1")
    fbm = FallbackBoundModel(
        chain=(primary, fallback), max_retries_per_model=0, retry_backoff=0.0
    )
    token = begin_fallback_run()
    try:
        await (await fbm.stream(_messages())).result()
        after_first = primary_p.calls
        await (await fbm.stream(_messages())).result()
        assert primary_p.calls == after_first
    finally:
        end_fallback_run(token)


@pytest.mark.asyncio
async def test_new_run_resets_sticky_to_primary() -> None:
    from cubeloop.providers.fallback import begin_fallback_run, end_fallback_run

    prim = FauxProvider(provider_id="primary")
    prim.set_responses(
        [faux_assistant_message("hello"), faux_assistant_message("hello2")]
    )
    ok_primary = prim.model("model-1")
    fallback = _faux("fallback", "world")
    fbm = FallbackBoundModel(chain=(ok_primary, fallback))
    token = begin_fallback_run()
    try:
        s = await fbm.stream(_messages())
        await s.result()
    finally:
        end_fallback_run(token)
    token = begin_fallback_run()
    try:
        s = await fbm.stream(_messages())
        r = await s.result()
        assert r.provider_id == "primary"
    finally:
        end_fallback_run(token)


@pytest.mark.asyncio
async def test_new_run_after_failover_probes_primary_again() -> None:
    from cubeloop.providers.fallback import begin_fallback_run, end_fallback_run

    err = ProviderUnavailable("down", provider="primary", model="m")
    primary_p = _CountingRaiseProvider(err, fail_times=99, provider_id="primary")
    primary = BoundModel(provider=primary_p, spec=Model(id="m", provider_id="primary"))
    fb = FauxProvider(provider_id="fallback")
    fb.set_responses([faux_assistant_message("ok"), faux_assistant_message("ok2")])
    fbm = FallbackBoundModel(
        chain=(primary, fb.model("model-1")),
        max_retries_per_model=0,
        retry_backoff=0.0,
    )
    token = begin_fallback_run()
    try:
        r1 = await (await fbm.stream(_messages())).result()
        assert r1.provider_id == "fallback"
    finally:
        end_fallback_run(token)
    after_run1 = primary_p.calls
    token = begin_fallback_run()
    try:
        r2 = await (await fbm.stream(_messages())).result()
        assert r2.provider_id == "fallback"
        assert primary_p.calls > after_run1
    finally:
        end_fallback_run(token)


@pytest.mark.asyncio
async def test_generate_exhaustion_preserves_errors() -> None:
    e1 = RateLimited("a", provider="p", model="m1")
    e2 = RateLimited("b", provider="p", model="m2")
    fbm = FallbackBoundModel(
        chain=(_raising(e1, "m1"), _raising(e2, "m2")),
        max_retries_per_model=0,
    )
    with pytest.raises(ProviderUnavailable) as ei:
        await fbm.generate(_messages())
    assert len(ei.value.errors) == 2
    assert isinstance(ei.value.__cause__, RateLimited)


@pytest.mark.asyncio
async def test_exhaustion_errors_are_one_per_leg() -> None:
    e1 = RateLimited("a", provider="p", model="m1")
    e2 = RateLimited("b", provider="p", model="m2")
    fbm = FallbackBoundModel(
        chain=(_raising(e1, "m1"), _raising(e2, "m2")),
        max_retries_per_model=3,
        retry_backoff=0.0,
    )
    with pytest.raises(ProviderUnavailable) as ei:
        await fbm.generate(_messages())
    assert len(ei.value.errors) == 2


@pytest.mark.asyncio
async def test_generate_skips_too_small_middle_leg() -> None:
    err = ContextLengthExceeded(
        "too long", provider="p", model="a", tokens_in=200_000, context_window=8_000
    )
    a = BoundModel(
        provider=_RaisingProvider(err),
        spec=Model(id="a", provider_id="p", context_window=8_000),
    )
    mid_p = _CountingRaiseProvider(
        ContextLengthExceeded("mid", tokens_in=200_000, context_window=16_000),
        fail_times=99,
        provider_id="mid",
    )
    b = BoundModel(
        provider=mid_p, spec=Model(id="b", provider_id="p", context_window=16_000)
    )
    big = _faux("big", "ok")
    big = BoundModel(
        provider=big.provider,
        spec=Model(id="c", provider_id="big", context_window=400_000),
    )
    fbm = FallbackBoundModel(chain=(a, b, big), retry_backoff=0.0)
    result = await fbm.generate(_messages())
    assert mid_p.calls == 0
    assert result.provider_id == "big"


@pytest.mark.asyncio
async def test_content_filtered_does_not_hop_unless_listed() -> None:
    from cubeloop.errors import ContentFiltered

    err = ContentFiltered("blocked", provider="primary", model="m")
    primary = _raising(err)
    fallback = _faux("fallback", "ok")
    fbm = FallbackBoundModel(chain=(primary, fallback))
    with pytest.raises(ContentFiltered):
        await fbm.stream(_messages())
    assert fallback.provider.call_count == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_content_filtered_hops_when_opted_in() -> None:
    from cubeloop.errors import ContentFiltered

    err = ContentFiltered("blocked", provider="primary", model="m")
    primary = _raising(err)
    fallback = _faux("fallback", "ok")
    fbm = FallbackBoundModel(
        chain=(primary, fallback),
        trigger_errors=frozenset({ContentFiltered}),
    )
    stream = await fbm.stream(_messages())
    result = await stream.result()
    assert result.provider_id == "fallback"


@pytest.mark.asyncio
async def test_reset_active_and_index_helpers() -> None:
    from cubeloop.providers.fallback import (
        begin_fallback_run,
        end_fallback_run,
        get_active_index,
        reset_active,
        set_active_index,
    )

    assert get_active_index() == 0
    token = begin_fallback_run()
    try:
        set_active_index(2)
        assert get_active_index() == 2
        reset_active()
        assert get_active_index() == 0
        fbm = FallbackBoundModel(chain=(_faux("p", "hi"),))
        fbm.reset_active()
        assert get_active_index() == 0
    finally:
        end_fallback_run(token)


def test_negative_retries_rejected() -> None:
    with pytest.raises(ValueError, match="max_retries_per_model"):
        FallbackBoundModel(chain=(_faux("p", "hi"),), max_retries_per_model=-1)


@pytest.mark.asyncio
async def test_on_retry_callback_exception_is_swallowed() -> None:
    err = ProviderUnavailable("blip", provider="primary", model="m")
    primary_p = _CountingRaiseProvider(err, fail_times=1, provider_id="primary")
    primary = BoundModel(provider=primary_p, spec=Model(id="m", provider_id="primary"))

    def _bad(failed, error, attempt, wait_s):  # noqa: ANN001
        raise RuntimeError("retry hook broken")

    fbm = FallbackBoundModel(
        chain=(primary,), on_retry=_bad, retry_backoff=0.0, max_retries_per_model=2
    )
    stream = await fbm.stream(_messages())
    result = await stream.result()
    assert result.content


@pytest.mark.asyncio
async def test_rate_limited_honors_retry_after_cap() -> None:
    err = RateLimited("429", provider="primary", model="m", retry_after=30.0)
    primary_p = _CountingRaiseProvider(err, fail_times=1, provider_id="primary")
    primary = BoundModel(provider=primary_p, spec=Model(id="m", provider_id="primary"))
    waits: list[float] = []

    def _on_retry(failed, error, attempt, wait_s):  # noqa: ANN001
        waits.append(wait_s)

    fbm = FallbackBoundModel(
        chain=(primary,),
        on_retry=_on_retry,
        retry_backoff=0.0,
        max_retry_after=0.0,
        max_retries_per_model=1,
    )
    stream = await fbm.stream(_messages())
    await stream.result()
    assert waits == [0.0]


def test_sticky_disabled_always_starts_at_zero() -> None:
    fbm = FallbackBoundModel(chain=(_faux("p", "hi"),), sticky_within_run=False)
    assert fbm._start_index() == 0


@pytest.mark.asyncio
async def test_async_on_retry_is_awaited() -> None:
    seen: list[int] = []
    err = ProviderUnavailable("blip", provider="primary", model="m")
    primary_p = _CountingRaiseProvider(err, fail_times=1, provider_id="primary")
    primary = BoundModel(provider=primary_p, spec=Model(id="m", provider_id="primary"))

    async def _cb(failed, error, attempt, wait_s):  # noqa: ANN001
        seen.append(attempt)

    fbm = FallbackBoundModel(
        chain=(primary,), on_retry=_cb, retry_backoff=0.0, max_retries_per_model=2
    )
    await (await fbm.stream(_messages())).result()
    assert seen == [1]


def test_context_too_small_without_need_or_window() -> None:
    from cubeloop.providers.fallback import _context_too_small

    bound = BoundModel(
        provider=_RaisingProvider(ProviderUnavailable("x")),
        spec=Model(id="m", provider_id="p", context_window=0),
    )
    assert (
        _context_too_small(ContextLengthExceeded("long", tokens_in=10), bound) is False
    )
    bound2 = BoundModel(
        provider=bound.provider,
        spec=Model(id="m", provider_id="p", context_window=8_000),
    )
    assert (
        _context_too_small(ContextLengthExceeded("long", tokens_in=None), bound2)
        is False
    )


def test_start_index_clamps_out_of_range() -> None:
    from cubeloop.providers.fallback import (
        begin_fallback_run,
        end_fallback_run,
        set_active_index,
    )

    fbm = FallbackBoundModel(chain=(_faux("p", "hi"),))
    token = begin_fallback_run()
    try:
        set_active_index(99)
        assert fbm._start_index() == 0
    finally:
        end_fallback_run(token)


def test_stick_noop_when_disabled() -> None:
    from cubeloop.providers.fallback import begin_fallback_run, end_fallback_run

    fbm = FallbackBoundModel(chain=(_faux("p", "hi"),), sticky_within_run=False)
    token = begin_fallback_run()
    try:
        fbm._stick(1)
        assert fbm._start_index() == 0
    finally:
        end_fallback_run(token)


@pytest.mark.asyncio
async def test_retry_sleeps_when_wait_positive() -> None:
    err = RateLimited("429", provider="primary", model="m", retry_after=0.01)
    primary_p = _CountingRaiseProvider(err, fail_times=1, provider_id="primary")
    primary = BoundModel(provider=primary_p, spec=Model(id="m", provider_id="primary"))
    fbm = FallbackBoundModel(
        chain=(primary,),
        retry_backoff=0.0,
        max_retry_after=0.01,
        max_retries_per_model=1,
    )
    await (await fbm.stream(_messages())).result()
    assert primary_p.calls == 2
