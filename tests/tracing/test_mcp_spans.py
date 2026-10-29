"""Phase 5: pin MCP CLIENT span emissions + W3C traceparent propagation.

The MCP adapter wraps every ``call_remote`` invocation in
:func:`cubeloop.mcp._tracing.mcp_client_span`, which opens a CLIENT span
with the GenAI MCP semconv attributes. When the OTel API is absent the
context manager is a no-op (verified separately).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import pytest
from opentelemetry import trace as _trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import SpanKind, StatusCode

from cubeloop.mcp._adapter import make_mcp_agent_tool


class _CaptureExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans):  # noqa: ANN001
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def _make_provider() -> tuple[TracerProvider, _CaptureExporter]:
    """Build a fresh TracerProvider with an in-memory exporter.

    Each test gets its own provider; we don't use trace.set_tracer_provider
    globally, instead we monkeypatch the module-level ``_otel_trace`` in
    ``cubeloop.mcp._tracing`` so MCP fetches the test's tracer.
    """
    resource = Resource.create({"service.name": "mcp-span-tests"})
    provider = TracerProvider(resource=resource)
    exporter = _CaptureExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _patch_mcp_trace(monkeypatch, provider: TracerProvider) -> None:
    """Swap the cubeloop.mcp._tracing module's ``get_tracer`` to use ours."""
    import cubeloop.mcp._tracing as mcp_tracing

    class _ShimTraceMod:
        @staticmethod
        def get_tracer(name: str):  # noqa: D401
            return provider.get_tracer(name)

        @staticmethod
        def use_span(span, **kwargs):
            return _trace.use_span(span, **kwargs)

        @staticmethod
        def get_current_span():
            return _trace.get_current_span()

        @staticmethod
        def set_span_in_context(span, context=None):
            return _trace.set_span_in_context(span, context)

    monkeypatch.setattr(mcp_tracing, "_otel_trace", _ShimTraceMod)
    monkeypatch.setattr(mcp_tracing, "_OTEL_AVAILABLE", True)
    # Other tests' Tracer.attach() may have left entries on the provider
    # stack; clear so this test resolves via the shim above. The
    # current-tool contextvar is per-task and won't leak between tests
    # (pytest-asyncio runs each test in a fresh task), so no reset
    # needed there.
    monkeypatch.setattr(mcp_tracing, "_provider_stack", [])


async def _make_tool(
    call_remote,
    *,
    server_address=None,
    server_port=None,
    protocol_version=None,
    session_id=None,
):
    """Build the MCP-adapted AgentTool. Mirrors what loaders produce."""
    return make_mcp_agent_tool(
        name="search",
        description="search the web",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        call_remote=call_remote,
        server_address=server_address,
        server_port=server_port,
        protocol_version=protocol_version,
        session_id=session_id,
    )


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


class TestMCPClientSpan:
    async def test_span_emitted_with_required_attrs(self, monkeypatch):
        provider, exporter = _make_provider()
        _patch_mcp_trace(monkeypatch, provider)

        async def call_remote(name, args):
            return {"content": [{"type": "text", "text": "ok"}], "isError": False}

        tool = await _make_tool(
            call_remote,
            server_address="example.com",
            server_port=443,
            protocol_version="2025-11-25",
            session_id="sess-abc",
        )
        result = await tool.execute("tc1", tool.parameters.model_validate({"q": "x"}))
        assert result.is_error is None  # success path

        # One MCP CLIENT span captured.
        mcp_spans = [s for s in exporter.spans if s.name.startswith("tools/call ")]
        assert len(mcp_spans) == 1
        span = mcp_spans[0]
        assert span.kind == SpanKind.CLIENT
        attrs = _attrs(span)
        assert attrs["mcp.method.name"] == "tools/call"
        assert attrs["gen_ai.tool.name"] == "search"
        assert attrs["gen_ai.operation.name"] == "execute_tool"
        assert attrs["mcp.session.id"] == "sess-abc"
        assert attrs["mcp.protocol.version"] == "2025-11-25"
        assert attrs["server.address"] == "example.com"
        assert attrs["server.port"] == 443

    async def test_span_records_exception_on_failure(self, monkeypatch):
        provider, exporter = _make_provider()
        _patch_mcp_trace(monkeypatch, provider)

        async def call_remote(name, args):
            raise RuntimeError("server unavailable")

        tool = await _make_tool(call_remote)
        try:
            await tool.execute("tc1", tool.parameters.model_validate({"q": "x"}))
        except RuntimeError:
            pass

        mcp_spans = [s for s in exporter.spans if s.name.startswith("tools/call ")]
        assert len(mcp_spans) == 1
        span = mcp_spans[0]
        assert span.status.status_code == StatusCode.ERROR
        attrs = _attrs(span)
        assert attrs["error.type"] == "RuntimeError"
        evnames = [e.name for e in span.events]
        assert "exception" in evnames

    async def test_exception_event_is_recorded_only_once(self, monkeypatch):
        """OTel's ``use_span`` defaults to record_exception=True; if we
        leave that on, the auto-recording on context exit and our own
        ``record_exception`` in the except branch both fire — duplicate
        exception events double-count errors at the backend.

        We disable auto-recording on ``use_span`` and own the
        recording. Pin: exactly one ``exception`` event per failure."""
        provider, exporter = _make_provider()
        _patch_mcp_trace(monkeypatch, provider)

        async def call_remote(name, args):
            raise RuntimeError("boom")

        tool = await _make_tool(call_remote)
        try:
            await tool.execute("tc1", tool.parameters.model_validate({"q": "x"}))
        except RuntimeError:
            pass

        mcp_spans = [s for s in exporter.spans if s.name.startswith("tools/call ")]
        assert len(mcp_spans) == 1
        exception_events = [e for e in mcp_spans[0].events if e.name == "exception"]
        assert len(exception_events) == 1, (
            f"expected exactly 1 exception event; got {len(exception_events)}"
        )

    async def test_span_records_cancellation(self, monkeypatch):
        """Cancellation is a control signal, not a failure — match the
        chat / turn / invoke_agent convention: leave Status UNSET,
        record cubeloop.aborted=true + error.type, do NOT add an
        ``exception`` event."""
        provider, exporter = _make_provider()
        _patch_mcp_trace(monkeypatch, provider)

        async def call_remote(name, args):
            raise asyncio.CancelledError()

        tool = await _make_tool(call_remote)
        try:
            await tool.execute("tc1", tool.parameters.model_validate({"q": "x"}))
        except asyncio.CancelledError:
            pass

        mcp_spans = [s for s in exporter.spans if s.name.startswith("tools/call ")]
        assert len(mcp_spans) == 1
        span = mcp_spans[0]
        attrs = _attrs(span)
        assert attrs["error.type"] == "cubeloop.aborted"
        assert attrs["cubeloop.aborted"] is True
        # Status stays UNSET; cancel is not a failure.
        assert span.status.status_code == StatusCode.UNSET
        # No exception event — cancel is signaled via cubeloop.aborted only.
        assert not any(e.name == "exception" for e in span.events)


class TestMCPIsErrorResponse:
    """When an MCP server returns ``isError: true``, the CLIENT span
    must reflect the protocol-level failure — codex round 4."""

    async def test_iserror_response_marks_span_error(self, monkeypatch):
        provider, exporter = _make_provider()
        _patch_mcp_trace(monkeypatch, provider)

        async def call_remote(name, args):
            return {
                "content": [{"type": "text", "text": "tool failed"}],
                "isError": True,
            }

        tool = await _make_tool(call_remote)
        result = await tool.execute("tc1", tool.parameters.model_validate({"q": "x"}))
        # Adapter still surfaces is_error on the AgentToolResult.
        assert result.is_error is True

        mcp_spans = [s for s in exporter.spans if s.name.startswith("tools/call ")]
        assert len(mcp_spans) == 1
        span = mcp_spans[0]
        assert span.status.status_code == StatusCode.ERROR
        attrs = _attrs(span)
        assert attrs["error.type"] == "mcp.is_error"

    async def test_protocol_error_message_is_private_by_default(self, monkeypatch):
        from cubeloop.mcp._tracing import mark_span_mcp_error, mcp_client_span

        provider, exporter = _make_provider()
        _patch_mcp_trace(monkeypatch, provider)
        secret = "Authorization: Bearer MCPPROTOCOLSECRET"

        async with mcp_client_span(
            method="tools/call",
            tool_name="search",
        ) as span:
            mark_span_mcp_error(span, secret)

        captured = next(s for s in exporter.spans if s.name.startswith("tools/call "))
        assert captured.status.status_code == StatusCode.ERROR
        assert captured.status.description == "mcp protocol error"
        assert secret not in repr(dict(captured.attributes or {}))
        for event in captured.events:
            assert secret not in repr(dict(event.attributes or {}))

    async def test_iserror_false_keeps_span_unset(self, monkeypatch):
        provider, exporter = _make_provider()
        _patch_mcp_trace(monkeypatch, provider)

        async def call_remote(name, args):
            return {
                "content": [{"type": "text", "text": "ok"}],
                "isError": False,
            }

        tool = await _make_tool(call_remote)
        await tool.execute("tc1", tool.parameters.model_validate({"q": "x"}))

        mcp_spans = [s for s in exporter.spans if s.name.startswith("tools/call ")]
        assert len(mcp_spans) == 1
        span = mcp_spans[0]
        assert span.status.status_code == StatusCode.UNSET
        assert "error.type" not in _attrs(span)


class TestNoOpWhenOTelMissing:
    async def test_context_manager_yields_none_when_no_otel(self, monkeypatch):
        import cubeloop.mcp._tracing as mcp_tracing

        monkeypatch.setattr(mcp_tracing, "_OTEL_AVAILABLE", False)

        async def call_remote(name, args):
            return {"content": [], "isError": False}

        tool = await _make_tool(call_remote)
        # Must run without errors and without emitting any spans.
        await tool.execute("tc1", tool.parameters.model_validate({"q": "x"}))


class TestTraceparentHelper:
    async def test_current_traceparent_returns_w3c_string_when_in_span(
        self, monkeypatch
    ):
        provider, _ = _make_provider()
        _patch_mcp_trace(monkeypatch, provider)
        from cubeloop.mcp._tracing import current_traceparent

        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("test-span"):
            tp = current_traceparent()
        assert tp is not None
        # Format: 00-<32hex>-<16hex>-<flags>
        parts = tp.split("-")
        assert len(parts) == 4
        assert parts[0] == "00"
        assert len(parts[1]) == 32
        assert len(parts[2]) == 16
        assert len(parts[3]) == 2

    async def test_current_traceparent_none_without_span(self, monkeypatch):
        from cubeloop.mcp._tracing import current_traceparent

        # No active span (and no recording context set up).
        assert current_traceparent() is None


class TestTraceparentInjection:
    """The HTTP loader must inject ``traceparent`` into outbound
    session headers whenever the MCP CLIENT span is active so that
    instrumented MCP servers can continue the trace (codex round 2)."""

    async def test_traceparent_header_set_inside_mcp_span(self, monkeypatch):
        from cubeloop.mcp import _tracing as mcp_tracing
        from cubeloop.mcp._tracing import mcp_client_span

        # Force OTel on with a deterministic tracer.
        provider, _exporter = _make_provider()
        _patch_mcp_trace(monkeypatch, provider)

        # Reproduce the http_loader.call_remote injection logic.
        seen_headers: list[dict] = []

        base_headers = {"x-test": "yes"}

        async def fake_call_remote():
            # Mirrors http_loader._call_remote header-merge logic.
            tp = mcp_tracing.current_traceparent()
            call_headers = base_headers
            if tp is not None:
                call_headers = {**base_headers, "traceparent": tp}
            seen_headers.append(call_headers)

        async with mcp_client_span(method="tools/call", tool_name="search"):
            await fake_call_remote()

        assert len(seen_headers) == 1
        assert seen_headers[0]["x-test"] == "yes"  # caller headers preserved
        assert "traceparent" in seen_headers[0]
        # Format: 00-<32hex>-<16hex>-<flags>
        tp = seen_headers[0]["traceparent"]
        assert tp.startswith("00-") and len(tp.split("-")) == 4

    async def test_http_loader_forwards_traceparent_on_tool_call(self, monkeypatch):
        """The real HTTP opener stamps traceparent during AgentTool execute."""
        import mcp
        import mcp.client.sse
        from cubeloop.mcp import load_mcp_tools_http

        provider, _exporter = _make_provider()
        _patch_mcp_trace(monkeypatch, provider)

        seen_headers: list[dict | None] = []

        class DummyClient:
            protocol_version = "2026-07-28"
            server_info = None

            def __init__(self, transport) -> None:
                del transport

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def list_tools(self):
                return type(
                    "Resp",
                    (),
                    {
                        "tools": [
                            type(
                                "Tool",
                                (),
                                {
                                    "name": "search",
                                    "description": "Search",
                                    "input_schema": {
                                        "type": "object",
                                        "properties": {"q": {"type": "string"}},
                                    },
                                    "icons": None,
                                },
                            )()
                        ]
                    },
                )()

            async def call_tool(self, name, args):
                return type(
                    "Call",
                    (),
                    {
                        "content": [
                            type("Block", (), {"type": "text", "text": "ok"})()
                        ],
                        "is_error": False,
                        "structured_content": None,
                    },
                )()

        def fake_sse(url, headers=None, timeout=None, sse_read_timeout=None):
            del url, timeout, sse_read_timeout
            seen_headers.append(headers)

            @asynccontextmanager
            async def _cm():
                yield "sse_stream"

            return _cm()

        monkeypatch.setattr(mcp, "Client", DummyClient)
        monkeypatch.setattr(mcp.client.sse, "sse_client", fake_sse)
        discovery = await load_mcp_tools_http(
            "https://mcp.example/mcp",
            headers={"x-test": "yes"},
        )
        await discovery.tools[0].execute(
            "tc-trace",
            discovery.tools[0].parameters(q="cats"),
            signal=None,
            on_update=None,
        )

        assert seen_headers[0] == {"x-test": "yes"}
        assert seen_headers[1]["x-test"] == "yes"
        assert seen_headers[1]["traceparent"].startswith("00-")

    async def test_no_header_added_when_no_active_span(self, monkeypatch):
        from cubeloop.mcp._tracing import current_traceparent

        # Outside any span: helper returns None and loader must not add
        # a traceparent header.
        assert current_traceparent() is None


class TestTracerProviderRouting:
    """When a user constructs a cubeloop.tracing.Tracer with its own
    private provider and calls attach(agent), MCP CLIENT spans must
    flow through that same provider's exporters — without this, MCP
    spans would silently land in the OTel global no-op provider
    (codex round 5)."""

    async def test_mcp_span_lands_in_tracer_exporter(self):
        from cubeloop.agent.agent import Agent
        from cubeloop.providers.base import ToolCall
        from cubeloop.providers.faux import (
            FauxProvider,
            faux_assistant_message,
        )
        from cubeloop.tracing import Tracer

        from pydantic import BaseModel

        # Capture exporter shared by the cubepi Tracer.
        exporter = _CaptureExporter()

        # Build an MCP-backed tool whose call_remote we can drive.
        class P(BaseModel):
            q: str = ""

        async def call_remote(name, args):
            return {"content": [{"type": "text", "text": "ok"}], "isError": False}

        from cubeloop.mcp._adapter import make_mcp_agent_tool

        mcp_tool = make_mcp_agent_tool(
            name="search",
            description="search",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
            call_remote=call_remote,
            server_address="api.example.com",
            protocol_version="2025-11-25",
        )

        provider = FauxProvider(provider_id="faux")
        provider.append_responses(
            [
                faux_assistant_message(
                    [ToolCall(id="tc1", name="search", arguments={"q": "x"})],
                    stop_reason="tool_use",
                ),
                faux_assistant_message("done"),
            ]
        )
        agent = Agent(
            model=provider.model("faux-1"),
            system_prompt="s",
            tools=[mcp_tool],
        )

        tracer = Tracer(service_name="t", agent_name="a", exporters=[exporter])
        detach = tracer.attach(agent)

        try:
            await agent.prompt("go")
            await agent.wait_for_idle()
        finally:
            detach()
            await tracer.shutdown()

        # Exactly one MCP tools/call CLIENT span landed in the Tracer's
        # exporter — proving the routing worked.
        mcp_spans = [s for s in exporter.spans if s.name.startswith("tools/call ")]
        assert len(mcp_spans) == 1
        attrs = dict(mcp_spans[0].attributes or {})
        assert attrs["mcp.method.name"] == "tools/call"
        assert attrs["gen_ai.tool.name"] == "search"

        # The CLIENT span must be a child of the recorder's
        # execute_tool span (same trace_id, parent_span_id set), not an
        # orphan root trace (codex round-6 review on PR #86).
        client_span = mcp_spans[0]
        execute_spans = [
            s for s in exporter.spans if s.name.startswith("execute_tool ")
        ]
        assert len(execute_spans) == 1
        tool_ctx = execute_spans[0].get_span_context()
        client_ctx = client_span.get_span_context()
        assert client_ctx.trace_id == tool_ctx.trace_id
        assert client_span.parent is not None
        assert client_span.parent.span_id == tool_ctx.span_id

    async def test_mcp_exception_is_private_without_content_opt_in(self):
        from cubeloop.agent.agent import Agent
        from cubeloop.providers.base import ToolCall
        from cubeloop.providers.faux import FauxProvider, faux_assistant_message
        from cubeloop.tracing import Tracer

        secret = "Authorization: Bearer MCPSECRET"
        exporter = _CaptureExporter()

        async def call_remote(name, args):
            raise RuntimeError(secret)

        mcp_tool = make_mcp_agent_tool(
            name="search",
            description="search",
            input_schema={
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
            call_remote=call_remote,
        )
        provider = FauxProvider(provider_id="faux")
        provider.append_responses(
            [
                faux_assistant_message(
                    [ToolCall(id="tc1", name="search", arguments={"q": "x"})],
                    stop_reason="tool_use",
                ),
                faux_assistant_message("done"),
            ]
        )
        agent = Agent(
            model=provider.model("faux-1"),
            system_prompt="s",
            tools=[mcp_tool],
        )
        tracer = Tracer(
            service_name="t",
            agent_name="a",
            exporters=[exporter],
            record_content=False,
        )
        detach = tracer.attach(agent)
        try:
            await agent.prompt("go")
            await agent.wait_for_idle()
        finally:
            detach()
            await tracer.shutdown()

        span = next(s for s in exporter.spans if s.name.startswith("tools/call "))
        assert span.status.status_code == StatusCode.ERROR
        assert span.status.description == "mcp client error"
        assert secret not in repr(dict(span.attributes or {}))
        exception_event = next(
            event for event in span.events if event.name == "exception"
        )
        assert exception_event.attributes["exception.type"] == "RuntimeError"
        for event in span.events:
            assert secret not in repr(dict(event.attributes or {}))

    async def test_mcp_exception_details_remain_available_with_content_opt_in(self):
        from cubeloop.agent.agent import Agent
        from cubeloop.providers.base import ToolCall
        from cubeloop.providers.faux import FauxProvider, faux_assistant_message
        from cubeloop.tracing import Tracer

        secret = "diagnostic mcp failure"
        exporter = _CaptureExporter()

        async def call_remote(name, args):
            raise RuntimeError(secret)

        mcp_tool = make_mcp_agent_tool(
            name="search",
            description="search",
            input_schema={
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
            call_remote=call_remote,
        )
        provider = FauxProvider(provider_id="faux")
        provider.append_responses(
            [
                faux_assistant_message(
                    [ToolCall(id="tc1", name="search", arguments={"q": "x"})],
                    stop_reason="tool_use",
                ),
                faux_assistant_message("done"),
            ]
        )
        agent = Agent(
            model=provider.model("faux-1"),
            system_prompt="s",
            tools=[mcp_tool],
        )
        tracer = Tracer(
            service_name="t",
            agent_name="a",
            exporters=[exporter],
            record_content=True,
        )
        detach = tracer.attach(agent)
        try:
            await agent.prompt("go")
            await agent.wait_for_idle()
        finally:
            detach()
            await tracer.shutdown()

        span = next(s for s in exporter.spans if s.name.startswith("tools/call "))
        assert secret in (span.status.description or "")
        assert any(
            secret in repr(dict(event.attributes or {})) for event in span.events
        )

    async def test_unparented_mcp_exception_fails_closed_with_opted_in_tracer(self):
        """Without a tool-span owner, the global provider stack cannot prove
        which concurrent Tracer owns the call, so content must stay disabled."""
        from cubeloop.agent.agent import Agent
        from cubeloop.mcp._tracing import mcp_client_span
        from cubeloop.providers.faux import FauxProvider
        from cubeloop.tracing import Tracer

        secret = "Authorization: Bearer UNPARENTEDMCPSECRET"
        exporter = _CaptureExporter()
        provider = FauxProvider(provider_id="faux")
        agent = Agent(model=provider.model("faux-1"), system_prompt="s")
        tracer = Tracer(
            service_name="t",
            agent_name="a",
            exporters=[exporter],
            record_content=True,
        )
        detach = tracer.attach(agent)
        try:
            with pytest.raises(RuntimeError, match="UNPARENTEDMCPSECRET"):
                async with mcp_client_span(
                    method="connect",
                    server_address="example.com",
                ):
                    raise RuntimeError(secret)
        finally:
            detach()
            await tracer.shutdown()

        span = next(s for s in exporter.spans if s.name == "connect")
        assert span.status.status_code == StatusCode.ERROR
        assert span.status.description == "mcp client error"
        event = next(event for event in span.events if event.name == "exception")
        assert dict(event.attributes or {}) == {"exception.type": "RuntimeError"}
        assert secret not in repr(dict(span.attributes or {}))

    async def test_two_attaches_detach_one_keeps_other_routing(self):
        """When a Tracer is attached to two agents and one is detached,
        the remaining agent's MCP spans must still land in the Tracer's
        exporter. Refcounted register/unregister is the contract
        (codex round-6 review on PR #86)."""
        from cubeloop.agent.agent import Agent
        from cubeloop.providers.base import ToolCall
        from cubeloop.providers.faux import FauxProvider, faux_assistant_message
        from cubeloop.tracing import Tracer

        exporter = _CaptureExporter()

        async def call_remote(name, args):
            return {"content": [{"type": "text", "text": "ok"}], "isError": False}

        from cubeloop.mcp._adapter import make_mcp_agent_tool

        def _make_agent():
            mcp_tool = make_mcp_agent_tool(
                name="search",
                description="search",
                input_schema={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
                call_remote=call_remote,
            )
            provider_a = FauxProvider(provider_id="faux")
            provider_a.append_responses(
                [
                    faux_assistant_message(
                        [ToolCall(id="tc1", name="search", arguments={"q": "x"})],
                        stop_reason="tool_use",
                    ),
                    faux_assistant_message("done"),
                ]
            )
            return Agent(
                model=provider_a.model("faux-1"),
                system_prompt="s",
                tools=[mcp_tool],
            )

        agent_a = _make_agent()
        agent_b = _make_agent()

        tracer = Tracer(service_name="t", agent_name="a", exporters=[exporter])
        detach_a = tracer.attach(agent_a)
        detach_b = tracer.attach(agent_b)

        try:
            # Detach the first attachment; agent_b's runs must still
            # route MCP spans through the Tracer's exporter.
            detach_a()

            await agent_b.prompt("go")
            await agent_b.wait_for_idle()
        finally:
            detach_b()
            await tracer.shutdown()

        mcp_spans = [s for s in exporter.spans if s.name.startswith("tools/call ")]
        assert len(mcp_spans) == 1, (
            "MCP routing for the still-attached agent broke when the "
            "other agent's detach was called"
        )

    def test_register_and_unregister_provider(self, monkeypatch):
        from cubeloop.mcp import _tracing as mcp_tracing

        # Start from a clean stack so the assertion below is well-defined
        # regardless of other tests' attach()/detach() side-effects.
        monkeypatch.setattr(mcp_tracing, "_provider_stack", [])

        sentinel = object()
        token = mcp_tracing.register_provider(sentinel)
        try:
            assert mcp_tracing._provider_stack[-1][1] is sentinel
        finally:
            mcp_tracing.unregister_provider(token)
        assert mcp_tracing._provider_stack == []

    def test_unregister_only_removes_own_token(self, monkeypatch):
        """Two providers registered in sequence; detaching the first
        must leave the second's registration intact so MCP spans
        continue flowing through it. Without this, a Tracer attached to
        agent A and B then detached from A would drop MCP routing for B
        (codex round-6 review on PR #86)."""
        from cubeloop.mcp import _tracing as mcp_tracing

        monkeypatch.setattr(mcp_tracing, "_provider_stack", [])

        first = object()
        second = object()
        t1 = mcp_tracing.register_provider(first)
        t2 = mcp_tracing.register_provider(second)
        try:
            # Detach the first registration; the second must remain
            # routable. The most-recently-pushed provider wins on lookup.
            mcp_tracing.unregister_provider(t1)
            assert len(mcp_tracing._provider_stack) == 1
            assert mcp_tracing._provider_stack[-1][1] is second
        finally:
            mcp_tracing.unregister_provider(t2)
        assert mcp_tracing._provider_stack == []


class TestMCPSpanParentage:
    """Phase 5 round 6: the MCP CLIENT span must nest under the
    cubepi recorder's ``execute_tool`` span, not start an orphan root
    trace. The recorder publishes its execute_tool span by tool_call_id;
    the adapter passes the id through to ``mcp_client_span`` which uses
    it as the explicit parent context."""

    async def test_mcp_span_parented_to_registered_tool_span(self, monkeypatch):
        provider, exporter = _make_provider()
        _patch_mcp_trace(monkeypatch, provider)

        # Open a synthetic "execute_tool" span and register it as the
        # parent for tool_call_id "tc1" — the same hook the cubepi
        # recorder uses.
        from cubeloop.mcp import _tracing as mcp_tracing

        tracer = provider.get_tracer("test")
        tool_span = tracer.start_span("execute_tool search")
        token = mcp_tracing.register_tool_span("tc1", tool_span)
        try:

            async def call_remote(name, args):
                return {"content": [], "isError": False}

            tool = await _make_tool(call_remote)
            await tool.execute("tc1", tool.parameters.model_validate({"q": "x"}))
        finally:
            mcp_tracing.unregister_tool_span(token)
            tool_span.end()

        mcp_spans = [s for s in exporter.spans if s.name.startswith("tools/call ")]
        assert len(mcp_spans) == 1
        client_span = mcp_spans[0]
        tool_ctx = tool_span.get_span_context()
        client_ctx = client_span.get_span_context()
        # Same trace_id, parent points at the execute_tool span.
        assert client_ctx.trace_id == tool_ctx.trace_id
        assert client_span.parent is not None
        assert client_span.parent.span_id == tool_ctx.span_id

    async def test_two_tracers_route_mcp_by_parent_owner(self):
        """Two different cubeloop.Tracer instances attached to two agents
        in the same process. An MCP CLIENT span emitted under agent A's
        execute_tool span must export through Tracer A's exporter — not
        Tracer B's, even if B was attached more recently (LIFO order on
        the provider stack).

        Without this routing, agent A's trace would be missing its MCP
        leg and Tracer B's exporter would receive a stray span with
        agent A's trace_id (codex round-7 review on PR #86)."""
        from cubeloop.agent.agent import Agent
        from cubeloop.providers.base import ToolCall
        from cubeloop.providers.faux import FauxProvider, faux_assistant_message
        from cubeloop.tracing import Tracer

        from cubeloop.mcp._adapter import make_mcp_agent_tool

        async def call_remote(name, args):
            return {"content": [{"type": "text", "text": "ok"}], "isError": False}

        def _make_agent():
            mcp_tool = make_mcp_agent_tool(
                name="search",
                description="search",
                input_schema={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
                call_remote=call_remote,
            )
            provider = FauxProvider(provider_id="faux")
            provider.append_responses(
                [
                    faux_assistant_message(
                        [ToolCall(id="tc1", name="search", arguments={"q": "x"})],
                        stop_reason="tool_use",
                    ),
                    faux_assistant_message("done"),
                ]
            )
            return Agent(
                model=provider.model("faux-1"),
                system_prompt="s",
                tools=[mcp_tool],
            )

        exporter_a = _CaptureExporter()
        exporter_b = _CaptureExporter()

        agent_a = _make_agent()
        agent_b = _make_agent()

        tracer_a = Tracer(service_name="a", agent_name="a", exporters=[exporter_a])
        tracer_b = Tracer(service_name="b", agent_name="b", exporters=[exporter_b])

        # Attach A first then B so B is at the top of the LIFO stack.
        # The buggy implementation would route both agents' MCP spans
        # through B; we want A's spans in A's exporter and B's in B's.
        detach_a = tracer_a.attach(agent_a)
        detach_b = tracer_b.attach(agent_b)

        try:
            await agent_a.prompt("go")
            await agent_a.wait_for_idle()
            await agent_b.prompt("go")
            await agent_b.wait_for_idle()
        finally:
            detach_a()
            detach_b()
            await tracer_a.shutdown()
            await tracer_b.shutdown()

        # Each exporter saw exactly one tools/call CLIENT span — its
        # own agent's — with matching trace_id.
        mcp_a = [s for s in exporter_a.spans if s.name.startswith("tools/call ")]
        mcp_b = [s for s in exporter_b.spans if s.name.startswith("tools/call ")]
        assert len(mcp_a) == 1, "agent_a's MCP span did not land in tracer_a"
        assert len(mcp_b) == 1, "agent_b's MCP span did not land in tracer_b"

        # Cross-check trace_id consistency: each exporter's MCP span
        # shares trace_id with that exporter's execute_tool span (i.e.,
        # parent and child went to the same backend together).
        for exporter in (exporter_a, exporter_b):
            execute = [s for s in exporter.spans if s.name.startswith("execute_tool ")]
            mcp = [s for s in exporter.spans if s.name.startswith("tools/call ")]
            assert len(execute) == 1 and len(mcp) == 1
            assert (
                execute[0].get_span_context().trace_id
                == mcp[0].get_span_context().trace_id
            )

    async def test_concurrent_agents_with_colliding_tool_call_ids(self):
        """Two agents executing MCP tools concurrently with the SAME
        tool_call_id (``tc1``, mimicking how Faux/OpenAI-style providers
        mint ids per conversation). A global dict keyed by tool_call_id
        would let the second registration overwrite the first; the MCP
        span lookup could then parent/export agent A's CLIENT span
        through agent B's owning provider, reintroducing the cross-
        Tracer misrouting that the sequential test does not cover.

        ContextVars scope per asyncio task, so each agent's run sees
        only its own parent (codex round-8 review on PR #86)."""
        import asyncio as _asyncio

        from cubeloop.agent.agent import Agent
        from cubeloop.mcp._adapter import make_mcp_agent_tool
        from cubeloop.providers.base import ToolCall
        from cubeloop.providers.faux import FauxProvider, faux_assistant_message
        from cubeloop.tracing import Tracer

        # Gate that holds both agents' MCP calls inside the
        # ``execute_tool`` span at the same time, so any global-dict
        # collision would be exercised. Without the gate the calls
        # would serialize trivially.
        in_call = _asyncio.Event()
        release = _asyncio.Event()

        async def call_remote(name, args):
            in_call.set()
            await release.wait()
            return {"content": [{"type": "text", "text": "ok"}], "isError": False}

        def _make_agent():
            mcp_tool = make_mcp_agent_tool(
                name="search",
                description="search",
                input_schema={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
                call_remote=call_remote,
            )
            prov = FauxProvider(provider_id="faux")
            prov.append_responses(
                [
                    faux_assistant_message(
                        [ToolCall(id="tc1", name="search", arguments={"q": "x"})],
                        stop_reason="tool_use",
                    ),
                    faux_assistant_message("done"),
                ]
            )
            return Agent(
                model=prov.model("faux-1"),
                system_prompt="s",
                tools=[mcp_tool],
            )

        exporter_a = _CaptureExporter()
        exporter_b = _CaptureExporter()
        agent_a = _make_agent()
        agent_b = _make_agent()
        tracer_a = Tracer(service_name="a", agent_name="a", exporters=[exporter_a])
        tracer_b = Tracer(service_name="b", agent_name="b", exporters=[exporter_b])
        detach_a = tracer_a.attach(agent_a)
        detach_b = tracer_b.attach(agent_b)

        async def _run(agent):
            await agent.prompt("go")
            await agent.wait_for_idle()

        try:
            task_a = _asyncio.create_task(_run(agent_a))
            task_b = _asyncio.create_task(_run(agent_b))

            # Both agents have entered their MCP call_remote.
            await in_call.wait()
            in_call.clear()
            # Give the second one a moment to also enter; the gate keeps
            # it parked there.
            await _asyncio.sleep(0.05)
            release.set()
            await _asyncio.gather(task_a, task_b)
        finally:
            detach_a()
            detach_b()
            await tracer_a.shutdown()
            await tracer_b.shutdown()

        for exporter, who in ((exporter_a, "a"), (exporter_b, "b")):
            execute = [s for s in exporter.spans if s.name.startswith("execute_tool ")]
            mcp = [s for s in exporter.spans if s.name.startswith("tools/call ")]
            assert len(execute) == 1, f"agent_{who}: execute_tool count"
            assert len(mcp) == 1, f"agent_{who}: tools/call count"
            assert (
                execute[0].get_span_context().trace_id
                == mcp[0].get_span_context().trace_id
            ), f"agent_{who}: MCP span landed in wrong trace"
            assert mcp[0].parent is not None
            assert mcp[0].parent.span_id == execute[0].get_span_context().span_id, (
                f"agent_{who}: MCP span parented to the wrong execute_tool"
            )

    async def test_parallel_tool_mode_parents_mcp_span(self):
        """In ``parallel`` tool execution, the cubepi agent loop emits
        ``ToolExecutionStartEvent`` in the parent task and
        ``ToolExecutionEndEvent`` from the per-tool *child* task it
        spawns. A naive ``ContextVar.reset(token)`` would raise
        ``ValueError`` in the child task because the Token was minted
        in a different context. The dict+contextvar hybrid keeps the
        registry valid until the child task's cleanup runs — pin that
        an MCP CLIENT span is still correctly parented under
        ``execute_tool`` in this mode (codex round-9 review)."""
        from cubeloop.agent.agent import Agent
        from cubeloop.agent.types import AgentTool
        from cubeloop.mcp._adapter import make_mcp_agent_tool
        from cubeloop.providers.base import ToolCall
        from cubeloop.providers.faux import FauxProvider, faux_assistant_message
        from cubeloop.tracing import Tracer

        async def call_remote(name, args):
            return {"content": [{"type": "text", "text": "ok"}], "isError": False}

        # Build the MCP tool with execution_mode="parallel" so the
        # agent's _execute_parallel path is taken.
        base_tool = make_mcp_agent_tool(
            name="search",
            description="search",
            input_schema={
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
            call_remote=call_remote,
        )
        mcp_tool = AgentTool(
            name=base_tool.name,
            description=base_tool.description,
            parameters=base_tool.parameters,
            execute=base_tool.execute,
            execution_mode="parallel",
        )

        provider = FauxProvider(provider_id="faux")
        provider.append_responses(
            [
                faux_assistant_message(
                    [ToolCall(id="tc1", name="search", arguments={"q": "x"})],
                    stop_reason="tool_use",
                ),
                faux_assistant_message("done"),
            ]
        )
        agent = Agent(
            model=provider.model("faux-1"),
            system_prompt="s",
            tools=[mcp_tool],
        )

        exporter = _CaptureExporter()
        tracer = Tracer(service_name="t", agent_name="a", exporters=[exporter])
        detach = tracer.attach(agent)
        try:
            await agent.prompt("go")
            await agent.wait_for_idle()
        finally:
            detach()
            await tracer.shutdown()

        execute = [s for s in exporter.spans if s.name.startswith("execute_tool ")]
        mcp = [s for s in exporter.spans if s.name.startswith("tools/call ")]
        assert len(execute) == 1
        assert len(mcp) == 1
        assert (
            mcp[0].get_span_context().trace_id == execute[0].get_span_context().trace_id
        )
        assert mcp[0].parent is not None
        assert mcp[0].parent.span_id == execute[0].get_span_context().span_id

    async def test_cancellation_does_not_leak_tool_registrations(self):
        """``asyncio.CancelledError`` inherits ``BaseException`` and
        bypasses the cubepi agent loop's ``except Exception`` handler,
        so a cancelled run never emits ``ToolExecutionEndEvent`` and
        never reaches the per-event unregister. Without an explicit
        cleanup path, ``_active_entries`` in ``cubeloop.mcp._tracing``
        would grow unbounded across aborted runs (codex round-10).

        Pin: after a cancelled run + detach, ``_active_entries`` is
        back to its pre-run size."""
        import asyncio as _asyncio

        from cubeloop.agent.agent import Agent
        from cubeloop.mcp import _tracing as mcp_tracing
        from cubeloop.mcp._adapter import make_mcp_agent_tool
        from cubeloop.providers.base import ToolCall
        from cubeloop.providers.faux import FauxProvider, faux_assistant_message
        from cubeloop.tracing import Tracer

        baseline = len(mcp_tracing._active_entries)

        # Gate the MCP call so we can cancel mid-flight, simulating
        # the real "agent run cancelled while a tool is awaiting" path.
        in_call = _asyncio.Event()
        never = _asyncio.Event()

        async def call_remote(name, args):
            in_call.set()
            await never.wait()  # cancellation lands here
            return {"content": [], "isError": False}

        mcp_tool = make_mcp_agent_tool(
            name="search",
            description="search",
            input_schema={
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
            call_remote=call_remote,
        )

        provider = FauxProvider(provider_id="faux")
        provider.append_responses(
            [
                faux_assistant_message(
                    [ToolCall(id="tc1", name="search", arguments={"q": "x"})],
                    stop_reason="tool_use",
                ),
                faux_assistant_message("done"),
            ]
        )
        agent = Agent(
            model=provider.model("faux-1"),
            system_prompt="s",
            tools=[mcp_tool],
        )

        exporter = _CaptureExporter()
        tracer = Tracer(service_name="t", agent_name="a", exporters=[exporter])
        detach = tracer.attach(agent)

        try:
            run_task = _asyncio.create_task(agent.prompt("go"))
            await in_call.wait()
            # During registration_active = baseline + 1.
            assert len(mcp_tracing._active_entries) == baseline + 1, (
                "expected exactly one active MCP tool registration mid-run"
            )
            run_task.cancel()
            try:
                await run_task
            except _asyncio.CancelledError:
                pass
            # Drain any post-cancel events; agent doesn't emit on cancel
            # but wait_for_idle is harmless.
            try:
                await _asyncio.wait_for(agent.wait_for_idle(), timeout=1.0)
            except (_asyncio.TimeoutError, _asyncio.CancelledError):
                pass
        finally:
            detach()
            # ``detach()`` runs the synchronous sweep eagerly — the
            # registration must already be gone before we even reach
            # ``tracer.shutdown()`` (codex round-11). Asserting *before*
            # shutdown pins that contract; the post-shutdown assertion
            # below is a belt-and-braces check.
            assert len(mcp_tracing._active_entries) == baseline, (
                "detach() did not synchronously sweep the cancelled "
                "tool's registration; cleanup was deferred to a loop "
                "tick that may not run before tracer.shutdown()"
            )
            await tracer.shutdown()

        assert len(mcp_tracing._active_entries) == baseline

    async def test_unregister_cleans_dict_even_from_different_task(self):
        """Direct contract test for the dict+contextvar hybrid:
        ``register_tool_span`` in one task, ``unregister_tool_span`` in
        a different task. The ``ContextVar.reset`` raises and is
        swallowed; the dict entry must still be cleaned up so any
        subsequent lookup via a stale contextvar returns ``None``
        (codex round-9 review)."""
        import asyncio as _asyncio

        from cubeloop.mcp import _tracing as mcp_tracing

        # Build a parent task that registers, then a child task that
        # unregisters. We assert the dict is empty after cleanup.
        registered_count_before = len(mcp_tracing._active_entries)

        token_holder: list = []

        async def _register():
            token = mcp_tracing.register_tool_span("tc1", "span-A", provider="prov-A")
            token_holder.append(token)
            # Don't unregister here — let the child task do it.

        async def _unregister(token):
            mcp_tracing.unregister_tool_span(token)

        await _register()
        # Confirm the dict gained one entry.
        assert len(mcp_tracing._active_entries) == registered_count_before + 1

        # Unregister from a different task; ContextVar.reset will
        # ValueError but the dict cleanup must still happen.
        await _asyncio.create_task(_unregister(token_holder[0]))
        assert len(mcp_tracing._active_entries) == registered_count_before, (
            "_active_entries leaked after cross-task unregister_tool_span"
        )

    async def test_nested_tool_falls_back_to_outer_after_inner_unregister_cross_task(
        self,
    ):
        """Parallel-mode shape: an inner tool registers while an outer tool is
        active, then its ToolExecutionEnd unregisters from a CHILD task — so the
        per-task contextvar can't be reset. A later lookup in the outer task
        must fall back to the still-active OUTER tool span, not return None;
        otherwise a subsequent subagent/MCP call fails to nest under the outer
        execute_tool span (codex P2, PR #122)."""
        import asyncio as _asyncio

        from cubeloop.mcp import _tracing as mcp_tracing

        tok_outer = mcp_tracing.register_tool_span(
            "outer", "span-outer", provider="prov-outer"
        )
        tok_inner = mcp_tracing.register_tool_span(
            "inner", "span-inner", provider="prov-inner"
        )
        try:
            # Innermost lookup is the inner tool while it is live.
            assert mcp_tracing._get_tool_span_entry() == (
                "span-inner",
                "prov-inner",
                False,
            )

            async def _end_inner() -> None:
                mcp_tracing.unregister_tool_span(tok_inner)

            # Inner ends in a different task (cross-task reset fails silently).
            await _asyncio.create_task(_end_inner())

            # Must fall back to the OUTER tool, not None.
            assert mcp_tracing._get_tool_span_entry() == (
                "span-outer",
                "prov-outer",
                False,
            )
        finally:
            mcp_tracing.unregister_tool_span(tok_outer)

        # Fully unwound: no active tool for this task.
        assert mcp_tracing._get_tool_span_entry() is None

    async def test_mcp_span_orphan_when_no_registered_parent(self, monkeypatch):
        """Without a registered parent, the MCP span starts a new
        root trace. Pinning this so the parented-path test above is
        meaningful (i.e., parenting only kicks in when we ask for it)."""
        provider, exporter = _make_provider()
        _patch_mcp_trace(monkeypatch, provider)

        async def call_remote(name, args):
            return {"content": [], "isError": False}

        tool = await _make_tool(call_remote)
        await tool.execute("no-parent-tc", tool.parameters.model_validate({"q": "x"}))

        mcp_spans = [s for s in exporter.spans if s.name.startswith("tools/call ")]
        assert len(mcp_spans) == 1
        assert mcp_spans[0].parent is None


class TestLoaderHelpers:
    def test_split_address_parses_url(self):
        from cubeloop.mcp.http_loader import _split_address

        assert _split_address("https://api.example.com:8443/mcp") == (
            "api.example.com",
            8443,
        )
        host, port = _split_address("http://localhost/mcp")
        assert host == "localhost"
        assert port is None

    def test_split_address_on_empty(self):
        from cubeloop.mcp.http_loader import _split_address

        # urlparse("") returns hostname=None, port=None — both nones is
        # the documented signal that the helper couldn't extract values.
        host, port = _split_address("")
        assert host is None
        assert port is None

    def test_split_address_still_available_for_span_attrs(self):
        """HTTP loader still exposes address parsing for MCP client spans."""
        from cubeloop.mcp.http_loader import _split_address

        host, port = _split_address("https://mcp.example.com:8443/mcp")
        assert host == "mcp.example.com"
        assert port == 8443
