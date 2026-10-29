"""Tests for the MCP v2 HTTP tool loader."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from cubeloop.mcp.http_loader import _open_client, open_mcp_http_client


class _FakeClient:
    def __init__(self, *, protocol_version: str = "2026-07-28") -> None:
        self.protocol_version = protocol_version
        self.server_info = SimpleNamespace(
            name="fake-server",
            version="2.0.0",
            website_url="https://example.test",
            icons=None,
        )
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def list_tools(self):
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="search",
                    description="Search the web",
                    input_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                    icons=None,
                )
            ]
        )

    async def call_tool(self, name: str, args: dict[str, object]):
        self.calls.append((name, args))
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="hello"),
                SimpleNamespace(
                    type="image", data="base64payload", mime_type="image/png"
                ),
            ],
            is_error=False,
            structured_content={"source": "fake"},
        )


def _install_fake_client(monkeypatch, *, protocol_version: str = "2026-07-28"):
    from cubeloop.mcp import http_loader

    clients: list[_FakeClient] = []

    @asynccontextmanager
    async def fake_open_client(*args, **kwargs):
        client = _FakeClient(protocol_version=protocol_version)
        clients.append(client)
        yield client

    monkeypatch.setattr(http_loader, "open_mcp_http_client", fake_open_client)
    return clients


def test_import_http_loader() -> None:
    from cubeloop.mcp import load_mcp_tools_http

    assert callable(load_mcp_tools_http)


def test_mcp_package_exports_cached_descriptor_primitives() -> None:
    from cubeloop import mcp

    assert mcp.open_mcp_http_client is open_mcp_http_client
    assert callable(mcp.make_mcp_agent_tool)
    assert callable(mcp.serialize_mcp_call_tool_response)
    assert "open_mcp_http_client" in mcp.__all__
    assert "make_mcp_agent_tool" in mcp.__all__
    assert "serialize_mcp_call_tool_response" in mcp.__all__
    # CubePlex #703 imported the private name from http_loader.
    assert _open_client is open_mcp_http_client


@pytest.mark.asyncio
async def test_load_mcp_tools_http_uses_negotiated_v2_client(monkeypatch) -> None:
    """A modern connection exposes its negotiated version and serializes calls."""
    from cubeloop.mcp import load_mcp_tools_http
    from cubeloop.providers.base import ImageContent, TextContent

    clients = _install_fake_client(monkeypatch)
    discovery = await load_mcp_tools_http("https://mcp.example/mcp")

    assert discovery.server is not None
    assert discovery.server.name == "fake-server"
    assert discovery.tools[0].name == "search"
    args = discovery.tools[0].parameters(query="cats")
    result = await discovery.tools[0].execute("tc-1", args, signal=None, on_update=None)

    assert len(clients) == 2
    assert clients[1].calls == [("search", {"query": "cats"})]
    assert isinstance(result.content[0], TextContent)
    assert isinstance(result.content[1], ImageContent)
    assert result.content[1].media_type == "image/png"
    assert result.details["structuredContent"] == {"source": "fake"}


@pytest.mark.asyncio
async def test_load_mcp_tools_http_preserves_legacy_fallback_protocol(
    monkeypatch,
) -> None:
    """The loader accepts the legacy protocol selected by MCP Client auto mode."""
    from cubeloop.mcp import load_mcp_tools_http

    clients = _install_fake_client(monkeypatch, protocol_version="2025-11-25")
    discovery = await load_mcp_tools_http("https://mcp.example/mcp")

    tool = discovery.tools[0]
    assert tool.name == "search"
    args = tool.parameters(query="legacy")
    await tool.execute("tc-legacy", args, signal=None, on_update=None)
    assert len(clients) == 2


@pytest.mark.asyncio
async def test_load_mcp_tools_http_times_out_tool_calls(monkeypatch) -> None:
    from cubeloop.mcp import http_loader, load_mcp_tools_http

    class _SlowClient(_FakeClient):
        async def call_tool(self, name: str, args: dict[str, object]):
            await asyncio.sleep(1)

    @asynccontextmanager
    async def fake_open_client(*args, **kwargs):
        yield _SlowClient()

    monkeypatch.setattr(http_loader, "open_mcp_http_client", fake_open_client)
    discovery = await load_mcp_tools_http("https://mcp.example/mcp", timeout=0.01)
    with pytest.raises(TimeoutError):
        await discovery.tools[0].execute(
            "tc-timeout",
            discovery.tools[0].parameters(query="x"),
            signal=None,
            on_update=None,
        )


@pytest.mark.asyncio
async def test_load_mcp_tools_http_rejects_unknown_transport() -> None:
    from cubeloop.mcp import load_mcp_tools_http

    with pytest.raises(ValueError, match="unsupported MCP transport"):
        await load_mcp_tools_http("https://mcp.example/mcp", transport="websocket")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_open_client_streamable_http_branches(monkeypatch) -> None:
    """Directly test open_mcp_http_client with streamable_http transport."""
    captured_init = []

    class DummyClient:
        def __init__(self, transport_or_url):
            captured_init.append(transport_or_url)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    import mcp
    import mcp.client.streamable_http

    seen_clients: list[object] = []

    class _FakeHttpClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> None:
            self.closed = True

    def fake_create_mcp_http_client(**kwargs: object) -> _FakeHttpClient:
        client = _FakeHttpClient(**kwargs)
        seen_clients.append(client)
        return client

    @asynccontextmanager
    async def fake_streamable(url, http_client=None):
        yield http_client

    monkeypatch.setattr(mcp, "Client", DummyClient)
    monkeypatch.setattr(
        mcp.client.streamable_http, "streamable_http_client", fake_streamable
    )
    monkeypatch.setattr(
        "mcp.shared._httpx_utils.create_mcp_http_client",
        fake_create_mcp_http_client,
    )

    # Test with headers
    async with open_mcp_http_client(
        "https://mcp.example/mcp",
        headers={"Authorization": "Bearer test"},
        timeout=15.0,
        transport="streamable_http",
    ) as client:
        assert isinstance(client, DummyClient)
        assert seen_clients[-1].kwargs["headers"] == {"Authorization": "Bearer test"}
        assert seen_clients[-1].closed is False

    assert seen_clients[-1].closed is True

    # Test without headers
    async with open_mcp_http_client(
        "https://mcp.example/mcp",
        headers=None,
        timeout=15.0,
        transport="streamable_http",
    ) as client:
        assert isinstance(client, DummyClient)
        assert seen_clients[-1].kwargs["headers"] is None

    assert seen_clients[-1].closed is True


@pytest.mark.asyncio
async def test_open_client_sse_branch(monkeypatch) -> None:
    """Directly test open_mcp_http_client with sse transport."""
    captured_init = []

    class DummyClient:
        def __init__(self, transport):
            captured_init.append(transport)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    import mcp
    import mcp.client.sse

    @asynccontextmanager
    async def fake_sse(url, headers=None, timeout=None, sse_read_timeout=None):
        yield "sse_stream"

    monkeypatch.setattr(mcp, "Client", DummyClient)
    monkeypatch.setattr(mcp.client.sse, "sse_client", fake_sse)

    async with open_mcp_http_client(
        "https://mcp.example/sse",
        headers={"x-test": "1"},
        timeout=20.0,
        transport="sse",
    ) as client:
        assert isinstance(client, DummyClient)


_FAKE_TRACEPARENT = "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"


def _patch_traceparent(monkeypatch, value: str | None) -> None:
    from cubeloop.mcp import _tracing as mcp_tracing

    monkeypatch.setattr(mcp_tracing, "current_traceparent", lambda: value)


class _DummyMcpClient:
    def __init__(self, transport_or_url) -> None:
        del transport_or_url

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.mark.asyncio
async def test_open_mcp_http_client_stamps_traceparent_on_streamable_http(
    monkeypatch,
) -> None:
    """Opener stamps W3C traceparent without mutating the caller's headers dict."""
    import mcp
    import mcp.client.streamable_http

    captured_headers: list[object] = []
    caller_headers = {"Authorization": "Bearer test"}

    class _FakeHttpClient:
        def __init__(self, **kwargs: object) -> None:
            captured_headers.append(kwargs.get("headers"))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

    def fake_create_mcp_http_client(**kwargs: object) -> _FakeHttpClient:
        return _FakeHttpClient(**kwargs)

    @asynccontextmanager
    async def fake_streamable(url, http_client=None):
        yield http_client

    monkeypatch.setattr(mcp, "Client", _DummyMcpClient)
    monkeypatch.setattr(
        mcp.client.streamable_http, "streamable_http_client", fake_streamable
    )
    monkeypatch.setattr(
        "mcp.shared._httpx_utils.create_mcp_http_client",
        fake_create_mcp_http_client,
    )
    _patch_traceparent(monkeypatch, _FAKE_TRACEPARENT)

    async with open_mcp_http_client(
        "https://mcp.example/mcp",
        headers=caller_headers,
        timeout=15.0,
        transport="streamable_http",
    ):
        pass

    async with open_mcp_http_client(
        "https://mcp.example/mcp",
        headers=None,
        timeout=15.0,
        transport="streamable_http",
    ):
        pass

    assert captured_headers == [
        {
            "Authorization": "Bearer test",
            "traceparent": _FAKE_TRACEPARENT,
        },
        {"traceparent": _FAKE_TRACEPARENT},
    ]
    assert caller_headers == {"Authorization": "Bearer test"}


@pytest.mark.asyncio
async def test_open_mcp_http_client_stamps_traceparent_on_sse(monkeypatch) -> None:
    import mcp
    import mcp.client.sse

    captured_headers: list[object] = []

    def fake_sse(url, headers=None, timeout=None, sse_read_timeout=None):
        del url, timeout, sse_read_timeout
        captured_headers.append(headers)

        @asynccontextmanager
        async def _cm():
            yield "sse_stream"

        return _cm()

    monkeypatch.setattr(mcp, "Client", _DummyMcpClient)
    monkeypatch.setattr(mcp.client.sse, "sse_client", fake_sse)
    _patch_traceparent(monkeypatch, _FAKE_TRACEPARENT)

    async with open_mcp_http_client(
        "https://mcp.example/sse",
        headers={"x-test": "1"},
        timeout=20.0,
        transport="sse",
    ):
        pass

    assert captured_headers == [
        {"x-test": "1", "traceparent": _FAKE_TRACEPARENT},
    ]
