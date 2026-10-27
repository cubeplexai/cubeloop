"""Tests for the MCP v2 HTTP tool loader."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from cubeloop.mcp.http_loader import _open_client


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

    monkeypatch.setattr(http_loader, "_open_client", fake_open_client)
    return clients


def test_import_http_loader() -> None:
    from cubeloop.mcp import load_mcp_tools_http

    assert callable(load_mcp_tools_http)


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

    monkeypatch.setattr(http_loader, "_open_client", fake_open_client)
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
    """Directly test _open_client with streamable_http transport to ensure lines are covered."""
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

    @asynccontextmanager
    async def fake_streamable(url, http_client=None):
        yield "streamable_stream"

    monkeypatch.setattr(mcp, "Client", DummyClient)
    monkeypatch.setattr(
        mcp.client.streamable_http, "streamable_http_client", fake_streamable
    )

    # Test with headers
    async with _open_client(
        "https://mcp.example/mcp",
        headers={"Authorization": "Bearer test"},
        timeout=15.0,
        transport="streamable_http",
    ) as client:
        assert isinstance(client, DummyClient)

    # Test without headers
    async with _open_client(
        "https://mcp.example/mcp",
        headers=None,
        timeout=15.0,
        transport="streamable_http",
    ) as client:
        assert isinstance(client, DummyClient)


@pytest.mark.asyncio
async def test_open_client_sse_branch(monkeypatch) -> None:
    """Directly test _open_client with sse transport."""
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

    async with _open_client(
        "https://mcp.example/sse",
        headers={"x-test": "1"},
        timeout=20.0,
        transport="sse",
    ) as client:
        assert isinstance(client, DummyClient)
