"""Tests for the MCP v2 stdio tool loader."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_import_stdio_loader() -> None:
    from cubeloop.mcp import load_mcp_tools_stdio

    assert callable(load_mcp_tools_stdio)


@pytest.mark.asyncio
async def test_stdio_loader_uses_mcp_v2_client_auto_mode(monkeypatch) -> None:
    """The v2 Client supplies a stateless or legacy-negotiated connection."""
    from cubeloop.mcp import stdio_loader

    clients = []

    class FakeClient:
        protocol_version = "2026-07-28"
        server_info = SimpleNamespace(name="fake", version="2.0.0", icons=None)

        def __init__(self, params):
            self.params = params
            self.calls = []
            clients.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def list_tools(self):
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="echo",
                        description="Echo text",
                        input_schema={
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                        },
                        icons=None,
                    )
                ]
            )

        async def call_tool(self, name, args):
            self.calls.append((name, args))
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=args["text"])],
                is_error=False,
                structured_content=None,
            )

    import mcp

    monkeypatch.setattr(mcp, "Client", FakeClient)
    discovery = await stdio_loader.load_mcp_tools_stdio("fake-server", ["--stdio"])
    assert discovery.server is not None
    assert discovery.server.name == "fake"
    assert discovery.tools[0].name == "echo"

    tool = discovery.tools[0]
    result = await tool.execute(
        "tc-stdio", tool.parameters(text="hello"), signal=None, on_update=None
    )
    assert clients[1].calls == [("echo", {"text": "hello"})]
    assert result.content[0].text == "hello"


@pytest.mark.asyncio
async def test_stdio_loader_preserves_legacy_protocol_version(monkeypatch) -> None:
    """Client auto-mode may negotiate down to an initialize-era server."""
    from cubeloop.mcp import stdio_loader

    class FakeClient:
        protocol_version = "2025-11-25"
        server_info = SimpleNamespace(name="legacy", version="1.0.0", icons=None)

        def __init__(self, params):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def list_tools(self):
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="echo", description="", input_schema={}, icons=None
                    )
                ]
            )

        async def call_tool(self, name, args):
            return SimpleNamespace(content=[], is_error=False, structured_content=None)

    import mcp

    monkeypatch.setattr(mcp, "Client", FakeClient)
    discovery = await stdio_loader.load_mcp_tools_stdio("fake-server", [])
    assert discovery.tools[0].name == "echo"
