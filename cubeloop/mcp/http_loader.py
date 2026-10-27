"""HTTP transport MCP tool loader."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from cubeloop.mcp._adapter import (
    get_mcp_field,
    make_mcp_agent_tool,
    serialize_mcp_call_tool_response,
)
from cubeloop.mcp.types import (
    MCPDiscoveryResult,
    server_info_from_result,
    tool_info_from_desc,
)

Transport = Literal["sse", "streamable_http"]


@asynccontextmanager
async def _open_client(
    server_url: str,
    *,
    headers: dict[str, str] | None,
    timeout: float,
    transport: Transport,
) -> AsyncIterator[Any]:
    """Open an MCP v2 client over the requested transport.

    ``Client`` negotiates protocol version ``2026-07-28`` by probing
    ``server/discover`` and automatically falls back to ``initialize`` for
    pre-v2 servers.  The v2 Streamable HTTP transport stamps the protocol and
    routing headers itself, so callers must not synthesize them.
    """
    from mcp import Client

    if transport == "streamable_http":
        import httpx2
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client

        http_client = create_mcp_http_client(
            headers=headers,
            timeout=httpx2.Timeout(timeout),
        )
        async with Client(
            streamable_http_client(server_url, http_client=http_client)
        ) as client:
            yield client
        return

    if transport == "sse":
        from mcp.client.sse import sse_client

        async with Client(
            sse_client(
                server_url,
                headers=headers,
                timeout=timeout,
                sse_read_timeout=timeout,
            )
        ) as client:
            yield client
        return

    raise ValueError(f"unsupported MCP transport '{transport}'")


async def load_mcp_tools_http(
    server_url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    transport: Transport = "sse",
) -> MCPDiscoveryResult:
    """Connect to an MCP server and discover executable tools.

    Supports both the stateless MCP 2026-07-28 protocol and legacy servers.
    MCP SDK v2 performs protocol negotiation; in auto mode it uses
    ``server/discover`` for modern servers and falls back to ``initialize``.
    Each returned tool opens a fresh client for invocation, matching the
    loader's previous no-pooling behaviour.

    ``timeout`` wraps the entire connection, handshake, and RPC execution.
    """

    async def _call_remote(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        async with asyncio.timeout(timeout):
            async with _open_client(
                server_url, headers=headers, timeout=timeout, transport=transport
            ) as client:
                response = await client.call_tool(tool_name, args)
                return serialize_mcp_call_tool_response(response)

    async with asyncio.timeout(timeout):
        async with _open_client(
            server_url, headers=headers, timeout=timeout, transport=transport
        ) as client:
            tools_resp = await client.list_tools()
            tool_descs = tools_resp.tools
            protocol_version = client.protocol_version
            server_info = client.server_info

    address, port = _split_address(server_url)
    tools = [
        make_mcp_agent_tool(
            name=desc.name,
            description=desc.description or "",
            input_schema=get_mcp_field(desc, "input_schema", "inputSchema")
            or {"type": "object", "properties": {}},
            call_remote=_call_remote,
            server_address=address,
            server_port=port,
            protocol_version=protocol_version,
        )
        for desc in tool_descs
    ]
    return MCPDiscoveryResult(
        tools=tools,
        server=server_info_from_result(server_info),
        tool_infos=[tool_info_from_desc(desc) for desc in tool_descs],
    )


def _split_address(server_url: str) -> tuple[str | None, int | None]:
    try:
        from urllib.parse import urlparse

        parsed = urlparse(server_url)
        return parsed.hostname, parsed.port
    except Exception:
        return None, None
