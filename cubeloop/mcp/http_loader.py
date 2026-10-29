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


def _headers_with_traceparent(
    headers: dict[str, str] | None,
) -> dict[str, str] | None:
    """Return ``headers``, adding W3C ``traceparent`` when a span is active."""
    from cubeloop.mcp._tracing import current_traceparent

    traceparent = current_traceparent()
    if traceparent is None:
        return headers
    return {**(headers or {}), "traceparent": traceparent}


@asynccontextmanager
async def open_mcp_http_client(
    server_url: str,
    *,
    headers: dict[str, str] | None,
    timeout: float,
    transport: Transport,
) -> AsyncIterator[Any]:
    """Open a negotiated MCP HTTP client over ``sse`` or ``streamable_http``.

    The SDK ``Client`` is auto-mode: it probes ``server/discover``
    (2026-07-28) and falls back to ``initialize`` for 1.x servers.
    Callers must not invoke ``initialize()`` themselves.

    When a CubeLoop MCP CLIENT span is active, the opener stamps a W3C
    ``traceparent`` header so an instrumented server can continue the
    trace. Hosts that persist ``tools/list`` should pair this with
    :func:`cubeloop.mcp.make_mcp_agent_tool` and
    :func:`cubeloop.mcp.serialize_mcp_call_tool_response`.
    """
    from mcp import Client

    call_headers = _headers_with_traceparent(headers)

    if transport == "streamable_http":
        import httpx2
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client

        # streamable_http_client has no headers/timeout kwargs. A caller-supplied
        # client is not closed by the SDK (client_provided=True), so this loader
        # owns the AsyncClient lifecycle. create_mcp_http_client lives in a
        # private module because that is the SDK's supported factory for the
        # transport's default timeout and redirect behaviour.
        async with create_mcp_http_client(
            headers=call_headers,
            timeout=httpx2.Timeout(timeout),
        ) as http_client:
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
                headers=call_headers,
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
            async with open_mcp_http_client(
                server_url,
                headers=headers,
                timeout=timeout,
                transport=transport,
            ) as client:
                response = await client.call_tool(tool_name, args)
                return serialize_mcp_call_tool_response(response)

    async with asyncio.timeout(timeout):
        async with open_mcp_http_client(
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
