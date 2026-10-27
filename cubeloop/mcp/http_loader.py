"""HTTP transport MCP tool loader."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from cubeloop.mcp._adapter import make_mcp_agent_tool
from cubeloop.mcp.types import MCPDiscoveryResult, server_info_from_result, tool_info_from_desc

Transport = Literal["sse", "streamable_http"]


@asynccontextmanager
async def _open_client(
    server_url: str,
    *,
    headers: dict[str, str] | None,
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
        if headers:
            import httpx2
            from mcp.client.streamable_http import streamable_http_client

            async with httpx2.AsyncClient(headers=headers) as http_client:
                async with Client(streamable_http_client(server_url, http_client=http_client)) as client:
                    yield client
        else:
            async with Client(server_url) as client:
                yield client
        return

    if transport == "sse":
        from mcp.client.sse import sse_client

        async with Client(sse_client(server_url, headers=headers)) as client:
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
    """

    async def _call_remote(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        async with _open_client(server_url, headers=headers, transport=transport) as client:
            response = await asyncio.wait_for(client.call_tool(tool_name, args), timeout=timeout)
            return _serialize_call_tool_response(response)

    async with _open_client(server_url, headers=headers, transport=transport) as client:
        tools_resp = await asyncio.wait_for(client.list_tools(), timeout=timeout)
        tool_descs = tools_resp.tools
        protocol_version = client.protocol_version
        server_info = client.server_info

    address, port = _split_address(server_url)
    tools = [
        make_mcp_agent_tool(
            name=desc.name,
            description=desc.description or "",
            input_schema=_field(desc, "input_schema", "inputSchema") or {"type": "object", "properties": {}},
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


def _field(value: object, snake_name: str, camel_name: str) -> Any:
    return getattr(value, snake_name, None) or getattr(value, camel_name, None)


def _split_address(server_url: str) -> tuple[str | None, int | None]:
    try:
        from urllib.parse import urlparse

        parsed = urlparse(server_url)
        return parsed.hostname, parsed.port
    except Exception:
        return None, None


def _serialize_call_tool_response(resp: Any) -> dict[str, Any]:
    """Normalize a v2 ``CallToolResult`` to the adapter response shape."""
    content: list[dict[str, Any]] = []
    for block in resp.content or []:
        if block.type == "text":
            content.append({"type": "text", "text": block.text})
        elif block.type == "image":
            content.append(
                {
                    "type": "image",
                    "data": getattr(block, "data", ""),
                    "mimeType": _field(block, "mime_type", "mimeType") or "",
                }
            )
    result: dict[str, Any] = {
        "content": content,
        "isError": bool(_field(resp, "is_error", "isError")),
    }
    structured = _field(resp, "structured_content", "structuredContent")
    if structured is not None:
        result["structuredContent"] = structured
    return result
