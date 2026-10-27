"""stdio transport MCP tool loader."""

from __future__ import annotations

import asyncio
from typing import Any

from cubeloop.mcp._adapter import make_mcp_agent_tool
from cubeloop.mcp.types import MCPDiscoveryResult, server_info_from_result, tool_info_from_desc


async def load_mcp_tools_stdio(
    command: str,
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout: float = 30.0,
) -> MCPDiscoveryResult:
    """Spawn an MCP server and load tools using the MCP v2 client.

    ``mcp.Client`` negotiates the stateless 2026-07-28 protocol by default and
    falls back to the legacy ``initialize`` handshake for older servers.  A
    process is deliberately opened for discovery and again for every tool call,
    preserving this loader's existing no-pooling behaviour.
    """
    from mcp import Client, StdioServerParameters

    server_params = StdioServerParameters(command=command, args=args, env=env, cwd=cwd)

    async def _call_remote(tool_name: str, args_dict: dict[str, Any]) -> dict[str, Any]:
        async with Client(server_params) as client:
            response = await asyncio.wait_for(
                client.call_tool(tool_name, args_dict), timeout=timeout
            )
            return _serialize_call_tool_response(response)

    async with Client(server_params) as client:
        tools_resp = await asyncio.wait_for(client.list_tools(), timeout=timeout)
        tool_descs = tools_resp.tools
        protocol_version = client.protocol_version
        server_info = client.server_info

    tools = [
        make_mcp_agent_tool(
            name=desc.name,
            description=desc.description or "",
            input_schema=_field(desc, "input_schema", "inputSchema") or {"type": "object", "properties": {}},
            call_remote=_call_remote,
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
