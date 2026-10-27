"""stdio transport MCP tool loader."""

from __future__ import annotations

import asyncio
from typing import Any

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

    ``timeout`` wraps the entire connection, handshake, and RPC execution.
    """
    from mcp import Client, StdioServerParameters

    server_params = StdioServerParameters(command=command, args=args, env=env, cwd=cwd)

    async def _call_remote(tool_name: str, args_dict: dict[str, Any]) -> dict[str, Any]:
        async with asyncio.timeout(timeout):
            async with Client(server_params) as client:
                response = await client.call_tool(tool_name, args_dict)
                return serialize_mcp_call_tool_response(response)

    async with asyncio.timeout(timeout):
        async with Client(server_params) as client:
            tools_resp = await client.list_tools()
            tool_descs = tools_resp.tools
            protocol_version = client.protocol_version
            server_info = client.server_info

    tools = [
        make_mcp_agent_tool(
            name=desc.name,
            description=desc.description or "",
            input_schema=get_mcp_field(desc, "input_schema", "inputSchema")
            or {"type": "object", "properties": {}},
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
