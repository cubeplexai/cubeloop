"""Model Context Protocol (MCP) tool integration for cubeloop.

cubeloop[mcp] extra required.
"""

from cubeloop.mcp._adapter import (
    make_mcp_agent_tool,
    serialize_mcp_call_tool_response,
)
from cubeloop.mcp.http_loader import load_mcp_tools_http, open_mcp_http_client
from cubeloop.mcp.stdio_loader import load_mcp_tools_stdio
from cubeloop.mcp.types import (
    MCPDiscoveryResult,
    MCPIcon,
    MCPServerInfo,
    MCPToolInfo,
)

__all__ = [
    "MCPDiscoveryResult",
    "MCPIcon",
    "MCPServerInfo",
    "MCPToolInfo",
    "load_mcp_tools_http",
    "load_mcp_tools_stdio",
    "make_mcp_agent_tool",
    "open_mcp_http_client",
    "serialize_mcp_call_tool_response",
]
