"""Minimal stdio MCP server for testing.

Uses the MCP Python SDK v2 server primitives and advertises one ``echo`` tool.

Run as: python -m tests.mcp._fake_stdio_server
"""

from __future__ import annotations

import asyncio


async def main() -> None:
    from mcp.server import MCPServer

    mcp = MCPServer("fake-cubepi-test-server", version="0.0.1")

    @mcp.tool()
    def echo(text: str) -> str:
        """Echo the input back."""
        return text

    await mcp.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
