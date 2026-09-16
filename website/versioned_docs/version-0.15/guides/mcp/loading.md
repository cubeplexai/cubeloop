---
title: Loading MCP Tools
description: "Load tools from MCP servers into your CubeLoop agent, including SSE-based remote servers."
---

# Loading MCP Tools

The [Model Context Protocol](https://modelcontextprotocol.io) defines
a standard way for tool servers to expose capabilities to agents.
CubeLoop ships two loaders that connect to an MCP server, enumerate its
tools, and turn each one into a regular `AgentTool` you can hand to
`Agent(tools=…)`.

Install the extra:

```bash
pip install "cubeloop[mcp]"
```

This pulls in the `mcp` SDK.

## stdio transport: local subprocess

For tools that run as a local process (e.g. npm-published servers, a
Python module, an internal CLI):

```python
import asyncio
import sys
from cubeloop import Agent
from cubeloop.mcp import load_mcp_tools_stdio
from cubeloop.providers.anthropic import AnthropicProvider


async def main():
    # Spawn a stdio MCP server and discover its tools.
    tools = await load_mcp_tools_stdio(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/sandbox"],
        timeout=30.0,
    )

    agent = Agent(
        model=AnthropicProvider(provider_id="anthropic", api_key="…").model("claude-sonnet-4-6"),
        tools=tools,                # all server tools, ready to use
    )
    agent.subscribe(lambda e, s=None: None)
    await agent.prompt("List files in /tmp/sandbox.")


asyncio.run(main())
```

`load_mcp_tools_stdio` connects, calls `list_tools`, disconnects, and
returns the `AgentTool` list. **Each tool call spawns a fresh
subprocess** — v0.3 keeps things simple, no process pool.

Arguments:

| Parameter | Purpose |
|---|---|
| `command` | The executable (e.g. `"npx"`, `sys.executable`, `/usr/bin/uvx`) |
| `args` | argv for the server |
| `env` | Environment variables (optional) |
| `cwd` | Working directory (optional) |
| `timeout` | Per-call wall-clock timeout for `initialize` / `list` / `call` |

## HTTP/SSE transport: remote server

For hosted MCP servers (Sentry, GitHub, internal services):

```python
from cubeloop.mcp import load_mcp_tools_http

tools = await load_mcp_tools_http(
    server_url="https://mcp.example.com/sse",
    headers={"Authorization": "Bearer <token>"},
    timeout=30.0,
)
```

`load_mcp_tools_http` uses the MCP SDK's SSE client. Same model as
stdio: one connection to enumerate, then a fresh connection per tool
call.

See [MCP Auth](./auth) for the auth patterns you'll likely need with
HTTP servers.

## What you get back

Each entry in the returned list is an `AgentTool`:

- `name` — the MCP tool name.
- `description` — straight from the server (no rewrites).
- `parameters` — a Pydantic model synthesised from the MCP
  `inputSchema` (JSON Schema → Pydantic via `pydantic.create_model`).
- `execute` — closure that calls `tools/call` over the same transport.

The synthesised Pydantic model covers: `string`, `integer`, `number`,
`boolean`, `array`, `object` (as `dict[str, Any]`), and enum (via
`Literal`). Top-level constraints are preserved: `description`,
`pattern`, `minLength`/`maxLength`, `minimum`/`maximum` (incl.
exclusive variants), `minItems`/`maxItems`.

## Mixing MCP tools with hand-written tools

They're the same type. Just concatenate the lists:

```python
mcp_tools = await load_mcp_tools_stdio(command="…", args=[…])
my_tools = [weather_tool, search_tool]

agent = Agent(
    model=model,
    tools=my_tools + mcp_tools,
)
```

The model sees one combined JSON Schema; the loop dispatches each
call to the right implementation.

:::tip Many MCP servers? Defer them.
If you wire up several MCP servers and only a few are needed per
conversation, eagerly loading every schema into the system prompt
becomes the dominant context cost. Wrap each server's tools in a
[`DeferredToolGroup`](../middleware/deferred-tools) — CubeLoop replaces
the full schemas with a compact catalog and lets the model expand a
group on demand via the built-in `load_tools` tool.
:::

## Per-call vs reusable connections

CubeLoop opens a new transport per `execute` call. That's:

- ✅ Simple — no pool lifecycle to manage.
- ✅ Robust — a hung connection can't poison other tools.
- ⚠️ Slower for stdio servers with heavy startup (a `npx` server can
  add seconds per call).

For high-throughput stdio servers, run the server as a persistent
HTTP service instead and use `load_mcp_tools_http`.

## Image and structured content

If an MCP tool returns image content blocks, CubeLoop maps them to
`ImageContent` and includes them in the `AgentToolResult.content`.
Anthropic provider relays these as image blocks in tool results;
OpenAI providers currently strip them (the wire format doesn't
support image-bearing tool results).

If the server returns `structuredContent`, it's exposed under
`AgentToolResult.details["structuredContent"]` — useful for
downstream programmatic access, but not shown to the model.

## Common pitfalls

- **`asyncio.TimeoutError` immediately on first call** — Server didn't
  finish `initialize` within `timeout`. Bump `timeout=60` or higher;
  some servers do heavy setup.
- **Each tool call is slow** — stdio subprocess spawn overhead. Run
  the server as HTTP, or write a custom adapter that keeps the
  subprocess alive.
- **Tools missing from the list** — Server failed to advertise them.
  Run the server in isolation and call `list_tools` manually with the
  MCP CLI to inspect.
- **Pydantic validation rejects model output** — The MCP `inputSchema`
  has constraints the model isn't honouring. Either loosen the schema
  on the server side or add a `before_tool_call` middleware that
  coerces.

## See also

- [MCP Auth](./auth) — bearer tokens, headers, env-based credentials.
- [Tool Use](../agents/tool-use) — how tools (MCP or otherwise) are
  dispatched.
- [`make_mcp_agent_tool` source](https://github.com/cubeplexai/cubeloop/blob/main/cubeloop/mcp/_adapter.py)
  — the schema → Pydantic adapter, if you need to customise.
- [Deferred Tool Groups](../middleware/deferred-tools) — hide MCP
  schemas from the system prompt and let the model expand them on
  demand. Useful when several MCP servers add up to a context-heavy
  toolbelt.
