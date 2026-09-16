---
title: MCP Server Authentication
description: "Configure authentication — API keys, OAuth, and custom headers — for MCP servers in CubeLoop."
---

# MCP Server Authentication

The MCP transport layer doesn't dictate an auth scheme — servers
decide. In practice, three patterns cover almost every case:

1. **Bearer token in `Authorization` header** (HTTP transport).
2. **Arbitrary custom headers** (HTTP transport).
3. **Environment variables** (stdio transport — server reads them
   from its own process env).

This page walks through each.

## HTTP: bearer tokens

The dominant pattern for hosted MCP servers (GitHub, Sentry,
internal). Pass an `Authorization` header to `load_mcp_tools_http`:

```python
import os
from cubeloop.mcp import load_mcp_tools_http

tools = await load_mcp_tools_http(
    server_url="https://mcp.example.com/sse",
    headers={"Authorization": f"Bearer {os.environ['MCP_TOKEN']}"},
)
```

`headers` is forwarded to *every* request the transport makes,
including subsequent `tools/call` invocations. There's no separate
"login then call" step — the token rides every connection.

## HTTP: custom headers

Some servers use API-key headers instead:

```python
tools = await load_mcp_tools_http(
    server_url="https://mcp.internal/sse",
    headers={
        "X-API-Key": os.environ["MCP_API_KEY"],
        "X-Tenant-Id": "acme-corp",
    },
)
```

Combine as needed:

```python
headers = {
    "Authorization": f"Bearer {token}",
    "X-Trace-Id": str(uuid.uuid4()),
}
```

## HTTP: short-lived tokens / refresh

CubeLoop's loaders take a static `headers` dict at load time. For tokens
that expire (OAuth, JWT with short TTL), you have two options:

### Option A — Re-load on expiry

Catch the error, re-fetch the token, re-load the tools:

```python
async def load_with_fresh_token():
    token = await fetch_token()
    return await load_mcp_tools_http(
        server_url="…",
        headers={"Authorization": f"Bearer {token}"},
    )

tools = await load_with_fresh_token()
```

Practical for tokens that live longer than a single agent run.

### Option B — Wrap and refresh inside the tool

Build the `AgentTool` yourself with a closure that knows how to
refresh:

```python
from cubeloop.mcp._adapter import make_mcp_agent_tool
from cubeloop.mcp import load_mcp_tools_http

async def call_remote_with_refresh(tool_name, args):
    headers = {"Authorization": f"Bearer {await fetch_token()}"}
    # Re-implement the http_loader's call_remote with fresh headers each time
    from mcp.client.sse import sse_client
    from mcp import ClientSession
    async with sse_client(server_url, headers=headers, timeout=30) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            resp = await session.call_tool(tool_name, args)
            return _serialize_call_tool_response(resp)

# Use the adapter directly:
my_tool = make_mcp_agent_tool(
    name="…",
    description="…",
    input_schema={…},
    call_remote=call_remote_with_refresh,
)
```

Only worth it for tokens with very short TTLs. For anything longer,
Option A is simpler.

## stdio: environment variables

stdio servers read credentials from their own process environment.
Pass an `env` dict:

```python
import os
from cubeloop.mcp import load_mcp_tools_stdio

tools = await load_mcp_tools_stdio(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-github"],
    env={
        "GITHUB_PERSONAL_ACCESS_TOKEN": os.environ["GH_TOKEN"],
        **os.environ,                         # inherit the rest
    },
)
```

If you omit `env`, the subprocess inherits **all** parent env vars
(standard subprocess behaviour). For a clean slate, pass an explicit
dict:

```python
env = {
    "PATH": os.environ["PATH"],
    "HOME": os.environ["HOME"],
    "GITHUB_PERSONAL_ACCESS_TOKEN": token,
}
```

## stdio: file-based credentials

When a server reads credentials from `~/.config/...`, pass `cwd` to
ensure consistent path resolution, and rely on the inherited
environment:

```python
tools = await load_mcp_tools_stdio(
    command="/usr/local/bin/my-mcp",
    args=["--config", "config.yaml"],
    cwd="/etc/myapp",
)
```

## Per-user / per-tenant credentials

In a multi-tenant service, each agent invocation needs different
credentials. **Load tools per-request**:

```python
async def build_agent_for_user(user_id: str) -> Agent:
    token = await fetch_user_token(user_id)
    tools = await load_mcp_tools_http(
        server_url="https://mcp.example.com/sse",
        headers={"Authorization": f"Bearer {token}"},
    )
    return Agent(model=model, tools=tools)
```

Don't cache a single `tools` list across users — the closures retain
the auth headers from load time.

## Auditing / observability

Log MCP calls inside a [`before_tool_call`](../middleware/hooks#before_tool_call)
middleware. Tools loaded from MCP look identical to hand-written
tools in event streams, so existing logging middleware Just Works.

To tag MCP tools specifically, check for the synthesised parameter
model's name (prefix `MCP_`):

```python
class MCPAuditMiddleware(Middleware):
    async def before_tool_call(self, ctx, *, signal=None):
        param_name = type(ctx.args).__name__
        if param_name.startswith("MCP_"):
            log.info("mcp_call", extra={"tool": ctx.tool_call.name})
        return None
```

## Common pitfalls

- **`401 Unauthorized` only on call, not on `list_tools`** — Some
  servers gate per-tool. Ensure the token has scopes for every tool
  you intend the agent to use.
- **Tokens leaking into logs** — Don't log the `headers` dict.
  Especially watch out for exception messages that include URLs with
  query-string credentials.
- **stdio server fails silently** — Server prints auth errors to its
  own stderr. Add `stdout`/`stderr` redirection or use the MCP SDK's
  diagnostic logging.
- **Re-loading every request is slow** — Cache the loaded tools per
  `(user_id, token)` pair. Just remember tokens have TTLs.

## See also

- [Loading MCP Tools](./loading) — the basic loader API.
- [Middleware → Examples → Logging](../middleware/examples#structured-logging)
  — pairs naturally with MCP audit needs.
