---
title: MCP 服务器认证
description: "为 CubeLoop 中的 MCP 服务器配置认证——API 密钥、OAuth 及自定义请求头。"
---

# MCP 服务器认证

MCP 传输层本身不规定认证方案——由服务器自行决定。实践中，以下三种模式几乎覆盖所有场景：

1. **`Authorization` 请求头中的 Bearer token**（HTTP 传输）。
2. **任意自定义请求头**（HTTP 传输）。
3. **环境变量**（stdio 传输——服务器从自身进程环境中读取）。

本页逐一介绍每种模式。

## HTTP：Bearer token

这是托管 MCP 服务器（GitHub、Sentry、内部服务等）的主流认证模式。向 `load_mcp_tools_http` 传递 `Authorization` 请求头：

```python
import os
from cubeloop.mcp import load_mcp_tools_http

tools = await load_mcp_tools_http(
    server_url="https://mcp.example.com/sse",
    headers={"Authorization": f"Bearer {os.environ['MCP_TOKEN']}"},
)
```

`headers` 会随传输层的**每一次**请求发送，包括后续的 `tools/call` 调用。不存在"先登录再调用"的步骤——token 随每个连接传递。

## HTTP：自定义请求头

部分服务器使用 API Key 请求头代替 Bearer token：

```python
tools = await load_mcp_tools_http(
    server_url="https://mcp.internal/sse",
    headers={
        "X-API-Key": os.environ["MCP_API_KEY"],
        "X-Tenant-Id": "acme-corp",
    },
)
```

也可按需组合：

```python
headers = {
    "Authorization": f"Bearer {token}",
    "X-Trace-Id": str(uuid.uuid4()),
}
```

## HTTP：短期 token / 刷新

CubeLoop 的加载器在加载时接受静态的 `headers` 字典。对于会过期的 token（OAuth、短 TTL 的 JWT），有两种处理方案：

### 方案 A——到期后重新加载

捕获错误，重新获取 token，然后重新加载工具：

```python
async def load_with_fresh_token():
    token = await fetch_token()
    return await load_mcp_tools_http(
        server_url="…",
        headers={"Authorization": f"Bearer {token}"},
    )

tools = await load_with_fresh_token()
```

适用于生命周期长于单次 agent 运行的 token。

### 方案 B——在工具内部封装刷新逻辑

自己构建 `AgentTool`，通过闭包持有刷新逻辑：

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

仅适合 TTL 极短的 token。对于周期较长的 token，方案 A 更简洁。

## stdio：环境变量

stdio 服务器从自身进程环境中读取凭据。传入 `env` 字典即可：

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

如果省略 `env`，子进程会继承**全部**父进程环境变量（标准子进程行为）。如需从干净状态启动，请传入显式字典：

```python
env = {
    "PATH": os.environ["PATH"],
    "HOME": os.environ["HOME"],
    "GITHUB_PERSONAL_ACCESS_TOKEN": token,
}
```

## stdio：基于文件的凭据

当服务器从 `~/.config/...` 读取凭据时，可传入 `cwd` 确保路径解析一致，并依赖继承的环境变量：

```python
tools = await load_mcp_tools_stdio(
    command="/usr/local/bin/my-mcp",
    args=["--config", "config.yaml"],
    cwd="/etc/myapp",
)
```

## 多用户 / 多租户凭据

在多租户服务中，每次 agent 调用需要使用不同的凭据。请**按请求加载工具**：

```python
async def build_agent_for_user(user_id: str) -> Agent:
    token = await fetch_user_token(user_id)
    tools = await load_mcp_tools_http(
        server_url="https://mcp.example.com/sse",
        headers={"Authorization": f"Bearer {token}"},
    )
    return Agent(model=model, tools=tools)
```

不要跨用户缓存同一份 `tools` 列表——闭包在加载时已保留了各自的认证请求头。

## 审计 / 可观测性

在 [`before_tool_call`](../middleware/hooks#before_tool_call) middleware 中记录 MCP 调用日志。从 MCP 加载的工具在事件流中与手写工具完全一致，因此现有日志 middleware 开箱即用。

若要专门标记 MCP 工具，可检查合成参数模型的名称（前缀为 `MCP_`）：

```python
class MCPAuditMiddleware(Middleware):
    async def before_tool_call(self, ctx, *, signal=None):
        param_name = type(ctx.args).__name__
        if param_name.startswith("MCP_"):
            log.info("mcp_call", extra={"tool": ctx.tool_call.name})
        return None
```

## 常见问题

- **调用时 `401 Unauthorized`，但 `list_tools` 正常** —— 部分服务器对单个工具设置了权限。确认 token 拥有所有待用工具的 scope。
- **Token 泄露到日志中** —— 不要记录 `headers` 字典。尤其注意包含查询字符串凭据的 URL 出现在异常消息中的情况。
- **stdio 服务器静默失败** —— 服务器会将认证错误打印到自身的 stderr。添加 `stdout`/`stderr` 重定向，或使用 MCP SDK 的诊断日志。
- **每次请求都重新加载很慢** —— 按 `(user_id, token)` 对缓存已加载的工具。记住 token 有 TTL。

## 参见

- [加载 MCP 工具](./loading) —— 基础加载器 API。
- [Middleware → 示例 → 日志记录](../middleware/examples#structured-logging) —— 与 MCP 审计需求天然配合。
