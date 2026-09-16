---
title: 加载 MCP 工具
description: "将 MCP 服务器中的工具加载到 CubeLoop agent，包括基于 SSE 的远程服务器。"
---

# 加载 MCP 工具

[Model Context Protocol](https://modelcontextprotocol.io) 定义了一套标准方式，让工具服务器向 agent 暴露能力。CubeLoop 内置两个加载器，可连接 MCP 服务器、枚举其工具，并将每个工具转换为标准的 `AgentTool`，直接传给 `Agent(tools=…)` 使用。

安装额外依赖：

```bash
pip install "cubeloop[mcp]"
```

这将引入 `mcp` SDK。

## stdio 传输：本地子进程

适用于以本地进程方式运行的工具（如 npm 发布的服务器、Python 模块、内部 CLI 等）：

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

`load_mcp_tools_stdio` 会连接服务器、调用 `list_tools`、断开连接，然后返回 `AgentTool` 列表。**每次工具调用都会启动一个新子进程**——v0.3 保持简单，不维护进程池。

参数说明：

| 参数 | 用途 |
|---|---|
| `command` | 可执行文件（如 `"npx"`、`sys.executable`、`/usr/bin/uvx`） |
| `args` | 服务器的 argv |
| `env` | 环境变量（可选） |
| `cwd` | 工作目录（可选） |
| `timeout` | `initialize` / `list` / `call` 各步骤的挂钟超时时间 |

## HTTP/SSE 传输：远程服务器

适用于托管的 MCP 服务器（Sentry、GitHub、内部服务等）：

```python
from cubeloop.mcp import load_mcp_tools_http

tools = await load_mcp_tools_http(
    server_url="https://mcp.example.com/sse",
    headers={"Authorization": "Bearer <token>"},
    timeout=30.0,
)
```

`load_mcp_tools_http` 使用 MCP SDK 的 SSE 客户端。模式与 stdio 相同：一次连接用于枚举，每次工具调用建立新连接。

HTTP 服务器常用的认证模式参见 [MCP 认证](./auth)。

## 返回值说明

返回列表中的每个元素都是一个 `AgentTool`：

- `name` —— MCP 工具名称。
- `description` —— 直接来自服务器（不做改写）。
- `parameters` —— 从 MCP `inputSchema` 合成的 Pydantic 模型（JSON Schema → Pydantic，通过 `pydantic.create_model`）。
- `execute` —— 通过相同传输层调用 `tools/call` 的闭包。

合成的 Pydantic 模型支持：`string`、`integer`、`number`、`boolean`、`array`、`object`（映射为 `dict[str, Any]`）以及枚举（通过 `Literal`）。顶层约束均会保留：`description`、`pattern`、`minLength`/`maxLength`、`minimum`/`maximum`（含 exclusive 变体）、`minItems`/`maxItems`。

## 将 MCP 工具与手写工具混用

两者类型相同，直接拼接列表即可：

```python
mcp_tools = await load_mcp_tools_stdio(command="…", args=[…])
my_tools = [weather_tool, search_tool]

agent = Agent(
    model=model,
    tools=my_tools + mcp_tools,
)
```

模型看到的是合并后的统一 JSON Schema；循环会将每次调用分发给对应的实现。

:::tip MCP server 很多？延迟加载它们
如果你接了好几个 MCP server、但每次对话只用到其中一两个，把每个 server
的 schema 全部塞进 system prompt 就成了主要的上下文成本。把每组 tool
包成一个 [`DeferredToolGroup`](../middleware/deferred-tools)——CubeLoop 会把
完整 schema 换成一份紧凑目录，让模型通过内置的 `load_tools` 工具按需
展开。
:::

## 按次连接与复用连接

CubeLoop 每次 `execute` 调用都会建立新的传输连接。这样做：

- ✅ 简单——无需管理连接池的生命周期。
- ✅ 健壮——挂起的连接不会污染其他工具。
- ⚠️ 对于启动开销较大的 stdio 服务器较慢（`npx` 服务器每次调用可能增加数秒延迟）。

对于高吞吐量的 stdio 服务器，建议将其作为持久 HTTP 服务运行，并改用 `load_mcp_tools_http`。

## 图片与结构化内容

如果 MCP 工具返回图片内容块，CubeLoop 会将其映射为 `ImageContent` 并包含在 `AgentToolResult.content` 中。Anthropic provider 会将其作为工具结果中的图片块转发；OpenAI provider 目前会将其去除（wire 格式不支持携带图片的工具结果）。

如果服务器返回 `structuredContent`，它会暴露在 `AgentToolResult.details["structuredContent"]` 下——便于下游代码访问，但不会展示给模型。

## 常见问题

- **第一次调用时立即出现 `asyncio.TimeoutError`** —— 服务器未在 `timeout` 内完成 `initialize`。将 `timeout=60` 或更高；部分服务器初始化较重。
- **每次工具调用都很慢** —— stdio 子进程启动开销。将服务器改为 HTTP 运行，或编写自定义适配器以保持子进程常驻。
- **工具列表中缺少某些工具** —— 服务器未能广播它们。单独运行服务器并用 MCP CLI 手动调用 `list_tools` 进行排查。
- **Pydantic 校验拒绝模型输出** —— MCP `inputSchema` 中有模型不遵守的约束。可在服务器端放宽 schema，或添加 `before_tool_call` middleware 进行强制转换。

## 参见

- [MCP 认证](./auth) —— Bearer token、请求头、基于环境变量的凭据。
- [工具使用](../agents/tool-use) —— 工具（MCP 或其他）的分发机制。
- [`make_mcp_agent_tool` 源码](https://github.com/cubeplexai/cubeloop/blob/main/cubeloop/mcp/_adapter.py) —— schema → Pydantic 适配器，如需自定义可参考。
- [延迟工具组](../middleware/deferred-tools) —— 把 MCP schema 从 system
  prompt 里藏起来，让模型按需展开。当多个 MCP server 加起来构成一个
  上下文沉重的工具集时很有用。
