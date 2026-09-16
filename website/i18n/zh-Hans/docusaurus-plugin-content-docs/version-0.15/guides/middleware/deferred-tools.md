---
title: 延迟工具组
description: "默认对模型隐藏 MCP 工具 schema，按需加载——且不破坏 prompt 缓存。"
---

# 延迟工具组

当一个 agent 接入多个 MCP 服务器时，它们合起来的 tool schema 在每一轮都
能吃掉数千个 token 上下文——即使模型这一回合只需要其中一两组。
`DeferredToolGroup` 用一份紧凑的目录替代完整 schema，让模型按需加载
工具组。

## 两种策略

延迟工具支持两种策略，由 `deferred_tool_strategy` 选择（默认
`"dispatch"`）：

| | `tools` 参数 | system prompt | 每次加载的缓存代价 | 调用路径 |
|---|---|---|---|---|
| **`dispatch`**（默认） | 静态 | 静态 | **零**——schema 以消息尾部追加交付 | `deferred_tool_call` 转发器，引擎解包 |
| **`inject`** | 随加载增长 | 目录计数变化 | system + 全部历史重读一次 | 原生工具调用 |

**为什么默认 dispatch。** 在所有前缀缓存的 provider 上，工具定义渲染在
prompt 最前端。会话中途注入工具等于在整个历史**之前**插入字节，每次加载
都要按未缓存价重读全部对话。dispatch 模式在首个请求之后再不触碰 tools
数组和 system prompt——schema 通过 `load_tools` 的工具结果交付，追加在
消息历史末尾，像普通轮次一样增量缓存。

**何时选 `inject`。** 原生工具调用享有 provider 侧 schema 校验，也是模型
训练中最熟悉的调用方式。如果你的工具参数复杂、会话较短（单次加载的缓存
代价小），`inject` 用缓存效率换取调用可靠性。

## dispatch 模式如何工作

1. system prompt 携带一份简短的**静态**目录——每组一行，含描述和工具
   名。它永不改变。
2. 模型调用内置的 `load_tools(group_id)`，在**工具结果里**拿到该组的
   完整 schema。
3. 模型通过内置转发器调用已加载的工具：
   `deferred_tool_call(tool_name=..., arguments=...)`。
4. 引擎在一切流程之前解包转发调用：参数校验、
   `before_tool_call`/`after_tool_call` 钩子、权限系统、事件和 tracing
   看到的都是**真实**工具名和参数——而非信封。

```
# Deferred tool groups

These tool groups are available but not yet loaded. Call `load_tools(group_id)`
to get their full schemas, then invoke them via
`deferred_tool_call(tool_name=..., arguments=...)`.

- `mcp:github` — GitHub: Issues, PRs, repos, code search (4 tools)
  create_issue, search_repos, create_pr, list_comments
- `mcp:linear` — Linear: Project management and issue tracking (6 tools)
  create_issue, update_issue, list_projects, ...
```

几个值得了解的性质：

- **隐式加载。** 模型对从未显式加载的工具直接调用
  `deferred_tool_call` 时，中间件会即时加载并校验参数。校验失败时，
  错误结果附带完整 schema，模型一个来回即可自我纠正。
- **压缩自救。** `load_tools` 幂等——若上下文压缩丢掉了旧结果，模型
  再调一次即可拿回字节相同的 schema。
- **Fork。** fork 出的 agent（`fork_once`）继承 dispatch resolver，
  父 agent 已加载的工具在 fork 内仍可调用。

## 基本用法

向 `Agent` 传入 `deferred_tool_groups`，中间件自动创建，无需手动接线：

```python
from cubeloop import Agent
from cubeloop.deferred import DeferredToolGroup

# load_github_tools / load_linear_tools 是零参 async 可调用对象，
# 返回 list[AgentTool]。具体怎么写见下面的「编写 loader」一节，
# 给了 MCP 后端和手写 @tool 函数两种常见形态。

github_group = DeferredToolGroup(
    group_id="mcp:github",
    display_name="GitHub",
    description="Issues, PRs, repos, code search",
    tool_names=["create_issue", "search_repos", "create_pr", "list_comments"],
    loader=load_github_tools,
)

linear_group = DeferredToolGroup(
    group_id="mcp:linear",
    display_name="Linear",
    description="Project management and issue tracking",
    tool_names=["create_issue", "update_issue", "list_projects"],
    loader=load_linear_tools,
)

agent = Agent(
    model=provider.model("claude-sonnet-4-6"),
    tools=[search_tool, calculator],              # 始终可用的工具
    deferred_tool_groups=[github_group, linear_group],
    # deferred_tool_strategy="inject",            # 选用 v1 行为
)
```

### `DeferredToolGroup` 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `group_id` | `str` | 模型在 `load_tools` 调用中使用的唯一标识（如 `"mcp:github"`） |
| `display_name` | `str` | 目录中展示的人类可读标签 |
| `description` | `str` | 该组能力的一行摘要 |
| `tool_names` | `list[str]` | 目录里列出的 tool 名。**必须和 loader 返回的每个工具的 `AgentTool.name` 完全一致** —— 选择性展开（`load_tools(group_id, tool_names=[…])`）就靠这个字段匹配。 |
| `loader` | `async () -> list[AgentTool]` | 返回该组完整工具集的回调 |

### 编写 loader

`loader` 是一个零参 async 可调用对象，返回 `list[AgentTool]`。CubeLoop
只看返回类型——里面的 `AgentTool` 怎么来由你决定。两种典型写法：

**从 MCP server 加载。** `load_mcp_tools_stdio` / `load_mcp_tools_http`
返回 `MCPDiscoveryResult`，里面 `.tools` 字段就是你想要的
`list[AgentTool]`。包一层：

```python
from cubeloop.deferred import DeferredToolGroup
from cubeloop.mcp import load_mcp_tools_stdio

async def load_github_tools():
    result = await load_mcp_tools_stdio(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_TOKEN": "ghp_…"},
    )
    return result.tools   # list[AgentTool]

github_group = DeferredToolGroup(
    group_id="mcp:github",
    display_name="GitHub",
    description="Issues, PRs, repos, code search",
    tool_names=["create_issue", "search_repos", "create_pr"],
    loader=load_github_tools,
)
```

`tool_names` 里写的名字必须和 MCP server 公布的工具名一致——这些名字
discovery 之后会成为 `AgentTool.name`。如果目录里写 `create_issue` 但
server 发布的是 `github_create_issue`，选择性展开就匹配不到。

**从手写 `@tool` 函数加载。** 任何被 `@tool` 装饰的函数都是一个
`AgentTool`（`.name` 默认取函数名，可用 `@tool(name="…")` 覆盖）。
loader 就是一个返回列表的 async 函数：

```python
from cubeloop import tool
from cubeloop.deferred import DeferredToolGroup

@tool
async def create_issue(*, repo: str, title: str, body: str) -> str:
    "Open a GitHub issue."
    ...

@tool
async def search_repos(*, query: str) -> str:
    "Search public repos."
    ...

async def load_github_tools():
    return [create_issue, search_repos]   # 已经是 AgentTool

github_group = DeferredToolGroup(
    group_id="mcp:github",
    display_name="GitHub",
    description="Issues, PRs, repos, code search",
    tool_names=["create_issue", "search_repos"],
    loader=load_github_tools,
)
```

两种可以混用——同一个 list 里既有 MCP 工具又有手写工具——只要
`tool_names` 里每一个名字都能在返回的 list 里找到对应的
`AgentTool.name` 就行。如果 loader 抛异常，错误会作为 tool error 回给
模型，组保持未展开。

## `load_tools` 工具

模型调用 `load_tools` 加载一组工具，两种模式：

```
# 加载整组
load_tools(group_id="mcp:github")

# 只加载指定工具
load_tools(group_id="mcp:github", tool_names=["create_issue", "search_repos"])
```

dispatch 模式下结果携带完整 schema：

```json
{
  "group_id": "mcp:github",
  "expanded": true,
  "tool_names": ["create_issue", "search_repos"],
  "remaining": 2,
  "schemas": [
    {"name": "create_issue", "description": "...", "parameters": {"...": "..."}},
    {"name": "search_repos", "description": "...", "parameters": {"...": "..."}}
  ]
}
```

（`inject` 模式下省略 `schemas`——定义直接加入模型可见的 tools 数组。）

加载后工具在同一轮内即刻可用。

### Loader 缓存

`loader` 回调每组每 run 恰好调用**一次**。首次加载触发；后续的选择性
加载从缓存结果中过滤。loader 失败时错误返回给模型，该组保持未加载。
已加载的工具幂等——重复请求是 no-op（dispatch 模式下会重新返回相同的
schema）。

## 加载状态

中间件在 `ctx.extra` 里记录各组的加载情况：

```python
ctx.extra["expanded_groups"] = {
    "mcp:github": None,                    # 全量加载（None = 全部工具）
    "mcp:linear": ["create_issue"],        # 部分加载
    # mcp:slack 不存在 = 未加载
}
```

该状态随 checkpoint 持久化，驱动跨 run 重放。

## 跨 run 重放

从上一个 run 恢复会话时，需要还原加载状态，让转发调用立即可解析。
`prepare_resumed_state` 负责这件事——`strategy` 参数**必填**，且必须与
中间件的 strategy 一致：

```python
from cubeloop.deferred import DeferredToolsMiddleware

# saved_extra 是上一个 run 持久化的 ctx.extra
resumed = await DeferredToolsMiddleware.prepare_resumed_state(
    groups=all_groups,
    expanded=saved_extra["expanded_groups"],
    strategy="dispatch",
)

agent = Agent(
    model=model,
    tools=[*builtin_tools, *resumed.pre_loaded_tools],
    deferred_tool_groups=resumed.remaining_groups,
)
```

`prepare_resumed_state` 返回的 `ResumedState` 包含：

| 字段 | 说明 |
|---|---|
| `pre_loaded_tools` | 之前已加载组的工具，随时可被解析（dispatch 模式下对 payload 隐藏） |
| `remaining_groups` | 仍可通过 `load_tools` 加载的组 |
| `loader_cache` | 预载的工具缓存（传给 `resumed_loader_cache` 避免重复调用 loader） |

dispatch 模式下没有其他要恢复的东西：模型见过的 schema 留在消息历史
里，由 checkpointer 随会话带回。`inject` 模式下，全量加载的组退出延迟
集合（与 v1 相同）。

## 进阶：直接构造中间件

需要完全控制目录 header 或恢复种子时，自行构造
`DeferredToolsMiddleware`：

```python
from cubeloop.deferred import DeferredToolsMiddleware

mw = DeferredToolsMiddleware(
    groups=[github_group, linear_group],
    extra_ref=lambda: agent_extra,
    strategy="dispatch",
    catalog_header="# Available integrations\n\nLoad with load_tools().",
)

agent = Agent(
    model=model,
    tools=[search_tool],
    middleware=[mw],
)
```

### 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `groups` | `list[DeferredToolGroup]` | 必填 | 要延迟的组 |
| `extra_ref` | `() -> dict` | 必填 | 返回实时的 `ctx.extra` 字典 |
| `strategy` | `"dispatch" \| "inject"` | `"dispatch"` | 披露策略（见上文） |
| `catalog_header` | `str \| None` | *（按策略内置）* | 目录段的 header 文本 |
| `resumed_loader_cache` | `dict[str, list[AgentTool]] \| None` | `None` | 上一个 run 的工具缓存（避免恢复时重复调用 loader） |
| `on_tools_expanded` | `(list[AgentTool]) -> None \| None` | `None` | 新工具加载后回调（内部用于跨轮持久化） |

使用 `Agent(deferred_tool_groups=...)` 简写时，`extra_ref` 自动绑定到
`self._extra`。

## 从 0.10 迁移

延迟工具组在 CubeLoop 0.10 发布时的行为即现在的 `inject` 策略。升级后
行为变化：

- **默认策略现在是 `dispatch`。** 目录措辞改变，出现 `deferred_tool_call`
  内置工具，加载的工具不再加入模型可见的 tools 数组。用
  `Agent(deferred_tool_strategy="inject")` 或
  `DeferredToolsMiddleware(strategy="inject")` 恢复 0.10 行为。
- **`inject` 模式不再把 schema 渲染进 system prompt。** 定义本来就在
  tools 数组里；重复渲染（及其双倍 token 计费）已移除。因此
  `resumed_schemas` 构造参数和 `ResumedState.expanded_schemas` 不复存在。
- **`prepare_resumed_state` 要求显式传 `strategy=`**，避免恢复时与中间件
  的策略静默错配。

## 何时使用

**适合：**

- agent 接入 5 个以上 MCP 服务器，但每次会话通常只用 1–2 个。
- tool schema 很大（参数多、描述长）。
- 你希望跨轮保持高 prompt 缓存命中率。

**不适合：**

- agent 只有少量工具——目录和 `load_tools` 调用的开销不值得。
- 每一轮都需要全部工具——延迟只是徒增一个来回。
- tool schema 很小——上下文节省微乎其微。

## 另请参阅

- [加载 MCP 工具](../mcp/loading)——如何从 MCP 服务器获得 `AgentTool`
  列表。
- [9 个 Hook](./hooks)——驱动延迟工具的中间件 hook
  （`transform_system_prompt`、`after_tool_call`、`resolve_tool_call`）。
- [组合](./composition)——与其他中间件叠加时的组合行为。
