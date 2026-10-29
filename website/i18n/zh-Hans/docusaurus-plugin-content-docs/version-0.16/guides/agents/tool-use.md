---
title: 工具调用
description: "注册工具、并行或顺序执行、并通过 Pydantic 校验处理结果的完整指南。"
---

# 工具使用与并行执行

工具是 Agent 影响世界的方式。CubeLoop 把每个 `AgentTool` 转成模型可用
的 JSON Schema、用 Pydantic 校验参数、执行你的代码、把结果作为
`ToolResultMessage` 喂回。默认情况下,只要模型在一轮里发出多个工具
调用,它们就会并行执行。

## `@tool` 装饰器

定义工具最快的方式是给一个 async 函数加装饰器。CubeLoop 会从参数生成输入
schema,不需要单独写参数模型,也不用写样板的 `execute` 签名:

```python
from typing import Annotated
from pydantic import Field
from cubeloop import tool


@tool
async def search(
    query: Annotated[str, Field(description="自然语言查询")],
    limit: Annotated[int, Field(ge=1, le=100)] = 10,
) -> str:
    "在内部知识库里搜索。"
    results = await my_search_backend(query, limit)
    return "\n".join(results)
```

这就是一个完整、可注册的 `AgentTool`。装饰器会推断:

- **name** 来自函数名(用 `@tool(name=...)` 覆盖);
- **description** 来自 docstring(用 `@tool(description=...)` 覆盖);
- **输入 schema** 来自带类型的参数 —— `Field(...)` 的默认值与元数据和手写
  模型完全一致。

返回值可以是普通 `str`(如上,自动包成文本)、`TextContent`、内容列表,或
在你需要 `details`、`is_error`、`terminate` 时返回完整的 `AgentToolResult`:

```python
from cubeloop import tool, AgentToolResult, TextContent


@tool
async def search(query: str, limit: int = 10) -> AgentToolResult:
    "在内部知识库里搜索。"
    results = await my_search_backend(query, limit)
    return AgentToolResult(
        content=[TextContent(text="\n".join(results))],
        details={"raw_results": results},   # 透传到 ToolResultMessage.details
    )
```

要让工具顺序执行,传 `@tool(execution_mode="sequential")`。如果函数需要由
循环提供的参数,直接声明即可 —— `tool_call_id`、`signal`、`on_update` 中
任意一个声明了就会被注入,且不会出现在 schema 里:

```python
@tool
async def long_job(prompt: str, *, signal=None, on_update=None) -> str:
    "运行一个长任务并流式上报进度。"
    ...
```

## 工具的结构(完整写法)

装饰器只是 `AgentTool` 的语法糖。下面这种写法与之等价,且会一直支持 ——
当你需要动态构建工具,或在多个工具间共享同一个参数模型时,用它:

```python
from pydantic import BaseModel, Field
from cubeloop import AgentTool, AgentToolResult, TextContent


class SearchParams(BaseModel):
    query: str = Field(..., description="自然语言查询")
    limit: int = Field(10, ge=1, le=100)


async def search(tool_call_id, params: SearchParams, *, signal=None, on_update=None):
    results = await my_search_backend(params.query, params.limit)
    return AgentToolResult(
        content=[TextContent(text="\n".join(results))],
        details={"raw_results": results},   # 透传到 ToolResultMessage.details
    )


search_tool = AgentTool(
    name="search",
    description="在内部知识库里搜索。",
    parameters=SearchParams,
    execute=search,
)
```

`description` 是直接展示给模型的 —— 给模型写,不是给人看。Pydantic
的 `Field(description=…)` 会进 JSON Schema,帮模型理解每个参数。

## 默认并行

模型在一条 assistant 消息里发多个工具调用时,CubeLoop 会用
`asyncio.create_task()` 调度它们并 gather。这通常就是你想要的。

```python
agent = Agent(
    model=model,
    tools=[search_tool, fetch_url_tool, summarise_tool],
)
```

事件流会先一次性发出所有 `tool_execution_start`,中间穿插每个工具
报告进度的 `tool_execution_update`,最后按完成顺序发
`tool_execution_end`。

## 强制顺序执行

两种方式：

1. **整个 agent 级别** —— `Agent(tool_execution="sequential")`。所有
   工具批次按模型发出的顺序逐个执行。

2. **单个工具级别** —— 在 `AgentTool` 上设
   `execution_mode="sequential"`。一旦当前批次里 *任意* 一个工具是
   sequential,整个批次都退化为顺序执行。

    ```python
    write_db_tool = AgentTool(
        name="write_db",
        description="持久化一条记录。",
        parameters=WriteDbParams,
        execute=write_db,
        execution_mode="sequential",   # 出于安全考虑放弃并行
    )
    ```

内置的 `ask_user` HITL 工具（见 [HITL 指南](../hitl/overview)）设置了
`execution_mode="sequential"` —— 它会暂停 agent 等待人类输入，因此
工具批次会逐个运行。

工具会修改共享状态（DB、计数器）且你需要确定顺序时,选 sequential。

## 流式回报工具进度

长耗时工具可以推送增量更新,以 `tool_execution_update` 事件呈现：

```python
async def slow_search(tool_call_id, params, *, signal=None, on_update=None):
    for i, page in enumerate(await fetch_pages(params.query)):
        if signal and signal.is_set():
            break
        if on_update:
            on_update({"progress": i, "total": len(pages), "url": page.url})
        await process(page)
    return AgentToolResult(content=[TextContent(text="done")])
```

事件里的 `partial_result` 就是你传给 `on_update` 的对象。用小 dict 就好,
它不会进模型的 context。

## 取消正在跑的工具

`signal` 就是 `agent.abort()` set 的那个 `asyncio.Event`。在任何循环
里检查它：

```python
async def long_running(tool_call_id, params, *, signal=None, on_update=None):
    for chunk in big_dataset:
        if signal and signal.is_set():
            return AgentToolResult(content=[TextContent(text="cancelled")])
        await process_chunk(chunk)
```

如果工作是一个大 `await`,用 `asyncio.wait_for(..., timeout=…)` 包一下,
或调用底层库自己的取消方法。

## 返回错误

两种姿势：

1. **抛异常。** CubeLoop 捕获后转成 `is_error=True` 的 `AgentToolResult`,
   异常字符串作为 `TextContent`。
2. **显式返回 `is_error=True`。** 适合你想给结构化错误体的场景：

    ```python
    return AgentToolResult(
        content=[TextContent(text="超出限流,60 秒后再试")],
        is_error=True,
    )
    ```

不管哪种方式,模型都会收到一个明确标记错误的工具结果,通常会自适应
（换参数重试、问用户等）。

## 从工具结束本轮:`terminate`

工具可以声明 *"这次之后,不要再循环到模型了。"* 设
`terminate=True`：

```python
async def submit_final_answer(tool_call_id, params, *, signal=None, on_update=None):
    save_answer(params.answer)
    return AgentToolResult(
        content=[TextContent(text="submitted")],
        terminate=True,
    )
```

CubeLoop 仅在当前批次中 *每个* 工具结果都是 `terminate=True` 时才终止。
然后循环发 `turn_end`、`agent_end`,退出。

## 工具很多？把 schema 延迟加载

工具集变大——通常是因为你接了好几个 MCP server——所有 JSON schema
合起来每一轮都会吃掉几千 token 的 system prompt，即使模型这一回合只
需要其中一两组。这些 schema 还是 prompt cache 的雷区：任何 tool 改动
都会让之后每一轮的 cache 失效。

`DeferredToolGroup` 的解法是用一份紧凑的目录替代完整 schema。模型看到
每个组一行描述，需要时通过内置的 `load_tools` 工具按需加载——该组的
完整 schema 随工具结果返回，之后通过 `deferred_tool_call` 转发器调用。
tools 数组和 system prompt 整个 run 保持字节稳定，加载永不破坏 prompt
cache。（v1 行为——把加载的工具注入模型可见的 tools 数组、原生调用——
仍可通过 `deferred_tool_strategy="inject"` 选用。）

```python
from cubeloop import Agent
from cubeloop.deferred import DeferredToolGroup
from cubeloop.mcp import load_mcp_tools_stdio

async def load_github_tools():
    result = await load_mcp_tools_stdio(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
    )
    return result.tools   # list[AgentTool]

github_group = DeferredToolGroup(
    group_id="mcp:github",
    display_name="GitHub",
    description="Issues, PRs, repos, code search",
    tool_names=["create_issue", "search_repos", "create_pr"],
    loader=load_github_tools,
)

agent = Agent(
    model=provider.model("claude-sonnet-4-6"),
    tools=[search_tool, calculator],     # 始终可用
    deferred_tool_groups=[github_group], # 按需展开
)
```

loader 是一个零参 async 可调用对象，返回 `list[AgentTool]` —— 你可以
像上面这样包一次 MCP discovery，也可以直接返回你自己写的 `@tool` 装饰
函数。`tool_names` 里的名字必须和返回的每个工具的 `AgentTool.name`
完全一致，选择性展开才能找到。

适合 ≥ 5 个工具组、但每次对话通常只用一两组的场景，或者 schema 大到
足以明显撑大每一轮 system prompt 的情况。每一轮都要全部 tool 的话就
别用，延迟反而多一次往返。

完整 API（含手写工具 loader 示例）、跨 run 恢复、高级中间件构造器见
[延迟工具组](../middleware/deferred-tools)。

## 常见坑

- **忘了 keyword-only 参数** —— 开发时 `execute(tool_call_id, params)`
  能跑,但框架传 `signal=` 时会崩。签名一定保留
  `*, signal=None, on_update=None`。
- **`details` 塞太大** —— `details` 透传到 agent 事件里,但 **不会**
  给模型看。除非你下游有消费者,否则别堆大 blob。
- **Pydantic 严格度的意外** —— `Field(..., min_length=1)` 让模型通过
  JSON Schema 看到约束 —— 约束有帮助,但模型仍然偶尔发坏 JSON。
  CubeLoop 把 `ValidationError` 转成工具的 error result,你不用自己包。
- **`tools=[]` 但模型还是想用工具** —— 一般是 system prompt 里提到了
  工具。要么删掉提示,要么真的把工具给它。

## 另请参阅

- [流式事件](./streaming) —— `tool_execution_*` 事件如何嵌入到事件
  分类体系。
- [Middleware → before_tool_call](../middleware/hooks#before_tool_call)
  和 [after_tool_call](../middleware/hooks#after_tool_call) ——
  拦截、策略、重试。
- [Recipes → Weather Agent](../../recipes/weather-agent) —— 一个真发
  HTTP 请求的工具,端到端。
- [MCP 加载](../mcp/loading) —— 一次性把一个 MCP server 的整套工具
  拉下来。
