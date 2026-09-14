---
title: 9 个 Hook
description: "CubeLoop 的 9 个中间件 hook 参考——transform_context、resolve_tool_call、before_tool_call、after_tool_call、on_run_end 等。"
---

# 9 个 Hook

`Middleware` 是一个最多包含九个可选异步方法的类。每个 hook 在 agent
循环中的精确位置触发。只实现你需要的——CubeLoop 只会连接你重写了的方法。

```python
from cubeloop import Middleware

class MyMiddleware(Middleware):
    async def transform_context(self, messages, *, ctx, signal=None):
        ...
```

将实例传入 `Agent(middleware=[MyMiddleware(), …])`。

## `transform_context`

```python
async def transform_context(
    self,
    messages: list[Message],
    *,
    ctx: AgentContext,
    signal=None,
) -> list[Message]:
    ...
```

**每次调用模型之前**触发，作用于完整的消息列表。`ctx` 是当前
`AgentContext`；使用 `ctx.extra` 存放每次运行或 checkpointer 可持久化的
middleware 状态。用于：

- 截断或总结以适配上下文窗口。
- 注入系统提醒（更好用 `transform_system_prompt`）。
- 动态添加或删除特定消息。

返回（可能是新的）列表。**不要修改输入**——返回一个新列表以免其他持有原引用的代码感到意外。

组合方式：链式——每个中间件看到上一个的输出。

## `convert_to_llm`

```python
async def convert_to_llm(
    self,
    messages: list[Message],
    *,
    ctx: AgentContext,
) -> list[Message]:
    ...
```

**在序列化给 provider 之前**触发。这是调整 LLM 看到的最终内容的最后机会。用于：

- 将工具结果精简为纯文本。
- 为非多模态 provider 将图片内容替换为文字描述。
- 压缩较长的工具输出。

组合方式：**最后一个实现胜出**（非链式）。当多个中间件可能冲突而你希望只有一个所有者时使用。

## `transform_system_prompt`

```python
async def transform_system_prompt(
    self,
    system_prompt: str,
    *,
    ctx: AgentContext,
    signal=None,
) -> str:
    ...
```

**每次调用模型之前**触发，作用于 system prompt 字符串。用于：

- 注入运行时信息（当前时间、用户角色）。
- 组合模块化的 system prompt 片段。
- A/B 测试 prompt 变体。

组合方式：链式。

`transform_context`、`convert_to_llm` 和 `transform_system_prompt` 完成后，
CubeLoop 会捕获一个只读的 `TurnExecutionContext`。它记录生成本轮请求的逻辑输入和
具体工具绑定，但不会向模型输入中加入标识符。同一模型的传输重试复用这份快照；切换到
fallback 模型时，会在同一个 turn 下创建一份新快照。工具调用会保留快照中的绑定，
而 `before_tool_call` 仍会在实际执行时按当前状态完成权限和策略检查。

## `resolve_tool_call`

```python
async def resolve_tool_call(self, tool_call: ToolCall, *, context: AgentContext, signal=None) -> ToolCall | None:
    ...
```

**每个工具调用时、先于其他一切**触发——参数校验、`before_tool_call`、
执行、事件和 tracing 都基于这个 hook 的返回值运作。返回改写后的
`ToolCall` 以重定向调用，返回 `None` 则原样通过。

返回的 `ToolCall` **必须保留原始 `id`**——工具结果消息在线上以它为键。

用于：dispatcher 解包（[延迟工具组](./deferred-tools) 正是用它把
`deferred_tool_call` 路由到真实工具）、工具别名、版本重定向。

组合方式：**第一个非 `None` 返回值胜出**——第一个返回改写结果的中间件
会短路整个链。这与顺序链式的 `before_tool_call` 不同；一个 resolver
永远看不到另一个 resolver 的输出。

## `before_tool_call`

```python
async def before_tool_call(self, ctx: BeforeToolCallContext, *, signal=None) -> BeforeToolCallResult | None:
    ...
```

**每个工具调用时**触发，在参数校验之后、`tool.execute` 之前。
context 提供：

- `ctx.assistant_message` —— 发起调用的消息。
- `ctx.tool_call` —— `ToolCall` 块。
- `ctx.args` —— *校验后的* Pydantic 实例。
- `ctx.context` —— 完整的 `AgentContext`。

返回 `BeforeToolCallResult(block=True, reason="…")` 以短路——
CubeLoop 会将该原因作为工具结果返回，并标记 `is_error=True`。
返回 `None`（或无返回）则继续执行。

用于：权限控制、速率限制、dry-run 模式、沙箱、
人机协同确认（参见 [HITL 指南](../hitl/overview)）。

组合方式：**第一个 `block=True` 短路**整个链。

## `after_tool_call`

```python
async def after_tool_call(self, ctx: AfterToolCallContext, *, signal=None) -> AfterToolCallResult | None:
    ...
```

**每个工具调用时**触发，在 `tool.execute` 返回（或抛出异常）之后。
context 额外提供：

- `ctx.result` —— execute 返回的 `AgentToolResult`。
- `ctx.is_error` —— 工具是否出错。

返回 `AfterToolCallResult(content=…, details=…, is_error=…, terminate=…)`
以覆盖结果的单个字段（任何 `None` 字段保持原值）。返回 `None` 则原样通过。

用于：编辑、重试、日志记录、结果转换。

组合方式：后面的覆盖前面的（返回值中的每个非 `None` 字段覆盖前一个值）。

## `should_stop_after_turn`

```python
async def should_stop_after_turn(self, ctx: ShouldStopAfterTurnContext) -> bool:
    ...
```

**在每个轮次边界**触发（任何工具批次之后）。返回 `True` 以结束本次运行，
不进行下一次模型调用。

用于：最大轮次限制、预算上限、应用定义的停止条件。

组合方式：**任意一个返回 `True` 即停止**（逻辑 OR 跨链）。

## `after_model_response`

```python
async def after_model_response(
    self,
    response: AssistantMessage,
    ctx: AgentContext,
    *,
    signal=None,
) -> TurnAction | None:
    ...
```

**在 assistant 消息落定后立即**触发，在 `message_end` 发出**之前**、
在任何工具调用分发**之前**。该 hook 返回一个 `TurnAction`：

```python
from cubeloop.middleware.base import TurnAction
from cubeloop.providers.base import synthetic_user_message

TurnAction(
    response=modified_message,            # 替换消息；None 则保留原消息
    inject_messages=[                     # 在下一轮之前追加的额外消息
        synthetic_user_message("…", source="my_middleware"),
    ],
    decision="natural",                   # "natural" | "stop" | "loop_to_model"
)
```

注入的 user 角色消息**必须**用 `synthetic_user_message(text, source=...)`
构造，不要直接 `UserMessage(...)`。工厂会写入 `metadata["synthetic"] = True`，
下游消费者（重放历史的 UI）就能把框架插入的提示和用户真正输入的区分开来；
`source` 是个自由格式标签，只用于 trace。判断可用 `is_synthetic_message(msg)`。

三个控制流旋钮：

- `decision="natural"`（默认）—— 正常进入工具执行 / 下一轮。
- `decision="stop"` —— 在发出 `turn_end` 和 `agent_end` 后结束运行。
  不执行工具，不再调用模型。
- `decision="loop_to_model"` —— 跳过工具执行，立即重新调用模型（配合
  `inject_messages` 使用以先添加上下文）。

用于：响应审核、带重新提示的结构化输出验证、条件路由。

组合方式：链式——每个中间件看到前一个中间件的 `response`；
`inject_messages` 跨链拼接；最后一个中间件的 `decision` 胜出。

## `on_run_end`

```python
async def on_run_end(
    self,
    ctx: AgentContext,
    *,
    signal=None,
) -> list[Message] | None:
    ...
```

**每次外层循环迭代后触发**——所有轮次和工具调用完成、循环本应退出之前。
返回非空 `list[Message]` 会将这些消息注入上下文并继续循环（agent 再跑一次）。
返回 `None` 或 `[]` 什么也不做（循环退出）。和 `inject_messages` 一样，
返回的 user 角色消息要用 `synthetic_user_message(...)` 构造，让它们带上
synthetic 标记。

每次 `prompt()` 内**可以多次触发**。中间件每次返回消息，worker 就再跑一轮。

**何时触发：**
- 正常完成（所有轮次结束后循环自然 break）。
- `should_stop_after_turn` 返回 `True`。
- `after_model_response` 返回 `decision="stop"`。

**何时不触发：**
- run 以 `stop_reason` 为 `"error"` 或 `"aborted"` 结束。
- HITL 中断（`HitlDetached` / `HitlAborted`）——run 是暂停而非结束。

用于：run 结束后的记忆提炼、对话摘要、需要完整轮次上下文的审计日志。

组合方式：所有中间件返回的消息**拼接**为单一列表，一起注入额外模型轮次。

## 中间件的构成

一个中间件不需要实现每一个 hook。只覆盖你需要的即可；基类中未实现
的 hook 会抛出 `NotImplementedError`，但 `compose_middleware` 会自动跳过它们。

```python
from cubeloop import Middleware

class MaxTurnsMiddleware(Middleware):
    def __init__(self, max_turns: int) -> None:
        self.max_turns = max_turns
        self.turns = 0

    async def should_stop_after_turn(self, ctx) -> bool:
        self.turns += 1
        return self.turns >= self.max_turns


agent = Agent(model=…, middleware=[MaxTurnsMiddleware(5)])
```

## 另请参阅

- [组合规则](./composition) —— 多个中间件定义同一 hook 时的精确语义。
- [示例](./examples) —— 速率限制、日志、重试、滑动窗口上下文、run 结束后记忆的实用中间件。
