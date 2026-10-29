---
title: 组合规则
description: "了解 CubeLoop 如何通过 per-hook 规则组合多个中间件。"
---

# 组合规则

当你传入多个中间件——`Agent(middleware=[m1, m2, m3])`——CubeLoop 按照
**每个 hook 各自的规则**进行组合。正确的理解方式是：每个 hook 采用
最适合其工作的组合规则，你不需要记忆 "before" 或 "after" 优先级猜测。

## 规则速览

| Hook | 规则 | 顺序重要吗？ |
|---|---|---|
| `transform_context` | **链式**——每个看到上一个的输出 | 是 |
| `convert_to_llm` | **最后一个胜出** | 只有一个运行 |
| `transform_system_prompt` | **链式** | 是 |
| `resolve_tool_call` | **第一个非 `None` 胜出**——在校验前改写调用 | 第一个命中即短路 |
| `before_tool_call` | **第一个 block 停止**；非 block 累积 | block 胜出；`edited_args` 最后写入者胜出；`hitl_trace` 合并 |
| `after_tool_call` | **后面覆盖前面** | 最后写入胜出 |
| `should_stop_after_turn` | **任一 True 即停止**（OR） | 否 |
| `after_model_response` | **带合并语义的链式** | 见下文 |
| `on_run_end` | **消息拼接**——非空触发一轮额外调用 | 否 |

## `transform_context` 和 `transform_system_prompt`

链式：`m1` 的输出成为 `m2` 的输入，`m2` 的输出成为 `m3` 的输入。适用于分层转换：

```python
agent = Agent(
    middleware=[
        SlidingWindow(max_messages=20),    # m1: 丢弃最旧的
        InjectSummary(),                    # m2: 前置摘要块
    ],
)
```

`m2` 看到的是截断后的列表。用户可见的 `agent.state.messages` 不受影响——
中间件只改变模型接收到的内容。

## `convert_to_llm`

有意采用最后胜出：这是发送前的最终转换。多个所有者会打架；只选一个。
CubeLoop 强制**列表中的最后一个**实现了 `convert_to_llm` 的中间件运行。

如果你发现自己需要两个 `convert_to_llm` 中间件，将它们合并为一个
（调用点组合：写一个调用两者的中间件）。

## `resolve_tool_call`

**第一个非 `None` 返回值胜出。** 第一个返回改写后 `ToolCall` 的中间件
会短路整个链——后面的 resolver 看不到原始调用，任何 resolver 也看不到
其他 resolver 的输出。全部返回 `None` 时调用原样继续。

这与 `before_tool_call` 的链式组合是刻意不同的：一次 resolution 是
单一改写（例如把[延迟工具](./deferred-tools)的 dispatcher 解包成
真实工具），不是累积。

## `before_tool_call`

第一个 `block=True` 短路其余中间件。**非 block 返回累积：**
`edited_args` 向下游传播（每个中间件看到的是前一个编辑后的形式），
`hitl_trace` 跨链合并（当被覆盖时，旧键在 `_chain` 下归档）。

用于按从最严格到最宽松的顺序链式组合策略层：

```python
agent = Agent(
    middleware=[
        RateLimiter(),       # 配额不足时阻止
        SafetyFilter(),      # 参数危险时阻止；可能编辑
        AuditLogger(),       # 永不阻止；记录用于可观测性
    ],
)
```

如果 `RateLimiter` 返回 `block=True`，`SafetyFilter` 和 `AuditLogger`
的 `before_tool_call` 不会运行。如果 `SafetyFilter` 返回
`edited_args={"cmd": "rm /tmp/foo"}`，工具将使用编辑后的参数运行，
`AuditLogger` 通过重建的 `ctx.args` 看到它们。
`AuditLogger.after_tool_call` 仍然触发，因为那是另一个 hook。

## `after_tool_call`

每个中间件可以返回一个设置了部分字段的 `AfterToolCallResult`；
CubeLoop 合并它们，后面的结果在非 `None` 字段上覆盖前面的。完整结果：

```python
class AfterToolCallResult(BaseModel):
    content: list[Content] | None = None
    details: Any = None
    is_error: bool | None = None
    terminate: bool | None = None
```

模式：早期中间件添加丰富的 `details`，后期中间件为模型净化 `content`。
两者都运行；合并后的结果组合了一个的 `details` 和另一个的脱敏 `content`。

## `should_stop_after_turn`

任一中间件返回 `True` 即结束运行。链中其余中间件不会被执行。

```python
agent = Agent(
    middleware=[
        MaxTurns(10),
        BudgetCap(usd=0.5),
        FinalAnswerSentinel(),   # 当 assistant 说 "FINAL ANSWER" 时停止
    ],
)
```

## `after_model_response`

带结构化合并的链式。每个中间件看到**当前的 response**（可能已被前面的
中间件替换），并返回一个 `TurnAction`：

- `response: AssistantMessage | None` —— 若非 None，则替换当前的 response
  供下游中间件和循环最终持久化使用。
- `inject_messages: list[Message]` —— 跨整个链追加到单个列表中，
  然后在下一轮之前添加到上下文中。
- `decision: "natural" | "stop" | "loop_to_model"` —— **最后一个中间件的值胜出**。

```python
agent = Agent(
    middleware=[
        ProfanityRedactor(),    # 重写 response
        StructuredOutputValidator(),  # 可能返回 decision="loop_to_model"
        EventLogger(),          # 不改变 decision
    ],
)
```

如果 `StructuredOutputValidator` 返回 `decision="loop_to_model"` 而
`EventLogger` 返回 `decision="natural"`，循环看到的是 `"natural"`——
因为最后胜出。如果这不是你想要的，请重新排序。

## 混合中间件与构造函数可调用对象

`Agent(...)` 也接受显式的 hook 可调用对象（`convert_to_llm=…`、
`before_tool_call=…` 等）。当两者都存在时，**显式可调用对象胜出**：

```python
agent = Agent(
    middleware=[LoggingMiddleware()],
    before_tool_call=my_explicit_hook,   # 覆盖中间件版本
)
```

一次性 hook 使用显式形式；当行为是一个连贯的包时使用中间件类。

**例外：`resolve_tool_call` 是组合而非替换。** 显式 resolver 成为链头
——它先运行，返回 `None` 时中间件 resolver 才运行（first-non-`None`，
与该 hook 自身的组合规则一致）。如果是替换语义，
`Agent(deferred_tool_groups=…, resolve_tool_call=…)` 会静默废掉延迟
工具的 dispatch——那个中间件是框架自动创建的内部接线。

## 关于 `Middleware` 基类的说明

基类 `Middleware` 中未实现的方法会抛出 `NotImplementedError`。
`compose_middleware` 通过对比基方法检测到这一点，并**只连接**中间件
实际覆盖的 hook。

```python
class JustTransform(Middleware):
    async def transform_context(self, messages, *, ctx, signal=None):
        return messages[-10:]
    # 没有其他 hook。CubeLoop 不会调用它们。
```

## `on_run_end`

所有返回非空列表的中间件都会贡献消息；消息拼接为单一列表，在额外模型轮次
之前注入。返回 `None` 或 `[]` 的中间件被跳过。由于所有中间件都贡献，顺序
不影响消息是否被注入——只影响注入列表内的相对顺序。

## 另请参阅

- [9 个 Hook](./hooks) —— 每个 hook 的作用及触发时机。
- [示例](./examples) —— 组合在实践中的应用。
