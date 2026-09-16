---
title: 中间件与 ask_user
sidebar_position: 2
description: "用 ConfirmToolCallMiddleware / ApprovalPolicyMiddleware 把关工具调用，并用 ask_user 工具让模型发起结构化提问。"
---

# 中间件与  工具

## 内置中间件

### `ConfirmToolCallMiddleware`

"在集合中的工具名一律询问人类。"

```python
from cubeloop.hitl import ConfirmToolCallMiddleware

agent = Agent(
    ...,
    middleware=[
        ConfirmToolCallMiddleware(
            channel,
            require_confirm={"bash", "write_file"},
            timeout_seconds=180,
        ),
    ],
)
```

`require_confirm` 选项：

| 值 | 行为 |
|---|---|
| `None`（默认） | 确认**所有**工具 |
| `set[str]` | 仅当 `tool_call.name` 在集合中时确认 |
| `Callable[[BeforeToolCallContext], bool]` | 自定义断言 —— 检查参数、上下文等 |

### `ApprovalPolicyMiddleware`

适用于需要**策略引擎**将工具调用分为 auto-allow、hard-deny 和
human-confirm 三类的宿主。

```python
from cubeloop.hitl import Approve, ApprovalPolicyMiddleware, AskUser, Deny

def my_policy(ctx):
    if ctx.tool_call.name == "read_file":
        return Approve()                              # 直接放行
    if ctx.tool_call.name.startswith("dangerous_"):
        return Deny(reason="策略阻止")                  # 硬阻止，不询问人类
    return AskUser(timeout_seconds=180)               # 人类确认

agent = Agent(
    ...,
    middleware=[ApprovalPolicyMiddleware(channel, policy=my_policy)],
)
```

策略函数可以是同步或异步（可 `await`）。返回值：

| 返回 | 效果 |
|---|---|
| `Approve()` | 工具运行；channel 从未调用 |
| `Deny(reason)` | 工具阻塞；`hitl_trace["decision"]="policy_deny"` |
| `AskUser(timeout_seconds=..., details=...)` | 调用 channel；人类选择 approve/deny/edit |

## `ask_user` 内置工具

模型在需要用户结构化输入时调用的工具。工厂函数返回一个名为
`"ask_user"` 的 `AgentTool`，`execution_mode="sequential"`。

```python
from cubeloop.hitl import ask_user_tool

agent = Agent(
    ...,
    tools=[bash_tool, ask_user_tool(channel)],
)
```

工具描述明确引导模型不要拿 `ask_user` 做自由文本澄清（"对于自由文本
提问，直接用文本结束 turn——用户的下一条消息就是答案"）。

取消和超时以 `tool_result.is_error=True` 体现，
`details["hitl"]["outcome"]="cancelled"` / `"timed_out"` —— 模型看到
干净的���误工具结果并能做出反应。其他 HITL 控制异常（HitlDetached、
HitlAborted）传播到 Agent 层，不暴露给模型。


## `ask_user` 对比 end of turn

| 目标 | 用法 |
|---|---|
| 对用户的自由文本追问 | 直接用文本结束 turn——用户的下一条消息就是答案 |
| 结构化选择 (N 选一) | `ask_user` 工具带 `options` |
| 多选 ("任意选择") | `ask_user` 工具带 `multi_select=True` |
| "其他" 可自由文本输入 | `ask_user` 工具选项带 `allow_input=True` |
| 确认/编辑工具参数后才运行 | `ConfirmToolCallMiddleware` 或 `ApprovalPolicyMiddleware` |
