---
title: 待办列表
description: "使用 TodoListMiddleware 给 agent 添加 write_todos 工具，跟踪多步骤任务进度。"
---

# 待办列表

`TodoListMiddleware` 给 agent 提供一个 `write_todos` 工具，用于在多步骤任务中维护结构化的待办列表。模型调用该工具来创建和更新条目；middleware 负责确保列表在运行结束前保持同步。

适用场景：需要跨多个步骤追踪进度的 agent，或者你希望模型向用户实时展示任务拆解情况。

## 基本用法

`TodoListMiddleware` 需要一个 `extra_ref` 可调用对象，它必须返回 `AgentContext.extra` 的实时引用。middleware 和工具都通过这个引用读写状态，从而在 checkpoint 后存活。

```python
from cubeloop import Agent
from cubeloop.middleware import TodoListMiddleware

agent_extra: dict = {}

agent = Agent(
    model=provider.model("claude-sonnet-4-6"),
    system_prompt="你是一个认真负责的助手。",
    middleware=[
        TodoListMiddleware(extra_ref=lambda: agent_extra),
    ],
)
```

当 agent 使用 checkpointer 时，`extra_ref` 必须指向 `AgentContext.extra` 的同一个对象，这样 todo 状态才能跨会话持久化和恢复：

```python
from cubeloop import Agent
from cubeloop.checkpointer import PostgresCheckpointer
from cubeloop.middleware import TodoListMiddleware

ctx_holder: dict[str, dict] = {}

def extra_ref() -> dict:
    return ctx_holder.setdefault("extra", {})

agent = Agent(
    model=provider.model("claude-sonnet-4-6"),
    checkpointer=PostgresCheckpointer(...),
    thread_id="conv_123",
    middleware=[
        TodoListMiddleware(extra_ref=extra_ref),
    ],
)
```

最简单的写法是 `lambda: agent.state.extra`，但 `agent.state` 只在 agent
构造完成后才有效，所以延迟绑定的 lambda 或共享字典引用都可以。

## `write_todos` 工具

该工具接受一个 `todos` 列表，每项包含：

- `content` — 简短的任务描述。
- `status` — `"pending"`、`"in_progress"` 或 `"completed"` 之一。

模型每次调用都会替换整个列表。middleware 会验证 payload，拒绝会使列表进入不一致状态的调用：

- `content` 不能为空字符串。
- 除非所有条目都已 `"completed"`，否则恰好只能有一条 `"in_progress"`。
- 同一轮中多次调用 `write_todos` 会被拒绝，列表回滚到本轮开始前的状态。
- 只有当之前所有条目都已完成时，才允许传入空列表。

## 完成守卫

当模型在未完成条目仍存在的情况下给出纯文本回复（无工具调用）时，middleware
会注入一条纠正消息，将模型循环回一轮以更新待办列表。强制轮完成后，运行正常继续。

这防止了模型完成工作后忘记将条目标为已完成就直接回复的常见情况。

## 过期提醒

如果模型连续多轮不调用 `write_todos`，middleware 会注入一条软性提醒，请模型同步列表。模型可以忽略该提醒，它不是硬性阻断。

默认阈值是连续 5 轮未调用，两次提醒之间最少间隔 5 轮。

## 自定义工具描述和系统提示

通过 `tool_description` 和 `system_prompt` 覆盖默认值：

```python
TodoListMiddleware(
    extra_ref=extra_ref,
    tool_description="为当前任务维护一份步骤清单。",
    system_prompt="## 任务追踪\n所有多步骤工作都使用 write_todos。",
)
```

`tool_description` 是模型在工具列表中看到的文字；`system_prompt` 由
`transform_system_prompt` hook 追加到 agent 的系统提示末尾。

## `ctx.extra` 状态布局

所有状态都存储在 `AgentContext.extra` 的固定 key 下：

| Key | 类型 | 说明 |
|---|---|---|
| `todos` | `list[Todo] \| None` | 当前待办列表 |
| `todo_guard_retries` | `dict` | 各守卫的重试计数 |
| `todo_guard_blocked` | `TodoGuardBlocked \| None` | 当前激活的守卫升级 payload |
| `todo_guard_suppressed` | `bool` | 守卫阻断事件后的压制标志 |
| `todo_stale_iterations` | `int` | 上次调用 `write_todos` 以来的轮数 |
| `todo_finalization_correction` | `bool \| None` | 本轮是否注入了完成纠正消息 |

这些 key 跨版本保持稳定。checkpointer 将它们作为 `ctx.extra` 的一部分持久化，恢复的会话会从模型上次保留的待办列表继续。

## 不适用场景

短会话或纯对话型 agent 不需要 `TodoListMiddleware`——工具描述和系统提示指令每轮都会消耗 token。另外，该工具由模型自主决定何时调用。如果你需要每步骤都强制产出结构化输出，考虑直接定义专用工具。
