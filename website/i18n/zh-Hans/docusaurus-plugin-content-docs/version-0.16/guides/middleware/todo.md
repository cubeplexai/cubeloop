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

agent = Agent(
    model=provider.model("claude-sonnet-4-6"),
    system_prompt="你是一个认真负责的助手。",
    middleware=[
        TodoListMiddleware(extra_ref=lambda: agent.session.state_context),
    ],
)
```

当 agent 使用 checkpointer 时，`extra_ref` 必须指向 `AgentContext.extra` 的同一个对象，这样 todo 状态才能跨会话持久化和恢复：

```python
from cubeloop import Agent
from cubeloop.checkpointer import PostgresCheckpointer
from cubeloop.middleware import TodoListMiddleware

agent = Agent(
    model=provider.model("claude-sonnet-4-6"),
    checkpointer=PostgresCheckpointer(...),
    thread_id="conv_123",
    middleware=[
        TodoListMiddleware(extra_ref=lambda: agent.session.state_context),
    ],
)
```

公开的 `agent.session.state_context` 就是实时 checkpoint extra，lambda 在 agent
构造后才求值。恢复时，先调用 `await agent.session.load_checkpoint()`，再执行新请求。

## `write_todos` 工具

该工具接受 `todos` 列表，以及可选的 `wait_for_tasks` 任务 ID。每个 Todo 包含：

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

## 等待宿主管理的后台工作

应用有跨 run 后台任务时，可以给 middleware 传入异步 `validate_task_wait`：

```python
from cubeloop.agent.types import AgentContext
from cubeloop.middleware import TaskWaitBinding, TaskWaitValidation

async def validate_task_wait(
    task_ids: list[str],
    ctx: AgentContext,
    prior: TaskWaitBinding | None,
) -> TaskWaitValidation:
    return await task_service.validate_wait(task_ids, ctx, prior)

middleware = TodoListMiddleware(
    extra_ref=lambda: agent.session.state_context,
    validate_task_wait=validate_task_wait,
)
```

示例的 `task_service` 是应用自己的服务，不是 CubeLoop 内置功能。宿主负责校验当前
scope、发起者权限、取消状态，以及任务是否仍有可观察、待交付的结果。成功时返回
`TaskWaitValidation(status="valid", validation={...})`；`validation` 只能包含可 JSON
序列化的非敏感证据，它会随 checkpoint 保存，并通过下次回调的 `prior` 返回。
没有配置回调时，非空 `wait_for_tasks` 会被拒绝。

成功的声明绑定任务 ID、Todo 快照、run 和当前输入边界。纯文本收尾触发未完成守卫前，
CubeLoop 再次调用宿主校验；仍有效则正常结束当前 run，不勾选未完成 Todo，也不强制
追加模型调用。不带任务 ID 的普通 Todo 更新会清除声明。新的用户或内部输入、新 run
的初始消息或 HITL 答复（包括人工批准、拒绝或编辑工具调用）都会使旧声明失效。
策略自动批准不属于新的人工输入；压缩上下文后无法确认原输入边界时也需重新声明。

用户后来取消已验证的工作时，宿主证明取消发生在原校验之后，可返回
`status="cancelled"` 和非空 `reason`，允许自然收尾并保留未完成列表和取消原因。
cancelled 不能用来建立新声明。无效 ID、查询失败、旧绑定仍走原有守卫；不能借等待
绕过 Todo 参数错误、显式停止、HITL 或其他 middleware 的控制。

可恢复声明需要 checkpointer 和实时 extra 引用。工具轮次结束后、下一次模型调用前，
以及发布 HITL 暂停和正常结束时都会保存 extra；保存失败不算成功持久化等待声明。
CubeLoop 不负责启动、轮询、取消或
投递后台任务，也不保持等待中的 Session；宿主负责任务生命周期及后续输入或新 run。
没有 Todo 的 agent 不需要为了结束 run 而额外调用该工具。

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
| `todo_task_wait` | `dict \| None` | 已校验任务 ID、Todo 快照、run／输入边界和宿主证据 |
| `todo_task_wait_outcome` | `dict \| None` | 最近一次 valid／cancelled 收尾判断及原因 |

这些 key 跨版本保持稳定。checkpointer 将它们作为 `ctx.extra` 的一部分持久化，恢复的会话会从模型上次保留的待办列表继续。

## 不适用场景

短会话或纯对话型 agent 不需要 `TodoListMiddleware`——工具描述和系统提示指令每轮都会消耗 token。另外，该工具由模型自主决定何时调用。如果你需要每步骤都强制产出结构化输出，考虑直接定义专用工具。
