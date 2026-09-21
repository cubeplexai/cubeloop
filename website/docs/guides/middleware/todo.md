---
title: Todo List
description: "Use TodoListMiddleware to give your agent a write_todos tool for tracking multi-step work."
---

# Todo List

`TodoListMiddleware` gives the agent a `write_todos` tool for maintaining a
structured checklist during multi-step tasks. The model calls the tool to
create and update items; the middleware enforces that the list stays in sync
before the run ends.

Use it when agents need to track progress across many steps, or when you want
the model to show the user a live breakdown of what it's doing.

## Basic setup

`TodoListMiddleware` requires an `extra_ref` callable that returns the live
`AgentContext.extra` dict. The middleware and the tool both read and write
through this reference so state survives checkpointing.

```python
from cubeloop import Agent
from cubeloop.middleware import TodoListMiddleware

agent = Agent(
    model=provider.model("claude-sonnet-4-6"),
    system_prompt="You are a thorough assistant.",
    middleware=[
        TodoListMiddleware(extra_ref=lambda: agent.session.state_context),
    ],
)
```

When the agent is checkpointed, pass the same `extra_ref` that points to
`AgentContext.extra` so todo state is persisted and restored across sessions:

```python
from cubeloop import Agent
from cubeloop.checkpointer import PostgresCheckpointer
from cubeloop.middleware import TodoListMiddleware

# extra_ref must return the same object as AgentContext.extra.
# The helper below is the standard pattern with a checkpointed agent.
agent = Agent(
    model=provider.model("claude-sonnet-4-6"),
    checkpointer=PostgresCheckpointer(...),
    thread_id="conv_123",
    middleware=[
        TodoListMiddleware(extra_ref=lambda: agent.session.state_context),
    ],
)
```

The public `agent.session.state_context` is the live checkpoint extra mapping.
The lambda is evaluated after construction. On recovery, call
`await agent.session.load_checkpoint()` before executing a new request.

## The `write_todos` tool

The tool accepts a `todos` list and optional `wait_for_tasks` task IDs. Each Todo has:

- `content` — a short task description.
- `status` — one of `"pending"`, `"in_progress"`, or `"completed"`.

The model replaces the entire list on every call. The middleware validates the
payload and rejects calls that would leave the list in an inconsistent state:

- Empty content strings are rejected.
- Exactly one item must be `"in_progress"` unless all items are `"completed"`.
- Calling `write_todos` more than once in a single turn is rejected; the list
  rolls back to its pre-turn state.
- An empty list is only accepted when all prior items were already completed.

## Finalization guard

When the model delivers a plain-text response (no tool calls) while unfinished
items remain in the list, the middleware injects a correction message and loops
the model back for one extra turn to update the checklist. After that forced
turn the run proceeds normally regardless of what the model does.

This prevents the common pattern where a model completes work but forgets to
mark items as done before responding.

## Waiting for host-managed background work

Pass `validate_task_wait` when your application owns background tasks that can
outlive a run. The async callback has this signature:

```python
from cubeloop.agent.types import AgentContext
from cubeloop.middleware import TaskWaitBinding, TaskWaitValidation

async def validate_task_wait(
    task_ids: list[str],
    ctx: AgentContext,
    prior: TaskWaitBinding | None,
) -> TaskWaitValidation:
    # Your service checks scope, actor permissions, cancellation, and whether
    # the tasks still have an observable result to deliver.
    return await task_service.validate_wait(task_ids, ctx, prior)

middleware = TodoListMiddleware(
    extra_ref=lambda: agent.session.state_context,
    validate_task_wait=validate_task_wait,
)
```

The host service in this example is application code, not part of CubeLoop.
It returns `TaskWaitValidation(status="valid", validation={...})` only after
checking the tasks. Store only JSON-compatible, non-secret evidence in
`validation`; it is checkpointed and returned in the next `prior` binding.
No validator means a nonempty `wait_for_tasks` is rejected.

The declaration binds task IDs to the successful Todo update, run, and current
input boundary. Before forcing a finalization correction, CubeLoop asks the
host again. A valid wait lets the run end naturally without completing unfinished
Todos or making another model call. An ordinary Todo update without task IDs
clears the declaration. New user or internal input, including a new run's initial
message or a HITL answer, invalidates it. A changed or unprovable boundary after compaction also
requires a new declaration.

If the user cancels previously validated work, the callback may return
`status="cancelled"` with a nonempty `reason`, after proving cancellation happened
after the original validation. This also allows natural completion, preserving
the unfinished list and cancellation reason. It cannot establish a new wait.
Invalid tasks, lookup failures, or stale bindings keep the ordinary guard.
Malformed Todo updates, explicit stop, HITL, and other middleware decisions are
not bypassed.

Use a checkpointer and the live extra mapping for recoverable declarations.
Extra is saved after tool turns before another model call, before a HITL
suspension is published, and at normal run end. A failed save does not count as
a successfully checkpointed wait.
CubeLoop does not start, poll, cancel, or deliver background tasks, and does not
keep a waiting Session alive. The host remains responsible for their lifecycle
and for delivering results in a later input or run. Agents without a Todo list
do not need to call `write_todos` just to end a run.

## Stale-todo reminder

If the model makes several tool calls in a row without touching `write_todos`,
the middleware injects a soft reminder asking it to sync the list. The model is
free to ignore the reminder; it is never a hard block.

The threshold is 5 consecutive non-`write_todos` turns, with a minimum of 5
turns between successive reminders.

## Customizing the tool description and system prompt

Pass `tool_description` and `system_prompt` to override the defaults:

```python
TodoListMiddleware(
    extra_ref=extra_ref,
    tool_description="Maintain a checklist of steps for the current task.",
    system_prompt="## Task tracking\nUse write_todos for all multi-step work.",
)
```

`tool_description` is the text the model sees in its tool list; `system_prompt`
is appended to the agent's system prompt by the `transform_system_prompt` hook.

## State layout in `ctx.extra`

All state is stored under well-known keys in `AgentContext.extra`:

| Key | Type | Description |
|---|---|---|
| `todos` | `list[Todo] \| None` | Current checklist |
| `todo_guard_retries` | `dict` | Per-guard retry counters |
| `todo_guard_blocked` | `TodoGuardBlocked \| None` | Active guard escalation payload |
| `todo_guard_suppressed` | `bool` | Guard suppression flag after a blocked episode |
| `todo_stale_iterations` | `int` | Turns since last `write_todos` call |
| `todo_finalization_correction` | `bool \| None` | Whether a finalization correction was injected this turn |
| `todo_task_wait` | `dict \| None` | Validated task IDs, Todo snapshot, run/input boundary, and host evidence |
| `todo_task_wait_outcome` | `dict \| None` | Last valid/cancelled finalization decision and reason |

These keys are stable across versions. Checkpointers persist them as part of
`ctx.extra`, so a resumed session starts with the same checklist the model left.

## When not to use it

Skip `TodoListMiddleware` for short or conversational agents — the tool
description and system prompt instructions consume tokens on every turn. The
tool is also self-governed: the model decides whether and when to call it. For
workflows where you need guaranteed structured output at each step, consider
explicit tool definitions instead.
