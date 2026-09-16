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

agent_extra: dict = {}

agent = Agent(
    model=provider.model("claude-sonnet-4-6"),
    system_prompt="You are a thorough assistant.",
    middleware=[
        TodoListMiddleware(extra_ref=lambda: agent_extra),
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

In practice the simplest pattern is to pass `lambda: agent.state.extra` — but
`agent.state` is only valid after the agent is constructed, so a late-binding
lambda or a shared dict reference both work.

## The `write_todos` tool

The tool accepts a single `todos` list. Each item has:

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

These keys are stable across versions. Checkpointers persist them as part of
`ctx.extra`, so a resumed session starts with the same checklist the model left.

## When not to use it

Skip `TodoListMiddleware` for short or conversational agents — the tool
description and system prompt instructions consume tokens on every turn. The
tool is also self-governed: the model decides whether and when to call it. For
workflows where you need guaranteed structured output at each step, consider
explicit tool definitions instead.
