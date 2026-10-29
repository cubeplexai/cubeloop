---
title: Subagents
description: "Use SubagentMiddleware to delegate self-contained work to child CubeLoop agents."
---

# Subagents

`SubagentMiddleware` adds a `subagent` tool. When the model calls that tool,
CubeLoop creates an ephemeral child `Agent`, runs a self-contained prompt, and
returns the child agent's final assistant text as the tool result.

Use it when a parent agent needs to delegate bounded work such as research,
review, extraction, or a focused implementation pass.

## Define subagent specs

```python
from cubeloop import Agent
from cubeloop.middleware import SubagentMiddleware, SubagentSpec

subagents = {
    "researcher": SubagentSpec(
        name="researcher",
        description="Researches a narrow question and returns concise notes.",
        system_prompt="You are a precise research assistant.",
    ),
    "reviewer": SubagentSpec(
        name="reviewer",
        description="Reviews code for bugs, regressions, and missing tests.",
        system_prompt="You are a rigorous senior code reviewer.",
    ),
}

model = provider.model("claude-sonnet-4-6")

agent = Agent(
    model=model,
    middleware=[
        SubagentMiddleware(
            subagents=subagents,
            default_model=model,
            shared_tools=[web_search],
        ),
    ],
)
```

If the model requests an unknown `subagent_type`, CubeLoop falls back to the
`general-purpose` subagent. If you do not define one, the middleware supplies a
basic default.

## Tool access and middleware inheritance

Subagents only receive the tools you pass in:

- `shared_tools` are available to every child agent.
- `SubagentSpec.tools` are available only to that subagent type.
- `excluded_tool_names` prevents recursive or host-only tools from being shared.

Use `inherited_middleware` for middleware every child should run. Use
`SubagentSpec.middleware` for behavior specific to one subagent type.

### Checkpointed HITL bindings are stripped

A tool or middleware whose `.hitl` is a checkpointed `HitlBinding` carries
the parent run's `run_id`. The child runs under its own fresh `run_id`, so
inheriting such an element would crash with `Agent has checkpointed HITL
elements bound to run_ids ...` at the child's `prompt()` entry. The
middleware drops these elements automatically before constructing the
child agent. The most common case is the parent's `ask_user_tool(...)`
appearing in `shared_tools` — it simply won't be in the child's tool list.

Elements with no `.hitl`, or with a non-checkpointed binding, are
inherited as-is. Use a non-HITL approval / policy mechanism if a subagent
needs gating.

## Stream child events to your host

Applications often need child events in their own UI or audit log. Provide an
`event_mapper` and optional `event_handler`:

```python
def map_event(event):
    if event.type == "text_delta":
        return {"type": "subagent_text_delta", "delta": event.delta}
    return None

async def handle_event(agent_id, payload):
    await ui_stream.send({"agent_id": agent_id, **payload})

SubagentMiddleware(
    subagents=subagents,
    default_model=model,
    event_mapper=map_event,
    event_handler=handle_event,
)
```

Mapped payloads are also stored in the parent tool result under
`details["subagent_events"]`.

## Tracing and aborts

Pass a `Tracer` via `tracer=...` to attach tracing to each child run. Nested
subagent spans share the parent trace, so `cubeloop trace view <trace_id>` renders
the parent tool call and child run together.

The parent run's abort signal is forwarded to the child agent. If the parent is
aborted while a subagent is running, the child agent is aborted as well.

## Prompting guidelines

The parent model should send a self-contained `prompt` in the `subagent` tool
call. Do not rely on the child seeing the parent's full hidden context. Include
the goal, relevant facts, constraints, and desired output shape in the prompt.
