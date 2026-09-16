---
title: Cross-process & durability
sidebar_position: 3
description: "Suspend a CubeLoop agent on a HITL request and resume it in another process with a CheckpointedChannel; durable-scope rules."
---

# Cross-process flow & durability

## Cross-process (web service) flow

Full lifecycle of a suspend/resume cycle:

```python
import asyncio
from cubeloop.agent.agent import Agent
from cubeloop.checkpointer.postgres import PostgresCheckpointer
from cubeloop.hitl import (
    ApproveAnswer, CheckpointedChannel, ConfirmToolCallMiddleware,
)

# ---------- Process 1: HTTP POST /chat ----------

async with PostgresCheckpointer("postgresql://...") as cp:
    channel = CheckpointedChannel(checkpointer=cp, thread_id="conv-42")

    agent = Agent(
        model=...,
        tools=[bash_tool],
        middleware=[
            ConfirmToolCallMiddleware(channel, require_confirm={"bash"}),
        ],
        channel=channel,
        checkpointer=cp,
        thread_id="conv-42",
    )

    # prompt() blocks until the HITL request is answered OR the host
    # calls detach(). Same-process host coroutines can answer directly
    # via channel.answer(); cross-process hosts use respond().
    task = asyncio.create_task(agent.prompt("delete temp files"))

    # Poll for pending (or subscribe to channel for SSE push)
    for _ in range(1000):
        pending = channel.pending
        if pending is not None:
            break
        await asyncio.sleep(0.1)

    assert pending is not None
    # Render pending.payload to the frontend (an ApproveRequest:
    # tool_name="bash", args={"cmd":"rm /tmp/foo"}, ...)

    # Graceful suspend — persist the assistant message + unresolved
    # tool_calls and keep pending_request in DB. detach() commits the
    # HitlDetached transition; prompt() then emits AgentSuspendedEvent
    # after recording outcome=suspended.
    # The HTTP handler returns 200 { status: "awaiting_approval" }.
    await agent.detach()
    await task


# ---------- Process 2: HTTP POST /respond ----------

async with PostgresCheckpointer("postgresql://...") as cp:
    channel = CheckpointedChannel(checkpointer=cp, thread_id="conv-42")

    agent = Agent(
        model=...,
        tools=[bash_tool],
        middleware=[
            ConfirmToolCallMiddleware(channel, require_confirm={"bash"}),
        ],
        channel=channel,
        checkpointer=cp,
        thread_id="conv-42",
    )

    # Loads the persisted history + pending, validates the question_id
    # matches, attaches the answer to the channel, re-enters the loop
    # where the last assistant message had unresolved tool calls.
    await agent.respond(
        question_id=request.json["call_id"],
        answer=ApproveAnswer(decision="approve"),
    )
    # The bash tool executes, the model receives the tool_result and
    # produces the next assistant turn. Conversation continues normally.
```

**If the user closes the tab without answering:**

```python
await agent.abort_pending(reason="user closed tab")
# Phase 1: signals the in-flight HITL await (if any) to raise HitlAborted.
# Phase 2: appends synthetic deny ToolResultMessage(s) for unresolved
#   tool_calls, appends a terminal AssistantMessage(stop_reason="aborted"),
#   clears persisted pending, emits AgentAbortedEvent.
# No model call is made. The conversation is closed.
```


## Durable scope

Durable cross-process resume (survives process death) is supported at two
well-defined safe suspension points:

1. **`before_tool_call` approval gate** — the approval middleware calls
   `channel.approve(...)` *before* the tool's `execute()` body runs. No tool
   side effects exist yet. Resume re-enters the loop and either runs the
   (possibly edited) tool body or substitutes a synthetic deny tool_result.
2. **`ask_user` tool body** — whose entire `execute()` body is
   `return await channel.ask(...)`. Resume replays nothing because nothing
   else happened.

**Custom tools that mix HITL with other work inside `execute()` are NOT
durable cross-process by default.** If such a tool's process dies mid-execute,
anything that ran before the channel call would be lost. CubeLoop will raise
`HitlDurabilityNotGuaranteed` unless the `CheckpointedChannel` is constructed
with `allow_inside_custom_tool=True` — the caller must acknowledge the
idempotency contract (the tool body must be a pure HITL wait with no
preceding observable side effects).
