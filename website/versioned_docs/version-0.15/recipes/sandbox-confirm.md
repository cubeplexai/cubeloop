---
title: Sandbox Confirm with ApprovalPolicyMiddleware
description: "Implement sandbox tool confirmation with CubeLoop's ApprovalPolicyMiddleware — auto-allow, deny, or human-confirm."
---

# Recipe: Sandbox Confirm with `ApprovalPolicyMiddleware`

Use case: a web service where every tool call goes through a rule engine that
classifies it as auto-allow, hard-deny, or human-confirm.

## Step 1: Define your policy function

The policy receives a `BeforeToolCallContext` and returns an
`ApprovalDecision` — `Approve()`, `Deny(reason)`, or `AskUser(...)`.

```python
from cubeloop.hitl import Approve, AskUser, Deny

# Mock rule engine — replace with your actual policy catalog.
def classify_command(cmd: str) -> tuple[str, str | None]:
    """(tier, reason) — "allow", "block", or "confirm"."""
    if cmd.startswith(("ls", "cat", "head", "grep", "find")):
        return ("allow", None)
    if "rm -rf /" in cmd or cmd.startswith("dd"):
        return ("block", "destructive I/O")
    return ("confirm", "needs human review")


def sandbox_policy(ctx):
    cmd = ctx.args.cmd  # ctx.args is the validated pydantic model
    tier, reason = classify_command(cmd)

    if tier == "allow":
        return Approve()
    if tier == "block":
        return Deny(reason=reason or "blocked by policy")
    return AskUser(
        timeout_seconds=180,
        details={"matched_rule": tier, "impact": reason or "unknown"},
    )
```

`ctx.args` is the **validated pydantic model**. Access fields as
attributes (`ctx.args.cmd`). The middleware internally converts it to a
dict via `_args_to_dict` for the channel's approve payload, but your
policy receives the typed model.

## Step 2: Wire into the agent

```python
from cubeloop.agent.agent import Agent
from cubeloop.checkpointer.postgres import PostgresCheckpointer
from cubeloop.hitl import ApprovalPolicyMiddleware, CheckpointedChannel

async def main():
    async with PostgresCheckpointer("postgresql://...") as cp:
        channel = CheckpointedChannel(checkpointer=cp, thread_id="session-1")

        agent = Agent(
            model=anthropic.model("claude-sonnet-4-6"),
            system_prompt="You are a helpful assistant with access to a bash shell.",
            tools=[bash_tool],
            middleware=[
                ApprovalPolicyMiddleware(channel, policy=sandbox_policy),
            ],
            channel=channel,
            checkpointer=cp,
            thread_id="session-1",
        )

        await agent.prompt("list files then delete temp logs")
        # Agent runs; when bash is invoked, sandbox_policy decides:
        #   ls → Approve() → runs immediately
        #   rm /tmp/logs → AskUser() → channel suspends, HitlRequestEvent fires
```

## Step 3: Host handles the pending request

```python
async def host_loop(channel: CheckpointedChannel):
    async for req in channel.subscribe():
        if req.payload.kind == "approve":
            tool_name = req.payload.tool_name
            command = req.payload.args.get("cmd", "")
            details = req.payload.details or {}
            timeout = req.timeout_seconds  # seconds for the frontend countdown

            # Render to the frontend: tool_name, command, details["matched_rule"],
            # details["impact"], and a countdown based on timeout.
            human_answer = await my_frontend.show_confirm(
                tool_name=tool_name,
                command=command,
                details=details,
                timeout=timeout,
            )
            # Build an ApproveAnswer from the human's decision.
            from cubeloop.hitl import ApproveAnswer
            human_answer = ApproveAnswer(
                decision=ui_response["decision"],          # "approve" | "deny" | "edit"
                reason=ui_response.get("reason"),           # only for deny
                edited_args=ui_response.get("edited_args"), # only for edit
            )
            await channel.answer(req.question_id, human_answer)
        elif req.payload.kind == "ask":
            await channel.answer(req.question_id, await my_frontend.show_form(req))
        else:  # confirm
            await channel.answer(req.question_id, await my_frontend.show_confirm(req))
```

## Decision semantics

| Human chose | Tool outcome | `hitl_trace["decision"]` | Model sees |
|---|---|---|---|
| Approve | Runs with original args | unset (passthrough, no HITL details) | Normal `tool_result` |
| Deny | Blocked | `"human_deny"` | `tool_result.is_error=True` with user's reason |
| Edit | Runs with edited args | `"edit"` + `original_args` / `edited_args` | Normal `tool_result` (from the edited execution) |

Policy decisions (hard-deny without asking the human) carry
`hitl_trace["decision"]="policy_deny"`.

## Timeout behaviour

If the human doesn't respond within `timeout_seconds`, the middleware
translates to `BeforeToolCallResult(block=True, deny_reason="approval_timeout")`.
The model sees `tool_result.is_error=True` with
`details.hitl.decision="timed_out"` and naturally produces a follow-up turn
explaining the timeout.

## Aborting

If the user closes the tab or an admin kills the conversation:

```python
await agent.abort_pending(reason="user closed tab")
```

This closes the conversation cleanly: synthetic deny tool_results are
appended for any unresolved tool calls, a terminal
`AssistantMessage(stop_reason="aborted")` is persisted, and
`AgentAbortedEvent` is emitted. The next `agent.prompt(...)` starts fresh.

## Run the example

A self-contained, runnable version of this recipe is in the repository at
[`examples/sandbox_confirm.py`](https://github.com/cubeplexai/cubeloop/blob/main/examples/sandbox_confirm.py).
It wires up a simulated bash tool and a policy that auto-allows reads, denies
destructive writes, and auto-approves everything else via the host loop.

```bash
git clone https://github.com/cubeplexai/cubeloop && cd cubeloop
uv sync

export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY [+ OPENAI_BASE_URL]
uv run python examples/sandbox_confirm.py
```
