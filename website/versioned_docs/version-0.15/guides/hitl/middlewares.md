---
title: Middlewares & ask_user
sidebar_position: 2
description: "Gate tool calls with ConfirmToolCallMiddleware / ApprovalPolicyMiddleware, and let the model ask structured questions with the ask_user tool."
---

# Middlewares & the `ask_user` tool

## Built-in middlewares

### `ConfirmToolCallMiddleware`

"Always ask the human for tool *names* in this set."

```python
from cubeloop.hitl import ConfirmToolCallMiddleware

# Set-based matching — only ask for these tool names
agent = Agent(
    ...,
    middleware=[
        ConfirmToolCallMiddleware(
            channel,
            require_confirm={"bash", "write_file", "http_post"},
            timeout_seconds=180,
        ),
    ],
)
```

`require_confirm` options:

| Value | Behavior |
|---|---|
| `None` (default) | Confirm **every** tool. |
| `set[str]` | Confirm if `tool_call.name` is in the set. |
| `Callable[[BeforeToolCallContext], bool]` | Custom predicate — inspect args, context, etc. |

An optional `details_fn(ctx: BeforeToolCallContext) -> dict` enriches the
approve request payload with extra context the frontend can render (e.g.
matched rule name, impact preview, affected file list).

### `ApprovalPolicyMiddleware`

For hosts with a **policy engine** that classifies tool calls into three
tiers — auto-allow, hard-deny, or human-confirm.

```python
from cubeloop.hitl import Approve, ApprovalPolicyMiddleware, AskUser, Deny

def my_policy(ctx):
    if ctx.tool_call.name == "read_file":
        return Approve()                               # passthrough
    if ctx.tool_call.name.startswith("dangerous_"):
        return Deny(reason="blocked by policy")        # hard block, no human asked
    return AskUser(timeout_seconds=180)                # human confirm

agent = Agent(
    ...,
    middleware=[ApprovalPolicyMiddleware(channel, policy=my_policy)],
)
```

The policy function can be sync or async (`await`-able). It returns one of:

| Return | Effect |
|---|---|
| `Approve()` | Tool runs; channel never invoked. |
| `Deny(reason)` | Tool blocked; `hitl_trace["decision"]="policy_deny"`. |
| `AskUser(timeout_seconds=..., details=...)` | Channel invoked; human chooses approve/deny/edit. |

Policy-deny and human-deny produce different `hitl_trace` keys (`policy_deny`
vs `human_deny`) so audit and trace can distinguish them.

## `ask_user` built-in tool

A tool the **model** invokes when it needs structured input from the user.
The factory returns an `AgentTool` named `"ask_user"` with
`execution_mode="sequential"` — it can't run in parallel with other tools.

```python
from cubeloop.hitl import ask_user_tool

agent = Agent(
    ...,
    tools=[bash_tool, ask_user_tool(channel)],
)
```

The tool description explicitly steers the model away from using `ask_user`
for free-form clarification ("for free-form questions, end your turn with
text — the user's next message is your answer"). The model should only
invoke it when a **structured** answer is needed.

The `Parameters` prompt schema the model sees:

| Field | Type | Description |
|---|---|---|
| `questions` | array | One or more question objects. |
| `questions[].key` | string | Field name in the answer dict. |
| `questions[].prompt` | string | The question text. |
| `questions[].options` | array (optional) | Selection options. `None` = free text. |
| `questions[].options[].label` | string | Human-facing label. |
| `questions[].options[].value` | string | Value returned to agent. |
| `questions[].options[].allow_input` | bool (default `false`) | "Other / please specify." |
| `questions[].multi_select` | bool (default `false`) | Allow multiple selections. |
| `questions[].required` | bool (default `true`) | Can the user skip this? |

Cancel and timeout are surfaced as `tool_result.is_error=True` with
`details["hitl"]["outcome"]="cancelled"` / `"timed_out"` — the model sees
a clean error tool result and can react. Other HITL control exceptions
(HitlDetached, HitlAborted) propagate to the Agent layer, not the model.


## When to use `ask_user` vs end of turn

| Goal | Use |
|---|---|
| Free-text follow-up question to user | Just end the turn with the question as text; the user's next message is your answer. |
| Structured selection (one of N) | `ask_user` tool with `options`. |
| Multi-select ("pick any of") | `ask_user` tool with `multi_select=True`. |
| "Other" with free-text input | `ask_user` tool option with `allow_input=True`. |
| Confirm/edit tool args before run | `ConfirmToolCallMiddleware` or `ApprovalPolicyMiddleware`. |
