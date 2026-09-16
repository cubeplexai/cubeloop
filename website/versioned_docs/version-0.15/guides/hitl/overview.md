---
title: Overview
sidebar_position: 1
description: "CubeLoop human-in-the-loop: the channel, the three verbs (confirm, approve, ask), and timeouts."
---

# Human-in-the-Loop (HITL)

CubeLoop's HITL channel lets an agent **pause and ask a human** before proceeding.
It handles two recurring patterns with a single primitive:

1. **Sandbox tool confirmation** — a dangerous tool (bash, file writes, API
   mutations) needs approve / deny / edit from a human before running.
2. **Mid-run structured questions** — the agent needs a specific selection or
   form answer from the user before it can continue.

The channel is an `await`-able coroutine collaborator. Tool authors write
`await channel.ask(...)` and the channel handles the pause. Host code
(subscribers) renders the pending request and posts an answer. Two backends
cover the full spectrum:

- `InMemoryChannel` — CLI, notebooks, tests. Process dies, pending is lost.
- `CheckpointedChannel` — web services. Persists the pending request to a
  `Checkpointer` so a different process (or the same process after restart)
  can pick up and answer hours later.

## Architecture

```text
┌────────────────────────────────────────────────────────┐
│ Host (cubebox web service / CLI / TUI)                │
│                                                        │
│   subscribe to channel.pending  ◄──── answer / cancel  │
└──────────────────────┬─────────────────────────────────┘
                       │
         ┌─────────────▼──────────────┐
         │     HitlChannel (Protocol) │
         │  confirm / approve / ask   │
         │  pending / answer / cancel │
         └──────┬──────────────┬──────┘
                │              │
     ┌──────────▼───┐   ┌──────▼────────────┐
     │InMemoryChannel│   │CheckpointedChannel│
     │ (Future+Queue)│   │  (Future + persi- │
     │               │   │   sts to Checkpo- │
     │               │   │   inter)          │
     └───────────────┘   └───────────────────┘
                │              │
                │  used by ────┤
                │              │
     ┌──────────▼──────────────▼────────────┐
     │  ask_user tool    ConfirmToolCallMW  │
     │  (structured form) (approve/deny/    │
     │                     edit per-tool)   │
     └──────────────────────────────────────┘
                       │
         ┌─────────────▼──────────────┐
         │   cubeloop Agent loop      │
         │   (BeforeToolCallResult    │
         │    carries hitl_trace)     │
         └────────────────────────────┘
```

## Channel types

### `InMemoryChannel`

For single-process use. Holds an `asyncio.Future` internally — the host
calls `channel.answer()` from the same event loop.

```python
from cubeloop.hitl import InMemoryChannel

channel = InMemoryChannel(
    default_timeout=180.0,  # per-call timeout; None = no timeout (default)
)
```

### `CheckpointedChannel`

For cross-process use. On every `confirm / approve / ask`, the `HitlRequest`
is persisted via `Checkpointer.save_pending_request(thread_id, ...)`. On
`channel.answer()`, the pending is cleared. On `HitlDetached` (graceful
suspend), the pending stays so a later `Agent.respond()` can resume.

```python
from cubeloop.hitl import CheckpointedChannel
from cubeloop.checkpointer.sqlite import SQLiteCheckpointer  # or postgres / mysql

async with SQLiteCheckpointer("path/to.db") as cp:
    channel = CheckpointedChannel(
        checkpointer=cp,
        thread_id="conversation-42",
        default_timeout=None,              # cross-process: typical to disable
        allow_inside_custom_tool=False,    # safety gate (see Cross-process & durability)
    )
```

`CheckpointedChannel.__init__` validates the checkpointer implements
`save_pending_request` and `load_pending_request` — it fails early with
`HitlError` if not.

## The three verbs

### `confirm(prompt, *, details, timeout, signal) → bool`

A simple yes/no question. The host answers `True` or `False`.

```python
if await channel.confirm("Deploy to production?", details={"env": "prod"}):
    await deploy()
```

### `approve(tool_name, tool_call_id, args, *, details, timeout, signal) → ApproveAnswer`

The sandbox-confirm verb. Returns an `ApproveAnswer` with one of three decisions:

| Decision | Result |
|---|---|
| `"approve"` | Tool runs with the original args. |
| `"deny"` | Tool is blocked; `tool_result.is_error=True` with `details["hitl"]["decision"]="human_deny"`. |
| `"edit"` | Tool runs with the edited args (re-validated against the tool's pydantic parameter model). |

For `approve` requests, the envelope's `question_id` is set to the LLM's
`tool_call_id` — no separate UUID, so host code can correlate by the same
ID it already tracks from the tool stream.

### `ask(questions, *, timeout, signal) → dict[str, str | list[str]]`

A structured form with one or more `Question` objects. Each question can be:

- **Free-text** (`options=None`)
- **Single-select** (`options=[...]`, `multi_select=False`)
- **Multi-select** (`options=[...]`, `multi_select=True`)
- **"Other" with input** (option has `allow_input=True` — user types free text)

```python
from cubeloop.hitl.types import Question, Option

answers = await channel.ask([
    Question(key="framework", prompt="Which framework?", options=[
        Option(label="React", value="react"),
        Option(label="Vue", value="vue"),
        Option(label="Other", value="other", allow_input=True),
    ]),
    Question(key="features", prompt="Enable:", multi_select=True, options=[
        Option(label="Auth", value="auth"),
        Option(label="Payments", value="payments"),
    ]),
])
# answers == {"framework": "react", "features": ["auth", "payments"]}
```

## Timeout

Both channels accept a `default_timeout` constructor arg and every verb
accepts a per-call `timeout` kwarg (per-call overrides default).

| Layer | API |
|---|---|
| Channel constructor | `InMemoryChannel(default_timeout=30.0)` |
| Per-call override | `channel.confirm("ok?", timeout=10.0)` |
| Per-call off | `channel.approve(..., timeout=None)` when channel default is set |

Timeout expiry raises `HitlTimedOut(BaseException)` from the agent-side
`await`. The surrounding tool or middleware translates it to
`tool_result.is_error=True` with `details["hitl"]["decision"]="timed_out"`,
so the model sees a clean denial and can react naturally. The envelope's
`HitlRequest.timeout_seconds` is filled automatically so the frontend can
render a countdown.
