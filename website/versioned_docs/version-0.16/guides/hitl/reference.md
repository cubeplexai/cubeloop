---
title: API, events & reference
sidebar_position: 4
description: "CubeLoop HITL reference: Agent API, events, trace spans, error reference, testing helpers, and architecture notes."
---

# API, events & reference

## Agent API

### Constructor

```python
Agent(
    ...,
    channel: HitlChannel | None = None,    # bind a channel to this agent
)
```

If a channel is bound, the Agent wires its `_process_event` as the channel's
emit callback so `HitlRequestEvent` / `HitlAnswerEvent` are dispatched to all
agent listeners. A `_run_lock` (`asyncio.Lock`) serialises `prompt`,
`resume`, and `respond` calls.

### Properties

| Property | Returns | Description |
|---|---|---|
| `agent.channel` | `HitlChannel \| None` | The bound channel or `None`. |
| `agent.in_flight_hitl_request` | `HitlRequest \| None` | The channel's current pending payload. Raises `HitlError` if no channel bound. |

### Methods

| Method | Signature | Description |
|---|---|---|
| `load_pending_hitl_request()` | `async → HitlRequest \| None` | Reads the pending from the checkpointer (even post-detach). |
| `detach()` | `async → None` | Snapshots the pending request and triggers `HitlDetached` on the channel future. After the owning run task records `outcome=suspended` and clears its active run, it emits `AgentSuspendedEvent(pending_request=...)`. The assistant message retains unresolved tool calls and `pending_request` stays persisted. |
| `respond(*, question_id=, answer=)` | `async → None` | Resumes a suspended run. Validates qid matches persistent pending, attaches answer to channel, re-enters the loop via `run_agent_loop_resume`. |
| `abort_pending(reason=)` | `async → None` | Closes the conversation. Two-phase: Phase 1 signals in-flight await (no lock). Phase 2 appends synthetic deny tool_results + terminal `stop_reason="aborted"` assistant (under lock). |

`in_flight_hitl_request` is a synchronous property (reads the in-memory
channel slot). `load_pending_hitl_request()` is async (reads from the
checkpointer; useful post-detach or in a fresh process).

## Events

Four new events are emitted on the agent's event stream:

| Event | When | Key fields |
|---|---|---|
| `HitlRequestEvent` | Channel receives a new `confirm/approve/ask`. | `request: HitlRequest` |
| `HitlAnswerEvent` | `channel.answer()` or `channel.cancel()` fires. | `question_id: str`, `answer: Any`, `cancelled: bool`, `timed_out: bool` |
| `AgentSuspendedEvent` | A detach-triggered run has committed `outcome=suspended`. | `pending_request: HitlRequest` |
| `AgentAbortedEvent` | `agent.abort_pending()` closes the conversation. | `reason: str` |

These are all included in the `AgentEvent` union, so typed listeners
automatically cover them. `HitlRequestEvent` and `HitlAnswerEvent` are
emitted by the channel through the agent's emit binding. `AgentSuspendedEvent`
and `AgentAbortedEvent` are emitted by the Agent layer. Suspension payload is
snapshotted by `detach()`, then published by the owning run task only after the
suspended transition is committed.

## Trace spans

When the `cubeloop[tracing]` extra is installed, each HITL await is wrapped
in an OpenTelemetry span:

| Span name | Attributes |
|---|---|
| `hitl.approve` | `hitl.tool_name`, `hitl.tool_call_id`, `hitl.outcome`, `hitl.from_resume`, `hitl.duration_seconds` |
| `hitl.confirm` | `hitl.question_id`, `hitl.outcome`, `hitl.duration_seconds` |
| `hitl.ask` | `hitl.question_id`, `hitl.outcome`, `hitl.duration_seconds` |

`hitl.outcome` is one of: `approved`, `denied`, `edited`, `answered`,
`cancelled`, `timed_out`, `aborted`, `detached`.

The tracing import is lazy — if `opentelemetry` is not installed, the
channel silently falls back to a no-op span.

## Error reference

| Exception | Base class | Meaning |
|---|---|---|
| `HitlCancelled(reason)` | `BaseException` | Host called `channel.cancel(qid)`. |
| `HitlTimedOut(seconds)` | `BaseException` | Per-call or channel-default timeout expired. |
| `HitlDetached` | `BaseException` | `agent.detach()` called during HITL await. |
| `HitlAborted` | `BaseException` | `agent.abort_pending()` signalled the agent. |
| `HitlConcurrencyError` | `Exception` | `confirm/approve/ask` called while channel already has a pending request. |
| `HitlStaleAnswer` | `Exception` | `channel.answer(qid)` with a `question_id` that doesn't match the current pending. |
| `HitlNoPendingRequest` | `Exception` | `agent.respond(…)` called but no `pending_request` on the thread. |
| `HitlDurabilityNotGuaranteed` | `Exception` | Custom tool called `CheckpointedChannel.ask()` without `allow_inside_custom_tool=True`. |

`HitlControlException` (the parent of the four `BaseException` subclasses)
is intentionally NOT caught by the existing broad `except Exception:` handlers
in `cubeloop.agent.tools._prepare_tool_call` and `_execute_prepared` — this
mirrors `asyncio.CancelledError`'s pattern.


## Testing helpers

```python
from cubeloop.hitl.testing import ScriptedChannel, NoopChannel

# ScriptedChannel: pre-programmed answers, consumed in order.
ch = ScriptedChannel(answers=[
    ApproveAnswer(decision="approve"),
    {"color": "red"},                       # ask answer
    lambda req: ApproveAnswer(decision="deny", reason="test")  # callable
])
assert len(ch.history) == 3  # all HitlRequests ever seen

# NoopChannel: auto-approves everything. Useful for subagents.
ch = NoopChannel()
assert (await ch.approve("bash", "tc", {})).decision == "approve"
assert (await ch.confirm("?")) is True
assert await ch.ask([Question(key="k", prompt="p")]) == {"k": ""}
```

## Architecture notes

- **Single pending per thread.** The agent loop is sequential — at most one
  HITL request is outstanding per `thread_id`. Concurrent `confirm/approve/ask`
  raises `HitlConcurrencyError`.
- **Parallel approval batches collect answers before execution.** If one
  assistant turn contains multiple parallel tool calls that require approval,
  CubeLoop still exposes one pending request at a time. Each approved answer is
  persisted by `question_id` and replayed on the next resume attempt. Tool
  bodies start only after every gate in the parallel batch has been answered.
- **Prompt-cache prefix invariant.** Between pause and resume, the messages
  list changes only by appending tool-result messages and the next assistant
  turn at the tail. No inserting, reordering, or mutating prior messages —
  that would invalidate the provider-side prompt cache.
- **`question_id == tool_call_id` for approve requests.** No aliasing or
  mapping needed — hosts that already track `call_id` from the tool stream
  pass it directly.
- **Resume re-enters the unresolved tool cycle.** The last assistant message's
  unresolved tool calls dictate what prepares and executes next. Persisted HITL
  answers can be replayed by `question_id`, but CubeLoop does not use node-based
  graph replay semantics.
