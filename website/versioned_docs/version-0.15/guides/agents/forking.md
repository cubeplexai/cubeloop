---
title: Conversation Forking
description: "Branch a conversation at a completed-run boundary — persistent forks for new threads, ephemeral one-shot probes, and the HITL binding rules that keep them safe."
---

# Conversation Forking

A **fork** branches a conversation at a completed-run boundary. CubeLoop
gives you two variants:

- **`Agent.fork(...)`** — a *persistent* fork. Copies messages from the
  source thread up to and including a chosen `run_id` into a brand-new
  thread you can continue to converse with.
- **`Agent.fork_once(...)`** — an *ephemeral* one-shot probe. Loads a
  snapshot in memory, runs one prompt, and returns the result without
  writing anything back to the source thread.

Both APIs key off `run_id` — a stable identifier for one
`prompt() → final assistant message` cycle. Every message produced
during a prompt carries the same `run_id`, so "the boundary after
run R" is an unambiguous, reproducible cut.

## The cubebox copy-button UX

The driving use case is the cubebox "branch this reply" button: under
every assistant message in the UI there is a small affordance that
forks the conversation into a fresh thread starting from the message
above it. Users use it to explore "what if the assistant had answered
this differently?" without polluting the original thread.

In that flow:

- the host generates the new thread's id (cubebox uses uuid7),
- calls `Agent.fork(src, new, after_run_id=...)` to materialise the
  branch, and
- opens a fresh agent against the new thread.

The features below — explicit `run_id`, `active_run_id`, fork APIs,
and the HITL binding rules — are exactly what that UX needs.

## `Agent.prompt(run_id=...)` — accept-or-generate

`Agent.prompt()` now returns the `run_id` it used for the call:

```python
run_id = await agent.prompt("hello")
print(run_id)  # → "8c0b…" — server-generated
```

Hosts that want to control the id (cubebox, multi-machine relays,
anything that needs the id *before* the call completes) can supply it:

```python
import uuid_extensions   # or whatever uuid7 source you prefer

my_run_id = uuid_extensions.uuid7str()
run_id = await agent.prompt("hello", run_id=my_run_id)
assert run_id == my_run_id
```

While the run is in flight, the id is available on
`Agent.state.active_run_id`:

```python
async def watch(agent: Agent) -> None:
    while agent.state.is_streaming:
        print("running:", agent.state.active_run_id)
        await asyncio.sleep(0.1)
```

`active_run_id` is `None` between runs; it's set on entry to
`prompt()` and cleared on clean exit. If a prompt fails mid-flight
it stays set so `respond()` can resume the same run after a HITL
suspension or process restart.

## `Agent.fork(...)` — persistent branch

```python
await agent.fork(
    src_thread_id="conv_123",
    new_thread_id="conv_456",
    after_run_id="R1",
    metadata={"label": "branch experiment"},
)
```

What it does:

1. **Copies messages.** Every message on `conv_123` whose `run_id`
   belongs to a *completed* run up to and including `R1` is appended
   verbatim to `conv_456` (with new `seq` numbers). Pending or
   aborted runs after `R1` are not included.
2. **Records lineage.** The new thread row is written with
   `parent_thread_id = "conv_123"` and `forked_at_seq` equal to the
   source-thread seq of the last copied message, so you can trace
   parent/child relationships later.
3. **Stamps fork metadata.** `metadata` is written under
   `extra["fork"]` on the new thread (merged with `save_extra`
   semantics — existing keys survive).

After `fork` returns, `conv_456` exists as a fresh thread that
shares history with `conv_123` up to R1; further prompts on either
thread are independent. The source thread is untouched.

`fork` requires the checkpointer to implement the v4 Protocol
methods (`claim_run`, `mark_run_complete`, `fork`); on a v3-only
backend it raises `CheckpointerError`.

## `Agent.fork_once(...)` — ephemeral one-shot probe

```python
result = await agent.fork_once(
    src_thread_id="conv_123",
    message="What if you had said yes?",
    after_run_id="R1",
)

print(result.text)         # final assistant text
print(result.stop_reason)  # "stop" | "max_tokens" | "error" | ...
for m in result.messages:  # only the new messages added during the probe
    ...
```

`fork_once` returns a `ForkOnceResult` dataclass:

```python
@dataclass(frozen=True)
class ForkOnceResult:
    text: str               # final assistant text content joined to one string
    messages: list[Message] # messages emitted during the probe (no history prefix)
    stop_reason: str        # final assistant stop_reason
```

Under the hood it builds a transient `Agent` seeded with a snapshot
of the source thread up to `R1`, runs your prompt against it once,
and discards everything when the call returns. The probe gets its
own fresh `run_id` (you don't supply one).

### Isolation contract

`fork_once` is isolated **only at the checkpointer layer**:

- Nothing the probe does is written to the source thread, the new
  thread, or any thread — the transient agent has no `thread_id`.
- **Tool side effects are NOT isolated.** If your tool sends an
  email, calls an HTTP API, or writes a file, it will do so during a
  `fork_once`. Design tools to be safe to invoke during a probe, or
  skip `fork_once` for tools that aren't.
- **HITL is banned outright.** If any tool or middleware on the
  agent carries a `HitlBinding`, `fork_once` raises `RuntimeError`
  immediately. Build a separate agent (without `ask_user_tool` /
  `ApprovalPolicyMiddleware`) for ephemeral probes.

## HITL binding requirement

When you use `ask_user_tool` or `ApprovalPolicyMiddleware` with a
**checkpointed** HITL channel, the channel must be bound to the same
`run_id` you pass to `prompt()`:

```python
import uuid
from cubeloop import Agent
from cubeloop.hitl import CheckpointedChannel, ask_user_tool

run_id = uuid.uuid4().hex
channel = CheckpointedChannel(checkpointer=cp, thread_id="conv_123", run_id=run_id)

agent = Agent(
    model=provider.model("claude-sonnet-4-6"),
    checkpointer=cp,
    thread_id="conv_123",
    tools=[ask_user_tool(channel)],
    channel=channel,
)

# run_id MUST be supplied — it's the same id the channel is bound to
result_run_id = await agent.prompt("…", run_id=run_id)
```

`prompt()` enforces this on entry:

- If a checkpointed HITL element is present and you omit `run_id`,
  it raises `ValueError` (no auto-generation — you have to pin it).
- If the supplied `run_id` doesn't match the channel's bound id, it
  raises `ValueError` with both ids in the message.

The reason: a HITL request persists across process restarts, and
when you later `agent.respond(answer)` the framework needs to know
which `run_id` to resume under. Binding it up front makes that
deterministic.

In-memory (`InMemoryChannel`) HITL doesn't have this requirement —
nothing is persisted, so there's no resume contract to honour.

## Schema v3 → v4 migration

The fork feature requires the v4 schema. Each backend has its own
upgrade path:

- **Postgres** — see [Postgres → Schema v3→v4](../checkpointing/postgres#schema-v3--v4-migration).
- **MySQL** — see [MySQL → Schema v3→v4](../checkpointing/mysql#schema-v3--v4-migration).
- **SQLite** — auto-migrated at `__aenter__`; no host action.
- **Memory** — no schema; works out of the box.

## Legacy data behaviour

CubeLoop treats messages from before this feature (no `run_id`
column populated, i.e. `run_id IS NULL`) gracefully:

- **Mixed threads** — a thread that already has legacy messages and
  then receives a post-upgrade `prompt()` is fully forkable from any
  post-upgrade `run_id`. The fork carries the legacy messages
  through as a prefix to the new thread; they remain `run_id=NULL`
  on the copy.
- **All-legacy threads** — a thread with no post-upgrade runs has no
  `run_id` markers, so there is nothing to point `after_run_id=` at.
  `fork` / `fork_once` against such a thread raise
  `CheckpointerError`. To make a legacy thread forkable, send one
  post-upgrade `prompt()` and then fork after its run id.

`Agent.prompt()` continues to work on legacy threads either way —
the new schema is fully backwards-compatible for normal use.

## Known limitation: cross-process concurrent prompts

If two processes drive `prompt()` against the same `thread_id` at
the same time, the per-thread row lock makes message rows
linearisable, but the *semantic* snapshot a fork captures can miss
context from a sibling run that interleaved. Specifically:

- Process A starts run `Ra`, appends `[user, assistant, ...]`.
- Before `Ra` finishes, Process B starts run `Rb`, appends its own
  messages, and completes.
- A `fork(..., after_run_id=Ra)` may copy Rb's messages too,
  depending on interleaving.

The rows are correct; the conversation slice may not be what you
intended. Avoid concurrent prompts on the same thread or coordinate
at the application layer if you need strict slicing. Single-process
deployments (cubebox's default) aren't affected.

## See also

- [Postgres Checkpointing](../checkpointing/postgres) — production backend with v4 schema.
- [MySQL Checkpointing](../checkpointing/mysql) — MySQL sibling with v4 schema.
- [SQLite Checkpointing](../checkpointing/sqlite) — single-process backend with auto-migration.
- [Custom Backends](../checkpointing/custom) — Protocol details, including the new
  `snapshot`, `fork`, `claim_run`, `mark_run_complete`, `load_pending`
  methods you need to implement for fork support.
- [HITL Overview](../hitl/overview) — channel/binding mechanics.
