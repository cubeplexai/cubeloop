---
title: Multi-turn Conversations
description: "Build multi-turn conversational agents with CubeLoop's stateful agent loop and message history."
---

# Multi-turn Conversations

A "turn" in CubeLoop is one round of: user input → model response (and
maybe tools) → optional more model responses to tool results. The
agent's `_messages` list grows across turns; this guide covers how to
drive multi-turn flows correctly and how to inject input while the
agent is mid-thought.

## The basic pattern

Just call `prompt` again after the previous call returns:

```python
await agent.prompt("Hi, my name is Sam.")
await agent.prompt("What's my name?")
# → "Your name is Sam."
```

History lives on `agent.state.messages`. CubeLoop appends each user
message, each assistant message, and each tool result. The provider
gets the full list every time, so context windows matter (see
[Context Management](#context-management) below).

## In-flight steering: `agent.steer()`

Sometimes the user has a correction or extra context while the model
is still mid-turn. Use `steer()`:

```python
import asyncio

async def main():
    task = asyncio.create_task(agent.prompt("Plan a 5-day trip to Kyoto."))
    await asyncio.sleep(2)
    # User changed their mind:
    agent.steer(UserMessage(content=[TextContent(text="Make it 3 days, not 5.")]))
    await task
```

A steering message is enqueued; the loop picks it up between turns (in
practice, between a tool batch and the next model call). The agent
sees it before its next response — it's not lost.

`steering_mode` on `Agent` controls drain behaviour:

- `"one-at-a-time"` (default) — one queued message per pickup point.
- `"all"` — every queued message is drained at once.

Service hosts that must correlate a queued message with durable storage can
use the identified session API:

```python
from cubeloop import InputEnvelope

receipt = agent.session.submit_input(
    InputEnvelope(
        input_id=client_message_id,
        message=UserMessage(content=[TextContent(text=content)]),
        mode="steer",
    )
)
```

`queued` means the in-memory queue accepted the message. It is not a durable
acknowledgement. Subscribe to the session's `input_committed` event; its
durability is `checkpoint` only after the message append succeeds. Steering
and follow-up remain separate queues with their existing pickup points and
batch policies.

## Queued follow-ups: `agent.follow_up()`

`follow_up` is for *"after the current run, start a new turn with
this."* It's the typical pattern for a chat UI: the user types while
the assistant is still responding.

```python
agent.follow_up(UserMessage(content=[TextContent(text="And what about Osaka?")]))
# When the current prompt() finishes, the loop picks this up
# automatically and starts a new turn.
```

If the agent is idle when you call `follow_up`, you still need to
trigger a run — most apps call `await agent.resume()` once `prompt()`
returns, to drain the queue.

## `resume()` — continue from the last message

`resume()` is the "pick up where we left off" entry point. Two uses:

1. **After loading from a checkpointer.** The state has messages but
   no in-flight prompt. `resume()` looks at the last message and acts:
   - assistant → expects a queued steer/follow_up to convert into a
     new user turn; otherwise raises.
   - tool_result → re-invokes the model with the tool output.
   - user → re-invokes the model on the user message.
2. **After abort.** Once `agent.abort()` has cleanly torn down a run,
   you can resume.

```python
async with SQLiteCheckpointer("conv.db") as cp:
    agent = Agent(model=…, checkpointer=cp, thread_id="conv-1")
    await agent.prompt("hello")    # loads existing history first
    await agent.prompt("how are you?")
```

The first `prompt()` after instantiation loads any existing thread.
Subsequent calls just append.

## Context management

CubeLoop does **not** truncate or summarise context on your behalf. The
full message list is sent to the model on every turn. Strategies:

- **Manual truncation** — Implement a
  [`transform_context`](../middleware/hooks#transform_context)
  middleware that returns a sliding window.
- **Summarisation pass** — Periodically inject a summary message and
  drop older ones via `transform_context`.
- **Different `convert_to_llm`** — Reshape the history just before
  serialisation (the last opportunity), without mutating
  `agent.state.messages`. The user-visible history stays full; the
  model sees less.

See [Middleware → Examples](../middleware/examples#sliding-window-truncation)
for a working example.

## Cancellation and idle waits

```python
agent.abort()                # signals the current run to stop
await agent.wait_for_idle()  # awaits the run-cleanup
```

`wait_for_idle()` is a no-op if the agent is already idle. It's safe
to call anywhere.

## Restoring state from disk

```python
from cubeloop.checkpointer import SQLiteCheckpointer

async with SQLiteCheckpointer("conv.db") as cp:
    agent = Agent(
        model=model,
        checkpointer=cp,
        thread_id="user-42",
    )
    # First prompt() restores the saved history if any.
    await agent.prompt("continue our chat")
```

The checkpoint `extra` mapping is also restored. Middleware that wants to
persist per-thread state should write into `context.extra`; the checkpointer's
`save_extra` is called after a turn that produced tool results, when the run
suspends, and at `agent_end`. Hosts should use
`agent.session.state_context`, which is the stable live mapping, and
`await agent.session.load_checkpoint()` when they need an explicit idle-time
restore. Messages and extra are installed together, including checkpoints
whose message list is empty.

## Common pitfalls

- **Starting any execution while another is in flight or finalizing** raises
  `ExecutionBusy`. Use `steer()` or `follow_up()`, or `wait_for_idle()` first.
- **`resume()` with last message = assistant and no queue** raises
  `"Cannot continue from message role: assistant"`. Either queue a
  follow-up first or call `prompt()` instead.
- **History grows unbounded** — without a `transform_context`
  middleware you'll eventually hit context limits. Plan for
  truncation/summarisation early.
- **Concurrent agents on the same `thread_id`** — Append-only is safe
  for ordering but two agents writing the same thread will interleave
  messages. Use one agent instance per thread or coordinate at the
  application layer.

## See also

- [Streaming Events](./streaming) — exact event order around
  steering/follow_up.
- [Checkpointing → SQLite](../checkpointing/sqlite) — persisting history.
- [Recipes → Persistent Chat](../../recipes/persistent-chat) — full
  multi-turn app with history reload.
