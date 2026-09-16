---
title: Streaming Events
description: "Stream events and tokens from CubeLoop agents in real time using subscribers and MessageStream."
---

# Streaming Events

CubeLoop exposes two streams, layered:

1. **Provider stream** — `MessageStream` you get from
   `provider.stream(...)`. Yields `StreamEvent`s describing the raw
   wire output: text deltas, thinking deltas, tool-call deltas, then
   `done` or `error`.
2. **Agent event stream** — what listeners registered with
   `agent.subscribe(...)` see. Eleven event types covering the full
   lifecycle of a `prompt()` call, including provider events wrapped
   inside `MessageUpdateEvent`.

Most app code only needs the agent event stream.

## The eleven agent events

| Event | Fires when |
|---|---|
| `agent_start` | At the very beginning of `prompt()` / `resume()` |
| `turn_start` | Before each model invocation (one or more per `prompt`) |
| `message_start` | Right before a new message (user / assistant / tool result) is added to history |
| `message_update` | On every provider `StreamEvent` (deltas, etc.); has `event.stream_event` attached |
| `message_end` | After a message is finalised |
| `tool_execution_start` | A tool call is dispatched (one per call, before parallel `asyncio.gather`) |
| `tool_execution_update` | A tool reported partial progress via `on_update(...)` |
| `tool_execution_end` | A tool finished (or failed) |
| `turn_end` | After all tools in a batch settled, or after a tool-less assistant response |
| `agent_end` | The whole `prompt()` call has finished — clean exit, abort, or error |

`MessageStartEvent` and `MessageEndEvent` apply to *every* message,
not just assistant ones. User and tool-result messages also get them.

## Event order for a tool-using turn

A typical "user asks question → model calls one tool → model responds"
sequence:

```
agent_start
turn_start
  message_start         (UserMessage from prompt)
  message_end           (UserMessage)
  message_start         (AssistantMessage — empty partial)
  message_update × N    (text_delta, toolcall_delta, …)
  message_end           (AssistantMessage — finalised)
  tool_execution_start
  tool_execution_end
  message_start         (ToolResultMessage)
  message_end           (ToolResultMessage)
turn_end
turn_start              (loop continues with the tool result in context)
  message_start
  message_update × N
  message_end
turn_end
agent_end
```

## Subscribing

```python
def on_event(event, signal=None):
    if event.type == "message_update" and event.stream_event.type == "text_delta":
        print(event.stream_event.delta, end="", flush=True)

unsubscribe = agent.subscribe(on_event)
```

`agent.subscribe(...)` never receives an event whose top-level `.type`
equals `"text_delta"` — that's a *provider* event type. The agent
wraps every provider event in a `MessageUpdateEvent` and tucks the
original under `event.stream_event`. Match on both fields, as above.

Listeners can be sync or async; async ones are awaited. The second
argument is the run-level `asyncio.Event` (the abort signal) — you can
inspect `signal.is_set()` to know if the run was cancelled.

To stop receiving events, call the function returned by `subscribe`.

## Filtering for text deltas (the common case)

```python
def on_event(event, signal=None):
    if event.type == "message_update":
        sub = event.stream_event
        if sub.type == "text_delta":
            print(sub.delta, end="", flush=True)
```

Equivalent defensive form using `getattr`, since only
`MessageUpdateEvent` carries a `stream_event` attribute:

```python
def on_event(event, signal=None):
    if getattr(event, "stream_event", None) and event.stream_event.type == "text_delta":
        print(event.stream_event.delta, end="", flush=True)
```

The shape that CubeLoop guarantees is the one in the table above
(`MessageUpdateEvent.stream_event.delta`). Always match the outer
event's `type == "message_update"` (or check `stream_event` exists)
before reaching into `stream_event.type`.

## Provider `StreamEvent` types

Inside a `MessageUpdateEvent.stream_event`, the type tells you what
the model is emitting:

| `stream_event.type` | Meaning | Useful fields |
|---|---|---|
| `start` | Start of an assistant message | `partial` |
| `text_start` | Beginning of a text block | `content_index` |
| `text_delta` | Token chunk | `delta` |
| `text_end` | End of a text block | `content_index` |
| `thinking_start` / `thinking_delta` / `thinking_end` | Extended thinking blocks | `delta` |
| `toolcall_start` / `toolcall_delta` / `toolcall_end` | Streaming JSON args for a tool call | `delta` (partial JSON) |
| `done` | Stream finished normally | — |
| `error` | Stream errored | `error_message` |

The `partial` field on every event is a deep-copied snapshot of the
in-progress `AssistantMessage` — handy for UIs that re-render
on every event without tracking deltas themselves.

## Iterating a raw provider stream

If you're skipping `Agent` entirely (rare — usually means you're
writing a test or a custom orchestrator), iterate the stream
directly:

```python
stream = await provider.stream(
    model=model,
    messages=[UserMessage(content=[TextContent(text="hello")])],
)
async for event in stream:
    if event.type == "text_delta":
        print(event.delta, end="", flush=True)
final = await stream.result()   # waits for the AssistantMessage
```

`stream.result()` returns even after iteration ends; it's the
canonical way to get the final message.

## Common pitfalls

- **Subscribing after `prompt()`** — Early events are gone.
  `agent.subscribe(...)` first, always.
- **Listener exceptions crash the loop?** — They don't, but they
  propagate up to the next `await`. Wrap risky work in `try/except`.
- **Order on parallel tools** — `tool_execution_start` events come in
  the model's emit order; `tool_execution_end` events come in
  completion order. Don't rely on pairs being adjacent.
- **`message_update` fires for every delta** — High-frequency tokens
  can flood a slow listener (e.g. a websocket). Batch on the consumer
  side if needed.
- **`text_delta` for thinking?** — No. Thinking blocks emit
  `thinking_*` events. Filter on `event.stream_event.type` if you only
  want visible text.

## See also

- [Tool Use](./tool-use) — the `tool_execution_*` triplet in detail.
- [Multi-turn](./multi-turn) — event order around steering and resume.
- [API Reference → StreamEvent](../../api/cubeloop-providers#streamevent)
  for the field-level schema.
