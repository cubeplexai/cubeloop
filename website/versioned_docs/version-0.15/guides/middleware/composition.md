---
title: Composition Rules
description: "Learn how CubeLoop composes multiple middlewares with per-hook rules: chain, last-wins, first-block-stops, and more."
---

# Composition Rules

When you pass multiple middlewares — `Agent(middleware=[m1, m2,
m3])` — CubeLoop composes them according to **per-hook rules** that
differ on purpose. The right way to think about it is: each hook has
the composition rule that makes sense for its job, and you don't have
to remember "before" or "after" precedence guesses.

## The rules at a glance

| Hook | Rule | Order matters? |
|---|---|---|
| `transform_context` | **Chain** — each sees previous output | Yes |
| `convert_to_llm` | **Last wins** | Only the last one runs |
| `transform_system_prompt` | **Chain** | Yes |
| `resolve_tool_call` | **First non-`None` wins** — rewrites the call before validation | First match short-circuits |
| `before_tool_call` | **First block stops**; non-block accumulates | Block wins; `edited_args` last-writer-wins; `hitl_trace` merges |
| `after_tool_call` | **Later overrides earlier** | Last write wins |
| `should_stop_after_turn` | **Any `True` stops** (OR) | No |
| `after_model_response` | **Chain with merge semantics** | See below |
| `on_run_end` | **Messages concatenate** — non-empty triggers one extra turn | No |

## `transform_context` and `transform_system_prompt`

Chain: `m1`'s output becomes `m2`'s input becomes `m3`'s input. Useful
for layered transforms:

```python
agent = Agent(
    middleware=[
        SlidingWindow(max_messages=20),    # m1: drop oldest
        InjectSummary(),                    # m2: prepend a summary block
    ],
)
```

`m2` sees the truncated list. The user-visible
`agent.state.messages` is untouched — middleware only changes what
the model receives.

## `convert_to_llm`

Last-wins on purpose: this is the final transform before wire
serialisation. Multiple owners would fight; pick one. CubeLoop enforces
that the **last** middleware in the list that implements
`convert_to_llm` is the one that runs.

If you find yourself needing two `convert_to_llm` middlewares,
collapse them into one (call site composition: write one that calls
both).

## `resolve_tool_call`

**First non-`None` wins.** The first middleware to return a rewritten
`ToolCall` short-circuits the chain — later resolvers never see the
original call, and no resolver ever sees another resolver's output. If
every middleware returns `None`, the call proceeds unchanged.

This deliberately differs from `before_tool_call`'s chaining: a
resolution is a single rewrite (e.g. unwrapping the
[deferred-tools](./deferred-tools) dispatcher to the real tool), not an
accumulation.

## `before_tool_call`

First `block=True` short-circuits the rest. **Non-block returns
accumulate:** `edited_args` propagates downstream (each middleware sees
the edited form from the one above), and `hitl_trace` merges across the
chain (with older keys archived under `_chain` when overwritten).

Use to chain policy layers from most-restrictive to least:

```python
agent = Agent(
    middleware=[
        RateLimiter(),       # blocks on rate quota
        SafetyFilter(),      # blocks on dangerous args; may edit
        AuditLogger(),       # never blocks; records for observability
    ],
)
```

If `RateLimiter` returns `block=True`, `SafetyFilter` and
`AuditLogger`'s `before_tool_call` don't run. If `SafetyFilter` returns
`edited_args={"cmd": "rm /tmp/foo"}`, the *tool* runs with the edited
args and `AuditLogger` sees them via the rebuilt `ctx.args`.
`AuditLogger.after_tool_call` still fires because that's a different
hook.

## `after_tool_call`

Each middleware can return an `AfterToolCallResult` with some fields
set; CubeLoop merges them, with later results overriding earlier ones
for any field that's not `None`. The full result:

```python
class AfterToolCallResult(BaseModel):
    content: list[Content] | None = None
    details: Any = None
    is_error: bool | None = None
    terminate: bool | None = None
```

Pattern: an early middleware adds rich `details`, a later one
sanitises `content` for the model. Both run; the merged result
combines `details` from one with the redacted `content` from the
other.

## `should_stop_after_turn`

Any middleware returning `True` ends the run. The rest of the chain
isn't evaluated.

```python
agent = Agent(
    middleware=[
        MaxTurns(10),
        BudgetCap(usd=0.5),
        FinalAnswerSentinel(),   # stops when assistant says "FINAL ANSWER"
    ],
)
```

## `after_model_response`

Chain with structured merge. Each middleware sees the **current
response** (which may have been replaced by an earlier middleware) and
returns a `TurnAction`:

- `response: AssistantMessage | None` — if non-None, replaces the
  current response for downstream middlewares and for what the loop
  ultimately persists.
- `inject_messages: list[Message]` — appended into a single list
  across the whole chain, then added to context before the next turn.
- `decision: "natural" | "stop" | "loop_to_model"` — the **last
  middleware's value wins**.

```python
agent = Agent(
    middleware=[
        ProfanityRedactor(),    # rewrites response
        StructuredOutputValidator(),  # may decide="loop_to_model"
        EventLogger(),          # decision unchanged
    ],
)
```

If `StructuredOutputValidator` returns `decision="loop_to_model"` and
`EventLogger` returns `decision="natural"`, the loop sees `"natural"`
— because last wins. Reorder if that's not what you wanted.

## Mixing middleware with constructor callables

`Agent(...)` also accepts explicit hook callables (`convert_to_llm=…`,
`before_tool_call=…`, etc.). When both are present, the **explicit
callable wins**:

```python
agent = Agent(
    middleware=[LoggingMiddleware()],
    before_tool_call=my_explicit_hook,   # overrides the middleware version
)
```

Use the explicit form for one-off hooks; use middleware classes when
behaviour is a coherent bundle.

**Exception: `resolve_tool_call` composes instead of replacing.** An
explicit resolver becomes the chain head — it runs first, and middleware
resolvers run only if it returns `None` (first-non-`None`, matching the
hook's own composition rule). Replacing would silently disable deferred
dispatch for `Agent(deferred_tool_groups=…, resolve_tool_call=…)`, where
the middleware is auto-created internal wiring.

## A note on `Middleware` base class

The base `Middleware` class's unimplemented methods raise
`NotImplementedError`. `compose_middleware` detects this by comparing
to the base method and **only** wires hooks the middleware actually
overrides. You don't need to `pass`-implement every method.

```python
class JustTransform(Middleware):
    async def transform_context(self, messages, *, ctx, signal=None):
        return messages[-10:]
    # No other hooks. CubeLoop won't call them.
```

## `on_run_end`

All middleware returning non-empty lists contribute; their messages are
concatenated into a single list and injected before the extra model
turn. Middleware returning `None` or `[]` are skipped. Because all
contribute, order doesn't affect whether messages are injected — only
their relative order within the injected list.

## See also

- [The 9 Hooks](./hooks) — what each hook does and when it fires.
- [Examples](./examples) — composition in practice.
