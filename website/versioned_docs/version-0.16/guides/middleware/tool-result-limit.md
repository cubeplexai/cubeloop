---
title: Tool Result Limit
description: "Use ToolResultLimitMiddleware to cap tool-result text before it is emitted to the model and host event consumers."
---

# Tool Result Limit

`ToolResultLimitMiddleware` truncates oversized tool-result text in
`after_tool_call`, before CubeLoop publishes `ToolExecutionEndEvent`.
The model, the checkpointer, and any host event consumer all see the
truncated result.

Use it when a tool can return unbounded output — shell commands,
web fetches, MCP tools — and a huge result would blow the context
window or a host event-size budget.

## Basic setup

```python
from cubeloop import Agent
from cubeloop.middleware import ToolResultLimitMiddleware

agent = Agent(
    model=provider.model("claude-sonnet-4-6"),
    middleware=[
        ToolResultLimitMiddleware(),
    ],
)
```

The default cap is 20,000 characters of text content. Results at or
under the cap pass through unchanged (`after_tool_call` returns
`None`). Over the cap, text is cut to the first `max_chars` characters
and a notice is appended:

```text
[truncated: tool result was 1082017 characters; showing first 20000. Narrow the tool call and retry.]
```

Image blocks are kept. Only `TextContent` counts toward the cap.

## Options

```python
ToolResultLimitMiddleware(
    max_chars=20_000,
    exclude_tool_names={"load_skill"},
)
```

- `max_chars` — maximum characters of text kept from the original
  result. Must be `>= 1`.
- `exclude_tool_names` — tool names that skip truncation. Use this for
  tools whose full payload is load-bearing (for example a skill body
  returned by `load_skill`).

## Placement

Put this **after** other `after_tool_call` middleware so they rewrite
the full result first, and the truncated text is what
`ToolExecutionEndEvent` carries.

CubeLoop's default composer is last-non-`None` wins. A later
middleware that returns an `AfterToolCallResult` without `content`
would drop this rewrite. Keep `ToolResultLimitMiddleware` last, or
use a composer that merges `content` and `details`.

Subagent child agents do not inherit parent middleware. Pass the same
instance (or a second one) via `SubagentMiddleware(inherited_middleware=...)`
if child tool results should be capped too.
