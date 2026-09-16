---
title: Examples
description: "Working middleware examples for CubeLoop: rate limiting, retries with backoff, structured logging, context truncation, and HITL."
---

# Middleware Examples

Working middleware for the four most common needs: rate limiting,
retries, structured logging, and context truncation.

## Rate limiting

Block tool calls when a user exceeds quota. Combine
`before_tool_call` with an external rate-limiter (a token bucket, a
Redis INCR, …).

```python
import time
from cubeloop import Middleware
from cubeloop.agent.types import BeforeToolCallResult


class RateLimitMiddleware(Middleware):
    def __init__(self, max_calls_per_min: int) -> None:
        self.max = max_calls_per_min
        self._timestamps: list[float] = []

    async def before_tool_call(self, ctx, *, signal=None):
        now = time.monotonic()
        # Drop entries older than 60 s.
        self._timestamps = [t for t in self._timestamps if now - t < 60]
        if len(self._timestamps) >= self.max:
            return BeforeToolCallResult(
                block=True,
                reason=f"Rate limit: {self.max} tool calls/min exceeded. Try again shortly.",
            )
        self._timestamps.append(now)
        return None
```

Use:

```python
agent = Agent(model=…, middleware=[RateLimitMiddleware(max_calls_per_min=30)])
```

When the limit hits, the model sees a tool result that says "Rate
limit exceeded…" and usually waits or asks the user.

## Retries with backoff

Retry failed tool calls inside `after_tool_call`. Up to N times, with
exponential backoff, only for transient errors.

```python
import asyncio
from cubeloop import Middleware
from cubeloop.agent.types import AfterToolCallResult


class RetryMiddleware(Middleware):
    def __init__(self, max_retries: int = 3, base_delay: float = 0.5) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def after_tool_call(self, ctx, *, signal=None):
        if not ctx.is_error:
            return None

        # Find the tool by name and re-execute up to max_retries times.
        tool = next(
            (t for t in (ctx.context.tools or []) if t.name == ctx.tool_call.name),
            None,
        )
        if tool is None:
            return None

        for attempt in range(1, self.max_retries + 1):
            await asyncio.sleep(self.base_delay * (2 ** (attempt - 1)))
            try:
                new_result = await tool.execute(
                    ctx.tool_call.id,
                    ctx.args,
                    signal=signal,
                    on_update=None,
                )
                return AfterToolCallResult(
                    content=new_result.content,
                    details={"retried": attempt, "original_error": ctx.result.content},
                    is_error=False,
                )
            except Exception:
                continue

        return None  # give up — original error stays
```

Combine with caution: retrying non-idempotent tools (writes, sends,
deletes) can cause real damage. Mark such tools `execution_mode="sequential"`
and skip them here based on `ctx.tool_call.name`.

## Structured logging

Log every tool call with its arguments, duration, and outcome.
Pairs `before_tool_call` (to record start time) with `after_tool_call`
(to record the result). Stash the start time in `ctx.context.extra`.

```python
import time, logging
from cubeloop import Middleware

log = logging.getLogger("cubeloop.tools")


class ToolLoggingMiddleware(Middleware):
    async def before_tool_call(self, ctx, *, signal=None):
        ctx.context.extra.setdefault("_tool_starts", {})[ctx.tool_call.id] = time.monotonic()
        return None

    async def after_tool_call(self, ctx, *, signal=None):
        started = ctx.context.extra.get("_tool_starts", {}).pop(ctx.tool_call.id, None)
        duration_ms = int((time.monotonic() - started) * 1000) if started else None
        log.info(
            "tool_call",
            extra={
                "tool_name": ctx.tool_call.name,
                "args": ctx.args.model_dump() if hasattr(ctx.args, "model_dump") else ctx.args,
                "is_error": ctx.is_error,
                "duration_ms": duration_ms,
            },
        )
        return None
```

`ctx.context.extra` is the right place to stash per-run state because
it's:

- Visible to other middleware via the same `ctx.context`.
- Persisted by checkpointers via `save_extra` at `agent_end`.
- Reset when a new conversation starts (a new `thread_id`).

## Sliding-window truncation

Keep the model's context bounded by retaining only the most recent N
messages, plus the system prompt:

```python
from cubeloop import Middleware


class SlidingWindow(Middleware):
    def __init__(self, max_messages: int = 20) -> None:
        self.max_messages = max_messages

    async def transform_context(self, messages, *, ctx, signal=None):
        if len(messages) <= self.max_messages:
            return messages
        return messages[-self.max_messages:]
```

`transform_context` doesn't touch `agent.state.messages` — the user
keeps seeing the full history. The model just sees the last N.

Pairs well with a `transform_system_prompt` that injects a summary of
what was dropped:

```python
class SummaryInjector(Middleware):
    async def transform_system_prompt(self, system_prompt, *, ctx, signal=None):
        summary = "Earlier in this conversation we discussed: …"
        return f"{system_prompt}\n\nContext: {summary}".strip()
```

## Built-in compaction

`CompactionMiddleware` summarizes older turns into `ctx.extra` and
passes the model a compressed view: one summary message plus recent
messages. The full conversation history remains in `agent.state`.

```python
from cubeloop.middleware import CompactionMiddleware

main_model = main_provider.model("claude-sonnet-4-6")
summary_model = cheap_provider.model("claude-haiku-4-5")

agent = Agent(
    model=main_model,
    middleware=[
        CompactionMiddleware(
            summary_model=summary_model,
            max_tokens_before_compact=80_000,
            keep_tail_tokens=8_000,
        ),
    ],
)
```

The summary call uses `Provider.generate(...)` with
`temperature=0.0`, `reasoning=ReasoningControl(mode="off")`, and
`max_output_tokens` set from `max_summary_tokens`.

## Built-in subagents

`SubagentMiddleware` adds one `subagent` tool that runs a temporary
child `Agent` with a self-contained prompt. CubeLoop captures child
events and returns the child agent's final assistant text as the tool
result.

```python
from cubeloop.middleware import SubagentMiddleware, SubagentSpec

subagents = {
    "researcher": SubagentSpec(
        name="researcher",
        description="Researches a narrow question",
        system_prompt="You are a concise research assistant.",
    )
}

agent = Agent(
    model=model,
    middleware=[
        SubagentMiddleware(
            subagents=subagents,
            default_model=model,
            shared_tools=[web_search],
        ),
    ],
)
```

Host applications can pass `event_mapper` and `event_handler` to map
child `AgentEvent`s into their own UI stream or audit log. Billing,
SSE payload shape, and product-specific tool filtering stay in the
host application.

## Max turns / budget cap

Hard-stop the agent after a maximum number of turns or a cost cap:

```python
class MaxTurns(Middleware):
    def __init__(self, max_turns: int) -> None:
        self.max_turns = max_turns
        self.turns = 0

    async def should_stop_after_turn(self, ctx):
        self.turns += 1
        return self.turns >= self.max_turns


class BudgetCap(Middleware):
    def __init__(self, usd: float, model_cost) -> None:
        self.cap = usd
        self.cost = model_cost   # cubeloop.providers.ModelCost or similar
        self.spent = 0.0

    async def should_stop_after_turn(self, ctx):
        m = ctx.message
        if m.usage:
            self.spent += (
                (m.usage.input_tokens / 1_000_000) * self.cost.input
                + (m.usage.output_tokens / 1_000_000) * self.cost.output
            )
        return self.spent >= self.cap
```

## Structured output with `after_model_response`

Validate JSON output and re-prompt if it doesn't parse:

```python
import json
from cubeloop import Middleware
from cubeloop.middleware.base import TurnAction
from cubeloop.providers.base import TextContent, UserMessage


class JSONOutputValidator(Middleware):
    def __init__(self, schema_cls) -> None:
        self.schema = schema_cls

    async def after_model_response(self, response, ctx, *, signal=None):
        text = "".join(
            c.text for c in response.content if isinstance(c, TextContent)
        )
        try:
            obj = json.loads(text)
            self.schema.model_validate(obj)
            return None  # valid — proceed naturally
        except Exception as e:
            return TurnAction(
                inject_messages=[
                    UserMessage(content=[TextContent(text=f"Invalid output: {e}. Return valid JSON.")]),
                ],
                decision="loop_to_model",
            )
```

The agent will skip tool execution and immediately re-prompt the
model with the feedback message in context.

## Human-in-the-loop tool confirmation

CubeLoop ships two built-in HITL middlewares in `cubeloop.hitl`:

**`ConfirmToolCallMiddleware`** — "always ask the human for this tool":

```python
from cubeloop.hitl import ConfirmToolCallMiddleware, InMemoryChannel

channel = InMemoryChannel()
agent = Agent(
    model=…,
    middleware=[
        ConfirmToolCallMiddleware(
            channel,
            require_confirm={"bash", "write_file"},
        ),
    ],
)
```

The agent pauses on every `bash` or `write_file` call and waits for the
host to `channel.answer(qid, ApproveAnswer(decision="approve"))`. The
result drives the tool: `approve` runs it, `deny` blocks with a reason,
`edit` re-validates and runs the edited args.

**`ApprovalPolicyMiddleware`** — for hosts that classify tool calls via
a policy engine:

```python
from cubeloop.hitl import Approve, ApprovalPolicyMiddleware, AskUser, Deny

def my_policy(ctx):
    if ctx.tool_call.name in ("read_file", "grep"):
        return Approve()
    if ctx.tool_call.name.startswith("dangerous_"):
        return Deny(reason="blocked")
    return AskUser(timeout_seconds=180)

agent = Agent(
    model=…,
    middleware=[ApprovalPolicyMiddleware(channel, policy=my_policy)],
)
```

`Deny` skips the channel entirely (hard block). `AskUser` triggers the
channel's approve flow. `Approve` returns immediately.

Full details — timeout semantics, edit semantics, events, trace spans,
cross-process suspend/resume — are in the [HITL guide](../hitl/overview).

## See also

- [The 9 Hooks](./hooks) — exact semantics of each hook.
- [Composition Rules](./composition) — how multiple middlewares
  combine.
- [Recipes](../../recipes/weather-agent) — middleware composed into
  real-world apps.
