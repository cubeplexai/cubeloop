---
title: 示例
description: "CubeLoop 中间件实用示例：速率限制、重试、结构化日志、上下文截断和 HITL。"
---

# 中间件示例

针对四种最常见需求的实用中间件：速率限制、重试、结构化日志和上下文截断。

## 速率限制 {#rate-limiting}

当用户超过配额时阻止工具调用。将 `before_tool_call` 与外部速率限制器
（令牌桶、Redis INCR……）结合。

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
        # 丢弃超过 60 秒的记录。
        self._timestamps = [t for t in self._timestamps if now - t < 60]
        if len(self._timestamps) >= self.max:
            return BeforeToolCallResult(
                block=True,
                reason=f"Rate limit: {self.max} tool calls/min exceeded. Try again shortly.",
            )
        self._timestamps.append(now)
        return None
```

使用：

```python
agent = Agent(model=…, middleware=[RateLimitMiddleware(max_calls_per_min=30)])
```

当达到限制时，模型会看到一个工具结果说"Rate limit exceeded…"并通常会等待或询问用户。

## 带退避的重试 {#retries-with-backoff}

在 `after_tool_call` 中重试失败的工具调用。最多 N 次，指数退避，
仅针对临时错误。

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

        # 按名称查找工具并重新执行，最多 max_retries 次。
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

        return None  # 放弃——保留原始错误
```

谨慎组合：重试非幂等工具（写入、发送、删除）可能造成实际损害。
将这类工具标记为 `execution_mode="sequential"` 并基于 `ctx.tool_call.name` 跳过。

## 结构化日志 {#structured-logging}

记录每一次工具调用及其参数、时长和结果。将 `before_tool_call`（记录开始时间）
与 `after_tool_call`（记录结果）配对。将开始时间存储在 `ctx.context.extra` 中。

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

`ctx.context.extra` 是存储每次运行状态的最佳位置，因为：

- 其他中间件可通过同一个 `ctx.context` 看到。
- Checkpointer 在 `agent_end` 时通过 `save_extra` 持久化。
- 新对话开始时（新的 `thread_id`）会自动重置。

## 滑动窗口截断 {#sliding-window-truncation}

通过仅保留最近的 N 条消息（加上 system prompt）来保持模型上下文边界：

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

`transform_context` 不接触 `agent.state.messages`——用户仍然看到完整历史。
模型只看到最后 N 条。

与注入被丢弃内容摘要的 `transform_system_prompt` 配合效果很好：

```python
class SummaryInjector(Middleware):
    async def transform_system_prompt(self, system_prompt, *, ctx, signal=None):
        summary = "Earlier in this conversation we discussed: …"
        return f"{system_prompt}\n\nContext: {summary}".strip()
```

## 最大轮次 / 预算上限

在达到最大轮次数或成本上限时硬停止 agent：

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
        self.cost = model_cost   # cubeloop.providers.ModelCost 或类似
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

## 用 `after_model_response` 实现结构化输出

验证 JSON 输出，如果解析失败则重新提示：

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
            return None  # 有效——正常进行
        except Exception as e:
            return TurnAction(
                inject_messages=[
                    UserMessage(content=[TextContent(text=f"Invalid output: {e}. Return valid JSON.")]),
                ],
                decision="loop_to_model",
            )
```

Agent 将跳过工具执行，并立即用上下文中的反馈消息重新提示模型。

## 人机协同工具确认

CubeLoop 在 `cubeloop.hitl` 中内置了两个 HITL 中间件：

**`ConfirmToolCallMiddleware`** —— "对此工具始终询问人类"：

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

Agent 在每次 `bash` 或 `write_file` 调用时暂停，等待宿主调用
`channel.answer(qid, ApproveAnswer(decision="approve"))`。结果驱动工具：
`approve` 执行，`deny` 带原因阻止，`edit` 重新校验并运行编辑后的参数。

**`ApprovalPolicyMiddleware`** —— 适用于通过策略引擎对工具调用进行分类的宿主：

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

`Deny` 完全跳过 channel（硬阻止）。`AskUser` 触发 channel 的 approve 流程。
`Approve` 立即返回。

完整细节——超时语义、编辑语义、事件、追踪 span、跨进程挂起/恢复——
请参见 [HITL 指南](../hitl/overview)。

## 另请参阅

- [9 个 Hook](./hooks) —— 每个 hook 的精确语义。
- [组合规则](./composition) —— 多个中间件如何组合。
- [配方](../../recipes/weather-agent) —— 中间件在真实应用中的组合。
