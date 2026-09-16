---
title: 多 Provider 故障转移
description: "用内置 FallbackBoundModel 在 LLM provider 之间自动降级。"
---

# Recipe：多 Provider 故障转移

主 provider 限流、不可用或撞上上下文窗口时，自动切到下一条，不要把 agent 打崩。
CubeLoop 内置 `FallbackBoundModel`，不必再手写适配器。

**预计耗时：** 5 分钟。
**依赖：** `cubepi`，以及至少两套 provider API key。

## 内置：`FallbackBoundModel`

```python
import os
from cubeloop import Agent, FallbackBoundModel
from cubeloop.providers.anthropic import AnthropicProvider
from cubeloop.providers.openai import OpenAIProvider

anthropic = AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"])
openai = OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])

model = FallbackBoundModel(
    chain=(
        anthropic.model("claude-opus-4-8"),   # 主模型
        openai.model("gpt-5"),                # 备援
    )
)

agent = Agent(model=model, system_prompt="You answer concisely.")
await agent.prompt("Capital of Mongolia?")
```

`FallbackBoundModel` 会看每条腿的**第一个** stream 事件。该事件上的类型字段
（`error_type`、`error_code`、`status_code` …）和抛出的异常走**同一套判定**。
一旦出现非 error 的首包，流原样转发——流中途出错不会再 hop。

一轮里的策略：

1. **先同模型重试**（默认 3 次重试 / 共 4 次尝试），仅针对瞬时的
   `RateLimited` 和 `ProviderUnavailable`。会遵守
   `RateLimited.retry_after`，并受 `max_retry_after`（默认 5s）封顶。
2. **再 hop** 到下一条。residual `ProviderBadRequest` 和 `ModelNotFound`
   立刻 hop（不同模型重试）——分类漏报「其实是这个模型/网关的问题」比
   「全链都会炸的真 schema 错误」更常见。
3. **粘住**本轮第一次**成功**的那条腿：同一 `Agent.prompt` 里后续
   `stream` / `generate` 直接打这条。HITL `respond()` 继续该 run，保留
   sticky。下一次用户 `prompt` 重置回 `chain[0]`。Subagent 自己开 run，
   **不继承**父级 index。Sticky 依赖 `begin_fallback_run()`（Agent 会调）；
   单独调用 `FallbackBoundModel` 不会粘。

## 默认触发条件

默认会 **failover（hop）** 的错误：

| 错误 | 同模型重试 | Hop |
|---|---|---|
| `RateLimited` | 是 | 重试耗尽后 |
| `ProviderUnavailable` | 是 | 重试耗尽后 |
| `ContextLengthExceeded` | 否 | 是，但会跳过 `context_window` 装不下 `tokens_in` 的腿 |
| `ModelNotFound` | 否 | 是 |
| `ProviderBadRequest`（其余 4xx） | 否 | 是 |
| `ProviderAuthFailed` | 否 | 否（fail-closed） |
| `ContentFiltered` | 否 | 否（fail-closed；可用 `trigger_errors` 打开） |

相对早期版本这是有意的行为变化：residual 4xx 以前会直接打死这一轮。
真 schema 错误仍会很快耗尽整条链；聚合后的
`ProviderUnavailable.errors` 会留下每一腿。

## 自定义触发条件

用 `trigger_errors` 覆盖默认集合：

```python
from cubeloop import FallbackBoundModel
from cubeloop.errors import ProviderAuthFailed, ProviderUnavailable, RateLimited

model = FallbackBoundModel(
    chain=(primary, fallback),
    trigger_errors=frozenset({RateLimited, ProviderUnavailable, ProviderAuthFailed}),
)
```

## 监控 failover

用 `on_failover` 接计费或告警：

```python
import logging

log = logging.getLogger(__name__)

async def record_failover(failed, next_model, error):
    log.warning(
        "provider failover: %s/%s → %s/%s (%s)",
        failed.spec.provider_id, failed.spec.id,
        next_model.spec.provider_id if next_model else "none",
        next_model.spec.id if next_model else "—",
        error,
    )

model = FallbackBoundModel(
    chain=(primary, fallback),
    max_retries_per_model=3,   # 首次失败后再重试 3 次（共 4 次）
    max_retry_after=5.0,
    on_failover=record_failover,
)
```

`on_failover` 的参数是 `(failed: BoundModel, next_model: BoundModel | None,
error: BaseException | str)`，**只在真正 hop 时**触发，同模型重试不会调它。
同模型重试用可选的 `on_retry(failed, error, attempt, wait_s)`。
同步 / 异步回调都可以；回调里抛错会被记日志并吞掉。

整条链都失败时抛 `ProviderUnavailable`，`.errors` 是每条腿的失败
（能分类则带类型），`__cause__` 是最后一条 typed error。

## `provider` 和 `spec` 始终指向主模型

`FallbackBoundModel.provider` / `.spec` 代理 `chain[0]`。读
`agent._model.provider` 或 `agent._model.spec` 的 tracing / 计费代码看到的
仍是用户选的主模型。真正应答的那条腿写在返回的
`AssistantMessage.provider_id` / `model_id` 上。

## 常见陷阱

- **不同 provider 的工具 schema** — 两个内置 provider 都吃同一套
  `ToolDefinition`，但厂商私货（如 OpenAI `parallel_tool_calls=False`）
  带不到 Anthropic。跨 provider 行为放在 `transform_context` middleware，
  不要塞 `extra_body`。
- **成本不同** — hop 会改单价。用 `AssistantMessage.provider_id` 记账；
  `on_failover` 是记录切换的合适位置。
- **流中途错误不重试** — 看到健康首包就提交该 provider，后半段错误原样转发。
- **`ContextLengthExceeded` 会跳过装不下的腿** — 已知 `tokens_in` 且下一腿
  `context_window` 更小时不会打它。标准窗口主模型请配大窗口备援。
- **Sticky 只作用于本轮，不是长期偏好** — 下一次 `Agent.prompt` 仍从用户选的
  主模型开始。除非产品层显式持久化，不要把 active index 存下来。

## 运行示例

仓库里有可运行版本：

```bash
git clone https://github.com/cubeplexai/cubeloop && cd cubeloop
uv sync

export ANTHROPIC_API_KEY=sk-ant-...   # 或 OPENAI_API_KEY [+ OPENAI_BASE_URL]
uv run python examples/multi_provider_failover.py
```

示例故意给主模型一个坏 key。鉴权默认 fail-closed，所以示例通过
`trigger_errors` **显式加入** `ProviderAuthFailed`，再由真实备援给出答案。

## 另请参见

- [Providers 总览](../guides/providers/overview) — provider 配置与 `CapabilityDescriptor`。
- [Providers / Anthropic](../guides/providers/anthropic) 和 [OpenAI](../guides/providers/openai) — 厂商细节。
- [编写自定义 Provider](../guides/providers/custom) — 内置不够用时。
