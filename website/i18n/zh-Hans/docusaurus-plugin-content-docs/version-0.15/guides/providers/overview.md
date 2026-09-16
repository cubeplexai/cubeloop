---
title: Providers Overview
description: "CubeLoop 的 provider 配置、能力和预设总览。"
---

# Providers Overview

_从这里开始看 provider 配置。_

本页是 provider 配置的入口。先讲 Anthropic / OpenAI 的默认路径，
再讲当你需要接入别的后端时，如何用数据的方式描述 wire 差异和
预设覆盖，而不是写各家厂商 glue code。整体顺序是：

1. **默认零配置。** 对于 Anthropic 和 OpenAI，只需创建 provider 和 model，本页的内容一概不需要。
2. **对于非默认端点，以数据方式描述差异。** `CapabilityDescriptor` 以声明式方式捕获差异——无需子类化，无需 fork。

## `provider.model(...)` 参数

用 `provider.model(model_id, ...)` 为 Agent 创建绑定模型。`model_id` 为必填位置参数，其他参数均为可选关键字参数：

- `api: str` — 覆盖接口名/路由标签，供后续集成读取。
- `reasoning: bool` — 启用推理模式与 reasoning-level 协商。
- `context_window: int` — 上下文窗口提示（主要用于校验与提示词规划）。
- `max_tokens: int` — 该模型默认最大生成 token 上限。
- `temperature: float` — 该模型默认采样温度。
- `cost: ModelCost | None` — 可选的成本元数据对象。
- `thinking_level_map: dict[str, str | None] | None` — 推理等级映射；将某级别设为 `None` 表示该 level 不可用。

## CapabilityDescriptor：当后端模型线不同时该怎么配

`CapabilityDescriptor` 在初始化 provider 时传入，用于统一描述这个 provider
后端在 wire 上的差异。

```python
from cubeloop import CapabilityDescriptor
from cubeloop.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    api_key="...",
    base_url="https://api.deepseek.com",
    capability=CapabilityDescriptor(
        reasoning_on_payload={"extra_body": {"thinking": True}},
        reasoning_off_payload={"extra_body": {"thinking": False}},
        max_tokens_field="max_completion_tokens",
    ),
)
```

如果只有部分模型例外，使用 `model_capability_overrides`：

```python
from cubeloop import CapabilityDescriptor
from cubeloop.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    api_key="...",
    base_url="https://openrouter.ai/api/v1",
    capability=CapabilityDescriptor(
        reasoning_on_payload={"extra_body": {"thinking": True}},
    ),
    model_capability_overrides={
        "deepseek-r1": CapabilityDescriptor(
            reasoning_on_payload={"extra_body": {"thinking": "enabled"}},
        ),
    },
)
```

`model_capability_overrides` 使用精确 `model_id` 匹配。

`CapabilityDescriptor` 常用字段：

- `reasoning_on_payload / reasoning_off_payload` — 在 reasoning 开/关时，深度
  合并到最终 payload。
- `reasoning_level`（`ReasoningLevelSpec`）— 将 `off`/`low`/... 映射到后端字段。
- `temperature`（`TemperatureSpec`）— 裁剪、固定或去掉温度参数。
- `max_tokens_field` — 选 `max_tokens` 或 `max_completion_tokens`。
- `supports_tools` / `supports_images` / `supports_streaming` — 供宿主应用或前端消费的能力元数据。

:::note 预设目录由宿主应用维护
CubeLoop 提供的是**机制**（`CapabilityDescriptor` 及其 wire 运行时），而非厂商目录。包含 base URL、认证、区域/代码规划端点、模型列表的现成 provider 列表属于产品数据，应由嵌入 CubeLoop 的应用自行维护（例如，cubebox 维护自己的 provider 目录）。如需接入特定厂商，请按下文所示，使用正确的 `base_url` + `CapabilityDescriptor` 构建 provider。
:::

## 1. 简单场景——完全零配置

大多数用户无需关心能力描述符。内置 provider 已有合理的默认值：

```python
import cubeloop
from cubeloop import Agent
from cubeloop.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(provider_id="anthropic")  # reads ANTHROPIC_API_KEY
agent = Agent(model=provider.model("claude-sonnet-4-6"))
await agent.prompt("Hello!")
```

这就是全部配置。不带 `capability=` 构建的 provider，其输出与 CubeLoop `0.4` 完全一致——下文的机制仅在显式使用时才会生效。

## 2. 非默认端点——CapabilityDescriptor

当你需要使用 OpenAI 或 Anthropic 之外的模型——DeepSeek、Qwen、豆包、OpenRouter 路由、本地服务器——麻烦在于每家的 wire 方言不同（是用 `max_tokens` 还是 `max_completion_tokens`？如何开关推理？）。你不需要子类化 provider；只需将差异描述为一个 [`CapabilityDescriptor`](pathname:///pydoc/cubeloop/providers/capability.html)，连同正确的 `base_url` 和对应端点 wire 形状的 provider 类一起传入：

```python
import os
from cubeloop import CapabilityDescriptor
from cubeloop.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    capability=CapabilityDescriptor(
        reasoning_off_payload={"extra_body": {"reasoning": {"exclude": True}}},
        reasoning_on_payload={"extra_body": {"reasoning": {"exclude": False}}},
    ),
)
```

根据端点的 wire 方言选择 provider 类：

| Wire 方言 | Provider 类 |
| --- | --- |
| `anthropic-messages` | `AnthropicProvider` |
| `openai-completions` | `OpenAIProvider` |
| `openai-responses` | `OpenAIResponsesProvider` |

每个字段对应一种 wire 行为，未设置的字段不产生任何影响——因此只需声明实际存在差异的部分。

### `max_tokens_field`

`"max_tokens"`（默认）或 `"max_completion_tokens"`。部分 OpenAI 兼容服务器只接受其中一种拼写；此字段在请求发出前重命名该 key。**效果：** 选错会导致服务器忽略输出长度限制或返回 400。

### `temperature`

`TemperatureSpec` 控制调用方传入的 temperature 如何处理：

```python
from cubeloop import TemperatureSpec

TemperatureSpec(mode="free", min=0.0, max=2.0, default=1.0)  # clamp into [min, max]
TemperatureSpec(mode="fixed", fixed_value=1.0)               # always overwrite
TemperatureSpec(mode="ignored")                              # drop the key
```

- **`free`** —— 调用方的值被截断到 `[min, max]` 范围内；若未传入则不写入任何值。**效果：** 防止超出范围的值被后端拒绝。
- **`fixed`** —— 始终使用 `fixed_value`。**效果：** 适用于只允许固定 temperature 的模型（如部分 o 系列推理模型）。
- **`ignored`** —— 完全去除该 key。**效果：** 适用于在推理时对任何 `temperature` 字段返回 400 的后端。

### 推理开关：`reasoning_off_payload` / `reasoning_on_payload`

关闭推理时，将 `reasoning_off_payload` 深度合并到请求中；开启时合并 `reasoning_on_payload`。**效果：** 这就是"开关推理"如何映射到厂商期望的字段：

```python
CapabilityDescriptor(
    reasoning_off_payload={"extra_body": {"enable_thinking": False}},
    reasoning_on_payload={"extra_body": {"enable_thinking": True}},
)
```

合并会递归处理嵌套 dict；数组是原子的；冲突时 capability 值优先。

### 推理级别：`reasoning_level`（三种形状）

在开/关之外，CubeLoop 将 `ThinkingLevel`（`off`/`low`/`medium`/`high`/`xhigh`）映射到通过点路径 `path` 写入的具体 wire 值。`kind` 决定形状：

`ReasoningLevelSpec` 只负责「`thinking`/`low`/... 具体映射成后端字段」；要真正生效，还要配两个参数：

- 在 `provider.model(...)` 时把 `reasoning=True`（把这个模型设为推理模型）
- 在 `Agent(...)` 初始化时把 `thinking` 设成 `off|low|medium|high|xhigh`（默认 `off`）

```python
from cubeloop import Agent, CapabilityDescriptor, ReasoningLevelSpec
from cubeloop.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    api_key="...",
    capability=CapabilityDescriptor(
        reasoning_on_payload={"extra_body": {"reasoning": {"enabled": True}}},
        reasoning_level=ReasoningLevelSpec(
            path="reasoning.effort",
            kind="effort",
            level_to_effort={
                "off": "low",
                "low": "low",
                "medium": "medium",
                "high": "high",
                "xhigh": "high",
            },
        ),
    ),
)

agent = Agent(model=provider.model("deepseek-r1", reasoning=True), thinking="high")
```

```python
from cubeloop import ReasoningLevelSpec

# int_budget — a token budget (Anthropic).
ReasoningLevelSpec(
    path="thinking.budget_tokens", kind="int_budget",
    level_budgets={"off": 0, "low": 2048,
                   "medium": 8192, "high": 16384, "xhigh": 16384},
)

# effort — an effort string (OpenAI Responses).
ReasoningLevelSpec(
    path="reasoning.effort", kind="effort",
    level_to_effort={"low": "low", "medium": "medium",
                     "high": "high", "xhigh": "high"},
)

# enum — a vendor-specific state (Doubao's 3-state thinking).
ReasoningLevelSpec(
    path="thinking.type", kind="enum",
    level_to_enum={"off": "disabled", "low": "enabled", "high": "enabled"},
)
```

**效果：** 映射表中缺失的级别不会被写入，端点将保持该级别的自身默认值。

### `supports_tools` / `supports_images` / `supports_streaming`

声明式标志，由宿主应用和前端读取（例如，用于置灰图片上传按钮）。provider 本身不依赖这些标志做行为控制。

### 共享端点上的按模型覆盖

一个网关（OpenRouter、LiteLLM、内部代理）通常同时服务推理型和非推理型模型。`model_capability_overrides` 将 `model_id` 映射到一个描述符，该描述符会**替换**该模型的基础描述符：

```python
provider = OpenAIProvider(
    api_key="…",
    base_url="https://openrouter.ai/api/v1",
    capability=base_cap,                        # default for unlisted models
    model_capability_overrides={
        "deepseek/deepseek-r1": reasoning_cap,  # this model only
    },
)
```

解析方式为对 `model_id` 精确匹配；未列出的模型回退到 `capability`。

## 图片生成 provider

图片生成有独立的 provider 表面（`cubeloop.providers.images`），范式与上文
描述完全一致：provider 上的 `provider_id`、`provider.model("id", ...)`
工厂、类型化的 `ProviderError` 错误，以及处理后端字段差异的 capability
descriptor。完整指南见 [图片生成](./image-generation)。

## 参见

- [OpenAI Provider](./openai) —— OpenAI / OpenAI 兼容端点的具体配置。
- [Anthropic Provider](./anthropic) —— `int_budget` 推理形状的实际用法。
- [自定义 Provider](./custom) —— 当端点甚至不是 OpenAI/Anthropic 形状时。
- [API 参考 → `cubeloop.providers`](../../api/cubeloop-providers)。
