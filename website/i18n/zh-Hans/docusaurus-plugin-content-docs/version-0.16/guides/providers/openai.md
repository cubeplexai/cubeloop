---
title: OpenAI
description: "在 CubeLoop 中使用 OpenAI GPT 模型——OpenAIProvider 与 Chat Completions API 集成。"
---

# OpenAI Provider

CubeLoop 内置两个 OpenAI provider，分别覆盖两种 API 接口：

- **`OpenAIProvider`** —— Chat Completions API（`/v1/chat/completions`）。适用于 GPT-4/5 系列以及大多数 OpenAI 兼容服务器（vLLM、LiteLLM、DeepSeek、Qwen、MiniMax、豆包……）。
- **`OpenAIResponsesProvider`** —— Responses API（`/v1/responses`）。适用于需要服务端状态和推理摘要的场景。

两者实现相同的 `Provider` 协议；每个 agent 选择其一。

## Chat Completions：`OpenAIProvider`

```python
from cubeloop.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    provider_id="openai",
    api_key="sk-…",      # or reads OPENAI_API_KEY
    base_url=None,        # set for OpenAI-compatible servers
    extra_body=None,      # merged into every request
    extra_headers=None,
)

model = provider.model(
    "gpt-5",
    reasoning=True,        # enables thinking level mapping
    max_tokens=8192,
    context_window=128_000,
)
```

### Chat Completions 中的推理

OpenAI 通过 o 系列和 gpt-5 模型的 `delta.reasoning_content` 暴露推理内容。CubeLoop 将其捕获为 `ThinkingContent`，并以与 Anthropic 完全相同的方式发出 `thinking_*` 事件。相同的 `ThinkingLevel` 枚举（`"off"` → `"high"`）同样适用。

许多 OpenAI 兼容的开源后端在不同字段下暴露推理内容。CubeLoop 按优先级顺序识别以下三种：

1. `delta.reasoning_content`（DeepSeek、Qwen、豆包）
2. `delta.reasoning`（vLLM）
3. `delta.reasoning_details`（MiniMax）

无需任何配置——provider 会自动选择存在的字段。

### `extra_body` 处理开源后端的差异

大多数 OpenAI 兼容服务器通过请求体接受扩展字段。在构造时一次性设置：

```python
provider = OpenAIProvider(
    api_key="…",
    base_url="https://api.deepseek.com/v1",
    extra_body={"enable_thinking": True, "stream_options": {"include_usage": True}},
)
```

如需按请求修改，请使用 `on_payload`（见下文）。

### 能力描述符

OpenAI 与 OpenAI 兼容后端之间的 wire 格式差异（如 `max_tokens` vs `max_completion_tokens`、推理字段名称、temperature 处理方式）可通过构造时传入的 [`CapabilityDescriptor`](pathname:///pydoc/cubeloop/providers/capability.html) 来配置。例如，`max_tokens_field="max_completion_tokens"` 会在请求发出前重命名该 key。详见 [能力描述符](./overview) 了解完整配置项（CubeLoop `0.5+`）。

### 指向 vLLM / LiteLLM / DeepSeek

```python
provider = OpenAIProvider(
    api_key="dummy",                                    # vLLM ignores it
    base_url="http://localhost:8000/v1",
    extra_headers={"Authorization": "Bearer dummy"},
)
```

LiteLLM 示例：

```python
provider = OpenAIProvider(
    api_key=os.environ["LITELLM_KEY"],
    base_url="https://litellm.internal/v1",
)
```

关于这些后端的 wire 差异（推理开关、token 字段重命名……），参见 [能力描述符](./overview)。

## Responses API：`OpenAIResponsesProvider`

```python
from cubeloop.providers.openai_responses import OpenAIResponsesProvider

provider = OpenAIResponsesProvider(provider_id="openai_responses", api_key="sk-…")
model = provider.model("gpt-5", reasoning=True)
```

Responses API 在服务端维护状态（通过 `previous_response_id` 引用）。CubeLoop 追踪 `AssistantMessage.response_id` 并自动回传——你的代码看起来与 Chat Completions 路径完全相同。

在以下情况下使用 Responses provider：

- 需要推理**摘要**（而不只是文本）作为 thinking 块暴露。
- 使用 `o` 系列模型，并希望服务端跨轮次保持推理链（更小的 payload，更快的复用）。

如果需要完全控制消息列表和提示缓存策略，请继续使用 `OpenAIProvider`。

## `on_payload` / `on_response`

形状与 [Anthropic](./anthropic) provider 相同。payload 字典有所不同（OpenAI 风格中 `system` 不单独存放，`tools` schema 格式也不同），因此在修改前先检查一次。

```python
async def add_user_metadata(payload, model):
    payload["user"] = "u-42"     # billable user attribution
    return payload

agent = Agent(model=model, on_payload=add_user_metadata)
```

## 工具调用

工具定义会自动转换为 OpenAI 的 `{"type": "function", "function": {...}}` 格式。流式格式在 `toolcall_delta` 下发出增量 JSON 参数；CubeLoop 通过 [`cubeloop.utils.json_parse.parse_streaming_json`](../../api/cubeloop-utils) 对其进行缓冲和解析，确保部分内容始终能校验为最接近的合法对象。

一条 assistant 消息中的多个并行工具调用开箱即用——它们与 Anthropic provider 使用相同的并行执行器进行路由。

## 常见问题

- **`stream_options.include_usage` 被拒绝** —— 部分兼容服务器会拒绝整个 `stream_options` 字段。**`on_payload` 无法修复此问题**：CubeLoop 0.3 在你的回调执行**之后**才调用 `kwargs.setdefault("stream_options", {})`，因此在 `on_payload` 中删除该 key 会被静默地撤销。解决方案：
  - 子类化 `OpenAIProvider` 并覆盖 `stream()`，对你的后端跳过 `setdefault`。
  - 在 `on_payload` 中设置 `include_usage=False`（字段仍会发出，但通常被严格后端接受为无操作）。
  - 使用 [`CapabilityDescriptor`](./overview)（CubeLoop `0.5+`）以声明式方式描述你的后端推理配置。
- **有推理内容但没有 `thinking_*` 事件** —— 你的后端在非标准字段下暴露推理内容。可通过 PR 添加第四个分支，或使用 `on_payload` 进行转码。
- **同一进程中混用多个 provider** —— 每个 provider 持有自己的 HTTP 客户端。按 `(base_url, api_key)` 对复用单个实例，而非每个 agent 各创建一个。
- **usage 显示 0 个输入 token** —— 大多数兼容服务器完全省略 usage，或只在最后一个 chunk 中发出。可在 `on_payload` 中检查尾部 chunk 获取提示，或将 token 计数视为这些后端的尽力而为值。

## 参见
- [图片生成](./image-generation) —— 使用 `openai-images` 与 OpenAI 图片模型。

- [Anthropic Provider](./anthropic) —— 另一个内置 provider。
- [自定义 Provider](./custom) —— 从头编写你自己的 provider。
- [Recipes → 多 Provider 故障转移](../../recipes/multi-provider-failover) —— 结合两个 provider 提升弹性。
