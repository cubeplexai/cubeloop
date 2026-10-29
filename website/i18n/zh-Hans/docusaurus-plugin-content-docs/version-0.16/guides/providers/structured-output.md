---
title: 结构化输出
description: "通过 generate_structured() 和 tool_choice 从 LLM 调用获得经过校验的 Pydantic 模型。"
---

# 结构化输出

`BoundModel.generate_structured()` 从一次 LLM 调用中提取强类型、经过
校验的数据。传入一个 Pydantic 模型类，拿回一个已校验的实例——不用
解析 JSON、也不用手写 schema。

```python
from pydantic import BaseModel
from cubeloop.providers.base import TextContent, UserMessage

class Sentiment(BaseModel):
    label: str
    confidence: float

result = await model.generate_structured(
    Sentiment,
    messages=[UserMessage(content=[TextContent(text="I love this product!")])],
)
print(result)  # label='positive' confidence=0.95
```

## 工作方式

底层 `generate_structured()` 做的事：

1. 把 Pydantic 模型的 JSON schema 转换成一个合成的 tool 定义。
2. 用 `tools=[synthetic_tool]` 和 `tool_choice=tool_name` 调用
   `generate()`，强制模型调用这个 tool。
3. 从响应里取出 `ToolCall.arguments`。
4. 通过 `output_type.model_validate()` 校验。

这和 pydantic-ai 默认使用的方式（`ToolOutput` 模式）一样。它对所有
provider 都能用，因为每个 LLM API 都支持 tool 调用。

## 参数

```python
await model.generate_structured(
    output_type,          # Pydantic 模型类（必填）
    messages,             # list[Message]（必填）
    *,
    system_prompt="",     # 可选的 system prompt（永远会追加一段使用 tool 的提示）
    tool_name="structured_output",
    tool_description="Return the structured output",
    max_output_tokens=None,
    temperature=None,
    max_retries=1,        # Pydantic 校验失败时的重试次数（把错误回灌给模型）
)
```

| 参数 | 默认 | 说明 |
|-----------|---------|-------------|
| `output_type` | 必填 | Pydantic `BaseModel` 子类 |
| `messages` | 必填 | 对话消息 |
| `system_prompt` | `""` | 自定义 system prompt（永远会追加一段使用 tool 的提示） |
| `tool_name` | `"structured_output"` | 合成 tool 的名称 |
| `tool_description` | `"Return the structured output"` | 发给模型的 tool 描述 |
| `max_retries` | `1` | Pydantic 校验失败时的重试次数 |
| `max_output_tokens` | `None` | 覆盖模型的默认 max tokens |
| `temperature` | `None` | 覆盖模型的默认 temperature |

## 错误处理

`generate_structured()` 在两种情况下抛 `StructuredOutputError`：

```python
from cubeloop.providers.base import StructuredOutputError

try:
    result = await model.generate_structured(MySchema, messages=[...])
except StructuredOutputError as e:
    print(e)  # "no tool call" 或 "validation failed after retries"
```

- **No tool call**：模型返回了文本而不是调用 tool。
- **Validation failed**：所有尝试都通不过 Pydantic 校验（首次 + 重试）。

校验失败时，错误会作为 `UserMessage` 回灌给模型，给它再一次机会
（最多 `max_retries` 次）。

## `tool_choice`

`generate_structured()` 内部用 `tool_choice` 强制模型调用合成 tool。
你也可以在 `stream()` 和 `generate()` 上直接用 `tool_choice`：

```python
reply = await model.generate(
    messages=[...],
    tools=[my_tool_def],
    tool_choice="my_tool",  # 强制使用这个 tool
)
```

可选值：

| 值 | 行为 |
|-------|----------|
| `None` | Provider 默认（模型自决） |
| `"auto"` | 模型自己决定是否调用 tool |
| `"required"` | 必须调用某个 tool |
| `"none"` | 不允许 tool 调用 |
| `"tool_name"` | 强制使用指定名称的 tool |

`tool_choice` 在所有内置 provider（Anthropic、OpenAI、OpenAI Responses）
上都能用。每个 provider 会把这个值映射到它原生的 wire 格式。
