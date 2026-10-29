---
title: Structured Output
description: "Get validated Pydantic models from LLM calls with generate_structured() and tool_choice."
---

# Structured Output

`BoundModel.generate_structured()` extracts typed, validated data from
an LLM call. Pass a Pydantic model class and get a validated instance
back — no JSON parsing, no schema wrangling.

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

## How it works

Under the hood, `generate_structured()`:

1. Converts the Pydantic model's JSON schema into a synthetic tool definition.
2. Calls `generate()` with `tools=[synthetic_tool]` and
   `tool_choice=tool_name` to force the model to call that tool.
3. Extracts the `ToolCall.arguments` from the response.
4. Validates through `output_type.model_validate()`.

This is the same approach pydantic-ai uses by default (`ToolOutput` mode).
It works across all providers because every LLM API supports tool calling.

## Parameters

```python
await model.generate_structured(
    output_type,          # Pydantic model class (required)
    messages,             # list[Message] (required)
    *,
    system_prompt="",     # Optional system prompt (a tool-use hint is always appended)
    tool_name="structured_output",
    tool_description="Return the structured output",
    max_output_tokens=None,
    temperature=None,
    max_retries=1,        # Retries on validation failure (feeds error back to model)
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `output_type` | required | Pydantic `BaseModel` subclass |
| `messages` | required | Conversation messages |
| `system_prompt` | `""` | Custom system prompt (tool-use hint always appended) |
| `tool_name` | `"structured_output"` | Name of the synthetic tool |
| `tool_description` | `"Return the structured output"` | Tool description sent to model |
| `max_retries` | `1` | Retries on Pydantic validation failure |
| `max_output_tokens` | `None` | Override model's default max tokens |
| `temperature` | `None` | Override model's default temperature |

## Error handling

`generate_structured()` raises `StructuredOutputError` in two cases:

```python
from cubeloop.providers.base import StructuredOutputError

try:
    result = await model.generate_structured(MySchema, messages=[...])
except StructuredOutputError as e:
    print(e)  # "no tool call" or "validation failed after retries"
```

- **No tool call**: the model returned text instead of calling the tool.
- **Validation failed**: Pydantic validation failed on all attempts
  (initial + retries).

On validation failure, the error is fed back to the model as a
`UserMessage` and it gets another chance (up to `max_retries` times).

## `tool_choice`

`generate_structured()` uses `tool_choice` internally to force the model
to call the synthetic tool. You can also use `tool_choice` directly on
`stream()` and `generate()`:

```python
reply = await model.generate(
    messages=[...],
    tools=[my_tool_def],
    tool_choice="my_tool",  # Force this specific tool
)
```

Values:

| Value | Behavior |
|-------|----------|
| `None` | Provider default (model decides) |
| `"auto"` | Model decides whether to call a tool |
| `"required"` | Must call some tool |
| `"none"` | No tool calls allowed |
| `"tool_name"` | Force a specific tool by name |

`tool_choice` works on all built-in providers (Anthropic, OpenAI, OpenAI
Responses). Each provider maps the value to its native wire format.
