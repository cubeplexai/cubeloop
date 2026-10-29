---
title: 工具结果长度限制
description: "用 ToolResultLimitMiddleware 在 ToolExecutionEndEvent 发出前截断过长的工具结果文本。"
---

# 工具结果长度限制

`ToolResultLimitMiddleware` 在 `after_tool_call` 里截断过长的工具结果文本，发生在 CubeLoop 发布 `ToolExecutionEndEvent` 之前。模型、checkpointer 和宿主的事件消费者看到的都是截断后的结果。

工具可能返回不受控的输出时使用它，比如 shell、网页抓取、MCP 工具。过大的结果会撑破上下文窗口，或者超出宿主的事件大小限制。

## 基本设置

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

默认上限是 20,000 个字符的文本内容。不超过上限的结果原样通过（`after_tool_call` 返回 `None`）。超过上限时，文本保留前 `max_chars` 个字符，并附上提示：

```text
[truncated: tool result was 1082017 characters; showing first 20000. Narrow the tool call and retry.]
```

图片块会保留。只有 `TextContent` 计入上限。

## 选项

```python
ToolResultLimitMiddleware(
    max_chars=20_000,
    exclude_tool_names={"load_skill"},
)
```

- `max_chars`：从原始结果保留的最大文本字符数，必须 `>= 1`。
- `exclude_tool_names`：跳过截断的工具名。完整内容本身就是负载时使用，例如 `load_skill` 返回的 skill 正文。

## 放置顺序

把它放在其他 `after_tool_call` middleware **之后**，让它们先改写完整结果，再由截断后的文本进入 `ToolExecutionEndEvent`。

CubeLoop 默认的组合规则是最后一个非 `None` 返回值生效。排在后面的 middleware 如果返回了不含 `content` 的 `AfterToolCallResult`，这次改写会被丢掉。把 `ToolResultLimitMiddleware` 放在最后，或者换一个会合并 `content` 和 `details` 的 composer。

子 agent 不会继承父 agent 的 middleware。如果子工具的结果也要限制，通过 `SubagentMiddleware(inherited_middleware=...)` 传入同一个实例或再创建一个。
