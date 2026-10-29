---
title: 子 Agent
description: "使用 SubagentMiddleware 把自包含任务委派给 CubeLoop 子 Agent。"
---

# 子 Agent

`SubagentMiddleware` 会添加一个 `subagent` 工具。模型调用该工具时，CubeLoop
创建一个临时子 `Agent`，运行一段自包含 prompt，并把子 agent 最终 assistant
文本作为工具结果返回。

适合父 agent 需要委派有边界的工作：研究、审查、抽取、聚焦实现等。

## 定义子 agent

```python
from cubeloop import Agent
from cubeloop.middleware import SubagentMiddleware, SubagentSpec

subagents = {
    "researcher": SubagentSpec(
        name="researcher",
        description="Researches a narrow question and returns concise notes.",
        system_prompt="You are a precise research assistant.",
    ),
    "reviewer": SubagentSpec(
        name="reviewer",
        description="Reviews code for bugs, regressions, and missing tests.",
        system_prompt="You are a rigorous senior code reviewer.",
    ),
}

model = provider.model("claude-sonnet-4-6")

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

如果模型请求未知 `subagent_type`，CubeLoop 回退到 `general-purpose` 子 agent。
如果你没有定义它，middleware 会提供一个基础默认版本。

## 工具访问和 middleware 继承

子 agent 只获得你传入的工具：

- `shared_tools` 对所有子 agent 可用。
- `SubagentSpec.tools` 只对该类型子 agent 可用。
- `excluded_tool_names` 防止递归或 host 专用工具被共享。

用 `inherited_middleware` 传入所有子 agent 都应运行的 middleware。用
`SubagentSpec.middleware` 定义某个子 agent 类型专属行为。

## 把子事件流式传给 host

应用通常需要把子事件写入自己的 UI 或审计日志。传入 `event_mapper` 和可选
`event_handler`：

```python
def map_event(event):
    if event.type == "text_delta":
        return {"type": "subagent_text_delta", "delta": event.delta}
    return None

async def handle_event(agent_id, payload):
    await ui_stream.send({"agent_id": agent_id, **payload})

SubagentMiddleware(
    subagents=subagents,
    default_model=model,
    event_mapper=map_event,
    event_handler=handle_event,
)
```

映射后的 payload 也会存入父工具结果的 `details["subagent_events"]`。

## Tracing 和 abort

通过 `tracer=...` 传入 `Tracer`，即可给每个子运行附加 tracing。嵌套子
agent span 共享父 trace，所以 `cubeloop trace view <trace_id>` 会把父工具调用
和子运行一起渲染出来。

父运行的 abort signal 会转发给子 agent。如果父运行在子 agent 执行中被
abort，子 agent 也会被 abort。

## Prompt 约定

父模型在 `subagent` 工具调用里应发送自包含的 `prompt`。不要假设子 agent 能
看到父 agent 的完整隐藏上下文。prompt 里应包含目标、相关事实、约束和期望
输出格式。
