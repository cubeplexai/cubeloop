---
title: 概览
sidebar_position: 1
description: "CubeLoop 人机协同：channel、三种动词（confirm、approve、ask）与超时。"
---

# 人机协同 (HITL)

CubeLoop 的 HITL channel 让 agent 能够**暂停并等待人类输入**后再继续。
它用一个统一原语覆盖两种典型模式：

1. **沙箱工具确认** —— 危险工具（bash、文件写入、API 变更）在运行前
   需要人类 approve / deny / edit。
2. **运行中结构化提问** —— agent 需要用户做出选择或填写表单才能继续。

Channel 是一个可 `await` 的协程协作者。工具作者写
`await channel.ask(...)`，channel 处理暂停。宿主代码订阅挂起请求并回填
答案。两种后端覆盖全场景：

- `InMemoryChannel` —— CLI、notebook、测试。进程死亡，挂起丢失。
- `CheckpointedChannel` —— web 服务。将挂起请求持久化到 `Checkpointer`，
  让不同进程（或重启后的同一进程）在数小时后拾起并回答。

## 三种动词

### `confirm(prompt, *, details, timeout, signal) → bool`

简单的 yes/no 问题。宿主回答 `True` 或 `False`。

### `approve(tool_name, tool_call_id, args, *, details, timeout, signal) → ApproveAnswer`

沙箱确认动词。返回一个 `ApproveAnswer`，三种决策：

| 决策 | 结果 |
|---|---|
| `"approve"` | 以原始参数运行工具 |
| `"deny"` | 阻塞工具；`tool_result.is_error=True`，`details["hitl"]["decision"]="human_deny"` |
| `"edit"` | 以编辑后的参数运行（会拿工具的 pydantic 参数模型重新校验） |

对 `approve` 请求，信封的 `question_id` 设为 LLM 的 `tool_call_id` —
没有独立的 UUID，所以宿主代码可以直接用它已经在工具流中追踪的同一个
ID 来关联。

### `ask(questions, *, timeout, signal) → dict[str, str | list[str]]`

一个包含一个或多个 `Question` 对象的结构化表单。每个问题可以是：

- **自由文本** (`options=None`)
- **单选** (`options=[...]`, `multi_select=False`)
- **多选** (`options=[...]`, `multi_select=True`)
- **"其他" 可输入**（选项有 `allow_input=True` —— 用户输入自由文本）

```python
from cubeloop.hitl.types import Question, Option

answers = await channel.ask([
    Question(key="framework", prompt="选择框架？", options=[
        Option(label="React", value="react"),
        Option(label="Vue", value="vue"),
        Option(label="其他", value="other", allow_input=True),
    ]),
    Question(key="features", prompt="启用功能：", multi_select=True, options=[
        Option(label="认证", value="auth"),
        Option(label="支付", value="payments"),
    ]),
])
# answers == {"framework": "react", "features": ["auth", "payments"]}
```

## 超时

两个 channel 都在构造函数接受 `default_timeout`，每个动词接受一個
per-call 的 `timeout` kwarg（per-call 覆盖默认）。

超时到期从 agent 侧的 `await` 抛出 `HitlTimedOut(BaseException)`。
周围的工具或中间件将其转换为 `tool_result.is_error=True`，
`details["hitl"]["decision"]="timed_out"`，模型看到干净的拒绝结果
并能自然反应。信封的 `HitlRequest.timeout_seconds` 会自动填写，
前端可以渲染倒计时。
