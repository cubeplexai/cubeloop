---
title: 通过 ask_user 实现多问题表单
description: "使用 CubeLoop 的 ask_user HITL 工具构建多问题表单以收集结构化用户输入。"
---

# 配方：通过 `ask_user` 实现多问题表单

适用场景：agent 在继续之前需要用户的**结构化答案** —— 配置向导、
偏好选择器、功能开关。

## 步骤 1：注册工具

```python
from cubeloop.agent.agent import Agent
from cubeloop.hitl import InMemoryChannel, ask_user_tool

channel = InMemoryChannel()

agent = Agent(
    model=...,
    system_prompt=(
        "当您需要用户在选项中选择时，请使用 ask_user 工具。"
        "对于自由形式的澄清问题，直接用文本结束当前轮次——"
        "用户的下一条消息就是答案。"
    ),
    tools=[ask_user_tool(channel)],
    channel=channel,
)
```

`ask_user` 工具像其他工具一样注册。它的
`execution_mode="sequential"` 使工具批次逐一执行——
HITL 暂停不能与并行工具执行重叠。

## 步骤 2：宿主渲染表单

模型调用 `ask_user` 并传入一个问题对象列表。宿主在 channel 上收到
一个 `AskRequest` payload：

```python
async def host():
    async for req in channel.subscribe():
        if req.payload.kind == "ask":
            answers = {}
            for q in req.payload.questions:
                if q.options is None:
                    # 自由文本问题
                    answers[q.key] = await my_ui.text_input(q.prompt)
                elif q.multi_select:
                    answers[q.key] = await my_ui.checkbox_group(
                        q.prompt, [(o.label, o.value) for o in q.options],
                    )
                else:
                    answers[q.key] = await my_ui.radio_group(
                        q.prompt,
                        [(o.label, o.value) for o in q.options],
                        allow_input_indexes=[
                            i for i, o in enumerate(q.options) if o.allow_input
                        ],
                    )
            await channel.answer(req.question_id, answers)
```

## 模型看到的工具参数

模型可以传入混合了自由文本、单选和多选字段的问题：

```json
{
  "questions": [
    {
      "key": "project_type",
      "prompt": "什么类型的项目？",
      "options": [
        {"label": "Web 应用", "value": "web"},
        {"label": "CLI 工具", "value": "cli"},
        {"label": "库", "value": "lib"}
      ]
    },
    {
      "key": "framework",
      "prompt": "哪个框架？",
      "options": [
        {"label": "React", "value": "react"},
        {"label": "Vue", "value": "vue"},
        {"label": "其他", "value": "other", "allow_input": true}
      ]
    },
    {
      "key": "features",
      "prompt": "你需要哪些功能？",
      "multi_select": true,
      "options": [
        {"label": "认证", "value": "auth"},
        {"label": "支付", "value": "payments"},
        {"label": "文件上传", "value": "uploads"}
      ]
    },
    {
      "key": "project_name",
      "prompt": "这个项目叫什么名字？"
    }
  ]
}
```

## 答案结构

宿主用一个 `key → value` 的 dict 回答：

```python
# 上述表单的答案示例：
{
    "project_type": "web",
    "framework": "svelte",       # 用户选择了"其他"并输入了"svelte"
    "features": ["auth", "uploads"],
    "project_name": "my-saas"
}
```

答案被填入工具结果的 `details["hitl"]["answers"]`。
模型可以通过文本内容看到人类可读的摘要，也可以通过 dict 进行结构化消费。

## 取消与超时

如果宿主通过 `channel.cancel(qid, reason)` 取消：

```python
await channel.cancel(req.question_id, reason="user closed the form")
```

工具会向模型显示一个错误结果：

```
tool_result.is_error = True
tool_result.details["hitl"]["outcome"] = "cancelled"
tool_result.details["hitl"]["reason"] = "user closed the form"
```

如果超时到期：

```
tool_result.details["hitl"]["outcome"] = "timed_out"
tool_result.details["hitl"]["seconds"] = 30.0
```

两种情况下模型都会看到干净的错误结果，并能做出相应反应——
再次提问、回退到默认值或向用户报告。

## 进程内示例（完整可运行代码段）

```python
import asyncio
from cubeloop.agent.agent import Agent
from cubeloop.hitl import InMemoryChannel, ask_user_tool

channel = InMemoryChannel()

agent = Agent(
    model=...,
    tools=[ask_user_tool(channel)],
    channel=channel,
)

async def host():
    async for req in channel.subscribe():
        if req.payload.kind == "ask":
            answers = {
                q.key: q.options[0].value if q.options else ""
                for q in req.payload.questions
            }
            await channel.answer(req.question_id, answers)

async def main():
    host_task = asyncio.create_task(host())
    try:
        await agent.prompt("Scaffold a new project.")
    finally:
        host_task.cancel()

asyncio.run(main())
```

## 运行示例

仓库中有一份完整可运行的代码，位于
[`examples/ask_user_form.py`](https://github.com/cubeplexai/cubeloop/blob/main/examples/ask_user_form.py)。
Host 循环以编程方式回答所有问题，无需接入真实 UI 即可观察完整的交互过程。

```bash
git clone https://github.com/cubeplexai/cubeloop && cd cubeloop
uv sync

export ANTHROPIC_API_KEY=sk-ant-...   # 或 OPENAI_API_KEY [+ OPENAI_BASE_URL]
uv run python examples/ask_user_form.py
```
