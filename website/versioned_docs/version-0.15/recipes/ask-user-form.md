---
title: Multi-question Form via ask_user
description: "Build a multi-question form with CubeLoop's ask_user HITL tool for structured user input."
---

# Recipe: Multi-question Form via `ask_user`

Use case: the agent needs a structured answer from the user before it can
proceed — a configuration wizard, a preference selector, a feature toggle.

## Step 1: Register the tool

```python
from cubeloop.agent.agent import Agent
from cubeloop.hitl import InMemoryChannel, ask_user_tool

channel = InMemoryChannel()

agent = Agent(
    model=...,
    system_prompt=(
        "When you need the user to choose among options, use the ask_user tool. "
        "For free-form clarification questions, just end your turn with text — "
        "the user's next message will be your answer."
    ),
    tools=[ask_user_tool(channel)],
    channel=channel,
)
```

The `ask_user` tool is registered like any other tool. Its
`execution_mode="sequential"` makes the tool batch run one-by-one —
the HITL pause can't overlap with parallel tool execution.

## Step 2: Host renders the form

The model invokes `ask_user` with a list of question objects. The host
receives an `AskRequest` payload on the channel:

```python
async def host():
    async for req in channel.subscribe():
        if req.payload.kind == "ask":
            answers = {}
            for q in req.payload.questions:
                if q.options is None:
                    # Free-text question
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

## What the model sees as tool parameters

The model can pass questions that mix free-text, single-select, and
multi-select fields in a single call:

```json
{
  "questions": [
    {
      "key": "project_type",
      "prompt": "What kind of project?",
      "options": [
        {"label": "Web app", "value": "web"},
        {"label": "CLI tool", "value": "cli"},
        {"label": "Library", "value": "lib"}
      ]
    },
    {
      "key": "framework",
      "prompt": "Which framework?",
      "options": [
        {"label": "React", "value": "react"},
        {"label": "Vue", "value": "vue"},
        {"label": "Other", "value": "other", "allow_input": true}
      ]
    },
    {
      "key": "features",
      "prompt": "Which features do you need?",
      "multi_select": true,
      "options": [
        {"label": "Authentication", "value": "auth"},
        {"label": "Payments", "value": "payments"},
        {"label": "File uploads", "value": "uploads"}
      ]
    },
    {
      "key": "project_name",
      "prompt": "What should we call this project?"
    }
  ]
}
```

## Answer shape

The host answers with a dict mapping `key → value`:

```python
# Example answer for the above form:
{
    "project_type": "web",
    "framework": "svelte",       # user chose "Other" and typed "svelte"
    "features": ["auth", "uploads"],
    "project_name": "my-saas"
}
```

The answer is stuffed into the tool result as
`details["hitl"]["answers"]`. The model receives a human-readable summary
in the text content and can reference the dict for structured consumption.

## Cancel and timeout

If the host cancels via `channel.cancel(qid, reason)`:

```python
await channel.cancel(req.question_id, reason="user closed the form")
```

The tool surfaces an error result to the model:

```
tool_result.is_error = True
tool_result.details["hitl"]["outcome"] = "cancelled"
tool_result.details["hitl"]["reason"] = "user closed the form"
```

If the timeout expires:

```
tool_result.details["hitl"]["outcome"] = "timed_out"
tool_result.details["hitl"]["seconds"] = 30.0
```

In both cases the model sees a clean error result and can react
accordingly — ask again, fall back to a default, or report to the user.

## In-process example (full runnable snippet)

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

## Run the example

A self-contained, runnable version of this recipe is in the repository at
[`examples/ask_user_form.py`](https://github.com/cubeplexai/cubeloop/blob/main/examples/ask_user_form.py).
The host loop answers all questions programmatically so you can observe the
full round-trip without wiring up a real UI.

```bash
git clone https://github.com/cubeplexai/cubeloop && cd cubeloop
uv sync

export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY [+ OPENAI_BASE_URL]
uv run python examples/ask_user_form.py
```
