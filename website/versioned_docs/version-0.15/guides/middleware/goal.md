---
title: Goal
description: "Use GoalMiddleware to keep an agent working until an independent evaluator model confirms a completion condition is met."
---

# Goal

`GoalMiddleware` keeps an agent running autonomously until a completion
condition is met. A separate evaluator model (e.g. Haiku) judges
whether the condition has been achieved — the worker model is not grading
its own homework.

Inspired by Claude Code's `/goal` command: set a condition, let the agent
work, stop when a second model says the condition is satisfied.

## Basic setup

```python
from cubeloop import Agent
from cubeloop.providers.anthropic import AnthropicProvider
from cubeloop.middleware.goal import GoalMiddleware

provider = AnthropicProvider(api_key="...")

goal = GoalMiddleware(
    evaluator=provider.model("claude-haiku-4-5-20251001"),
    max_evaluations=10,
)

agent = Agent(
    model=provider.model("claude-sonnet-4-6"),
    middleware=[goal],
    tools=[...],
)

# /goal prefix activates goal mode
await agent.prompt("/goal all tests in tests/auth pass and ruff check is clean")

# Check outcome
print(agent.state.extra["goal"])
# {"status": "achieved", "condition": "all tests in ...", "evaluations": 2, ...}
```

## How it works

### Activation

GoalMiddleware activates when the user message starts with `/goal `.
Everything after the prefix is the completion condition.

```python
# Goal mode — evaluator will judge this condition
await agent.prompt("/goal make the homepage load in under 2 seconds")

# Normal mode — middleware is fully transparent
await agent.prompt("fix the bug in auth.py")
```

The `/goal` prefix is stripped before the worker sees the message. The
worker receives only the condition text as its work directive.

### Evaluation loop

After the worker finishes a complete run (all tool calls exhausted):

1. The evaluator reads the condition + last 20 messages from the conversation.
2. It returns `{achieved: bool, reason: str}` via structured output.
3. If **achieved** — the loop ends, status is `"achieved"`.
4. If **not achieved** and evaluations remain — feedback is injected
   (`"Goal not yet met: {reason}. Continue working."`) and the worker
   resumes.
5. If **max_evaluations** reached — the loop ends, status is `"exhausted"`.

```
Worker run → on_run_end → evaluator judges
                              ├─ achieved=True  → stop (status: "achieved")
                              ├─ achieved=False → inject feedback, continue
                              └─ max evals hit  → stop (status: "exhausted")
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `evaluator` | `BoundModel` | required | Model that judges the condition |
| `max_evaluations` | `int` | `10` | Safety cap on evaluator calls |

Use a fast, cheap model for the evaluator (e.g. Haiku). It only reads
the conversation transcript — it cannot call tools.

## Reading the outcome

After `agent.prompt()` returns, check `agent.state.extra["goal"]`:

```python
goal_state = agent.state.extra["goal"]

match goal_state["status"]:
    case "achieved":
        print(f"Done in {goal_state['evaluations']} evaluations")
        print(f"Reason: {goal_state['last_reason']}")
    case "exhausted":
        print(f"Gave up after {goal_state['max_evaluations']} evaluations")
        print(f"Last reason: {goal_state['last_reason']}")
```

Full state shape:

```python
{
    "status": "achieved" | "active" | "exhausted",
    "condition": "all tests pass...",
    "evaluations": 3,
    "max_evaluations": 10,
    "last_reason": "2 tests still failing in test_auth.py",
}
```

## Tracing

GoalMiddleware declares its evaluator via `extra_llm_calls()`, so the
tracing Recorder can subscribe to the evaluator's provider and attribute
evaluation spans correctly — they show up as evaluator calls, not worker
calls.

## Tips

- **Keep conditions specific and verifiable.** "All tests pass" is
  better than "code works well." The evaluator judges from transcript
  text, not by running tools.
- **Use a cheap evaluator.** Haiku is usually sufficient — the judgment
  is binary (achieved/not) with a short reason.
- **Set `max_evaluations` conservatively.** The default of 10 prevents
  runaway loops. Increase only when you expect the agent to need many
  iterations.
- **Combine with other middleware.** GoalMiddleware composes with
  `TodoListMiddleware`, compaction, etc. via standard middleware
  composition.
