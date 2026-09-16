---
title: Goal
description: "用 GoalMiddleware 让 agent 持续运行，直到一个独立的评估模型确认完成条件已满足。"
---

# Goal

`GoalMiddleware` 让 agent 自主运行直到完成条件满足为止。一个独立的
评估模型（例如 Haiku）来判断条件是否达成——不让工作模型自己给自己
打分。

灵感来自 Claude Code 的 `/goal` 命令：设定一个条件，让 agent 干活，
直到第二个模型说条件满足时停下。

## 基础用法

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

# /goal 前缀激活 goal 模式
await agent.prompt("/goal all tests in tests/auth pass and ruff check is clean")

# 查看结果
print(agent.state.extra["goal"])
# {"status": "achieved", "condition": "all tests in ...", "evaluations": 2, ...}
```

## 工作方式

### 激活

用户消息以 `/goal ` 开头时 GoalMiddleware 激活。前缀之后的全部内容是
完成条件。

```python
# Goal 模式——评估模型会判断这个条件
await agent.prompt("/goal make the homepage load in under 2 seconds")

# 普通模式——中间件完全透明
await agent.prompt("fix the bug in auth.py")
```

`/goal` 前缀在 worker 看到消息前被剥离。worker 只拿到条件文本作为
工作指令。

### 评估循环

每次 worker 跑完一次完整 run（所有 tool 调用都用尽）之后：

1. 评估模型读条件 + 对话的最后 20 条消息。
2. 通过 structured output 返回 `{achieved: bool, reason: str}`。
3. 如果 **achieved**——循环结束，状态是 `"achieved"`。
4. 如果 **not achieved** 且还有评估次数——把反馈
   （`"Goal not yet met: {reason}. Continue working."`）注入回去，
   worker 继续。
5. 如果到 **max_evaluations**——循环结束，状态是 `"exhausted"`。

```
Worker run → on_run_end → 评估模型判断
                              ├─ achieved=True  → 停（状态："achieved"）
                              ├─ achieved=False → 注入反馈，继续
                              └─ 达到最大评估次数 → 停（状态："exhausted"）
```

## 参数

| 参数 | 类型 | 默认 | 说明 |
|-----------|------|---------|-------------|
| `evaluator` | `BoundModel` | 必填 | 判断条件的模型 |
| `max_evaluations` | `int` | `10` | 评估调用次数的安全上限 |

evaluator 建议用便宜快速的模型（如 Haiku）。它只读对话脚本——不能
调工具。

## 读取结果

`agent.prompt()` 返回后，看 `agent.state.extra["goal"]`：

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

完整的状态结构：

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

GoalMiddleware 通过 `extra_llm_calls()` 声明自己的 evaluator，所以
tracing 的 Recorder 能订阅 evaluator 的 provider 并正确归属评估 span
——它们会作为 evaluator 的调用出现，而不是 worker 的调用。

## 小贴士

- **条件要具体、可验证。** "All tests pass" 比 "code works well" 好得多。
  evaluator 是从对话脚本判断的，不是真的去跑工具。
- **用便宜的 evaluator。** Haiku 通常足够——判断是二选一（达成与否）
  加一句简短理由。
- **`max_evaluations` 设保守点。** 默认 10 防止失控循环。只有当你确实
  需要多次迭代时才提高。
- **可以和其他中间件叠加。** GoalMiddleware 通过标准中间件组合规则
  与 `TodoListMiddleware`、compaction 等组合。
