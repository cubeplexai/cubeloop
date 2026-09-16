---
title: 使用 ApprovalPolicyMiddleware 做沙箱确认
description: "使用 CubeLoop 的 ApprovalPolicyMiddleware 实现沙箱工具确认——自动放行、拒绝或人工确认。"
---

# 配方：使用 `ApprovalPolicyMiddleware` 做沙箱确认

适用场景：一个 web 服务中每个工具调用都经过规则引擎，被分类为自动放行、
硬阻止或人类确认。

## 步骤 1：定义策略函数

策略接收一个 `BeforeToolCallContext` 并返回一个 `ApprovalDecision` ——
`Approve()`、`Deny(reason)` 或 `AskUser(...)`。

```python
from cubeloop.hitl import Approve, AskUser, Deny

# 模拟规则引擎 —— 替换为你实际的策略目录。
def classify_command(cmd: str) -> tuple[str, str | None]:
    """(tier, reason) — "allow", "block", 或 "confirm"."""
    if cmd.startswith(("ls", "cat", "head", "grep", "find")):
        return ("allow", None)
    if "rm -rf /" in cmd or cmd.startswith("dd"):
        return ("block", "destructive I/O")
    return ("confirm", "needs human review")


def sandbox_policy(ctx):
    cmd = ctx.args.cmd  # ctx.args 是校验后的 pydantic 模型
    tier, reason = classify_command(cmd)

    if tier == "allow":
        return Approve()
    if tier == "block":
        return Deny(reason=reason or "blocked by policy")
    return AskUser(
        timeout_seconds=180,
        details={"matched_rule": tier, "impact": reason or "unknown"},
    )
```

`ctx.args` 是**校验后的 pydantic 模型**。通过属性访问字段
（`ctx.args.cmd`）。中间件内部会将其转为 dict 用于 channel 的 approve payload，
但你的策略收到的仍然是类型化的模型。

## 步骤 2：接入 agent

```python
from cubeloop.agent.agent import Agent
from cubeloop.checkpointer.postgres import PostgresCheckpointer
from cubeloop.hitl import ApprovalPolicyMiddleware, CheckpointedChannel

async def main():
    async with PostgresCheckpointer("postgresql://...") as cp:
        channel = CheckpointedChannel(checkpointer=cp, thread_id="session-1")

        agent = Agent(
            model=anthropic.model("claude-sonnet-4-6"),
            system_prompt="You are a helpful assistant with access to a bash shell.",
            tools=[bash_tool],
            middleware=[
                ApprovalPolicyMiddleware(channel, policy=sandbox_policy),
            ],
            channel=channel,
            checkpointer=cp,
            thread_id="session-1",
        )

        await agent.prompt("list files then delete temp logs")
        # Agent 运行中；当 bash 被调用时，sandbox_policy 决定：
        #   ls → Approve() → 立即执行
        #   rm /tmp/logs → AskUser() → channel 挂起，HitlRequestEvent 触发
```

## 步骤 3：宿主处理挂起的请求

```python
async def host_loop(channel: CheckpointedChannel):
    async for req in channel.subscribe():
        if req.payload.kind == "approve":
            tool_name = req.payload.tool_name
            command = req.payload.args.get("cmd", "")
            details = req.payload.details or {}
            timeout = req.timeout_seconds  # 前端倒计时用的秒数

            # 渲染给前端：tool_name、command、details["matched_rule"]、
            # details["impact"]、以及基于 timeout 的倒计时。
            human_answer = await my_frontend.show_confirm(
                tool_name=tool_name,
                command=command,
                details=details,
                timeout=timeout,
            )
            # 根据人类的决定构建 ApproveAnswer。
            from cubeloop.hitl import ApproveAnswer
            human_answer = ApproveAnswer(
                decision=ui_response["decision"],          # "approve" | "deny" | "edit"
                reason=ui_response.get("reason"),           # 仅用于 deny
                edited_args=ui_response.get("edited_args"), # 仅用于 edit
            )
            await channel.answer(req.question_id, human_answer)
        elif req.payload.kind == "ask":
            await channel.answer(req.question_id, await my_frontend.show_form(req))
        else:  # confirm
            await channel.answer(req.question_id, await my_frontend.show_confirm(req))
```

## 决策语义

| 人类选择 | 工具结果 | `hitl_trace["decision"]` | 模型看到 |
|---|---|---|---|
| 批准 | 使用原始参数执行 | 未设置（直通，无 HITL 细节） | 正常 `tool_result` |
| 拒绝 | 被阻止 | `"human_deny"` | `tool_result.is_error=True` 及用户的原因 |
| 编辑 | 使用编辑后的参数执行 | `"edit"` + `original_args` / `edited_args` | 正常 `tool_result`（来自编辑后的执行） |

策略决策（不经询问人类直接硬阻止）携带
`hitl_trace["decision"]="policy_deny"`。

## 超时行为

如果人类在 `timeout_seconds` 内未响应，中间件会转换为
`BeforeToolCallResult(block=True, deny_reason="approval_timeout")`。
模型看到 `tool_result.is_error=True`，其中
`details.hitl.decision="timed_out"`，并自然产生一个解释超时的后续轮次。

## 中止

如果用户关闭标签页或管理员终止对话：

```python
await agent.abort_pending(reason="user closed tab")
```

这会干净地关闭对话：为所有未解决的 tool call 追加合成的 deny tool_result，
持久化一个终止性的 `AssistantMessage(stop_reason="aborted")`，并触发
`AgentAbortedEvent`。下一次 `agent.prompt(...)` 重新开始。

## 运行示例

仓库中有一份完整可运行的代码，位于
[`examples/sandbox_confirm.py`](https://github.com/cubeplexai/cubeloop/blob/main/examples/sandbox_confirm.py)。
示例接入了一个模拟的 bash 工具和一个策略函数：读操作自动允许、危险写操作硬拒绝、其他操作由 host 循环自动确认。

```bash
git clone https://github.com/cubeplexai/cubeloop && cd cubeloop
uv sync

export ANTHROPIC_API_KEY=sk-ant-...   # 或 OPENAI_API_KEY [+ OPENAI_BASE_URL]
uv run python examples/sandbox_confirm.py
```
