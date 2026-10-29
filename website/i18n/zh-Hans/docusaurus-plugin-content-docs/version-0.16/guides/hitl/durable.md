---
title: 跨进程与持久化
sidebar_position: 3
description: "用 CheckpointedChannel 在一个进程挂起 HITL 请求、在另一个进程恢复；持久化范围规则。"
---

# 跨进程流程与持久化

## 跨进程（web 服务）流程

```python
# ───── 进程 1：HTTP POST /chat ─────

async with PostgresCheckpointer("postgresql://...") as cp:
    channel = CheckpointedChannel(checkpointer=cp, thread_id="conv-42")

    agent = Agent(
        model=…,
        tools=[bash_tool],
        middleware=[ConfirmToolCallMiddleware(channel, require_confirm={"bash"})],
        channel=channel, checkpointer=cp, thread_id="conv-42",
    )

    task = asyncio.create_task(agent.prompt("删除临时文件"))

    # 轮询挂起（或订阅 channel 用于 SSE 推送）
    for _ in range(1000):
        pending = channel.pending
        if pending is not None:
            break
        await asyncio.sleep(0.1)

    # 优雅挂起 — 持久化 assistant message + 未决 tool_calls,
    # pending_request 留在 DB, 发射 AgentSuspendedEvent.
    await agent.detach()
    await task


# ───── 进程 2：HTTP POST /respond ─────

async with PostgresCheckpointer("postgresql://...") as cp:
    channel = CheckpointedChannel(checkpointer=cp, thread_id="conv-42")

    agent = Agent(
        model=…,
        tools=[bash_tool],
        middleware=[ConfirmToolCallMiddleware(channel, require_confirm={"bash"})],
        channel=channel, checkpointer=cp, thread_id="conv-42",
    )

    await agent.respond(
        question_id=request.json["call_id"],
        answer=ApproveAnswer(decision="approve"),
    )
    # Bash 工具执行，模型收到 tool_result，产生下一个 assistant turn。
```

**用户关闭 tab 未回答时：**

```python
await agent.abort_pending(reason="用户关闭了 tab")
# Phase 1：发信号给进行中的 HITL await（如有），触发 HitlAborted。
# Phase 2：为未决的 tool_calls 追加合成 deny ToolResultMessage，
#   追加终止 AssistantMessage(stop_reason="aborted")，
#   清除持久化 pending, 发射 AgentAbortedEvent。
# 不调用模型。对话关闭。
```


## 持久化范围

持久的跨进程恢复（进程死亡后仍能继续）在两个定义明确的安全暂停点支持：

1. **`before_tool_call` 确认门** —— 确认中间件在工具的 `execute()` body
   运行*之前*调用 `channel.approve(...)`。此时不存在工具副作用。恢复时
   重新进入循环，执行（可能被编辑过的）工具体或替换为合成 deny
   tool_result。
2. **`ask_user` 工具体** —— 其整个 `execute()` body 就是
   `return await channel.ask(...)`。恢复时不会重放任何内容。

**默认情况下，在 `execute()` 内将 HITL 与其他工作混合的自定义工具不
支持跨进程持久化。** 如果此类工具的进程在执行中途死亡，channel 调用
前运行的所有内容都会丢失。除非使用 `allow_inside_custom_tool=True`
构造 `CheckpointedChannel`，否则将抛出 `HitlDurabilityNotGuaranteed` —
调用者必须承认等幂性契约（工具体在这一点必须是纯 HITL 等待，前面没有
可观察的副作用）。
