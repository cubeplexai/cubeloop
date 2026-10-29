---
title: API、事件与参考
sidebar_position: 4
description: "CubeLoop HITL 参考：Agent API、事件、追踪 span、错误参考、测试辅助与架构说明。"
---

# API、事件与参考

## Agent API

| 属性/方法 | 签名 | 说明 |
|---|---|---|
| `agent.channel` | `HitlChannel \| None` | 绑定的 channel 或 `None` |
| `agent.in_flight_hitl_request` | `HitlRequest \| None` | channel 当前 pending（同步属性） |
| `load_pending_hitl_request()` | `async → HitlRequest \| None` | 从 checkpointer 读取 pending（detach 后也能用） |
| `detach()` | `async → None` | 发射 `AgentSuspendedEvent` 然后触发 `HitlDetached` |
| `respond(*, question_id=, answer=)` | `async → None` | 恢复挂起的运行 |
| `abort_pending(reason=)` | `async → None` | 关闭对话（两阶段：发信号 + 追加合成拒绝） |

## 事件

四个新事件在 agent 事件流上发射：

| 事件 | 何时发射 | 关键字段 |
|---|---|---|
| `HitlRequestEvent` | channel 收到新的 `confirm/approve/ask` | `request: HitlRequest` |
| `HitlAnswerEvent` | `channel.answer()` 或 `channel.cancel()` 触发 | `question_id`, `answer`, `cancelled`, `timed_out` |
| `AgentSuspendedEvent` | `agent.detach()` 被调用时 HITL pending | `pending_request: HitlRequest` |
| `AgentAbortedEvent` | `agent.abort_pending()` 关闭对话 | `reason: str` |

## 追踪 span

安装 `cubeloop[tracing]` extra 后，每次 HITL await 都会包装在一个
OpenTelemetry span 中：

| Span 名称 | 属性 |
|---|---|
| `hitl.approve` | `hitl.tool_name`, `hitl.tool_call_id`, `hitl.outcome`, `hitl.from_resume`, `hitl.duration_seconds` |
| `hitl.confirm` | `hitl.question_id`, `hitl.outcome`, `hitl.duration_seconds` |
| `hitl.ask` | `hitl.question_id`, `hitl.outcome`, `hitl.duration_seconds` |

`hitl.outcome` 可以是：`approved`, `denied`, `edited`, `answered`,
`cancelled`, `timed_out`, `aborted`, `detached`。

追踪导入是懒式的 —— 未安装 `opentelemetry` 时，channel 静默回退到
无操作 span。

## 错误参考

| 异常 | 基类 | 含义 |
|---|---|---|
| `HitlCancelled(reason)` | `BaseException` | 宿主调了 `channel.cancel(qid)` |
| `HitlTimedOut(seconds)` | `BaseException` | per-call 或 channel 默认超时到期 |
| `HitlDetached` | `BaseException` | HITL await 期间调了 `agent.detach()` |
| `HitlAborted` | `BaseException` | `agent.abort_pending()` 向 agent 发信号 |
| `HitlConcurrencyError` | `Exception` | channel 已有 pending 时再次调用 `confirm/approve/ask` |
| `HitlStaleAnswer` | `Exception` | `channel.answer(qid)` 的 qid 与当前 pending 不匹配 |
| `HitlNoPendingRequest` | `Exception` | 调了 `agent.respond(…)` 但线程没有 `pending_request` |
| `HitlDurabilityNotGuaranteed` | `Exception` | 自定义工具调了 `CheckpointedChannel.ask()` 但没有 `allow_inside_custom_tool=True` |

`HitlControlException`（四个 `BaseException` 子类的父类）**故意**不被
`cubeloop.agent.tools._prepare_tool_call` 和 `_execute_prepared` 中现有
的广泛 `except Exception:` 处理器捕获 —— 这模仿了 `asyncio.CancelledError`
的模式。


## 测试辅助

```python
from cubeloop.hitl.testing import ScriptedChannel, NoopChannel

# ScriptedChannel：预编程答案，按顺序消费
ch = ScriptedChannel(answers=[
    ApproveAnswer(decision="approve"),
    {"color": "red"},                       # ask 答案
])
assert len(ch.history) == 2  # 所有见过的 HitlRequest

# NoopChannel：自动批准一切。用于测试中的 subagent。
ch = NoopChannel()
assert (await ch.approve("bash", "tc", {})).decision == "approve"
```

## 架构说明

- **每线程单一 pending。** Agent 循环是顺序的——每个 `thread_id` 最多
  有一个 HITL 请求。并发 `confirm/approve/ask` 会抛 `HitlConcurrencyError`。
- **Prompt-cache 前缀不变量。** 暂停与恢复之间，消息列表仅通过在末尾
  追加 tool-result 消息和下一条 assistant turn 来改变。不插入、重排或
  修改先前的消息——否则会使 provider 端的 prompt cache 失效。
- **`question_id == tool_call_id` 适用于 approve 请求。** 无需别名或
  映射——已从工具流中追踪 `call_id` 的宿主可直接传递。
- **恢复不重放。** 恢复时将答案预加载到 channel 并重新进入循环。最后一条
  assistant message 中未决的 tool call 决定了接下来执行什么。没有基于
  节点的重放语义。
