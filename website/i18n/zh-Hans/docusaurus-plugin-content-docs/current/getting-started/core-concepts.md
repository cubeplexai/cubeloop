---
title: 核心概念
description: "了解 CubeLoop 的六个核心概念：Agent、Tool、Provider、Stream & Events、Middleware 和 Checkpointer。"
---

# 核心概念

CubeLoop 的全部能力可以归纳为六个概念。这一页读一遍,后面的文档基本就
变成查表了。

## Agent

`Agent` 是有状态的门面：你用 provider、model、可选的工具、可选的
middleware/checkpointer 构造它。普通应用通过三个方法驱动它：

- `await agent.prompt(message)` —— 用一条 user 消息开启新一轮。
- `await agent.resume()` —— 从最后一条已持久化的消息继续（配合
  checkpointer 使用）。
- `agent.steer(message)` / `agent.follow_up(message)` —— 在正在跑的
  过程中插入消息,或为当前运行结束后排队下一条。

Agent 持有一个 `AgentState`（system prompt、tools、model、消息历史、
未结束的 tool call、流式标志）和一个 subscribers 列表：

```python
unsubscribe = agent.subscribe(my_listener)
# ...
unsubscribe()
```

Subscriber 会收到循环发出的每一个 `AgentEvent`。可以是同步或异步函数。

需要明确获知每次执行结果的宿主，可以通过 `agent.session` 使用同一套执行引擎：

```python
from cubeloop import PromptExecutionRequest

result = await agent.session.execute(
    PromptExecutionRequest(
        run_id="run-42",
        attempt_id="worker-attempt-1",
        message="继续分析",
    )
)

if result.outcome == "suspended":
    render_form(result.pending_request)
elif result.outcome != "completed":
    record_failure(result.error)
```

一个执行会话属于一个 `Agent`，其中的多次尝试按顺序运行。它不是分布式锁，
也不是持久化的 worker 会话。执行尚未结束时再次调用 `execute()` 会抛出
`ExecutionBusy`；准入、权限校验、租约和传输层状态仍由应用负责。

## Tool

工具就是一个模型可以调用的 async 函数。用 `@tool` 装饰它,CubeLoop 会从
参数生成输入 schema:

```python
from cubeloop import tool

@tool
async def search(query: str, limit: int = 10) -> str:
    "搜索语料库"
    # 干活；需要的话在签名里声明 signal / on_update
    return "…"
```

`@tool` 会构建一个 `AgentTool`(name + description + Pydantic 参数模型 +
异步 `execute`);schema 自动转成 JSON Schema 喂给模型。参数解析、错误
包装、并行执行都由框架处理。长写法 `AgentTool(...)`、`execution_mode`、
`on_update`（增量进度）、`terminate`（在工具里结束本轮）见
[工具使用](../guides/agents/tool-use)。

## Provider

任何匹配下面 Protocol 的对象就是 Provider：

```python
class Provider(Protocol):
    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream: ...
```

它返回一个 `MessageStream` —— 一个统一的异步迭代器,产出 `StreamEvent`,
并通过 `await stream.result()` 暴露最终的 `AssistantMessage`。内置 Provider：

- `AnthropicProvider` —— Claude(Messages API,支持思考、缓存、工具使用)。
- `OpenAIProvider` —— GPT 家族(Chat Completions API)。
- `OpenAIResponsesProvider` —— GPT 家族(Responses API,服务端状态)。
- `FauxProvider` —— 确定性测试替身(不发任何网络请求)。

实现一个方法就能写自己的。见 [Providers / 自定义](../guides/providers/custom)。

## Stream 和事件

流和事件分两层：

- **Provider 流** —— `MessageStream` 产出的是 *provider* 事件：
  `start`、`text_start`、`text_delta`、`text_end`、`thinking_*`、
  `toolcall_*`、`done`、`error`。原始 token 流。
- **Agent 事件** —— `agent.subscribe(...)` 收到的内容。十四种类型
  覆盖整个循环 + HITL：`agent_start`、`agent_end`、`turn_start`、
  `turn_end`、`message_start`、`message_update`、`message_end`、
  `tool_execution_start`、`tool_execution_update`、
  `tool_execution_end`、`hitl_request`、`hitl_answer`、
  `agent_suspended`、`agent_aborted`。

做 UI 订阅 Agent 事件；做底层 token 路由就钻 `event.stream_event`。
见 [流式事件](../guides/agents/streaming)。

`agent.session.subscribe(...)` 在此之上增加了面向宿主的事件层。每个事件都带有
`run_id`、`attempt_id` 和本次尝试内单调递增的序号。注入的消息成功追加到
checkpointer 后，会话会发出 `input_committed`；每次已接受的尝试只会发出一个
`execution_finished`。只有在缺少某个消费者就无法安全继续执行时，才应将它标记为
`required=True`；观察者发生故障不会改变执行结果。

## Middleware

`Middleware` 是有最多九个类型化 hook 的类：

| Hook | 何时触发 | 组合规则 |
|---|---|---|
| `transform_context` | 每次调模型之前,处理消息列表 | 链式 —— 每个收到上一个的输出 |
| `convert_to_llm` | provider 序列化之前 | 最后一个实现生效 |
| `transform_system_prompt` | 每次调模型之前,处理 system prompt | 链式 |
| `resolve_tool_call` | 每个工具调用之前（参数校验之前，可改写调用） | 第一个非 `None` 胜出 |
| `before_tool_call` | 每个工具调用之前(在参数校验后) | 第一个 `block=True` 短路 |
| `after_tool_call` | 每个工具调用之后(在 `execute` 之后) | 后写覆盖先写 |
| `after_model_response` | assistant 消息落定之后 | 返回 `TurnAction` 控制流向 |
| `should_stop_after_turn` | 每个轮次结束时 | 任一返回 `True` 即停 |
| `on_run_end` | 所有轮次完成后一次，`agent_end` 之前 | 消息拼接；非空触发一轮额外调用 |

通过 `Agent(middleware=[...])` 传入。见
[Middleware → 组合规则](../guides/middleware/composition)。

## Checkpointer

任何匹配下面 Protocol 的对象就是 Checkpointer：

```python
class Checkpointer(Protocol):
    async def load(self, thread_id: str) -> CheckpointData | None: ...
    async def append(self, thread_id: str, messages: list[Message]) -> None: ...
    async def save_extra(self, thread_id: str, extra: dict) -> None: ...
```

通过 `Agent(checkpointer=cp, thread_id="…")` 绑定到 Agent,循环就会在
每条消息落定时追加一行,并在第一次 `prompt()` 时恢复历史。内置后端：
`MemoryCheckpointer`、`SQLiteCheckpointer`、`PostgresCheckpointer`、`MySQLCheckpointer`。
见 [Checkpointing → SQLite](../guides/checkpointing/sqlite)。

HITL 为跨进程挂起/恢复新增了两个可选方法：`save_pending_request` /
`load_pending_request`。所有第一方后���都已实现。见 [HITL 指南](../guides/hitl/overview)。

## HITL（人机协同）

CubeLoop 内置了 `cubeloop.hitl` 模块，用于 agent 需要**暂停并等待人类输入**的
场景：

- **沙箱确认** —— 危险工具（bash、写入文件）在执行前需要人类
  approve / deny / edit。
- **运行中提问** —— agent 在运行中途向用户弹出一个结构化表单，等待回答。

```python
from cubeloop.hitl import InMemoryChannel, ConfirmToolCallMiddleware, ask_user_tool

channel = InMemoryChannel()

agent = Agent(
    model=…,
    tools=[bash_tool, ask_user_tool(channel)],
    middleware=[ConfirmToolCallMiddleware(channel, require_confirm={"bash"})],
    channel=channel,
)
```

Channel 是一个可 `await` 的协程协作者：工具和中间件作者写
`await channel.ask(...)` 或 `await channel.confirm(...)`，channel
处理暂停。宿主代码（你的 web 应用 / TUI）订阅 `channel.subscribe()` 或
轮询 `channel.pending`，把请求渲染给用户，然后通过
`channel.answer(qid, answer)` 回填答案。

内置两种 channel 后端：
- **`InMemoryChannel`** —— 单进程（CLI、notebook、测试）。
- **`CheckpointedChannel`** —— 跨进程（web 服务）。挂起的请求持久化到
  checkpointer；另一个进程可以在数小时后通过
  `Agent.respond(question_id=, answer=)` 回答。

完整细节——三种 HITL 动词、两套内置中间件、跨进程挂起/恢复协议、事件、
追踪 span 和错误参考——见 [HITL 指南](../guides/hitl/overview)。

## Tracer（可选）

`Tracer` 输出符合 [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
的 span,任何 OTLP 后端（Jaeger、Tempo、Honeycomb、Datadog、AWS
X-Ray 等）都能直接接收,无需额外 instrumentation。先装 extra：

```bash
pip install "cubeloop[tracing]"           # OTel SDK
pip install "cubeloop[tracing-otlp]"      # + OTLP/HTTP 导出器
```

然后用 `async with` 包住 Agent：

```python
from cubeloop.tracing import Tracer
from cubeloop.tracing.exporters import JsonlSpanExporter

async with (
    Tracer(
        service_name="my-bot",
        agent_name="assistant",
        exporters=[JsonlSpanExporter(directory="./cubeloop-traces")],
    ) as tracer,
    tracer.attached(agent),
):
    await agent.prompt("…")
```

每次 run 会发出一个 `invoke_agent` 根 span,其下每轮 LLM 往返对应
一个 `cubeloop.turn`,再嵌套 `chat`（CLIENT）和 `execute_tool` 子
span。**默认不记录任何 prompt 内容或模型输出** —— 需要的话用
`Tracer(record_content=True)` 显式打开,搭配 `redact` 回调脱敏。配
合 `Meter(...)` 还能拿到 token / 时延 / TTFC 直方图。完整指南：
[追踪 → 概览](../guides/tracing/overview)。

## 拼起来

```
用户代码
   │
   ▼
┌──────────────────────────────────────────┐
│ Agent                                     │
│  ├─ AgentState (messages, tools, …)       │
│  ├─ Middleware ── compose_middleware()    │
│  ├─ Checkpointer ── message_end 时追加    │
│  └─ run_agent_loop  ◀──── 真实的循环      │
│       │                                   │
│       ▼                                   │
│  Provider.stream() → MessageStream        │
│       │                                   │
│       └─ events → emit → subscribers      │
└──────────────────────────────────────────┘
```

这张图就是整个框架。文档站的其余部分,都是细节而已。
