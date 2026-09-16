---
title: 流式输出
description: "通过 subscriber 和 MessageStream 实时流式接收 CubeLoop agent 的事件和 token。"
---

# 流式事件

CubeLoop 暴露两层流：

1. **Provider 流** —— `provider.stream(...)` 返回的 `MessageStream`,
   产出 `StreamEvent` 描述原始线缆输出：文本 delta、思考 delta、
   工具调用 delta,最后是 `done` 或 `error`。
2. **Agent 事件流** —— `agent.subscribe(...)` 注册的 listener 看到的
   内容。十一种事件覆盖 `prompt()` 调用的完整生命周期,包括把 provider
   事件包在 `MessageUpdateEvent` 里。

大部分应用代码只需要 Agent 事件流。

## 十一种 Agent 事件

| 事件 | 何时触发 |
|---|---|
| `agent_start` | `prompt()` / `resume()` 一开始 |
| `turn_start` | 每次调模型之前(一次 `prompt` 可能调多次) |
| `message_start` | 一条新消息(user / assistant / tool result)即将加入历史 |
| `message_update` | 每个 provider `StreamEvent`(deltas 等);`event.stream_event` 字段会带原始事件 |
| `message_end` | 一条消息已 finalize |
| `tool_execution_start` | 一个工具调用被派发(在并行 `asyncio.gather` 之前各发一次) |
| `tool_execution_update` | 一个工具通过 `on_update(...)` 报告了局部进度 |
| `tool_execution_end` | 一个工具结束(成功或失败) |
| `turn_end` | 当前批次的工具都尘埃落定,或一条无工具的 assistant 回复结束 |
| `agent_end` | 整个 `prompt()` 调用结束 —— 正常结束、abort、或 error 都会发 |

`MessageStartEvent` 和 `MessageEndEvent` 对 *每种* 消息都触发,不只是
assistant。User 消息和 tool result 消息也有。

## 工具调用一轮的事件顺序

典型 "用户提问 → 模型调一个工具 → 模型回答" 的事件序列：

```
agent_start
turn_start
  message_start         (从 prompt 来的 UserMessage)
  message_end           (UserMessage)
  message_start         (空的 AssistantMessage partial)
  message_update × N    (text_delta、toolcall_delta、…)
  message_end           (finalize 后的 AssistantMessage)
  tool_execution_start
  tool_execution_end
  message_start         (ToolResultMessage)
  message_end           (ToolResultMessage)
turn_end
turn_start              (循环带着工具结果再进模型)
  message_start
  message_update × N
  message_end
turn_end
agent_end
```

## 订阅

```python
def on_event(event, signal=None):
    if event.type == "message_update" and event.stream_event.type == "text_delta":
        print(event.stream_event.delta, end="", flush=True)

unsubscribe = agent.subscribe(on_event)
```

`agent.subscribe(...)` 永远不会收到 `event.type == "text_delta"`
的事件 —— 那是 *provider* 事件的 type。Agent 把每个 provider 事件
都包成 `MessageUpdateEvent`,原事件挂在 `event.stream_event` 上。
所以要同时匹配外层和内层。

Listener 可以是同步或异步,异步的会被 await。第二个参数是 run 级别
的 `asyncio.Event`(abort signal)—— 你可以查 `signal.is_set()` 判断
本次运行是否被取消。

要取消订阅,调用 `subscribe` 返回的那个函数。

## 过滤文本增量(最常见的用法)

```python
def on_event(event, signal=None):
    if event.type == "message_update":
        sub = event.stream_event
        if sub.type == "text_delta":
            print(sub.delta, end="", flush=True)
```

CubeLoop 保证的稳定结构是上面表格里那一种(`message_update.stream_event.delta`)。
代码里对内部类型做防御性判断。

## Provider 的 `StreamEvent` 类型

在 `MessageUpdateEvent.stream_event` 内部,type 字段告诉你模型正在
吐什么：

| `stream_event.type` | 含义 | 关键字段 |
|---|---|---|
| `start` | assistant 消息开始 | `partial` |
| `text_start` | 一个文本块开始 | `content_index` |
| `text_delta` | token 片段 | `delta` |
| `text_end` | 文本块结束 | `content_index` |
| `thinking_start` / `thinking_delta` / `thinking_end` | 扩展思考块 | `delta` |
| `toolcall_start` / `toolcall_delta` / `toolcall_end` | 工具调用的流式 JSON 参数 | `delta`(partial JSON) |
| `done` | 流正常结束 | — |
| `error` | 流出错 | `error_message` |

每个事件的 `partial` 字段是当前 `AssistantMessage` 的深拷贝快照,
方便 UI 不追 delta、每个事件直接重渲染。

## 直接迭代 provider 流

如果你完全跳过 `Agent`(罕见,通常意味着你在写测试或自定义编排),
直接迭代流即可：

```python
stream = await provider.stream(
    model=model,
    messages=[UserMessage(content=[TextContent(text="hello")])],
)
async for event in stream:
    if event.type == "text_delta":
        print(event.delta, end="", flush=True)
final = await stream.result()   # 拿最终的 AssistantMessage
```

`stream.result()` 在迭代结束后也能调用 —— 这是拿最终消息的标准
方式。

## 常见坑

- **`prompt()` 之后才订阅** —— 早期事件已经发完了。先 subscribe,
  再 prompt。
- **Listener 抛异常会让循环崩吗？** —— 不会,但异常会向上传播到下一个
  `await`。把风险代码包在 try/except 里。
- **并行工具的事件顺序** —— `tool_execution_start` 按模型发出顺序,
  `tool_execution_end` 按完成顺序。别依赖事件成对相邻。
- **`message_update` 触发太频繁** —— 高频 token 可能压垮慢消费者
  （比如 websocket）。消费端做批处理。
- **思考块会触发 `text_delta` 吗？** —— 不会,思考块发 `thinking_*`
  事件。只想看可见文本就过滤 `event.stream_event.type`。

## 另请参阅

- [工具使用](./tool-use) —— 详解 `tool_execution_*` 三联事件。
- [多轮会话](./multi-turn) —— steering 和 resume 周边的事件顺序。
- [API Reference → StreamEvent](../../api/cubeloop-providers#streamevent)
  里有字段级 schema。
