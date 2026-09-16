---
title: 自定义 Provider
description: "通过实现 Provider 协议，为 CubeLoop 编写自定义 provider。"
---

# 自定义 Provider

provider 是任意实现了以下单个方法的类：

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

这就是完整的接口。实现它，`Agent(model=…)` 就可以接受你的类。

本页涵盖两种场景：

1. **新的真实 provider** —— Bedrock、Vertex、Replicate、内部 LLM 网关……
2. **`FauxProvider`** —— 内置的确定性 provider，是单元测试的利器。

## 最小化真实 provider

模式：创建一个 `MessageStream`，启动一个向其推送事件的生产者任务，然后立即返回 stream。

```python
import asyncio
import time
from cubeloop.providers.base import (
    AssistantMessage,
    Message,
    MessageStream,
    Model,
    StreamEvent,
    StreamOptions,
    TextContent,
    ToolDefinition,
    Usage,
)


class MyProvider:
    def __init__(self, *, api_key: str) -> None:
        self._api_key = api_key

    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream:
        opts = options or StreamOptions()
        ms = MessageStream()

        async def _produce():
            try:
                partial = AssistantMessage(
                    content=[TextContent(text="")],
                    usage=Usage(),
                    timestamp=time.time(),
                    provider_id=model.provider_id,
                    model_id=model.id,
                )
                ms.push(StreamEvent(type="start", partial=partial.model_copy(deep=True)))

                # Call your backend. Stream tokens:
                async for token in call_my_backend(messages, model.id, signal=opts.signal):
                    if opts.signal and opts.signal.is_set():
                        ms.push(StreamEvent(type="error", error_message="aborted"))
                        ms.set_result(partial.model_copy(update={"stop_reason": "aborted"}))
                        return
                    partial.content[-1] = TextContent(text=partial.content[-1].text + token)
                    ms.push(StreamEvent(
                        type="text_delta",
                        delta=token,
                        partial=partial.model_copy(deep=True),
                    ))

                ms.push(StreamEvent(type="done"))
                ms.set_result(partial)

            except Exception as exc:
                error_msg = AssistantMessage(
                    content=[],
                    stop_reason="error",
                    error_message=str(exc),
                    usage=Usage(),
                    timestamp=time.time(),
                )
                ms.push(StreamEvent(type="error", error_message=str(exc)))
                ms.set_result(error_msg)

        ms.attach_task(asyncio.create_task(_produce()))
        return ms
```

需要注意的要点：

1. **始终先推送 `start` 事件。** 订阅者依赖它进行 UI 初始化。
2. **始终以 `done` 或 `error` 结束。** agent 循环会等待 `MessageStream` 直到收到其中之一。
3. **始终调用 `ms.set_result(...)`**，以便 `await stream.result()` 能够完成。即使出错也需调用。
4. **如果生产者是独立任务，`ms.attach_task(...)` 是必须的** —— 它将任务的异常状态接入 stream，使崩溃表现为 `error` 而非挂起。
5. **遵守 `opts.signal`。** 在读取循环内部检查它；发出 `aborted` stop_reason，让 agent 能够干净地关闭。

## 支持工具调用

如果你的模型会产生工具调用，在流式传输时将 `ToolCall` 块追加到 `partial.content`，并发出 `toolcall_start` / `toolcall_delta` / `toolcall_end` 事件：

```python
from cubeloop.providers.base import ToolCall

tc = ToolCall(id=block_id, name=tool_name, arguments={})
partial.content.append(tc)
ms.push(StreamEvent(type="toolcall_start", content_index=len(partial.content)-1,
                    partial=partial.model_copy(deep=True)))
# ... as JSON args arrive:
ms.push(StreamEvent(type="toolcall_delta", delta=partial_json_chunk, …))
# ... on completion:
# replace tc.arguments with the parsed dict, push toolcall_end
```

CubeLoop 的 agent 循环会在收到 `done` 事件后自动分发工具调用。

## 挂钩 `on_payload` / `on_response`

如果你的 provider 发送 HTTP 请求，请调用 `cubeloop.providers.base` 中的辅助函数：

```python
from cubeloop.providers.base import (
    ProviderResponse,
    invoke_on_payload,
    invoke_on_response,
)

payload = await invoke_on_payload(opts.on_payload, payload, model)
http_resp = await self._client.post(..., json=payload)
await invoke_on_response(
    opts.on_response,
    ProviderResponse(status=http_resp.status_code, headers=dict(http_resp.headers)),
    model,
)
```

这与内置 provider 的做法一致；你的用户可以免费获得相同的检查点。

## 在测试中使用 `FauxProvider` {#using-fauxprovider-in-tests}

CubeLoop 内置 `FauxProvider`，用于确定性测试——无网络调用，无不稳定性，且有真实的流式事件：

```python
from cubeloop import Agent
from cubeloop.providers import FauxProvider, faux_assistant_message, faux_text, faux_tool_call


def test_my_agent():
    provider = FauxProvider(provider_id="faux")
    provider.set_responses([
        faux_assistant_message([
            faux_tool_call("search", {"query": "python"}),
        ]),
        faux_assistant_message("Here are the results: …"),
    ])

    agent = Agent(
        model=provider.model("test"),
        tools=[my_search_tool],
    )
    events = []
    agent.subscribe(lambda e, signal=None: events.append(e))
    await agent.prompt("Search for python")

    assert any(e.type == "tool_execution_start" for e in events)
    assert events[-1].type == "agent_end"
```

`set_responses` 是一个 FIFO 队列：每次模型调用弹出一个响应。faux provider 以真实的增量方式（逐 token）回放，因此你的流式代码路径会被真正执行。

辅助函数：

- `faux_text("Hello!")` —— 将字符串包装为 `TextContent` 块。
- `faux_thinking("Pondering…")` —— 一个 `ThinkingContent` 块。
- `faux_tool_call("name", {"arg": …})` —— 一个 `ToolCall` 块。
- `faux_assistant_message(content_or_text)` —— 构建完整的 `AssistantMessage`。

## 常见问题

- **缺少 `start` 事件** —— 订阅者看不到部分消息。始终先推送它。
- **忘记调用 `ms.set_result(...)`** —— `await agent.prompt()` 永远挂起。在成功路径和错误路径都要设置结果。
- **同步的 `produce`** —— `stream()` 必须**立即**返回——将工作放在任务内部。如果在 `return ms` 之前 `await` 了你的后端，就阻塞了调用方。
- **推送后修改 `partial`** —— 推送时使用 `deep=True` 拷贝；调用方异步迭代事件，在迭代过程中发生变更会产生极难调试的别名问题。

## 参见
- [图片生成](./image-generation) —— 使用 `openai-images` 与 OpenAI 图片模型。

- [Providers Overview](./overview) —— 在从头编写类之前，先检查你的后端是否只是一个 `CapabilityDescriptor` 或内置预设已经覆盖的 OpenAI/Anthropic 兼容端点。
- [API 参考 → providers/base](../../api/cubeloop-providers) —— 完整类型列表。
- [Anthropic Provider 源码](https://github.com/cubeplexai/cubeloop/blob/main/cubeloop/providers/anthropic.py) —— 一个真实完整的示例。
- [`FauxProvider` 源码](https://github.com/cubeplexai/cubeloop/blob/main/cubeloop/providers/faux.py) —— 测试原语，包含流式真实性的细节。
