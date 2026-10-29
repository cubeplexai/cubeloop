---
title: Anthropic
description: "通过 AnthropicProvider 在 CubeLoop 中使用 Claude 模型——支持思考、缓存和工具调用。"
---

# Anthropic Provider

`AnthropicProvider` 封装了官方 `anthropic` SDK 的 Messages API。
支持流式、扩展思考、prompt caching 和工具调用。

## 构造

```python
from cubeloop.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(
    api_key="sk-ant-…",          # 也可让 SDK 自己读 ANTHROPIC_API_KEY
    base_url=None,                # 指向代理 / Bedrock 兼容端点
    cache_retention="short",      # "short"(5 分钟,默认) | "long"(1 小时) | "none"
)
```

`api_key=None` 时 SDK 会从环境变量读取。

## `Model`

```python
provider = AnthropicProvider(provider_id="anthropic")
model = provider.model(
    "claude-sonnet-4-6",
    reasoning=True,           # 启用思考等级(见下)
    max_tokens=8192,          # 回复 token 上限
    context_window=200_000,   # 模型硬上限
    temperature=0.7,
)
```

第一个参数就是你传给 SDK 的模型名。`provider_id` 是 CubeLoop 内部用的
自由标签 —— 保持稳定即可,需要时可以让 tracing 和错误信息显示一个
明确的来源标签。

## 扩展思考

CubeLoop 把 `ThinkingLevel` 枚举映射到 Anthropic 的 `budget_tokens`:

| Level | 默认 budget |
|---|---|
| `"off"` | 关闭思考 |
| `"low"` | 2048 |
| `"medium"` | 8192 |
| `"high"` | 16384 |
| `"xhigh"` | clamp 到 `"high"` |

在 Agent 上设置：

```python
agent = Agent(model=model, thinking="medium")
```

要自定义 budget,可以通过自己的 `on_payload` hook 传入
`StreamOptions(thinking_budgets=ThinkingBudgets(low=4096, medium=12288))`。

思考打开时,CubeLoop **会把 `temperature` 字段省掉** —— 因为 Anthropic
API 在扩展思考模式下不接受非默认的 temperature（[兼容性文档](https://platform.claude.com/docs/en/build-with-claude/extended-thinking#feature-compatibility)）。
关闭思考时再按 `Model.temperature` 走;CubeLoop 自己处理切换。

思考内容以 `thinking_start` / `thinking_delta` / `thinking_end` 事件
流式产出,最终作为 `ThinkingContent` 块保存在 `AssistantMessage.content`
里 —— 后续轮次也会带着,以保证模型的思路连贯性。

## Prompt caching

默认情况下,provider 在每个请求上插入三个缓存断点：

- **system prompt**(最稳定)。
- **最后一个工具定义**(变化少)。
- **最后一条消息**(每轮往前推,把之前的历史全部缓存)。

缓存保留期默认 `"short"`(5 分钟,免费)。如果你的回合间隔较长,
切到 `"long"`:

```python
AnthropicProvider(provider_id="anthropic", api_key=…, cache_retention="long")  # 1 小时 TTL
AnthropicProvider(provider_id="anthropic", api_key=…, cache_retention="none")  # 完全禁用
```

每个 `AssistantMessage` 上的 `Usage` 对象会报告 `cache_read_tokens`
和 `cache_write_tokens`,方便你看命中率。

需要自定义缓存策略(比如换个断点策略)？实现 `CacheMarkerPolicy`
Protocol 然后 `cache_policy=…` 传入。默认策略类位于
`cubeloop.providers.anthropic.DefaultCacheMarkerPolicy`。

## 用 `on_payload` 自定义请求

`on_payload` 让你在请求发出前查看 / 替换 request dict:

```python
async def my_payload(payload, model):
    payload.setdefault("metadata", {})["user_id"] = "u-42"
    return payload     # 返回 None 或不返回保留原 payload

agent = Agent(model=model, on_payload=my_payload)
```

典型用法：加 `metadata.user_id`(计费)、强制 beta header,或留一份
调试面板用的 payload 副本。

## 用 `on_response` 自定义响应处理

`on_response` 在收到 HTTP 响应(状态、头)后、流式开始前触发：

```python
async def my_response(resp, model):
    if resp.status >= 400:
        logger.warning("bad status %s", resp.status)
    rate = resp.headers.get("anthropic-ratelimit-requests-remaining")
    if rate is not None:
        metrics.gauge("rate_remaining", int(rate))

agent = Agent(model=model, on_response=my_response)
```

两个回调都可以是同步或异步。

## 指向 Bedrock / Vertex / 代理

Anthropic SDK 接受 `base_url`,CubeLoop 透传：

```python
provider = AnthropicProvider(
    api_key="…",
    base_url="https://my-litellm.internal/v1",
)
```

对于 Bedrock,使用 `anthropic-bedrock` 适配器、通过 [自定义 provider](./custom)
注入。

## 常见坑

- **`temperature` 被忽略** —— 预期之内。思考打开时 CubeLoop 故意 drop
  掉,这是 API 约束,不是 bug。
- **`xhigh` 和 `high` 看起来一样** —— Anthropic 没有更高一档的 budget,
  所以 CubeLoop 把 `xhigh` clamp 到 `high`,token budget 相同。
- **缓存未命中** —— 缓存按 (内容, ttl) 索引。改 system prompt 会让
  整块失效;改工具列表则从工具往后失效。要最大化命中,保持这两个
  跨轮稳定。
- **`anthropic.RateLimitError`** —— 会作为 stream error 事件传出,
  错误消息来自 SDK 的 `str(exc)`。在 `agent_end` 里捕获,决定是否
  重试。

## 另请参阅

- [OpenAI Provider](./openai) —— 同 Protocol,不同形态。
- [自定义 Provider](./custom) —— 把不内建的 API 接进来。
- [Recipes → 多 Provider 容错](../../recipes/multi-provider-failover)
  —— Anthropic 故障时切到 OpenAI。
