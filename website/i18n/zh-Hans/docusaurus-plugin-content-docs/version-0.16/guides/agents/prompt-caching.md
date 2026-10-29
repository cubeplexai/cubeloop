---
title: Prompt 缓存
description: "CubeLoop 如何最大化 prompt 缓存命中率——追加式消息存储、自动缓存断点，以及如何避免破坏缓存。"
---

# Prompt 缓存

Prompt 缓存让 LLM provider 在 stable prefix 上复用 KV 计算，而不是每一轮
都重新处理。收益非常可观：Anthropic 对命中缓存的 token 按**正常输入价格
的 10%** 收费，命中也会降低 time-to-first-token。

CubeLoop 从底层架构就为最大化缓存命中而设计。这篇指南讲解缓存是怎么工作
的、为什么 CubeLoop 的架构能让缓存常驻，以及你可以做什么（或不要做什么）
来保住命中率。

---

## Prompt 缓存如何工作

每一次 LLM API 请求都是一串内容块：system prompt、tool 定义、对话历史。
provider 对每一块的字节内容做哈希。如果当前请求与最近 TTL 内见过的某次
请求共享一段前缀，那段前缀对应的 KV 状态会被复用——provider 只处理超出
缓存边界的新 token。

```
Request N:   [system] [tools] [msg 1] [msg 2] [msg 3]
                  ↑         ↑             ↑
             cached     cached        cached      ← 第 N+1 轮全部命中
                                              [msg 4]  ← 只算新 token
```

命中需要两个条件：

1. **前缀必须字节一致**。任何改动——重排、重新格式化、加一个字符——都
   会 miss。
2. **TTL 没过期**。Anthropic 提供 5 分钟（`"short"`）和 1 小时
   （`"long"`）两档；OpenAI 对超过 1 024 token 的 prompt 自动缓存。

---

## 为什么追加式存储很关键

很多 Agent 框架在每一轮都从 checkpoint 快照重建完整的消息列表。如果
序列化细节变了——dict key 顺序、空白字符、一个时间戳字段——产出的字节
序列就跟上一次的请求不一样了，于是**从那一位置开始的所有缓存断点全部
miss**。

CubeLoop 的 checkpointer 是追加式的：只往 thread 末尾添加新消息。内存里
的 `agent.state.messages` 列表通过 append 增长，从不重建、从不重排。
这意味着每一次请求的前缀都和上一轮字节一致——这是命中缓存最基础的
要求。

---

## Anthropic：自动缓存断点

`AnthropicProvider` 通过 `DefaultCacheMarkerPolicy` 自动插入
`cache_control` 标记。默认每个请求放三个断点：

| 断点 | 缓存什么 | 为什么放这里 |
|---|---|---|
| System prompt | 完整的 system prompt 块 | 内容最稳定——很少跨轮变化 |
| 最后一个 tool 定义 | 截至并包含最后一个的全部 tool schema | tool 列表很少变 |
| 历史最后一条消息 | 此前的全部对话历史 | 每一轮前移；早先的轮次保持温热 |

到第 N+1 轮，前两个断点和第 N 轮完全一致（同一个 system prompt、同一组
tool），所以命中缓存。最后一条消息的断点向前移了一位，所以会写一条新
缓存覆盖稍长的历史——这条会在第 N+2 轮命中。

### 配置

```python
from cubeloop.providers.anthropic import AnthropicProvider

# 默认：short TTL（5 分钟），自动断点
provider = AnthropicProvider(api_key="…")

# Long TTL（1 小时）——轮次慢或用户不频繁时用
provider = AnthropicProvider(api_key="…", cache_retention="long")

# 完全关闭缓存
provider = AnthropicProvider(api_key="…", cache_retention="none")
```

### 读取缓存指标

每条 `AssistantMessage` 都带一个 `Usage`。缓存相关字段：

```python
agent.subscribe(lambda event, signal=None: None)
await agent.prompt("Summarise the document")

last_msg = agent.state.messages[-1]   # AssistantMessage
usage = last_msg.usage
print(usage.input_tokens)        # 未命中缓存的 prompt token（不含 cache_read）
print(usage.cache_read_tokens)   # 从缓存读取的 token  ← 省钱处
print(usage.cache_write_tokens)  # 本轮写入缓存的 token
```

这一轮的缓存命中率：
`cache_read_tokens / (input_tokens + cache_read_tokens + cache_write_tokens)`。

:::tip
CubeLoop 的 `Usage.input_tokens` 是**未命中缓存**的那部分——本轮真正被模型
处理的 token。完整 prompt token 数是
`input_tokens + cache_read_tokens + cache_write_tokens`。100% 命中时
`input_tokens` 为 0。
:::

`cubeloop trace` CLI 输出和 `Tracer` 发出的 OTel span 里也能看到这些字段。
注意 OTel span 属性 `gen_ai.usage.input_tokens` 遵循 GenAI 语义约定，
报的是**包含**未命中和命中的总数：

```
gen_ai.usage.input_tokens          = 8 420   （包含：未命中 + 命中）
gen_ai.usage.cache_read.input_tokens = 7 980  （命中那部分）
```

---

## OpenAI：自动缓存

OpenAI 对超过 1 024 token 的 prompt 自动缓存——不需要显式的
`cache_control` 标记。CubeLoop 通过同一个 `Usage` 接口暴露命中数据：

```python
usage.cache_read_tokens   # 对应 prompt_tokens_details.cached_tokens
```

CubeLoop 端没什么可配。保持 system prompt 和 tool 定义在轮次间稳定，
OpenAI 的缓存会自然预热起来。

---

## 如何避免破坏缓存

### ✅ 该做

- **保持 system prompt 跨轮稳定。** system prompt 是最外层的缓存
  断点。改了它，后面全废。
- **保持 tool 列表稳定。** 增删 tool 在**会话之间**做，不在**轮次
  之间**做。tool 定义断点覆盖整张 tool 列表；任何改动都从那里开始失效。
- 如果 agent 一轮要超过 5 分钟（长思考、慢工具、低频用户），
  **用 `cache_retention="long"`**。
- **用 `agent.steer()`** 注入中途指令，而不是往历史前面塞新消息——
  `steer` 是 append，不是 insert。

### ❌ 该避免

- **往历史中间插入消息**。在最后位置之前插一条，会把后续所有消息
  位移，序列就跟 provider 缓存的不一样了。
- **把按请求变化的数据（当前时间戳、请求 ID）放进 system prompt**。
  易变数据放到 user message 里。
- **重排 tool 定义。** tool 列表是按顺序序列化的；顺序变了就 miss，
  即使 tool 本身一样。
- **轮次间改 tool 的描述**（同一 thread）。描述会被序列化进 schema，
  会破坏 tool 定义断点。

---

## 自定义缓存策略（Anthropic）

如果默认的三断点策略不适合你的场景，实现 `CacheMarkerPolicy` 并传给
provider：

```python
from cubeloop.providers.anthropic import CacheMarkerPolicy, AnthropicProvider
from cubeloop.providers.base import Message

class SystemOnlyPolicy:
    """只缓存 system prompt——tool 列表经常变时有用。"""

    def mark_system(self) -> bool:
        return True

    def mark_last_tool(self) -> bool:
        return False

    def message_breakpoint_indices(self, messages: list[Message]) -> list[int]:
        return []   # 不在消息上设断点

provider = AnthropicProvider(
    api_key="…",
    cache_policy=SystemOnlyPolicy(),
)
```

三个方法直接对应三个默认断点。返回 `True` / 非空列表则启用断点，
返回 `False` / `[]` 则跳过。

---

## 多租户考虑

多租户场景里每个 `thread_id` 都是一段独立的对话。缓存命中是按 thread
来的：用户 A 的历史预热的缓存只惠及用户 A 后续的轮次。

跨租户的缓存效率最高的做法：

- **所有租户共用一份稳定的 system prompt**，不要在 system prompt 里
  塞租户特定数据。每租户的上下文放到第一条 user message 或 tool 结果
  里。
- 所有租户的 agent 用相同的 tool 定义。如果不同租户需要不同的 tool 集，
  考虑用不同的 `Agent` 实例和不同的 `AnthropicProvider` 配置，而不是
  动态地改 tool 列表。

---

## 参见

- [Anthropic Provider](../providers/anthropic)——`cache_retention`、
  `CacheMarkerPolicy`、用于检视原始请求的 `on_payload` hook。
- [多轮对话](./multi-turn)——消息历史在轮次间如何增长，为什么追加
  语义很重要。
- [Tracing](../tracing/overview)——从 OTel span 读
  `cache_read.input_tokens`。
