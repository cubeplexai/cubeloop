---
title: 图片生成
description: "用 CubeLoop 的 image provider 生成图片 —— OpenAI、豆包 Seedream、SiliconFlow、Together AI 等 OpenAI 兼容后端。"
---

# 图片生成

CubeLoop 的图片生成路径与 chat provider 范式同构：Provider 装连接信息
（`provider_id`、`api_key`、`base_url`、`capability`），model spec 装模型级
默认值，per-call 走类型化的 `ImagesContext` 加可选的 `ImagesOptions`
跨切面选项。调用失败抛出类型化的 `ProviderError` 子类——UI 端跟 chat
错误一样 catch 即可。

唯一一个具体 provider 类 `OpenAIImagesProvider` 通过
`ImagesCapabilityDescriptor` 把后端的字段差异**声明为数据**，从而能打多个
OpenAI 形态的后端（OpenAI 官方、豆包 Seedream、SiliconFlow、Together AI 等）。

> **异步任务后端**（阿里万相、Google Imagen、Stability、Replicate、
> fal、FLUX 官方）走的是 submit→poll→fetch 模式，不在本版 capability
> descriptor 的覆盖范围内。今天可以通过继承 `BaseImagesProvider` 自定义实现；
> 一线异步后端的统一脚手架是 [Roadmap 项](#roadmap)。

## Quickstart — OpenAI

```python
import os
from cubeloop.providers.images import OpenAIImagesProvider, ImagesContext

provider = OpenAIImagesProvider(
    provider_id="openai",
    api_key=os.environ["OPENAI_API_KEY"],
)
model = provider.model(
    "gpt-image-1",
    default_size="1024x1024",
    default_quality="high",
)

result = await provider.generate_images(
    model,
    ImagesContext(prompt="一只黎明时分的小机器人"),
)

# result.stop_reason ∈ {"stop", "aborted"}；失败会抛 ProviderError。
for block in result.output:
    print(block.type, block.media_type, len(block.source))
```

## `provider.model("id", ...)` — 模型工厂

`provider.model(...)` 返回 `ImagesModel`。provider 的 `provider_id` 会自动
拷进 model（用于 tracing、错误信息、response metadata）。

| 参数 | 类型 | 作用 |
|---|---|---|
| `id`（位置参数） | `str` | wire 上的模型 ID（如 `"gpt-image-1"`） |
| `api` | `str` | 路由 tag（如 `"openai-images"`） |
| `default_size` | `str \| None` | `ImagesContext.size` 为 `None` 时使用 |
| `default_n` | `int \| None` | `ImagesContext.n` 为 `None` 时使用 |
| `default_quality` | `Literal["low","medium","high"] \| None` | context 未指定时的默认值 |
| `default_output_format` | `Literal["png","jpeg","webp"] \| None` | 默认输出格式 |
| `cost` | `ImagesCost \| None` | 单图/百万像素计价元数据 |
| `max_input_images` | `int \| None` | 编辑路径输入图上限；仅 capability 支持 edit 时有意义 |

## `ImagesContext` — per-call 请求负载

```python
ctx = ImagesContext(
    prompt="一只机器人",
    size="1024x1024",
    n=2,
    quality="high",
    output_format="png",
    seed=42,                # 仅 capability.supports_seed=True 时写入
    negative_prompt="...",  # 仅 capability.supports_negative_prompt=True 时写入
    steps=20,               # 仅 capability.supports_steps=True 时写入
    guidance=7.5,           # 仅 capability.supports_guidance=True 时写入
    extra={"watermark": False},  # 始终透传
    input_images=[...],     # ImageContent 列表，触发 edit 路径
)
```

字段合并规则：`ctx.<字段>` 优先于 `model.default_<字段>`；都为 `None` 时
该字段不写进 payload（后端用自家默认值）。值的语义——`"1024x1024"` /
`"1K"` / `"1:1"` 怎么写——仍是用户自己的责任；capability 只换 wire 上的
**字段名**。

## `ImagesOptions` — per-call 跨切面选项

```python
from cubeloop.providers.images import ImagesOptions

opts = ImagesOptions(
    signal=cancel_event,         # asyncio.Event；set 后中止当前调用
    on_payload=lambda p, m: p,   # 发包前的 payload mutator（per-call）
    on_response=lambda r, m: None,  # response observer（per-call）
)
```

中途 `signal.set()` 时，SDK 请求被取消，provider 返回
`AssistantImages(stop_reason="aborted", output=[])`，不向外抛出。
外部的 `task.cancel()` / `asyncio.wait_for(...)` 取消仍以
`asyncio.CancelledError` 形式正常传播——只有 signal 触发的 abort 会被转换。

Response 观察者通过 `ImagesAborted` 异常（普通 `Exception` 子类，**故意不**
继承 `asyncio.CancelledError`，这样 async listener 还能走 awaited 快路径
在 generate 返回前完成）来看到 abort：

```python
from cubeloop.providers.images import ImagesAborted

def on_response(body, model, exc):
    if isinstance(exc, ImagesAborted):
        ...    # signal 触发的 abort —— body 为 None
    elif exc is not None:
        ...    # 真实失败（ProviderError 子类）
    else:
        ...    # 成功 —— body 是已拼好的响应
```

`on_payload` / `on_response` 是 per-call hook；对于持久观察者（tracing、
审计），用 `provider.subscribe_request()` / `provider.subscribe_response()`
—— 见 [可观察性](#可观察性)。

## `ImagesCapabilityDescriptor` —— 对接其它 OpenAI 形态后端

不同的 OpenAI 形态后端字段名不同，descriptor 让同一个
`OpenAIImagesProvider` 都能打：

### 火山方舟豆包 Seedream

基本 OpenAI 兼容，多了 `watermark` 扩展和 `seed` 支持：

```python
OpenAIImagesProvider(
    provider_id="doubao",
    api_key=os.environ["ARK_API_KEY"],
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    capability=ImagesCapabilityDescriptor(
        supports_seed=True,
        extra_payload={"watermark": False},
    ),
)
```

### SiliconFlow

URL 长得像 OpenAI，但字段名要换：

```python
from cubeloop.providers.images.capability import ImagesCapabilityDescriptor, SizeSpec

OpenAIImagesProvider(
    provider_id="siliconflow",
    api_key=os.environ["SILICONFLOW_API_KEY"],
    base_url="https://api.siliconflow.cn/v1",
    capability=ImagesCapabilityDescriptor(
        size_spec=SizeSpec(kind="image_size_string"),
        count_field="batch_size",
        supports_seed=True,
        supports_steps=True, steps_field="num_inference_steps",
        supports_guidance=True, guidance_field="guidance_scale",
        supports_negative_prompt=True,
        output_format_field=None,    # 该后端不支持
    ),
)
```

### Together AI — FLUX schnell

FLUX schnell 用 `aspect_ratio` 不是 `size`：

```python
OpenAIImagesProvider(
    provider_id="together",
    api_key=os.environ["TOGETHER_API_KEY"],
    base_url="https://api.together.xyz/v1",
    capability=ImagesCapabilityDescriptor(
        size_spec=SizeSpec(kind="aspect_ratio"),
        supports_seed=True,
        supports_steps=True, steps_field="steps",
    ),
)
```

### 多模型混合网关

一个网关服务多个不同形态的模型时，用 `model_capability_overrides`：

```python
provider = OpenAIImagesProvider(
    provider_id="together",
    api_key="...",
    base_url="https://api.together.xyz/v1",
    capability=together_pro_cap,       # 默认
    model_capability_overrides={
        "black-forest-labs/FLUX.1-schnell": together_schnell_cap,
    },
)
```

按 `model.id` 精确匹配；未匹配的退回到基础 `capability`。

## 错误处理

所有内置 image provider 在失败时抛出类型化的 `cubeloop.errors.ProviderError`
子类——不再用 in-band 错误字符串：

```python
from cubeloop.errors import RateLimited, ProviderAuthFailed, ProviderUnavailable

try:
    result = await provider.generate_images(model, ctx)
except RateLimited as exc:
    # exc.retry_after 可能有值
    ...
except ProviderAuthFailed:
    ...
except ProviderUnavailable:
    # 5xx / timeout / network——通常可重试
    ...
```

`AssistantImages.stop_reason` 现在只有 `"stop"`（成功）和 `"aborted"`
（信号触发的中止）。没有 `"error"` 值，也没有 `error_message` 字段。

## 可观察性

持久观察者注册在 provider 上：

```python
provider.subscribe_request(lambda payload, model: log_payload(payload))
provider.subscribe_response(lambda body, model, exc: log_response(body, exc))
```

- `subscribe_request` 每次调用 SDK 发包前触发一次，拿到最终拼好的
  payload dict（`on_payload` mutator 之后的版本）。
- `subscribe_response` 每次调用结束在 `finally` 块触发一次，拿到 response
  body（失败时为 `None`）和异常（成功时为 `None`）。

**没有** `subscribe_chunk`——图片生成是 one-shot。

## 编辑路径

传入 `input_images` 触发 edit 路径（前提是 capability 声明支持）：

```python
import base64
from cubeloop.providers.base import ImageContent

with open("source.png", "rb") as fh:
    source_b64 = base64.b64encode(fh.read()).decode("ascii")

ctx = ImagesContext(
    prompt="调亮、调暖一点。",
    input_images=[ImageContent(source=source_b64, media_type="image/png")],
)
result = await provider.generate_images(model, ctx)
```

设 `capability=ImagesCapabilityDescriptor(supports_edit=False)` 即使
`input_images` 非空也回到 generate 路径——目标模型不支持编辑时用。

## 测试 stub `FauxImagesProvider`

```python
from cubeloop.providers.images import FauxImagesProvider
from cubeloop.errors import RateLimited

# 正常路径：
provider = FauxImagesProvider(png_b64="iVBORw0KGgo...")

# 注入错误（测重试中间件）：
provider = FauxImagesProvider(
    png_b64="iVBORw0KGgo...",
    raise_on_call=RateLimited,
)
```

`FauxImagesProvider` 继承 `BaseImagesProvider`，自带 listener 注册表、
`.model()` 工厂和 `provider_id` 传播，所以涉及可观察性的测试可以跟
`OpenAIImagesProvider` 互换使用。

## Roadmap

- **异步任务后端**：阿里万相、Google Imagen on Vertex、Stability、Replicate、
  fal、FLUX 官方——这些走的是 submit→poll→fetch 模式，本版没做一级支持。
  今天可以继承 `BaseImagesProvider` 自己实现；未来版本会加 `AsyncTaskImagesProvider`
  这种共享 polling 脚手架。
- **Tracing 接入**：本版加了 image provider 的 listener 注册表，但
  `cubeloop.tracing` 还没自动订阅 image 调用。需要 image 调用 span 的
  Host 暂时手动 `subscribe_*`。

## 另见

- [Providers Overview](./overview) —— chat-provider 的配置方式；image
  provider 跟它共享 `provider_id` / `.model()` / capability 范式。
- [OpenAI Provider](./openai) —— chat 那边 OpenAI 形态的共通配置。
- [API Reference → `cubeloop.providers.images`](../../api/cubeloop-providers)。
