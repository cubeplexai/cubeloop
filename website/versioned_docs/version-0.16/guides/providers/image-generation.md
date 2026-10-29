---
title: Image Generation
description: "Generate images with CubeLoop image providers — OpenAI, Doubao Seedream, SiliconFlow, Together AI, and other OpenAI-compatible backends."
---

# Image Generation

CubeLoop's image-generation path mirrors the chat-provider conventions: a
provider holds the connection (`provider_id`, `api_key`, `base_url`,
`capability`), a model spec holds model-level defaults, and per-call work
goes through a typed `ImagesContext` plus an optional `ImagesOptions`
cross-cutting bag. Failures raise typed `ProviderError` subclasses; UI
hosts catch them like they would any chat error.

The single concrete provider class — `OpenAIImagesProvider` — reaches
multiple OpenAI-shape backends through an `ImagesCapabilityDescriptor`
that declares wire differences as data (size field name, count field
name, supports_seed/steps/guidance gates, …).

> **Async-task backends** (Aliyun Wanxiang, Google Imagen, Stability,
> Replicate, fal, FLUX official) follow a submit→poll→fetch pattern that
> is not modeled by the capability descriptor. Custom subclasses can
> implement them today; first-class async-task scaffolding is a Roadmap
> item — see the [bottom of this page](#roadmap).

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
    ImagesContext(prompt="A cute robot at sunrise"),
)

# result.stop_reason in {"stop", "aborted"}; failures raise ProviderError.
for block in result.output:
    print(block.type, block.media_type, len(block.source))
```

## `provider.model("id", ...)` — model factory

`provider.model(...)` builds an `ImagesModel`. The provider's
`provider_id` is propagated automatically so the model knows where it
came from (used for tracing, error messages, and response metadata).

| Parameter | Type | Effect |
|---|---|---|
| `id` (positional) | `str` | The wire model id (e.g. `"gpt-image-1"`) |
| `api` | `str` | Routing tag (e.g. `"openai-images"`) |
| `default_size` | `str \| None` | Used when `ImagesContext.size` is `None` |
| `default_n` | `int \| None` | Used when `ImagesContext.n` is `None` |
| `default_quality` | `Literal["low","medium","high"] \| None` | Default when context omits it |
| `default_output_format` | `Literal["png","jpeg","webp"] \| None` | Default output format |
| `cost` | `ImagesCost \| None` | Per-image / per-megapixel pricing metadata |
| `max_input_images` | `int \| None` | Edit-path cap; meaningful only when the capability supports edit |

## `ImagesContext` — per-call payload

```python
ctx = ImagesContext(
    prompt="A robot",
    size="1024x1024",
    n=2,
    quality="high",
    output_format="png",
    seed=42,                # only written if capability.supports_seed
    negative_prompt="...",  # only written if capability.supports_negative_prompt
    steps=20,               # only written if capability.supports_steps
    guidance=7.5,           # only written if capability.supports_guidance
    extra={"watermark": False},  # always written verbatim
    input_images=[...],     # ImageContent list; triggers edit path
)
```

Field merge rule: `ctx.<field>` wins over `model.default_<field>`; if both
are `None`, the field is omitted from the wire payload (the backend uses
its own default). Value semantics — `"1024x1024"` vs `"1K"` vs `"1:1"` —
remain the user's responsibility; the capability descriptor only renames
the wire **key**.

## `ImagesOptions` — per-call cross-cutting

```python
from cubeloop.providers.images import ImagesOptions

opts = ImagesOptions(
    signal=cancel_event,         # asyncio.Event; set to abort
    on_payload=lambda p, m: p,   # pre-send payload mutator (per-call)
    on_response=lambda r, m: None,  # response observer (per-call)
)
```

When `signal` is set mid-call, the SDK request is cancelled and the
provider returns `AssistantImages(stop_reason="aborted", output=[])`.
The cancellation does not escape — external `task.cancel()` and
`asyncio.wait_for(...)` cancellations still propagate as
`asyncio.CancelledError`, only the signal-driven abort is converted.

Response observers see the abort via an `ImagesAborted` exception (a
plain `Exception` subclass, intentionally distinct from
`asyncio.CancelledError` so async listeners still run on the awaited
fast-path before the call returns):

```python
from cubeloop.providers.images import ImagesAborted

def on_response(body, model, exc):
    if isinstance(exc, ImagesAborted):
        ...    # signal-driven abort — body is None
    elif exc is not None:
        ...    # real failure (a ProviderError subclass)
    else:
        ...    # success — body holds the assembled response
```

`on_payload` and `on_response` are per-call hooks; for persistent
observers (tracing, audit), use `provider.subscribe_request()` /
`provider.subscribe_response()` — see [Observability](#observability).

## `ImagesCapabilityDescriptor` — reach other OpenAI-shape backends

Different OpenAI-shape backends use different field names. The
descriptor lets one `OpenAIImagesProvider` reach all of them:

### Volcengine Ark / Doubao Seedream

Mostly OpenAI-compatible with a `watermark` extra and `seed` support:

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

OpenAI-shape URL but field names differ:

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
        output_format_field=None,    # not supported
    ),
)
```

### Together AI — FLUX schnell

FLUX schnell uses `aspect_ratio` instead of `size`:

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

### Mixed-model gateways

When one gateway serves multiple models with different shapes, use
`model_capability_overrides`:

```python
provider = OpenAIImagesProvider(
    provider_id="together",
    api_key="...",
    base_url="https://api.together.xyz/v1",
    capability=together_pro_cap,       # default
    model_capability_overrides={
        "black-forest-labs/FLUX.1-schnell": together_schnell_cap,
    },
)
```

Resolution is exact-match on `model.id`; unmatched models fall back to
the base `capability`.

## Error handling

All built-in image providers raise typed `cubeloop.errors.ProviderError`
subclasses on failure — never in-band error strings:

```python
from cubeloop.errors import RateLimited, ProviderAuthFailed, ProviderUnavailable

try:
    result = await provider.generate_images(model, ctx)
except RateLimited as exc:
    # exc.retry_after may be populated
    ...
except ProviderAuthFailed:
    ...
except ProviderUnavailable:
    # 5xx, timeout, network — typically retryable
    ...
```

`AssistantImages.stop_reason` is now only `"stop"` (success) or
`"aborted"` (signal-triggered cancel). There is no `"error"` value and
no `error_message` field.

## Observability

Persistent observers register on the provider:

```python
provider.subscribe_request(lambda payload, model: log_payload(payload))
provider.subscribe_response(lambda body, model, exc: log_response(body, exc))
```

- `subscribe_request` fires once per call, just before the SDK send, with
  the final assembled payload dict (after `on_payload` mutators).
- `subscribe_response` fires once per call in the provider's `finally`
  block, with the assembled response body (or `None` on failure) and the
  exception (or `None` on success).

There is **no** `subscribe_chunk` — image generation is one-shot.

## Edit path

Passing `input_images` triggers the edit path when the capability
supports it:

```python
import base64
from cubeloop.providers.base import ImageContent

with open("source.png", "rb") as fh:
    source_b64 = base64.b64encode(fh.read()).decode("ascii")

ctx = ImagesContext(
    prompt="Make it brighter and warmer.",
    input_images=[ImageContent(source=source_b64, media_type="image/png")],
)
result = await provider.generate_images(model, ctx)
```

Setting `capability=ImagesCapabilityDescriptor(supports_edit=False)`
falls back to the generate path even when `input_images` is non-empty —
useful when targeting a backend whose model can't edit.

## Faux provider for tests

```python
from cubeloop.providers.images import FauxImagesProvider
from cubeloop.errors import RateLimited

# Happy path:
provider = FauxImagesProvider(png_b64="iVBORw0KGgo...")

# Inject an error to exercise retry middleware:
provider = FauxImagesProvider(
    png_b64="iVBORw0KGgo...",
    raise_on_call=RateLimited,
)
```

`FauxImagesProvider` inherits the listener registry, `.model()` factory,
and `provider_id` propagation from `BaseImagesProvider`, so tests that
exercise observability against the image path can use it interchangeably
with `OpenAIImagesProvider`.

## Roadmap

- **Async-task backends.** Aliyun Wanxiang, Google Imagen on Vertex,
  Stability, Replicate, fal, FLUX official all follow a submit→poll→fetch
  pattern that this version does not model first-class. Subclasses of
  `BaseImagesProvider` can implement them by hand today; a future release
  will likely add an `AsyncTaskImagesProvider` base with shared polling
  scaffolding.
- **Tracing wiring.** This release adds the listener registry on image
  providers, but `cubeloop.tracing` does not yet auto-subscribe to image
  calls. Hosts that want image-call spans should subscribe manually for
  now.

## See also

- [Providers Overview](./overview) — chat-provider configuration; image
  providers follow the same `provider_id` / `.model()` / capability
  conventions.
- [OpenAI Provider](./openai) — shared OpenAI-shape patterns on the chat
  side.
- [API Reference → `cubeloop.providers.images`](../../api/cubeloop-providers).
