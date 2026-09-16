---
title: Providers Overview
description: "Provider setup, capabilities, and presets in CubeLoop."
---

# Providers Overview

_Start here for provider setup in CubeLoop._

This page is the entry point for provider configuration. It covers the
default path for Anthropic and OpenAI, then explains how to describe
wire-level differences and preset overrides when you need to reach a
different backend without writing per-vendor glue. The progression is
deliberate:

1. **The default is zero config.** For Anthropic and OpenAI you write a
   provider and a model. Nothing on this page is required.
2. **For an off-default endpoint, describe the quirks as data.** A
   `CapabilityDescriptor` captures the differences declaratively — no
   subclassing, no forking.

## `provider.model(...)` parameters

Use `provider.model(model_id, ...)` to create a bound model for an agent.
`model_id` is required and positional; everything else is optional keyword
arguments:

- `api: str` — alternate API name/route tag for downstream integrations.
- `reasoning: bool` — enable reasoning for this model. A model bound with
  `reasoning=False` never receives reasoning fields on the wire, regardless
  of the agent's `ReasoningControl`.
- `context_window: int` — context-capacity hint used for validation and prompt planning.
- `max_tokens: int` — default max generation cap for this model.
- `temperature: float` — default sampling temperature for this model.
- `cost: ModelCost | None` — optional cost metadata object.

### Calling a bound model directly

`provider.model(...)` returns a `BoundModel` that you can invoke without
fishing the provider back out:

```python
from cubeloop.providers.base import TextContent, UserMessage

bound = provider.model("claude-sonnet-4-6")

# Single-shot call.
reply = await bound.generate(
    messages=[UserMessage(content=[TextContent(text="hi")])],
    system_prompt="Be brief.",
)

# Streaming.
stream = await bound.stream(messages=[...])
async for event in stream:
    ...
```

Both methods forward to the bound provider with `model=bound.spec` — useful
for utilities (summarizers, classifiers) where you already hold a
`BoundModel` and want to skip the agent loop.

## `CapabilityDescriptor` is what to use when behavior differs by backend

`CapabilityDescriptor` is passed to a provider to express wire differences for
all models served by that provider:

```python
from cubeloop import CapabilityDescriptor, ReasoningCapability
from cubeloop.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    api_key="...",
    base_url="https://api.deepseek.com",
    capability=CapabilityDescriptor(
        reasoning=ReasoningCapability(
            mode_payloads={
                "on": {"extra_body": {"thinking": True}},
                "off": {"extra_body": {"thinking": False}},
            },
        ),
        max_tokens_field="max_completion_tokens",
    ),
)
```

If only one model needs an override, use
`model_capability_overrides`:

```python
from cubeloop import CapabilityDescriptor, ReasoningCapability
from cubeloop.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    api_key="...",
    base_url="https://openrouter.ai/api/v1",
    capability=CapabilityDescriptor(
        reasoning=ReasoningCapability(
            mode_payloads={"on": {"extra_body": {"thinking": True}}},
        ),
    ),
    model_capability_overrides={
        "deepseek-r1": CapabilityDescriptor(
            reasoning=ReasoningCapability(
                mode_payloads={"on": {"extra_body": {"thinking": "enabled"}}},
            ),
        ),
    },
)
```

`model_capability_overrides` is matched by exact `model_id`.

`CapabilityDescriptor` supports these fields:

- `reasoning` (`ReasoningCapability | None`) — maps `ReasoningControl`
  (mode/effort/summary) onto this endpoint's wire payload.
- `temperature` (`TemperatureSpec`) — clip, force, or strip temperature.
- `max_tokens_field` — pick `max_tokens` or `max_completion_tokens`.
- `supports_tools` / `supports_images` / `supports_streaming` — metadata consumed
  by host UI and product code.

:::note Preset catalogs live in the host application
CubeLoop ships the **mechanism** (the `CapabilityDescriptor` and the wire
runtime that applies it), not a catalog of vendors. A ready-made list of
providers — base URLs, auth, regional/coding-plan endpoints, model lists —
is product data and belongs to the application embedding CubeLoop (for
example, cubebox maintains its own provider catalog). To reach a specific
vendor, build the provider with the right `base_url` + `CapabilityDescriptor`
as shown below.
:::

## 1. The simple case — no config at all

Most users never touch capabilities. The built-in providers ship with
sensible defaults:

```python
import cubeloop
from cubeloop import Agent
from cubeloop.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(provider_id="anthropic")  # reads ANTHROPIC_API_KEY
agent = Agent(model=provider.model("claude-sonnet-4-6"))
await agent.prompt("Hello!")
```

That's the whole setup. A provider built without `capability=` produces
byte-identical output to CubeLoop `0.4` — the machinery below only kicks in
when you ask for it.

## 2. Off-default endpoints — the CapabilityDescriptor

When you want a model that isn't OpenAI or Anthropic — DeepSeek, Qwen,
Doubao, an OpenRouter route, a local server — the awkward part is each
one's wire dialect (does it want `max_tokens` or `max_completion_tokens`?
how is reasoning toggled?). You don't subclass a provider; you describe the
quirks as a [`CapabilityDescriptor`](pathname:///pydoc/cubeloop/providers/capability.html)
and pass it in, along with the right `base_url` and provider class for the
endpoint's wire shape:

```python
import os
from cubeloop import CapabilityDescriptor, ReasoningCapability
from cubeloop.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    capability=CapabilityDescriptor(
        reasoning=ReasoningCapability(
            mode_payloads={
                "off": {"extra_body": {"reasoning": {"exclude": True}}},
                "on": {"extra_body": {"reasoning": {"exclude": False}}},
            },
        ),
    ),
)
```

Pick the provider class by the endpoint's wire dialect:

| Wire dialect | Provider class |
| --- | --- |
| `anthropic-messages` | `AnthropicProvider` |
| `openai-completions` | `OpenAIProvider` |
| `openai-responses` | `OpenAIResponsesProvider` |

Each field maps to one wire behavior, and an unset field does nothing — so
you only declare what's actually different.

### `max_tokens_field`

`"max_tokens"` (default) or `"max_completion_tokens"`. Some
OpenAI-compatible servers accept only one spelling; this renames the key
on the way out. **Effect:** wrong choice → the server ignores your output
cap or 400s.

### `temperature`

A `TemperatureSpec` controlling how the caller's temperature is treated:

```python
from cubeloop import TemperatureSpec

TemperatureSpec(mode="free", min=0.0, max=2.0, default=1.0)  # clamp into [min, max]
TemperatureSpec(mode="fixed", fixed_value=1.0)               # always overwrite
TemperatureSpec(mode="ignored")                              # drop the key
```

- **`free`** — the caller's value is clamped into `[min, max]`; if none was
  sent, nothing is written. **Effect:** protects against an out-of-range
  value the backend would reject.
- **`fixed`** — `fixed_value` always wins. **Effect:** use for models that
  permit only one temperature (e.g. some o-series reasoning models).
- **`ignored`** — the key is stripped entirely. **Effect:** for backends
  that 400 on any `temperature` while reasoning.

### Reasoning: `ReasoningCapability`

`ReasoningCapability` maps the provider-independent `ReasoningControl`
(`mode`, `effort`, `summary`) onto whatever fields the vendor expects.
`mode_payloads` is the on/off toggle — the payload for the request's
`ReasoningControl.mode` is deep-merged into the request:

```python
from cubeloop import CapabilityDescriptor, ReasoningCapability

CapabilityDescriptor(
    reasoning=ReasoningCapability(
        mode_payloads={
            "off": {"extra_body": {"enable_thinking": False}},
            "on": {"extra_body": {"enable_thinking": True}},
        },
    ),
)
```

The merge recurses into nested dicts; arrays are atomic; on a collision
the capability value wins. `mode_payloads["off"]` is applied even for a
model bound with `reasoning=False` (so a hybrid model's "disable
thinking" quirk always fires); anything else in `ReasoningCapability`
(effort, summary, `mode_payloads` for any mode other than `"off"`) is
skipped entirely for such a model.

Beyond on/off, `effort_path` + `effort_values` map a `ReasoningEffort`
(`minimal`/`low`/`medium`/`high`/`max`) onto a concrete wire value
written at a dotted path — a token budget, an effort string, or a
vendor-specific enum, depending on what you put in `effort_values`.
`summary_path` + `summary_values` do the same for `ReasoningSummary`.
You still need two call-site controls:

- set `reasoning=True` when binding the model (enable reasoning for that model)
- set the agent's `reasoning` argument to a `ReasoningControl(mode=..., effort=...)`
  (defaults to `mode="off"`).

```python
from cubeloop import (
    Agent,
    CapabilityDescriptor,
    ReasoningCapability,
    ReasoningControl,
)
from cubeloop.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    api_key="...",
    capability=CapabilityDescriptor(
        reasoning=ReasoningCapability(
            mode_payloads={"on": {"extra_body": {"reasoning": {"enabled": True}}}},
            effort_path="reasoning.effort",
            effort_values={
                "low": "low",
                "medium": "medium",
                "high": "high",
                "max": "high",
            },
        ),
    ),
)

agent = Agent(
    model=provider.model("deepseek-r1", reasoning=True),
    reasoning=ReasoningControl(mode="on", effort="high"),
)
```

```python
from cubeloop import ReasoningCapability

# A token budget (Anthropic).
ReasoningCapability(
    effort_path="thinking.budget_tokens",
    effort_values={"minimal": 1024, "low": 2048,
                   "medium": 8192, "high": 16384, "max": 16384},
)

# An effort string (OpenAI Responses).
ReasoningCapability(
    effort_path="reasoning.effort",
    effort_values={"low": "low", "medium": "medium",
                   "high": "high", "max": "xhigh"},
)

# A vendor-specific enum (Doubao's 3-state thinking) via mode_payloads.
ReasoningCapability(
    mode_payloads={
        "off": {"thinking": {"type": "disabled"}},
        "on": {"thinking": {"type": "enabled"}},
    },
)
```

**Effect:** an effort/summary value missing from the map is simply not
written, so the endpoint keeps its own default for that value.

### `supports_tools` / `supports_images` / `supports_streaming`

Declarative flags read by host applications and frontends (for example, to
grey out image upload). The providers themselves don't gate on them.

### Per-model overrides on a shared endpoint

One gateway (OpenRouter, LiteLLM, an internal proxy) often serves both
reasoning and non-reasoning models. `model_capability_overrides` maps a
`model_id` to a descriptor that **replaces** the base one for that model:

```python
provider = OpenAIProvider(
    api_key="…",
    base_url="https://openrouter.ai/api/v1",
    capability=base_cap,                        # default for unlisted models
    model_capability_overrides={
        "deepseek/deepseek-r1": reasoning_cap,  # this model only
    },
)
```

Resolution is exact-match on `model_id`; anything not listed falls back to
`capability`.

## Image providers

Image generation has its own provider surface (`cubeloop.providers.images`)
that follows the same conventions described above: `provider_id` on the
provider, `provider.model("id", ...)` factory, typed `ProviderError`
failures, and a capability descriptor for backend wire differences. See
[Image Generation](./image-generation) for the full guide.

## Failover chain

`FallbackBoundModel` wraps an ordered chain of `BoundModel` instances. On a
`RateLimited`, `ProviderUnavailable`, or `ContextLengthExceeded` error — or on
a first-event stream error — it transparently tries the next model:

```python
from cubeloop import FallbackBoundModel

model = FallbackBoundModel(
    chain=(
        anthropic.model("claude-opus-4-8"),   # primary
        openai.model("gpt-5"),                # fallback
    )
)
agent = Agent(model=model, ...)
```

`trigger_errors` is configurable; an optional `on_failover` callback fires on
each switchover for billing/metrics. See the
[Multi-Provider Failover recipe](../../recipes/multi-provider-failover) for the
full guide.

## See also

- [OpenAI Provider](./openai) — concrete OpenAI / OpenAI-compatible setup.
- [Anthropic Provider](./anthropic) — the token-budget reasoning shape in practice.
- [Writing a Custom Provider](./custom) — when the endpoint isn't even OpenAI/Anthropic-shaped.
- [Multi-Provider Failover](../../recipes/multi-provider-failover) — `FallbackBoundModel` in action.
- [API Reference → `cubeloop.providers`](../../api/cubeloop-providers).
