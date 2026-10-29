---
title: Anthropic
description: "Use Anthropic Claude models with CubeLoop's AnthropicProvider — supports thinking, caching, and tool use."
---

# Anthropic Provider

`AnthropicProvider` wraps the official `anthropic` SDK against the
Messages API. It supports streaming, extended thinking, prompt
caching, and tool use.

## Construction

```python
from cubeloop.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(
    api_key="sk-ant-…",          # or read from os.environ["ANTHROPIC_API_KEY"]
    base_url=None,                # set to point at a proxy / Bedrock-compatible endpoint
    cache_retention="short",      # "short" (5 min, default) | "long" (1 h) | "none"
)
```

`api_key=None` lets the underlying SDK read `ANTHROPIC_API_KEY` from
the environment.

## The `Model`

```python
provider = AnthropicProvider(provider_id="anthropic")
model = provider.model(
    "claude-sonnet-4-6",
    reasoning=True,           # enables thinking levels (see below)
    max_tokens=8192,          # response cap
    context_window=200_000,   # hard model limit
    temperature=0.7,
)
```

The first argument is the model name exactly as you'd pass it to the
SDK. `provider_id` is a free-form label used by CubeLoop internals — keep
it stable across your codebase, and set it when you want tracing and
error messages to show a specific source label.

## Extended thinking (reasoning)

CubeLoop exposes a provider-independent `ReasoningControl(mode, effort,
summary)` and maps it onto Anthropic's `thinking` + `budget_tokens`:

| `effort` | Budget |
|---|---|
| `"minimal"` | 1024 |
| `"low"` | 2048 |
| `"medium"` | 8192 |
| `"high"` | 16384 |
| `"max"` | 16384 (Anthropic has no tier above `"high"`) |

`mode="off"` disables thinking; `mode="on"` (or `"auto"`) enables it at
the given `effort`. Set it per-agent:

```python
from cubeloop import ReasoningControl

agent = Agent(
    model=model,
    reasoning=ReasoningControl(mode="on", effort="medium"),
)
```

A non-reasoning model (`Model(reasoning=False)`) never receives an
enabled `thinking` payload, regardless of the requested `mode` — CubeLoop
clamps it to `"off"` for you.

To change the per-effort budgets, supply a
[`CapabilityDescriptor`](./overview) with a `ReasoningCapability` whose
`effort_values` map is the single source of truth for budget values:

```python
from cubeloop import CapabilityDescriptor, ReasoningCapability
from cubeloop.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(
    api_key="sk-ant-…",
    capability=CapabilityDescriptor(
        reasoning=ReasoningCapability(
            mode_payloads={
                "off": {"thinking": {"type": "disabled"}},
                "on": {"thinking": {"type": "enabled"}},
            },
            effort_path="thinking.budget_tokens",
            effort_values={
                "low": 4096, "medium": 12288, "high": 16384, "max": 16384,
            },
        ),
    ),
)
```

:::note Changed in 0.13
The `thinking` / `ThinkingLevel` / `ThinkingBudgets` API (and
`StreamOptions(thinking_budgets=…)`) is replaced by `ReasoningControl` +
`ReasoningCapability`. `Agent(thinking=…)` is now `Agent(reasoning=…)`.
:::

When reasoning is on, CubeLoop **omits `temperature`** because the
Anthropic API rejects non-default temperatures alongside extended
thinking ([feature compatibility](https://platform.claude.com/docs/en/build-with-claude/extended-thinking#feature-compatibility)).
Set `Model.temperature` to the value you want when reasoning is off;
CubeLoop handles the rest.

Thinking content streams as `thinking_start` / `thinking_delta` /
`thinking_end` events and ends up in `AssistantMessage.content` as
`ThinkingContent` blocks, preserved on subsequent turns so the model
keeps continuity.

## Prompt caching

By default the provider marks three cache breakpoints on each request:

- The **system prompt** (most stable).
- The **last tool definition** (changes rarely).
- The **last message** (the cache moves forward each turn so prior
  history stays warm).

Cache retention is `"short"` (5 minutes, free). Bump to `"long"` if
your turns are slower than that:

```python
AnthropicProvider(provider_id="anthropic", api_key=…, cache_retention="long")  # 1-hour TTL
AnthropicProvider(provider_id="anthropic", api_key=…, cache_retention="none")  # disable entirely
```

The `Usage` object on each `AssistantMessage` reports
`cache_read_tokens` and `cache_write_tokens` so you can see your hit
rate.

For custom cache strategies (a different breakpoint policy), implement
the `CacheMarkerPolicy` Protocol and pass `cache_policy=…`. The
default policy lives at `cubeloop.providers.anthropic.DefaultCacheMarkerPolicy`.

## Custom payloads with `on_payload`

`on_payload` lets you inspect or replace the request dict right before
it's sent:

```python
async def my_payload(payload, model):
    payload.setdefault("metadata", {})["user_id"] = "u-42"
    return payload     # return None or no return to keep the original

agent = Agent(model=model, on_payload=my_payload)
```

Use this for: adding `metadata.user_id` (for billing), forcing
beta-header flags, or tracking the exact payloads you sent for a
debug pane.

## Custom response handling with `on_response`

`on_response` fires after the HTTP response is received (status,
headers), before streaming begins:

```python
async def my_response(resp, model):
    if resp.status >= 400:
        logger.warning("bad status %s", resp.status)
    rate = resp.headers.get("anthropic-ratelimit-requests-remaining")
    if rate is not None:
        metrics.gauge("rate_remaining", int(rate))

agent = Agent(model=model, on_response=my_response)
```

Both callbacks may be sync or async.

## Pointing at Bedrock / Vertex / proxies

The Anthropic SDK accepts a `base_url`; CubeLoop forwards it:

```python
provider = AnthropicProvider(
    api_key="…",
    base_url="https://my-litellm.internal/v1",
)
```

For Bedrock specifically, use the `anthropic-bedrock` adapter directly
and inject it via a [custom provider](./custom).

## Common pitfalls

- **`temperature` ignored** — Expected. CubeLoop drops it when reasoning
  is on; that's an API constraint, not a bug.
- **`effort="max"` looks the same as `"high"`** — Anthropic doesn't
  expose a budget tier above `high`, so the built-in profile maps both
  to the same token budget.
- **Cache misses you didn't expect** — Caches are keyed by content +
  ttl. Changing the system prompt invalidates everything; changing the
  tool list invalidates from the tools onward. Make those two stable
  across turns to maximise hits.
- **`anthropic.RateLimitError`** — Propagates as a stream error event
  with the SDK's `str(exc)`. Catch in `agent_end` and decide whether
  to retry.

## See also

- [OpenAI Provider](./openai) — same protocol, different shape.
- [Providers Overview](./overview) — tune
  reasoning budgets and temperature handling as data.
- [Custom Provider](./custom) — wrap a non-built-in API.
- [Recipes → Multi-Provider Failover](../../recipes/multi-provider-failover)
  — fall back to OpenAI when Anthropic is down.
