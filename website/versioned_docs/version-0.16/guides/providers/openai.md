---
title: OpenAI
description: "Use OpenAI GPT models with CubeLoop's OpenAIProvider — Chat Completions API integration."
---

# OpenAI Provider

CubeLoop ships two OpenAI providers covering the two API surfaces:

- **`OpenAIProvider`** — Chat Completions API
  (`/v1/chat/completions`). Use this for the GPT-4/5 family and most
  OpenAI-compatible servers (vLLM, LiteLLM, DeepSeek, Qwen, MiniMax,
  DouBao, …).
- **`OpenAIResponsesProvider`** — Responses API
  (`/v1/responses`). Use this when you want server-side state and
  reasoning summaries.

Both implement the same `Provider` protocol; pick one per agent.

## Chat Completions: `OpenAIProvider`

```python
from cubeloop.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    provider_id="openai",
    api_key="sk-…",      # or reads OPENAI_API_KEY
    base_url=None,        # set for OpenAI-compatible servers
    extra_body=None,      # merged into every request
    extra_headers=None,
)

model = provider.model(
    "gpt-5",
    reasoning=True,        # enables reasoning-effort mapping
    max_tokens=8192,
    context_window=128_000,
)
```

### Reasoning on Chat Completions

OpenAI exposes reasoning content through `delta.reasoning_content` on
o-series and gpt-5 models. CubeLoop captures it as `ThinkingContent` and
emits `thinking_*` events identically to Anthropic. Control it with
`ReasoningControl`:

```python
from cubeloop import ReasoningControl

agent = Agent(
    model=model,
    reasoning=ReasoningControl(mode="on", effort="high"),
)
```

`effort` accepts `"minimal" | "low" | "medium" | "high" | "max"` and is
written to `reasoning_effort` on the wire (`"max"` maps to OpenAI's
`"xhigh"` tier). A model bound with `reasoning=False` never receives a
`reasoning_effort` field, regardless of the requested mode.

Many OpenAI-compatible OSS backends emit reasoning under different
fields. CubeLoop understands three in priority order:

1. `delta.reasoning_content` (DeepSeek, Qwen, DouBao)
2. `delta.reasoning` (vLLM)
3. `delta.reasoning_details` (MiniMax)

No configuration needed — the provider picks whichever field is
present.

### `extra_body` for OSS quirks

Most OpenAI-compatible servers accept extensions through the request
body. Set them once at construction:

```python
provider = OpenAIProvider(
    api_key="…",
    base_url="https://api.deepseek.com/v1",
    extra_body={"enable_thinking": True, "stream_options": {"include_usage": True}},
)
```

If you need per-request mutation, use `on_payload` (see below).

### Capability descriptor

Wire-shape differences between OpenAI and OpenAI-compatible backends
(e.g. `max_tokens` vs `max_completion_tokens`, reasoning field names,
temperature handling) are configured through a
[`CapabilityDescriptor`](pathname:///pydoc/cubeloop/providers/capability.html)
passed at construction. For example, `max_tokens_field="max_completion_tokens"`
renames the key on the way out. See [Providers Overview](./overview)
for the full set of knobs (CubeLoop `0.5+`).

### Pointing at vLLM / LiteLLM / DeepSeek

```python
provider = OpenAIProvider(
    api_key="dummy",                                    # vLLM ignores it
    base_url="http://localhost:8000/v1",
    extra_headers={"Authorization": "Bearer dummy"},
)
```

For LiteLLM:

```python
provider = OpenAIProvider(
    api_key=os.environ["LITELLM_KEY"],
    base_url="https://litellm.internal/v1",
)
```

For the wire quirks of these backends (reasoning toggles, token-field
renames, …), see [Providers Overview](./overview).

## Responses API: `OpenAIResponsesProvider`

```python
from cubeloop.providers.openai_responses import OpenAIResponsesProvider

provider = OpenAIResponsesProvider(provider_id="openai_responses", api_key="sk-…")
model = provider.model("gpt-5", reasoning=True)
```

The Responses API keeps state server-side (referenced by
`previous_response_id`). CubeLoop tracks `AssistantMessage.response_id`
and feeds it back automatically — your code looks identical to the
Chat Completions path.

Use the Responses provider when:

- You want reasoning **summaries** (not just text) surfaced as
  thinking blocks.
- You're using the `o`-series and want the server to hold the
  reasoning chain across turns (smaller payloads, faster reuse).

Stay on `OpenAIProvider` when you want full control over the message
list and prompt caching strategy.

## `on_payload` / `on_response`

Same shape as the [Anthropic](./anthropic) provider. The payload dict
differs (`messages` instead of `messages` + `system` separately,
OpenAI-style `tools` schema), so inspect it once before mutating.

```python
async def add_user_metadata(payload, model):
    payload["user"] = "u-42"     # billable user attribution
    return payload

agent = Agent(model=model, on_payload=add_user_metadata)
```

## Tool calling

Tool definitions are auto-converted to OpenAI's
`{"type": "function", "function": {...}}` shape. The streaming format
emits incremental JSON arguments under `toolcall_delta`; CubeLoop
buffers and parses them through
[`cubeloop.utils.json_parse.parse_streaming_json`](../../api/cubeloop-utils)
so partials always validate to the closest well-formed object.

Multiple parallel tool calls in one assistant message just work —
they're routed through the same parallel executor as the Anthropic
provider.

## Common pitfalls

- **`stream_options.include_usage` rejected** — Some compatibles
  reject the whole `stream_options` field. **`on_payload` cannot fix
  this**: CubeLoop 0.3 calls `kwargs.setdefault("stream_options", {})`
  *after* your callback runs, so deleting the key in `on_payload` is
  silently undone. Workarounds:
  - Subclass `OpenAIProvider` and override `stream()` to skip the
    `setdefault` for your backend.
  - Set `include_usage=False` in `on_payload` (the field still goes
    out, but is usually accepted as a no-op even by strict
    backends).
  - Use a [`CapabilityDescriptor`](./overview) (CubeLoop `0.5+`) to
    describe your backend's reasoning wiring declaratively.
- **Thinking events but no `thinking_*` events** — Your backend
  surfaces reasoning under a non-standard field. Either add a fourth
  branch via PR or transcode it with `on_payload`.
- **Mixed providers in one process** — Each provider holds its own
  HTTP client. Reuse a single instance per `(base_url, api_key)`
  pair instead of creating one per agent.
- **Usage shows 0 input tokens** — Most compatibles omit usage
  entirely or only emit it on the final chunk. Inspect the trailing
  chunk in `on_payload` for a hint, or treat token counts as
  best-effort on those backends.

## See also
- [Image Generation](./image-generation) — generate images with `openai-images` and image editing.

- [Anthropic Provider](./anthropic) — the other built-in.
- [Custom Provider](./custom) — write your own from scratch.
- [Recipes → Multi-Provider Failover](../../recipes/multi-provider-failover)
  — combine both providers for resilience.
