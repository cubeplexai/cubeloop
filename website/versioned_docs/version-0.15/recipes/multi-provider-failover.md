---
title: Multi-Provider Failover
description: "Automatic failover between LLM providers using FallbackBoundModel."
---

# Recipe: Multi-Provider Failover

When the primary provider is rate-limited, unavailable, or has hit its context
limit, fall over to the next one automatically — without crashing the agent.
CubeLoop ships `FallbackBoundModel` for this out of the box.

**Time to read:** 5 minutes.
**Deps:** `cubeloop`, API keys for two providers.

## The built-in: `FallbackBoundModel`

```python
import os
from cubeloop import Agent, FallbackBoundModel
from cubeloop.providers.anthropic import AnthropicProvider
from cubeloop.providers.openai import OpenAIProvider

anthropic = AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"])
openai = OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])

model = FallbackBoundModel(
    chain=(
        anthropic.model("claude-opus-4-8"),   # primary
        openai.model("gpt-5"),                # fallback
    )
)

agent = Agent(model=model, system_prompt="You answer concisely.")
await agent.prompt("Capital of Mongolia?")
```

`FallbackBoundModel` peeks at the first stream event from each provider. Typed
error fields on that event (`error_type`, `error_code`, `status_code`, …) use
the **same predicate** as a raised exception. Once a non-error first event
arrives the stream is forwarded as-is — mid-stream errors are not retried.

Policy in one turn:

1. **Same-model retry** (default 3 retries / 4 attempts) for transient
   `RateLimited` and `ProviderUnavailable` only. `RateLimited.retry_after` is
   honoured, capped by `max_retry_after` (default 5s).
2. **Then hop** to the next chain entry. Residual `ProviderBadRequest` and
   `ModelNotFound` hop immediately (no same-model retry) — classification
   misses model-specific 400s more often than a true schema bug that fails
   every provider.
3. **Stick** to the first *successful* leg for later `stream` / `generate`
   calls in the same `Agent.prompt`. HITL `respond()` continues that run and
   keeps the sticky index. The next user `prompt` resets to `chain[0]`.
   Subagents start their own run and do **not** inherit the parent's index.
   Sticky requires `begin_fallback_run()` (Agent does this); standalone
   `FallbackBoundModel` calls do not stick.

## Default trigger conditions

By default **failover** is triggered on:

| Error | Same-model retry | Hop |
|---|---|---|
| `RateLimited` | Yes | After retries |
| `ProviderUnavailable` | Yes | After retries |
| `ContextLengthExceeded` | No | Yes, but skip legs whose `context_window` cannot fit `tokens_in` |
| `ModelNotFound` | No | Yes |
| `ProviderBadRequest` (residual 4xx) | No | Yes |
| `ProviderAuthFailed` | No | No (fail-closed) |
| `ContentFiltered` | No | No (fail-closed; opt in via `trigger_errors`) |

This is an intentional change from earlier releases: residual 4xx used to
hard-fail the turn. A true schema bug still exhausts the chain quickly; the
aggregated `ProviderUnavailable.errors` list keeps every leg.

## Custom trigger conditions

Pass `trigger_errors` to override:

```python
from cubeloop import FallbackBoundModel
from cubeloop.errors import ProviderAuthFailed, ProviderUnavailable, RateLimited

model = FallbackBoundModel(
    chain=(primary, fallback),
    trigger_errors=frozenset({RateLimited, ProviderUnavailable, ProviderAuthFailed}),
)
```

## Monitoring failovers

Pass `on_failover` to hook into billing or alerting:

```python
import logging

log = logging.getLogger(__name__)

async def record_failover(failed, next_model, error):
    log.warning(
        "provider failover: %s/%s → %s/%s (%s)",
        failed.spec.provider_id, failed.spec.id,
        next_model.spec.provider_id if next_model else "none",
        next_model.spec.id if next_model else "—",
        error,
    )
    # e.g. await billing.record_fallback_failure(failed.spec, error)

model = FallbackBoundModel(
    chain=(primary, fallback),
    max_retries_per_model=3,   # 3 retries after the first failure (4 attempts)
    max_retry_after=5.0,
    on_failover=record_failover,
)
```

`on_failover` receives `(failed: BoundModel, next_model: BoundModel | None,
error: BaseException | str)` and fires **only on a chain hop**, not on each
same-model retry. Optional `on_retry(failed, error, attempt, wait_s)` covers
retries. Both sync and async callables are accepted. Exceptions raised inside
either callback are logged and swallowed.

When every leg fails, CubeLoop raises `ProviderUnavailable` whose `.errors`
list holds the per-leg failures (typed when possible). `__cause__` is the
last typed error.

## `provider` and `spec` always reflect the primary

`FallbackBoundModel.provider` and `.spec` proxy `chain[0]`. Tracing and
billing code that reads `agent._model.provider` or `agent._model.spec` sees
the primary — which is the intended provider. The `AssistantMessage` returned
by the actual call carries `provider_id` and `model_id` of whichever model
really responded.

## Common pitfalls

- **Different tool schemas across providers** — Both built-in providers accept
  the same `ToolDefinition`, but vendor-specific extras (e.g. OpenAI
  `parallel_tool_calls=False`) won't carry to Anthropic. Keep cross-provider
  behaviour in `transform_context` middleware, not in `extra_body`.
- **Different cost** — Failover changes per-token cost. Track which provider
  answered via `AssistantMessage.provider_id` and bill accordingly; the
  `on_failover` callback is the right place to record the switch.
- **Mid-stream errors aren't retried** — Once a healthy first event arrives,
  `FallbackBoundModel` commits to that provider. Errors during the rest of
  the stream are forwarded to the agent as-is.
- **`ContextLengthExceeded` skips legs that cannot fit** — if `tokens_in` is
  known and the next model's `context_window` is smaller, that leg is not
  called. Pair a standard-window primary with a large-context fallback.
- **Sticky is per run, not a long-term preference** — the next
  `Agent.prompt` starts at the user-selected primary again. Do not persist
  the active index unless the product layer opts in.

## Run the example

A runnable version is in the repository:

```bash
git clone https://github.com/cubeplexai/cubeloop && cd cubeloop
uv sync

export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY [+ OPENAI_BASE_URL]
uv run python examples/multi_provider_failover.py
```

The example deliberately uses a bad key for the primary. Auth is fail-closed
by default, so the example **opts in** via `trigger_errors` to include
`ProviderAuthFailed`, then answers correctly via the real fallback.

## See also

- [Providers Overview](../guides/providers/overview) — provider setup and `CapabilityDescriptor`.
- [Providers / Anthropic](../guides/providers/anthropic) and [OpenAI](../guides/providers/openai) — provider-specific details.
- [Writing a Custom Provider](../guides/providers/custom) — when neither built-in provider fits.
