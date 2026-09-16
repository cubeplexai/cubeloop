---
title: Prompt Caching
description: "How CubeLoop maximises prompt cache hit rates — append-only message storage, automatic cache breakpoints, and how to avoid breaking the cache."
---

# Prompt Caching

Prompt caching lets LLM providers reuse the KV computation for the
stable prefix of a request instead of reprocessing it on every turn.
The savings are significant: Anthropic charges cached tokens at
**10% of the normal input price**, and cache hits also reduce
time-to-first-token.

CubeLoop is designed from the ground up to maximise cache hit rates.
This guide explains how caching works, why CubeLoop's architecture
keeps the cache warm, and what you can do (or avoid doing) to
preserve those hits.

---

## How prompt caching works

Every LLM API request is a sequence of content blocks: system prompt,
tool definitions, then the conversation history. The provider hashes
the byte content of each block. If an incoming request shares a
prefix with a recently-seen request up to the TTL, the cached KV
state for that prefix is reused — the provider only processes the
new tokens beyond the cache boundary.

```
Request N:   [system] [tools] [msg 1] [msg 2] [msg 3]
                  ↑         ↑             ↑
             cached     cached        cached      ← all hits on turn N+1
                                              [msg 4]  ← new tokens only
```

Two things are required for a cache hit:

1. **The prefix is byte-identical** to the previous request. Any
   change — reordering, reformatting, adding a character — is a miss.
2. **The TTL hasn't expired.** Anthropic offers 5-minute (`"short"`)
   and 1-hour (`"long"`) retention; OpenAI caches automatically for
   prompts over 1 024 tokens.

---

## Why append-only storage matters

Many agent frameworks rebuild the full message list from a checkpoint
snapshot on every turn. If serialisation details change — dict key
ordering, whitespace, a timestamp field — the resulting byte sequence
differs from the previous request and **every cache breakpoint from
that point on misses**.

CubeLoop's checkpointer is append-only: it only ever adds new messages
to the end of the thread. The in-memory `agent.state.messages` list
grows by appending; it is never rebuilt or reshuffled. This means the
prefix of every request is guaranteed to be byte-identical to the
previous turn, which is the most fundamental requirement for cache
hits.

---

## Anthropic: automatic cache breakpoints

`AnthropicProvider` inserts `cache_control` markers automatically
via `DefaultCacheMarkerPolicy`. By default, three breakpoints are
placed on each request:

| Breakpoint | What gets cached | Why here |
|---|---|---|
| System prompt | The full system prompt block | Most stable content — rarely changes across turns |
| Last tool definition | All tool schemas up to and including the last one | Tool lists change infrequently |
| Last message in history | All prior conversation history | Moves forward each turn; prior turns stay warm |

On turn N+1, the first two breakpoints are identical to turn N (same
system prompt, same tools), so they get cache hits. The
last-message breakpoint has moved one message forward, so it writes
a new cache entry covering the slightly-longer history — and that
entry will be a hit on turn N+2.

### Configuration

```python
from cubeloop.providers.anthropic import AnthropicProvider

# Default: short TTL (5 min), automatic breakpoints
provider = AnthropicProvider(api_key="…")

# Long TTL (1 h) — use when turns are slow or infrequent
provider = AnthropicProvider(api_key="…", cache_retention="long")

# Disable caching entirely
provider = AnthropicProvider(api_key="…", cache_retention="none")
```

### Reading cache metrics

Each `AssistantMessage` carries a `Usage` object. The cache fields:

```python
agent.subscribe(lambda event, signal=None: None)
await agent.prompt("Summarise the document")

last_msg = agent.state.messages[-1]   # AssistantMessage
usage = last_msg.usage
print(usage.input_tokens)        # uncached prompt tokens (excludes cache_read)
print(usage.cache_read_tokens)   # tokens served from cache  ← savings here
print(usage.cache_write_tokens)  # tokens written to cache this turn
```

Cache hit rate for a turn:
`cache_read_tokens / (input_tokens + cache_read_tokens + cache_write_tokens)`.

:::tip
`input_tokens` in CubeLoop's `Usage` is the **uncached** portion — tokens that
were actually processed by the model this turn. The full prompt token count is
`input_tokens + cache_read_tokens + cache_write_tokens`. On a 100 % cache hit,
`input_tokens` is 0.
:::

You can also see these fields in the `cubeloop trace` CLI output and in
the OTel spans emitted by `Tracer`. Note that the OTel span attribute
`gen_ai.usage.input_tokens` follows the GenAI Semantic Convention and reports
the **inclusive** total (uncached + cached):

```
gen_ai.usage.input_tokens          = 8 420   (inclusive: uncached + cached)
gen_ai.usage.cache_read.input_tokens = 7 980  (cached subset)
```

---

## OpenAI: automatic caching

OpenAI caches automatically for prompts longer than 1 024 tokens —
no explicit `cache_control` markers are needed. CubeLoop surfaces the
hit data through the same `Usage` interface:

```python
usage.cache_read_tokens   # maps to prompt_tokens_details.cached_tokens
```

There is nothing to configure on the CubeLoop side. Keep the system
prompt and tool definitions stable across turns and OpenAI's cache
will warm up naturally.

---

## How to avoid breaking the cache

### ✅ Do

- **Keep the system prompt stable across turns.** The system prompt
  is the outermost cache breakpoint. Changing it invalidates
  everything.
- **Keep the tool list stable.** Add or remove tools between
  *conversations*, not between *turns*. The tool-definition
  breakpoint covers the entire tool list; any change invalidates
  from there.
- **Use `cache_retention="long"`** if your agent turns take more than
  5 minutes (long thinking runs, slow tools, infrequent users).
- **Use `agent.steer()`** to inject mid-turn guidance instead of
  prepending a new message to history — steer appends, not inserts.

### ❌ Avoid

- **Injecting messages into the middle of history.** Inserting a
  message before the last position shifts all subsequent messages,
  making the sequence differ from what the provider cached.
- **Changing the system prompt to include per-request data** (e.g.
  current timestamp, request ID). Put volatile data in the user
  message instead.
- **Reordering tool definitions.** The tool list is serialised in
  order; changing the order is a cache miss even if the tools
  themselves are identical.
- **Modifying a tool's description between turns** for the same
  thread. Description changes are serialised into the schema and
  will break the tool-definition breakpoint.

---

## Custom cache policy (Anthropic)

If the default three-breakpoint strategy does not fit your use case,
implement `CacheMarkerPolicy` and pass it to the provider:

```python
from cubeloop.providers.anthropic import CacheMarkerPolicy, AnthropicProvider
from cubeloop.providers.base import Message

class SystemOnlyPolicy:
    """Cache only the system prompt — useful when tool lists change often."""

    def mark_system(self) -> bool:
        return True

    def mark_last_tool(self) -> bool:
        return False

    def message_breakpoint_indices(self, messages: list[Message]) -> list[int]:
        return []   # no message breakpoints

provider = AnthropicProvider(
    api_key="…",
    cache_policy=SystemOnlyPolicy(),
)
```

The three methods map directly onto the three default breakpoints.
Return `True` / a non-empty list to enable a breakpoint, `False` /
`[]` to skip it.

---

## Multi-tenant considerations

In a multi-tenant setup each `thread_id` is an independent
conversation. Cache hits are per-thread: user A's history warms a
cache entry that benefits only user A's subsequent turns.

For maximum cache efficiency across tenants:

- Use a **stable, shared system prompt** for all tenants rather than
  embedding per-tenant data in the system prompt. Per-tenant context
  belongs in the first user message or in tool results.
- Keep tool definitions identical across all tenant agents. If
  different tenants need different tool sets, consider separate
  `Agent` instances with separate `AnthropicProvider` configurations
  rather than dynamically mutating the tool list.

---

## See also

- [Anthropic Provider](../providers/anthropic) — `cache_retention`,
  `CacheMarkerPolicy`, and the `on_payload` hook for inspecting raw requests.
- [Multi-turn Conversations](./multi-turn) — how message history grows
  across turns and why append semantics matter.
- [Tracing](../tracing/overview) — reading `cache_read.input_tokens`
  from OTel spans.
