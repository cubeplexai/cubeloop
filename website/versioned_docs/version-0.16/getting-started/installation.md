---
title: Installation
description: "Install CubeLoop via pip. Python 3.11+ required. Supports Linux, macOS, and Windows."
---

# Installation

CubeLoop runs on **Python 3.11+**. The core has three runtime
dependencies: `pydantic`, `anthropic`, `openai`. Optional features
(SQLite, Postgres, MCP, OpenTelemetry tracing) are gated behind extras
so you only install what you use.

## With pip

```bash
pip install cubeloop
```

Optional extras:

```bash
pip install "cubeloop[sqlite]"        # adds aiosqlite for SQLiteCheckpointer
pip install "cubeloop[postgres]"      # adds asyncpg + sqlalchemy + msgpack
pip install "cubeloop[mcp]"           # adds the MCP SDK for tool loaders
pip install "cubeloop[tracing]"       # adds opentelemetry-sdk for Tracer / Meter
pip install "cubeloop[tracing-otlp]"  # adds the OTLP/HTTP span exporter
pip install "cubeloop[sqlite,mcp,tracing]"  # combine
```

## With uv

[`uv`](https://github.com/astral-sh/uv) is significantly faster than
pip and is the recommended workflow:

```bash
uv add cubeloop
uv add "cubeloop[sqlite,postgres,mcp,tracing,tracing-otlp]"
```

In an existing uv project, `uv sync` re-locks the environment after
edits to `pyproject.toml`.

## With Poetry

```bash
poetry add cubeloop
poetry add "cubeloop[sqlite,postgres,mcp,tracing,tracing-otlp]"
```

## Verifying the install

```bash
python -c "import cubeloop; print(cubeloop.__doc__)"
# cubeloop — Pythonic async-native agent framework.
```

If you see an `ImportError`, your interpreter is likely \< 3.11 — check
`python --version`.

## Configuring provider credentials

CubeLoop providers read credentials from constructor arguments. Most
deployments pull them from environment variables:

```python
import os
from cubeloop.providers.anthropic import AnthropicProvider
from cubeloop.providers.openai import OpenAIProvider

anthropic = AnthropicProvider(provider_id="anthropic", api_key=os.environ["ANTHROPIC_API_KEY"])
openai = OpenAIProvider(provider_id="openai", api_key=os.environ["OPENAI_API_KEY"])
```

You can also pass `base_url=...` to either provider to point at a
self-hosted endpoint or compatible proxy (e.g. Anthropic Bedrock,
LiteLLM, vLLM).

For the [FauxProvider](../guides/providers/custom#using-fauxprovider-in-tests)
(used in tests), no credentials are required.

## Choosing extras: which to install

| Extra | Pulls in | Install when |
|---|---|---|
| (none) | core only | You only need in-memory state, no MCP, no tracing |
| `[sqlite]` | `aiosqlite` | Single-process app needs disk persistence |
| `[postgres]` | `asyncpg`, `sqlalchemy`, `msgpack` | Multi-instance / production — see [Postgres guide](../guides/checkpointing/postgres) |
| `[mcp]` | `mcp` | You want to mount MCP server tools into your agent |
| `[tracing]` | `opentelemetry-sdk` | You want OpenTelemetry traces (and optionally metrics) — see [Tracing guide](../guides/tracing/overview) |
| `[tracing-otlp]` | `opentelemetry-exporter-otlp-proto-http` | Ship traces to an OTLP/HTTP backend (Jaeger ≥1.50, Tempo, Honeycomb, Datadog, …) |
| `[docs]` | `griffe` | You're building the docs site (contributors only) |

## Next steps

- [Quick Start](./quick-start) — your first agent in five minutes.
- [Core Concepts](./core-concepts) — what `Agent` / `Tool` / `Provider`
  / `Checkpointer` actually mean before you start gluing them.
