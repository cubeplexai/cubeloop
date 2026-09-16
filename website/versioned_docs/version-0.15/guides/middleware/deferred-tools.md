---
title: Deferred Tool Groups
description: "Hide MCP tool schemas from the model by default, expanding them on demand — without invalidating the prompt cache."
---

# Deferred Tool Groups

When an agent connects to many MCP servers, their combined tool schemas
can consume thousands of tokens of context on every turn — even if the
model only needs one or two groups for the current task.
`DeferredToolGroup` solves this by replacing full schemas with a compact
catalog, letting the model load groups on demand.

## Two strategies

Deferred tools support two strategies, chosen with
`deferred_tool_strategy` (default: `"dispatch"`):

| | `tools` param | system prompt | cache cost per expansion | calling path |
|---|---|---|---|---|
| **`dispatch`** (default) | static | static | **zero** — schemas are message-suffix appends | `deferred_tool_call` dispatcher, engine-unwrapped |
| **`inject`** | grows per expansion | catalog counts change | full re-read of system + history | native tool calls |

**Why dispatch is the default.** Tool definitions render at the very
front of the prompt on every prefix-cached provider. Injecting a tool
mid-conversation inserts bytes *before* the entire history, so each
expansion re-reads the whole conversation at uncached rates. Dispatch
mode never touches the tools array or the system prompt after the first
request — schemas travel in `load_tools` tool results, which append to
the end of the message history and cache incrementally like any other
turn.

**When to pick `inject`.** Native tool calls get provider-side schema
validation and the calling ergonomics models are trained on. If your
tool arguments are complex and your conversations are short (so the
per-expansion cache cost is small), `inject` trades cache efficiency for
calling reliability.

## How dispatch mode works

1. The system prompt carries a short, **static** catalog — one line per
   group with a description and tool names. It never changes.
2. The model calls the built-in `load_tools(group_id)` and receives the
   group's **full schemas in the tool result**.
3. The model invokes a loaded tool through the built-in dispatcher:
   `deferred_tool_call(tool_name=..., arguments=...)`.
4. The engine unwraps the dispatcher call before anything else sees it:
   validation, `before_tool_call`/`after_tool_call` hooks, permission
   systems, emitted events, and tracing all observe the **real** tool
   name and arguments — never the envelope.

```
# Deferred tool groups

These tool groups are available but not yet loaded. Call `load_tools(group_id)`
to get their full schemas, then invoke them via
`deferred_tool_call(tool_name=..., arguments=...)`.

- `mcp:github` — GitHub: Issues, PRs, repos, code search (4 tools)
  create_issue, search_repos, create_pr, list_comments
- `mcp:linear` — Linear: Project management and issue tracking (6 tools)
  create_issue, update_issue, list_projects, ...
```

A few properties worth knowing:

- **Implicit loading.** If the model calls `deferred_tool_call` for a
  tool it never explicitly loaded, the middleware loads it on the fly
  and validates the arguments. On a validation failure the error result
  includes the full schema, so the model self-corrects in one round
  trip.
- **Compaction self-rescue.** `load_tools` is idempotent — if context
  compaction drops an old result, the model can simply call it again
  and get byte-identical schemas back.
- **Forks.** Forked agents (`fork_once`) inherit the dispatch resolver,
  so tools the parent loaded remain invocable inside forks.

## Basic setup

Pass `deferred_tool_groups` to `Agent`. The middleware is created
automatically — no manual wiring needed:

```python
from cubeloop import Agent
from cubeloop.deferred import DeferredToolGroup

# load_github_tools / load_linear_tools are zero-arg async callables
# returning list[AgentTool]. See "Writing a loader" below for the two
# common shapes (MCP-backed and hand-written @tool functions).

github_group = DeferredToolGroup(
    group_id="mcp:github",
    display_name="GitHub",
    description="Issues, PRs, repos, code search",
    tool_names=["create_issue", "search_repos", "create_pr", "list_comments"],
    loader=load_github_tools,
)

linear_group = DeferredToolGroup(
    group_id="mcp:linear",
    display_name="Linear",
    description="Project management and issue tracking",
    tool_names=["create_issue", "update_issue", "list_projects"],
    loader=load_linear_tools,
)

agent = Agent(
    model=provider.model("claude-sonnet-4-6"),
    tools=[search_tool, calculator],              # always-available tools
    deferred_tool_groups=[github_group, linear_group],
    # deferred_tool_strategy="inject",            # opt into v1 behavior
)
```

### `DeferredToolGroup` fields

| Field | Type | Description |
|---|---|---|
| `group_id` | `str` | Unique identifier the model uses in `load_tools` calls (e.g. `"mcp:github"`) |
| `display_name` | `str` | Human-readable label shown in the catalog |
| `description` | `str` | One-line summary of the group's capabilities |
| `tool_names` | `list[str]` | Tool names shown in the catalog. **Must match the `AgentTool.name` of each tool the loader returns** — selective expansion (`load_tools(group_id, tool_names=[…])`) matches on this. |
| `loader` | `async () -> list[AgentTool]` | Callback that returns the full tool set for this group |

### Writing a loader

The loader is a zero-argument async callable that returns
`list[AgentTool]`. CubeLoop only cares about its return type — where the
`AgentTool` objects come from is up to you. Two common shapes:

**From an MCP server.** `load_mcp_tools_stdio` / `load_mcp_tools_http`
return an `MCPDiscoveryResult` whose `.tools` is the `list[AgentTool]`
you want. Wrap it:

```python
from cubeloop.deferred import DeferredToolGroup
from cubeloop.mcp import load_mcp_tools_stdio

async def load_github_tools():
    result = await load_mcp_tools_stdio(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_TOKEN": "ghp_…"},
    )
    return result.tools   # list[AgentTool]

github_group = DeferredToolGroup(
    group_id="mcp:github",
    display_name="GitHub",
    description="Issues, PRs, repos, code search",
    tool_names=["create_issue", "search_repos", "create_pr"],
    loader=load_github_tools,
)
```

The names in `tool_names` must match the MCP server's tool names —
those become `AgentTool.name` after discovery. If the catalog lists
`create_issue` but the server publishes it as `github_create_issue`,
selective expansion misses.

**From hand-written `@tool` functions.** Any function decorated with
`@tool` produces an `AgentTool` (its `.name` defaults to the function
name, overridable via `@tool(name="…")`). A loader for hand-written
tools is just `async lambda` over a list:

```python
from cubeloop import tool
from cubeloop.deferred import DeferredToolGroup

@tool
async def create_issue(*, repo: str, title: str, body: str) -> str:
    "Open a GitHub issue."
    ...

@tool
async def search_repos(*, query: str) -> str:
    "Search public repos."
    ...

async def load_github_tools():
    return [create_issue, search_repos]   # already AgentTools

github_group = DeferredToolGroup(
    group_id="mcp:github",
    display_name="GitHub",
    description="Issues, PRs, repos, code search",
    tool_names=["create_issue", "search_repos"],
    loader=load_github_tools,
)
```

You can mix the two — return MCP tools and hand-written tools in the
same list — as long as every name in `tool_names` matches an
`AgentTool.name` in the returned list. If the loader raises, the
error is reported to the model as a tool error and the group stays
unexpanded.

## The `load_tools` tool

The model calls `load_tools` to load a group's tools. Two modes:

```
# Load everything in the group
load_tools(group_id="mcp:github")

# Load specific tools only
load_tools(group_id="mcp:github", tool_names=["create_issue", "search_repos"])
```

In dispatch mode the result carries the full schemas:

```json
{
  "group_id": "mcp:github",
  "expanded": true,
  "tool_names": ["create_issue", "search_repos"],
  "remaining": 2,
  "schemas": [
    {"name": "create_issue", "description": "...", "parameters": {"...": "..."}},
    {"name": "search_repos", "description": "...", "parameters": {"...": "..."}}
  ]
}
```

(In `inject` mode `schemas` is omitted — the definitions join the
model-visible tools array instead.)

After loading, the tools are immediately available in the same turn.

### Loader caching

The `loader` callback is invoked exactly **once per group per run**.
The first load triggers it; subsequent selective loads filter from the
cached result. If the loader fails, the error is returned to the model
and the group remains unloaded. Already-loaded tools are idempotent —
re-requesting them is a no-op (and in dispatch mode re-serves the same
schemas).

## Expansion state

The middleware tracks which groups are loaded in `ctx.extra`:

```python
ctx.extra["expanded_groups"] = {
    "mcp:github": None,                    # fully loaded (None = all tools)
    "mcp:linear": ["create_issue"],        # partially loaded
    # mcp:slack not present = unloaded
}
```

This state survives checkpointing and drives cross-run replay.

## Cross-run replay

When resuming a conversation from a previous run, restore the expansion
state so dispatched calls resolve immediately. `prepare_resumed_state`
handles this — the `strategy` argument is **required** and must match
the middleware's strategy:

```python
from cubeloop.deferred import DeferredToolsMiddleware

# saved_extra is the persisted ctx.extra from the previous run
resumed = await DeferredToolsMiddleware.prepare_resumed_state(
    groups=all_groups,
    expanded=saved_extra["expanded_groups"],
    strategy="dispatch",
)

agent = Agent(
    model=model,
    tools=[*builtin_tools, *resumed.pre_loaded_tools],
    deferred_tool_groups=resumed.remaining_groups,
)
```

`prepare_resumed_state` returns a `ResumedState` with:

| Field | Description |
|---|---|
| `pre_loaded_tools` | Tools from previously-loaded groups, ready to resolve (hidden from the payload in dispatch mode) |
| `remaining_groups` | Groups still loadable via `load_tools` |
| `loader_cache` | Pre-loaded tool cache (pass to `resumed_loader_cache` to avoid redundant loader calls) |

In dispatch mode there is nothing else to restore: the schemas the model
saw live in the message history, and the checkpointer brings them back
with the conversation. In `inject` mode, fully loaded groups leave the
deferred set (as in v1).

## Advanced: constructing the middleware directly

For full control over the catalog header or resume seeding, construct
`DeferredToolsMiddleware` yourself:

```python
from cubeloop.deferred import DeferredToolsMiddleware

mw = DeferredToolsMiddleware(
    groups=[github_group, linear_group],
    extra_ref=lambda: agent_extra,
    strategy="dispatch",
    catalog_header="# Available integrations\n\nLoad with load_tools().",
)

agent = Agent(
    model=model,
    tools=[search_tool],
    middleware=[mw],
)
```

### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `groups` | `list[DeferredToolGroup]` | required | Groups to defer |
| `extra_ref` | `() -> dict` | required | Returns the live `ctx.extra` dict |
| `strategy` | `"dispatch" \| "inject"` | `"dispatch"` | Disclosure strategy (see above) |
| `catalog_header` | `str \| None` | *(strategy-specific built-in)* | Header text for the catalog section |
| `resumed_loader_cache` | `dict[str, list[AgentTool]] \| None` | `None` | Pre-loaded tool cache from a previous run (avoids re-calling loaders on resume) |
| `on_tools_expanded` | `(list[AgentTool]) -> None \| None` | `None` | Called after new tools are loaded (used internally for cross-turn persistence) |

When using the `Agent(deferred_tool_groups=...)` shorthand, `extra_ref`
is automatically bound to `self._extra`.

## Migrating from 0.10

Deferred tool groups shipped in CubeLoop 0.10 with what is now the
`inject` strategy. Upgrading changes behavior:

- **The default strategy is now `dispatch`.** The catalog wording
  changes, a `deferred_tool_call` builtin appears, and loaded tools no
  longer join the model-visible tools array. Restore the 0.10 behavior
  with `Agent(deferred_tool_strategy="inject")` or
  `DeferredToolsMiddleware(strategy="inject")`.
- **`inject` mode no longer renders schemas into the system prompt.**
  The definitions were already in the tools array; the duplicate
  rendering (and its double token billing) is gone. As a consequence,
  the `resumed_schemas` constructor parameter and
  `ResumedState.expanded_schemas` no longer exist.
- **`prepare_resumed_state` requires `strategy=`** so a resume can't
  silently mismatch the middleware's strategy.

## When to use it

**Good fit:**

- Agent has access to 5+ MCP servers but typically uses 1–2 per conversation.
- Tool schemas are large (many parameters, long descriptions).
- You want to keep prompt-cache hit rates high across turns.

**Skip it when:**

- The agent has only a few tools — the overhead of the catalog and
  `load_tools` call isn't worth it.
- All tools are needed on every turn — deferring just adds a round trip.
- Tool schemas are small — the context savings are minimal.

## See also

- [Loading MCP Tools](../mcp/loading) — how to get `AgentTool` lists from
  MCP servers.
- [The 9 Hooks](./hooks) — the middleware hooks that power deferred tools
  (`transform_system_prompt`, `after_tool_call`, `resolve_tool_call`).
- [Composition](./composition) — how middleware composes when stacked with
  other middleware.
