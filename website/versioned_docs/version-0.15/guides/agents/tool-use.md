---
title: Tool Use & Parallel Execution
description: "Register tools, execute them in parallel or sequentially, and handle results with Pydantic validation in CubeLoop."
---

# Tool Use & Parallel Execution

Tools are how an agent acts on the world. CubeLoop turns each `AgentTool`
into a JSON Schema for the model, validates arguments with Pydantic,
runs the work, and feeds the result back as a `ToolResultMessage`. By
default tools run in parallel when the model calls more than one in a
single turn.

## The `@tool` decorator

The quickest way to define a tool is to decorate an async function. CubeLoop
generates the input schema from the parameters, so there's no separate model
or boilerplate `execute` signature to write:

```python
from typing import Annotated
from pydantic import Field
from cubeloop import tool


@tool
async def search(
    query: Annotated[str, Field(description="The natural-language query")],
    limit: Annotated[int, Field(ge=1, le=100)] = 10,
) -> str:
    "Search the internal knowledge base."
    results = await my_search_backend(query, limit)
    return "\n".join(results)
```

That's a complete, registrable `AgentTool`. The decorator infers:

- **name** from the function name (override with `@tool(name=...)`);
- **description** from the docstring (override with `@tool(description=...)`);
- **the input schema** from the typed parameters — `Field(...)` defaults and
  metadata are honoured exactly as in a hand-written model.

The return value can be a plain `str` (wrapped as text, as above), a
`TextContent`, a `list` of content, or a full `AgentToolResult` when you need
`details`, `is_error`, or `terminate`:

```python
from cubeloop import tool, AgentToolResult, TextContent


@tool
async def search(query: str, limit: int = 10) -> AgentToolResult:
    "Search the internal knowledge base."
    results = await my_search_backend(query, limit)
    return AgentToolResult(
        content=[TextContent(text="\n".join(results))],
        details={"raw_results": results},   # passes through to ToolResultMessage.details
    )
```

To run a tool sequentially, pass `@tool(execution_mode="sequential")`. If the
function needs the loop-supplied arguments, just declare them — any of
`tool_call_id`, `signal`, or `on_update` are injected when present and never
appear in the schema:

```python
@tool
async def long_job(prompt: str, *, signal=None, on_update=None) -> str:
    "Run a long job, streaming progress."
    ...
```

## Anatomy of a tool (longhand)

The decorator is sugar over the explicit `AgentTool`. The form below is
equivalent and remains fully supported — reach for it when you want to build
tools dynamically or share one params model across several tools:

```python
from pydantic import BaseModel, Field
from cubeloop import AgentTool, AgentToolResult, TextContent


class SearchParams(BaseModel):
    query: str = Field(..., description="The natural-language query")
    limit: int = Field(10, ge=1, le=100)


async def search(tool_call_id, params: SearchParams, *, signal=None, on_update=None):
    results = await my_search_backend(params.query, params.limit)
    return AgentToolResult(
        content=[TextContent(text="\n".join(results))],
        details={"raw_results": results},   # passes through to ToolResultMessage.details
    )


search_tool = AgentTool(
    name="search",
    description="Search the internal knowledge base.",
    parameters=SearchParams,
    execute=search,
)
```

The `description` is shown to the model verbatim — write it for the
model, not for humans. Pydantic `Field(description=...)` propagates
into the JSON Schema and helps the model understand each parameter.

## Parallel by default

When the model emits multiple tool calls in one assistant message,
CubeLoop schedules them on `asyncio.create_task()` and gathers them.
That's almost always what you want.

```python
agent = Agent(
    model=model,
    tools=[search_tool, fetch_url_tool, summarise_tool],
)
```

The event stream emits `tool_execution_start` for all of them up
front, interleaved `tool_execution_update` events as each tool reports
progress, then `tool_execution_end` per tool in completion order.

### Fault isolation

Failures are isolated per call: every tool in the batch runs to
completion regardless of what its siblings do, and each call gets
exactly one `tool_result` — a real one, or a synthesized error result
if the tool (or an `after_tool_call` hook) raised something
unexpected. One misbehaving tool never discards a sibling's completed
work, and a persisted assistant message is never left with unanswered
`tool_calls` (which providers reject on the next turn).

Two exception classes still propagate, by design:

- **HITL control exceptions** (a tool detaching for human input) — the
  batch first waits for every sibling to finish and persists their
  results, then the suspend propagates. Only the detaching call stays
  unanswered, and the HITL resume/abort flow answers it later.
- **Cancellation** — siblings are cancelled and awaited (no leaked
  tasks); results of tools that completed before the cancel are
  persisted so a resumed thread doesn't re-run them and duplicate
  their side effects.

## Forcing sequential execution

There are two ways to opt out of parallel mode:

1. **Per agent** — `Agent(tool_execution="sequential")`. All tool
   batches run one-by-one in the order the model emitted them.

2. **Per tool** — set `execution_mode="sequential"` on the
   `AgentTool`. If *any* tool in the current batch is sequential, the
   whole batch falls back to sequential.

    ```python
    write_db_tool = AgentTool(
        name="write_db",
        description="Persist a record.",
        parameters=WriteDbParams,
        execute=write_db,
        execution_mode="sequential",   # opt out of parallelism for safety
    )
    ```

The built-in `ask_user` HITL tool (see [HITL guide](../hitl/overview)) sets
`execution_mode="sequential"` — it pauses the agent for human input, so
the tool batch runs one-by-one.

Use sequential mode when tools mutate shared state (a DB, a counter)
and you want a deterministic order.

## Streaming tool progress

Long-running tools can stream partial updates that show up as
`tool_execution_update` events on the agent stream:

```python
async def slow_search(tool_call_id, params, *, signal=None, on_update=None):
    for i, page in enumerate(await fetch_pages(params.query)):
        if signal and signal.is_set():
            break
        if on_update:
            on_update({"progress": i, "total": len(pages), "url": page.url})
        await process(page)
    return AgentToolResult(content=[TextContent(text="done")])
```

`partial_result` in the event is whatever object you handed to
`on_update`. Use a small dict; it doesn't end up in the model's
context.

## Cancelling in-flight tools

The `signal` argument is the same `asyncio.Event` that
`agent.abort()` sets. Check it in any loop:

```python
async def long_running(tool_call_id, params, *, signal=None, on_update=None):
    for chunk in big_dataset:
        if signal and signal.is_set():
            return AgentToolResult(content=[TextContent(text="cancelled")])
        await process_chunk(chunk)
```

If your work is one big `await`, wrap it in
`asyncio.wait_for(..., timeout=…)` or use the abort callbacks of the
underlying library.

## Returning errors

Two ways:

1. **Raise an exception.** CubeLoop catches it, turns it into an
   `AgentToolResult` with `is_error=True` and the exception string as
   `TextContent`.
2. **Return `is_error=True` explicitly.** Useful when you want a
   structured error body:

    ```python
    return AgentToolResult(
        content=[TextContent(text="Rate limit exceeded; try again in 60s")],
        is_error=True,
    )
    ```

Either way, the model receives a tool result that's clearly marked as
an error and usually adapts (retries with different args, asks the
user, etc.).

## Ending the turn from a tool: `terminate`

A tool can declare *"after this, stop looping — don't ask the model
again."* Set `terminate=True`:

```python
async def submit_final_answer(tool_call_id, params, *, signal=None, on_update=None):
    save_answer(params.answer)
    return AgentToolResult(
        content=[TextContent(text="submitted")],
        terminate=True,
    )
```

CubeLoop only terminates if *every* tool result in the current batch is
`terminate=True`. The agent loop emits `turn_end`, then `agent_end`,
and exits.

## Many tools? Defer their schemas

When the toolbelt grows — usually because you've wired up several MCP
servers — the combined JSON schemas can consume thousands of tokens of
system prompt on every turn, even though the model only needs one or
two groups for the current task. The schemas are also a prompt-cache
landmine: any tool change anywhere invalidates the cache for every
turn that follows.

`DeferredToolGroup` solves this by replacing the full schemas with a
compact catalog. The model sees a one-line description per group and
loads a group on demand via the built-in `load_tools` tool, which
returns the group's full schemas in its tool result; loaded tools are
then invoked through the `deferred_tool_call` dispatcher. The tools
array and system prompt stay byte-stable for the whole run, so loading
never invalidates the prompt cache. (The v1 behavior — injecting loaded
tools into the model-visible tools array as native tools — remains
available via `deferred_tool_strategy="inject"`.)

```python
from cubeloop import Agent
from cubeloop.deferred import DeferredToolGroup
from cubeloop.mcp import load_mcp_tools_stdio

async def load_github_tools():
    result = await load_mcp_tools_stdio(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
    )
    return result.tools   # list[AgentTool]

github_group = DeferredToolGroup(
    group_id="mcp:github",
    display_name="GitHub",
    description="Issues, PRs, repos, code search",
    tool_names=["create_issue", "search_repos", "create_pr"],
    loader=load_github_tools,
)

agent = Agent(
    model=provider.model("claude-sonnet-4-6"),
    tools=[search_tool, calculator],     # always-available
    deferred_tool_groups=[github_group], # expanded on demand
)
```

The loader is a zero-arg async callable returning `list[AgentTool]` —
wrap an MCP discovery call (as above), or just return your own
`@tool`-decorated functions. The names in `tool_names` must match each
tool's `AgentTool.name` so selective expansion can find them.

Good fit when you have ≥5 tool groups but typically only use one or
two per conversation, or when schemas are large enough to noticeably
inflate every system prompt. Skip it when all the tools are needed on
every turn — deferring just adds a round trip.

See [Deferred Tool Groups](../middleware/deferred-tools) for the full
API (including a hand-written-tools loader example), cross-run replay,
and the advanced middleware constructor.

## Common pitfalls

- **Forgetting the keyword-only args** — `execute(tool_call_id,
  params)` will work in dev but crash when the framework passes
  `signal=`. Keep `*, signal=None, on_update=None` in the signature.
- **Heavy `details` payloads** — `details` is preserved through the
  agent event but is *not* shown to the model. Don't pack massive blobs
  there unless you have a downstream consumer.
- **Pydantic strictness surprises** — `Field(..., min_length=1)` lets
  the model see the constraint via JSON Schema; constraints help, but
  remember the model still sometimes sends bad JSON. CubeLoop turns the
  `ValidationError` into a tool error result; you don't need to wrap
  validation yourself.
- **`tools=[]` and the model still asks for a tool** — Usually means
  the prompt mentions one. Either remove the suggestion from the
  system prompt or pass the tool.

## See also

- [Streaming Events](./streaming) — how `tool_execution_*` events fit
  the larger event taxonomy.
- [Middleware → before_tool_call](../middleware/hooks#before_tool_call)
  and [after_tool_call](../middleware/hooks#after_tool_call) —
  interception, policy, retries.
- [Recipes → Weather Agent](../../recipes/weather-agent) — a real
  HTTP-calling tool, end-to-end.
- [MCP Loading](../mcp/loading) — pull an entire toolset off an MCP
  server in one call.
