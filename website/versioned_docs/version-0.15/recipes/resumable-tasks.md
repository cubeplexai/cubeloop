---
title: Resumable Long Tasks
description: "Build crash-resilient long-running tasks with CubeLoop checkpointing and recovery."
---

# Recipe: Resumable Long Tasks

When an agent is mid-flight through a long-running operation (a series
of tool calls, a multi-turn reasoning session) and the process dies,
you want to come back and pick up where it left off — not start over.
CubeLoop's append-only checkpointing plus `agent.resume()` makes this
trivial *between turns*; for resumption *mid-tool*, you need a little
more care.

**Time to run:** 15 minutes.
**Deps:** `cubeloop[sqlite]`, an `ANTHROPIC_API_KEY`.

## The pattern

There are three crash points to think about:

1. **Between turns** — The model has answered, no tools to run, the
   loop is between iterations. `resume()` re-invokes the model. *Free
   with the checkpointer.*
2. **After tool results, before model call** — Tool results are
   persisted. `resume()` sees the last message is a `ToolResultMessage`
   and re-invokes the model. *Free with the checkpointer.*
3. **Mid-tool** — The tool started but didn't finish. Nothing is
   persisted yet (CubeLoop only persists messages). You need
   tool-internal idempotency. *Requires care.*

This recipe focuses on case 3.

## Idempotent tools with external state

The pattern: each tool action has a deterministic, idempotent key.
Before doing the work, check whether it's been done.

```python title="tools.py"
import os
import json
from pathlib import Path

from cubeloop import AgentToolResult, TextContent, tool


# Simple file-backed job store; replace with Redis / Postgres in prod.
JOB_DIR = Path(os.environ.get("JOB_DIR", "/tmp/cubeloop-jobs"))
JOB_DIR.mkdir(parents=True, exist_ok=True)


# execution_mode="sequential" → one transcode at a time. Only `signal` is
# declared, so that's the only loop-supplied arg injected.
@tool(execution_mode="sequential")
async def transcode_video(source_path: str, output_path: str, *, signal=None) -> AgentToolResult:
    "Transcode a video file. Idempotent — safe to retry."
    job_key = f"transcode:{source_path}->{output_path}"
    job_file = JOB_DIR / f"{job_key.replace('/', '_')}.json"

    if job_file.exists():
        # Already done in a previous run.
        state = json.loads(job_file.read_text())
        return AgentToolResult(
            content=[TextContent(text=f"Already transcoded to {state['output_path']}.")],
            details=state,
        )

    # Do the actual work (long-running, expensive).
    # Use signal to abort cleanly if cancelled.
    output = await run_ffmpeg(source_path, output_path, signal=signal)

    # Commit the job-done marker AFTER the work succeeds.
    job_file.write_text(json.dumps({"output_path": output}))

    return AgentToolResult(
        content=[TextContent(text=f"Transcoded to {output}.")],
        details={"output_path": output},
    )
```

Now if the process dies during `run_ffmpeg`, the next agent run sees
`job_file.exists() == False`, redoes the work, and only writes the
marker on success. If the process dies *after* the marker was
written, the next run sees the marker, returns the cached result
immediately, and the agent continues as if it had just finished.

## Resuming the agent

```python title="resume.py"
import asyncio
import os
import sys

from cubeloop import Agent
from cubeloop.checkpointer import SQLiteCheckpointer
from cubeloop.providers.anthropic import AnthropicProvider

from tools import transcode_video   # the @tool-decorated AgentTool from above


async def main(thread_id: str, initial_prompt: str | None):
    async with SQLiteCheckpointer("jobs.db") as cp:
        agent = Agent(
            model=AnthropicProvider(provider_id="anthropic", api_key=os.environ["ANTHROPIC_API_KEY"]).model("claude-sonnet-4-6"),
            system_prompt="You orchestrate video transcoding jobs.",
            tools=[transcode_video],
            checkpointer=cp,
            thread_id=thread_id,
        )
        agent.subscribe(lambda e, s=None: None)

        if initial_prompt:
            # Fresh run. prompt() auto-loads history on first call before
            # appending the new user message.
            await agent.prompt(initial_prompt)
        else:
            # Resume. agent.resume() does NOT auto-load — only prompt() does.
            # Hydrate the agent state manually first.
            data = await cp.load(thread_id)
            if data is None:
                raise RuntimeError(f"No saved state for thread {thread_id!r}")
            agent.state.messages = list(data.messages)
            # `extra` is restored too; it's private on Agent, so use the
            # checkpointer's view if your middleware reads it.

            # Resume picks up from the last persisted message:
            #   ToolResultMessage / UserMessage → re-invokes the model
            #   AssistantMessage with no queued steer/follow_up → raises
            last = agent.state.messages[-1]
            if type(last).__name__ == "AssistantMessage":
                from cubeloop.providers.base import ToolCall
                has_pending_tools = any(isinstance(c, ToolCall) for c in last.content)
                if not has_pending_tools:
                    # Job finished normally — no pending tool calls.
                    # Ask the user for the next prompt instead.
                    print("Last run completed. Nothing to resume.")
                    return
                # Has unresolved tool calls — resume() will execute them.
            await agent.resume()


if __name__ == "__main__":
    thread_id = sys.argv[1]
    initial = sys.argv[2] if len(sys.argv) > 2 else None
    asyncio.run(main(thread_id, initial))
```

Workflow:

```bash
# Start a job:
python resume.py job-1 "Transcode /videos/a.mov to /out/a.mp4 and /videos/b.mov to /out/b.mp4"

# Kill it mid-flight: Ctrl-C.

# Resume — agent picks up from the last persisted message:
python resume.py job-1
```

## The three resume scenarios in code

```python
async def smart_resume(agent, cp, thread_id):
    # resume() doesn't auto-load — hydrate the agent first if its state is empty.
    if not agent.state.messages:
        data = await cp.load(thread_id)
        if data is None or not data.messages:
            return False           # nothing to resume from
        agent.state.messages = list(data.messages)

    last = agent.state.messages[-1]
    last_role = type(last).__name__

    if last_role == "AssistantMessage":
        # Either the run finished naturally, or it died right after a
        # turn ended. resume() raises unless there's queued steering.
        # Easiest path: ask the user what's next.
        return False

    # Last is UserMessage or ToolResultMessage — safe to resume.
    await agent.resume()
    return True
```

## Persistence + abort

`agent.abort()` triggers a clean teardown that emits `agent_end`. The
last *fully persisted* message is whatever made it through
`message_end`. Aborts during a tool's execution don't persist the
tool result (the tool didn't return), so `resume()` will re-dispatch
the model with the last `AssistantMessage` containing the unfinished
`ToolCall`. The model will usually re-issue the call — your
idempotency guards handle the rest.

## What about persisting partial tool state?

CubeLoop doesn't expose a "persist a partial tool result" API. The
intended pattern is: keep partial state in the tool's own
external store (filesystem, Redis, S3), keyed deterministically by the
tool args. That's what `transcode_video` above does with `JOB_DIR`.

## Common pitfalls

- **Non-idempotent tools** — Without deterministic keys, retries can
  charge a credit card twice or send a duplicate email. Always wrap
  external side-effects in an idempotency key.
- **Job markers in `/tmp`** — Cleared on reboot. Use a real
  persistence layer for production jobs.
- **`resume()` after an assistant message with no queue** — Raises.
  Either prompt the user for the next message or call `prompt()`
  fresh.
- **`resume()` on a fresh agent** — Raises `No messages to continue
  from`. `resume()` does not auto-load from the checkpointer; only
  `prompt()` does. Hydrate manually with `agent.state.messages =
  (await cp.load(thread_id)).messages` first.
- **Forgetting the signal check inside the tool** — A long
  `await asyncio.sleep(...)` or a `for ... in stream` that ignores
  `signal.is_set()` won't honour `abort`. Drop a check inside any
  hot loop.

## Run the example

A self-contained, runnable version of this recipe is in the repository at
[`examples/resumable_tasks.py`](https://github.com/cubeplexai/cubeloop/blob/main/examples/resumable_tasks.py).

```bash
git clone https://github.com/cubeplexai/cubeloop && cd cubeloop
uv sync --extra sqlite

export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY [+ OPENAI_BASE_URL]

# Start the job (kill mid-flight with Ctrl-C to test recovery):
uv run python examples/resumable_tasks.py job-1 start

# Resume from where it stopped — already-done items are skipped:
uv run python examples/resumable_tasks.py job-1
```

## See also

- [Multi-turn → `resume()`](../guides/agents/multi-turn#resume--continue-from-the-last-message)
  — full semantics.
- [Persistent Chat](./persistent-chat) — the simpler restartable
  scenario.
- [SQLite Checkpointing](../guides/checkpointing/sqlite) — what's
  persisted, when.
