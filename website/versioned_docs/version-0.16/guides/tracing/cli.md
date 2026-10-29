---
title: The trace CLI
description: "Use the CubeLoop trace CLI to inspect, filter, and display OpenTelemetry trace spans."
---

# Inspecting traces with `cubeloop trace`

The `JsonlSpanExporter` writes one file per trace under
`./cubeloop-traces/<date>/<trace_id>.jsonl`. The `cubeloop trace` CLI (provided by
the `trace-cli` extra) reads those files so you can see exactly what a run did
— which LLM and tool calls fired, in what order, what each returned, where it
errored, and the token counts — without re-running it.

```bash
pip install 'cubeloop[trace-cli]'      # or: uv sync --extra trace-cli
cubeloop trace --help
```

`--dir` defaults to `./cubeloop-traces`; pass `--dir <path>` if your traces live
elsewhere. Each file is one **trace**: the run plus any nested subagent runs,
which inherit the parent's `trace_id` and so land in the same file.

## `ls` — list recent traces

```bash
cubeloop trace ls          # newest first; -n N to limit
```

| column | meaning |
|---|---|
| `started` | trace start time (UTC) |
| `trace_id` | the id you pass to `view` / `follow` / `stats` |
| `spans` | span count for the whole trace (incl. subagents) |
| `status` | `ok` or `error` |
| `duration` | wall-clock span of the trace |
| `input` | the user's prompt, to identify the run |

### Filter by run metadata (`--meta`)

If the host stamped run-scoped metadata onto the trace (via
`tracing_context(metadata=…)` — e.g. cubebox records `conversation_id`,
`user_id`, `org_id`, `workspace_id` on the root `invoke_agent` span), filter to
just those traces:

```bash
cubeloop trace ls --meta conversation_id=conv_123
cubeloop trace ls --meta user_id=usr_9 --meta org_id=org_1   # repeatable = AND, exact match
```

Traces produced by [`Tracer.oneshot()`](../../api/cubeloop-tracing#tracer-oneshot)
(background LLM calls without a full agent loop) are also indexed here. Filter
them by the `operation` name passed to `oneshot()`:

```bash
cubeloop trace ls --meta oneshot_operation=consolidate_memory
cubeloop trace stats --by model --meta oneshot_operation=consolidate_memory
```

Each `--meta KEY=VALUE` is matched exactly against the trace's root metadata;
repeating the flag ANDs the conditions.

To **display** metadata values as columns (rather than only filter by them),
add `--show-meta KEY[,KEY…]`:

```bash
cubeloop trace ls --show-meta conversation_id,user_id
cubeloop trace ls --meta org_id=org_1 --show-meta conversation_id   # filter + show
```

(Or see all of a single trace's metadata with `cubeloop trace view <id> -v`.)

## `view` — render a trace as a span tree

A trace-id **prefix** is enough (the table truncates ids); an ambiguous prefix
lists candidates.

```bash
cubeloop trace view 1cd97cdb
```

```
trace
└── invoke_agent  14425.8ms  [0x1cd97cdb]
    ├── cubeloop.turn  1283.1ms  [0x5cfda93e]
    │   ├── chat deepseek-v4-flash  1208.7ms  tok 6845/68  [0x0d130229]
    │   └── execute_tool subagent  9610.2ms  subagent  [0x38bdd10a]
    │       └── invoke_agent  9601.0ms  [0x8094f99b]   ← subagent run, nested
    │           └── cubeloop.turn  9598.4ms  [0x57c5cfc7]
    │               ├── chat deepseek-v4-flash  1190.3ms  [0x8205ca6b]
    │               └── execute_tool web_search  6500.2ms  web_search  [0xca4e59fc]
    └── cubeloop.turn  491.9ms  ERROR  [0xce25f242]
        └── chat deepseek-v4-flash  427.2ms  ERROR  [0x0bff68ec]
            └── error: Error code: 400 - ... `tool_use` ids were found without
                `tool_result` blocks immediately after: call_01_...
```

Read it top-down: `invoke_agent` (a run) → `cubeloop.turn` (one agent-loop turn)
→ `chat <model>` (an LLM call, with `tok <input>/<output>`) and
`execute_tool <name>` (a tool call). A **subagent** shows up as
`execute_tool subagent` with its own `invoke_agent → cubeloop.turn → …` nested
directly beneath it. The `[0x…]` suffix on each node is the span's `span_id` —
grep it in the raw JSONL to inspect that exact span. Errors print inline under
the failing span.

Flags:

```bash
cubeloop trace view <id> --content   # expand gen_ai prompts / tool args / results
cubeloop trace view <id> -v          # expand ALL span attributes (verbose, large)
```

`--content` requires the run to have been recorded with
`record_content=True` (see [Content & Redaction](./content-recording)).

## `follow` — watch a trace live

```bash
cubeloop trace follow <id>           # polls as spans complete; good for a run in progress
```

## `stats` — aggregate across traces

```bash
cubeloop trace stats --by model                  # latency p50/p95, error rate, tokens
cubeloop trace stats --by tool --since 2026-05-20
```

`stats` also accepts `--meta KEY=VALUE` (same semantics as `ls`) to aggregate
only the traces that match — e.g. latency / error-rate / tokens for one user or
conversation:

```bash
cubeloop trace stats --by model --meta user_id=usr_9
cubeloop trace stats --by tool --meta conversation_id=conv_123
```

## `convert` — reconstruct an API request body

When you need to replay a specific LLM call — reproduce a failure, test a prompt
change, or run a raw `curl` against the same context — `convert` reads a recorded
`chat` span and outputs the complete request body.

Requires `record_content=True`.

```bash
# Default: last chat span in the trace, OpenAI JSON format
cubeloop trace convert <trace_id>

# Select which LLM call to reconstruct
cubeloop trace convert <trace_id> --turn 2        # 2nd chat span (1-indexed)
cubeloop trace convert <trace_id> --span 0xbb7eb1 # by span_id prefix (from `view`)

# Output formats
cubeloop trace convert <trace_id> --format openai     # default — JSON request body
cubeloop trace convert <trace_id> --format anthropic  # Anthropic Messages API body
cubeloop trace convert <trace_id> --format curl       # shell-executable curl command
```

The `[0x…]` span id from `view` output goes directly into `--span`:

```
├── chat kimi-k2.6  31704.5ms  [0xbb7eb192]   ← paste as: --span 0xbb7eb1
├── chat kimi-k2.6  32420.2ms  [0x7c76f48d]   ← or: --span 0x7c76f4
```

The reconstructed body includes the full conversation history, the system prompt,
all tool definitions, and request parameters (`model`, `max_tokens`,
`temperature`). Pipe to `python -m json.tool`, `jq`, or directly to a replay script.

## Beyond the CLI

The files are plain JSONL — one span per line — so you can parse them directly
(`jq`, `python -c`) to pull a specific attribute (`gen_ai.usage.*`,
`gen_ai.tool.call.result`, `gen_ai.input.messages`, …). Error detail lives in a
span **event** named `gen_ai.client.operation.exception`.

:::tip For AI agents
A bundled `cubeloop-trace` skill drives this CLI for debugging ("why did the run
end with no reply?", "the tool result is wrong"). It encodes the fast path
(`ls` → `view <prefix>`) and the token/cache-rate conventions.

```bash
npx skills add cubeplexai/cubeloop@cubeloop-trace -a claude-code
```
:::
