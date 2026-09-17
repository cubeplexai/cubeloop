# Changelog

All notable changes to CubeLoop are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`ToolResultLimitMiddleware` caps tool-result text in `after_tool_call`.**
  Oversized text is truncated to `max_chars` (default 20,000) with a notice
  telling the model to narrow the call. The rewrite happens before
  `ToolExecutionEndEvent`, so the model, checkpointer, and host consumers
  all see the truncated result. Image blocks are kept; tools in
  `exclude_tool_names` pass through.

## [0.15.0] - 2026-09-16

### Added

- **Public execution sessions for service hosts.** Every `Agent` exposes
  `agent.session` with typed prompt, continue, and HITL response requests.
  Each attempt carries a run ID and attempt ID and returns a settled
  `ExecutionResult` that distinguishes completion, suspension, cancellation,
  failure, and incomplete history. Existing `Agent.prompt()`, `resume()`, and
  `respond()` use the same sequential admission gate.
- **Identified live input and session events.** Hosts can submit steering or
  follow-up messages with an input ID, observe when an input is committed to
  the checkpoint, and subscribe to bounded, attempt-scoped event delivery.
  `TurnExecutionContext` captures the effective model, prompt, and tool
  bindings for each model request.

### Fixed

- **Execution finalization and recovery are more consistent.** Failed or
  cancelled attempts settle queued input and tool calls, preserve suspension
  state across durable retries, and report checkpoint and event-delivery
  failures without exposing a false successful outcome. Pending HITL cleanup
  on Postgres and MySQL is fenced by run ID to avoid clearing another run's
  request.

## [0.14.1] - 2026-09-10

### Fixed

- **The CubeLoop rename no longer renames the persistent Postgres/MySQL
  schema.** Physical tables, partitions, indexes, and the version marker remain
  `cubepi_*` at schema version 5, so an existing 0.13.6 database opens without
  DDL and historical host Alembic revisions remain reproducible.
- **The `cubepi` package is now a fail-fast tombstone.** It has no dependencies,
  exports, import aliases, or CLI proxy. Imports and commands fail with explicit
  instructions to install and migrate to `cubeloop`.

## [0.13.6] - 2026-09-04

### Fixed

- **Compaction no longer lets trailing synthetic user controls hide
  current-turn evidence.** Transform middleware can append a synthetic
  `UserMessage` after this turn's tool results. Compaction previously
  treated that control as the protected tail and a legal boundary, so
  the results immediately before it became eligible for size-only
  pruning (`[tool] N chars`) even though they were still in durable
  history. Trailing synthetic user messages are now split off before
  tail, boundary, prune, and summarize decisions; they still count
  toward the send-size threshold and are reattached unchanged (`#217`).
- **GitHub Actions workflows now declare least-privilege permissions.**
  CI and publish workflows explicitly request read-only repository
  contents access.

## [0.13.5] - 2026-08-16

### Changed

- **`FallbackBoundModel` retries the active model before hopping**, then
  sticks to the first successful leg for the rest of the agent run
  (`#211`). Transient `RateLimited` / `ProviderUnavailable` retry up to
  `max_retries_per_model=3` (4 attempts) and honour a capped
  `retry_after`. Residual `ProviderBadRequest` and `ModelNotFound` hop
  without same-model retry; `ProviderAuthFailed` and `ContentFiltered`
  stay fail-closed. Stream first-event errors now carry typed fields so
  the same predicate applies to exceptions and error events. Exhaustion
  raises `ProviderUnavailable` with `.errors` listing every leg. New
  subclasses `ModelNotFound` and `ContentFiltered` inherit
  `ProviderBadRequest`. `ProviderError.error_code` is extracted from
  vendor bodies when present.

## [0.13.4] - 2026-08-10

### Changed

- **`cubepi.run_id` now follows the agent business run id.** On
  `AgentStart`, the recorder stamps `agent.state.active_run_id` (the same
  string as `prompt(run_id=…)` / `Message.run_id`) onto every span in the
  activation. A tracer-private uuid is minted only as a fallback when
  `active_run_id` is unset (e.g. oneshot still mints its own session id).
  Hosts can filter traces with the same id they use for SSE, messages, and
  billing. OTel `trace_id` / `span_id` remain the tree identity; JSONL still
  shards by `trace_id`.

### Fixed

- **Durable HITL pauses now finalize traces as suspended rather than aborted.**
  `Agent.detach()` commits the pending request first, then the owning activation
  emits a terminal suspension event after runtime-state cleanup. Open agent,
  turn, chat, and tool spans end with `cubepi.run.outcome="suspended"`; observer
  failures cannot hide the terminal event, cancellation still propagates, and
  `respond()` opens a distinct correlated activation trace.
- **Raw tracing diagnostics now follow the content-recording opt-in.** With
  `record_content=False`, provider, turn, one-shot, and MCP spans retain typed
  error classification and ERROR status but use bounded generic descriptions
  and omit exception messages and stack traces. Stream logs retain structural
  timing/size fields while omitting tool-argument previews and raw error text.
  `record_content=True` preserves the prior diagnostic detail for failures.
- **One-shot cancellation now matches Agent and MCP cancellation semantics.** It
  records `cubepi.aborted=true` with status UNSET and does not emit an exception
  event, because cancellation is a control signal rather than a failure.

## [0.13.3] - 2026-08-02

### Fixed

- **OpenAI `chat_completions`/`responses` capability profiles no longer emit
  `reasoning_effort: "minimal"`.** OpenAI renamed the lowest reasoning-effort
  tier to `"none"` (gpt-5.5+), and some OpenAI-compatible proxies reject the
  now-stale `"minimal"` value outright. The built-in "off" mode payload and
  the `minimal` → wire-value mapping in `_OPENAI_EFFORT_VALUES` both now emit
  `"none"`.
- **Live `ToolResultMessage`s now carry the owning turn's `run_id`.** On a
  run-aware Agent, tool results emitted during the live turn (events, hooks,
  live context, and the next provider call) previously had `run_id=None`
  while the checkpointer copy was stamped correctly. The assistant message is
  now stamped in place at the message-end seam so tool-result construction
  can inherit the active run ID for every sequential, parallel, salvage, and
  HITL-sibling outcome. Direct `execute_tool_calls` with an unstamped
  assistant still produces `run_id=None`.

## [0.13.2] - 2026-07-23

### Fixed

- **`gen_ai.output.messages` now recorded on the `chat` span.** Per the
  tracing design spec (§10.3/§10.5), the provider-level `chat` span's
  `gen_ai.output.messages` should reflect what the provider actually
  returned - independent of the turn/agent-level rollup. A code comment
  in `_on_provider_response` already claimed it would record "the
  normalized output messages where derivable", but only
  `cubepi.llm.raw_response` was ever set on this span;
  `gen_ai.output.messages` was written only on `invoke_agent` and
  `cubeloop.turn`. The recorder now reconstructs the semconv
  output-message parts (text / reasoning / tool_call) directly from the
  assembled response body via a new `_derive_output_message_from_body()`
  helper, mirroring the existing three-way provider-shape dispatch
  (Anthropic-shaped, OpenAI `chat.completion`-shaped, OpenAI
  Responses-shaped). Unrecognized shapes or empty content set nothing,
  identical to prior behavior (no regression).

## [0.13.1] - 2026-07-15

### Fixed

- **Non-blocking tracing flush.** `Tracer.force_flush` and `Meter.force_flush`
  now run the synchronous provider flush in a worker thread via
  `asyncio.to_thread`, so awaiting either method never stalls the event loop
  on a slow or backlogged OTLP collector.
- **Background flush mode for `trace()`.** The `trace()` context manager
  accepts a new `flush` parameter (`"await"` | `"background"`, default
  `"await"`). With `flush="background"` the block exits immediately after
  detaching listeners while span export continues as a supervised background
  task — useful on request-serving paths where a caller is waiting on the
  block's completion. The tracer keeps a strong reference to in-flight flush
  tasks (`_pending_flushes`) so they can't be garbage-collected mid-export;
  `await tracer.shutdown()` settles any still-pending flushes before closing
  exporters, so no spans are lost on clean shutdown.

## [0.13.0] - 2026-07-05

### Added

- **Reasoning capability primitives** — `ReasoningProfile`, `ThinkingControl`,
  and `AppliedReasoningControl` types for fine-grained reasoning configuration.
  Models that support extended thinking (Anthropic extended-thinking API) now
  expose budget, level, and token tracking through the provider layer.
- **Reasoning controls render in all providers** — `BoundModel` streams now
  carry reasoning state per-turn. Rendered thinking blocks are available in
  message content for inspection and tracing.
- **Run-scoped compaction and real-token triggering** — `CompactionMiddleware`
  can now compress history *within* a single long agentic run (previously
  compression only occurred at run boundaries). Compaction triggers on true
  context fill (input + cache_read + cache_write from the last turn) instead of
  cache-blind character estimates, so prompt caching can't mask genuine
  over-limit context.

### Fixed

- **Tool-batch fault isolation.** One escaping exception in a parallel tool
  batch (HitlControlException, CancelledError, or BaseException from
  after_tool_call) used to abort the bare-await collection loop, dropping every
  ToolResultMessage in the batch — including succeeded siblings — and leaving
  dangling tool_calls in the checkpoint. Now `_execute_parallel` settles every
  task and synthesizes error results for failures, preserving all checkpoint
  state and emitting events in correct order on suspend/resume.
- **Parallel HITL approval replay.** Fixed answer ledger replay to correctly
  handle parallel tool batches with mixed approved/pending tool calls. Added
  explicit answer ledger persistence across all checkpointer backends
  (sqlite/postgres/mysql) so HITL resume paths have a durable record of prior
  approvals.
- **Checkpointer corruption detection.** Load paths now wrap per-row
  deserialization in CheckpointCorruptionError with thread_id/backend context.
  Unknown roles are now treated as corruption in all backends instead of
  silently passing through.
- **Reasoning field gating.** Fixed rendering of reasoning fields to only
  appear when `model.reasoning` is enabled, preventing type mismatches on
  models that don't support extended thinking.
- **Checkpointer v5 schema support.** Restored compatibility with v5 schema
  migrations across all backends.

## [0.12.0] - 2026-06-24

### Added

- **`CompactionMiddleware(tool_result_compressor=...)`** — a
  `Callable[[ToolResultMessage], str | None]` callback for selective tool
  result preservation during compaction. Return a `str` to preserve that
  text verbatim in the summary (for grounding/citation); return `None` to
  fall through to default pruning. Preserved results are appended to the
  summary as a labeled reference section, excluded from the summarizer
  input to save budget, and accumulated across compaction rounds via
  `CompactionState` persistence.
- **Sender attribution at the provider boundary.** `UserMessage.metadata`
  can now carry `sender_user_id` / `sender_display_name`; providers prefix
  the first text block with `[Name]:` when converting to the API format.
  Keeps stored message content clean while letting the model know who sent
  each turn in group-chat scenarios.
- **Deferred tool ordering hint.** The dispatcher description now hints
  models to emit `tool_name` before arguments, smoothing streaming UX for
  dispatch-mode deferred tools.

### Fixed

- Removed stale `(latest)` labels from Chinese 0.7 and 0.8 version docs.

## [0.11.0] - 2026-06-17

### Changed (BREAKING)

- **Deferred tool groups default to the new `dispatch` strategy.** Tool
  schemas are delivered through `load_tools` results and invoked via the
  `deferred_tool_call` dispatcher; the tools array and system prompt stay
  byte-stable, so expansions no longer invalidate the prompt cache.
  Restore the v0.10 behavior with `Agent(deferred_tool_strategy="inject")`
  / `DeferredToolsMiddleware(strategy="inject")`.
- **`DeferredToolsMiddleware(resumed_schemas=...)` and
  `ResumedState.expanded_schemas` are removed**; `prepare_resumed_state`
  takes a **required** `strategy` keyword (a default could silently resume
  an inject-mode host with hidden tools).
- **Inject mode no longer renders expanded schemas into the system
  prompt.** The definitions were already in the tools array — the
  duplicate rendering (double token billing per turn) is gone.

### Added

- **`resolve_tool_call` middleware hook** — rewrite a tool call before
  validation, `before_tool_call`, execution, events, and tracing see it.
  Composition is first-non-None-wins. Powers the deferred dispatcher;
  also usable for tool aliasing/redirection.
- **`AgentTool.expose_to_model`** — when `False`, the tool is resolvable
  and executable by the engine but its definition is never sent to the
  provider. Dispatch-mode deferred tools use this.
- **`Agent(deferred_tool_strategy=...)`** and
  `DeferredToolsMiddleware(strategy=...)` — choose `"dispatch"` (default)
  or `"inject"`.
- Resolved dispatcher calls that fail argument validation get the tool's
  **full schema appended to the error result**, so the model can
  self-correct in one round trip.

### Fixed

- **HITL resume short-circuit now emits `HitlAnswerEvent`.** Previously,
  `_await_answer`'s resume path returned the pre-loaded answer without
  emitting the event, so subscribers (e.g. IM outbound tailers) never
  learned the question was answered.
- **Explicit `resolve_tool_call` composes with middleware resolvers**
  instead of replacing the chain. An explicit resolver passed to
  `Agent(resolve_tool_call=...)` becomes the chain head
  (first-non-None-wins) rather than silently disabling middleware-provided
  resolvers like the deferred dispatcher.

## [0.10.0] - 2026-06-10

### Removed (BREAKING)

- **`"minimal"` removed from `ThinkingLevel`.** `ThinkingLevel` now reads
  `Literal["off", "low", "medium", "high", "xhigh"]`; the `.minimal` field
  is gone from `ThinkingBudgets`; `THINKING_LEVELS` no longer contains it;
  Anthropic's default `level_budgets` and OpenAI Responses' `_THINKING_TO_EFFORT`
  no longer map it. **Callers that previously passed `thinking="minimal"`
  must switch to `thinking="low"` (or `"off"`).** Rationale: DeepSeek's
  Anthropic-shape endpoint rejects `effort=minimal` on `output_config`,
  and OpenAI's `reasoning.effort` path rewrote it to `"low"` downstream
  anyway — keeping it was a footgun that surfaced as a 400 + fallback.

### Added

- **`synthetic_user_message(text, *, source) -> UserMessage`** and
  **`is_synthetic_message(message) -> bool`** — public marker for
  framework-injected user-role messages. Middleware-injected nudges
  (todo guard errors, goal continuations, compaction summaries,
  `generate_structured` retry feedback) now stamp
  `metadata["synthetic"] = True` so downstream UIs can tell internal
  scaffolding apart from real human input. Real `Agent.prompt()` /
  `Agent.steer()` messages remain unmarked. Closes #171. Exported from
  `cubepi` and `cubeloop.providers`. Use this factory (not bare
  `UserMessage`) when returning messages from `TurnAction.inject_messages`
  or `on_run_end`.

- **`DeferredToolGroup` / `DeferredToolsMiddleware`** — progressive tool
  disclosure primitive. Hides MCP tool schemas from the model by default,
  injecting a compact catalog into the system prompt instead. The model
  expands groups on demand via the built-in `load_tools` tool (full or
  selective). Key properties:
  - Catalog sorted by `group_id` for byte-stable system prompt prefix.
  - Expanded schemas append-only (expansion order, never reordered) for
    prompt-cache prefix stability across turns.
  - Loader called once per group per run; selective expansions filter from
    the cached result.
  - `Agent(deferred_tool_groups=[...])` — primary API. Middleware is
    auto-created internally with `extra_ref` bound to `self._extra`.
  - Cross-run replay via `DeferredToolsMiddleware.prepare_resumed_state()`,
    which returns pre-loaded tools, remaining groups, and expanded schemas
    for prompt-cache continuity.
  - Exported from `cubepi.deferred` as `DeferredToolGroup`,
    `DeferredToolsMiddleware`, and `ResumedState`.

- **`tool_choice` on Provider** — new `tool_choice: ToolChoice | None`
  parameter on `BoundModel.stream()`, `BoundModel.generate()`, and the
  `Provider` protocol. Accepts `"auto"`, `"required"`, `"none"`, or a
  specific tool name string. Each built-in provider maps the value to its
  native wire format (Anthropic: `{"type": "any"}` for `"required"`,
  OpenAI: `"required"`, etc.). `FauxProvider` accepts and ignores the
  parameter. Type alias: `ToolChoice = Literal["auto", "required",
  "none"] | str`, exported from `cubepi.providers.base`.

- **`BoundModel.generate_structured()`** — tool-based structured output.
  Pass a Pydantic `BaseModel` subclass and get a validated instance back:

  ```python
  from pydantic import BaseModel

  class Sentiment(BaseModel):
      label: str
      confidence: float

  result = await model.generate_structured(
      Sentiment,
      messages=[UserMessage(content=[TextContent(text="Great product!")])],
  )
  ```

  Injects a synthetic tool from the model's JSON schema, forces the call
  via `tool_choice`, and validates the response with
  `output_type.model_validate()`. Retries on validation failure (configurable
  `max_retries`, default 1). Raises `StructuredOutputError` on no tool call
  or validation exhaustion.

- **`GoalMiddleware`** — autonomous goal-driven agent runs. A separate
  evaluator model judges whether a `/goal` condition has been met after
  each worker run (dual-model architecture — the agent isn't grading its
  own homework). Continues until the evaluator confirms or
  `max_evaluations` is hit. Outcome in `agent.state.extra["goal"]`.

  ```python
  from cubepi.middleware.goal import GoalMiddleware

  goal = GoalMiddleware(
      evaluator=provider.model("claude-haiku-4-5-20251001"),
      max_evaluations=10,
  )
  agent = Agent(model=provider.model("claude-sonnet-4-6"), middleware=[goal])
  await agent.prompt("/goal all tests pass")
  ```

  Exported from `cubepi.middleware` as `GoalMiddleware`.

### Changed

- **`on_run_end` fires on every outer-loop iteration** instead of once per
  `prompt()` call. The `_reflection_fired` single-fire guard has been
  removed. Existing middlewares that return `None` after one injection are
  unaffected. This enables evaluation loops like `GoalMiddleware`.

### Changed

- **Internal logging now uses stdlib `logging` exclusively.** Previously
  `FallbackBoundModel` and the provider listener-exception path tried to
  import `loguru` first and fell back to stdlib. The loguru path was
  silently incorrect — loguru does not perform `%s` argument substitution,
  so failover warnings rendered literal `%s` placeholders instead of the
  resolved labels. cubepi has never declared loguru as a dependency; hosts
  that prefer loguru should intercept stdlib logging records into it. No
  public API change.

### Fixed

- **`FallbackBoundModel` failover log line now substitutes its placeholders.**
  Before the loguru removal above, the WARNING emitted on every failover
  read `failed=%s  →  next=%s  reason=%s  attempt=%s/%s` literally because
  the loguru-backed logger ignored the positional args. Now renders as
  `failed=anthropic/claude-opus-4-5  →  next=openai/gpt-5  reason=…  attempt=1/2`.

- **`Recorder.attach()` and `Meter.attach()` now subscribe to every provider in
  a `FallbackBoundModel` chain** (closes #167). Previously they only listened
  to `chain[0].provider`, so post-failover calls executed against `chain[1..]`
  were invisible to provider-level observability — chat spans, token usage,
  cache metrics, and cost telemetry were missing for fallback legs. Adds a
  new `cubepi.providers.fallback.chain_providers()` helper used by both
  attach paths to walk and dedupe the chain. Agent-event-driven observability
  (e.g. cost middleware reading `MessageEvent`) was already correct and is
  unchanged.

## [0.9.0] - 2026-06-08

### Added

- **`TodoListMiddleware`** — built-in task-tracking middleware for multi-step
  agents. Adds a `write_todos` tool that lets the model maintain a structured
  checklist (`pending` / `in_progress` / `completed`). Includes:
  - **Finalization guard** — if the model delivers a plain-text final response
    while items remain unfinished, it is looped back once to update the list
    before the run ends.
  - **Stale-todo reminder** — a soft `UserMessage` is injected after several
    turns without a `write_todos` call, prompting the model to keep the list
    in sync without blocking.
  - **Parallel-call guard** — if the model calls `write_todos` more than once
    in a single turn, the duplicates are rejected and the checklist is rolled
    back to its pre-turn state.
  - State (`todos`, guard counters) lives in `AgentContext.extra` and survives
    checkpointing.
  - Constructor: `TodoListMiddleware(extra_ref=..., tool_description=...,
    system_prompt=...)`. `extra_ref` must return the live `AgentContext.extra`
    dict (same object, not a copy) so the tool executor can write into it.
  - Exported from `cubepi.middleware` as `TodoListMiddleware`, `Todo`,
    `WriteTodosInput`, and `TodoGuardBlocked`.

- **`FallbackBoundModel`** — built-in failover chain at the `BoundModel` level.
  Wrap an ordered `chain` of `BoundModel` instances; on `RateLimited`,
  `ProviderUnavailable`, or `ContextLengthExceeded` (configurable via
  `trigger_errors`), or on a first-event stream error, the next model in the
  chain is tried transparently. Optional `on_failover` callback for
  billing/metrics hooks. Exported from `cubepi` and `cubeloop.providers`.

- **`DEFAULT_TRIGGER_ERRORS`** — `frozenset({RateLimited, ProviderUnavailable,
  ContextLengthExceeded})`. The default set of error types that trigger failover
  in `FallbackBoundModel`.

- **`BoundModel.generate()` / `BoundModel.stream()`** — the handle returned by
  `provider.model(...)` now drives a provider call directly. Useful for
  utilities (summarizers, classifiers) where you already hold a `BoundModel`
  and want to skip the agent loop:

  ```python
  bound = provider.model("claude-sonnet-4-6")
  reply = await bound.generate(
      messages=[UserMessage(content=[TextContent(text="hi")])],
      system_prompt="Be brief.",
  )
  ```

  Both methods forward to the bound provider with `model=bound.spec` and
  mirror the `Provider.generate` / `Provider.stream` signatures exactly.

### Breaking

- **`Middleware.extra_llm_calls()` returns `Iterable[BoundModel]`** instead
  of `Iterable[tuple[Provider, Model]]`. Third-party middleware overriding
  this hook must update the return shape (see Migration). The recorder
  consumer in `cubeloop.tracing` was adapted in lock-step; built-in
  `CompactionMiddleware` already updated.
- **`cubepi.middleware.compaction.summarizer.summarize()` takes
  `model: BoundModel`** instead of separate `provider: Provider, model: Model`
  kwargs. Direct callers (rare — this is internal to `CompactionMiddleware`)
  must wrap the pair. The public `CompactionMiddleware(summary_model=...)`
  API is unchanged.
- **`cubepi.run_agent_loop` and `cubepi.run_agent_loop_continue` take
  `model: BoundModel`** instead of separate `provider: Provider, model: Model`
  kwargs. Stateless-loop callers driving the loop outside of `Agent` must
  update. The `Agent` API is unchanged — it already took `model: BoundModel`.

### Migration

- Middleware authors overriding `extra_llm_calls()`:

  ```python
  from cubepi.providers.base import BoundModel

  # Before
  def extra_llm_calls(self):
      return [(self._provider, self._model_spec)]

  # After — either build one explicitly…
  def extra_llm_calls(self):
      return [BoundModel(provider=self._provider, spec=self._model_spec)]

  # …or, if your middleware already holds a BoundModel (recommended),
  # just return it:
  def extra_llm_calls(self):
      return [self._bound_model]
  ```

- Direct `summarize()` callers (uncommon):

  ```python
  # Before
  await summarize(provider=provider, model=model_spec, ...)

  # After
  await summarize(model=BoundModel(provider=provider, spec=model_spec), ...)
  ```

- Stateless-loop callers (uncommon — most users build an `Agent`):

  ```python
  # Before
  await run_agent_loop(
      prompts=[...],
      context=ctx,
      provider=provider,
      model=model_spec,
      convert_to_llm=...,
      emit=...,
  )

  # After
  await run_agent_loop(
      prompts=[...],
      context=ctx,
      model=provider.model("id", ...),
      convert_to_llm=...,
      emit=...,
  )
  ```

### Fixed

- **`StructuredValue` fields now preserve `BaseModel` payloads on
  serialization.** Fields typed `StructuredValue` (the
  `JsonPrimitive | BaseModel | list | dict` union used by tool-result
  `details`, `AgentToolResult`, `HitlAnswerEvent.answer`, and compaction
  message-ref hashing) silently serialized `BaseModel` instances to `{}`
  on `model_dump()`. Pydantic's union dispatch picks the dump schema from
  the declared base, not the runtime subclass, so the concrete instance's
  fields were ignored with no error or warning — data gone. Annotating the
  `BaseModel` branch with `SerializeAsAny[BaseModel]` fixes the silent loss
  across all five affected sites: checkpointer save, compaction state,
  `ToolExecutionEndEvent`, `ToolExecutionUpdateEvent`, and `HitlAnswerEvent`.

- **`SubagentMiddleware` now strips checkpointed-HITL elements from a child
  agent's inherited tools / middleware.** Previously, passing the parent
  agent's `ask_user_tool(channel)` in `shared_tools` (the common pattern
  when the host wants tools shared between parent and children) caused
  the child's first `prompt()` to raise `Agent has checkpointed HITL
  elements bound to run_ids ...` because the binding's parent `run_id`
  didn't match the child's fresh `run_id`. The middleware now drops any
  element whose `.hitl` is a checkpointed `HitlBinding` before
  constructing the child — the subagent runs autonomously without the
  parent's HITL channel, matching its "ephemeral and autonomous" design
  intent. Elements without `.hitl`, or with non-checkpointed bindings,
  are inherited as-is.

## [0.8.0] - 2026-06-06

### Added

- **`@tool` decorator** — `cubepi.tool` builds an `AgentTool` from a plain
  async function: the input schema is generated from the typed parameters
  (honouring `Field(...)` metadata), the docstring becomes the description,
  and the loop-supplied `tool_call_id` / `signal` / `on_update` are injected
  only when declared. Tools may return a `str`, a `Content`, a `list` of
  content, or a full `AgentToolResult`. The longhand `AgentTool(...)` remains
  fully supported.
- **Conversation fork** — fork a thread at a completed-run boundary, or run
  a one-shot ephemeral continuation against a snapshot:
  - `Agent.fork(src, new, *, after_run_id, metadata=None)` — physical-copy
    fork at a completed-run boundary.
  - `Agent.fork_once(src, message, *, after_run_id) -> ForkOnceResult` —
    single-turn ephemeral continuation, no checkpointer writes.
  - `Agent.prompt(message, *, run_id=None) -> str` now accept-or-generates
    the `run_id` and returns it.
  - `Agent.state.active_run_id` exposes the in-flight `run_id`.
  - `Agent(messages=...)` constructor arg for ephemeral pre-seeded history.
  - `Checkpointer.snapshot`, `fork`, `claim_run`, `mark_run_complete`,
    `load_pending` Protocol methods.
  - `cubepi_runs` table per backend (Postgres / MySQL schema v3 → v4).
  - `Message.run_id: str | None` field on all three Message variants.
  - `HitlBinding` attribute on `AgentTool` / `Middleware`; `ask_user_tool`
    and `ApprovalPolicyMiddleware` populate it.

### Fixed

- **Tool results that set `is_error=True` are now reported as errors.** A tool
  body that returned `AgentToolResult(is_error=True)` without raising was
  surfaced to the model as a successful result; the flag is now honored on the
  execution path (affects both `@tool` and longhand `AgentTool` tools).

### Breaking

- `Agent.prompt()` return type changed from `None` to `str`. Callers ignoring
  the return value keep working.
- `Checkpointer` Protocol gained 5 new methods. Third-party v3-only
  checkpointers continue to work for vanilla `prompt()` via degraded mode;
  fork APIs raise `CheckpointerError` on such backends.

### Migration

- Postgres / MySQL: run the new alembic helper (see backend guides).
- SQLite: auto-migration at connect time.
- Legacy `run_id=NULL` messages remain readable; threads with only such
  messages are not forkable.

## [0.7.0] - 2026-06-05

### Added

- **`Provider.generate(...)` one-shot helper** — providers now expose a
  non-streaming call that consumes `stream()` and returns the final
  `AssistantMessage`. It supports per-call overrides for
  `max_output_tokens`, `temperature`, `thinking`, and `thinking_budgets`.
- **Compaction middleware** — `cubepi.middleware.CompactionMiddleware`
  summarizes older conversation turns into JSON-safe `AgentContext.extra`
  state and sends the model a compressed view while preserving full agent
  history.
- **Subagent middleware** — `cubepi.middleware.SubagentMiddleware` adds a
  `subagent` tool that runs an ephemeral child `Agent`, supports shared tools
  and middleware inheritance, captures child events, and exposes host callbacks
  for application-specific event streaming.
- **Typed provider error taxonomy** — built-in providers wrap SDK failures in
  `ProviderError` subclasses such as `ContextLengthExceeded`, `RateLimited`,
  `ProviderAuthFailed`, `ProviderUnavailable`, and `ProviderBadRequest`.
- **`Tracer.oneshot(...)`** — trace a single background LLM call without a full
  `Agent` loop; the trace CLI can filter these runs via
  `--meta oneshot_operation=...`.
- **Checkpointed HITL run IDs** — checkpointers now persist the owning
  `run_id` atomically with pending HITL requests so hosts can resume the
  correct detached run after a restart.
- **`on_run_end` middleware hook** — fires exactly once after all turns and tool
  calls complete, before `AgentEndEvent`. Return a `list[Message]` to inject
  additional messages and run one extra model turn (e.g. a memory-reflection
  pass); return `None` to do nothing.
  - `_reflection_fired` guard in the loop prevents the injected turn from
    re-triggering `on_run_end`.
  - `should_stop_after_turn` and `turn_action.decision == "stop"` paths now
    route through `on_run_end` before emitting `AgentEndEvent`. Error/aborted
    runs (`stop_reason in ("error", "aborted")`) skip the hook.
  - HITL-interrupted runs (HitlDetached / HitlAborted) also skip the hook —
    the conversation is paused, not finished.
- **Compaction summarizer tracing** — `CompactionMiddleware`'s summary call is
  now first-class in the trace tree. A `cubepi.compaction.summarize` parent
  span (carrying `cubepi.compaction.message_count`) wraps the summarizer LLM
  call, and the recorder auto-subscribes the middleware's `summary_provider`
  so its `chat` span lands as a child:
  ```
  invoke_agent
  └── cubeloop.turn
      ├── cubepi.compaction.summarize
      │   └── chat <summary-model>
      └── chat <main-model>
  ```
  Previously summarizer calls bypassed the trace entirely.
- **`Middleware.providers()` protocol** — middleware authors can override this
  method to expose any extra `BaseProvider` instances the middleware owns;
  `Recorder.attach()` walks `agent._middleware` and wires its listener registry
  on each one (id-deduped against `agent._provider`). Default is empty, so
  existing custom middleware needs no change.

### Changed

- **Breaking:** agent construction now takes a bound model from
  `provider.model(...)`. Replace
  `Agent(provider=provider, model=Model(...))` with
  `Agent(model=provider.model("model-id", ...))`. `provider_id` now lives on
  provider constructors and is copied into model metadata used for tracing,
  response metadata, and error messages. `Tracer.oneshot(...)`,
  `CompactionMiddleware`, and `SubagentMiddleware` use the same bound-model
  shape.
- **Breaking:** pre-model middleware hooks now receive `AgentContext` directly.
  Update custom middleware and explicit hook callables from the old signatures:
  - `transform_context(messages, *, signal=None)`
  - `transform_system_prompt(system_prompt, *, signal=None)`
  - `convert_to_llm(messages)`

  to the new signatures:
  - `transform_context(messages, *, ctx, signal=None)`
  - `transform_system_prompt(system_prompt, *, ctx, signal=None)`
  - `convert_to_llm(messages, *, ctx)`

  Use `ctx.extra` for middleware state that should survive checkpointing. This
  release does not include old-signature compatibility shims.
- **Breaking:** custom providers must implement `Provider.generate(...)` or
  inherit from `BaseProvider`, which supplies `generate()` by consuming
  `stream()`.
- **Breaking:** Postgres and MySQL checkpointer schemas are now version 3. Host
  applications must add the nullable `run_id` column to `cubepi_threads` and
  call `write_schema_version_op()` in their Alembic migration before using this
  release.
- **Breaking:** the `cubepi.providers.images` surface has been redesigned
  to align with the chat-provider 0.7 conventions: providers now take
  `provider_id` and an optional `ImagesCapabilityDescriptor`; models are
  built via `provider.model("id", ...)` (renamed `provider` field to
  `provider_id`, added `default_size/n/quality/output_format` and `cost`
  metadata); `ImagesContext` is typed (`size/n/quality/output_format/seed/
  negative_prompt/steps/guidance/extra`); per-call options live on a new
  `ImagesOptions` bag (`signal`, `on_payload`, `on_response`); failures
  raise `cubepi.errors.ProviderError` subclasses instead of in-band
  `AssistantImages.error_message`; the `create_images_provider` /
  `register_images_provider_class` registry is removed. The new shape
  reaches OpenAI, Doubao Seedream, SiliconFlow, and Together AI through
  a single `OpenAIImagesProvider` configured with the right
  `ImagesCapabilityDescriptor`.
- CI now runs `mypy cubepi` in addition to pytest and ruff.

### Fixed

- Tool argument `ValidationError`s are formatted as model-readable tool
  results, including literal and extra-field errors.
- Provider stream-level SDK exceptions are classified into typed cubepi errors
  instead of leaking raw vendor exception types.
- `Tracer.oneshot()` now closes chat spans on failure/cancellation, awaits
  stream completion and flushes, forwards abort signals, and surfaces silent
  producer failures.
- **Anthropic empty-assistant recovery** — a persisted
  `AssistantMessage(content=[], stop_reason="error")` (e.g. from a transient
  provider failure that the checkpointer recorded) no longer poisons the
  conversation:
  - Trailing empty assistant turns are dropped before send (the next request
    regenerates from the preceding user prompt instead of treating the
    placeholder as an assistant prefill).
  - Mid-history empty assistant content is replaced with an `[empty response]`
    text block so the wire payload satisfies the Anthropic API's
    `messages.N: all messages must have non-empty content` rule.
  - The trim happens in `stream()` before the cache policy resolves
    breakpoints, so the user's last-message `cache_control` marker survives
    on retry paths.
- **Tracing root attribution under middleware-driven providers** — when a
  middleware provider (e.g. `CompactionMiddleware.summary_provider`) issues
  the first chat span of a run before the agent's main call, the root
  `invoke_agent` span's `gen_ai.provider.name` / `cubepi.agent.system_prompt_sha256`
  / `cubepi.agent.tools` are no longer overwritten by the middleware's values.
  The recorder now gates root attribution by model identity (not listener
  identity), and falls back to first-call-wins when a middleware declares the
  same `(provider, model)` pair as the agent's main.
- The compaction summarizer's wrapper span now installs `turn_span` as the
  OTel current span, so `cubepi.compaction.summarize` lands as a child of
  `cubeloop.turn` instead of becoming an orphan root.

### Removed

- Removed unused `PyYAML` from the default dependency set; the core dependency
  set is back to `anthropic`, `openai`, and `pydantic`.

## [0.6.0] - 2026-05-31

### Added

- **Human-in-the-Loop (HITL)** — first-class suspend/resume for agent runs that
  need a human decision mid-flight:
  - `HitlChannel` protocol with `InMemoryChannel` and `CheckpointedChannel`
    implementations. `CheckpointedChannel` persists pending requests so an
    agent can be resumed after a process restart.
  - `Agent(channel=...)` wiring; `agent.respond()`, `agent.detach()`, and
    `agent.abort_pending()` for external controllers.
  - `ApprovalPolicyMiddleware` and `ConfirmToolCallMiddleware` for declarative
    tool-call gating; `ask_user` built-in tool for open-ended prompts.
  - `ScriptedChannel` and `NoopChannel` for deterministic testing.
  - `HitlRequestEvent`, `HitlAnswerEvent`, `AgentSuspendedEvent`,
    `AgentAbortedEvent` stream events.
  - Lazy OTel `hitl.ask` / `hitl.confirm` spans with outcome attributes.
  - Pending-request persistence on Memory, SQLite, Postgres, and MySQL
    checkpointers (schema v2 — additive migration, backwards compatible).
- **`MySQLCheckpointer`** — full-featured MySQL/MariaDB checkpointer with
  Alembic schema management, matching the Postgres implementation.
- **Stream recording + `trace convert`** — `record_stream()` captures a raw
  provider `MessageStream` to JSONL; `cubeloop trace convert` replays the
  recording as a structured trace. Useful for offline debugging and testing
  without live API calls.
- **Trace CLI — run-metadata filtering** (`--meta` / `--show-meta`): filter
  and display `tracing_context` key/value tags attached to a run.
- **Trace CLI — span IDs in `trace view`** node labels for easier
  cross-referencing with external OTLP backends.
- **Agent steering by ID** — `agent.steer(...)` returns a `steer_id`; pass it
  to `agent.cancel_steer(steer_id)` to cancel a not-yet-drained steering
  message.

### Fixed

- **OpenAI provider**: deduplicate `finish_reason` processing that caused
  spurious extra events on streamed responses.
- **Agent**: `ToolExecutionStartEvent` is now deferred until the tool coroutine
  is actually scheduled, preventing early events for tools that are never run.
- **Tracing**: clear `stream_tool_accumulated` at turn start to avoid stale
  tool-call data leaking across turns.

## [0.5.0] - 2026-05-25

### Added

- **Image generation subsystem** (`cubepi.providers.images`): a pluggable,
  per-vendor model interface for image generation, with a class factory so new
  backends slot in without touching call sites.
- **`CapabilityDescriptor`**: a declarative, per-model description of what a
  model supports — temperature mode (free / fixed / ignored), reasoning-level
  mapping (int budget / effort / enum), and the `max_tokens` field name. It now
  drives the OpenAI, OpenAI Responses, Anthropic, and DeepSeek providers, and is
  exported from the top-level package.
- **`cubeloop trace` CLI** (install with the `trace-cli` extra): discover, list,
  view, follow, and aggregate stats over local agent-run traces, with rich
  rendering and run-id prefix matching.
- **Tracing**: an OTLP exporter and a best-effort `trace()` scope helper. The
  tracing package now imports cleanly without `opentelemetry` installed.
- **Self-describing provider errors** that carry provider / model / cause
  context for easier debugging.

### Changed

- Provider reasoning/thinking and temperature handling is now driven by
  `CapabilityDescriptor` instead of per-provider ad-hoc payload quirks, giving
  consistent behavior across OpenAI, Anthropic, and DeepSeek.

### Fixed

- **Anthropic**: merge parallel tool results into a single user message; carry
  parsed tool arguments through the streaming `toolcall_end` event; compute
  `max_tokens` from the actual capability budget; honor per-request
  `thinking_budgets` overrides.
- **Agent loop / steering**: drain steering at the turn boundary;
  `after_model_response` now injects after tool results; backfill tool results
  for tool calls orphaned by a cancel.
- **DeepSeek**: correct reasoning-effort path and temperature range handling.

## Earlier releases

- **[0.4.0]** - 2026-05-19 — see the [release notes](https://github.com/cubeplexai/cubepi/releases/tag/v0.4.0).
- **[0.3.0]** - 2026-05-14 — see the [release notes](https://github.com/cubeplexai/cubepi/releases/tag/v0.3.0).
- **[0.2.0]** - 2026-05-10 — see the [release notes](https://github.com/cubeplexai/cubepi/releases/tag/v0.2.0).
- **[0.1.0]** - 2026-05-09 — initial release. See the [release notes](https://github.com/cubeplexai/cubepi/releases/tag/v0.1.0).

[Unreleased]: https://github.com/cubeplexai/cubeloop/compare/v0.15.0...HEAD
[0.15.0]: https://github.com/cubeplexai/cubeloop/compare/v0.14.1...v0.15.0
[0.14.1]: https://github.com/cubeplexai/cubeloop/compare/v0.13.6...v0.14.1
[0.13.6]: https://github.com/cubeplexai/cubeloop/compare/v0.13.5...v0.13.6
[0.13.5]: https://github.com/cubeplexai/cubeloop/compare/v0.13.4...v0.13.5
[0.13.4]: https://github.com/cubeplexai/cubeloop/compare/v0.13.3...v0.13.4
[0.13.3]: https://github.com/cubeplexai/cubeloop/compare/v0.13.2...v0.13.3
[0.13.2]: https://github.com/cubeplexai/cubeloop/compare/v0.13.1...v0.13.2
[0.13.1]: https://github.com/cubeplexai/cubeloop/compare/v0.13.0...v0.13.1
[0.13.0]: https://github.com/cubeplexai/cubeloop/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/cubeplexai/cubeloop/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/cubeplexai/cubeloop/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/cubeplexai/cubeloop/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/cubeplexai/cubeloop/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/cubeplexai/cubeloop/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/cubeplexai/cubeloop/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/cubeplexai/cubeloop/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/cubeplexai/cubeloop/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/cubeplexai/cubeloop/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/cubeplexai/cubeloop/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/cubeplexai/cubeloop/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cubeplexai/cubepi/releases/tag/v0.1.0
