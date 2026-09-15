# Execution Session Lifecycle

## Problem

`Agent.prompt()`, `respond()`, and `resume()` historically exposed the agent loop
directly. A host runtime could observe agent events, but it had no public object
that owned one execution attempt, accepted concurrent input, reported durability,
or returned a settled terminal result. Hosts therefore depended on private agent
state and could not distinguish execution failure, suspension, cancellation,
incomplete history, finalization failure, or event-delivery failure.

## Contract

Every `Agent` owns one `ExecutionSession`. The session accepts typed prompt,
respond, and continue requests. A request carries a stable `run_id` and a unique
`attempt_id`; only one attempt may execute or finalize at a time. Existing
`Agent` entry points delegate to this same admission gate.

`execute()` always settles ordinary execution and cancellation into an immutable
`ExecutionResult`. The result reports the outcome, pending HITL request when
suspended, checkpoint commitment, history consistency, a typed error, and any
consumer-delivery diagnostics. Process-control exceptions such as
`KeyboardInterrupt` and `SystemExit` are not converted.

`wait_for_idle()` covers the complete session lifecycle, including checkpoint
settlement and terminal consumer delivery. A caller released by it may start the
next attempt without racing finalization.

## Input and durability

While admission is open, a host may submit a steer or follow-up message with an
idempotency key. The receipt is `queued`, `committed`, `cancelled`, or `closed`.
The session stamps the active run and input identity, rejects conflicting run
IDs, and keeps a bounded terminal receipt ledger across attempts.

An input becomes committed immediately after the agent has appended it to its
history, before external event delivery. `InputCommitted` is then published in
normal message-event order. Failed or cancelled attempts remove undrained queue
entries before the next attempt, so stale input cannot leak into another run.
Checkpoint-backed agents report checkpoint durability; other agents report
in-memory durability.

## Event delivery

Subscribers receive monotonic, attempt-scoped `ExecutionEventEnvelope` values.
Each subscriber has a bounded queue and finite delivery timeout. Required
consumer failure stops non-terminal publication and fails the attempt, while
terminal publication is still attempted exactly once. Each consumer receives an
isolated snapshot, so it cannot mutate agent state or another consumer's view.

`ExecutionFinished` is a session lifecycle event, not a product transport's
user-facing completion event. Hosts remain responsible for projecting events to
their own durable log or SSE protocol.

## Turn execution context

Each concrete model request captures an immutable `TurnExecutionContext`:
effective model and reasoning settings, transformed prompt and messages, policy
revision, and frozen tool execution bindings. A binding fixes the schema,
validator, callback, HITL flag, and execution mode used for that request.

Retries on one fallback leg reuse its capture. A different fallback leg receives
a distinct identity even when two legs advertise identical provider and model
labels. Dynamically resolved tools extend the captured context without rebinding
an existing name.

The full context and executable callbacks are process-local. Checkpoints persist
conversation and extra state, not Python callables. A host resuming on another
process must reconstruct the same pinned/versioned tool catalog.

## HITL

Detach is valid only at a pending HITL safe point. The session preserves the
pending request from the committed suspension event, including for an
`InMemoryChannel`, and returns it in the suspended result. Respond requests
validate the recovered run ID under the execution lock before consuming an
answer.

## Compatibility

No storage schema, provider wire format, or existing agent event type changes.
Existing `Agent` methods remain available and delegate through the session.
