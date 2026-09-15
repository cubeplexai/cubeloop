# Execution Session Lifecycle Implementation Plan

## Goal

Add a public, host-facing execution lifecycle without moving product-specific
durability or transport policy into CubeLoop.

## Work

1. Define typed execution requests, results, errors, input envelopes and
   receipts, event envelopes, and turn execution contexts under
   `cubeloop/session/`; export them from the package root.
2. Give every `Agent` an `ExecutionSession`; route prompt, respond, and continue
   through one fail-fast admission gate while retaining the existing internal
   loop and checkpoint behavior.
3. Implement checkpoint loading and stable live `state_context` binding without
   exposing host code to `Agent._state` or `Agent._extra`.
4. Add bounded subscriber queues, finite acknowledgement timeouts, required and
   optional consumer semantics, isolated event snapshots, monotonic sequence
   numbers, and exactly one terminal event.
5. Add concurrent steer/follow-up submission with bounded cross-attempt
   deduplication, cancellation, run-ID validation, persistence-before-commit
   ordering, and stale-queue cleanup.
6. Capture immutable effective model requests and executable tool bindings,
   including fallback-leg identity and dynamically resolved tools.
7. Return explicit completed, suspended, cancelled, failed, and incomplete
   results. Preserve pending in-memory and checkpointed HITL requests and keep
   terminal publication alive through cancellation.
8. Extend `Agent.wait_for_idle()` through session finalization.
9. Document the public lifecycle in English and Simplified Chinese, including
   the cross-process tool-catalog reconstruction boundary.

## Verification

- Unit tests cover admission, input receipt transitions, cross-attempt
  deduplication, cancellation, HITL detach/respond, event isolation and delivery
  failure, terminal publication, immutable tool bindings, and fallback capture.
- Run ruff formatting and lint, strict mypy for the changed runtime modules, the
  complete Python test suite, both Docusaurus locale builds, and Codecov patch
  coverage.
- After every pushed fix, reply to each PR comment and request another Codex
  review until the latest commit receives no actionable feedback.

## Out of scope

- Persisting Python tool callbacks or the complete `TurnExecutionContext`.
- Product-specific Redis, SSE, database, and queue adapters.
- Parallel subagent orchestration.
