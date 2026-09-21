# Todo task-wait implementation

Goal: let a host-validated background wait finish the current run without falsely completing the Todo list.

1. Extend `cubeloop/middleware/todo.py` with typed validation/binding records, the optional callback, and `WriteTodosInput.wait_for_tasks`. Keep the callback optional and the existing guard as the default.
2. Use the public before-tool hook to obtain the live `AgentContext`. The tool validates before atomically updating Todo and declaration extra state; no private Agent fields or new Session lifecycle APIs. Reject a waiting tool call that lacks a matching hook context.
3. Bind to the run and current input fingerprint, including committed internal inputs and HITL answer evidence. Human tool approvals carry structured evidence; resume-slot JSON answers receive the same typed normalization as checkpoint-loaded answers. Cover approved, denied, and edited tool calls through Session suspend/respond. Persist only JSON-compatible data. Recheck binding and host evidence just before the pure-text unfinished guard. Cancelled is valid only with the previous successful declaration; record the reason without generating input or forcing a model turn.
4. Ensure parallel writes cannot retain one call's wait binding after checklist rollback. Normal updates clear waiting, and changed input or malformed saved bindings fail closed.
5. Add deterministic tool and real Session/checkpoint tests in `tests/middleware/test_todo_task_wait.py`. Keep `tests/test_synthetic_messages.py` and existing middleware behavior green. Update the current Todo guides and lazy exports in the same PR.
6. Persist existing Session extra in `Agent._process_event` at tool-turn end and suspension, before publishing the event. Keep normal run-end persistence. Mark a failed extra write as a checkpoint failure; cover the next-model boundary, HITL handoff, and failed writes without changing the execution lifecycle.

Verification: focused pytest while developing, then full upstream pytest, ruff, and mypy before PR; drive the upstream PR review loop. Publishing a release and switching the host dependency are separate integration steps, not implied by this module's tests passing.
