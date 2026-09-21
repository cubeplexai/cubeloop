# Host-validated Todo task waiting

This is the CubeLoop portion of the approved CubePlex lifecycle design (cubeplexai/cubeplex#633, unit R). It does not add a waiting Session, keep a run alive, manage background tasks, or deliver their results.

## Contract

`write_todos` accepts an optional `wait_for_tasks` list. Ordinary calls keep their current checklist validation. A nonempty list requires an optional asynchronous host validator; without one it is rejected, not treated as permission to finish with unfinished work.

The validator receives task IDs, the live `AgentContext`, and the prior successful declaration when rechecking. It returns `valid`, `cancelled`, or `invalid`, with a reason and a JSON validation binding. The host owns task lookup, scope, permissions, result delivery readiness, and proof that a cancellation happened after the original declaration. CubeLoop imports no host model or task adapter.

A successful declaration is saved with the Todo snapshot, run ID, and input boundary in checkpointed extra state. It is installed only after both checklist and host validation succeed. An ordinary successful Todo update clears it. New input, including the initial message of a new run, or a different Todo snapshot invalidates it. Framework reminder messages alone do not invalidate the boundary. If context compaction cannot preserve the boundary, fail closed and require a new declaration.

Extra must be saved at completed tool-turn boundaries before the next model call, and before publishing a durable HITL suspension, not only at normal run end. Otherwise a crash or HITL handoff loses a successfully validated declaration. This uses the existing checkpoint extra API and does not add a Session lifecycle state. A failed write marks the checkpoint inconsistent and must not publish a successful suspension or completion.

Before the existing unfinished-Todo pure-text guard forces another model call, revalidate the declaration. `valid` allows natural completion while leaving unfinished items unchanged. `cancelled` does the same only for an existing, still-bound successful declaration, and stores the user's cancellation reason. It never permits creation of a new declaration. Invalid IDs, missing authority, validation failures, changed input, or an unconfigured validator retain the normal guard.

HITL answers are new input too. Their structured tool-result evidence invalidates an earlier wait even when resuming the same run; the model must handle the answer before declaring a new wait. Human approval records evidence just like denial and editing; automatic policy approval is not new human input. JSON answers restored through the resume slot use the same typed normalization as checkpoint-loaded answers.

This exemption cannot bypass malformed Todo input, parallel write restrictions, explicit stop, HITL, or another middleware's stop decision. It emits no notification, starts no run, and does not modify historical messages or a dynamic system prompt prefix.

## Evidence required

Tests use the real tool/middleware/Session path with FauxProvider and checkpoint storage: valid natural completion, unchanged unconfigured behavior, invalid/cancelled first declarations, callback failure, checkpoint restore, input-boundary invalidation, ordinary updates, parallel writes, and cancellation at finalization. Both English and current Chinese Todo docs describe the optional host contract.
