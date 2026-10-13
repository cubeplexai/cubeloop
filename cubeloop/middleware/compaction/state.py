from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field

from cubeloop.providers.base import Message


class PreservedToolResult(BaseModel):
    """A tool result preserved verbatim across compaction boundaries."""

    tool_name: str
    tool_call_id: str
    text: str


class CompactionState(BaseModel):
    """JSON-safe summary state stored in ``AgentContext.extra``."""

    summary: str
    summarized_message_ids: list[str] = Field(default_factory=list)
    summarized_message_refs: list[str] = Field(default_factory=list)
    last_summarized_message_id: str | None = None
    is_fallback: bool = False
    preserved_tool_results: list[PreservedToolResult] = Field(default_factory=list)


# Bookkeeping that does NOT change what a summary of the message describes, and
# that the framework or the host rewrites after the message is stored. Hashing
# it makes the refs drift, which silently invalidates the persisted compaction
# state (`_state_matches_history` → False → the raw history is sent instead of
# the compacted view).
#
# - ``run_id``: re-stamped on every run that re-touches the message. Observed in
#   production on the framework's own injected user messages — the
#   ``todo_stale_reminder`` nudges (`metadata.synthetic_source`) carried a fresh
#   ``run_id`` each turn.
# - ``timestamp``: normalised on persistence.
# - ``metadata``: host-owned free-form bag (``UserMessage.metadata`` is
#   documented as such). Hosts put per-turn snapshots in it — CubePlex writes
#   ``memory_snapshot.captured_at``, a fresh timestamp on every turn.
#
# Impact when they *were* hashed: on a 901-message conversation 38 of 877 refs
# drifted between compaction runs, the state was dropped for 30+ turns, and the
# trigger could not see the raw history it was about to send (57,278 estimated
# vs >= 256,001 actually sent) — every request ended in
# ``ContextLengthExceeded``. See #233.
_VOLATILE_REF_FIELDS = ("run_id", "timestamp", "metadata")


def message_ref(message: Message) -> str:
    message_id = str(getattr(message, "id", "") or "")
    if message_id:
        return f"id:{message_id}"
    payload = message.model_dump(mode="json", exclude_none=True)
    for field in _VOLATILE_REF_FIELDS:
        payload.pop(field, None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def message_refs(messages: list[Message]) -> list[str]:
    return [message_ref(message) for message in messages]
