from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from cubeloop.hitl.types import HitlRequest
from cubeloop.providers.base import Message
from cubeloop.types import StructuredValue

ExecutionOutcome: TypeAlias = Literal[
    "completed", "suspended", "cancelled", "failed", "incomplete"
]


@dataclass(frozen=True)
class PromptExecutionRequest:
    run_id: str
    attempt_id: str
    message: str | Message | list[Message]
    kind: Literal["prompt"] = "prompt"


@dataclass(frozen=True)
class RespondExecutionRequest:
    run_id: str
    attempt_id: str
    answer: StructuredValue
    question_id: str | None = None
    kind: Literal["respond"] = "respond"


@dataclass(frozen=True)
class ContinueExecutionRequest:
    run_id: str
    attempt_id: str
    kind: Literal["continue"] = "continue"


ExecutionRequest: TypeAlias = (
    PromptExecutionRequest | RespondExecutionRequest | ContinueExecutionRequest
)


@dataclass(frozen=True)
class ExecutionError:
    kind: Literal["cancelled", "execution", "finalization", "inconsistent"]
    message: str
    cause: BaseException | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class DeliveryError:
    consumer: str
    seq: int
    reason: Literal["timeout", "closed", "error"]
    message: str = ""


@dataclass(frozen=True)
class ExecutionResult:
    run_id: str
    attempt_id: str
    outcome: ExecutionOutcome
    pending_request: HitlRequest | None = None
    error: ExecutionError | None = None
    checkpoint_committed: bool = False
    history_consistent: bool = True
    delivery_errors: tuple[DeliveryError, ...] = ()


class ExecutionBusy(RuntimeError):
    """Raised when an Agent already has an executing or finalizing attempt."""
