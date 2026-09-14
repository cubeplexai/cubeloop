from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from cubeloop.agent.types import AgentEvent
from cubeloop.session.input import InputDurability
from cubeloop.session.types import ExecutionResult


@dataclass(frozen=True)
class InputCommitted:
    input_id: str
    durability: InputDurability
    type: Literal["input_committed"] = "input_committed"


@dataclass(frozen=True)
class ExecutionFinished:
    result: ExecutionResult
    type: Literal["execution_finished"] = "execution_finished"


SessionEvent: TypeAlias = AgentEvent | InputCommitted | ExecutionFinished


@dataclass(frozen=True)
class ExecutionEventEnvelope:
    run_id: str
    attempt_id: str
    seq: int
    event: SessionEvent
