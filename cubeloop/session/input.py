from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cubeloop.providers.base import Message

InputMode = Literal["steer", "follow_up"]
InputStatus = Literal["queued", "committed", "cancelled", "closed"]
InputDurability = Literal["memory", "checkpoint"]


@dataclass(frozen=True)
class InputEnvelope:
    input_id: str
    message: Message
    mode: InputMode


@dataclass(frozen=True)
class InputReceipt:
    input_id: str
    status: InputStatus
    durability: InputDurability | None = None
