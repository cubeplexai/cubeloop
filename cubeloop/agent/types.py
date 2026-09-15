from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Generic, Literal, TypeVar

from pydantic import BaseModel

from cubeloop.hitl.binding import HitlBinding
from cubeloop.hitl.types import HitlRequest
from cubeloop.providers.base import (
    AssistantMessage,
    Content,
    Message,
    StreamEvent,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
)
from cubeloop.types import JsonObject, StructuredObject, StructuredValue

TParams = TypeVar("TParams", bound=BaseModel, covariant=True)
TMessage = TypeVar("TMessage")


class AgentToolResult(BaseModel):
    content: list[Content]
    details: StructuredValue = None
    is_error: bool | None = None
    terminate: bool | None = None


@dataclass
class AgentTool(Generic[TParams]):
    name: str
    description: str
    parameters: type[TParams]
    execute: Callable[..., Awaitable[AgentToolResult]]
    label: str = ""
    execution_mode: Literal["sequential", "parallel"] | None = None
    hitl_builtin: bool = False
    hitl: HitlBinding | None = None
    # When False the tool is resolvable/executable by the engine but its
    # definition is never sent to the provider (deferred dispatch mode).
    expose_to_model: bool = True

    def to_definition(self) -> ToolDefinition:
        schema = self.parameters.model_json_schema()
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=schema,
        )


@dataclass
class AgentContext:
    system_prompt: str
    messages: list[Message]
    tools: list[AgentTool] | None = None
    extra: JsonObject = field(default_factory=dict)
    run_id: str | None = None
    attempt_id: str | None = None
    turn_id: str | None = None
    turn_execution_context: Any = None
    on_turn_context: Callable[[Any], Any] | None = None
    set_input_admission: Callable[[bool], None] | None = None


# --- Hook context types ---


class BeforeToolCallResult(BaseModel):
    block: bool = False
    reason: str | None = None
    edited_args: JsonObject | None = None
    deny_reason: str | None = None
    hitl_trace: StructuredObject | None = None


class AfterToolCallResult(BaseModel):
    content: list[Content] | None = None
    details: StructuredValue = None
    is_error: bool | None = None
    terminate: bool | None = None


@dataclass
class BeforeToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCall
    args: BaseModel | JsonObject
    context: AgentContext


@dataclass
class AfterToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCall
    args: BaseModel | JsonObject
    result: AgentToolResult
    is_error: bool
    context: AgentContext


@dataclass
class ShouldStopAfterTurnContext:
    message: AssistantMessage
    tool_results: list[ToolResultMessage]
    context: AgentContext
    new_messages: list[Message]


# --- Event types (11 total, matching pi) ---


class AgentStartEvent(BaseModel):
    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(BaseModel):
    type: Literal["agent_end"] = "agent_end"
    messages: list[Message]


class TurnStartEvent(BaseModel):
    type: Literal["turn_start"] = "turn_start"


class TurnEndEvent(BaseModel):
    type: Literal["turn_end"] = "turn_end"
    message: AssistantMessage
    tool_results: list[ToolResultMessage]


class MessageStartEvent(BaseModel):
    type: Literal["message_start"] = "message_start"
    message: Message


class MessageUpdateEvent(BaseModel):
    type: Literal["message_update"] = "message_update"
    message: AssistantMessage
    stream_event: StreamEvent


class MessageEndEvent(BaseModel):
    type: Literal["message_end"] = "message_end"
    message: Message


class ToolExecutionStartEvent(BaseModel):
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    args: JsonObject


class ToolExecutionUpdateEvent(BaseModel):
    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    args: JsonObject | None = None
    partial_result: StructuredValue = None


class ToolExecutionEndEvent(BaseModel):
    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: StructuredValue = None
    is_error: bool = False
    terminate: bool = False
    """True iff the tool's AgentToolResult.terminate was True (or the
    after_tool_call hook set terminate=True). Recorders use this to mark
    the turn as terminated-by-tool without unwrapping ``result``."""
    blocked_by_hook: bool = False
    """True iff the tool call was blocked by a ``before_tool_call`` hook
    returning ``block=True``. Distinguishes hook-blocks from other
    immediate errors (tool-not-found, arg-validation failure)."""
    block_reason: str | None = None
    """When ``blocked_by_hook`` is True, the reason string from
    ``BeforeToolCallResult.reason`` (or ``None`` if the hook supplied
    no reason)."""


class HitlRequestEvent(BaseModel):
    type: Literal["hitl_request"] = "hitl_request"
    request: HitlRequest


class HitlAnswerEvent(BaseModel):
    type: Literal["hitl_answer"] = "hitl_answer"
    question_id: str
    answer: StructuredValue = None
    cancelled: bool = False
    timed_out: bool = False


class AgentSuspendedEvent(BaseModel):
    type: Literal["agent_suspended"] = "agent_suspended"
    pending_request: HitlRequest


class AgentAbortedEvent(BaseModel):
    type: Literal["agent_aborted"] = "agent_aborted"
    reason: str


AgentEvent = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
    | HitlRequestEvent
    | HitlAnswerEvent
    | AgentSuspendedEvent
    | AgentAbortedEvent
)

AgentEventSink = Callable[[AgentEvent], Awaitable[None]]

AgentListener = Callable[[AgentEvent, asyncio.Event | None], Awaitable[None] | None]


@dataclass(frozen=True)
class ForkOnceResult:
    text: str
    messages: list[Message]
    stop_reason: str
