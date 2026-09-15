from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from collections.abc import Awaitable, Callable
from typing import Any, Literal, Mapping

from pydantic import BaseModel

from cubeloop.agent.types import AgentTool, AgentToolResult
from cubeloop.providers.base import Message, Model, ReasoningControl


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    if hasattr(value, "model_dump"):
        return FrozenObject(_freeze(value.model_dump(mode="python")))
    return value


@dataclass(frozen=True)
class FrozenObject:
    _values: Mapping[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True)
class MessageView:
    role: str
    content: tuple[FrozenObject, ...]
    metadata: Mapping[str, Any]
    run_id: str | None
    _values: Mapping[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @classmethod
    def capture(cls, message: Message) -> MessageView:
        values = MappingProxyType(
            {
                name: _freeze(getattr(message, name))
                for name in type(message).model_fields
            }
        )
        return cls(
            role=values["role"],
            content=values["content"],
            metadata=values["metadata"],
            run_id=values["run_id"],
            _values=values,
        )


@dataclass(frozen=True)
class ToolExecutionBinding:
    name: str
    definition: FrozenObject
    parameters: type[BaseModel]
    execute: Callable[..., Awaitable[AgentToolResult]]
    hitl_builtin: bool
    execution_mode: Literal["sequential", "parallel"] | None
    _source_id: int = field(repr=False)

    @classmethod
    def capture(cls, tool: AgentTool) -> ToolExecutionBinding:
        return cls(
            name=tool.name,
            definition=_freeze(tool.to_definition()),
            parameters=tool.parameters,
            execute=tool.execute,
            hitl_builtin=tool.hitl_builtin,
            execution_mode=tool.execution_mode,
            _source_id=id(tool),
        )


@dataclass(frozen=True)
class TurnExecutionContext:
    turn_id: str
    run_id: str
    attempt_id: str
    model: FrozenObject
    reasoning: FrozenObject
    system_prompt: str
    messages: tuple[MessageView, ...]
    tools: tuple[ToolExecutionBinding, ...]
    policy_revision: str | None = None
    model_attempt_id: str | None = None

    @classmethod
    def capture(
        cls,
        *,
        turn_id: str,
        run_id: str,
        attempt_id: str,
        model: Model,
        reasoning: ReasoningControl,
        system_prompt: str,
        messages: list[Message],
        tools: list[AgentTool] | None,
        policy_revision: str | None = None,
        model_attempt_id: str | None = None,
    ) -> TurnExecutionContext:
        return cls(
            turn_id=turn_id,
            run_id=run_id,
            attempt_id=attempt_id,
            model=_freeze(model),
            reasoning=_freeze(reasoning),
            system_prompt=system_prompt,
            messages=tuple(MessageView.capture(message) for message in messages),
            tools=tuple(ToolExecutionBinding.capture(tool) for tool in tools or []),
            policy_revision=policy_revision,
            model_attempt_id=model_attempt_id,
        )

    def matches_model(self, model: Model, model_attempt_id: str | None) -> bool:
        return (
            self.model == _freeze(model) and self.model_attempt_id == model_attempt_id
        )

    def binding_for(self, name: str) -> ToolExecutionBinding | None:
        return next((binding for binding in self.tools if binding.name == name), None)

    def extend(self, tool: AgentTool) -> TurnExecutionContext:
        existing = self.binding_for(tool.name)
        if existing is not None:
            if existing._source_id != id(tool):
                raise ValueError(
                    f"turn execution context already binds tool {tool.name!r}"
                )
            return self
        binding = ToolExecutionBinding.capture(tool)
        return replace(self, tools=(*self.tools, binding))
