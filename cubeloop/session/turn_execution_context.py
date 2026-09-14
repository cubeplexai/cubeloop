from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

from cubeloop.agent.types import AgentTool
from cubeloop.providers.base import Message, Model, ReasoningControl, ToolDefinition


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
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

    @classmethod
    def capture(cls, message: Message) -> MessageView:
        return cls(
            role=message.role,
            content=tuple(_freeze(item) for item in message.content),
            metadata=_freeze(message.metadata),
            run_id=message.run_id,
        )


@dataclass(frozen=True)
class ToolExecutionBinding:
    name: str
    definition: ToolDefinition
    tool: AgentTool


@dataclass(frozen=True)
class TurnExecutionContext:
    turn_id: str
    run_id: str
    attempt_id: str
    model: Model
    reasoning: ReasoningControl
    system_prompt: str
    messages: tuple[MessageView, ...]
    tools: tuple[ToolExecutionBinding, ...]
    policy_revision: str | None = None

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
    ) -> TurnExecutionContext:
        return cls(
            turn_id=turn_id,
            run_id=run_id,
            attempt_id=attempt_id,
            model=model.model_copy(deep=True),
            reasoning=reasoning.model_copy(deep=True),
            system_prompt=system_prompt,
            messages=tuple(MessageView.capture(message) for message in messages),
            tools=tuple(
                ToolExecutionBinding(
                    name=tool.name,
                    definition=tool.to_definition().model_copy(deep=True),
                    tool=tool,
                )
                for tool in tools or []
            ),
            policy_revision=policy_revision,
        )

    def binding_for(self, name: str) -> ToolExecutionBinding | None:
        return next((binding for binding in self.tools if binding.name == name), None)

    def extend(self, tool: AgentTool) -> TurnExecutionContext:
        existing = self.binding_for(tool.name)
        if existing is not None:
            if existing.tool is not tool:
                raise ValueError(
                    f"turn execution context already binds tool {tool.name!r}"
                )
            return self
        binding = ToolExecutionBinding(
            name=tool.name,
            definition=tool.to_definition().model_copy(deep=True),
            tool=tool,
        )
        return replace(self, tools=(*self.tools, binding))
