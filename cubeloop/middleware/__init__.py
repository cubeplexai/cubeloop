from __future__ import annotations

from typing import Any

from cubeloop.middleware.base import Middleware, TurnAction, compose_middleware

__all__ = [
    "CompactionMiddleware",
    "CompactionState",
    "GoalMiddleware",
    "ToolResultCompressor",
    "ToolResultLimitMiddleware",
    "Middleware",
    "SubagentMiddleware",
    "SubagentRequest",
    "SubagentResult",
    "SubagentSpec",
    "Todo",
    "TodoGuardBlocked",
    "TodoGuardType",
    "TodoListMiddleware",
    "TurnAction",
    "WriteTodosInput",
    "WRITE_TODOS_SYSTEM_PROMPT",
    "WRITE_TODOS_TOOL_DESCRIPTION",
    "compose_middleware",
]

_LAZY = {
    "CompactionMiddleware": ("cubeloop.middleware.compaction", "CompactionMiddleware"),
    "CompactionState": ("cubeloop.middleware.compaction", "CompactionState"),
    "ToolResultCompressor": ("cubeloop.middleware.compaction", "ToolResultCompressor"),
    "ToolResultLimitMiddleware": (
        "cubeloop.middleware.tool_result_limit",
        "ToolResultLimitMiddleware",
    ),
    "GoalMiddleware": ("cubeloop.middleware.goal", "GoalMiddleware"),
    "SubagentMiddleware": ("cubeloop.middleware.subagents", "SubagentMiddleware"),
    "SubagentRequest": ("cubeloop.middleware.subagents", "SubagentRequest"),
    "SubagentResult": ("cubeloop.middleware.subagents", "SubagentResult"),
    "SubagentSpec": ("cubeloop.middleware.subagents", "SubagentSpec"),
    "Todo": ("cubeloop.middleware.todo", "Todo"),
    "TodoGuardBlocked": ("cubeloop.middleware.todo", "TodoGuardBlocked"),
    "TodoGuardType": ("cubeloop.middleware.todo", "TodoGuardType"),
    "TodoListMiddleware": ("cubeloop.middleware.todo", "TodoListMiddleware"),
    "WriteTodosInput": ("cubeloop.middleware.todo", "WriteTodosInput"),
    "WRITE_TODOS_SYSTEM_PROMPT": (
        "cubeloop.middleware.todo",
        "WRITE_TODOS_SYSTEM_PROMPT",
    ),
    "WRITE_TODOS_TOOL_DESCRIPTION": (
        "cubeloop.middleware.todo",
        "WRITE_TODOS_TOOL_DESCRIPTION",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, attr)
