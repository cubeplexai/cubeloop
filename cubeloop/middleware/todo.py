"""TodoListMiddleware — write_todos tool + guard state machine.

Full hook surface:
    tools                   — write_todos AgentTool
    transform_system_prompt — append todo-system instructions
    transform_context       — render current todos for model (system suffix)
    after_tool_call         — hook attachment point (tool execute() handles state)
    after_model_response    — guard state machine + TurnAction control flow

All 6 PlanningState channels live in ctx.extra and are mutated via the
``extra_ref`` callback pattern shared with compaction / skills:

    extra["todos"]                       list[Todo] | None
    extra["todo_guard_retries"]          dict[TodoGuardType, int]
    extra["todo_guard_blocked"]          TodoGuardBlocked | None
    extra["todo_guard_suppressed"]       bool
    extra["todo_stale_iterations"]       int
    extra["todo_finalization_correction"] bool | None

Validation helpers and message-inspection helpers operate on cubepi
AssistantMessage / ToolResultMessage types directly.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypeAlias, cast

from typing_extensions import TypedDict

from cubeloop.agent.types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
)
from cubeloop.middleware.base import Middleware, TurnAction
from cubeloop.providers.base import (
    synthetic_user_message,
    AssistantMessage,
    Message,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    is_synthetic_message,
)
from cubeloop.types import JsonObject, StructuredValue
from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------------------------
# Tool description + system prompt (provider-neutral constants).
# ---------------------------------------------------------------------------

WRITE_TODOS_TOOL_DESCRIPTION = """Use this tool to create and manage a structured task list for your current work session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.

Only use this tool if you think it will be helpful in staying organized. If the user's request is trivial and takes less than 3 steps, it is better to NOT use this tool and just do the task directly.

## When to Use This Tool
Use this tool in these scenarios:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests todo list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
5. The plan may need future revisions or updates based on results from the first few steps

## How to Use This Tool
1. When you start working on a task - Mark it as in_progress BEFORE beginning work.
2. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation.
3. You can also update future tasks, such as deleting them if they are no longer necessary, or adding new tasks that are necessary. Don't change previously completed tasks.
4. You can make several updates to the todo list at once. For example, when you complete a task, you can mark the next task you need to start as in_progress.

## When NOT to Use This Tool
It is important to skip using this tool when:
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no benefit
3. The task can be completed in less than 3 trivial steps
4. The task is purely conversational or informational

## Task States and Management

1. **Task States**: Use these states to track progress:
   - pending: Task not yet started
   - in_progress: Currently working on (unless all tasks are completed, only one task should be in_progress)
   - completed: Task finished successfully

2. **Task Management**:
   - Update task status in real-time as you work
   - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
   - Complete current tasks before starting new ones
   - Remove tasks that are no longer relevant from the list entirely
   - IMPORTANT: When you write this todo list, you should mark your first task as in_progress immediately!.
   - IMPORTANT: Unless all tasks are completed, only one task should be in_progress.

3. **Task Completion Requirements**:
   - ONLY mark a task as completed when you have FULLY accomplished it
   - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
   - When blocked, create a new task describing what needs to be resolved
   - Never mark a task as completed if:
     - There are unresolved issues or errors
     - Work is partial or incomplete
     - You encountered blockers that prevent completion
     - You couldn't find necessary resources or dependencies
     - Quality standards haven't been met

4. **Task Breakdown**:
   - Create specific, actionable items
   - Break complex tasks into smaller, manageable steps
   - Use clear, descriptive task names

Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully
Remember: If you only need to make a few tool calls to complete a task, and it is clear what you need to do, it is better to just do the task directly and NOT call this tool at all."""  # noqa: E501

WRITE_TODOS_TOOL_DESCRIPTION += """

If your host supports background task waiting and returns task IDs, you may include
wait_for_tasks when the remaining work depends on those tasks. Keep unfinished
todos unfinished. The host must validate the task IDs; this declaration does not
start tasks or keep this run alive. Omit it for ordinary checklist updates.
"""

WRITE_TODOS_SYSTEM_PROMPT = """## `write_todos`

You have access to the `write_todos` tool to help you manage and plan complex objectives.
Use this tool for complex objectives to ensure that you are tracking each necessary step and giving the user visibility into your progress.
This tool is very helpful for planning complex objectives, and for breaking down these larger complex objectives into smaller steps.

It is critical that you mark todos as completed as soon as you are done with a step. Do not batch up multiple steps before marking them as completed.
For simple objectives that only require a few steps, it is better to just complete the objective directly and NOT use this tool.
Writing todos takes time and tokens, use it when it is helpful for managing complex many-step problems! But not for simple few-step requests.
- Unless all tasks are completed, only one task should be in_progress.

## Important To-Do List Usage Notes to Remember
- The `write_todos` tool should never be called multiple times in parallel.
- Don't be afraid to revise the To-Do list as you go. New information may reveal new tasks that need to be done, or old tasks that are irrelevant."""  # noqa: E501


class Todo(TypedDict):
    """A single todo item with content and status."""

    content: str
    status: Literal["pending", "in_progress", "completed"]


TodoGuardType: TypeAlias = Literal["finalization"]

# Number of consecutive tool-call iterations (without write_todos) before
# injecting a soft stale-todo reminder. The model is free to ignore the
# reminder; it is never a hard block.
STALE_REMINDER_THRESHOLD = 5
# Minimum iterations between successive stale reminders.
STALE_REMINDER_INTERVAL = 5


class TodoGuardBlocked(TypedDict):
    """A guard escalation payload carried across the forced end turn."""

    guard_type: TodoGuardType
    message: str


_STALE_REMINDER_TEXT = (
    "The todo list has not been updated for several iterations. "
    "If you have completed the current task or started a new one, "
    "consider calling write_todos to keep the checklist in sync. "
    "Ignore this if the current work is still part of the active task."
)


def _guard_error_message(guard_type: TodoGuardType) -> str:
    return (
        "The todo list still has unfinished items. Call write_todos to update the "
        "remaining items. Your detailed response was already delivered to the user — "
        "after updating the list, give only a brief one-sentence closing. "
        "Do not repeat or re-summarize your earlier response."
    )


def _reset_guard_retries() -> dict[TodoGuardType, int]:
    return {}


def _blocked_todo_guard_message(blocked: TodoGuardBlocked) -> str:
    return (
        "Todo synchronization is already blocked. Do not call any tools. "
        "Respond to the user with a plain-text explanation that the run could "
        f"not continue safely because: {blocked['message']}"
    )


def _unfinished_todos(todos: list[Todo] | None) -> list[Todo]:
    return [todo for todo in (todos or []) if todo["status"] != "completed"]


def _write_todos_payload_error(todos: list[Todo]) -> str | None:
    if any(not todo["content"].strip() for todo in todos):
        return "Error: Todo content cannot be empty."

    in_progress_count = sum(1 for todo in todos if todo["status"] == "in_progress")
    if in_progress_count == 0 and any(todo["status"] != "completed" for todo in todos):
        return "Error: Unless all tasks are completed, exactly one todo must be in_progress."
    if in_progress_count > 1:
        return "Error: Unless all tasks are completed, exactly one todo must be in_progress."

    return None


def _write_todos_empty_payload_error(
    todos: list[Todo],
    prior_todos: list[Todo] | None,
) -> str | None:
    if todos:
        return None
    if any(todo["status"] != "completed" for todo in (prior_todos or [])):
        return (
            "Error: Cannot replace unfinished todos with an empty list. "
            "Update the active items first."
        )
    return None


def _validated_write_todos_payload(
    tool_call: dict[str, Any],
) -> tuple[list[Todo] | None, str | None]:
    args = tool_call.get("args")
    if not isinstance(args, dict):
        return (
            None,
            "Error: Received invalid `write_todos` payload. "
            "Call the tool again with a `todos` list.",
        )

    todos = args.get("todos")
    if not isinstance(todos, list):
        return (
            None,
            "Error: Received invalid `write_todos` payload. "
            "Call the tool again with a `todos` list.",
        )

    for todo in todos:
        if not isinstance(todo, dict):
            return None, (
                "Error: Received invalid `write_todos` payload. Each todo must include "
                "`content` and `status` fields."
            )
        if not isinstance(todo.get("content"), str) or not isinstance(
            todo.get("status"), str
        ):
            return None, (
                "Error: Received invalid `write_todos` payload. Each todo must include "
                "`content` and `status` fields."
            )
        if todo["status"] not in {"pending", "in_progress", "completed"}:
            return None, (
                "Error: Received invalid `write_todos` payload. Todo status must be one of "
                "`pending`, `in_progress`, or `completed`."
            )

    return cast("list[Todo]", todos), None


# ---------------------------------------------------------------------------
# Write_todos input schema
# ---------------------------------------------------------------------------


class WriteTodosInput(BaseModel):
    """Input schema for the ``write_todos`` tool."""

    todos: list[Todo]
    wait_for_tasks: list[str] = Field(
        default_factory=list,
        description="Background task IDs to await, only with host validation enabled.",
    )


class TaskWaitBinding(BaseModel):
    """A successful host check bound to the checklist and committed input boundary."""

    task_ids: list[str]
    todos: list[Todo]
    run_id: str | None
    input_boundary: str
    validation: JsonObject = Field(default_factory=dict)


class TaskWaitValidation(BaseModel):
    """Host evidence; cancelled is only meaningful for a prior successful binding."""

    status: Literal["valid", "cancelled", "invalid"]
    reason: str = ""
    validation: JsonObject = Field(default_factory=dict)


TaskWaitValidator: TypeAlias = Callable[
    [list[str], AgentContext, TaskWaitBinding | None], Awaitable[TaskWaitValidation]
]


def _input_boundary(ctx: AgentContext) -> str:
    inputs = [
        message
        for message in ctx.messages
        if (isinstance(message, UserMessage) and not is_synthetic_message(message))
        or isinstance(message.metadata.get("input_id"), str)
        or (
            isinstance(message, ToolResultMessage)
            and isinstance(message.details, dict)
            and "hitl" in message.details
        )
    ]
    return hashlib.sha256(
        json.dumps(
            [len(inputs), inputs[-1].model_dump(mode="json") if inputs else None],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _clear_task_wait(extra: dict[str, Any]) -> None:
    # Save an explicit null: checkpointers may merge extra rather than replace it.
    extra["todo_task_wait"] = None
    extra["todo_task_wait_outcome"] = None


# ---------------------------------------------------------------------------
# Helpers for inspecting AssistantMessage
# ---------------------------------------------------------------------------


def _last_assistant_message(
    messages: list[Message],
) -> AssistantMessage | None:
    """Return the last AssistantMessage from a cubepi message list."""
    return next(
        (msg for msg in reversed(messages) if isinstance(msg, AssistantMessage)),
        None,
    )


def _submitted_write_todos_calls(
    last_assistant_msg: AssistantMessage,
) -> list[ToolCall]:
    """Return all write_todos ToolCall objects in the AssistantMessage content."""
    return [
        block
        for block in last_assistant_msg.content
        if isinstance(block, ToolCall) and block.name == "write_todos"
    ]


def _non_todo_tool_calls(
    last_assistant_msg: AssistantMessage,
) -> list[ToolCall]:
    """Return all non-write_todos ToolCall objects in the AssistantMessage content."""
    return [
        block
        for block in last_assistant_msg.content
        if isinstance(block, ToolCall) and block.name != "write_todos"
    ]


def _pure_text_assistant_response(last_assistant_msg: AssistantMessage) -> bool:
    """True if the message has text content and no tool calls of any kind."""
    has_text = any(
        isinstance(block, TextContent) and block.text.strip()
        for block in last_assistant_msg.content
    )
    has_tool_calls = any(
        isinstance(block, ToolCall) for block in last_assistant_msg.content
    )
    return has_text and not has_tool_calls


def _todo_validation_errors_local(
    last_assistant_msg: AssistantMessage,
    prior_todos: list[Todo] | None,
) -> list[dict[str, Any]]:
    """Return validation error payloads for write_todos calls in the message.

    Returns a list of dicts with ``tool_call_id`` and ``error`` keys; the
    caller turns these into cubepi-compatible inject messages.
    """
    write_todos_calls = _submitted_write_todos_calls(last_assistant_msg)
    # Zero calls: nothing to validate.
    # More than one call: handled by parallel-write_todos check upstream.
    if len(write_todos_calls) != 1:
        return []

    tool_call = write_todos_calls[0]
    # Build a dict matching what _validated_write_todos_payload expects
    call_dict: dict[str, Any] = {
        "id": tool_call.id,
        "name": tool_call.name,
        "args": tool_call.arguments,
    }

    todos, payload_error = _validated_write_todos_payload(call_dict)
    if payload_error is not None:
        return [{"tool_call_id": tool_call.id, "error": payload_error}]
    assert todos is not None

    empty_error = _write_todos_empty_payload_error(todos, prior_todos)
    if empty_error is not None:
        return [{"tool_call_id": tool_call.id, "error": empty_error}]

    error = _write_todos_payload_error(todos)
    if error is None:
        return []

    return [{"tool_call_id": tool_call.id, "error": error}]


# ---------------------------------------------------------------------------
# UserMessage factory for injected messages
# ---------------------------------------------------------------------------


def _make_user_message(text: str, *, source: str) -> UserMessage:
    """Wrap plain text in a synthetic UserMessage for injection."""
    return synthetic_user_message(text, source=source)


# ---------------------------------------------------------------------------
# write_todos tool factory
# ---------------------------------------------------------------------------


def _make_write_todos_tool(
    extra_ref: Callable[[], dict[str, Any]],
    description: str = WRITE_TODOS_TOOL_DESCRIPTION,
    prepare_wait: Callable[[str, WriteTodosInput], Awaitable[TaskWaitBinding | str]]
    | None = None,
) -> AgentTool[WriteTodosInput]:
    """Build the ``write_todos`` AgentTool that stores results in extra.

    The tool:
    1. Validates the todos payload.
    2. On success, writes ``todos`` to ``extra["todos"]`` and returns the
       standard JSON tool-result content built by ``_build_todo_tool_message``.
    3. On validation failure, returns an error result.
    """

    async def _execute(
        tool_call_id: str,
        args: WriteTodosInput,
        *,
        signal: asyncio.Event | None = None,
        on_update: Callable[[StructuredValue], None] | None = None,
    ) -> AgentToolResult:
        del on_update  # unused
        todos = args.todos

        # Build a dict in the format _validated_write_todos_payload expects
        call_dict: dict[str, Any] = {
            "id": tool_call_id,
            "name": "write_todos",
            "args": {"todos": list(todos)},
        }

        # Run structural validation (schema already validated by Pydantic above)
        validated_todos, payload_error = _validated_write_todos_payload(call_dict)
        if payload_error is not None:
            return AgentToolResult(
                content=[TextContent(text=payload_error)],
                is_error=True,
            )
        assert validated_todos is not None

        prior_todos: list[Todo] | None = extra_ref().get("todos")

        empty_error = _write_todos_empty_payload_error(validated_todos, prior_todos)
        if empty_error is not None:
            return AgentToolResult(
                content=[TextContent(text=empty_error)],
                is_error=True,
            )

        invariant_error = _write_todos_payload_error(validated_todos)
        if invariant_error is not None:
            return AgentToolResult(
                content=[TextContent(text=invariant_error)],
                is_error=True,
            )

        binding: TaskWaitBinding | None = None
        if args.wait_for_tasks:
            prepared = (
                await prepare_wait(tool_call_id, args)
                if prepare_wait is not None
                else "Error: Task waiting is not configured by the host."
            )
            if signal is not None and signal.is_set():
                raise asyncio.CancelledError()
            if isinstance(prepared, str):
                return AgentToolResult(
                    content=[TextContent(text=prepared)], is_error=True
                )
            binding = prepared

        extra = extra_ref()
        extra["todos"] = validated_todos
        _clear_task_wait(extra)
        if binding is not None:
            extra["todo_task_wait"] = binding.model_dump(mode="json")

        # Build the JSON content identical to _build_todo_tool_message
        payload: dict[str, Any] = {"todos": validated_todos}
        if binding is not None:
            payload["wait_for_tasks"] = binding.task_ids
        if len(validated_todos) >= 3 and all(
            todo["status"] == "completed" for todo in validated_todos
        ):
            payload["reminder"] = (
                "All todo items are complete. Do a quick final check before responding."
            )
        content_text = json.dumps(payload, ensure_ascii=False)
        return AgentToolResult(content=[TextContent(text=content_text)])

    return AgentTool(
        name="write_todos",
        description=description,
        parameters=WriteTodosInput,
        execute=_execute,
    )


# ---------------------------------------------------------------------------
# Guard-retry helper working on extra dict
# ---------------------------------------------------------------------------


def _guard_retry_update_extra(
    extra: dict[str, Any],
    guard_type: TodoGuardType,
) -> tuple[int, dict[TodoGuardType, int]]:
    retries: dict[TodoGuardType, int] = dict(extra.get("todo_guard_retries", {}))
    retries[guard_type] = retries.get(guard_type, 0) + 1
    return retries[guard_type], retries


# ---------------------------------------------------------------------------
# Main middleware class
# ---------------------------------------------------------------------------


class TodoListMiddleware(Middleware):
    """Middleware that gives the agent a ``write_todos`` tool and enforces finalization.

    Hooks:
    - ``tools``: exposes ``write_todos`` AgentTool that writes to extra.
    - ``transform_system_prompt``: appends WRITE_TODOS_SYSTEM_PROMPT.
    - ``transform_context``: renders current todo list as a UserMessage
      appended to context (only when todos are present), so the model
      always has an up-to-date view of the checklist at each turn.
    - ``after_tool_call``: No-op for all tools except ``write_todos``;
      the write_todos tool execute() already writes to extra.  This hook
      exists as a hook attachment point if needed in future.
    - ``after_model_response``: full guard state machine —
        * blocked guard: if blocked and pure-text → stop, clear state
        * suppression: clear guard state once past the blocked episode
        * parallel write_todos detection → inject error messages
        * payload validation errors → inject error messages
        * stale-todo soft reminder → inject UserMessage after threshold
        * finalization hard guard → loop_to_model with correction message
        * otherwise → return None (natural flow)
    """

    def __init__(
        self,
        *,
        extra_ref: Callable[[], dict[str, Any]],
        system_prompt: str = WRITE_TODOS_SYSTEM_PROMPT,
        tool_description: str = WRITE_TODOS_TOOL_DESCRIPTION,
        validate_task_wait: TaskWaitValidator | None = None,
    ) -> None:
        # extra_ref must return the agent's persisted extra dict (i.e. the same
        # object as AgentContext.extra) so that todo state survives session
        # checkpointing.  AgentTool.execute() does not receive AgentContext, so
        # extra_ref is the only way for the tool to write into that dict.
        self._extra_ref = extra_ref
        self._system_prompt = system_prompt
        self._tool_description = tool_description
        self._validate_task_wait = validate_task_wait
        self._tool_contexts: dict[str, AgentContext] = {}
        self.tools = [
            _make_write_todos_tool(
                self._extra_ref, self._tool_description, self._prepare_task_wait
            )
        ]

    async def before_tool_call(
        self,
        ctx: BeforeToolCallContext,
        *,
        signal: asyncio.Event | None = None,
    ) -> BeforeToolCallResult | None:
        if ctx.tool_call.name == "write_todos":
            self._tool_contexts[ctx.tool_call.id] = ctx.context
        return None

    async def _prepare_task_wait(
        self, tool_call_id: str, args: WriteTodosInput
    ) -> TaskWaitBinding | str:
        ctx = self._tool_contexts.pop(tool_call_id, None)
        if self._validate_task_wait is None:
            return "Error: Task waiting is not configured by the host."
        if ctx is None or ctx.extra is not self._extra_ref():
            return "Error: Task waiting requires the live tool context."
        ids = args.wait_for_tasks
        if any(not value.strip() for value in ids) or len(ids) != len(set(ids)):
            return "Error: wait_for_tasks must contain distinct nonempty task IDs."
        if not _unfinished_todos(args.todos):
            return "Error: Task waiting requires unfinished todo items."
        boundary, run_id = _input_boundary(ctx), ctx.run_id
        try:
            validation = await self._validate_task_wait(list(ids), ctx, None)
            if validation.status != "valid":
                return f"Error: Task wait was not validated. {validation.reason}"
            if boundary != _input_boundary(ctx) or run_id != ctx.run_id:
                return "Error: Input changed while task waiting was being validated."
            return TaskWaitBinding(
                task_ids=list(ids),
                todos=copy.deepcopy(args.todos),
                run_id=run_id,
                input_boundary=boundary,
                validation=copy.deepcopy(validation.validation),
            )
        except Exception:
            return "Error: The host could not validate task waiting."

    def _bound_task_wait(self, ctx: AgentContext) -> TaskWaitBinding | None:
        extra = self._extra_ref()
        raw = extra.get("todo_task_wait")
        if raw is None:
            return None
        try:
            binding = TaskWaitBinding.model_validate(raw)
        except ValidationError:
            _clear_task_wait(extra)
            return None
        if (
            not binding.task_ids
            or any(not task_id.strip() for task_id in binding.task_ids)
            or len(binding.task_ids) != len(set(binding.task_ids))
            or binding.run_id != ctx.run_id
            or binding.todos != extra.get("todos")
            or binding.input_boundary != _input_boundary(ctx)
        ):
            _clear_task_wait(extra)
            return None
        return binding

    async def _task_wait_allows_finalization(
        self, ctx: AgentContext, signal: asyncio.Event | None
    ) -> bool:
        binding = self._bound_task_wait(ctx)
        if binding is None or self._validate_task_wait is None:
            return False
        try:
            result = await self._validate_task_wait(
                list(binding.task_ids), ctx, binding.model_copy(deep=True)
            )
            if result.status not in ("valid", "cancelled"):
                return False
            if result.status == "cancelled" and not result.reason.strip():
                return False
        except Exception:
            return False
        finally:
            # The new host await must not convert an explicit Stop to completion.
            if signal is not None and signal.is_set():
                raise asyncio.CancelledError()
        # A callback can await while another input is committed. Recheck after it.
        if self._bound_task_wait(ctx) != binding:
            return False
        extra = self._extra_ref()
        extra["todo_task_wait_outcome"] = {
            "status": result.status,
            "reason": result.reason,
        }
        extra["todo_guard_retries"] = _reset_guard_retries()
        extra["todo_stale_iterations"] = 0
        extra["todo_finalization_correction"] = None
        return True

    # ------------------------------------------------------------------
    # transform_system_prompt
    # ------------------------------------------------------------------

    async def transform_system_prompt(
        self,
        system_prompt: str,
        *,
        ctx: AgentContext,
        signal: asyncio.Event | None = None,
    ) -> str:
        """Append the write_todos system instructions."""
        del ctx, signal  # not used
        return system_prompt + "\n\n" + self._system_prompt

    # ------------------------------------------------------------------
    # transform_context
    # ------------------------------------------------------------------

    async def transform_context(
        self,
        messages: list[Message],
        *,
        ctx: AgentContext,
        signal: asyncio.Event | None = None,
    ) -> list[Message]:
        """Inject current todo state as a UserMessage suffix when todos exist.

        We do not rely on persisted ToolResultMessages being visible on
        replay, so a lightweight reminder is injected at the end of the
        context to ensure the model always sees the current todo list.

        When no todos are set, messages are returned unchanged (no injection).
        """
        del ctx, signal  # not used
        extra = self._extra_ref()
        todos: list[Todo] | None = extra.get("todos")
        if not todos:
            return messages

        payload: dict[str, Any] = {"todos": todos}
        if len(todos) >= 3 and all(todo["status"] == "completed" for todo in todos):
            payload["reminder"] = (
                "All todo items are complete. Do a quick final check before responding."
            )
        todo_text = "[Current todo list]\n" + json.dumps(payload, ensure_ascii=False)
        todo_msg = synthetic_user_message(todo_text, source="todo_list_view")
        return list(messages) + [todo_msg]

    # ------------------------------------------------------------------
    # after_tool_call
    # ------------------------------------------------------------------

    async def after_tool_call(
        self,
        ctx: AfterToolCallContext,
        *,
        signal: asyncio.Event | None = None,
    ) -> AfterToolCallResult | None:
        """Override duplicate write_todos results when parallel calls are detected.

        For the normal (single call) case this is a no-op; write_todos.execute()
        already wrote the validated list to extra["todos"].

        For the parallel case, after_model_response snapshotted the pre-turn todos
        in extra["_todos_snapshot"].  Each duplicate write_todos call's result is
        replaced with an error here, and extra["todos"] is restored to the snapshot
        so the turn leaves the checklist unchanged.
        """
        del signal  # not used
        if ctx.tool_call.name != "write_todos":
            return None
        extra = self._extra_ref()
        parallel_calls = _submitted_write_todos_calls(ctx.assistant_message)
        if len(parallel_calls) <= 1:
            return None
        # Restore pre-turn state and mark every parallel write as an error.
        # Use get (not pop) so the snapshot remains available for every
        # duplicate call's after_tool_call invocation.
        extra["todos"] = extra.get("_todos_snapshot", extra.get("todos"))
        extra["todo_task_wait"] = copy.deepcopy(extra.get("_todo_wait_snapshot"))
        extra["todo_task_wait_outcome"] = copy.deepcopy(
            extra.get("_todo_wait_outcome_snapshot")
        )
        return AfterToolCallResult(
            content=[
                TextContent(
                    text=(
                        "Error: write_todos was called multiple times in parallel. "
                        "Call it only once per response."
                    )
                )
            ],
            is_error=True,
        )

    # ------------------------------------------------------------------
    # after_model_response — full guard state machine
    # ------------------------------------------------------------------

    def _parallel_write_todos_error(
        self,
        last_assistant_msg: AssistantMessage,
    ) -> list[UserMessage] | None:
        """Detect parallel write_todos calls; return error UserMessages if found."""
        write_todos_calls = _submitted_write_todos_calls(last_assistant_msg)
        if len(write_todos_calls) <= 1:
            return None
        error_text = (
            "Error: The `write_todos` tool should never be called multiple times "
            "in parallel. Please call it only once per model invocation to update "
            "the todo list."
        )
        return [
            _make_user_message(error_text, source="todo_parallel_error")
            for _ in write_todos_calls
        ]

    def _guard_response(
        self,
        extra: dict[str, Any],
        guard_type: TodoGuardType,
    ) -> TurnAction:
        """Build the TurnAction for a finalization guard trigger."""
        retry_count, retries = _guard_retry_update_extra(extra, guard_type)
        message = _guard_error_message(guard_type)

        # Write retries to extra
        extra["todo_guard_retries"] = retries

        if retry_count >= 3:
            # Escalate to blocked state
            blocked: TodoGuardBlocked = {"guard_type": guard_type, "message": message}
            extra["todo_guard_blocked"] = blocked
            extra["todo_finalization_correction"] = None
            inject_text = (
                "Todo synchronization failed repeatedly. Do not call any tools. "
                "Respond to the user with a plain-text explanation that the run "
                f"could not continue safely because: {message}"
            )
            return TurnAction(
                inject_messages=[
                    _make_user_message(inject_text, source="todo_blocked")
                ],
                decision="loop_to_model",
            )

        extra["todo_finalization_correction"] = True
        return TurnAction(
            inject_messages=[_make_user_message(message, source="todo_guard")],
            decision="loop_to_model",
        )

    async def after_model_response(
        self,
        response: AssistantMessage,
        ctx: AgentContext,
        *,
        signal: asyncio.Event | None = None,
    ) -> TurnAction | None:
        """Guard state machine.

        Inspects the latest assistant message, transitions the blocked /
        stale-iteration state stored in ``extra``, and returns a
        ``TurnAction`` when the loop needs an injected nudge or hard stop.
        """
        extra = self._extra_ref()
        self._tool_contexts.clear()
        self._bound_task_wait(ctx)

        # We need the full message list from context to inspect the last AI msg.
        # In cubepi, ctx is AgentContext which has a .messages list.
        agent_ctx_messages = ctx.messages

        # Find last assistant message; if none, there's nothing to guard against.
        last_assistant_msg = _last_assistant_message(agent_ctx_messages + [response])
        if last_assistant_msg is None:
            return None

        # --- blocked-guard state machine (finalization escalation) ----------
        blocked_guard: TodoGuardBlocked | None = extra.get("todo_guard_blocked")
        if blocked_guard:
            if _pure_text_assistant_response(last_assistant_msg):
                # The blocked state resolved: model gave a plain-text explanation.
                extra["todo_guard_blocked"] = None
                extra["todo_guard_retries"] = _reset_guard_retries()
                extra["todo_guard_suppressed"] = True
                extra["todo_stale_iterations"] = 0
                extra["todo_finalization_correction"] = None
                return TurnAction(decision="stop")

            # Still in blocked state: re-inject the blocked guard message.
            extra["todo_finalization_correction"] = None
            return TurnAction(
                inject_messages=[
                    _make_user_message(
                        _blocked_todo_guard_message(blocked_guard),
                        source="todo_blocked",
                    )
                ],
                decision="loop_to_model",
            )

        # --- suppression after escalation ----------------------------------
        write_todos_calls = _submitted_write_todos_calls(last_assistant_msg)
        if extra.get("todo_guard_suppressed") and not write_todos_calls:
            extra["todo_guard_retries"] = _reset_guard_retries()
            extra["todo_guard_suppressed"] = True
            extra["todo_stale_iterations"] = 0
            extra["todo_finalization_correction"] = None
            return None

        # --- payload validation (parallel calls, schema, invariants) --------
        parallel_errors = self._parallel_write_todos_error(last_assistant_msg)
        if parallel_errors is not None:
            # Snapshot todos BEFORE any tool executes this turn so after_tool_call
            # can restore the list when overriding duplicate write_todos results.
            # decision="natural": cubepi defers inject_messages until after
            # ToolResultMessages, keeping Anthropic-style ordering intact.
            extra["_todos_snapshot"] = extra.get("todos")
            extra["_todo_wait_snapshot"] = copy.deepcopy(extra.get("todo_task_wait"))
            extra["_todo_wait_outcome_snapshot"] = copy.deepcopy(
                extra.get("todo_task_wait_outcome")
            )
            return TurnAction(inject_messages=cast("list[Any]", parallel_errors))

        validation_errors = _todo_validation_errors_local(
            last_assistant_msg,
            extra.get("todos"),
        )
        if validation_errors:
            _clear_task_wait(extra)
            inject: list[Any] = [
                _make_user_message(e["error"], source="todo_validation_error")
                for e in validation_errors
            ]
            return TurnAction(inject_messages=inject)

        # --- stale-todo soft reminder ---------------------------------------
        unfinished = _unfinished_todos(extra.get("todos"))
        has_write_todos = bool(write_todos_calls)
        has_non_todo_tools = bool(_non_todo_tool_calls(last_assistant_msg))

        # Compute stale counter update (deferred write until we know no hard guard fires)
        stale_count_new: int | None = None
        stale_injections: list[UserMessage] = []
        if unfinished and has_non_todo_tools and not has_write_todos:
            stale_count_new = extra.get("todo_stale_iterations", 0) + 1
            if stale_count_new >= STALE_REMINDER_THRESHOLD and (
                (stale_count_new - STALE_REMINDER_THRESHOLD) % STALE_REMINDER_INTERVAL
                == 0
            ):
                stale_injections.append(
                    _make_user_message(
                        _STALE_REMINDER_TEXT, source="todo_stale_reminder"
                    )
                )
        elif has_write_todos or not has_non_todo_tools:
            # Any write_todos call or non-tool turn resets the counter.
            stale_count_new = 0

        # Compute finalization_correction update (deferred)
        clear_finalization_correction = False
        if extra.get("todo_finalization_correction"):
            if not unfinished or not has_write_todos:
                clear_finalization_correction = True

        # --- finalization hard guard ----------------------------------------
        # NOTE: _guard_response reads extra["todo_guard_retries"] directly, so
        # we must call it BEFORE resetting retries in the clean-pass section below.
        if unfinished and _pure_text_assistant_response(last_assistant_msg):
            if await self._task_wait_allows_finalization(ctx, signal):
                return None
            return self._guard_response(extra, "finalization")

        # --- clean pass: commit deferred state updates ----------------------
        extra["todo_guard_retries"] = _reset_guard_retries()
        if extra.get("todo_guard_suppressed"):
            extra["todo_guard_suppressed"] = None
        if stale_count_new is not None:
            extra["todo_stale_iterations"] = stale_count_new
        if clear_finalization_correction:
            extra["todo_finalization_correction"] = None

        if stale_injections:
            return TurnAction(inject_messages=cast("list[Any]", stale_injections))

        return None
