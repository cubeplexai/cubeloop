from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import BaseModel

from cubeloop import Agent
from cubeloop.agent.types import AgentContext, AgentTool, AgentToolResult
from cubeloop.checkpointer.memory import MemoryCheckpointer
from cubeloop.hitl.ask_user import ask_user_tool
from cubeloop.hitl.channel import CheckpointedChannel
from cubeloop.hitl.middleware import ConfirmToolCallMiddleware
from cubeloop.middleware.base import Middleware, TurnAction
from cubeloop.middleware.todo import (
    TaskWaitBinding,
    TaskWaitValidation,
    TaskWaitValidator,
    TodoListMiddleware,
)
from cubeloop.providers.base import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    synthetic_user_message,
)
from cubeloop.providers.faux import FauxProvider, faux_assistant_message, faux_tool_call
from cubeloop.session import (
    InputEnvelope,
    PromptExecutionRequest,
    RespondExecutionRequest,
)

TODOS = [{"content": "Wait for build result", "status": "in_progress"}]


def waiting_call(*, tasks: list[str] | None = None) -> AssistantMessage:
    return faux_assistant_message(
        faux_tool_call(
            "write_todos",
            {
                "todos": TODOS,
                "wait_for_tasks": tasks if tasks is not None else ["task-a"],
            },
        )
    )


def agent_with_wait(
    validator: TaskWaitValidator | None,
    *,
    responses: list[Any] | None = None,
    checkpointer: MemoryCheckpointer | None = None,
) -> tuple[Agent, MemoryCheckpointer]:
    provider = FauxProvider(provider_id="faux")
    provider.set_responses(
        responses
        or [
            waiting_call(),
            faux_assistant_message("The build is running; I will report its result."),
            faux_assistant_message(
                faux_tool_call(
                    "write_todos",
                    {
                        "todos": [
                            {"content": "Wait for build result", "status": "completed"}
                        ]
                    },
                )
            ),
            faux_assistant_message("Forced correction."),
        ]
    )
    storage = checkpointer or MemoryCheckpointer()
    middleware = TodoListMiddleware(
        extra_ref=lambda: agent.session.state_context,
        validate_task_wait=validator,
    )
    agent = Agent(
        model=provider.model("faux"),
        middleware=[middleware],
        checkpointer=storage,
        thread_id="task-wait",
    )
    return agent, storage


async def execute(agent: Agent, *, run_id: str = "run-a") -> None:
    result = await agent.session.execute(
        PromptExecutionRequest(
            run_id=run_id, attempt_id=f"{run_id}-attempt", message="build"
        )
    )
    assert result.outcome == "completed", result
    assert result.checkpoint_committed


async def test_valid_wait_finishes_without_faking_todo_completion() -> None:
    calls: list[TaskWaitBinding | None] = []

    async def validate(
        ids: list[str], ctx: AgentContext, prior: TaskWaitBinding | None
    ) -> TaskWaitValidation:
        assert ids == ["task-a"] and ctx.run_id == "run-a"
        calls.append(prior)
        return TaskWaitValidation(status="valid", validation={"generation": 7})

    agent, storage = agent_with_wait(validate)
    await execute(agent)
    checkpoint = await storage.load("task-wait")
    assert checkpoint is not None
    assert checkpoint.extra["todos"] == TODOS
    assert len([m for m in checkpoint.messages if isinstance(m, AssistantMessage)]) == 2
    assert len(calls) == 2 and calls[0] is None and calls[1] is not None
    assert calls[1].validation == {"generation": 7}


@pytest.mark.parametrize("outcome", ["unconfigured", "invalid", "cancelled", "failure"])
async def test_new_declaration_requires_positive_host_validation(outcome: str) -> None:
    async def validate(
        ids: list[str], ctx: AgentContext, prior: TaskWaitBinding | None
    ) -> TaskWaitValidation:
        if outcome == "failure":
            raise RuntimeError("authority lookup unavailable")
        return TaskWaitValidation(
            status="cancelled" if outcome == "cancelled" else "invalid",
            reason="not authorized",
        )

    agent, storage = agent_with_wait(None if outcome == "unconfigured" else validate)
    await execute(agent)
    checkpoint = await storage.load("task-wait")
    assert checkpoint is not None
    result = next(m for m in checkpoint.messages if isinstance(m, ToolResultMessage))
    assert result.is_error
    assert checkpoint.extra.get("todo_task_wait") is None


async def test_later_user_cancellation_finishes_without_another_model_call() -> None:
    async def validate(
        ids: list[str], ctx: AgentContext, prior: TaskWaitBinding | None
    ) -> TaskWaitValidation:
        if prior is None:
            return TaskWaitValidation(status="valid", validation={"accepted_at": 10})
        assert prior.validation == {"accepted_at": 10}
        return TaskWaitValidation(status="cancelled", reason="user stopped the task")

    agent, storage = agent_with_wait(validate)
    await execute(agent)
    checkpoint = await storage.load("task-wait")
    assert checkpoint is not None
    assert checkpoint.extra["todos"] == TODOS
    assert checkpoint.extra["todo_task_wait_outcome"] == {
        "status": "cancelled",
        "reason": "user stopped the task",
    }
    assert len([m for m in checkpoint.messages if isinstance(m, AssistantMessage)]) == 2


async def valid(
    ids: list[str], ctx: AgentContext, prior: TaskWaitBinding | None
) -> TaskWaitValidation:
    return TaskWaitValidation(status="valid", validation={"accepted": True})


def complete_call() -> AssistantMessage:
    return faux_assistant_message(
        faux_tool_call(
            "write_todos",
            {"todos": [{"content": "Wait for build result", "status": "completed"}]},
        )
    )


async def test_checkpoint_restore_new_initial_input_invalidates_old_wait() -> None:
    first, storage = agent_with_wait(valid)
    await execute(first)
    first_checkpoint = await storage.load("task-wait")
    assert first_checkpoint is not None and first_checkpoint.extra["todo_task_wait"]
    second, _ = agent_with_wait(
        valid,
        checkpointer=storage,
        responses=[
            faux_assistant_message("New request needs work."),
            complete_call(),
            faux_assistant_message("Finished new request."),
        ],
    )
    await second.session.load_checkpoint()
    await execute(second, run_id="run-b")
    checkpoint = await storage.load("task-wait")
    assert checkpoint is not None
    assert checkpoint.extra["todo_task_wait"] is None
    assert any(
        message.run_id == "run-b"
        and message.metadata.get("synthetic_source") == "todo_guard"
        for message in checkpoint.messages
    )


async def test_successful_update_without_wait_clears_previous_declaration() -> None:
    agent, storage = agent_with_wait(
        valid,
        responses=[
            waiting_call(),
            faux_assistant_message(faux_tool_call("write_todos", {"todos": TODOS})),
            faux_assistant_message("Not a waiting declaration anymore."),
            complete_call(),
            faux_assistant_message("Done."),
        ],
    )
    await execute(agent)
    checkpoint = await storage.load("task-wait")
    assert checkpoint is not None and checkpoint.extra["todo_task_wait"] is None
    assert any(
        m.metadata.get("synthetic_source") == "todo_guard" for m in checkpoint.messages
    )


@pytest.mark.parametrize("kind", ["user", "background"])
async def test_committed_live_input_invalidates_wait(kind: str) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def waiting_response(messages: Any, model: Any) -> AssistantMessage:
        entered.set()
        await release.wait()
        return faux_assistant_message("Waiting for the background result.")

    agent, storage = agent_with_wait(
        valid,
        responses=[
            waiting_call(),
            waiting_response,
            faux_assistant_message("Handle the new input."),
            complete_call(),
            faux_assistant_message("Done."),
        ],
    )
    running = asyncio.create_task(execute(agent))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        message = (
            UserMessage(content=[TextContent(text="new instruction")])
            if kind == "user"
            else synthetic_user_message("task-a finished", source="background_task")
        )
        receipt = agent.session.submit_input(
            InputEnvelope(input_id="new-input", message=message, mode="steer")
        )
        assert receipt.status == "queued"
    finally:
        release.set()
        await running
    checkpoint = await storage.load("task-wait")
    assert checkpoint is not None
    assert checkpoint.extra["todo_task_wait"] is None
    assert any(m.metadata.get("input_id") == "new-input" for m in checkpoint.messages)
    assert any(
        m.metadata.get("synthetic_source") == "todo_guard" for m in checkpoint.messages
    )


@pytest.mark.parametrize("outcome", ["invalid", "failure", "cancelled_without_reason"])
async def test_failed_revalidation_retains_normal_finalization_guard(
    outcome: str,
) -> None:
    async def validate(
        ids: list[str], ctx: AgentContext, prior: TaskWaitBinding | None
    ) -> TaskWaitValidation:
        if prior is None:
            return TaskWaitValidation(status="valid")
        if outcome == "failure":
            raise RuntimeError("lookup failed")
        return TaskWaitValidation(
            status="cancelled" if outcome == "cancelled_without_reason" else "invalid"
        )

    agent, storage = agent_with_wait(validate)
    await execute(agent)
    checkpoint = await storage.load("task-wait")
    assert checkpoint is not None
    assert len([m for m in checkpoint.messages if isinstance(m, AssistantMessage)]) == 4
    assert any(
        m.metadata.get("synthetic_source") == "todo_guard" for m in checkpoint.messages
    )


async def test_parallel_updates_restore_wait_and_todos_together() -> None:
    agent, storage = agent_with_wait(
        valid,
        responses=[
            waiting_call(),
            faux_assistant_message(
                [
                    faux_tool_call(
                        "write_todos", {"todos": TODOS, "wait_for_tasks": ["task-b"]}
                    ),
                    faux_tool_call("write_todos", {"todos": TODOS}),
                ]
            ),
            faux_assistant_message("The original background task is still running."),
        ],
    )
    await execute(agent)
    checkpoint = await storage.load("task-wait")
    assert checkpoint is not None and checkpoint.extra["todos"] == TODOS
    assert checkpoint.extra["todo_task_wait"]["task_ids"] == ["task-a"]
    results = [m for m in checkpoint.messages if isinstance(m, ToolResultMessage)]
    assert not results[0].is_error and all(m.is_error for m in results[1:])


async def test_new_input_during_initial_validation_cannot_get_old_authorization() -> (
    None
):
    async def validate(
        ids: list[str], ctx: AgentContext, prior: TaskWaitBinding | None
    ) -> TaskWaitValidation:
        ctx.messages.append(UserMessage(content=[TextContent(text="new request")]))
        return TaskWaitValidation(status="valid")

    agent, storage = agent_with_wait(validate)
    await execute(agent)
    checkpoint = await storage.load("task-wait")
    assert checkpoint is not None
    first_result = next(
        m for m in checkpoint.messages if isinstance(m, ToolResultMessage)
    )
    assert first_result.is_error


async def test_binding_survives_json_checkpoint_restore_at_same_input_boundary() -> (
    None
):
    agent, storage = agent_with_wait(valid)
    await execute(agent)
    checkpoint = await storage.load("task-wait")
    assert checkpoint is not None
    extra = json.loads(json.dumps(checkpoint.extra))
    restored = TodoListMiddleware(extra_ref=lambda: extra, validate_task_wait=valid)
    ctx = AgentContext(
        system_prompt="", messages=checkpoint.messages, extra=extra, run_id="run-a"
    )
    assert (
        await restored.after_model_response(
            faux_assistant_message("Still waiting."), ctx
        )
        is None
    )
    assert extra["todos"] == TODOS and extra["todo_task_wait"]["task_ids"] == ["task-a"]


async def test_other_middleware_can_still_require_another_turn() -> None:
    class OneMoreTurn(Middleware):
        requested = False

        async def after_model_response(self, response, ctx, *, signal=None):
            if self.requested or any(isinstance(c, ToolCall) for c in response.content):
                return None
            self.requested = True
            return TurnAction(decision="loop_to_model")

    provider = FauxProvider(provider_id="faux")
    provider.set_responses(
        [
            waiting_call(),
            faux_assistant_message("First reply."),
            faux_assistant_message("Second."),
        ]
    )
    storage = MemoryCheckpointer()
    todo = TodoListMiddleware(
        extra_ref=lambda: agent.session.state_context, validate_task_wait=valid
    )
    agent = Agent(
        model=provider.model("faux"),
        middleware=[OneMoreTurn(), todo],
        checkpointer=storage,
        thread_id="task-wait",
    )
    await execute(agent)
    assert provider.call_count == 3
    assert agent.session.state_context["todos"] == TODOS


@pytest.mark.parametrize("phase", ["initial", "recheck"])
async def test_explicit_stop_during_wait_validation_is_still_cancelled(
    phase: str,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def validate(
        ids: list[str], ctx: AgentContext, prior: TaskWaitBinding | None
    ) -> TaskWaitValidation:
        if (prior is None) == (phase == "initial"):
            entered.set()
            await release.wait()
        return TaskWaitValidation(status="valid")

    agent, _ = agent_with_wait(validate)
    running = asyncio.create_task(
        agent.session.execute(
            PromptExecutionRequest(
                run_id="run-a", attempt_id="attempt-a", message="build"
            )
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        agent.session.request_cancel()
    finally:
        release.set()
    result = await running
    assert result.outcome == "cancelled"
    if phase == "initial":
        assert agent.session.state_context.get("todo_task_wait") is None
    else:
        assert agent.session.state_context["todos"] == TODOS
        assert agent.session.state_context.get("todo_task_wait_outcome") is None


async def test_invalid_todo_payload_does_not_get_wait_exemption() -> None:
    called = False

    async def validate(
        ids: list[str], ctx: AgentContext, prior: TaskWaitBinding | None
    ) -> TaskWaitValidation:
        nonlocal called
        called = True
        return TaskWaitValidation(status="valid")

    agent, storage = agent_with_wait(
        validate,
        responses=[
            faux_assistant_message(
                faux_tool_call(
                    "write_todos",
                    {
                        "todos": [{"content": "", "status": "in_progress"}],
                        "wait_for_tasks": ["task-a"],
                    },
                )
            ),
            faux_assistant_message("Invalid checklist rejected."),
        ],
    )
    await execute(agent)
    checkpoint = await storage.load("task-wait")
    assert checkpoint is not None and not called
    result = next(m for m in checkpoint.messages if isinstance(m, ToolResultMessage))
    assert result.is_error and checkpoint.extra.get("todo_task_wait") is None


async def test_wait_state_does_not_change_system_prompt_prefix() -> None:
    extra: dict[str, Any] = {}
    mw = TodoListMiddleware(extra_ref=lambda: extra, validate_task_wait=valid)
    ctx = AgentContext(system_prompt="base", messages=[], extra=extra)
    before = await mw.transform_system_prompt("base", ctx=ctx)
    extra.update(
        {
            "todos": TODOS,
            "todo_task_wait_outcome": {"status": "cancelled", "reason": "Stop"},
        }
    )
    assert await mw.transform_system_prompt("base", ctx=ctx) == before


@pytest.mark.parametrize(
    ("same_turn", "answer_kind"),
    [
        (False, "ask"),
        (True, "ask"),
        (False, "approve"),
        (False, "deny"),
        (False, "edit"),
    ],
)
async def test_valid_task_wait_does_not_complete_a_pending_hitl_request(
    same_turn: bool,
    answer_kind: str,
) -> None:
    storage = MemoryCheckpointer()
    channel = CheckpointedChannel(
        checkpointer=storage, thread_id="task-wait", run_id="run-a"
    )
    provider = FauxProvider(provider_id="faux")
    ask = faux_tool_call(
        "ask_user", {"questions": [{"key": "continue", "prompt": "Continue?"}]}
    )

    class ActionParams(BaseModel):
        command: str

    executed: list[str] = []

    async def action(tool_call_id, params, *, signal=None, on_update=None):
        executed.append(params.command)
        return AgentToolResult(content=[TextContent(text="Executed.")])

    if answer_kind != "ask":
        ask = faux_tool_call("action", {"command": "original"})
    provider.set_responses(
        [faux_assistant_message([*waiting_call().content, ask])]
        if same_turn
        else [waiting_call(), faux_assistant_message(ask)]
    )
    todo = TodoListMiddleware(
        extra_ref=lambda: agent.session.state_context, validate_task_wait=valid
    )
    agent = Agent(
        model=provider.model("faux"),
        tools=[
            ask_user_tool(channel),
            AgentTool(
                name="action",
                description="Act",
                parameters=ActionParams,
                execute=action,
            ),
        ],
        middleware=[
            todo,
            ConfirmToolCallMiddleware(channel, require_confirm=["action"]),
        ],
        checkpointer=storage,
        channel=channel,
        thread_id="task-wait",
    )
    pending = asyncio.Event()
    agent.session.subscribe(
        lambda envelope: (
            pending.set() if envelope.event.type == "hitl_request" else None
        )
    )
    running = asyncio.create_task(
        agent.session.execute(
            PromptExecutionRequest(
                run_id="run-a", attempt_id="attempt-a", message="build"
            )
        )
    )
    try:
        await asyncio.wait_for(pending.wait(), timeout=2)
        assert channel.pending is not None and not running.done()
        await agent.session.request_detach()
        result = await running
    finally:
        if not running.done():
            running.cancel()
            await running
    assert result.outcome == "suspended" and result.pending_request is not None
    checkpoint = await storage.load("task-wait")
    assert checkpoint is not None and checkpoint.extra["todos"] == TODOS
    assert checkpoint.extra["todo_task_wait_outcome"] is None
    assert await storage.load_pending("task-wait") is not None
    provider.set_responses(
        [
            faux_assistant_message(
                "The answer needs to be handled before waiting again."
            ),
            complete_call(),
            faux_assistant_message("Done."),
        ]
    )
    resumed = await agent.session.execute(
        RespondExecutionRequest(
            run_id="run-a",
            attempt_id="attempt-b",
            question_id=result.pending_request.question_id,
            answer=(
                {"continue": "yes"}
                if answer_kind == "ask"
                else {"decision": answer_kind, "edited_args": {"command": "edited"}}
            ),
        )
    )
    assert resumed.outcome == "completed"
    checkpoint = await storage.load("task-wait")
    if answer_kind != "ask":
        assert checkpoint is not None
        result_message = next(
            m
            for m in checkpoint.messages
            if isinstance(m, ToolResultMessage) and m.tool_name == "action"
        )
        assert (
            isinstance(result_message.details, dict)
            and "hitl" in result_message.details
        ), result_message
    assert checkpoint is not None and checkpoint.extra["todo_task_wait"] is None
    assert checkpoint.extra["todos"][0]["status"] == "completed"
    assert executed == (
        ["original"]
        if answer_kind == "approve"
        else ["edited"]
        if answer_kind == "edit"
        else []
    )


async def test_wait_binding_is_checkpointed_before_the_next_model_call() -> None:
    storage = MemoryCheckpointer()

    async def check_durable_binding(messages, model):
        checkpoint = await storage.load("task-wait")
        assert checkpoint is not None
        assert checkpoint.extra["todos"] == TODOS
        assert checkpoint.extra["todo_task_wait"]["task_ids"] == ["task-a"]
        return faux_assistant_message("Waiting for the build.")

    agent, _ = agent_with_wait(
        valid, checkpointer=storage, responses=[waiting_call(), check_durable_binding]
    )
    await execute(agent)


async def test_wait_checkpoint_failure_cannot_report_success_or_continue() -> None:
    class FailingExtraCheckpointer(MemoryCheckpointer):
        async def save_extra(self, thread_id, extra):
            if extra.get("todo_task_wait"):
                raise OSError("checkpoint unavailable")
            await super().save_extra(thread_id, extra)

    storage = FailingExtraCheckpointer()
    advanced = False

    async def must_not_advance(messages, model):
        nonlocal advanced
        advanced = True
        return faux_assistant_message("Waiting.")

    agent, _ = agent_with_wait(
        valid, checkpointer=storage, responses=[waiting_call(), must_not_advance]
    )
    result = await agent.session.execute(
        PromptExecutionRequest(run_id="run-a", attempt_id="attempt-a", message="build")
    )
    assert result.outcome == "failed" and not advanced
    assert not result.checkpoint_committed and not result.history_consistent
