import asyncio

import pytest

from cubeloop.agent._tool_cycle import (
    ToolCycleViolation,
    check_tool_cycle,
)
from cubeloop.agent.agent import Agent
from cubeloop.checkpointer.memory import MemoryCheckpointer
from cubeloop.hitl.ask_user import ask_user_tool
from cubeloop.hitl.channel import CheckpointedChannel
from cubeloop.middleware.base import TurnAction
from cubeloop.providers.base import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from cubeloop.providers.faux import FauxProvider
from cubeloop.session import PromptExecutionRequest


def _asst(call_ids, run_id="R"):
    return AssistantMessage(
        content=[ToolCall(id=cid, name="t", arguments={}) for cid in call_ids],
        run_id=run_id,
    )


def _res(call_id, run_id="R"):
    return ToolResultMessage(
        tool_call_id=call_id,
        tool_name="t",
        content=[TextContent(text="r")],
        run_id=run_id,
    )


def test_no_tool_calls_ok():
    check_tool_cycle(
        [
            UserMessage(content=[TextContent(text="hi")], run_id="R"),
            AssistantMessage(content=[TextContent(text="hi back")], run_id="R"),
        ]
    )


def test_complete_cycle_ok():
    check_tool_cycle(
        [
            _asst(["c1", "c2"]),
            _res("c1"),
            _res("c2"),
            AssistantMessage(content=[TextContent(text="done")], run_id="R"),
        ]
    )


def test_no_results_at_all_violation():
    try:
        check_tool_cycle([_asst(["c1"])])
    except ToolCycleViolation:
        return
    assert False, "expected ToolCycleViolation"


def test_intervening_user_message_violation():
    try:
        check_tool_cycle(
            [
                _asst(["c1"]),
                UserMessage(content=[TextContent(text="hi")], run_id="R"),
                _res("c1"),
            ]
        )
    except ToolCycleViolation:
        return
    assert False


def test_partial_cover_violation():
    try:
        check_tool_cycle([_asst(["c1", "c2"]), _res("c1")])
    except ToolCycleViolation:
        return
    assert False


def test_duplicate_ids_across_turns_violation():
    try:
        check_tool_cycle(
            [
                _asst(["c1"]),
                _asst(["c1"]),  # second assistant reuses id
                _res("c1"),
            ]
        )
    except ToolCycleViolation:
        return
    assert False


def test_multiset_mismatch_within_window_violation():
    """Assistant emits {c1, c2}; window has [c1, c1] — set-equality
    would have failed, but the bug is multiset-specific: window length
    matches K=2, and only the multiset check catches that c2 is missing
    while c1 is duplicated. (A trailing extra tool_result AFTER the
    K-window is a separate concern handled by the NEXT assistant turn's
    adjacency check; it's not what this test covers.)"""
    try:
        check_tool_cycle(
            [
                _asst(["c1", "c2"]),
                _res("c1"),
                _res("c1"),  # duplicate of c1; c2 missing
            ]
        )
    except ToolCycleViolation:
        return
    assert False


@pytest.mark.asyncio
async def test_incomplete_tool_cycle_does_not_mark():
    """after_model_response(decision='stop') on a tool-use response
    leaves an unresolved tool_call. _dispatch_outcome filters
    state.messages by run_id and demotes 'complete' to 'incomplete'
    via check_tool_cycle. Marker not written."""
    p = FauxProvider()
    p.set_responses(
        [
            AssistantMessage(
                content=[ToolCall(id="c1", name="t", arguments={})],
                stop_reason="tool_use",
            ),
        ]
    )

    async def _stop_after(response, ctx, *, signal=None):
        return TurnAction(decision="stop")

    cp = MemoryCheckpointer()
    a = Agent(
        model=p.model("faux-model"),
        checkpointer=cp,
        thread_id="t",
        after_model_response=_stop_after,
    )
    result = await a.session.execute(
        PromptExecutionRequest(run_id="R1", attempt_id="A1", message="hi")
    )
    assert result.outcome == "incomplete"
    assert result.checkpoint_committed is False
    assert cp._runs["t"]["R1"].completed_at is None


@pytest.mark.asyncio
async def test_tool_cycle_invariant_spans_hitl_resume():
    """Pause mid-tool-use (ask_user). Resume; provider then emits an
    assistant carrying an UNRELATED unresolved tool_call (no matching
    ToolResultMessage will follow). The invariant filters
    state.messages by m.run_id == 'R1' — sees the unresolved c1 →
    outcome demoted from 'complete' to 'incomplete' → marker NOT
    written."""
    cp = MemoryCheckpointer()
    p = FauxProvider()
    p.set_responses(
        [
            # Turn 1: ask_user → pause.
            AssistantMessage(
                content=[
                    ToolCall(
                        id="ask-1",
                        name="ask_user",
                        arguments={"questions": [{"key": "ans", "prompt": "?"}]},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Turn 2 (resume): assistant with an unresolved tool_call
            # that NO ToolResultMessage will satisfy. The agent loop's
            # after_model_response forces a stop on this turn so the
            # call never gets executed.
            AssistantMessage(
                content=[
                    ToolCall(id="orphan-1", name="lookup", arguments={}),
                ],
                stop_reason="tool_use",
            ),
        ]
    )

    async def _stop_after(response, ctx, *, signal=None):
        # Stop after the second assistant turn so the orphan tool_call
        # is NEVER executed.
        if any(getattr(c, "id", None) == "orphan-1" for c in response.content):
            return TurnAction(decision="stop")
        return None

    ch = CheckpointedChannel(checkpointer=cp, thread_id="t", run_id="R1")
    tool = ask_user_tool(ch)
    a = Agent(
        model=p.model("faux-model"),
        tools=[tool],
        checkpointer=cp,
        thread_id="t",
        channel=ch,
        after_model_response=_stop_after,
    )
    task = asyncio.create_task(a.prompt("hi", run_id="R1"))
    while (await cp.load_pending("t")) is None:
        await asyncio.sleep(0.01)
    # prompt() blocks until detach / abort / answer. Detach to
    # surface HitlDetached → loop returns with last_outcome="suspended".
    await a.detach()
    await task

    # Resume with a fresh Agent. The second turn emits orphan-1;
    # the after_model_response hook stops the loop. Pre-completion
    # invariant scans state.messages filtered by run_id=='R1' and
    # finds the unresolved orphan-1 → outcome 'incomplete' → no mark.
    ch2 = CheckpointedChannel(checkpointer=cp, thread_id="t", run_id="R1")
    tool2 = ask_user_tool(ch2)
    a2 = Agent(
        model=p.model("faux-model"),
        tools=[tool2],
        checkpointer=cp,
        thread_id="t",
        channel=ch2,
        after_model_response=_stop_after,
    )
    pending = await cp.load_pending("t")
    qid = pending[0].question_id
    await a2.respond(question_id=qid, answer="yes")

    assert cp._runs["t"]["R1"].completed_at is None
