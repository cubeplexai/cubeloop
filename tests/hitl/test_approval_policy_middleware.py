import asyncio

from cubeloop.agent.types import AgentContext, BeforeToolCallContext
from cubeloop.hitl import Approve, ApproveAnswer, AskUser, Deny
from cubeloop.hitl.channel import InMemoryChannel
from cubeloop.hitl.middleware import ApprovalPolicyMiddleware
from cubeloop.providers.base import AssistantMessage, TextContent, ToolCall


def _ctx(tool_call_id="tc-1") -> BeforeToolCallContext:
    return BeforeToolCallContext(
        assistant_message=AssistantMessage(
            content=[
                TextContent(text=""),
                ToolCall(id=tool_call_id, name="bash", arguments={"cmd": "ls"}),
            ],
            stop_reason="tool_use",
        ),
        tool_call=ToolCall(id=tool_call_id, name="bash", arguments={"cmd": "ls"}),
        args={"cmd": "ls"},
        context=AgentContext(system_prompt="", messages=[], tools=[]),
    )


async def test_approve_policy_passthrough():
    mw = ApprovalPolicyMiddleware(InMemoryChannel(), policy=lambda c: Approve())
    result = await mw.before_tool_call(_ctx())
    assert result is None


async def test_deny_policy_blocks_without_channel():
    ch = InMemoryChannel()
    mw = ApprovalPolicyMiddleware(ch, policy=lambda c: Deny(reason="forbidden"))
    result = await mw.before_tool_call(_ctx())
    assert result.block is True
    assert result.deny_reason == "forbidden"
    assert result.hitl_trace["decision"] == "policy_deny"
    assert ch.pending is None  # channel never invoked


async def test_ask_user_policy_invokes_channel_human_approve():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, ApproveAnswer(decision="approve"))

    asyncio.create_task(host())
    mw = ApprovalPolicyMiddleware(ch, policy=lambda c: AskUser())
    result = await mw.before_tool_call(_ctx())
    assert result is not None and not result.block
    assert result.hitl_trace == {"decision": "human_approve"}


async def test_ask_user_policy_human_deny_blocks():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(
            ch.pending.question_id, ApproveAnswer(decision="deny", reason="no")
        )

    asyncio.create_task(host())
    mw = ApprovalPolicyMiddleware(ch, policy=lambda c: AskUser())
    result = await mw.before_tool_call(_ctx())
    assert result.block is True
    assert result.hitl_trace["decision"] == "human_deny"
    assert result.deny_reason == "no"


async def test_ask_user_policy_human_edit_passes_edited_args():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(
            ch.pending.question_id,
            ApproveAnswer(decision="edit", edited_args={"cmd": "ls -l"}),
        )

    asyncio.create_task(host())
    mw = ApprovalPolicyMiddleware(ch, policy=lambda c: AskUser())
    result = await mw.before_tool_call(_ctx())
    assert result.edited_args == {"cmd": "ls -l"}
    assert result.hitl_trace["decision"] == "edit"
    assert result.hitl_trace["original_args"] == {"cmd": "ls"}


async def test_timeout_translates_to_approval_timeout_deny():
    ch = InMemoryChannel()
    mw = ApprovalPolicyMiddleware(
        ch,
        policy=lambda c: AskUser(timeout_seconds=0.05),
    )
    result = await mw.before_tool_call(_ctx())
    assert result.block is True
    assert result.deny_reason == "approval_timeout"
    assert result.hitl_trace["decision"] == "timed_out"


async def test_cancel_translates_to_cancelled_deny():
    ch = InMemoryChannel()

    async def canceller():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.cancel(ch.pending.question_id, reason="closed tab")

    asyncio.create_task(canceller())
    mw = ApprovalPolicyMiddleware(ch, policy=lambda c: AskUser())
    result = await mw.before_tool_call(_ctx())
    assert result.block is True
    assert "cancelled: closed tab" == result.deny_reason
    assert result.hitl_trace["decision"] == "cancelled"


async def test_async_policy_is_awaited():
    ch = InMemoryChannel()

    async def policy(c):
        return Approve()

    mw = ApprovalPolicyMiddleware(ch, policy=policy)
    assert (await mw.before_tool_call(_ctx())) is None
