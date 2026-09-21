import asyncio

from cubeloop.agent.types import AgentContext, BeforeToolCallContext
from cubeloop.hitl import ApproveAnswer
from cubeloop.hitl.channel import InMemoryChannel
from cubeloop.hitl.middleware import ConfirmToolCallMiddleware
from cubeloop.providers.base import AssistantMessage, TextContent, ToolCall


def _ctx(name="bash"):
    return BeforeToolCallContext(
        assistant_message=AssistantMessage(
            content=[TextContent(text=""), ToolCall(id="tc", name=name, arguments={})],
            stop_reason="tool_use",
        ),
        tool_call=ToolCall(id="tc", name=name, arguments={}),
        args={},
        context=AgentContext(system_prompt="", messages=[], tools=[]),
    )


async def test_set_based_require_confirm_only_asks_for_listed():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, ApproveAnswer(decision="approve"))

    asyncio.create_task(host())
    mw = ConfirmToolCallMiddleware(ch, require_confirm={"bash"})
    # bash: prompts
    result = await mw.before_tool_call(_ctx("bash"))
    assert result is not None and not result.block
    assert result.hitl_trace == {"decision": "human_approve"}
    # read_file: not in set — passes through silently
    assert (await mw.before_tool_call(_ctx("read_file"))) is None
    # bash prompted exactly once; read_file did not engage channel
    assert ch.pending is None


async def test_predicate_require_confirm():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, ApproveAnswer(decision="approve"))

    asyncio.create_task(host())

    def needs_confirm(ctx):
        return ctx.tool_call.name.startswith("dangerous_")

    mw = ConfirmToolCallMiddleware(ch, require_confirm=needs_confirm)
    result = await mw.before_tool_call(_ctx("dangerous_op"))
    assert result is not None and not result.block
    assert result.hitl_trace == {"decision": "human_approve"}


async def test_default_none_asks_for_every_tool():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, ApproveAnswer(decision="approve"))

    asyncio.create_task(host())
    mw = ConfirmToolCallMiddleware(ch)  # no require_confirm
    result = await mw.before_tool_call(_ctx("anything"))
    assert result is not None and not result.block
    assert result.hitl_trace == {"decision": "human_approve"}
