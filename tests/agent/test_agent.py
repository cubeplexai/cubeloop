import asyncio

import pytest

from cubeloop.agent.agent import Agent, _MessageQueue
from cubeloop.agent.types import AgentTool
from cubeloop.middleware.base import Middleware
from cubeloop.providers.base import (
    AssistantMessage,
    Model,
    ReasoningControl,
    TextContent,
    UserMessage,
)
from cubeloop.providers.faux import FauxProvider, faux_assistant_message
from pydantic import BaseModel


def make_model() -> Model:
    return Model(id="faux-1", provider_id="faux")


class TestAgentInit:
    def test_agent_unwraps_bound_model(self):
        provider = FauxProvider(provider_id="faux")
        bound_model = provider.model("faux-1")

        agent = Agent(model=bound_model)

        assert agent.state.model.id == "faux-1"
        assert agent.state.model.provider_id == "faux"

    def test_default_state(self):
        provider = FauxProvider(provider_id="faux")
        agent = Agent(model=provider.model("faux-1"))

        assert agent.state.system_prompt == ""
        assert agent.state.reasoning == ReasoningControl()
        assert agent.state.tools == []
        assert agent.state.messages == []
        assert agent.state.is_streaming is False
        assert agent.state.streaming_message is None
        assert agent.state.pending_tool_calls == set()
        assert agent.state.error_message is None
        assert agent._checkpoint_loaded is False

    def test_custom_initial_state(self):
        provider = FauxProvider(provider_id="faux")
        agent = Agent(
            model=provider.model("faux-1"),
            system_prompt="You are a helpful assistant.",
            reasoning=ReasoningControl(mode="on", effort="low"),
        )

        assert agent.state.system_prompt == "You are a helpful assistant."
        assert agent.state.reasoning == ReasoningControl(mode="on", effort="low")

    def test_middleware_tools_are_registered_on_agent(self):
        class Params(BaseModel):
            pass

        async def execute(tool_call_id, params, *, signal=None, on_update=None):
            return None

        tool = AgentTool(
            name="from_middleware",
            description="middleware tool",
            parameters=Params,
            execute=execute,
        )

        class ToolMiddleware(Middleware):
            tools = [tool]

        agent = Agent(
            model=FauxProvider(provider_id="faux").model("faux-1"),
            middleware=[ToolMiddleware()],
        )

        assert agent.state.tools == [tool]


class TestAgentSubscribe:
    def test_subscribe_and_unsubscribe(self):
        provider = FauxProvider(provider_id="faux")
        agent = Agent(model=provider.model("faux-1"))

        count = 0

        def listener(event, signal=None):
            nonlocal count
            count += 1

        unsub = agent.subscribe(listener)
        assert count == 0

        unsub()

    async def test_events_emitted_on_prompt(self):
        provider = FauxProvider(provider_id="faux")
        provider.set_responses([faux_assistant_message("ok")])
        agent = Agent(model=provider.model("faux-1"))

        events = []
        agent.subscribe(lambda e, s=None: events.append(e.type))

        await agent.prompt("hello")

        assert "agent_start" in events
        assert "message_start" in events
        assert "message_end" in events
        assert "agent_end" in events

    async def test_full_lifecycle_events_for_thrown_run_failures(self):
        async def bad_stream(*args, **kwargs):
            raise RuntimeError("provider exploded")

        provider = FauxProvider(provider_id="faux")
        provider.stream = bad_stream
        agent = Agent(model=provider.model("faux-1"))

        events = []
        agent.subscribe(lambda e, s=None: events.append(e.type))

        await agent.prompt("hello")

        assert events == [
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "turn_end",
            "agent_end",
        ]
        last_msg = agent.state.messages[-1]
        assert last_msg.role == "assistant"
        assert last_msg.stop_reason == "error"
        assert last_msg.error_message == "provider exploded"
        assert agent.state.error_message == "provider exploded"

    async def test_await_async_subscribers_before_prompt_resolves(self):
        provider = FauxProvider(provider_id="faux")
        provider.set_responses([faux_assistant_message("ok")])
        agent = Agent(model=provider.model("faux-1"))

        barrier = asyncio.Event()
        listener_finished = False

        async def listener(event, signal=None):
            nonlocal listener_finished
            if event.type == "agent_end":
                await barrier.wait()
                listener_finished = True

        agent.subscribe(listener)

        prompt_resolved = False

        async def run_prompt():
            nonlocal prompt_resolved
            await agent.prompt("hello")
            prompt_resolved = True

        task = asyncio.create_task(run_prompt())
        await asyncio.sleep(0.05)

        assert not prompt_resolved
        assert not listener_finished
        assert agent.state.is_streaming is True

        barrier.set()
        await task

        assert listener_finished
        assert prompt_resolved
        assert agent.state.is_streaming is False


class TestAgentState:
    def test_tools_are_copied(self):
        provider = FauxProvider(provider_id="faux")
        agent = Agent(model=provider.model("faux-1"))

        tools = [
            AgentTool(
                name="t",
                description="t",
                parameters=type(
                    "P", (object,), {"model_json_schema": classmethod(lambda cls: {})}
                ),
                execute=lambda *a, **k: None,
            )
        ]
        agent.state.tools = tools
        assert agent.state.tools is not tools

    def test_messages_are_copied(self):
        provider = FauxProvider(provider_id="faux")
        agent = Agent(model=provider.model("faux-1"))

        messages = [UserMessage(content=[TextContent(text="hi")])]
        agent.state.messages = messages
        assert agent.state.messages is not messages


class TestAgentQueues:
    def test_steer_queues_message(self):
        provider = FauxProvider(provider_id="faux")
        agent = Agent(model=provider.model("faux-1"))

        msg = UserMessage(content=[TextContent(text="steering")])
        agent.steer(msg)
        assert msg not in agent.state.messages

    def test_follow_up_queues_message(self):
        provider = FauxProvider(provider_id="faux")
        agent = Agent(model=provider.model("faux-1"))

        msg = UserMessage(content=[TextContent(text="follow-up")])
        agent.follow_up(msg)
        assert msg not in agent.state.messages


class TestAgentAbort:
    def test_abort_without_active_run_does_not_throw(self):
        provider = FauxProvider(provider_id="faux")
        agent = Agent(model=provider.model("faux-1"))
        agent.abort()


class TestAgentPromptGuards:
    async def test_raises_when_prompt_called_while_streaming(self):
        barrier = asyncio.Event()
        provider = FauxProvider(provider_id="faux")

        async def slow_stream(*args, **kwargs):
            from cubeloop.providers.base import MessageStream, StreamEvent

            ms = MessageStream()

            async def produce():
                await barrier.wait()
                msg = faux_assistant_message("ok")
                ms.push(StreamEvent(type="done"))
                ms.set_result(msg)

            asyncio.create_task(produce())
            return ms

        provider.stream = slow_stream
        agent = Agent(model=provider.model("faux-1"))

        task = asyncio.create_task(agent.prompt("first"))
        await asyncio.sleep(0.02)
        assert agent.state.is_streaming is True

        try:
            await agent.prompt("second")
            assert False, "Should have raised"
        except RuntimeError as e:
            assert "already processing" in str(e).lower()

        barrier.set()
        await task

    async def test_raises_when_resume_called_while_streaming(self):
        barrier = asyncio.Event()
        provider = FauxProvider(provider_id="faux")

        async def slow_stream(*args, **kwargs):
            from cubeloop.providers.base import MessageStream, StreamEvent

            ms = MessageStream()

            async def produce():
                await barrier.wait()
                msg = faux_assistant_message("ok")
                ms.push(StreamEvent(type="done"))
                ms.set_result(msg)

            asyncio.create_task(produce())
            return ms

        provider.stream = slow_stream
        agent = Agent(model=provider.model("faux-1"))

        task = asyncio.create_task(agent.prompt("first"))
        await asyncio.sleep(0.02)

        try:
            await agent.resume()
            assert False, "Should have raised"
        except RuntimeError as e:
            assert "already processing" in str(e).lower()

        barrier.set()
        await task


class TestAgentResume:
    async def test_resume_processes_follow_up_messages(self):
        provider = FauxProvider(provider_id="faux")
        provider.set_responses(
            [
                faux_assistant_message("Initial response"),
                faux_assistant_message("Processed"),
            ]
        )
        agent = Agent(model=provider.model("faux-1"))

        await agent.prompt("Initial")
        agent.follow_up(UserMessage(content=[TextContent(text="follow-up")]))
        await agent.resume()

        has_follow_up = any(
            isinstance(m, UserMessage)
            and any(
                isinstance(c, TextContent) and c.text == "follow-up" for c in m.content
            )
            for m in agent.state.messages
        )
        assert has_follow_up
        assert isinstance(agent.state.messages[-1], AssistantMessage)

    async def test_resume_drains_steering_queue_before_follow_up(self):
        """resume() should drain the steering queue first when the last
        message is from the assistant."""
        provider = FauxProvider(provider_id="faux")
        provider.set_responses(
            [
                faux_assistant_message("Initial"),
                faux_assistant_message("Steered"),
            ]
        )
        agent = Agent(model=provider.model("faux-1"))

        await agent.prompt("hello")
        agent.steer(UserMessage(content=[TextContent(text="steer-msg")]))
        await agent.resume()

        has_steer = any(
            isinstance(m, UserMessage)
            and any(
                isinstance(c, TextContent) and c.text == "steer-msg" for c in m.content
            )
            for m in agent.state.messages
        )
        assert has_steer
        assert isinstance(agent.state.messages[-1], AssistantMessage)

    async def test_resume_raises_on_assistant_last_with_empty_queues(self):
        """resume() raises RuntimeError when the last message is from the
        assistant and both steering and follow-up queues are empty."""
        provider = FauxProvider(provider_id="faux")
        provider.set_responses([faux_assistant_message("done")])
        agent = Agent(model=provider.model("faux-1"))

        await agent.prompt("hello")

        with pytest.raises(RuntimeError, match="Cannot continue from message role"):
            await agent.resume()


class TestMessageQueueAllMode:
    def test_drain_returns_all_messages_at_once(self):
        q = _MessageQueue(mode="all")
        m1 = UserMessage(content=[TextContent(text="a")])
        m2 = UserMessage(content=[TextContent(text="b")])
        m3 = UserMessage(content=[TextContent(text="c")])

        q.enqueue(m1)
        q.enqueue(m2)
        q.enqueue(m3)

        drained = q.drain()
        assert drained == [m1, m2, m3]
        assert not q.has_items()

    def test_drain_returns_empty_when_no_items(self):
        q = _MessageQueue(mode="all")
        assert q.drain() == []

    def test_has_items_reflects_state(self):
        q = _MessageQueue(mode="all")
        assert not q.has_items()
        q.enqueue(UserMessage(content=[TextContent(text="x")]))
        assert q.has_items()

    def test_clear_removes_all(self):
        q = _MessageQueue(mode="all")
        q.enqueue(UserMessage(content=[TextContent(text="a")]))
        q.enqueue(UserMessage(content=[TextContent(text="b")]))
        q.clear()
        assert not q.has_items()
        assert q.drain() == []


class TestAgentStatePendingToolCalls:
    def test_setter_makes_a_copy(self):
        provider = FauxProvider(provider_id="faux")
        agent = Agent(model=provider.model("faux-1"))

        original = {"call-1", "call-2"}
        agent.state.pending_tool_calls = original

        retrieved = agent.state.pending_tool_calls
        assert retrieved == {"call-1", "call-2"}
        # Must be a distinct set, not the same object
        assert retrieved is not original


class TestAgentReset:
    async def test_reset_clears_state_after_prompt(self):
        provider = FauxProvider(provider_id="faux")
        provider.set_responses([faux_assistant_message("response")])
        agent = Agent(model=provider.model("faux-1"))

        await agent.prompt("hello")
        assert len(agent.state.messages) > 0
        agent._checkpoint_loaded = True

        agent.reset()

        assert agent.state.messages == []
        assert agent.state.is_streaming is False
        assert agent.state.streaming_message is None
        assert agent.state.pending_tool_calls == set()
        assert agent.state.error_message is None

    async def test_reset_clears_queues(self):
        provider = FauxProvider(provider_id="faux")
        agent = Agent(model=provider.model("faux-1"))

        agent.steer(UserMessage(content=[TextContent(text="steer")]))
        agent.follow_up(UserMessage(content=[TextContent(text="follow")]))

        agent.reset()

        # After reset, queues should be empty — drain returns nothing
        assert agent._steering_queue.drain() == []
        assert agent._follow_up_queue.drain() == []

    async def test_reset_clears_session_input_receipts(self):
        provider = FauxProvider(provider_id="faux")
        agent = Agent(model=provider.model("faux-1"))
        agent.session._input_status["input-1"] = "committed"
        agent.session._input_durability["input-1"] = "memory"

        agent.reset()

        assert agent.session._input_status == {}
        assert agent.session._input_durability == {}
        assert agent.session._queued_inputs == {}


class TestAgentAbortSignal:
    async def test_abort_sets_signal_during_active_run(self):
        barrier = asyncio.Event()
        provider = FauxProvider(provider_id="faux")

        async def slow_stream(*args, **kwargs):
            from cubeloop.providers.base import MessageStream, StreamEvent

            ms = MessageStream()

            async def produce():
                await barrier.wait()
                msg = faux_assistant_message("ok")
                ms.push(StreamEvent(type="done"))
                ms.set_result(msg)

            asyncio.create_task(produce())
            return ms

        provider.stream = slow_stream
        agent = Agent(model=provider.model("faux-1"))

        task = asyncio.create_task(agent.prompt("hello"))
        await asyncio.sleep(0.02)

        assert agent._active_signal is not None
        assert not agent._active_signal.is_set()

        agent.abort()
        assert agent._active_signal.is_set()

        barrier.set()
        await task


class TestAgentWaitForIdle:
    async def test_returns_immediately_when_no_active_run(self):
        provider = FauxProvider(provider_id="faux")
        agent = Agent(model=provider.model("faux-1"))

        # No active run, _active_done is None — should return immediately
        await agent.wait_for_idle()

    async def test_waits_until_prompt_completes(self):
        barrier = asyncio.Event()
        provider = FauxProvider(provider_id="faux")

        async def slow_stream(*args, **kwargs):
            from cubeloop.providers.base import MessageStream, StreamEvent

            ms = MessageStream()

            async def produce():
                await barrier.wait()
                msg = faux_assistant_message("ok")
                ms.push(StreamEvent(type="done"))
                ms.set_result(msg)

            asyncio.create_task(produce())
            return ms

        provider.stream = slow_stream
        agent = Agent(model=provider.model("faux-1"))

        prompt_task = asyncio.create_task(agent.prompt("hello"))
        await asyncio.sleep(0.02)

        idle_resolved = False

        async def wait():
            nonlocal idle_resolved
            await agent.wait_for_idle()
            idle_resolved = True

        wait_task = asyncio.create_task(wait())
        await asyncio.sleep(0.02)
        assert not idle_resolved

        barrier.set()
        await prompt_task
        await wait_task
        assert idle_resolved


class TestAgentPromptInputTypes:
    async def test_prompt_with_message_object(self):
        provider = FauxProvider(provider_id="faux")
        provider.set_responses([faux_assistant_message("response")])
        agent = Agent(model=provider.model("faux-1"))

        msg = UserMessage(content=[TextContent(text="direct message")])
        await agent.prompt(msg)

        has_direct = any(
            isinstance(m, UserMessage)
            and any(
                isinstance(c, TextContent) and c.text == "direct message"
                for c in m.content
            )
            for m in agent.state.messages
        )
        assert has_direct
        assert isinstance(agent.state.messages[-1], AssistantMessage)

    async def test_prompt_with_list_of_messages(self):
        provider = FauxProvider(provider_id="faux")
        provider.set_responses([faux_assistant_message("response")])
        agent = Agent(model=provider.model("faux-1"))

        msgs = [
            UserMessage(content=[TextContent(text="first")]),
            UserMessage(content=[TextContent(text="second")]),
        ]
        await agent.prompt(msgs)

        texts = [
            c.text
            for m in agent.state.messages
            if isinstance(m, UserMessage)
            for c in m.content
            if isinstance(c, TextContent)
        ]
        assert "first" in texts
        assert "second" in texts
        assert isinstance(agent.state.messages[-1], AssistantMessage)


def _steer_msg(text: str, steer_id: str) -> UserMessage:
    return UserMessage(
        content=[TextContent(text=text)], metadata={"steer_id": steer_id}
    )


def test_message_queue_remove_by_steer_id():
    q = _MessageQueue(mode="all")
    q.enqueue(_steer_msg("a", "s1"))
    q.enqueue(_steer_msg("b", "s2"))
    assert q.remove("s1") is True
    assert [m.content[0].text for m in q.drain()] == ["b"]


def test_message_queue_remove_missing_returns_false():
    q = _MessageQueue(mode="all")
    q.enqueue(_steer_msg("a", "s1"))
    assert q.remove("nope") is False
    assert len(q.drain()) == 1


def test_message_queue_remove_ignores_messages_without_steer_id():
    q = _MessageQueue(mode="all")
    q.enqueue(UserMessage(content=[TextContent(text="x")]))  # no metadata.steer_id
    assert q.remove("s1") is False


@pytest.mark.asyncio
async def test_agent_cancel_steer_removes_queued_steer():
    agent = Agent(model=FauxProvider(provider_id="faux").model("faux-1"))
    agent.steer(_steer_msg("hi", "s1"))
    assert agent.cancel_steer("s1") is True
    assert agent.cancel_steer("s1") is False
