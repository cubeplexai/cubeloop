"""MCP adapter unit tests (D2.1)."""

from contextlib import asynccontextmanager

import pytest

from cubeloop.mcp._adapter import (
    make_mcp_agent_tool,
    mcp_schema_to_pydantic_model,
)


def test_schema_to_model_required_field() -> None:
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["city"],
    }
    M = mcp_schema_to_pydantic_model(tool_name="search", input_schema=schema)
    instance = M(city="Tokyo")
    assert instance.city == "Tokyo"
    assert instance.limit is None


def test_schema_to_model_array_field() -> None:
    schema = {
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["tags"],
    }
    M = mcp_schema_to_pydantic_model(tool_name="tag", input_schema=schema)
    instance = M(tags=["a", "b"])
    assert instance.tags == ["a", "b"]


def test_schema_to_model_boolean_and_number() -> None:
    schema = {
        "type": "object",
        "properties": {
            "active": {"type": "boolean"},
            "rate": {"type": "number"},
        },
        "required": ["active", "rate"],
    }
    M = mcp_schema_to_pydantic_model(tool_name="x", input_schema=schema)
    instance = M(active=True, rate=1.5)
    assert instance.active is True
    assert instance.rate == 1.5


def test_schema_to_model_preserves_enum() -> None:
    """enum becomes Literal — invalid values rejected by Pydantic."""
    schema = {
        "type": "object",
        "properties": {
            "unit": {"type": "string", "enum": ["c", "f"]},
        },
        "required": ["unit"],
    }
    M = mcp_schema_to_pydantic_model(tool_name="weather", input_schema=schema)
    assert M(unit="c").unit == "c"
    with pytest.raises(Exception):  # noqa: BLE001 - Pydantic ValidationError
        M(unit="kelvin")


def test_schema_to_model_preserves_string_constraints() -> None:
    schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "pattern": "^[A-Z]{3}$",
                "minLength": 3,
                "maxLength": 3,
                "description": "ISO airport code",
            },
        },
        "required": ["code"],
    }
    M = mcp_schema_to_pydantic_model(tool_name="ap", input_schema=schema)
    assert M(code="SFO").code == "SFO"
    with pytest.raises(Exception):
        M(code="sfo")  # lowercase fails pattern
    with pytest.raises(Exception):
        M(code="TOOLONG")  # exceeds maxLength
    assert M.model_fields["code"].description == "ISO airport code"


def test_schema_to_model_preserves_numeric_bounds() -> None:
    schema = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "exclusiveMinimum": 0,
            },
        },
        "required": ["limit"],
    }
    M = mcp_schema_to_pydantic_model(tool_name="pg", input_schema=schema)
    assert M(limit=50).limit == 50
    with pytest.raises(Exception):
        M(limit=0)
    with pytest.raises(Exception):
        M(limit=101)


def test_schema_to_model_preserves_array_size() -> None:
    schema = {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 3,
            },
        },
        "required": ["tags"],
    }
    M = mcp_schema_to_pydantic_model(tool_name="tag", input_schema=schema)
    assert M(tags=["a", "b"]).tags == ["a", "b"]
    with pytest.raises(Exception):
        M(tags=[])
    with pytest.raises(Exception):
        M(tags=["a", "b", "c", "d"])


def test_schema_to_model_unknown_type_becomes_any() -> None:
    """An unrecognized JSON Schema type falls back to typing.Any."""
    from typing import Any

    schema = {
        "type": "object",
        "properties": {
            # Neither a known scalar nor array/object — must hit the Any fallback.
            "weird": {"type": "totally-not-a-real-type"},
        },
        "required": [],
    }
    M = mcp_schema_to_pydantic_model(tool_name="x", input_schema=schema)
    # Field accepts arbitrary values because its type is Any.
    instance = M(weird={"anything": [1, 2, 3]})
    assert instance.weird == {"anything": [1, 2, 3]}
    annotations = M.model_fields["weird"].annotation
    assert annotations is Any


def test_schema_to_model_object_field_becomes_dict() -> None:
    schema = {
        "type": "object",
        "properties": {
            "config": {"type": "object"},
        },
        "required": [],
    }
    M = mcp_schema_to_pydantic_model(tool_name="x", input_schema=schema)
    instance = M(config={"a": 1})
    assert instance.config == {"a": 1}


def test_make_mcp_agent_tool_defaults_missing_description() -> None:
    async def _fake_call(name, args):
        return {"content": [], "isError": False}

    omitted = make_mcp_agent_tool(
        name="search",
        input_schema={"type": "object", "properties": {}},
        call_remote=_fake_call,
    )
    from_none = make_mcp_agent_tool(
        name="search",
        description=None,
        input_schema={"type": "object", "properties": {}},
        call_remote=_fake_call,
    )
    assert omitted.description == ""
    assert from_none.description == ""


@pytest.mark.asyncio
async def test_make_mcp_agent_tool_routes_to_call_remote() -> None:
    called: dict = {}

    async def _fake_call(name, args):
        called["name"] = name
        called["args"] = args
        return {
            "content": [{"type": "text", "text": "result"}],
            "isError": False,
        }

    tool = make_mcp_agent_tool(
        name="search",
        description="Search the web",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        call_remote=_fake_call,
    )
    assert tool.name == "search"
    assert tool.description == "Search the web"

    args = tool.parameters(query="cats")
    # Use the production signature: tool_call_id positional, then args, then keyword-only
    result = await tool.execute("test-call-id-1", args, signal=None, on_update=None)
    assert called == {"name": "search", "args": {"query": "cats"}}
    assert len(result.content) == 1
    from cubeloop.providers.base import TextContent

    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "result"


@pytest.mark.asyncio
async def test_make_mcp_agent_tool_carries_raw_response_in_details() -> None:
    async def _fake_call(name, args):
        return {"content": [], "isError": True, "errorMessage": "boom"}

    tool = make_mcp_agent_tool(
        name="bad",
        description="",
        input_schema={"type": "object", "properties": {}, "required": []},
        call_remote=_fake_call,
    )
    args = tool.parameters()
    result = await tool.execute("test-call-id-2", args, signal=None, on_update=None)
    assert result.details == {
        "raw_mcp_response": {"content": [], "isError": True, "errorMessage": "boom"}
    }
    # isError from MCP response must be reflected on AgentToolResult
    assert result.is_error is True


@pytest.mark.asyncio
async def test_make_mcp_agent_tool_omits_none_optional_args() -> None:
    """Optional schema field absent from args → omitted from call (not sent as null)."""
    captured: dict = {}

    async def _fake_call(name, args):
        captured["args"] = args
        return {"content": [], "isError": False}

    tool = make_mcp_agent_tool(
        name="search",
        description="",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
        call_remote=_fake_call,
    )
    args = tool.parameters(query="cats")  # limit not passed
    await tool.execute("tc-3", args, signal=None, on_update=None)
    # Critical: 'limit' must NOT appear in args (not even as null)
    assert "limit" not in captured["args"], (
        f"Optional field not provided should be omitted, got: {captured['args']!r}"
    )
    assert captured["args"] == {"query": "cats"}


async def test_signal_abort_cancels_in_flight_mcp_call() -> None:
    """``agent.abort()`` only sets a cooperative ``asyncio.Event``;
    it does NOT cancel the running task. An in-flight MCP
    ``tools/call`` would block until the response or transport
    timeout regardless of abort. The adapter must watch the signal
    and bail out with ``CancelledError`` so cancellation latency
    isn't bounded by MCP timeout (codex overall-review MAJOR)."""
    import asyncio as _asyncio

    from cubeloop.mcp._adapter import make_mcp_agent_tool

    started = _asyncio.Event()
    finished = _asyncio.Event()

    async def _slow_call(_name: str, _args: dict) -> dict:
        started.set()
        try:
            await _asyncio.sleep(10.0)  # would block well past test budget
        except _asyncio.CancelledError:
            finished.set()
            raise
        return {"content": [], "isError": False}

    tool = make_mcp_agent_tool(
        name="slow",
        description="slow tool",
        input_schema={"type": "object", "properties": {}, "required": []},
        call_remote=_slow_call,
    )
    signal = _asyncio.Event()
    args = tool.parameters()
    task = _asyncio.create_task(
        tool.execute("tc-abort", args, signal=signal, on_update=None)
    )
    await started.wait()
    # Cooperative abort: set the signal but do NOT cancel the task.
    signal.set()
    # Adapter must complete with a synthetic AgentToolResult (NOT
    # raise CancelledError) so the agent's normal abort lifecycle
    # runs — callers that do ``agent.abort(); await prompt_task``
    # must not see a new exception shape from MCP-backed tools
    # (codex round-1 follow-up on PR #88).
    try:
        result = await _asyncio.wait_for(task, timeout=1.0)
    except _asyncio.TimeoutError:
        task.cancel()
        raise AssertionError(
            "MCP adapter ignored abort signal; tools/call did not bail out"
        )
    # The inner call_remote was cancelled too.
    assert finished.is_set(), "call_remote was not cancelled when signal fired"
    # Synthetic tool result carries the aborted flag in details.
    assert result.details.get("aborted") is True
    # No spurious error status — abort isn't a failure.
    assert result.is_error is None or result.is_error is False
    assert result.content == []


async def test_agent_abort_during_mcp_call_does_not_raise() -> None:
    """The full ``agent.abort(); await prompt_task`` contract: when
    a real cubepi Agent calls an MCP tool that's still in flight and
    the user fires ``agent.abort()``, the prompt task must complete
    without surfacing ``CancelledError`` to the caller — matching
    the lifecycle non-MCP provider aborts use. Codex round-1
    follow-up on PR #88."""
    import asyncio as _asyncio

    from cubeloop.agent.agent import Agent
    from cubeloop.mcp._adapter import make_mcp_agent_tool
    from cubeloop.providers.base import ToolCall
    from cubeloop.providers.faux import FauxProvider, faux_assistant_message

    in_flight = _asyncio.Event()

    async def _slow(_name, _args):
        in_flight.set()
        await _asyncio.sleep(10.0)
        return {"content": [], "isError": False}

    mcp_tool = make_mcp_agent_tool(
        name="slow",
        description="slow",
        input_schema={"type": "object", "properties": {}, "required": []},
        call_remote=_slow,
    )
    provider = FauxProvider(provider_id="faux")
    provider.append_responses(
        [
            faux_assistant_message(
                [ToolCall(id="tc1", name="slow", arguments={})],
                stop_reason="tool_use",
            ),
            # Even though abort will kick in first, queue a follow-up
            # in case the loop somehow advances to a second turn.
            faux_assistant_message("done"),
        ]
    )
    agent = Agent(
        model=provider.model("faux-1"),
        system_prompt="s",
        tools=[mcp_tool],
    )

    prompt_task = _asyncio.create_task(agent.prompt("go"))
    await in_flight.wait()
    agent.abort()
    # The contract: prompt_task completes (returns None) without
    # raising. With the previous CancelledError-raising path this
    # would either propagate or wedge.
    try:
        await _asyncio.wait_for(prompt_task, timeout=2.0)
    except _asyncio.CancelledError:
        raise AssertionError(
            "agent.abort() during MCP call surfaced CancelledError to "
            "the caller — expected normal lifecycle completion"
        )
    except _asyncio.TimeoutError:
        prompt_task.cancel()
        raise AssertionError("prompt_task did not complete after abort()")


async def test_signal_abort_swallows_span_set_attribute_errors(monkeypatch) -> None:
    """The cooperative-abort branch in ``_execute`` marks the CLIENT
    span aborted before returning the synthetic result. If
    ``span.set_attribute`` raises (broken OTel SDK, recording-only
    span, …) the helper must swallow rather than corrupt the abort
    path (defensive-branch coverage)."""
    import asyncio as _asyncio

    from cubeloop.mcp._adapter import make_mcp_agent_tool
    from cubeloop.mcp import _tracing as mcp_tracing

    # Force mcp_client_span to yield a span whose set_attribute raises.
    class _BoomSpan:
        def set_attribute(self, *_a, **_kw):
            raise RuntimeError("set_attribute boom")

        def end(self):
            pass

        def get_span_context(self):
            class _Ctx:
                is_valid = False

            return _Ctx()

    @asynccontextmanager
    async def _fake_span(**_kw):
        yield _BoomSpan()

    monkeypatch.setattr("cubeloop.mcp._adapter.mcp_client_span", _fake_span)

    async def _call(_name, _args):
        return {"content": [], "isError": False}

    tool = make_mcp_agent_tool(
        name="t",
        description="t",
        input_schema={"type": "object", "properties": {}, "required": []},
        call_remote=_call,
    )
    signal = _asyncio.Event()
    signal.set()
    # set_attribute will raise; the abort branch must swallow it
    # and still return the synthetic aborted result.
    result = await tool.execute(
        "tc-x", tool.parameters(), signal=signal, on_update=None
    )
    assert result.details.get("aborted") is True

    # Silence the unused-import guard for mcp_tracing — it's documented
    # context for why span methods exist.
    del mcp_tracing


async def test_signal_abort_does_not_block_on_slow_call_cleanup() -> None:
    """``call_remote`` implementations sometimes catch
    ``CancelledError`` to do slow transport cleanup (close an HTTP
    socket, finalize an MCP session). The abort path must NOT block
    on that cleanup — otherwise we reintroduce the abort-latency
    problem the helper was meant to solve, just bounded by the
    cleanup duration instead of the MCP timeout (codex round-2
    follow-up on PR #88)."""
    import asyncio as _asyncio
    import time as _time

    from cubeloop.mcp._adapter import make_mcp_agent_tool

    in_flight = _asyncio.Event()
    cleanup_started = _asyncio.Event()
    cleanup_finished = _asyncio.Event()

    async def _slow_cleanup_call(_name, _args):
        in_flight.set()
        try:
            await _asyncio.sleep(10.0)
        except _asyncio.CancelledError:
            cleanup_started.set()
            # Simulate a slow teardown — 0.5s.
            try:
                await _asyncio.sleep(0.5)
            except _asyncio.CancelledError:
                pass
            cleanup_finished.set()
            raise
        return {"content": [], "isError": False}

    tool = make_mcp_agent_tool(
        name="slow-cleanup",
        description="slow cleanup tool",
        input_schema={"type": "object", "properties": {}, "required": []},
        call_remote=_slow_cleanup_call,
    )
    signal = _asyncio.Event()
    task = _asyncio.create_task(
        tool.execute("tc1", tool.parameters(), signal=signal, on_update=None)
    )
    await in_flight.wait()
    signal.set()
    t0 = _time.monotonic()
    result = await _asyncio.wait_for(task, timeout=1.0)
    elapsed = _time.monotonic() - t0
    # The fix: the helper does NOT await the cancelled call_task, so
    # _execute returns within a tick. Slow-cleanup work (the 0.5s
    # _asyncio.sleep inside the CancelledError handler) continues in
    # the background and must not delay this return.
    assert elapsed < 0.3, (
        f"abort path blocked on call_remote cleanup ({elapsed:.2f}s); "
        "expected immediate sentinel-based return"
    )
    assert result.details.get("aborted") is True
    # Cleanup started (cancellation dispatched) but may not be finished
    # yet — that's the whole point of the fix.
    assert cleanup_started.is_set()
    # Let the background cleanup finish so the test exits cleanly.
    await _asyncio.wait_for(cleanup_finished.wait(), timeout=2.0)


async def test_signal_already_set_before_call_aborts_immediately() -> None:
    """Pre-set signal: adapter returns the aborted synthetic result
    without invoking ``call_remote`` at all (codex round-1 follow-up
    on PR #88)."""
    import asyncio as _asyncio

    from cubeloop.mcp._adapter import make_mcp_agent_tool

    called = False

    async def _call(_name: str, _args: dict) -> dict:
        nonlocal called
        called = True
        return {"content": [], "isError": False}

    tool = make_mcp_agent_tool(
        name="t",
        description="t",
        input_schema={"type": "object", "properties": {}, "required": []},
        call_remote=_call,
    )
    signal = _asyncio.Event()
    signal.set()  # already aborted before _execute even starts
    result = await tool.execute(
        "tc-pre", tool.parameters(), signal=signal, on_update=None
    )
    assert called is False, "pre-set signal should skip call_remote entirely"
    assert result.details.get("aborted") is True
    assert result.content == []
