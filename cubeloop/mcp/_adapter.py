"""MCP tool descriptor → cubeloop.AgentTool adapter."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, Field, create_model

from cubeloop.agent.types import AgentTool, AgentToolResult
from cubeloop.mcp._tracing import mark_span_mcp_error, mcp_client_span
from cubeloop.providers.base import Content, ImageContent, TextContent


def mcp_schema_to_pydantic_model(
    *,
    tool_name: str,
    input_schema: dict[str, Any],
) -> type[BaseModel]:
    """Convert an MCP JSON-Schema inputSchema to a Pydantic model class.

    cubeloop.AgentTool requires `parameters: type[BaseModel]`. We synthesize
    a model from the schema's top-level properties.

    Type coverage: string/integer/number/boolean/array/object. Nested
    object schemas become dict[str, Any]. Constraint coverage on top-level
    properties: enum (via Literal), description, pattern, minLength,
    maxLength, minimum/maximum (incl. exclusive variants), minItems,
    maxItems. Anything else is preserved as a plain type without
    validation.
    """
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    fields: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        fields[prop_name] = _build_field(prop_schema, prop_name in required)

    model_name = f"MCP_{tool_name}_Input"
    return create_model(model_name, **fields)


def _build_field(prop_schema: dict[str, Any], required: bool) -> tuple[Any, Any]:
    """Map one JSON-Schema property to a (type, FieldInfo|default) tuple."""
    if "enum" in prop_schema and isinstance(prop_schema["enum"], list):
        enum_values = tuple(prop_schema["enum"])
        py_type: Any = Literal[enum_values] if enum_values else Any
    else:
        py_type = _json_schema_type_to_python(prop_schema)

    field_kwargs: dict[str, Any] = {}
    if "description" in prop_schema:
        field_kwargs["description"] = prop_schema["description"]

    # string constraints
    if "pattern" in prop_schema:
        field_kwargs["pattern"] = prop_schema["pattern"]
    if "minLength" in prop_schema:
        field_kwargs["min_length"] = prop_schema["minLength"]
    if "maxLength" in prop_schema:
        field_kwargs["max_length"] = prop_schema["maxLength"]

    # numeric constraints
    if "minimum" in prop_schema:
        field_kwargs["ge"] = prop_schema["minimum"]
    if "maximum" in prop_schema:
        field_kwargs["le"] = prop_schema["maximum"]
    if "exclusiveMinimum" in prop_schema:
        field_kwargs["gt"] = prop_schema["exclusiveMinimum"]
    if "exclusiveMaximum" in prop_schema:
        field_kwargs["lt"] = prop_schema["exclusiveMaximum"]

    # array constraints
    if "minItems" in prop_schema:
        field_kwargs["min_length"] = prop_schema["minItems"]
    if "maxItems" in prop_schema:
        field_kwargs["max_length"] = prop_schema["maxItems"]

    default = ... if required else None
    if field_kwargs:
        return (py_type, Field(default=default, **field_kwargs))
    return (py_type, default)


def _json_schema_type_to_python(schema: dict[str, Any]) -> Any:
    t = schema.get("type")
    if t == "string":
        return str
    if t == "integer":
        return int
    if t == "number":
        return float
    if t == "boolean":
        return bool
    if t == "array":
        item_t = _json_schema_type_to_python(schema.get("items", {}))
        return list[item_t]  # type: ignore[valid-type]
    if t == "object":
        return dict[str, Any]
    return Any


def make_mcp_agent_tool(
    *,
    name: str,
    description: str,
    input_schema: dict[str, Any],
    call_remote: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
    server_address: str | None = None,
    server_port: int | None = None,
    protocol_version: str | None = None,
    session_id: str | None = None,
) -> AgentTool:
    """Build a cubeloop.AgentTool wrapping an MCP tool call.

    call_remote is the transport-specific RPC: given (tool_name, args_dict),
    returns the MCP tools/call response dict normalized to:
        {"content": [{"type": "text", "text": ...}, ...], "isError": bool}

    The optional ``server_address`` / ``server_port`` / ``protocol_version``
    / ``session_id`` are recorded on the OTel CLIENT span around the
    remote call when ``cubepi[tracing]`` is installed. Loaders pass
    whichever values they have access to from the MCP handshake.
    """
    parameters_model = mcp_schema_to_pydantic_model(
        tool_name=name,
        input_schema=input_schema,
    )

    async def _execute(
        tool_call_id: str,
        args,
        *,
        signal=None,
        on_update=None,
    ) -> AgentToolResult:
        # on_update is unused by MCP semantics (MCP RPC is not
        # incremental). ``tool_call_id`` is forwarded as the parent
        # lookup key so the CLIENT span nests under the cubepi recorder's
        # execute_tool span when tracing is enabled.
        del on_update
        args_dict = (
            args.model_dump(exclude_none=True)
            if hasattr(args, "model_dump")
            else dict(args)
        )
        async with mcp_client_span(
            method="tools/call",
            tool_name=name,
            session_id=session_id,
            protocol_version=protocol_version,
            server_address=server_address,
            server_port=server_port,
            parent_tool_call_id=tool_call_id,
        ) as span:
            result = await _await_with_signal(call_remote(name, args_dict), signal)
            if result is _ABORTED_SENTINEL:
                # Cooperative abort fired while the MCP RPC was in
                # flight. Mark the CLIENT span aborted (Status UNSET,
                # cubeloop.aborted=true) so the trace shows the
                # interruption, then return a synthetic tool result.
                # Returning rather than raising preserves the agent
                # lifecycle that non-MCP provider aborts use: the
                # ``ToolExecutionEndEvent`` fires, the loop advances
                # to the next turn, the provider's stream loop sees
                # ``signal.is_set()`` and emits an aborted
                # ``AssistantMessage``, and ``TurnEnd`` / ``AgentEnd``
                # close the run normally. Callers that
                # ``agent.abort(); await prompt_task`` therefore see
                # no new exception type from MCP-backed tools (codex
                # round-1 follow-up on PR #88).
                if span is not None:
                    try:
                        span.set_attribute("cubeloop.aborted", True)
                        span.set_attribute("error.type", "cubeloop.aborted")
                    except Exception:
                        pass
                return AgentToolResult(
                    content=[],
                    details={"aborted": True},
                )
            # MCP server-reported tool failure (``isError: true``): the
            # JSON-RPC call succeeded but the tool's logic failed. Mark
            # the CLIENT span ERROR so backends count it as a failure,
            # matching the cubeloop.tool.is_error treatment for non-MCP
            # tool spans.
            if result.get("isError"):
                mark_span_mcp_error(span, "MCP server returned isError")
        content_blocks: list[Content] = []
        for c in result.get("content", []):
            ctype = c.get("type")
            if ctype == "text":
                content_blocks.append(TextContent(text=c.get("text", "")))
            elif ctype == "image":
                content_blocks.append(
                    ImageContent(
                        source=c.get("data", ""),
                        media_type=c.get("mimeType", ""),
                    )
                )
        details: dict[str, Any] = {"raw_mcp_response": result}
        if "structuredContent" in result:
            details["structuredContent"] = result["structuredContent"]
        return AgentToolResult(
            content=content_blocks,
            details=details,
            is_error=True if result.get("isError") else None,
        )

    return AgentTool(
        name=name,
        description=description,
        parameters=parameters_model,
        execute=_execute,
    )


def _consume_task_exception(task: "asyncio.Task[Any]") -> None:
    """``add_done_callback`` target that swallows the result/exception
    of a fire-and-forget cancelled task. Without this, asyncio logs
    "task exception was never retrieved" at GC time for tasks we
    cancelled and don't await."""
    try:
        task.exception()
    except (asyncio.CancelledError, asyncio.InvalidStateError):
        pass


# Sentinel returned by :func:`_await_with_signal` on cooperative abort.
# The MCP ``_execute`` adapter checks for it and synthesizes a tool
# result rather than raising ``asyncio.CancelledError``, so callers
# that do ``agent.abort(); await prompt_task`` don't see a new
# exception shape from MCP-backed tools — the agent loop runs its
# normal abort lifecycle (ToolExecutionEnd → next turn →
# provider stream sees signal → AssistantMessage(stop_reason="aborted")
# → TurnEnd → AgentEnd). See codex round-1 follow-up on PR #88.
_ABORTED_SENTINEL: Any = object()


async def _await_with_signal(coro: Awaitable[Any], signal: Any) -> Any:
    """Await ``coro``, but bail out with :data:`_ABORTED_SENTINEL` if
    ``signal`` (an ``asyncio.Event`` set by ``agent.abort()``) fires
    while the awaitable is in flight.

    ``Agent.abort()`` only sets the cooperative signal — it does NOT
    cancel the running task — so without a co-watching coroutine an
    in-flight MCP ``tools/call`` would block until response or
    transport timeout regardless of abort. Cancellation latency would
    be bounded by the MCP timeout (30s default), not by the user's
    intent. Racing the signal here lets ``abort()`` propagate
    through MCP within a tick (codex overall-review MAJOR).

    Returning a sentinel rather than raising ``asyncio.CancelledError``
    keeps the agent lifecycle intact: the MCP ``_execute`` adapter
    converts the sentinel into a synthetic ``AgentToolResult`` so the
    normal ``ToolExecutionEnd`` / ``TurnEnd`` / ``AgentEnd`` flow
    completes without surfacing a new exception type to callers that
    only catch ``Exception`` (codex round-1 follow-up on PR #88).

    When ``signal`` is ``None`` (e.g. tool invoked outside the agent
    loop, or by a non-cubepi caller), just await the coroutine.
    """
    if signal is None or not hasattr(signal, "wait"):
        return await coro
    if signal.is_set():
        # Already aborted before we even started. Close the coroutine
        # so the caller doesn't get a "coroutine was never awaited"
        # warning — ``coro`` is fresh and hasn't started executing.
        if hasattr(coro, "close"):
            coro.close()
        return _ABORTED_SENTINEL
    call_task = asyncio.ensure_future(coro)
    signal_task = asyncio.ensure_future(signal.wait())
    try:
        done, pending = await asyncio.wait(
            {call_task, signal_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
    except BaseException:
        # Outer cancellation: clean up both tasks and re-raise.
        # Same fire-and-forget cleanup as the signal-won branch —
        # don't block on slow transport teardown.
        for t in (call_task, signal_task):
            if not t.done():
                t.cancel()
                t.add_done_callback(_consume_task_exception)
        raise
    if signal_task in done:
        # Cooperative abort fired first — cancel the in-flight RPC
        # and return the sentinel so ``_execute`` can synthesize a
        # graceful tool result.
        #
        # Don't ``await call_task`` here: ``call_remote`` may catch
        # ``CancelledError`` for slow transport cleanup (HTTP socket
        # close, MCP session shutdown), and awaiting would reintroduce
        # the abort-latency problem this helper was meant to fix —
        # the prompt task would still hang until the MCP transport
        # timeout despite the signal having won the race (codex
        # round-2 follow-up on PR #88).
        #
        # Cancellation is dispatched eagerly; the cleanup finishes
        # out-of-band. The ``add_done_callback`` consumes the
        # eventual exception so asyncio doesn't log a
        # "task exception was never retrieved" warning at GC time.
        if not call_task.done():
            call_task.cancel()
            call_task.add_done_callback(_consume_task_exception)
        return _ABORTED_SENTINEL
    # call_task completed first; tidy the signal watcher.
    if not signal_task.done():
        signal_task.cancel()
        try:
            await signal_task
        except BaseException:
            pass
    return call_task.result()


def get_mcp_field(value: object, snake_name: str, camel_name: str) -> Any:
    """Read an attribute that may be snake_case (v2) or camelCase (v1/wire).

    ``None`` means the attribute is absent. Falsy values such as ``{}``,
    ``[]``, ``0``, and ``False`` are real payload and must not fall through.
    """
    if hasattr(value, snake_name):
        snake_value = getattr(value, snake_name)
        if snake_value is not None:
            return snake_value
    return getattr(value, camel_name, None)


def serialize_mcp_call_tool_response(resp: Any) -> dict[str, Any]:
    """Normalize an MCP SDK CallToolResult (v1 or v2) to the adapter response dict shape."""
    content: list[dict[str, Any]] = []
    for block in getattr(resp, "content", None) or []:
        b_type = getattr(block, "type", None)
        if b_type == "text":
            content.append({"type": "text", "text": getattr(block, "text", "")})
        elif b_type == "image":
            content.append(
                {
                    "type": "image",
                    "data": getattr(block, "data", ""),
                    "mimeType": get_mcp_field(block, "mime_type", "mimeType") or "",
                }
            )
    result: dict[str, Any] = {
        "content": content,
        "isError": bool(get_mcp_field(resp, "is_error", "isError")),
    }
    structured = get_mcp_field(resp, "structured_content", "structuredContent")
    if structured is not None:
        result["structuredContent"] = structured
    return result
