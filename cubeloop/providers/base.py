from __future__ import annotations

import asyncio
import copy
import inspect
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Literal,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field

from cubeloop.types import JsonObject, StructuredValue

BaseModelT = TypeVar("BaseModelT", bound=BaseModel)

ReasoningMode = Literal["off", "auto", "on"]
ReasoningEffort = Literal["minimal", "low", "medium", "high", "max"]
ReasoningSummary = Literal["none", "auto", "detailed", "summarized"]

ToolChoice = Literal["auto", "required", "none"] | str


class ReasoningControl(BaseModel):
    """Provider-independent reasoning controls."""

    mode: ReasoningMode = "off"
    effort: ReasoningEffort = "medium"
    summary: ReasoningSummary = "none"


class ModelCost(BaseModel):
    input: float = 0
    output: float = 0
    cache_read: float = 0
    cache_write: float = 0


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class Model(BaseModel):
    id: str
    provider_id: str = ""
    api: str = ""
    reasoning: bool = False
    context_window: int = 200_000
    max_tokens: int = 8192
    temperature: float = 0.7
    cost: ModelCost | None = None


class StructuredOutputError(Exception):
    """Raised when generate_structured() cannot extract or validate the output."""


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str = ""


class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    source: str = ""
    media_type: str = ""


class ThinkingContent(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str = ""


Content = TextContent | ImageContent


class ToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: JsonObject


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: list[Content]
    timestamp: float | None = None
    metadata: JsonObject = Field(default_factory=dict)
    run_id: str | None = None


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[Content | ThinkingContent | ToolCall]
    stop_reason: str = "stop"
    error_message: str | None = None
    error_type: str | None = None
    error_code: str | None = None
    status_code: int | None = None
    retry_after: float | None = None
    tokens_in: int | None = None
    context_window: int | None = None
    usage: Usage | None = None
    timestamp: float | None = None
    provider_id: str = ""
    model_id: str = ""
    response_id: str | None = None
    metadata: JsonObject = Field(default_factory=dict)
    run_id: str | None = None


class ToolResultMessage(BaseModel):
    role: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    content: list[Content]
    details: StructuredValue = None
    is_error: bool = False
    timestamp: float | None = None
    metadata: JsonObject = Field(default_factory=dict)
    run_id: str | None = None


Message = UserMessage | AssistantMessage | ToolResultMessage


# --- Synthetic message marker (issue #171) ---
#
# Middleware and the framework itself sometimes inject user-role messages
# into the conversation (todo guard nudges, goal continuations, …). Without
# a marker these are indistinguishable from messages a human typed, so
# downstream UIs render internal scaffolding as user chat bubbles. The
# marker lives in ``metadata`` (not a subclass) because class identity does
# not survive the checkpointer/API round-trip — persisted messages are
# revalidated into the plain union types.

SYNTHETIC_METADATA_KEY = "synthetic"
SYNTHETIC_SOURCE_METADATA_KEY = "synthetic_source"


def synthetic_user_message(text: str, *, source: str) -> UserMessage:
    """Build a user-role message injected by the framework, not typed by a human.

    All injected user-role messages MUST be created through this factory so
    downstream consumers can branch on a single marker
    (``metadata["synthetic"] is True``). ``source`` (e.g. ``"todo_guard"``,
    ``"goal_continuation"``) is for tracing/debugging only — UI behavior
    should never depend on it.
    """
    return UserMessage(
        content=[TextContent(text=text)],
        metadata={
            SYNTHETIC_METADATA_KEY: True,
            SYNTHETIC_SOURCE_METADATA_KEY: source,
        },
    )


def is_synthetic_message(message: Message) -> bool:
    """True if ``message`` was injected by the framework rather than a human."""
    return message.metadata.get(SYNTHETIC_METADATA_KEY) is True


# --- Sender attribution (group chat) ---
#
# When multiple humans share one conversation, each user message carries
# the sender's display name in metadata. Providers render the prefix at
# call-time rather than callers mutating ``content`` at write-time. This
# keeps stored messages clean (frontend can show sender via avatar/badge
# without stripping a prefix), handles attachment-only messages correctly,
# and concentrates the formatting in one place.

SENDER_DISPLAY_NAME_METADATA_KEY = "sender_display_name"
SENDER_USER_ID_METADATA_KEY = "sender_user_id"


def get_sender_display_name(message: Message) -> str | None:
    """Return the sender's display name, or None for non-attributed messages."""
    value = message.metadata.get(SENDER_DISPLAY_NAME_METADATA_KEY)
    return value if isinstance(value, str) and value else None


def apply_sender_attribution(
    message: UserMessage, content: list[Content]
) -> list[Content]:
    """Prefix the first non-empty text block with ``[name]: ``.

    Returns a new list — the original ``content`` is not mutated. If the
    message has no sender metadata, returns ``content`` unchanged. If the
    message has sender metadata but no text content (attachment-only),
    prepends a synthetic text block so the model still knows who sent
    the attachments.
    """
    display_name = get_sender_display_name(message)
    if not display_name:
        return content

    prefix = f"[{display_name}]: "
    out: list[Content] = []
    applied = False
    for block in content:
        if not applied and isinstance(block, TextContent) and block.text:
            out.append(TextContent(text=prefix + block.text))
            applied = True
        else:
            out.append(block)

    if not applied:
        # Attachment-only or empty text — prepend a sender announcement so
        # the agent doesn't see a faceless attachment list in a group chat.
        out.insert(0, TextContent(text=f"[{display_name}] sent:"))

    return out


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: JsonObject


class StreamEvent(BaseModel):
    type: Literal[
        "start",
        "text_start",
        "text_delta",
        "text_end",
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_end",
        "done",
        "error",
    ]
    content_index: int | None = None
    delta: str | None = None
    partial: AssistantMessage | None = None
    error_message: str | None = None
    error_type: str | None = None
    error_code: str | None = None
    status_code: int | None = None
    retry_after: float | None = None
    provider_id: str | None = None
    model_id: str | None = None
    tokens_in: int | None = None
    context_window: int | None = None


def typed_error_event(
    exc: BaseException,
    *,
    model: "Model",
    error_message: str,
) -> StreamEvent:
    """Build a typed error ``StreamEvent`` from a classified exception."""

    from cubeloop.errors import annotate_error_event

    fields = annotate_error_event(exc, fallback_message=error_message)
    error_type = fields.get("error_type")
    error_code = fields.get("error_code")
    status_code = fields.get("status_code")
    retry_after = fields.get("retry_after")
    tokens_in = fields.get("tokens_in")
    context_window = fields.get("context_window")
    return StreamEvent(
        type="error",
        error_message=str(fields.get("error_message") or error_message),
        error_type=error_type if isinstance(error_type, str) else None,
        error_code=error_code if isinstance(error_code, str) else None,
        status_code=status_code if isinstance(status_code, int) else None,
        retry_after=float(retry_after)
        if isinstance(retry_after, (int, float))
        else None,
        provider_id=str(fields.get("provider_id") or model.provider_id),
        model_id=str(fields.get("model_id") or model.id),
        tokens_in=tokens_in if isinstance(tokens_in, int) else None,
        context_window=context_window if isinstance(context_window, int) else None,
    )


def format_provider_error(
    exc: BaseException,
    model: "Model",
    base_url: str | None = None,
) -> str:
    """Build a self-describing error string for a failed provider call.

    SDK exceptions like openai's ``APIConnectionError`` stringify to bare
    ``"Connection error."`` and drop the transport failure into ``__cause__``
    / ``__context__``. Without provider/model/cause context, a logged or
    persisted error tells you nothing about *which* model failed or *why*.
    """
    target = f"{model.provider_id}/{model.id}" if model.provider_id else model.id
    if base_url:
        target = f"{target} @ {base_url}"

    chain: list[str] = [f"{type(exc).__name__}: {exc}"]
    seen: set[int] = {id(exc)}
    cur: BaseException | None = exc
    while cur is not None:
        nxt = cur.__cause__ or cur.__context__
        if nxt is None or id(nxt) in seen:
            break
        seen.add(id(nxt))
        chain.append(f"{type(nxt).__name__}: {nxt}")
        cur = nxt

    detail = " <- ".join(chain)
    return f"[{target}] {detail}"


class MessageStream:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._result_future: asyncio.Future[AssistantMessage] = (
            asyncio.get_running_loop().create_future()
        )
        self._producer_task: asyncio.Task | None = None

    def attach_task(self, task: asyncio.Task) -> None:
        self._producer_task = task
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc and not self._result_future.done():
            self._result_future.set_exception(exc)
            self._queue.put_nowait(None)

    def push(self, event: StreamEvent) -> None:
        self._queue.put_nowait(event)
        if event.type in ("done", "error"):
            self._queue.put_nowait(None)

    def set_result(self, message: AssistantMessage) -> None:
        if not self._result_future.done():
            self._result_future.set_result(message)

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self

    async def __anext__(self) -> StreamEvent:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def result(self) -> AssistantMessage:
        """Return the final assistant message.

        Blocks not just on the result future, but also on the producer
        task's completion — the producer's ``finally`` block runs
        :func:`_fire_response_listeners` (after ``set_result``), so
        without waiting for the task, callers under ``asyncio.run``
        teardown could exit before async response listeners have run.
        Producer exceptions are NOT re-raised here; they were already
        surfaced via the result future or the stream's error events.
        """
        msg = await self._result_future
        task = self._producer_task
        # Don't await ourselves — providers may call result() from inside
        # the producer task (e.g. to read an aborted result set by an
        # internal helper); that path would deadlock.
        if task is not None and not task.done() and task is not asyncio.current_task():
            # ``asyncio.shield`` lets the producer's finally (response
            # listener cleanup) keep running independently if our caller
            # is cancelled while we're waiting. We re-raise the caller's
            # CancelledError so cooperative shutdown/timeouts still
            # propagate; the producer continues unaffected.
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                raise
            except BaseException:
                # Producer raised — caller already has the result via the
                # future. Swallow so result() returns the message.
                pass
        return msg


@dataclass
class ProviderResponse:
    """HTTP response metadata exposed to on_response callbacks."""

    status: int
    headers: dict[str, str] = field(default_factory=dict)


OnPayloadCallback = Callable[[dict, Model], Awaitable[dict | None] | dict | None]
"""Optional callback for inspecting/replacing provider payloads before sending.
Return a dict to replace the payload, or None to keep unchanged."""

OnResponseCallback = Callable[["ProviderResponse", Model], Awaitable[None] | None]
"""Optional callback invoked after an HTTP response is received."""


OnRequestCallback = Callable[[dict, Model], Awaitable[None] | None]
"""Persistent observer. Fires just before HTTP send, after any per-call
``StreamOptions.on_payload`` mutation has been applied. Receives the final
wire payload dict and the Model. Return value is ignored."""

OnChunkCallback = Callable[["StreamEvent", Model], Awaitable[None] | None]
"""Persistent observer. Fires for every StreamEvent pushed onto the stream
(start, text_delta, thinking_delta, toolcall_delta, done, error, ...).
Heavy listeners should early-return on irrelevant event types — this hook
fires hot. Return value is ignored."""

OnResponseBodyCallback = Callable[
    [dict | None, Model, BaseException | None], Awaitable[None] | None
]
"""Persistent observer. Fires exactly once per ``stream()`` call, in a
finally block, after the stream terminates.

- body: assembled provider response as a dict (same shape a non-streaming
  call to the provider would have returned), or None if the stream failed
  before a response could be assembled.
- exc: the exception that ended the stream (including
  ``asyncio.CancelledError``), or None on normal completion.
Return value is ignored."""


async def _fire_listeners(listeners: list[Callable], *args: Any) -> None:
    """Invoke each listener with ``*args``. Listener return values and
    exceptions are ignored — a buggy listener must never crash the stream.

    Iterates a snapshot (``tuple(listeners)``) so a listener that detaches
    itself mid-iteration does not silently skip subsequent listeners.
    Callers are responsible for the hot-path guard
    (``if self._chunk_listeners:`` before ``await``) — see ``_emit`` helpers
    in each provider."""
    if not listeners:
        return
    for cb in tuple(listeners):
        try:
            result = cb(*args)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001 — intentional broad catch
            _log_listener_exception(cb, exc)


def _fire_listeners_sync(listeners: list[Callable], *args: Any) -> None:
    """Synchronous variant of :func:`_fire_listeners`. Used in producer
    ``finally`` blocks where awaiting another coroutine after a
    cancellation is unreliable — the outer task is already cancelling, so
    a subsequent ``await`` may not get a chance to run its callee's body.

    Sync listeners are invoked directly. Async listeners are scheduled as
    detached tasks wrapped in :func:`_safe_run_coroutine` so that any
    exception raised inside the coroutine body is logged via
    :func:`_log_listener_exception` rather than bubbling up to asyncio as
    an "unhandled task exception" warning."""
    if not listeners:
        return
    for cb in tuple(listeners):
        try:
            result = cb(*args)
            if inspect.isawaitable(result):
                wrapped = _safe_run_coroutine(cb, result)
                try:
                    asyncio.create_task(wrapped)
                except RuntimeError:
                    # No running event loop (e.g. teardown) — close both
                    # the wrapper and the inner listener coroutine so
                    # neither leaks as a "coroutine was never awaited"
                    # warning.
                    wrapped.close()
                    close = getattr(result, "close", None)
                    if callable(close):
                        close()
        except Exception as exc:  # noqa: BLE001 — intentional broad catch
            _log_listener_exception(cb, exc)


async def _fire_chunk_listeners(
    listeners: list[Callable], event: "StreamEvent", model: "Model"
) -> None:
    """Fire :func:`subscribe_chunk` listeners with **per-listener** deep
    copies of the event.

    The consumer's queued ``StreamEvent`` (already pushed onto the
    ``MessageStream``) and every other listener's copy are isolated.
    Without per-listener copies, a redacting listener that mutated
    ``event.partial`` would silently alter what later chunk listeners
    observed in the same stream.
    """
    if not listeners:
        return
    for cb in tuple(listeners):
        try:
            result = cb(event.model_copy(deep=True), model)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001 — intentional broad catch
            _log_listener_exception(cb, exc)


async def _fire_request_listeners(
    listeners: list[Callable], payload: dict, model: "Model"
) -> None:
    """Fire :func:`subscribe_request` listeners with **per-listener** deep
    copies of the payload.

    ``subscribe_request`` is documented as an **observer** — the
    mutation hook is the per-call ``StreamOptions.on_payload`` slot,
    which runs before this. Two isolation properties matter here:

    1. The dict the provider is about to send over the wire (the
       caller's ``kwargs``) must not be mutated by any listener.
    2. Multi-subscriber observability must be order-independent — one
       listener redacting fields in place must not affect what later
       listeners observe.

    Both are achieved by giving each listener its own ``deepcopy``.
    Cost: at most one deepcopy per registered listener per stream call,
    which is negligible compared to the HTTP call itself.
    """
    if not listeners:
        return
    for cb in tuple(listeners):
        try:
            snapshot = copy.deepcopy(payload)
            result = cb(snapshot, model)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001 — intentional broad catch
            _log_listener_exception(cb, exc)


async def _fire_response_listeners(
    listeners: list[Callable],
    body: dict | None,
    model: "Model",
    exc: BaseException | None,
) -> None:
    """Fire response listeners with the right strategy for each
    termination path.

    Normal completion (``exc is None``) or an in-stream exception that is
    NOT a cancel: ``await`` each async listener inline so the producer
    task doesn't end before the listener has run — important because
    callers that use ``asyncio.run(main())`` will tear down the loop the
    moment ``main()`` returns, cancelling any still-detached listener
    task.

    Producer task cancellation (``exc`` is :class:`asyncio.CancelledError`):
    fall back to :func:`_fire_listeners_sync`. Awaiting inside a finally
    block of a cancelled task is unreliable — the runtime may skip past
    the await without running the callee — so synchronous listeners run
    inline and async listeners are scheduled as detached best-effort
    tasks. This is the same contract the cancellation tests pin.

    In both paths, each listener receives its own ``deepcopy`` of
    ``body`` so multi-subscriber observability is order-independent.
    """
    if not listeners:
        return
    if isinstance(exc, asyncio.CancelledError):
        _fire_listeners_sync_per_listener(listeners, body, model, exc)
        return
    for cb in tuple(listeners):
        try:
            snapshot = copy.deepcopy(body) if body is not None else None
            result = cb(snapshot, model, exc)
            if inspect.isawaitable(result):
                await result
        except Exception as listener_exc:  # noqa: BLE001
            _log_listener_exception(cb, listener_exc)


def _fire_listeners_sync_per_listener(
    listeners: list[Callable],
    body: dict | None,
    model: "Model",
    exc: BaseException | None,
) -> None:
    """Sync fanout for response listeners on the cancellation path,
    giving each listener its own deep copy of ``body``."""
    if not listeners:
        return
    for cb in tuple(listeners):
        try:
            snapshot = copy.deepcopy(body) if body is not None else None
            result = cb(snapshot, model, exc)
            if inspect.isawaitable(result):
                wrapped = _safe_run_coroutine(cb, result)
                try:
                    asyncio.create_task(wrapped)
                except RuntimeError:
                    wrapped.close()
                    close = getattr(result, "close", None)
                    if callable(close):
                        close()
        except Exception as listener_exc:  # noqa: BLE001
            _log_listener_exception(cb, listener_exc)


async def _safe_run_coroutine(cb: Callable, coro: Any) -> None:
    """Await an arbitrary coroutine, logging and swallowing any exception
    it raises. Used by :func:`_fire_listeners_sync` so async listener
    failures don't become asyncio unhandled-task-exception warnings."""
    try:
        await coro
    except Exception as exc:  # noqa: BLE001 — intentional broad catch
        _log_listener_exception(cb, exc)


def _log_listener_exception(cb: Callable, exc: BaseException) -> None:
    logging.getLogger("cubeloop.providers").warning(
        "cubepi provider listener %r raised; swallowed", cb, exc_info=exc
    )


def _detach(listeners: list, cb: Callable) -> None:
    try:
        listeners.remove(cb)
    except ValueError:
        pass


OnModelAttemptCallback = Callable[[Model], Awaitable[None] | None]
"""Called immediately before each concrete provider-model attempt."""


class StreamOptions(BaseModel):
    """Options bag for Provider.stream(), transparent to the agent loop."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    reasoning: ReasoningControl = Field(default_factory=ReasoningControl)
    signal: asyncio.Event | None = None
    on_payload: OnPayloadCallback | None = None
    on_response: OnResponseCallback | None = None
    on_model_attempt: OnModelAttemptCallback | None = None


async def invoke_on_payload(
    callback: OnPayloadCallback | None,
    payload: dict,
    model: Model,
) -> dict:
    """Call *on_payload* and return the (possibly replaced) payload dict."""
    if callback is None:
        return payload
    result = callback(payload, model)
    if inspect.isawaitable(result):
        result = await result
    return result if isinstance(result, dict) else payload


async def invoke_on_response(
    callback: OnResponseCallback | None,
    response: ProviderResponse,
    model: Model,
) -> None:
    """Call *on_response* if provided."""
    if callback is None:
        return
    result = callback(response, model)
    if inspect.isawaitable(result):
        await result


@runtime_checkable
class Provider(Protocol):
    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream: ...

    async def generate(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: StreamOptions | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        reasoning: ReasoningControl | None = None,
    ) -> AssistantMessage: ...


class BaseProvider:
    """Concrete base class for built-in cubepi providers.

    Built-in providers (Anthropic, OpenAI, OpenAI Responses, Faux) inherit
    from this class to gain the persistent listener registry used by
    ``cubeloop.tracing`` and other observers. User-defined providers should
    inherit from ``BaseProvider`` unless they implement the full ``Provider``
    protocol themselves.

    Concrete subclasses must implement ``stream()`` and call
    ``_fire_listeners`` at three points: after the request payload is
    finalized, for each ``StreamEvent`` pushed onto the stream, and exactly
    once in a ``finally`` block after the stream terminates.

    Per-call mutators (``StreamOptions.on_payload``,
    ``StreamOptions.on_response``) retain their existing single-slot
    semantics and fire independently of the persistent listener registry
    below.
    """

    def __init__(self, *, provider_id: str = "") -> None:
        self.provider_id = provider_id
        self._request_listeners: list[OnRequestCallback] = []
        self._chunk_listeners: list[OnChunkCallback] = []
        self._response_listeners: list[OnResponseBodyCallback] = []

    def model(
        self,
        id: str,
        *,
        api: str = "",
        reasoning: bool = False,
        context_window: int = 200_000,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        cost: ModelCost | None = None,
    ) -> BoundModel:
        return BoundModel(
            provider=self,
            spec=Model(
                id=id,
                provider_id=self.provider_id,
                api=api,
                reasoning=reasoning,
                context_window=context_window,
                max_tokens=max_tokens,
                temperature=temperature,
                cost=cost,
            ),
        )

    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream:
        raise NotImplementedError

    async def generate(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: StreamOptions | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        reasoning: ReasoningControl | None = None,
    ) -> AssistantMessage:
        """Run a single provider call and return the final assistant message."""
        model_updates: dict[str, int | float] = {}
        if max_output_tokens is not None:
            model_updates["max_tokens"] = max_output_tokens
        if temperature is not None:
            model_updates["temperature"] = temperature
        if model_updates:
            model = model.model_copy(update=model_updates)

        option_updates: dict[str, ReasoningControl] = {}
        if reasoning is not None:
            option_updates["reasoning"] = reasoning
        if option_updates:
            base_options = options or StreamOptions()
            options = base_options.model_copy(update=option_updates)

        stream = await self.stream(
            model=model,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            tool_choice=tool_choice,
            options=options,
        )
        async for event in stream:
            if event.type in ("done", "error"):
                break
        return await stream.result()

    def _error_message(self, exc: BaseException, model: Model) -> str:
        """Format a failed-call error with provider/model/base_url + cause."""
        base_url: str | None = None
        client = getattr(self, "_client", None)
        if client is not None:
            base_url = str(getattr(client, "base_url", "") or "") or None
        return format_provider_error(exc, model, base_url)

    def subscribe_request(self, cb: OnRequestCallback) -> Callable[[], None]:
        """Register a persistent observer for request payloads.

        Returns a detach callable that removes this specific subscription.
        """
        self._request_listeners.append(cb)
        return lambda: _detach(self._request_listeners, cb)

    def subscribe_chunk(self, cb: OnChunkCallback) -> Callable[[], None]:
        """Register a persistent observer for stream chunks.

        Returns a detach callable.
        """
        self._chunk_listeners.append(cb)
        return lambda: _detach(self._chunk_listeners, cb)

    def subscribe_response(self, cb: OnResponseBodyCallback) -> Callable[[], None]:
        """Register a persistent observer for assembled responses.

        Returns a detach callable.
        """
        self._response_listeners.append(cb)
        return lambda: _detach(self._response_listeners, cb)

    async def _emit(
        self,
        ms: "MessageStream",
        event: "StreamEvent",
        model: Model | None,
    ) -> None:
        """Push an event to the message stream and fan out to chunk
        listeners with an isolated deep copy.

        The synchronous guard on ``self._chunk_listeners`` makes the
        no-listener case zero-await — important because this fires on
        every text delta. When listeners are present, each receives a
        ``model_copy(deep=True)`` of the event so a redacting/mutating
        observer cannot edit the same object that was already enqueued
        for the ``async for`` consumer.

        ``model`` may be ``None`` when invoked from internal helpers
        that bypass ``stream()`` (e.g. ``FauxProvider._stream_with_deltas``
        is exposed for direct calls in tests). In that case the listener
        fan-out is skipped — the test isn't observing listeners anyway.
        """
        ms.push(event)
        if model is not None and self._chunk_listeners:
            await _fire_chunk_listeners(self._chunk_listeners, event, model)


@dataclass(frozen=True)
class BoundModel:
    provider: Provider
    spec: Model

    async def stream(
        self,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream:
        # Forward ``model`` and ``messages`` positionally so a custom
        # provider that follows the protocol shape but uses different
        # parameter names (e.g. ``model_spec``, ``msgs``) keeps working —
        # the pre-BoundModel loop also called ``provider.stream(model,
        # messages, ...)`` positionally. The remaining args sit after
        # ``*`` in the protocol so they must stay keyword.
        if options is not None and options.on_model_attempt is not None:
            result = options.on_model_attempt(self.spec)
            if inspect.isawaitable(result):
                await result
        return await self.provider.stream(
            self.spec,
            messages,
            system_prompt=system_prompt,
            tools=tools,
            tool_choice=tool_choice,
            options=options,
        )

    async def generate(
        self,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: StreamOptions | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        reasoning: ReasoningControl | None = None,
    ) -> AssistantMessage:
        # Same positional-forwarding rationale as ``stream`` above.
        return await self.provider.generate(
            self.spec,
            messages,
            system_prompt=system_prompt,
            tools=tools,
            tool_choice=tool_choice,
            options=options,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            reasoning=reasoning,
        )

    async def generate_structured(
        self,
        output_type: type[BaseModelT],
        messages: list[Message],
        *,
        system_prompt: str = "",
        tool_name: str = "structured_output",
        tool_description: str = "Return the structured output",
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        max_retries: int = 1,
    ) -> BaseModelT:
        schema = output_type.model_json_schema()
        tool = ToolDefinition(
            name=tool_name,
            description=tool_description,
            parameters=schema,
        )
        default_hint = (
            f"You MUST respond by calling the '{tool_name}' tool. "
            "Do NOT respond with plain text."
        )
        full_system = (
            f"{system_prompt}\n\n{default_hint}".strip()
            if system_prompt
            else default_hint
        )

        attempt_messages = list(messages)
        last_error: Exception | None = None

        for _ in range(1 + max_retries):
            response = await self.generate(
                attempt_messages,
                system_prompt=full_system,
                tools=[tool],
                tool_choice=tool_name,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
            for block in response.content:
                if isinstance(block, ToolCall) and block.name == tool_name:
                    try:
                        return output_type.model_validate(block.arguments)
                    except Exception as exc:
                        last_error = exc
                        error_text = (
                            f"Validation error: {exc}. Fix the data and try again."
                        )
                        attempt_messages = [
                            *attempt_messages,
                            response,
                            ToolResultMessage(
                                tool_call_id=block.id,
                                tool_name=block.name,
                                content=[TextContent(text=error_text)],
                                is_error=True,
                            ),
                            synthetic_user_message(
                                error_text, source="structured_output_retry"
                            ),
                        ]
                        break
            else:
                raise StructuredOutputError(
                    f"Model returned no tool call for '{tool_name}'; "
                    f"got stop_reason={response.stop_reason!r}"
                )

        raise StructuredOutputError(
            f"Structured output validation failed after retries: {last_error}"
        ) from last_error


def chain_providers(model: object) -> list["BaseProvider"]:
    """Return the unique BaseProvider instances backing a bound model.

    Walks a model's ``chain`` (FallbackBoundModel-like) or its single
    ``.provider`` (plain BoundModel-like). Uses duck typing on ``.chain`` to
    avoid importing :mod:`cubeloop.providers.fallback` into this base module —
    ``fallback`` already depends on ``base``, so the reverse import would
    cycle.

    For ``FallbackBoundModel``-like inputs, iterates the chain and dedupes
    providers by identity. Chain entries whose ``.provider`` isn't a
    ``BaseProvider`` are logged at WARNING via stdlib ``logging`` and
    skipped — tracing / metrics need the ``subscribe_request`` /
    ``subscribe_chunk`` / ``subscribe_response`` interface that only
    ``BaseProvider`` exposes. (cubepi does not depend on loguru; hosts that
    use it can intercept stdlib logging.)

    For plain bound models, returns ``[model.provider]`` if it is a
    ``BaseProvider``. For ``None`` or any other input, returns ``[]``.

    Used by :meth:`cubeloop.tracing.recorder.Recorder.attach` and
    :meth:`cubeloop.tracing.meter.Meter.attach` to subscribe to every leg of a
    fallback chain so post-failover provider events land in the trace /
    metric stream.
    """
    if model is None:
        return []
    chain = getattr(model, "chain", None)
    if chain is not None:
        seen: set[int] = set()
        out: list[BaseProvider] = []
        for idx, bm in enumerate(chain):
            p = getattr(bm, "provider", None)
            if not isinstance(p, BaseProvider):
                logging.getLogger("cubeloop.providers.base").warning(
                    "cubeloop.providers.base.chain_providers: chain[%d] provider "
                    "%s is not a BaseProvider; tracing/metrics will skip this leg",
                    idx,
                    type(p).__name__,
                )
                continue
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
        return out
    provider = getattr(model, "provider", None)
    if isinstance(provider, BaseProvider):
        return [provider]
    return []


def collect_agent_providers(agent: Any) -> list["BaseProvider"]:
    """Return the unique BaseProvider instances backing an agent.

    Walks ``agent._model`` via :func:`chain_providers` and falls back to a
    public ``provider`` attribute on the agent for legacy code paths. Used
    by both :meth:`cubeloop.tracing.recorder.Recorder.attach` and
    :meth:`cubeloop.tracing.meter.Meter.attach` to dedupe the prelude that
    finds providers to subscribe to.
    """
    model = getattr(agent, "_model", None)
    providers = chain_providers(model)
    if not providers:
        legacy = getattr(agent, "provider", None)
        if isinstance(legacy, BaseProvider):
            providers = [legacy]
    return providers
