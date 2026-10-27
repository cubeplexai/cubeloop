from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from cubeloop.errors import EmptyCompletion
from cubeloop.providers.base import (
    AssistantMessage,
    BaseProvider,
    ImageContent,
    Message,
    MessageStream,
    Model,
    ProviderResponse,
    StreamEvent,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    ToolResultMessage,
    Usage,
    UserMessage,
    _fire_request_listeners,
    _fire_response_listeners,
    apply_sender_attribution,
    invoke_on_payload,
    invoke_on_response,
    is_empty_completion,
    typed_error_event,
)
from cubeloop.providers.capability import (
    CapabilityDescriptor,
    apply_reasoning_control,
    apply_temperature,
)
from cubeloop.providers.reasoning_profiles import get_capability_profile


class OpenAIResponsesProvider(BaseProvider):
    """Provider that uses the OpenAI Responses API.

    The Responses API supports reasoning models (o-series) with streaming,
    tool use, and reasoning effort control.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        capability: CapabilityDescriptor | None = None,
        model_capability_overrides: dict[str, CapabilityDescriptor] | None = None,
        provider_id: str = "",
    ) -> None:
        super().__init__(provider_id=provider_id)
        import openai

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.AsyncOpenAI(**kwargs)

        # Track whether capability was explicitly passed so the Responses path
        # stays behavior-identical for legacy callers (who rely on the inline
        # _THINKING_TO_EFFORT map). Spec §3.5.
        self._cap_active: bool = (
            capability is not None or model_capability_overrides is not None
        )
        self._capability: CapabilityDescriptor = capability or get_capability_profile(
            "openai.responses"
        )
        self._model_overrides: dict[str, CapabilityDescriptor] = (
            model_capability_overrides or {}
        )

    def _resolve_capability(self, model_id: str) -> CapabilityDescriptor:
        return self._model_overrides.get(model_id, self._capability)

    @staticmethod
    def _map_tool_choice(choice: str) -> str | dict:
        if choice in ("auto", "required", "none"):
            return choice
        return {"type": "function", "name": choice}

    @staticmethod
    def _assemble_response(resp: Any) -> dict[str, Any] | None:
        """Convert the OpenAI Responses SDK Response object to its canonical
        dict shape (same as a non-streaming Responses.create() return value).
        """
        if resp is None:
            return None
        if hasattr(resp, "model_dump"):
            return resp.model_dump(mode="json")
        return dict(resp) if isinstance(resp, dict) else None

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
        opts = options or StreamOptions()
        ms = MessageStream()

        api_input = self._build_input(messages)

        kwargs: dict[str, Any] = {
            "model": model.id,
            "input": api_input,
            "stream": True,
            "store": False,
        }

        if system_prompt:
            role = "developer" if model.reasoning else "system"
            kwargs["input"] = [{"role": role, "content": system_prompt}] + api_input

        if tools:
            kwargs["tools"] = [self._convert_tool(t) for t in tools]

        if tool_choice is not None:
            kwargs["tool_choice"] = self._map_tool_choice(tool_choice)

        async def _produce() -> None:
            body: dict | None = None
            exc: BaseException | None = None
            try:
                nonlocal kwargs
                kwargs = await invoke_on_payload(opts.on_payload, kwargs, model)

                # Capability-driven payload mutations run AFTER on_payload so a
                # caller's explicit values survive via setdefault. Mirrors the
                # ordering used by OpenAIProvider. Spec §3.5.
                if self._cap_active or model.reasoning:
                    # The max_tokens_field rename used by the chat-completions
                    # OpenAIProvider doesn't apply here — Responses always uses
                    # ``max_output_tokens`` natively, so we skip it. Reasoning
                    # models reject ``temperature``, so guard the setdefault.
                    if not model.reasoning:
                        kwargs.setdefault("temperature", model.temperature)
                    if model.max_tokens:
                        kwargs.setdefault("max_output_tokens", model.max_tokens)
                    cap = self._resolve_capability(model.id)
                    apply_temperature(kwargs, cap.temperature)
                    apply_reasoning_control(kwargs, cap, opts.reasoning, model=model)
                else:
                    if model.max_tokens:
                        kwargs.setdefault("max_output_tokens", model.max_tokens)

                    # Forward temperature for non-reasoning models; reasoning
                    # models don't support temperature on the Responses API.
                    if not model.reasoning:
                        kwargs.setdefault("temperature", model.temperature)

                await _fire_request_listeners(self._request_listeners, kwargs, model)

                try:
                    response = await self._client.responses.create(**kwargs)
                except Exception as _sdk_exc:
                    from cubeloop.errors import classify_and_raise

                    classify_and_raise(_sdk_exc, model=model, messages=messages)

                # Invoke on_response with HTTP metadata if available
                http_response = getattr(response, "response", None)
                if http_response is not None:
                    await invoke_on_response(
                        opts.on_response,
                        ProviderResponse(
                            status=http_response.status_code,
                            headers=dict(http_response.headers),
                        ),
                        model,
                    )
                partial = AssistantMessage(
                    content=[],
                    usage=Usage(),
                    timestamp=time.time(),
                    provider_id=model.provider_id,
                    model_id=model.id,
                )
                await self._emit(
                    ms,
                    StreamEvent(type="start", partial=partial.model_copy(deep=True)),
                    model,
                )

                # Track current streaming state
                current_thinking = ""
                current_text = ""
                current_item_type: str | None = None
                current_content_index = 0
                # Per-item tool call state keyed by item.id
                tool_state: dict[str, dict[str, str]] = {}
                active_tool_item_id: str | None = None

                async for event in response:
                    if opts.signal and opts.signal.is_set():
                        aborted = partial.model_copy(
                            update={
                                "stop_reason": "aborted",
                                "error_message": "Request was aborted",
                            }
                        )
                        await self._emit(
                            ms,
                            StreamEvent(
                                type="error",
                                error_message="Request was aborted",
                            ),
                            model,
                        )
                        ms.set_result(aborted)
                        return

                    etype = event.type

                    # --- Output item lifecycle ---
                    if etype == "response.output_item.added":
                        item = event.item
                        if item.type == "reasoning":
                            current_item_type = "reasoning"
                            current_thinking = ""
                            partial.content.append(ThinkingContent(thinking=""))
                            current_content_index = len(partial.content) - 1
                            await self._emit(
                                ms,
                                StreamEvent(
                                    type="thinking_start",
                                    content_index=current_content_index,
                                    partial=partial.model_copy(deep=True),
                                ),
                                model,
                            )
                        elif item.type == "message":
                            current_item_type = "message"
                            current_text = ""
                            partial.content.append(TextContent(text=""))
                            current_content_index = len(partial.content) - 1
                            await self._emit(
                                ms,
                                StreamEvent(
                                    type="text_start",
                                    content_index=current_content_index,
                                    partial=partial.model_copy(deep=True),
                                ),
                                model,
                            )
                        elif item.type == "function_call":
                            current_item_type = "function_call"
                            item_id = item.id or ""
                            tc_id = (
                                f"{item.call_id}|{item_id}" if item_id else item.call_id
                            )
                            tool_state[item_id or item.call_id] = {
                                "call_id": item.call_id,
                                "item_id": item_id,
                                "name": item.name,
                                "json": "",
                                "tc_id": tc_id,
                            }
                            active_tool_item_id = item_id or item.call_id
                            partial.content.append(
                                ToolCall(
                                    id=tc_id,
                                    name=item.name,
                                    arguments={},
                                )
                            )
                            current_content_index = len(partial.content) - 1
                            await self._emit(
                                ms,
                                StreamEvent(
                                    type="toolcall_start",
                                    content_index=current_content_index,
                                    partial=partial.model_copy(deep=True),
                                ),
                                model,
                            )

                    # --- Reasoning (thinking) deltas ---
                    elif etype == "response.reasoning_summary_text.delta":
                        if current_item_type == "reasoning":
                            current_thinking += event.delta
                            if partial.content and isinstance(
                                partial.content[-1], ThinkingContent
                            ):
                                partial.content[-1] = ThinkingContent(
                                    thinking=current_thinking
                                )
                            await self._emit(
                                ms,
                                StreamEvent(
                                    type="thinking_delta",
                                    delta=event.delta,
                                    content_index=current_content_index,
                                    partial=partial.model_copy(deep=True),
                                ),
                                model,
                            )

                    elif etype == "response.reasoning_text.delta":
                        if current_item_type == "reasoning":
                            current_thinking += event.delta
                            if partial.content and isinstance(
                                partial.content[-1], ThinkingContent
                            ):
                                partial.content[-1] = ThinkingContent(
                                    thinking=current_thinking
                                )
                            await self._emit(
                                ms,
                                StreamEvent(
                                    type="thinking_delta",
                                    delta=event.delta,
                                    content_index=current_content_index,
                                    partial=partial.model_copy(deep=True),
                                ),
                                model,
                            )

                    elif etype == "response.reasoning_summary_part.done":
                        # Append separator between summary parts
                        if current_item_type == "reasoning":
                            current_thinking += "\n\n"
                            if partial.content and isinstance(
                                partial.content[-1], ThinkingContent
                            ):
                                partial.content[-1] = ThinkingContent(
                                    thinking=current_thinking
                                )
                            await self._emit(
                                ms,
                                StreamEvent(
                                    type="thinking_delta",
                                    delta="\n\n",
                                    content_index=current_content_index,
                                    partial=partial.model_copy(deep=True),
                                ),
                                model,
                            )

                    # --- Text deltas ---
                    elif etype == "response.output_text.delta":
                        if current_item_type == "message":
                            current_text += event.delta
                            if partial.content and isinstance(
                                partial.content[-1], TextContent
                            ):
                                partial.content[-1] = TextContent(text=current_text)
                            await self._emit(
                                ms,
                                StreamEvent(
                                    type="text_delta",
                                    delta=event.delta,
                                    content_index=current_content_index,
                                    partial=partial.model_copy(deep=True),
                                ),
                                model,
                            )

                    elif etype == "response.refusal.delta":
                        if current_item_type == "message":
                            current_text += event.delta
                            if partial.content and isinstance(
                                partial.content[-1], TextContent
                            ):
                                partial.content[-1] = TextContent(text=current_text)
                            await self._emit(
                                ms,
                                StreamEvent(
                                    type="text_delta",
                                    delta=event.delta,
                                    content_index=current_content_index,
                                    partial=partial.model_copy(deep=True),
                                ),
                                model,
                            )

                    # --- Tool call argument deltas ---
                    elif etype == "response.function_call_arguments.delta":
                        if active_tool_item_id and active_tool_item_id in tool_state:
                            tool_state[active_tool_item_id]["json"] += event.delta
                            await self._emit(
                                ms,
                                StreamEvent(
                                    type="toolcall_delta",
                                    delta=event.delta,
                                    content_index=current_content_index,
                                    partial=partial.model_copy(deep=True),
                                ),
                                model,
                            )

                    elif etype == "response.function_call_arguments.done":
                        if active_tool_item_id and active_tool_item_id in tool_state:
                            tool_state[active_tool_item_id]["json"] = event.arguments

                    # --- Output item done ---
                    elif etype == "response.output_item.done":
                        item = event.item
                        if item.type == "reasoning":
                            # Finalize thinking content from item summary
                            summary_text = ""
                            if hasattr(item, "summary") and item.summary:
                                summary_text = "\n\n".join(
                                    s.text for s in item.summary if hasattr(s, "text")
                                )
                            final_thinking = summary_text or current_thinking
                            if partial.content and isinstance(
                                partial.content[-1], ThinkingContent
                            ):
                                partial.content[-1] = ThinkingContent(
                                    thinking=final_thinking
                                )
                            await self._emit(
                                ms,
                                StreamEvent(
                                    type="thinking_end",
                                    content_index=current_content_index,
                                    partial=partial.model_copy(deep=True),
                                ),
                                model,
                            )
                            current_item_type = None

                        elif item.type == "message":
                            # Finalize text from item content
                            final_text = ""
                            if hasattr(item, "content") and item.content:
                                parts = []
                                for c in item.content:
                                    if hasattr(c, "text"):
                                        parts.append(c.text)
                                    elif hasattr(c, "refusal"):
                                        parts.append(c.refusal)
                                final_text = "".join(parts)
                            if (
                                final_text
                                and partial.content
                                and isinstance(partial.content[-1], TextContent)
                            ):
                                partial.content[-1] = TextContent(text=final_text)
                            await self._emit(
                                ms,
                                StreamEvent(
                                    type="text_end",
                                    content_index=current_content_index,
                                    partial=partial.model_copy(deep=True),
                                ),
                                model,
                            )
                            current_item_type = None

                        elif item.type == "function_call":
                            item_key = (item.id or "") or item.call_id
                            ts = tool_state.pop(item_key, None)
                            final_json = getattr(item, "arguments", None) or (
                                ts["json"] if ts else ""
                            )
                            try:
                                args = json.loads(final_json) if final_json else {}
                            except json.JSONDecodeError:
                                args = {}
                            item_id = item.id or ""
                            tc_id = (
                                f"{item.call_id}|{item_id}" if item_id else item.call_id
                            )
                            for i, c in enumerate(partial.content):
                                if isinstance(c, ToolCall) and c.id == tc_id:
                                    partial.content[i] = ToolCall(
                                        id=tc_id,
                                        name=item.name,
                                        arguments=args,
                                    )
                            await self._emit(
                                ms,
                                StreamEvent(
                                    type="toolcall_end",
                                    content_index=current_content_index,
                                    partial=partial.model_copy(deep=True),
                                ),
                                model,
                            )
                            current_item_type = None

                    # --- Response completed ---
                    elif etype == "response.completed":
                        resp = event.response
                        body = self._assemble_response(resp)
                        usage = Usage()
                        if resp and resp.usage:
                            cached = 0
                            if hasattr(resp.usage, "input_tokens_details"):
                                details = resp.usage.input_tokens_details
                                if details and hasattr(details, "cached_tokens"):
                                    cached = details.cached_tokens or 0
                            usage = Usage(
                                input_tokens=(resp.usage.input_tokens or 0) - cached,
                                output_tokens=resp.usage.output_tokens or 0,
                                cache_read_tokens=cached,
                            )

                        stop_reason = self._map_stop_reason(
                            resp.status if resp else None, partial
                        )
                        partial = partial.model_copy(
                            update={
                                "usage": usage,
                                "stop_reason": stop_reason,
                                "response_id": getattr(resp, "id", None)
                                if resp
                                else None,
                            }
                        )
                        if is_empty_completion(partial):
                            raise EmptyCompletion(
                                "Model returned an empty completion with no content and no tool calls",
                                provider=model.provider_id,
                                model=model.id,
                            )
                        await self._emit(ms, StreamEvent(type="done"), model)
                        ms.set_result(partial)
                        return

                    # --- Errors ---
                    elif etype == "error":
                        error_msg_text = ""
                        if hasattr(event, "message"):
                            error_msg_text = event.message
                        elif hasattr(event, "code"):
                            error_msg_text = f"Error code {event.code}"
                        raise RuntimeError(error_msg_text or "Unknown error")

                    elif etype == "response.failed":
                        resp = event.response
                        body = self._assemble_response(resp)
                        error_detail = ""
                        if resp and hasattr(resp, "error") and resp.error:
                            code = getattr(resp.error, "code", "unknown")
                            msg = getattr(resp.error, "message", "no message")
                            error_detail = f"{code}: {msg}"
                        elif (
                            resp
                            and hasattr(resp, "incomplete_details")
                            and resp.incomplete_details
                        ):
                            reason = getattr(
                                resp.incomplete_details, "reason", "unknown"
                            )
                            error_detail = f"incomplete: {reason}"
                        raise RuntimeError(error_detail or "Unknown error (no details)")

                # If we get here without response.completed, finalize
                if is_empty_completion(partial):
                    raise EmptyCompletion(
                        "Model returned an empty completion with no content and no tool calls",
                        provider=model.provider_id,
                        model=model.id,
                    )
                await self._emit(ms, StreamEvent(type="done"), model)
                ms.set_result(partial)

            except BaseException as e:
                exc = e
                # Classify SDK stream-level errors that the narrow create()
                # try/except didn't catch — async-for-event iteration errors
                # arrive here raw.
                if isinstance(e, Exception):
                    from cubeloop.errors import classify_and_raise

                    try:
                        classify_and_raise(e, model=model, messages=messages)
                    except Exception as _classified:
                        exc = _classified
                err_text = self._error_message(exc, model)
                ev = typed_error_event(exc, model=model, error_message=err_text)
                error_msg = AssistantMessage(
                    content=[],
                    stop_reason="error",
                    error_message=ev.error_message,
                    error_type=ev.error_type,
                    error_code=ev.error_code,
                    status_code=ev.status_code,
                    retry_after=ev.retry_after,
                    tokens_in=ev.tokens_in,
                    context_window=ev.context_window,
                    usage=Usage(),
                    timestamp=time.time(),
                    provider_id=ev.provider_id or model.provider_id,
                    model_id=ev.model_id or model.id,
                )
                await self._emit(ms, ev, model)
                ms.set_result(error_msg)
                if not isinstance(e, Exception):
                    raise
            finally:
                await _fire_response_listeners(
                    self._response_listeners, body, model, exc
                )

        ms.attach_task(asyncio.create_task(_produce()))
        return ms

    @staticmethod
    def _map_stop_reason(status: str | None, partial: AssistantMessage) -> str:
        """Map OpenAI response status to cubepi stop reason."""
        has_tool_calls = any(isinstance(c, ToolCall) for c in partial.content)
        reason_map = {
            "completed": "stop",
            "incomplete": "length",
            "failed": "error",
            "cancelled": "error",
            "in_progress": "stop",
            "queued": "stop",
        }
        reason = reason_map.get(status or "completed", "stop")
        if has_tool_calls and reason == "stop":
            reason = "tool_use"
        return reason

    @staticmethod
    def _build_input(messages: list[Message]) -> list[dict[str, Any]]:
        """Convert cubepi messages to OpenAI Responses API input format."""
        api_input: list[dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg, UserMessage):
                content: list[dict[str, Any]] = []
                attributed = apply_sender_attribution(msg, msg.content)
                for c in attributed:
                    if isinstance(c, TextContent):
                        content.append({"type": "input_text", "text": c.text})
                    elif isinstance(c, ImageContent):
                        content.append(
                            {
                                "type": "input_image",
                                "image_url": f"data:{c.media_type};base64,{c.source}",
                            }
                        )
                if content:
                    api_input.append({"role": "user", "content": content})

            elif isinstance(msg, AssistantMessage):
                text_blocks: list[dict[str, Any]] = []
                for assistant_block in msg.content:
                    if isinstance(assistant_block, ThinkingContent):
                        pass
                    elif isinstance(assistant_block, TextContent):
                        text_blocks.append(
                            {
                                "type": "output_text",
                                "text": assistant_block.text,
                                "annotations": [],
                            }
                        )
                    elif isinstance(assistant_block, ToolCall):
                        if text_blocks:
                            api_input.append(
                                {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": text_blocks,
                                    "status": "completed",
                                }
                            )
                            text_blocks = []
                        parts = assistant_block.id.split("|", 1)
                        call_id = parts[0]
                        item_id = parts[1] if len(parts) > 1 else None
                        fc: dict[str, Any] = {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": assistant_block.name,
                            "arguments": json.dumps(assistant_block.arguments),
                        }
                        if item_id:
                            fc["id"] = item_id
                        api_input.append(fc)
                if text_blocks:
                    api_input.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": text_blocks,
                            "status": "completed",
                        }
                    )

            elif isinstance(msg, ToolResultMessage):
                parts = msg.tool_call_id.split("|", 1)
                call_id = parts[0]
                text_parts = [c.text for c in msg.content if isinstance(c, TextContent)]
                api_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": "\n".join(text_parts),
                    }
                )

        return api_input

    @staticmethod
    def _convert_tool(td: ToolDefinition) -> dict[str, Any]:
        """Convert a cubepi ToolDefinition to OpenAI Responses API tool format."""
        return {
            "type": "function",
            "name": td.name,
            "description": td.description,
            "parameters": td.parameters,
        }
