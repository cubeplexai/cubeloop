from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Literal, Protocol, runtime_checkable

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

CacheRetention = Literal["short", "long", "none"]


@runtime_checkable
class CacheMarkerPolicy(Protocol):
    """Policy controlling where Anthropic cache_control markers are inserted.

    See `cubepi/docs/specs/2026-05-13-cubepi-cubebox-readiness-design.md` § D3.
    """

    def mark_system(self) -> bool: ...
    def mark_last_tool(self) -> bool: ...
    def message_breakpoint_indices(
        self,
        messages: list[Message],
    ) -> list[int]: ...


class DefaultCacheMarkerPolicy:
    """Preserves cubepi v0.2 behavior: system + last message + last tool."""

    def mark_system(self) -> bool:
        return True

    def mark_last_tool(self) -> bool:
        return True

    def message_breakpoint_indices(self, messages: list[Message]) -> list[int]:
        return [len(messages) - 1] if messages else []


class AnthropicProvider(BaseProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_retention: CacheRetention = "short",
        cache_policy: CacheMarkerPolicy | None = None,
        capability: CapabilityDescriptor | None = None,
        model_capability_overrides: dict[str, CapabilityDescriptor] | None = None,
        provider_id: str = "",
    ) -> None:
        super().__init__(provider_id=provider_id)
        import anthropic

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url is not None:
            kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)
        self._cache_retention = cache_retention
        self._cache_policy: CacheMarkerPolicy = (
            cache_policy or DefaultCacheMarkerPolicy()
        )
        # Anthropic always runs the capability path; capability=None falls back
        # to the built-in Anthropic Messages profile, which mirrors legacy
        # wire bytes.
        self._capability: CapabilityDescriptor = (
            capability
            if capability is not None
            else get_capability_profile("anthropic", "messages")
        )
        self._model_overrides: dict[str, CapabilityDescriptor] = (
            model_capability_overrides or {}
        )

    def _resolve_capability(self, model_id: str) -> CapabilityDescriptor:
        return self._model_overrides.get(model_id, self._capability)

    @staticmethod
    def _map_tool_choice(choice: str) -> dict[str, str] | None:
        if choice == "auto":
            return {"type": "auto"}
        if choice == "required":
            return {"type": "any"}
        if choice == "none":
            return None
        return {"type": "tool", "name": choice}

    @staticmethod
    def _apply_anthropic_budget_guard(kwargs: dict[str, Any], model: Model) -> None:
        budget = 0
        thinking_block = kwargs.get("thinking")
        if isinstance(thinking_block, dict):
            budget = thinking_block.get("budget_tokens", 0) or 0
        kwargs["max_tokens"] = min(model.max_tokens + budget, model.context_window)

        min_output_tokens = 1024
        if budget <= 0 or kwargs["max_tokens"] - budget >= min_output_tokens:
            return

        new_budget = max(0, kwargs["max_tokens"] - min_output_tokens)
        if isinstance(thinking_block, dict):
            if new_budget > 0:
                thinking_block["budget_tokens"] = new_budget
            else:
                kwargs["thinking"] = {"type": "disabled"}

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
        cap = self._resolve_capability(model.id)

        cache_control = self._get_cache_control()
        # Strip persisted "this turn errored before any output" markers (empty
        # AssistantMessages at the tail) before BOTH the API conversion and the
        # cache-policy query: leaving them in would make Anthropic treat the
        # trailing empty assistant as a prefill, and querying the policy with
        # the original list would point cache_control at a now-dropped message.
        messages = self._strip_trailing_empty_assistants(messages)
        api_messages, breakpoints = self._build_api_messages(messages)
        if cache_control:
            indices = self._cache_policy.message_breakpoint_indices(messages)
            targets = [breakpoints[i] for i in indices if 0 <= i < len(breakpoints)]
            self._apply_breakpoint_markers(api_messages, targets, cache_control)

        kwargs: dict[str, Any] = {
            "model": model.id,
            "messages": api_messages,
        }
        if system_prompt:
            if cache_control and self._cache_policy.mark_system():
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": cache_control,
                    }
                ]
            else:
                kwargs["system"] = system_prompt
        if tools and tool_choice != "none":
            api_tools = [self._convert_tool(t) for t in tools]
            if cache_control and api_tools and self._cache_policy.mark_last_tool():
                api_tools[-1]["cache_control"] = cache_control
            kwargs["tools"] = api_tools

        if tool_choice is not None and tool_choice != "none":
            mapped = self._map_tool_choice(tool_choice)
            if mapped is not None:
                kwargs["tool_choice"] = mapped

        apply_reasoning_control(kwargs, cap, opts.reasoning, model=model)
        thinking_block = kwargs.get("thinking")
        thinking_enabled = (
            isinstance(thinking_block, dict) and thinking_block.get("type") == "enabled"
        )
        if not thinking_enabled:
            kwargs.setdefault("temperature", model.temperature)
            apply_temperature(kwargs, cap.temperature)
            kwargs["max_tokens"] = min(model.max_tokens, model.context_window)
        else:
            kwargs.pop("temperature", None)
            self._apply_anthropic_budget_guard(kwargs, model)

        async def _produce() -> None:
            body: dict | None = None
            exc: BaseException | None = None
            try:
                nonlocal kwargs
                kwargs = await invoke_on_payload(opts.on_payload, kwargs, model)
                await _fire_request_listeners(self._request_listeners, kwargs, model)

                try:
                    async with self._client.messages.stream(**kwargs) as stream:
                        # Invoke on_response with HTTP metadata if available
                        http_response = getattr(stream, "response", None)
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
                            StreamEvent(
                                type="start", partial=partial.model_copy(deep=True)
                            ),
                            model,
                        )

                        # Accumulate streamed tool-call args JSON per content index
                        # so toolcall_end can carry parsed arguments (the Anthropic
                        # stream only delivers them as incremental partial_json).
                        tool_args_buffers: dict[int, str] = {}
                        async for event in stream:
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

                            await self._handle_event(
                                event, partial, ms, model, tool_args_buffers
                            )

                        final_msg = await stream.get_final_message()
                        result = self._convert_response(final_msg, model)
                        if is_empty_completion(result):
                            raise EmptyCompletion(
                                "Model returned an empty completion with no content and no tool calls",
                                provider=model.provider_id,
                                model=model.id,
                            )
                        body = self._assemble_response(final_msg)
                        await self._emit(ms, StreamEvent(type="done"), model)
                        ms.set_result(result)
                except Exception as _sdk_exc:
                    from cubeloop.errors import classify_and_raise

                    classify_and_raise(_sdk_exc, model=model, messages=messages)

            except BaseException as e:
                exc = e
                err_text = self._error_message(e, model)
                ev = typed_error_event(e, model=model, error_message=err_text)
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

    def _get_cache_control(self) -> dict[str, str] | None:
        if self._cache_retention == "none":
            return None
        cc: dict[str, str] = {"type": "ephemeral"}
        if self._cache_retention == "long":
            cc["ttl"] = "1h"
        return cc

    def _apply_breakpoint_markers(
        self,
        api_messages: list[dict[str, Any]],
        targets: list[tuple[int, int]],
        cache_control: dict[str, str],
    ) -> None:
        """Apply cache_control to specific ``(message, block)`` positions.

        ``targets`` come from :meth:`_build_api_messages`, which tracks the exact
        block each source message contributed. Marking that block (rather than
        always ``content[-1]``) keeps a breakpoint on its intended message even
        after parallel tool results are merged into one message.
        """
        for msg_idx, block_idx in targets:
            if not (0 <= msg_idx < len(api_messages)):
                continue  # pragma: no cover — stream pre-filters indices
            msg = api_messages[msg_idx]
            content = msg.get("content")
            if isinstance(content, list) and content:
                if 0 <= block_idx < len(content) and isinstance(
                    content[block_idx], dict
                ):
                    content[block_idx] = {
                        **content[block_idx],
                        "cache_control": cache_control,
                    }
            elif isinstance(content, str):
                msg["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": cache_control,
                    }
                ]

    @staticmethod
    def _apply_message_cache_control(
        api_messages: list[dict[str, Any]],
        cache_control: dict[str, str],
    ) -> None:
        """Apply cache_control to the last content block of the last message.

        This caches the conversation prefix so subsequent turns get cache hits
        on all prior messages.
        """
        if not api_messages:
            return

        last_msg = api_messages[-1]
        content = last_msg.get("content")
        if not content:
            return

        if isinstance(content, list) and len(content) > 0:
            last_block = content[-1]
            if isinstance(last_block, dict):
                last_block["cache_control"] = cache_control
        elif isinstance(content, str):
            # Convert bare string content to a block so we can attach cache_control
            last_msg["content"] = [
                {"type": "text", "text": content, "cache_control": cache_control}
            ]

    @staticmethod
    def _build_api_messages(
        messages: list[Message],
    ) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
        """Convert messages, coalescing a contiguous run of tool results.

        Anthropic requires every ``tool_result`` for a parallel-tool-call turn
        to sit in the single ``user`` message immediately after the assistant's
        ``tool_use`` blocks. cubepi records one ``ToolResultMessage`` per call,
        so consecutive ones are merged into one user message here.

        Returns ``(api_messages, breakpoints)`` where ``breakpoints[i]`` is the
        ``(api_index, block_index)`` of the LAST content block that source
        message ``i`` contributed. Merging several sources into one message
        gives each a distinct block offset, so a cache breakpoint that targets
        an interior tool result stays on that result's block instead of sliding
        to the end of the merge.
        """
        api_messages: list[dict[str, Any]] = []
        breakpoints: list[tuple[int, int]] = []
        prev_tool_result = False
        for msg in messages:
            converted = AnthropicProvider._convert_message(msg)
            is_tool_result = isinstance(msg, ToolResultMessage)
            if is_tool_result and prev_tool_result and api_messages:
                merged = api_messages[-1]["content"]
                merged.extend(converted["content"])
                breakpoints.append((len(api_messages) - 1, len(merged) - 1))
            else:
                api_messages.append(converted)
                breakpoints.append(
                    (len(api_messages) - 1, len(converted["content"]) - 1)
                )
            prev_tool_result = is_tool_result
        return api_messages, breakpoints

    @staticmethod
    def _strip_trailing_empty_assistants(messages: list[Message]) -> list[Message]:
        """Drop trailing ``AssistantMessage`` entries with no content blocks.

        These are the persisted markers of a turn that errored before the
        model emitted anything. Two reasons to strip before doing anything
        else with the history:

        * Anthropic treats the last assistant message as a prefill, so a
          synthesised placeholder in that position would make the model
          continue *from* the empty turn instead of regenerating from the
          preceding user prompt.
        * The cache marker policy is queried with this same message list;
          if we trimmed only inside ``_build_api_messages`` the policy
          would still point ``cache_control`` at a message we just dropped.
        """
        while (
            messages
            and isinstance(messages[-1], AssistantMessage)
            and not messages[-1].content
        ):
            messages = messages[:-1]
        return messages

    @staticmethod
    def _convert_message(msg: Message) -> dict[str, Any]:
        if isinstance(msg, UserMessage):
            user_content: list[dict[str, Any]] = []
            attributed = apply_sender_attribution(msg, msg.content)
            for user_block in attributed:
                if isinstance(user_block, TextContent):
                    user_content.append({"type": "text", "text": user_block.text})
                elif isinstance(user_block, ImageContent):
                    user_content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": user_block.media_type,
                                "data": user_block.source,
                            },
                        }
                    )
            return {"role": "user", "content": user_content}

        elif isinstance(msg, AssistantMessage):
            assistant_content: list[dict[str, Any]] = []
            for assistant_block in msg.content:
                if isinstance(assistant_block, TextContent):
                    assistant_content.append(
                        {"type": "text", "text": assistant_block.text}
                    )
                elif isinstance(assistant_block, ThinkingContent):
                    assistant_content.append(
                        {"type": "thinking", "thinking": assistant_block.thinking}
                    )
                elif isinstance(assistant_block, ToolCall):
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": assistant_block.id,
                            "name": assistant_block.name,
                            "input": assistant_block.arguments,
                        }
                    )
            if not assistant_content:
                # Anthropic rejects messages with no content blocks ("all
                # messages must have non-empty content"). This happens when a
                # prior run failed before the model emitted anything and the
                # empty AssistantMessage was persisted (stop_reason="error").
                # Synthesize a single text block so replaying the history
                # doesn't break the next request. (A trailing empty error
                # assistant is dropped earlier in `_build_api_messages` — the
                # placeholder only ever appears in non-trailing positions.)
                assistant_content.append({"type": "text", "text": "[empty response]"})
            return {"role": "assistant", "content": assistant_content}

        elif isinstance(msg, ToolResultMessage):
            tool_content = []
            for c in msg.content:
                if isinstance(c, TextContent):
                    tool_content.append({"type": "text", "text": c.text})
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": tool_content,
                        "is_error": msg.is_error,
                    }
                ],
            }

        return {"role": "user", "content": []}

    @staticmethod
    def _convert_tool(td: ToolDefinition) -> dict[str, Any]:
        return {
            "name": td.name,
            "description": td.description,
            "input_schema": td.parameters,
        }

    async def _handle_event(
        self,
        event: Any,
        partial: AssistantMessage,
        ms: MessageStream,
        model: Model,
        tool_args_buffers: dict[int, str],
    ) -> None:
        etype = getattr(event, "type", "")
        if etype == "content_block_start":
            idx = getattr(event, "index", len(partial.content))
            block = event.content_block
            if block.type == "text":
                partial.content.append(TextContent(text=""))
                await self._emit(
                    ms,
                    StreamEvent(
                        type="text_start",
                        content_index=idx,
                        partial=partial.model_copy(deep=True),
                    ),
                    model,
                )
            elif block.type == "thinking":
                partial.content.append(ThinkingContent(thinking=""))
                await self._emit(
                    ms,
                    StreamEvent(
                        type="thinking_start",
                        content_index=idx,
                        partial=partial.model_copy(deep=True),
                    ),
                    model,
                )
            elif block.type == "tool_use":
                partial.content.append(
                    ToolCall(id=block.id, name=block.name, arguments={})
                )
                await self._emit(
                    ms,
                    StreamEvent(
                        type="toolcall_start",
                        content_index=idx,
                        partial=partial.model_copy(deep=True),
                    ),
                    model,
                )
        elif etype == "content_block_delta":
            idx = getattr(event, "index", len(partial.content) - 1)
            delta = event.delta
            if hasattr(delta, "text"):
                if partial.content and isinstance(partial.content[-1], TextContent):
                    partial.content[-1] = TextContent(
                        text=partial.content[-1].text + delta.text
                    )
                await self._emit(
                    ms,
                    StreamEvent(
                        type="text_delta",
                        delta=delta.text,
                        content_index=idx,
                        partial=partial.model_copy(deep=True),
                    ),
                    model,
                )
            elif hasattr(delta, "thinking"):
                if partial.content and isinstance(partial.content[-1], ThinkingContent):
                    partial.content[-1] = ThinkingContent(
                        thinking=partial.content[-1].thinking + delta.thinking
                    )
                await self._emit(
                    ms,
                    StreamEvent(
                        type="thinking_delta",
                        delta=delta.thinking,
                        content_index=idx,
                        partial=partial.model_copy(deep=True),
                    ),
                    model,
                )
            elif hasattr(delta, "partial_json"):
                tool_args_buffers[idx] = (
                    tool_args_buffers.get(idx, "") + delta.partial_json
                )
                await self._emit(
                    ms,
                    StreamEvent(
                        type="toolcall_delta",
                        delta=delta.partial_json,
                        content_index=idx,
                        partial=partial.model_copy(deep=True),
                    ),
                    model,
                )
        elif etype == "content_block_stop":
            idx = getattr(event, "index", len(partial.content) - 1)
            if partial.content:
                last = partial.content[-1]
                if isinstance(last, TextContent):
                    await self._emit(
                        ms,
                        StreamEvent(
                            type="text_end",
                            content_index=idx,
                            partial=partial.model_copy(deep=True),
                        ),
                        model,
                    )
                elif isinstance(last, ThinkingContent):
                    await self._emit(
                        ms,
                        StreamEvent(
                            type="thinking_end",
                            content_index=idx,
                            partial=partial.model_copy(deep=True),
                        ),
                        model,
                    )
                elif isinstance(last, ToolCall):
                    # Parse the accumulated args JSON onto the block so the
                    # toolcall_end partial exposes real arguments (not the empty
                    # dict from toolcall_start). Malformed/empty JSON falls back
                    # to {} — the final message still carries authoritative args.
                    raw = tool_args_buffers.get(idx, "")
                    try:
                        parsed = json.loads(raw) if raw.strip() else {}
                    except json.JSONDecodeError:
                        parsed = {}
                    if isinstance(parsed, dict) and parsed:
                        partial.content[-1] = ToolCall(
                            id=last.id, name=last.name, arguments=parsed
                        )
                    await self._emit(
                        ms,
                        StreamEvent(
                            type="toolcall_end",
                            content_index=idx,
                            partial=partial.model_copy(deep=True),
                        ),
                        model,
                    )

    @staticmethod
    def _assemble_response(final_msg: Any) -> dict[str, Any]:
        """Assemble Anthropic's non-streaming ``messages.create()`` response
        shape from the final streaming Message object.

        Mirrors the REST API response keys. Used to populate the assembled
        ``body`` argument passed to ``subscribe_response`` listeners.
        """
        content: list[dict[str, Any]] = []
        for block in final_msg.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                content.append({"type": "text", "text": block.text})
            elif btype == "thinking":
                content.append({"type": "thinking", "thinking": block.thinking})
            elif btype == "tool_use":
                content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        usage = getattr(final_msg, "usage", None)
        usage_dict: dict[str, Any] = {}
        if usage is not None:
            usage_dict = {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0)
                or 0,
                "cache_creation_input_tokens": getattr(
                    usage, "cache_creation_input_tokens", 0
                )
                or 0,
            }

        body: dict[str, Any] = {
            "id": getattr(final_msg, "id", None),
            "type": "message",
            "role": "assistant",
            "model": getattr(final_msg, "model", None),
            "content": content,
            "stop_reason": getattr(final_msg, "stop_reason", None),
            "stop_sequence": getattr(final_msg, "stop_sequence", None),
            "usage": usage_dict,
        }
        return body

    @staticmethod
    def _convert_response(response: Any, model: Model) -> AssistantMessage:
        content: list[Any] = []
        for block in response.content:
            if block.type == "text":
                content.append(TextContent(text=block.text))
            elif block.type == "thinking":
                content.append(ThinkingContent(thinking=block.thinking))
            elif block.type == "tool_use":
                content.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input)
                )

        stop_reason_map = {
            "end_turn": "stop",
            "tool_use": "tool_use",
            "max_tokens": "length",
        }

        return AssistantMessage(
            content=content,
            stop_reason=stop_reason_map.get(response.stop_reason, response.stop_reason),
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0)
                or 0,
                cache_write_tokens=getattr(
                    response.usage, "cache_creation_input_tokens", 0
                )
                or 0,
            ),
            timestamp=time.time(),
            provider_id=model.provider_id,
            model_id=model.id,
            response_id=getattr(response, "id", None),
        )
