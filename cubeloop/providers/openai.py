from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from cubeloop.utils.json_parse import parse_streaming_json

from cubeloop.providers.capability import (
    CapabilityDescriptor,
    apply_reasoning_control,
    apply_temperature,
)
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
from cubeloop.providers.reasoning_profiles import get_capability_profile


class OpenAIProvider(BaseProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
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
        if extra_headers:
            kwargs["default_headers"] = extra_headers
        self._client: openai.AsyncOpenAI = openai.AsyncOpenAI(**kwargs)
        self._extra_body: dict[str, Any] = extra_body or {}

        # Track whether capability was explicitly passed so the OpenAI path
        # (which today injects no temperature / no max_tokens) can stay
        # behavior-identical for legacy callers. Spec §3.5.
        self._cap_active: bool = (
            capability is not None or model_capability_overrides is not None
        )
        self._capability: CapabilityDescriptor = capability or get_capability_profile(
            "openai.chat_completions"
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
        return {"type": "function", "function": {"name": choice}}

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

        api_messages: list[dict[str, Any]] = []
        if system_prompt:
            api_messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )
        api_messages.extend(self._convert_message(m) for m in messages)

        kwargs: dict[str, Any] = {
            "model": model.id,
            "messages": api_messages,
            "stream": True,
        }
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

                # Merge instance-level extra_body into the request kwargs.
                # Provider-config extra_body (e.g. {"enable_thinking": false}) is
                # applied here so callers don't need to use on_payload for simple cases.
                if self._extra_body and "extra_body" not in kwargs:
                    kwargs["extra_body"] = dict(self._extra_body)
                elif self._extra_body:
                    kwargs["extra_body"] = {**self._extra_body, **kwargs["extra_body"]}

                # Request per-stream usage so we can populate AssistantMessage.usage.
                # Only set ``include_usage`` if the caller hasn't already
                # configured it via on_payload — some OpenAI-compatible
                # backends reject ``stream_options`` entirely, so callers
                # need to be able to opt out by setting it to False (or
                # removing the key).
                so = kwargs.setdefault("stream_options", {})
                if "include_usage" not in so:
                    so["include_usage"] = True

                # Capability-driven payload mutations. Gated on _cap_active so
                # legacy callers (no capability kwarg) keep today's wire bytes
                # exactly — the OpenAI path historically does not inject
                # temperature or max_tokens. Spec §3.5.
                cap = self._resolve_capability(model.id)
                if self._cap_active or model.reasoning:
                    # Reasoning models reject a non-default temperature, so
                    # only setdefault it for models that don't reason.
                    if not model.reasoning:
                        kwargs.setdefault("temperature", model.temperature)
                    # Don't inject a default max_tokens when the caller already
                    # set the renamed target field (e.g. max_completion_tokens
                    # via on_payload).
                    if cap.max_tokens_field not in kwargs:
                        kwargs.setdefault("max_tokens", model.max_tokens)
                    apply_temperature(kwargs, cap.temperature)
                    if cap.max_tokens_field != "max_tokens" and "max_tokens" in kwargs:
                        kwargs[cap.max_tokens_field] = kwargs.pop("max_tokens")
                    apply_reasoning_control(kwargs, cap, opts.reasoning, model=model)

                # Fire request listeners AFTER all kwargs mutations so observers
                # see the final wire payload (including extra_body merges,
                # capability-driven max_tokens field rename, and stream_options
                # injection). The on_payload mutator already ran above.
                # _fire_request_listeners deep-copies so a listener cannot
                # accidentally mutate the dict that's about to be sent.
                await _fire_request_listeners(self._request_listeners, kwargs, model)

                try:
                    response = await self._client.chat.completions.create(**kwargs)
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

                current_text = ""
                tool_calls_in_progress: dict[int, dict[str, Any]] = {}
                text_started = False
                text_content_index = 0
                response_id: str | None = None
                thinking_started = False
                thinking_content_index: int | None = None
                # Captured for _assemble_response. Populated from the first chunk
                # that exposes them; later chunks shouldn't overwrite.
                response_model: str | None = None
                response_created: int | None = None
                system_fingerprint: str | None = None
                service_tier: str | None = None
                final_finish_reason: str | None = None
                final_usage: Any = None

                async for chunk in response:
                    # Usage-only chunk (stream_options.include_usage=True sends a
                    # trailing chunk with no choices and usage populated).
                    if response_model is None:
                        rm = getattr(chunk, "model", None)
                        if rm:
                            response_model = rm
                    if response_created is None:
                        rc = getattr(chunk, "created", None)
                        if rc is not None:
                            response_created = rc
                    if system_fingerprint is None:
                        sf = getattr(chunk, "system_fingerprint", None)
                        if sf:
                            system_fingerprint = sf
                    if service_tier is None:
                        st = getattr(chunk, "service_tier", None)
                        if st:
                            service_tier = st

                    if getattr(chunk, "usage", None) is not None:
                        final_usage = chunk.usage
                        u = chunk.usage
                        prompt_tokens = getattr(u, "prompt_tokens", 0) or 0
                        cached_tokens = (
                            getattr(
                                getattr(u, "prompt_tokens_details", None),
                                "cached_tokens",
                                0,
                            )
                            or 0
                        )
                        # ``input_tokens`` is the uncached prompt portion; the
                        # cached prefix is reported separately. This matches
                        # the Responses/faux providers' accounting so cost
                        # aggregation across providers doesn't double-count.
                        partial.usage = Usage(
                            input_tokens=max(prompt_tokens - cached_tokens, 0),
                            output_tokens=getattr(u, "completion_tokens", 0) or 0,
                            cache_read_tokens=cached_tokens,
                        )

                    if response_id is None and getattr(chunk, "id", None):
                        response_id = chunk.id
                        partial.response_id = response_id
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

                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue

                    # OSS reasoning extraction: priority order is
                    #   1. delta.reasoning_content (DeepSeek/Qwen/DouBao)
                    #   2. delta.reasoning         (vLLM)
                    #   3. delta.reasoning_details (MiniMax)
                    reasoning_delta: str | None = None
                    rc = getattr(delta, "reasoning_content", None)
                    if rc:
                        reasoning_delta = rc
                    else:
                        r = getattr(delta, "reasoning", None)
                        if r:
                            reasoning_delta = r
                        else:
                            rds = getattr(delta, "reasoning_details", None)
                            if rds:
                                parts: list[str] = []
                                for d in rds:
                                    if hasattr(d, "text"):
                                        text = d.text
                                    elif isinstance(d, dict):
                                        text = d.get("text")
                                    else:
                                        text = None
                                    if text:
                                        parts.append(text)
                                if parts:
                                    reasoning_delta = "".join(parts)

                    if reasoning_delta:
                        if not thinking_started:
                            partial.content.append(ThinkingContent(thinking=""))
                            thinking_content_index = len(partial.content) - 1
                            await self._emit(
                                ms,
                                StreamEvent(
                                    type="thinking_start",
                                    content_index=thinking_content_index,
                                    partial=partial.model_copy(deep=True),
                                ),
                                model,
                            )
                            thinking_started = True
                        assert thinking_content_index is not None
                        existing = partial.content[thinking_content_index].thinking  # type: ignore[union-attr]
                        partial.content[thinking_content_index] = ThinkingContent(
                            thinking=existing + reasoning_delta
                        )
                        await self._emit(
                            ms,
                            StreamEvent(
                                type="thinking_delta",
                                delta=reasoning_delta,
                                content_index=thinking_content_index,
                                partial=partial.model_copy(deep=True),
                            ),
                            model,
                        )

                    if delta.content:
                        if not text_started:
                            partial.content.append(TextContent(text=""))
                            text_content_index = len(partial.content) - 1
                            await self._emit(
                                ms,
                                StreamEvent(
                                    type="text_start",
                                    content_index=text_content_index,
                                    partial=partial.model_copy(deep=True),
                                ),
                                model,
                            )
                            text_started = True
                        current_text += delta.content
                        if partial.content and isinstance(
                            partial.content[-1], TextContent
                        ):
                            partial.content[-1] = TextContent(text=current_text)
                        await self._emit(
                            ms,
                            StreamEvent(
                                type="text_delta",
                                delta=delta.content,
                                content_index=text_content_index,
                                partial=partial.model_copy(deep=True),
                            ),
                            model,
                        )

                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_calls_in_progress:
                                if text_started:
                                    await self._emit(
                                        ms,
                                        StreamEvent(
                                            type="text_end",
                                            content_index=text_content_index,
                                            partial=partial.model_copy(deep=True),
                                        ),
                                        model,
                                    )
                                    text_started = False
                                tool_calls_in_progress[idx] = {
                                    "id": tc_delta.id or "",
                                    "name": (
                                        tc_delta.function.name
                                        if tc_delta.function
                                        else ""
                                    ),
                                    "arguments": "",
                                }
                                partial.content.append(
                                    ToolCall(
                                        id=tool_calls_in_progress[idx]["id"],
                                        name=tool_calls_in_progress[idx]["name"],
                                        arguments={},
                                    )
                                )
                                tc_content_index = len(partial.content) - 1
                                tool_calls_in_progress[idx]["content_index"] = (
                                    tc_content_index
                                )
                                await self._emit(
                                    ms,
                                    StreamEvent(
                                        type="toolcall_start",
                                        content_index=tc_content_index,
                                        partial=partial.model_copy(deep=True),
                                    ),
                                    model,
                                )
                            if tc_delta.function and tc_delta.function.arguments:
                                tool_calls_in_progress[idx]["arguments"] += (
                                    tc_delta.function.arguments
                                )
                                await self._emit(
                                    ms,
                                    StreamEvent(
                                        type="toolcall_delta",
                                        delta=tc_delta.function.arguments,
                                        content_index=tool_calls_in_progress[idx][
                                            "content_index"
                                        ],
                                        partial=partial.model_copy(deep=True),
                                    ),
                                    model,
                                )

                    finish_reason = (
                        chunk.choices[0].finish_reason if chunk.choices else None
                    )
                    # Some providers (e.g. Volcano Engine) repeat the same
                    # finish_reason in a subsequent chunk. Only process it once
                    # to avoid duplicate toolcall_end / text_end events.
                    if finish_reason and not final_finish_reason:
                        final_finish_reason = finish_reason
                        if thinking_started:
                            await self._emit(
                                ms,
                                StreamEvent(
                                    type="thinking_end",
                                    content_index=thinking_content_index,
                                    partial=partial.model_copy(deep=True),
                                ),
                                model,
                            )
                            thinking_started = False

                        if text_started:
                            await self._emit(
                                ms,
                                StreamEvent(
                                    type="text_end",
                                    content_index=text_content_index,
                                    partial=partial.model_copy(deep=True),
                                ),
                                model,
                            )

                        for idx, tc_data in tool_calls_in_progress.items():
                            args = parse_streaming_json(tc_data["arguments"])
                            for i, c in enumerate(partial.content):
                                if isinstance(c, ToolCall) and c.id == tc_data["id"]:
                                    partial.content[i] = ToolCall(
                                        id=tc_data["id"],
                                        name=tc_data["name"],
                                        arguments=args,
                                    )
                            await self._emit(
                                ms,
                                StreamEvent(
                                    type="toolcall_end",
                                    content_index=tc_data["content_index"],
                                    partial=partial.model_copy(deep=True),
                                ),
                                model,
                            )

                        stop_map = {
                            "stop": "stop",
                            "tool_calls": "tool_use",
                            "length": "length",
                        }
                        # Build final now but do NOT emit yet — OpenAI sends a
                        # trailing usage-only chunk after finish_reason when
                        # stream_options.include_usage=True. Let the loop
                        # exhaust so the usage block above captures it before
                        # we close the stream.
                        partial = partial.model_copy(
                            update={
                                "stop_reason": stop_map.get(
                                    finish_reason, finish_reason
                                ),
                            }
                        )

                if is_empty_completion(partial):
                    raise EmptyCompletion(
                        "Model returned an empty completion with no content and no tool calls",
                        provider=model.provider_id,
                        model=model.id,
                    )

                body = self._assemble_response(
                    response_id=response_id,
                    model_id=response_model or model.id,
                    created=response_created,
                    system_fingerprint=system_fingerprint,
                    service_tier=service_tier,
                    text=current_text,
                    tool_calls_in_progress=tool_calls_in_progress,
                    finish_reason=final_finish_reason,
                    usage=final_usage,
                )
                await self._emit(ms, StreamEvent(type="done"), model)
                ms.set_result(partial)

            except BaseException as e:
                exc = e
                # Classify SDK stream-level errors that the narrow create()
                # try/except didn't catch — async-for-chunk iteration errors
                # (mid-stream connection drops, timeouts) arrive here raw.
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
    def _assemble_response(
        *,
        response_id: str | None,
        model_id: str,
        created: int | None,
        system_fingerprint: str | None,
        service_tier: str | None,
        text: str,
        tool_calls_in_progress: dict[int, dict[str, Any]],
        finish_reason: str | None,
        usage: Any,
    ) -> dict[str, Any]:
        """Assemble OpenAI's non-streaming ``chat.completion`` response shape
        from accumulated streaming state.
        """
        tool_calls: list[dict[str, Any]] = []
        for idx in sorted(tool_calls_in_progress):
            tc = tool_calls_in_progress[idx]
            tool_calls.append(
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": tc.get("arguments", ""),
                    },
                }
            )

        message: dict[str, Any] = {
            "role": "assistant",
            "content": text or None,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls

        usage_dict: dict[str, Any] = {}
        if usage is not None:
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            }
            details = getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                cached = getattr(details, "cached_tokens", None)
                if cached is not None:
                    usage_dict["prompt_tokens_details"] = {"cached_tokens": cached}

        body: dict[str, Any] = {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            ],
            "usage": usage_dict,
        }
        if system_fingerprint is not None:
            body["system_fingerprint"] = system_fingerprint
        if service_tier is not None:
            body["service_tier"] = service_tier
        return body

    @staticmethod
    def _convert_message(msg: Message) -> dict[str, Any]:
        if isinstance(msg, UserMessage):
            attributed = apply_sender_attribution(msg, msg.content)
            has_image = any(isinstance(c, ImageContent) for c in attributed)
            if has_image:
                parts: list[dict[str, Any]] = []
                for c in attributed:
                    if isinstance(c, TextContent):
                        parts.append({"type": "text", "text": c.text})
                    elif isinstance(c, ImageContent):
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{c.media_type};base64,{c.source}"
                                },
                            }
                        )
                return {"role": "user", "content": parts}
            text_parts = [c.text for c in attributed if isinstance(c, TextContent)]
            return {"role": "user", "content": "\n".join(text_parts)}

        elif isinstance(msg, AssistantMessage):
            text_parts = [c.text for c in msg.content if isinstance(c, TextContent)]
            tool_calls = [c for c in msg.content if isinstance(c, ToolCall)]

            result: dict[str, Any] = {"role": "assistant"}
            result["content"] = "\n".join(text_parts) if text_parts else None
            if tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in tool_calls
                ]
            return result

        elif isinstance(msg, ToolResultMessage):
            text_parts = [c.text for c in msg.content if isinstance(c, TextContent)]
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": "\n".join(text_parts),
            }

        return {"role": "user", "content": ""}

    @staticmethod
    def _normalise_tool_schema(
        schema: Any,
        *,
        defs: dict[str, Any] | None = None,
        top: bool = True,
        strip_title: bool = True,
        in_any_of: bool = False,
    ) -> Any:
        """Normalise a Pydantic model_json_schema() output to match langchain-openai's format.

        langchain-openai strips ``title`` from property-level fields and the top-level
        schema, but preserves ``title`` inside ``anyOf`` items (e.g. enum class names
        like ``"title": "MemoryScope"`` from Optional[SomeEnum] fields). It also strips
        the top-level ``description`` (class docstring) and resolves ``$defs``/``$ref``
        so the schema is self-contained.  Matching this format is required for
        byte-stream parity with the LangGraph runtime (OpenAI-compatible auto-cache
        hashes raw bytes).
        """
        N = OpenAIProvider._normalise_tool_schema  # local alias for brevity

        if isinstance(schema, dict):
            # Collect $defs at top level so we can resolve $refs below.
            if top and "$defs" in schema:
                defs = schema["$defs"]

            # $ref resolution: inline the definition.
            # Title kept only when we're inside an anyOf (enum option);
            # stripped otherwise (e.g. TypedDict in "items").
            if "$ref" in schema:
                ref_name = schema["$ref"].split("/")[-1]
                if defs and ref_name in defs:
                    resolved = N(
                        defs[ref_name],
                        defs=defs,
                        top=False,
                        strip_title=not in_any_of,
                        in_any_of=False,
                    )
                    # JSON Schema 2020-12 allows sibling keys alongside
                    # $ref; Pydantic emits ``description`` / ``default`` /
                    # validators on Optional[Enum] fields, etc. Merge them
                    # onto the resolved definition so field-level metadata
                    # isn't silently dropped when we inline.
                    siblings = {
                        k: v
                        for k, v in schema.items()
                        if k != "$ref" and not (k == "title" and not in_any_of)
                    }
                    if siblings and isinstance(resolved, dict):
                        merged: dict[str, Any] = dict(resolved)
                        for k, v in siblings.items():
                            merged.setdefault(
                                k,
                                N(
                                    v,
                                    defs=defs,
                                    top=False,
                                    strip_title=not in_any_of,
                                    in_any_of=False,
                                ),
                            )
                        return merged
                    return resolved
                # Unknown ref — leave as-is.
                return schema

            result: dict[str, Any] = {}
            for k, v in schema.items():
                if k == "title" and strip_title:
                    continue  # Strip at property / top level; keep inside anyOf defs.
                if k == "$defs":
                    continue  # Removed after $ref resolution.
                if k == "description" and top:
                    continue  # Strip class docstring from top-level schema only.
                if k == "properties" and isinstance(v, dict):
                    # Named fields: always strip the pydantic-generated title.
                    result[k] = {
                        pname: N(
                            pschema,
                            defs=defs,
                            top=False,
                            strip_title=True,
                            in_any_of=False,
                        )
                        for pname, pschema in v.items()
                    }
                elif k == "anyOf" and isinstance(v, list):
                    # anyOf items: keep title so enum class names survive.
                    result[k] = [
                        N(item, defs=defs, top=False, strip_title=True, in_any_of=True)
                        for item in v
                    ]
                else:
                    result[k] = N(
                        v,
                        defs=defs,
                        top=False,
                        strip_title=strip_title,
                        in_any_of=False,
                    )
            return result
        elif isinstance(schema, list):
            return [
                N(
                    item,
                    defs=defs,
                    top=False,
                    strip_title=strip_title,
                    in_any_of=in_any_of,
                )
                for item in schema
            ]
        return schema

    @staticmethod
    def _convert_tool(td: ToolDefinition) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": td.name,
                "description": td.description,
                "parameters": OpenAIProvider._normalise_tool_schema(td.parameters),
            },
        }
