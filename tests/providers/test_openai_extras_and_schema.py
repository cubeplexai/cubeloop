"""Coverage for OpenAIProvider's extra_body / extra_headers wiring and
``_normalise_tool_schema`` paths. These are the diff lines that landed
in PR #67 — adding focused unit tests so codecov/patch hits the bar.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from cubeloop.providers.base import (
    Model,
    StreamOptions,
    TextContent,
    UserMessage,
)
from cubeloop.providers.openai import OpenAIProvider


def _model() -> Model:
    return Model(id="gpt-4o", provider_id="openai", api="openai")


def _make_chunk(*, finish_reason=None, text: str | None = None):
    return SimpleNamespace(
        id="x",
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=text, tool_calls=None),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


async def _async_iter(items):
    for it in items:
        yield it


# ---------------------------------------------------------------------------
# extra_headers — constructor branch
# ---------------------------------------------------------------------------


def test_extra_headers_forwarded_to_async_openai() -> None:
    with patch("openai.AsyncOpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        OpenAIProvider(
            api_key="x",
            base_url="https://example/v1",
            extra_headers={"X-Custom": "y"},
        )
        kwargs = mock_openai.call_args.kwargs
        assert kwargs["default_headers"] == {"X-Custom": "y"}


def test_no_extra_headers_omits_default_headers() -> None:
    with patch("openai.AsyncOpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        OpenAIProvider(api_key="x")
        assert "default_headers" not in mock_openai.call_args.kwargs


# ---------------------------------------------------------------------------
# extra_body — merge into request kwargs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extra_body_added_when_kwargs_missing_it() -> None:
    captured: dict[str, Any] = {}
    with patch("openai.AsyncOpenAI") as mock_openai:
        client = MagicMock()
        mock_openai.return_value = client
        client.chat = MagicMock()
        client.chat.completions = MagicMock()

        async def capture(**kwargs):
            captured.update(kwargs)
            return _async_iter([_make_chunk(finish_reason="stop")])

        client.chat.completions.create = capture

        provider = OpenAIProvider(api_key="x", extra_body={"enable_thinking": False})
        provider._client = client

        ms = await provider.stream(
            _model(),
            [UserMessage(content=[TextContent(text="hi")])],
        )
        async for _ in ms:
            pass
        await ms.result()

    assert captured["extra_body"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_extra_body_merged_when_payload_hook_supplied_one() -> None:
    captured: dict[str, Any] = {}

    async def on_payload(payload, model):
        payload["extra_body"] = {"a": 1, "enable_thinking": True}  # collision on key
        return payload

    with patch("openai.AsyncOpenAI") as mock_openai:
        client = MagicMock()
        mock_openai.return_value = client
        client.chat = MagicMock()
        client.chat.completions = MagicMock()

        async def capture(**kwargs):
            captured.update(kwargs)
            return _async_iter([_make_chunk(finish_reason="stop")])

        client.chat.completions.create = capture

        provider = OpenAIProvider(api_key="x", extra_body={"enable_thinking": False})
        provider._client = client

        ms = await provider.stream(
            _model(),
            [UserMessage(content=[TextContent(text="hi")])],
            options=StreamOptions(on_payload=on_payload),
        )
        async for _ in ms:
            pass
        await ms.result()

    # Instance-level extra_body is the base; on_payload overrides on key collision.
    assert captured["extra_body"] == {"a": 1, "enable_thinking": True}


# ---------------------------------------------------------------------------
# Reasoning details fallback — delta entry that's neither attr nor dict
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reasoning_details_unknown_entry_type_yields_no_text() -> None:
    """A reasoning_details entry that's neither attr-bearing nor a dict
    should hit the ``text = None`` branch and produce no thinking_delta.
    """
    delta = SimpleNamespace(
        content=None,
        tool_calls=None,
        reasoning_content=None,
        reasoning=None,
        reasoning_details=[42],  # not a dict, no .text attr
    )
    chunk = SimpleNamespace(
        id="x",
        choices=[SimpleNamespace(delta=delta, finish_reason=None)],
        usage=None,
    )
    final = _make_chunk(finish_reason="stop")

    saw_thinking_delta = False
    with patch("openai.AsyncOpenAI") as mock_openai:
        client = MagicMock()
        mock_openai.return_value = client
        client.chat = MagicMock()
        client.chat.completions = MagicMock()

        async def create(**_):
            return _async_iter([chunk, final])

        client.chat.completions.create = create

        provider = OpenAIProvider(api_key="x")
        provider._client = client

        ms = await provider.stream(
            _model(),
            [UserMessage(content=[TextContent(text="hi")])],
        )
        async for evt in ms:
            if evt.type == "thinking_delta":
                saw_thinking_delta = True
        await ms.result()

    assert saw_thinking_delta is False


# ---------------------------------------------------------------------------
# _normalise_tool_schema — coverage for $defs / $ref / anyOf paths
# ---------------------------------------------------------------------------


def test_normalise_strips_top_level_title_description_and_defs() -> None:
    schema = {
        "title": "Foo",
        "description": "docstring",
        "type": "object",
        "$defs": {"Bar": {"title": "Bar", "type": "string"}},
        "properties": {"x": {"title": "X", "type": "integer"}},
    }
    out = OpenAIProvider._normalise_tool_schema(schema)
    assert "title" not in out
    assert "description" not in out
    assert "$defs" not in out
    assert "title" not in out["properties"]["x"]


def test_normalise_keeps_title_inside_anyof_via_ref() -> None:
    """anyOf items whose $ref resolves into a $def should keep title
    (enum class name needs to survive for cache parity)."""
    schema = {
        "title": "Outer",
        "type": "object",
        "$defs": {"Scope": {"title": "Scope", "enum": ["a", "b"], "type": "string"}},
        "properties": {
            "scope": {"anyOf": [{"$ref": "#/$defs/Scope"}, {"type": "null"}]},
        },
    }
    out = OpenAIProvider._normalise_tool_schema(schema)
    scope_options = out["properties"]["scope"]["anyOf"]
    # The Scope variant should keep its title; the null variant has none.
    titled = [o for o in scope_options if "title" in o]
    assert titled and titled[0]["title"] == "Scope"


def test_normalise_unknown_ref_left_unchanged() -> None:
    """A $ref that doesn't resolve to a $def should be passed through."""
    schema = {"$ref": "#/$defs/MissingName"}
    out = OpenAIProvider._normalise_tool_schema(schema)
    assert out == schema


def test_normalise_passes_through_non_dict_non_list() -> None:
    assert OpenAIProvider._normalise_tool_schema("plain") == "plain"
    assert OpenAIProvider._normalise_tool_schema(7) == 7


# ---------------------------------------------------------------------------
# end-to-end smoke — schema feeding through pydantic model
# ---------------------------------------------------------------------------


class _ToolParams(BaseModel):
    """A doc-string that should NOT make it into the wire schema."""

    name: str
    count: int = 1


def test_normalise_works_on_pydantic_generated_schema() -> None:
    raw = _ToolParams.model_json_schema()
    out = OpenAIProvider._normalise_tool_schema(raw)
    # Top-level title + description stripped.
    assert "title" not in out
    assert "description" not in out
    # Property titles stripped.
    for prop in out["properties"].values():
        assert "title" not in prop


# ---------------------------------------------------------------------------
# Codex review #5 — input_tokens excludes cached_tokens (cache-hit accounting)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_usage_subtracts_cached_tokens_from_input() -> None:
    """``input_tokens`` is the uncached portion; cached prefix reported
    separately, matching Responses/faux accounting (avoids double-count
    in cubebox cost aggregation)."""
    usage_chunk = SimpleNamespace(
        id="x",
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=50,
            prompt_tokens_details=SimpleNamespace(cached_tokens=800),
        ),
    )
    content_chunk = _make_chunk(text="ok")
    final = _make_chunk(finish_reason="stop")

    with patch("openai.AsyncOpenAI") as mock_openai:
        client = MagicMock()
        mock_openai.return_value = client
        client.chat = MagicMock()
        client.chat.completions = MagicMock()

        async def create(**_):
            return _async_iter([content_chunk, usage_chunk, final])

        client.chat.completions.create = create

        provider = OpenAIProvider(api_key="x")
        provider._client = client

        ms = await provider.stream(
            _model(),
            [UserMessage(content=[TextContent(text="hi")])],
        )
        async for _ in ms:
            pass
        msg = await ms.result()

    assert msg.usage.input_tokens == 200  # 1000 - 800
    assert msg.usage.cache_read_tokens == 800
    assert msg.usage.output_tokens == 50


@pytest.mark.asyncio
async def test_usage_no_cached_field_input_tokens_is_full_prompt() -> None:
    """No prompt_tokens_details → cached_tokens=0, input_tokens=prompt_tokens."""
    usage_chunk = SimpleNamespace(
        id="x",
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=200,
            completion_tokens=10,
            prompt_tokens_details=None,
        ),
    )
    content_chunk = _make_chunk(text="ok")
    final = _make_chunk(finish_reason="stop")

    with patch("openai.AsyncOpenAI") as mock_openai:
        client = MagicMock()
        mock_openai.return_value = client
        client.chat = MagicMock()
        client.chat.completions = MagicMock()

        async def create(**_):
            return _async_iter([content_chunk, usage_chunk, final])

        client.chat.completions.create = create

        provider = OpenAIProvider(api_key="x")
        provider._client = client

        ms = await provider.stream(
            _model(),
            [UserMessage(content=[TextContent(text="hi")])],
        )
        async for _ in ms:
            pass
        msg = await ms.result()

    assert msg.usage.input_tokens == 200
    assert msg.usage.cache_read_tokens == 0


# ---------------------------------------------------------------------------
# Codex review #6 — on_payload may opt out of stream_options.include_usage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_payload_can_disable_include_usage() -> None:
    """Caller-supplied ``stream_options.include_usage=False`` must survive
    the default-inject step."""
    captured: dict[str, Any] = {}

    async def on_payload(payload, model):
        payload["stream_options"] = {"include_usage": False}
        return payload

    with patch("openai.AsyncOpenAI") as mock_openai:
        client = MagicMock()
        mock_openai.return_value = client
        client.chat = MagicMock()
        client.chat.completions = MagicMock()

        async def capture(**kwargs):
            captured.update(kwargs)
            return _async_iter([_make_chunk(finish_reason="stop")])

        client.chat.completions.create = capture

        provider = OpenAIProvider(api_key="x")
        provider._client = client

        ms = await provider.stream(
            _model(),
            [UserMessage(content=[TextContent(text="hi")])],
            options=StreamOptions(on_payload=on_payload),
        )
        async for _ in ms:
            pass
        await ms.result()

    assert captured["stream_options"] == {"include_usage": False}


@pytest.mark.asyncio
async def test_default_adds_include_usage_when_not_set() -> None:
    """Without on_payload customization, default adds include_usage=True."""
    captured: dict[str, Any] = {}
    with patch("openai.AsyncOpenAI") as mock_openai:
        client = MagicMock()
        mock_openai.return_value = client
        client.chat = MagicMock()
        client.chat.completions = MagicMock()

        async def capture(**kwargs):
            captured.update(kwargs)
            return _async_iter([_make_chunk(finish_reason="stop")])

        client.chat.completions.create = capture

        provider = OpenAIProvider(api_key="x")
        provider._client = client

        ms = await provider.stream(
            _model(),
            [UserMessage(content=[TextContent(text="hi")])],
        )
        async for _ in ms:
            pass
        await ms.result()

    assert captured["stream_options"] == {"include_usage": True}


# ---------------------------------------------------------------------------
# Codex review #7 — preserve sibling metadata when $ref-resolving
# ---------------------------------------------------------------------------


def test_normalise_ref_resolution_preserves_sibling_description() -> None:
    """Pydantic emits ``$ref`` alongside ``description`` / ``default`` on
    Optional[Enum] fields; both must survive inlining."""
    schema = {
        "$defs": {"Color": {"title": "Color", "enum": ["r", "g"], "type": "string"}},
        "properties": {
            "c": {
                "$ref": "#/$defs/Color",
                "description": "primary colour",
                "default": "r",
            }
        },
    }
    out = OpenAIProvider._normalise_tool_schema(schema)
    resolved = out["properties"]["c"]
    assert resolved["enum"] == ["r", "g"]
    assert resolved["description"] == "primary colour"
    assert resolved["default"] == "r"


def test_normalise_ref_resolution_resolved_def_wins_on_key_collision() -> None:
    """If both $def and sibling supply the same key, the $def value wins
    (sibling is treated as additional metadata, not an override)."""
    schema = {
        "$defs": {"X": {"type": "string", "description": "from def"}},
        "properties": {
            "x": {"$ref": "#/$defs/X", "description": "from sibling"},
        },
    }
    out = OpenAIProvider._normalise_tool_schema(schema)
    # setdefault means the def's value sticks.
    assert out["properties"]["x"]["description"] == "from def"
