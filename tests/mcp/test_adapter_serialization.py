"""Tests for serialize_mcp_call_tool_response coverage."""

from __future__ import annotations

from types import SimpleNamespace

from cubeloop.mcp._adapter import get_mcp_field, serialize_mcp_call_tool_response


def test_serialize_mcp_call_tool_response_image_block_and_structured() -> None:
    resp = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="plain"),
            SimpleNamespace(
                type="image",
                data="b64data",
                mime_type="image/jpeg",
            ),
        ],
        is_error=False,
        structured_content={"ok": True},
    )
    serialized = serialize_mcp_call_tool_response(resp)
    assert serialized["content"] == [
        {"type": "text", "text": "plain"},
        {"type": "image", "data": "b64data", "mimeType": "image/jpeg"},
    ]
    assert serialized["isError"] is False
    assert serialized["structuredContent"] == {"ok": True}


def test_serialize_mcp_call_tool_response_camel_case_fallback() -> None:
    resp = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="image",
                data="b64data2",
                mimeType="image/png",
            ),
        ],
        isError=True,
        structuredContent={"fail": True},
    )
    serialized = serialize_mcp_call_tool_response(resp)
    assert serialized["content"] == [
        {"type": "image", "data": "b64data2", "mimeType": "image/png"},
    ]
    assert serialized["isError"] is True
    assert serialized["structuredContent"] == {"fail": True}


def test_get_mcp_field_keeps_falsy_snake_case_values() -> None:
    value = SimpleNamespace(structured_content={}, is_error=False, count=0, items=[])
    assert get_mcp_field(value, "structured_content", "structuredContent") == {}
    assert get_mcp_field(value, "is_error", "isError") is False
    assert get_mcp_field(value, "count", "count") == 0
    assert get_mcp_field(value, "items", "items") == []


def test_serialize_keeps_empty_structured_content_and_false_error() -> None:
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="")],
        is_error=False,
        structured_content={},
    )
    serialized = serialize_mcp_call_tool_response(resp)
    assert serialized["isError"] is False
    assert serialized["structuredContent"] == {}
