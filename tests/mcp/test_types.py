"""Unit tests for ``cubepi.mcp.types`` normalisation helpers.

The HTTP/stdio loader tests exercise the happy paths in aggregate. These
tests pin the edge cases of the conversion helpers directly so the
coverage gate doesn't regress when callers refactor.
"""

from __future__ import annotations

from types import SimpleNamespace

from cubeloop.mcp.types import (
    MCPIcon,
    MCPServerInfo,
    MCPToolInfo,
    icons_from_raw,
    server_info_from_init_result,
    tool_info_from_desc,
)


def test_icons_from_raw_returns_empty_for_non_iterable() -> None:
    """A ``None`` (or non-list) ``icons`` field yields an empty tuple, not a crash."""
    assert icons_from_raw(None) == ()
    assert icons_from_raw("not a list") == ()
    assert icons_from_raw({"some": "dict"}) == ()


def test_icons_from_raw_drops_entries_with_missing_src() -> None:
    """Icons missing the required ``src`` are silently skipped."""
    raws = [
        SimpleNamespace(src="", mimeType="image/svg+xml"),  # empty src
        SimpleNamespace(src=None),  # None src
        SimpleNamespace(src="https://x/icon.svg"),  # valid
    ]
    icons = icons_from_raw(raws)
    assert icons == (MCPIcon(src="https://x/icon.svg"),)


def test_icons_from_raw_preserves_theme_for_light_dark_variants() -> None:
    """MCP spec rev 2025-11-25 allows ``theme`` so clients can pick the
    variant matching the current UI theme."""
    raws = [
        SimpleNamespace(
            src="https://x/icon-light.svg", mimeType="image/svg+xml", theme="light"
        ),
        SimpleNamespace(
            src="https://x/icon-dark.svg", mimeType="image/svg+xml", theme="dark"
        ),
    ]
    icons = icons_from_raw(raws)
    assert icons == (
        MCPIcon(
            src="https://x/icon-light.svg", mime_type="image/svg+xml", theme="light"
        ),
        MCPIcon(src="https://x/icon-dark.svg", mime_type="image/svg+xml", theme="dark"),
    )


def test_icons_from_raw_accepts_snake_case_mime_type() -> None:
    """Some sources use ``mime_type`` instead of ``mimeType`` — accept both."""
    icons = icons_from_raw(
        [
            SimpleNamespace(
                src="data:image/svg+xml;base64,abc", mime_type="image/svg+xml"
            )
        ]
    )
    assert icons[0].mime_type == "image/svg+xml"


def test_server_info_returns_none_when_server_info_missing() -> None:
    """No ``serverInfo`` on the InitializeResult → ``None`` (no synthetic info)."""
    init = SimpleNamespace(serverInfo=None)
    assert server_info_from_init_result(init) is None


def test_server_info_returns_none_when_name_missing() -> None:
    """``name`` is required by the spec; without it we drop the whole entry
    rather than fabricate a placeholder."""
    init = SimpleNamespace(
        serverInfo=SimpleNamespace(name="", version="1.0", icons=None, websiteUrl=None)
    )
    assert server_info_from_init_result(init) is None


def test_server_info_captures_all_fields() -> None:
    init = SimpleNamespace(
        serverInfo=SimpleNamespace(
            name="Linear",
            version="1.4.2",
            websiteUrl="https://linear.app",
            icons=[
                SimpleNamespace(
                    src="https://linear.app/favicon.svg", mimeType="image/svg+xml"
                )
            ],
        )
    )
    info = server_info_from_init_result(init)
    assert info == MCPServerInfo(
        name="Linear",
        version="1.4.2",
        website_url="https://linear.app",
        icons=(
            MCPIcon(src="https://linear.app/favicon.svg", mime_type="image/svg+xml"),
        ),
    )


def test_server_info_reads_v2_snake_case_server_info() -> None:
    """SDK v2 InitializeResult nests the implementation under server_info."""
    init = SimpleNamespace(
        server_info=SimpleNamespace(
            name="v2-server",
            version="2.3.0",
            website_url="https://example.test",
            icons=None,
        )
    )
    info = server_info_from_init_result(init)
    assert info == MCPServerInfo(
        name="v2-server",
        version="2.3.0",
        website_url="https://example.test",
        icons=(),
    )


def test_tool_info_from_desc_handles_missing_icons() -> None:
    desc = SimpleNamespace(name="search", icons=None)
    assert tool_info_from_desc(desc) == MCPToolInfo(name="search", icons=())
