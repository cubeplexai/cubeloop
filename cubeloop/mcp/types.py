"""MCP discovery result types (spec rev 2025-11-25 icons + serverInfo)."""

from __future__ import annotations

from dataclasses import dataclass, field

from cubeloop.agent.types import AgentTool


@dataclass(frozen=True)
class MCPIcon:
    """One icon entry from an MCP ``Icon`` block.

    Mirrors ``mcp.types.Icon`` with snake_case fields. ``src`` is either an
    HTTP/HTTPS URL or a ``data:`` URI; ``sizes`` is a tuple of strings such
    as ``("48x48", "96x96")``. ``theme`` is ``"light"`` or ``"dark"`` when
    the server supplies separate variants — the spec lets clients pick the
    variant matching the current UI theme and fall back to the first entry
    otherwise.
    """

    src: str
    mime_type: str | None = None
    sizes: tuple[str, ...] | None = None
    theme: str | None = None


@dataclass(frozen=True)
class MCPServerInfo:
    """Server-level metadata captured from the ``initialize`` handshake.

    Sourced from ``InitializeResult.serverInfo`` (an ``Implementation``):
    the server's display name + version, an optional website URL, and an
    optional list of icons that clients can render as the server's logo.
    """

    name: str
    version: str
    website_url: str | None = None
    icons: tuple[MCPIcon, ...] = ()


@dataclass(frozen=True)
class MCPToolInfo:
    """Per-tool display metadata captured from ``tools/list``.

    Carries ``Tool.icons`` separately from the executable ``AgentTool`` so
    consumers can render tool-specific icons without coupling cubepi's
    core ``AgentTool`` type to MCP display concerns.
    """

    name: str
    icons: tuple[MCPIcon, ...] = ()


@dataclass(frozen=True)
class MCPDiscoveryResult:
    """Outcome of an MCP discovery handshake.

    ``tools`` is the executable surface (one ``AgentTool`` per tool the
    server exposes). ``server`` and ``tool_infos`` are display metadata
    captured from the same handshake; both follow MCP spec rev 2025-11-25.
    """

    tools: list[AgentTool]
    server: MCPServerInfo | None = None
    tool_infos: list[MCPToolInfo] = field(default_factory=list)


def _icon_from_raw(raw: object) -> MCPIcon | None:
    """Normalise an ``mcp.types.Icon`` (or duck-type) into ``MCPIcon``.

    Returns ``None`` when the raw entry is missing the required ``src``.
    Accepts attribute-bearing objects (Pydantic models, SimpleNamespace);
    callers are expected to pass entries from a parsed MCP response.
    """
    src = getattr(raw, "src", None)
    if not src:
        return None
    mime = getattr(raw, "mimeType", None) or getattr(raw, "mime_type", None)
    sizes_raw = getattr(raw, "sizes", None)
    sizes = tuple(sizes_raw) if sizes_raw else None
    theme = getattr(raw, "theme", None)
    return MCPIcon(src=src, mime_type=mime, sizes=sizes, theme=theme)


def icons_from_raw(raws: object) -> tuple[MCPIcon, ...]:
    """Build a tuple of ``MCPIcon`` from a raw ``icons`` field.

    Tolerates ``None``, an empty list, or a list of icon-shaped objects.
    Entries missing ``src`` are silently dropped.
    """
    if not isinstance(raws, list | tuple):
        return ()
    out: list[MCPIcon] = []
    for raw in raws:
        icon = _icon_from_raw(raw)
        if icon is not None:
            out.append(icon)
    return tuple(out)


def server_info_from_result(server: object | None) -> MCPServerInfo | None:
    """Extract ``MCPServerInfo`` from a v2 client ``server_info`` result.

    MCP Python SDK v2 exposes the server implementation directly and uses
    snake_case attributes, while a legacy handshake result wraps it in
    ``serverInfo``.  Supporting both shapes keeps this value useful when a
    client negotiates down to the pre-2026 protocol.
    """
    if server is None:
        return None
    # v2 Client.server_info is the Implementation itself (name/version).
    # A legacy InitializeResult wraps that object as server_info / serverInfo.
    # Descend only when the wrapper declares the field; an Implementation does
    # not, and getattr's default would otherwise hand the object back to itself.
    declared = set(getattr(server, "__dict__", {}))
    declared.update(getattr(type(server), "model_fields", {}))
    for field_name in ("server_info", "serverInfo"):
        if field_name not in declared:
            continue
        nested = getattr(server, field_name, None)
        if nested is not None:
            server = nested
        break
    name = getattr(server, "name", None)
    if not name:
        return None
    return MCPServerInfo(
        name=name,
        version=getattr(server, "version", "") or "",
        website_url=getattr(server, "website_url", None)
        or getattr(server, "websiteUrl", None),
        icons=icons_from_raw(getattr(server, "icons", None)),
    )


def server_info_from_init_result(init_result: object) -> MCPServerInfo | None:
    """Backward-compatible alias for legacy callers/tests."""
    return server_info_from_result(init_result)


def tool_info_from_desc(desc: object) -> MCPToolInfo:
    """Build a ``MCPToolInfo`` from a single ``tools/list`` entry."""
    return MCPToolInfo(
        name=getattr(desc, "name", "") or "",
        icons=icons_from_raw(getattr(desc, "icons", None)),
    )
