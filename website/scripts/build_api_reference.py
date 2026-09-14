"""Generate Docusaurus-compatible MDX from cubeloop public API via griffe.

Pipeline:

1. Load ``cubepi`` and its submodules with griffe.
2. Collect each module's public symbols (classes, functions, attributes
   listed in ``__all__`` or otherwise non-underscored).
3. Build a cross-reference index (``Name`` / ``Module.Name`` /
   ``Class.member`` → ``./cubepi-module#anchor``) so RST refs in
   docstrings become real markdown links inside MDX.
4. Render each module to MDX:
   - Module-level classes: signature + docstring + member listing
     (public methods get their own ``####`` subsection; public
     attributes / properties become a single ``Attributes`` bullet
     list with type annotations and first-line docstrings).
   - Module-level functions: signature + docstring.
   - Module-level attributes / constants: ``name: type = value`` line
     with full docstring (not just an empty ``_attribute_`` tag like
     the previous version).
5. Docstring rendering parses Google-style sections (``Args:``,
   ``Returns:``, ``Raises:``, ``Yields:``, ``Example:``, ``Note:``,
   ``Usage:``). Pre-pass strips RST refs (``:class:`Foo``` →
   ``[`Foo`](./cubepi-mod#foo)``) so we don't need to convert source
   docstrings to pure Google style up-front.

Private members (``_underscore`` names) are skipped entirely.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import griffe
from griffe import AliasResolutionError

MODULES = [
    ("cubeloop.agent",        "Agents",        1),
    ("cubeloop.session",      "Execution",     2),
    ("cubeloop.providers",    "Providers",     3),
    ("cubeloop.checkpointer", "Checkpointing", 4),
    ("cubeloop.middleware",   "Middleware",    5),
    ("cubeloop.mcp",          "MCP",           6),
    ("cubeloop.tracing",      "Tracing",       7),
    ("cubeloop.utils",        "Utils",         8),
]

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Public-surface detection
# ---------------------------------------------------------------------------


def _is_public(name: str, parent_all: list[str] | None) -> bool:
    if parent_all is not None:
        return name in parent_all
    return not name.startswith("_")


def _is_private(name: str) -> bool:
    """True for names skipped from the API reference entirely.

    ``__init__`` is rendered as the class constructor signature, not as
    a member — caller decides what to do with it. Dunder methods other
    than ``__init__`` are skipped (renaming the section to "Dunders"
    would add visual noise; users who need them can read the source).
    """
    return name.startswith("_")


def collect_public_symbols(module) -> list:
    """Return resolved Class/Function/Attribute objects for the
    module's public surface.

    Walks ``module.members`` first (eagerly imported symbols), then
    fills in any names in ``__all__`` that griffe missed — typically
    lazy exports surfaced via ``__getattr__``. The lazy ones are
    resolved at runtime via importlib to find their canonical dotted
    path, then loaded back through griffe so the rest of the pipeline
    (signatures, docstrings, source links) works identically.
    """
    parent_all: list[str] | None = None
    if hasattr(module, "exports") and module.exports is not None:
        try:
            parent_all = list(module.exports)
        except TypeError:
            parent_all = None

    out = []
    seen: set[str] = set()
    for member_name, member in module.members.items():
        if not _is_public(member_name, parent_all):
            continue
        if getattr(member, "is_alias", False):
            try:
                member = member.final_target
            except AliasResolutionError:
                continue
        out.append(member)
        seen.add(member_name)

    if parent_all:
        missing = [n for n in parent_all if n not in seen]
        if missing:
            try:
                runtime_mod = importlib.import_module(module.path)
            except Exception as e:
                print(
                    f"[warn] cannot import {module.path} to resolve lazy exports: {e}",
                    file=sys.stderr,
                )
                runtime_mod = None
            if runtime_mod is not None:
                for name in missing:
                    try:
                        obj = getattr(runtime_mod, name)
                    except (AttributeError, ImportError) as e:
                        print(
                            f"[warn] {module.path}.{name} unresolvable ({e}); skipping",
                            file=sys.stderr,
                        )
                        continue
                    real_module = getattr(obj, "__module__", None)
                    qualname = getattr(obj, "__qualname__", None) or name
                    if not real_module:
                        continue
                    try:
                        target = griffe.load(f"{real_module}.{qualname}")
                    except Exception as e:
                        print(
                            f"[warn] griffe.load({real_module}.{qualname}) failed: {e}",
                            file=sys.stderr,
                        )
                        continue
                    out.append(target)
    return out


# ---------------------------------------------------------------------------
# Cross-reference index
# ---------------------------------------------------------------------------


def _anchor(name: str) -> str:
    """Docusaurus auto-anchor convention for ``### Heading``: lowercase,
    underscores → hyphens, non-alphanumeric stripped, leading/trailing
    hyphens collapsed (so ``__init__`` becomes ``init`` rather than
    ``--init--``). Matches GitHub flavored markdown anchors."""
    s = name.lower().replace("_", "-")
    s = re.sub(r"[^a-z0-9\-]", "", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def _source_relpath(filepath: str | Path) -> str:
    """Return a GitHub-linkable source path relative to the repo root."""
    path = Path(str(filepath))
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        parts = path.as_posix().split("/")
        for index in range(len(parts) - 1, -1, -1):
            if parts[index] == "cubeloop":
                return "/".join(parts[index:])
        return path.as_posix()


def build_ref_index(modules_to_symbols: dict[str, list]) -> dict[str, str]:
    """Build ``Name`` → ``./cubepi-module#anchor`` map for RST refs.

    For each module's public symbols, record:

    - bare name → page anchor (``Tracer`` → ``./cubepi-tracing#tracer``)
    - fully qualified ``module.Name`` → same anchor
    - ``Class.member`` → method/attribute anchor on the class page
      (``Tracer.attach`` → ``./cubepi-tracing#tracer-attach``)
    """
    index: dict[str, str] = {}
    for module_name, symbols in modules_to_symbols.items():
        page_id = module_name.replace(".", "-")
        for sym in symbols:
            base_anchor = _anchor(sym.name)
            url = f"./{page_id}#{base_anchor}"
            index.setdefault(sym.name, url)
            index.setdefault(f"{module_name}.{sym.name}", url)
            # Class members get their own anchors prefixed with the class name.
            if hasattr(sym, "members"):
                for mname in sym.members:
                    if _is_private(mname):
                        continue
                    m_anchor = f"{base_anchor}-{_anchor(mname)}"
                    m_url = f"./{page_id}#{m_anchor}"
                    index.setdefault(f"{sym.name}.{mname}", m_url)
                    index.setdefault(f"{module_name}.{sym.name}.{mname}", m_url)
    return index


# ---------------------------------------------------------------------------
# Docstring rendering
# ---------------------------------------------------------------------------


# Common RST cross-reference shapes:
#   :class:`Foo`         → render as link to Foo if known
#   :class:`~mod.Foo`    → render as link, show only "Foo"
#   :class:`mod.Foo`     → render as link, show "mod.Foo"
_RST_REF_PATTERN = re.compile(
    r":(?:class|func|meth|attr|mod|obj|exc|data):`(?P<short>~?)(?P<ref>[\w\.]+)`"
)


def _rst_to_mdx_refs(text: str, ref_index: dict[str, str]) -> str:
    """Replace RST cross-refs (``:class:`Foo```) with markdown links to
    our generated pages, or with inline code when the target isn't in
    our public surface (external types, private helpers).

    The leading ``~`` modifier means "show only the last component of
    the dotted path" — common idiom for refs like
    ``:class:`~cubepi.tracing.Tracer``` rendering as just ``Tracer``.
    """

    def _replace(m: re.Match) -> str:
        short_mod = bool(m.group("short"))
        ref = m.group("ref")
        last = ref.split(".")[-1]
        # Try the most specific lookup first (full qualified name),
        # fall back to the bare last component.
        link = ref_index.get(ref) or ref_index.get(last)
        display = last if short_mod else ref
        if link:
            return f"[`{display}`]({link})"
        return f"`{display}`"

    return _RST_REF_PATTERN.sub(_replace, text)


_RST_DOUBLE_BACKTICK = re.compile(r"``([^`\n]+?)``")


def _strip_rst_role_inline_code(text: str) -> str:
    r"""RST double-backtick ``\`\`foo\`\``` denotes inline literal text.
    Markdown uses single backticks for the same meaning; collapse so
    the rendered output isn't a literal ``\`\`foo\`\``` with visible
    extra backticks.
    """
    return _RST_DOUBLE_BACKTICK.sub(r"`\1`", text)


_GOOGLE_SECTIONS = (
    "Args", "Arguments",
    "Returns", "Returns",
    "Raises",
    "Yields",
    "Example", "Examples",
    "Note", "Notes",
    "Usage",
    "Attributes",
    "See Also",
)
_CODE_SECTIONS = {"Example", "Examples", "Usage"}


def _escape_mdx_braces(text: str) -> str:
    """Escape ``{`` and ``}`` so MDX does not try to parse them as JSX.

    MDX treats unescaped curly braces in prose as inline JS expressions
    and passes them to acorn; Python dict literals in docstrings (e.g.
    ``{"k": "v"}``) are NOT valid expressions and crash the build. We
    escape every ``{`` / ``}`` that falls outside fenced code blocks
    (those are handled by the caller).
    """
    return text.replace("{", r"\{").replace("}", r"\}")


def render_docstring(text: str | None, ref_index: dict[str, str]) -> str:
    """Convert a docstring (RST refs + Google-style sections + prose)
    to MDX. Cross-refs are resolved against ``ref_index`` first, then
    Google sections are parsed line-by-line."""
    if not text:
        return ""
    # Resolve cross-references before any line-by-line processing — the
    # transformer is regex-safe across line boundaries and produces
    # markdown links that survive the per-line walk below.
    text = _rst_to_mdx_refs(text, ref_index)
    text = _strip_rst_role_inline_code(text)
    # RST underline-style headings (``Example\n-------``) render as
    # spurious markdown h2s that disrupt our heading hierarchy (we're
    # already inside an h3 ``### Tracer`` section). Convert to bold
    # text so they look like a sub-callout, matching how Google-style
    # sections render via ``**Args**`` etc.
    text = re.sub(
        r"^([A-Za-z][^\n]{0,80})\n[-=~^]{3,}\s*$",
        lambda m: f"**{m.group(1).strip()}**",
        text,
        flags=re.MULTILINE,
    )

    lines = text.strip("\n").splitlines()
    out: list[str] = []
    in_section: str | None = None
    code_buffer: list[str] | None = None

    def _flush_code() -> None:
        nonlocal code_buffer
        if code_buffer is None:
            return
        if any(s.strip() for s in code_buffer):
            # Trim leading/trailing blank lines so the fenced block is
            # tight — the original RST often left blank lines after
            # the ``::`` marker that bloated the rendered block.
            while code_buffer and not code_buffer[0].strip():
                code_buffer.pop(0)
            while code_buffer and not code_buffer[-1].strip():
                code_buffer.pop()
            # Ensure a blank line above the fence so the previous
            # paragraph (often a ``**Example**`` callout) doesn't run
            # straight into the code.
            if out and out[-1] != "":
                out.append("")
            out.append("```python")
            out.extend(code_buffer)
            out.append("```")
            out.append("")
        code_buffer = None

    for line in lines:
        stripped = line.strip()

        # RST literal-block marker: ``::`` either standalone on its
        # own line OR as the trailing token of a prose paragraph
        # ("Use it like this::") signals that the next indented block
        # is code. Standalone form: just open the buffer. Trailing
        # form: emit the line without the ``::`` then open the buffer.
        if stripped == "::":
            _flush_code()
            in_section = "Example"
            code_buffer = []
            continue
        if stripped.endswith("::") and not stripped.endswith(":::"):
            # Strip the trailing ``::`` from prose, leave a single
            # colon so the reader still sees the lead-in punctuation
            # ("Use it like this:").
            _flush_code()
            prose = line[: line.rfind("::")] + ":"
            out.append(_escape_mdx_braces(prose))
            in_section = "Example"
            code_buffer = []
            continue

        # Match both Google-style ("Example:") and RST-style
        # ("Example::") section markers. RST's ``::`` syntax denotes a
        # literal/code block follows; treat the same as Google's
        # ``Example:`` heading followed by an indented code block.
        m = re.match(r"^([A-Z][a-zA-Z]+):{1,2}\s*$", stripped)
        if m and m.group(1) in _GOOGLE_SECTIONS:
            _flush_code()
            in_section = m.group(1)
            out.append("")
            out.append(f"**{in_section}**")
            out.append("")
            if in_section in _CODE_SECTIONS:
                code_buffer = []
            continue
        if in_section in _CODE_SECTIONS:
            if line.strip() == "" or line.startswith(("    ", "\t")):
                if code_buffer is None:
                    code_buffer = []
                if line.startswith("    "):
                    code_buffer.append(line[4:])
                elif line.startswith("\t"):
                    code_buffer.append(line[1:])
                else:
                    code_buffer.append(line)
                continue
            _flush_code()
            in_section = None
        if in_section in {"Args", "Arguments"} and line.startswith(("    ", "\t")):
            stripped = line.strip()
            arg_m = re.match(r"^(\w+)\s*(\([^)]+\))?:\s*(.*)$", stripped)
            if arg_m:
                arg, _ty, desc = arg_m.groups()
                out.append(f"- `{arg}` — {_escape_mdx_braces(desc)}")
                continue
        if in_section in {"Returns", "Yields"} and line.startswith(("    ", "\t")):
            out.append(f"- {_escape_mdx_braces(line.strip())}")
            continue
        if in_section == "Raises" and line.startswith(("    ", "\t")):
            stripped = line.strip()
            raise_m = re.match(r"^(\w+):\s*(.*)$", stripped)
            if raise_m:
                exc, desc = raise_m.groups()
                out.append(f"- `{exc}` — {_escape_mdx_braces(desc)}")
                continue
        out.append(_escape_mdx_braces(line))
    _flush_code()
    return "\n".join(out).strip() + "\n"


# ---------------------------------------------------------------------------
# Signature rendering
# ---------------------------------------------------------------------------


def _params_of(symbol, skip_self: bool = False) -> list[tuple]:
    """Extract ``[(name, type, default, kind), …]`` tuples from a
    griffe function symbol.

    When rendering class methods, ``skip_self=True`` drops the
    leading ``self`` parameter from the displayed signature — the
    user already knows it's a method.
    """
    out = []
    params = getattr(symbol, "parameters", None) or []
    for p in params:
        pname = p.name
        if skip_self and pname == "self":
            continue
        ptype = str(p.annotation) if getattr(p, "annotation", None) else None
        default = getattr(p, "default", None)
        pdefault = str(default) if default is not None else None
        kind_attr = getattr(p, "kind", None)
        kind = (
            getattr(kind_attr, "name", None)
            or (str(kind_attr) if kind_attr else None)
        )
        out.append((pname, ptype, pdefault, kind))
    return out


def render_signature(name: str, parameters: list, returns: str | None) -> str:
    """Render a Python signature with positional/keyword-only markers."""
    parts: list[str] = []
    seen_var_positional = False
    seen_keyword_only_marker = False
    seen_positional_only = False
    pending_positional_only_marker = False
    for p in parameters:
        kind = p[3] if len(p) >= 4 else None
        pname, ptype, pdefault = p[0], p[1], p[2]

        if kind == "positional_only":
            seen_positional_only = True
        elif seen_positional_only and not pending_positional_only_marker:
            parts.append("/")
            pending_positional_only_marker = True

        if kind == "var_positional":
            rendered = f"*{pname}"
            seen_var_positional = True
        elif kind == "var_keyword":
            rendered = f"**{pname}"
        elif kind == "keyword_only":
            if not seen_var_positional and not seen_keyword_only_marker:
                parts.append("*")
                seen_keyword_only_marker = True
            rendered = pname
        else:
            rendered = pname

        if ptype:
            rendered += f": {ptype}"
        if pdefault:
            rendered += f" = {pdefault}"
        parts.append(rendered)

    ret_suffix = f" -> {returns}" if returns else ""

    # Single line when it comfortably fits; otherwise fall back to a
    # black-style multiline layout (one parameter per indented line) so
    # wide constructors like Agent(...) stay readable instead of
    # overflowing as one giant line in the rendered code block.
    single = f"{name}({', '.join(parts)}){ret_suffix}"
    if len(single) <= 88 or not parts:
        sig = single
    else:
        body = ",\n".join(f"    {part}" for part in parts)
        sig = f"{name}(\n{body},\n){ret_suffix}"

    return f"```python\n{sig}\n```"


# ---------------------------------------------------------------------------
# Class member listing
# ---------------------------------------------------------------------------


def _is_property(member) -> bool:
    """Return True when a griffe function symbol is actually a
    ``@property`` decorator-tagged method.

    griffe doesn't have a distinct kind for properties (they're both
    ``function``); the only signal is the decorator list. ``@property``
    or ``@cached_property`` both qualify.
    """
    decorators = getattr(member, "decorators", []) or []
    for d in decorators:
        dval = str(getattr(d, "value", d))
        if dval.endswith("property") or "cached_property" in dval:
            return True
    return False


def _first_paragraph(doc: str | None) -> str:
    """Return the first paragraph of a docstring, or empty string."""
    if not doc:
        return ""
    parts = doc.strip().split("\n\n", 1)
    return parts[0].replace("\n", " ").strip()


def render_class_members(
    klass, ref_index: dict[str, str], github_blob_root: str
) -> list[str]:
    """Render a class's public methods + properties + attributes.

    Outputs three sections in order: Attributes (data + property),
    Methods. Each method gets its own ``####`` subsection with a
    code-block signature and the full docstring. Attributes /
    properties are condensed to a bullet list — they're typically
    short. ``__init__`` is intentionally NOT listed; it's already
    rendered as the class's own signature by the caller.
    """
    lines: list[str] = []

    attributes: list[tuple[str, object]] = []
    properties: list[tuple[str, object]] = []
    methods: list[tuple[str, object]] = []

    for name, member in klass.members.items():
        if _is_private(name):
            continue
        if name == "__init__":
            continue  # rendered as the class signature
        kind = getattr(member.kind, "value", None) or str(member.kind)
        if kind == "function":
            if _is_property(member):
                properties.append((name, member))
            else:
                methods.append((name, member))
        elif kind == "attribute":
            # Properties might also be reported as 'attribute' by some
            # griffe versions — keep them in the Attributes section
            # since the user doesn't care about the implementation
            # detail; both render as ``name: type`` bullets.
            attributes.append((name, member))

    if attributes or properties:
        lines.append("")
        lines.append("**Attributes**")
        lines.append("")
        for name, member in attributes + properties:
            type_str = ""
            ann = getattr(member, "annotation", None)
            if ann:
                type_str = f": `{ann}`"
            elif getattr(member, "returns", None):  # for properties
                type_str = f": `{member.returns}`"
            line = f"- `{name}`{type_str}"
            doc = (
                member.docstring.value
                if getattr(member, "docstring", None)
                else None
            )
            first = _first_paragraph(doc)
            if first:
                first = _strip_rst_role_inline_code(
                    _rst_to_mdx_refs(first, ref_index)
                )
                line += f" — {_escape_mdx_braces(first)}"
            lines.append(line)
        lines.append("")

    if methods:
        lines.append("")
        lines.append("**Methods**")
        lines.append("")
        for name, member in methods:
            params = _params_of(member, skip_self=True)
            returns = (
                str(member.returns) if getattr(member, "returns", None) else None
            )
            anchor = f"{_anchor(klass.name)}-{_anchor(name)}"
            lines.append(f"#### `{name}` {{#{anchor}}}")
            lines.append("")
            lines.append(render_signature(name, params, returns))
            lines.append("")
            doc = (
                member.docstring.value
                if getattr(member, "docstring", None)
                else None
            )
            if doc:
                lines.append(render_docstring(doc, ref_index))

            fp = getattr(member, "filepath", None)
            ln = getattr(member, "lineno", None)
            if fp and ln:
                rel = _source_relpath(fp)
                link = f"{github_blob_root}/{rel}#L{ln}"
                lines.append(f"[source]({link})")
                lines.append("")
            lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Module-level symbol rendering
# ---------------------------------------------------------------------------


def _render_attribute_symbol(
    symbol, ref_index: dict[str, str], block: list[str]
) -> None:
    """Render a module-level attribute (constant, type alias, sentinel).

    Shows ``name: type = value`` when both annotation and value are
    available; falls back gracefully if one is missing. The previous
    implementation just tagged ``_attribute_`` with no value visible.
    """
    name = symbol.name
    ann = getattr(symbol, "annotation", None)
    val = getattr(symbol, "value", None)
    sig_parts = [f"{name}"]
    if ann:
        sig_parts.append(f": {ann}")
    if val is not None:
        # griffe gives us the source-level expression string; trim
        # giant values (could be a long literal) for readability.
        val_str = str(val)
        if len(val_str) > 200:
            val_str = val_str[:197] + "..."
        sig_parts.append(f" = {val_str}")
    block.append(f"```python\n{''.join(sig_parts)}\n```")
    block.append("")
    doc = symbol.docstring.value if getattr(symbol, "docstring", None) else None
    block.append(render_docstring(doc, ref_index))


def render_symbol(
    symbol,
    ref_index: dict[str, str],
    github_blob_root: str,
) -> str:
    """Render one top-level symbol (class / function / attribute) to
    MDX. Classes also recurse into their public members."""
    name = symbol.name
    kind = symbol.kind.value if hasattr(symbol.kind, "value") else str(symbol.kind)
    anchor = _anchor(name)

    # Custom anchor so the generated link from ref_index hits this
    # heading reliably (Docusaurus's auto-slug is fine for ASCII names
    # but we want determinism for cross-page refs).
    block: list[str] = [f"### {name} {{#{anchor}}}", "", f"_{kind}_", ""]

    if kind == "attribute":
        _render_attribute_symbol(symbol, ref_index, block)
    else:
        # Classes render their __init__ signature; drop the leading
        # ``self`` just like methods do (users call ``Agent(model=...)``,
        # never pass ``self``).
        params = _params_of(symbol, skip_self=(kind == "class"))
        returns = str(symbol.returns) if getattr(symbol, "returns", None) else None
        if params or returns:
            block.append(render_signature(name, params, returns))
            block.append("")
        doc = symbol.docstring.value if getattr(symbol, "docstring", None) else None
        block.append(render_docstring(doc, ref_index))

    fp = getattr(symbol, "filepath", None)
    ln = getattr(symbol, "lineno", None)
    if fp and ln:
        rel = _source_relpath(fp)
        link = f"{github_blob_root}/{rel}#L{ln}"
        block.append(f"[source]({link})")
        block.append("")

    # Classes: list members AFTER the class-level [source] link so the
    # top of the section is the contract overview, and the member dump
    # comes below as deep-dive material.
    if kind == "class":
        block.extend(render_class_members(symbol, ref_index, github_blob_root))

    return "\n".join(block)


# ---------------------------------------------------------------------------
# Page emission
# ---------------------------------------------------------------------------


def emit_module(
    out_path: Path,
    module_name: str,
    sidebar_position: int,
    symbols: Iterable,
    ref_index: dict[str, str],
    source_ref: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = (
        "---\n"
        f"id: {module_name.replace('.', '-')}\n"
        f"title: {module_name}\n"
        f"sidebar_position: {sidebar_position}\n"
        "hide_table_of_contents: false\n"
        "---\n\n"
    )
    body = [f"# `{module_name}`", ""]
    github_blob_root = f"https://github.com/cubeplexai/cubeloop/blob/{source_ref}"
    for sym in symbols:
        body.append(render_symbol(sym, ref_index, github_blob_root))
        body.append("")
    body.append("")
    body.append("<!-- GENERATED by build_api_reference.py — DO NOT EDIT -->")
    out_path.write_text(frontmatter + "\n".join(body), encoding="utf-8")


def resolve_source_ref(cli_ref: str | None) -> str:
    if cli_ref:
        return cli_ref
    env = os.environ.get("CUBELOOP_DOCS_SOURCE_REF") or os.environ.get(
        "CUBEPI_DOCS_SOURCE_REF"
    )
    if env:
        return env
    return "main"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory, typically website/docs/api/",
    )
    parser.add_argument(
        "--ref",
        default=None,
        help=(
            "Git ref for source links (default: main; override per "
            "snapshotted version with e.g. v0.4.0)."
        ),
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    source_ref = resolve_source_ref(args.ref)
    top: griffe.Module = griffe.load("cubeloop")  # type: ignore[assignment]

    # Two-pass: collect symbols first, build the cross-ref index, then
    # render. This lets RST refs in any module link to symbols defined
    # in any other module without forward-reference problems.
    modules_to_symbols: dict[str, list] = {}
    for mod_name, _label, _position in MODULES:
        short = mod_name.split(".")[-1]
        submod = top.members.get(short)
        if submod is None:
            try:
                submod = griffe.load(mod_name)
            except Exception as e:
                print(
                    f"[warn] {mod_name} not importable; skipping ({e})",
                    file=sys.stderr,
                )
                continue
        modules_to_symbols[mod_name] = collect_public_symbols(submod)

    ref_index = build_ref_index(modules_to_symbols)

    for mod_name, _label, position in MODULES:
        symbols = modules_to_symbols.get(mod_name)
        if symbols is None:
            continue
        out_path = args.out / f"{mod_name.replace('.', '-')}.mdx"
        emit_module(out_path, mod_name, position, symbols, ref_index, source_ref)
        print(f"[ok] wrote {out_path} ({len(symbols)} symbols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
