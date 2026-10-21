#!/usr/bin/env python3
"""Render one CHANGELOG.md version as GitHub release notes.

The release workflow refuses to publish when this command fails. The tag,
both package versions, and a non-empty dated changelog section must agree.
"""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

_TAG = re.compile(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)\Z")
_LINK_REF = re.compile(r"^\[[^\]\n]+\]: \S+\Z")
_PACKAGES = (
    Path("pyproject.toml"),
    Path("compat/cubepi/pyproject.toml"),
)


def release_version(tag: str) -> str:
    if _TAG.fullmatch(tag) is None:
        raise ValueError("Expected a release tag in the form v<major>.<minor>.<patch>")
    return tag[1:]


def render_release_notes(changelog: str, tag: str) -> str:
    """Return the dated section for ``tag``, including its heading."""
    version = release_version(tag)
    heading = re.compile(
        rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}[^\S\n]*$",
        re.MULTILINE,
    )
    found = heading.search(changelog)
    if found is None:
        raise ValueError(f"CHANGELOG.md has no '## [{version}] - YYYY-MM-DD' section")
    rest = changelog[found.end() :]
    following = re.search(r"^## ", rest, re.MULTILINE)
    if following is not None:
        rest = rest[: following.start()]
    lines = rest.splitlines()
    while lines and (not lines[-1].strip() or _LINK_REF.fullmatch(lines[-1])):
        lines.pop()
    body = "\n".join(lines).strip()
    if not body:
        raise ValueError(f"CHANGELOG.md section for {version} is empty")
    return f"{found.group(0).strip()}\n\n{body}\n"


def assert_package_versions(root: Path, tag: str) -> None:
    """Require every released package version to equal ``tag``."""
    version = release_version(tag)
    for relative in _PACKAGES:
        path = root / relative
        try:
            with path.open("rb") as handle:
                data = tomllib.load(handle)
        except OSError as error:
            raise ValueError(f"Cannot read {relative}") from error
        except tomllib.TOMLDecodeError as error:
            raise ValueError(f"Cannot parse {relative}") from error
        project = data.get("project")
        found = project.get("version") if isinstance(project, dict) else None
        if not isinstance(found, str) or found == "":
            raise ValueError(f"{relative} is missing project.version")
        if found != version:
            raise ValueError(f"{relative} version is {found}, release tag is {tag}")


def prepare_release_notes(root: Path, tag: str) -> str:
    assert_package_versions(root, tag)
    try:
        changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError("Cannot read CHANGELOG.md") from error
    return render_release_notes(changelog, tag)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        print(prepare_release_notes(root, args.tag), end="")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Cannot prepare release notes: {error}\n")


if __name__ == "__main__":
    main()
