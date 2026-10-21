"""The release tag, package versions, and changelog section stay in lockstep."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_renderer() -> ModuleType:
    path = ROOT / "scripts" / "release-notes.py"
    spec = importlib.util.spec_from_file_location("release_notes", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def renderer() -> ModuleType:
    return load_renderer()


def write_release(root: Path, changelog: str, cubeloop: str, cubepi: str) -> None:
    (root / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "cubeloop"\nversion = "{cubeloop}"\n',
        encoding="utf-8",
    )
    package = root / "compat" / "cubepi"
    package.mkdir(parents=True)
    (package / "pyproject.toml").write_text(
        f'[project]\nname = "cubepi"\nversion = "{cubepi}"\n',
        encoding="utf-8",
    )


def test_renders_only_the_matching_version_section(
    renderer: ModuleType, tmp_path: Path
) -> None:
    write_release(
        tmp_path,
        """# Changelog

## [Unreleased]

## [1.2.3] - 2026-01-02

### Fixed

- The useful fix (`#9`).

[example](../guide.md)

[1.2.3]: https://example.com/compare/1.2.3
[Unreleased]: https://example.com/unreleased

## [1.2.2] - 2026-01-01

### Added

- Older feature.
""",
        "1.2.3",
        "1.2.3",
    )
    notes = renderer.prepare_release_notes(tmp_path, "v1.2.3")
    assert notes == (
        "## [1.2.3] - 2026-01-02\n"
        "\n"
        "### Fixed\n"
        "\n"
        "- The useful fix (`#9`).\n"
        "\n"
        "[example](../guide.md)\n"
    )


def test_rejects_invalid_tags(renderer: ModuleType) -> None:
    for tag in ("0.15.1", "v0.15", "v01.2.3", "v1.2.3-rc1", "../v1.2.3"):
        with pytest.raises(ValueError, match="v<major>"):
            renderer.release_version(tag)


def test_missing_or_empty_section_blocks_publication(
    renderer: ModuleType, tmp_path: Path
) -> None:
    write_release(
        tmp_path,
        "## [1.2.3] - 2026-01-02\n\n## [1.2.2] - 2026-01-01\n\n- Older.\n",
        "1.2.3",
        "1.2.3",
    )
    with pytest.raises(ValueError, match="empty"):
        renderer.prepare_release_notes(tmp_path, "v1.2.3")

    (tmp_path / "CHANGELOG.md").write_text(
        "## [Unreleased]\n\n- Not this release.\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no '## \\[1.2.3\\]"):
        renderer.prepare_release_notes(tmp_path, "v1.2.3")


def test_package_version_must_match_the_tag(
    renderer: ModuleType, tmp_path: Path
) -> None:
    write_release(
        tmp_path,
        "## [1.2.3] - 2026-01-02\n\n- Ready.\n",
        "1.2.3",
        "1.2.2",
    )
    with pytest.raises(ValueError, match="compat/cubepi/pyproject.toml"):
        renderer.prepare_release_notes(tmp_path, "v1.2.3")


def test_checked_out_tree_renders_its_package_version(renderer: ModuleType) -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        version = tomllib.load(handle)["project"]["version"]
    notes = renderer.prepare_release_notes(ROOT, f"v{version}")
    assert notes.startswith(f"## [{version}] - ")
    assert "\n### " in notes
    assert "https://github.com/cubeplexai/cubeloop/compare/" not in notes

    with pytest.raises(ValueError, match="pyproject.toml version is"):
        renderer.prepare_release_notes(ROOT, "v0.0.0")
