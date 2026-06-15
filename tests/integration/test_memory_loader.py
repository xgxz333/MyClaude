from __future__ import annotations

from pathlib import Path

import pytest

from my_claude.core.memory.loader import load_context_file


def test_load_context_file_returns_empty_string_when_missing(tmp_path: Path) -> None:
    assert load_context_file(tmp_path / "missing.md") == ""


def test_load_context_file_reads_utf8_and_strips_text(tmp_path: Path) -> None:
    context_path = tmp_path / "context.md"
    context_path.write_text("\n  用户偏好：使用 uv  \n", encoding="utf-8")

    assert load_context_file(context_path) == "用户偏好：使用 uv"


def test_load_context_file_expands_user_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    context_dir = home / ".myclaude"
    context_dir.mkdir(parents=True)
    (context_dir / "context.md").write_text("global context\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    assert load_context_file(Path("~/.myclaude/context.md")) == "global context"
