from __future__ import annotations

import asyncio
from pathlib import Path

from my_claude.core.tools.builtin.list_dir import ListDirTool
from my_claude.core.tools.builtin.write_file import WriteFileTool


def test_write_file_creates_parent_directories_and_writes_text(tmp_path: Path) -> None:
    tool = WriteFileTool(root=tmp_path)

    result = asyncio.run(
        tool.run({"path": "notes/plan.txt", "content": "ship it"})
    )

    assert result.is_error is False
    assert result.content == "wrote 7 bytes to notes/plan.txt"
    assert (tmp_path / "notes" / "plan.txt").read_text(encoding="utf-8") == "ship it"


def test_write_file_rejects_path_traversal(tmp_path: Path) -> None:
    tool = WriteFileTool(root=tmp_path)

    result = asyncio.run(tool.run({"path": "../outside.txt", "content": "bad"}))

    assert result.is_error is True
    assert result.error == "path traversal is not allowed"
    assert not (tmp_path.parent / "outside.txt").exists()


def test_write_file_rejects_large_content(tmp_path: Path) -> None:
    tool = WriteFileTool(root=tmp_path, max_bytes=4)

    result = asyncio.run(tool.run({"path": "big.txt", "content": "12345"}))

    assert result.is_error is True
    assert result.error == "content too large: 5 bytes (limit 4)"
    assert not (tmp_path / "big.txt").exists()


def test_list_dir_returns_tree(tmp_path: Path) -> None:
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "module.py").write_text("print('ok')", encoding="utf-8")
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    tool = ListDirTool(root=tmp_path)

    result = asyncio.run(tool.run({"path": ".", "max_depth": 3}))

    assert result.is_error is False
    assert "./" in result.content
    assert "README.md" in result.content
    assert "src/" in result.content
    assert "pkg/" in result.content
    assert "module.py" in result.content


def test_list_dir_rejects_non_directory_and_traversal(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")
    tool = ListDirTool(root=tmp_path)

    file_result = asyncio.run(tool.run({"path": "file.txt"}))
    traversal_result = asyncio.run(tool.run({"path": "../"}))

    assert file_result.is_error is True
    assert file_result.error == "path is not a directory: file.txt"
    assert traversal_result.is_error is True
    assert traversal_result.error == "path traversal is not allowed"


def test_list_dir_truncates_total_entries(tmp_path: Path) -> None:
    for index in range(5):
        (tmp_path / f"file_{index}.txt").write_text(str(index), encoding="utf-8")
    tool = ListDirTool(root=tmp_path, max_entries=3)

    result = asyncio.run(tool.run({"path": "."}))

    assert result.is_error is False
    assert "... (truncated)" in result.content
