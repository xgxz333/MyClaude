from __future__ import annotations

import asyncio
from pathlib import Path

from my_claude.core.tools.builtin.read_file import ReadFileTool


def test_read_file_tool_reads_relative_file(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello", encoding="utf-8")
    tool = ReadFileTool(root=tmp_path)

    result = asyncio.run(tool.run({"path": "notes.txt"}))

    assert result.is_error is False
    assert result.content == "hello"


def test_read_file_tool_rejects_path_traversal(tmp_path: Path) -> None:
    tool = ReadFileTool(root=tmp_path)

    result = asyncio.run(tool.run({"path": "../secret.txt"}))

    assert result.is_error is True
    assert result.error == "path traversal is not allowed"


def test_read_file_tool_rejects_absolute_paths(tmp_path: Path) -> None:
    tool = ReadFileTool(root=tmp_path)

    result = asyncio.run(tool.run({"path": str(tmp_path / "notes.txt")}))

    assert result.is_error is True
    assert result.error == "absolute paths are not allowed"


def test_read_file_tool_truncates_output_by_character_limit(tmp_path: Path) -> None:
    file_path = tmp_path / "large.txt"
    file_path.write_text("abcdef", encoding="utf-8")
    tool = ReadFileTool(root=tmp_path, max_chars=4)

    result = asyncio.run(tool.run({"path": "large.txt"}))

    assert result.content == "abcd\n[truncated]"


def test_read_file_tool_decodes_invalid_utf8_with_replacement(tmp_path: Path) -> None:
    file_path = tmp_path / "binary.txt"
    file_path.write_bytes(b"ok\xff")
    tool = ReadFileTool(root=tmp_path)

    result = asyncio.run(tool.run({"path": "binary.txt"}))

    assert result.is_error is False
    assert result.content == "ok\ufffd"


def test_read_file_tool_exposes_strict_schema(tmp_path: Path) -> None:
    tool = ReadFileTool(root=tmp_path)

    assert tool.definition.input_schema["required"] == ["path"]
    assert tool.definition.input_schema["additionalProperties"] is False
