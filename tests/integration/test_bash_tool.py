from __future__ import annotations

import asyncio
from pathlib import Path

from my_claude.core.tools.base import ToolResult
from my_claude.core.tools.builtin.bash import BashTool
from my_claude.core.tools.registry import ToolRegistry


def test_bash_tool_returns_stdout_and_exit_status(tmp_path: Path) -> None:
    tool = BashTool(cwd=tmp_path)

    result = asyncio.run(tool.run({"command": "printf 'hello'"}))

    assert result.is_error is False
    assert "$ printf 'hello'" in result.content
    assert "[exit 0]" in result.content
    assert "stdout:\nhello" in result.content
    assert "stderr:\n[empty]" in result.content


def test_bash_tool_returns_error_feedback_for_nonzero_exit(tmp_path: Path) -> None:
    tool = BashTool(cwd=tmp_path)

    result = asyncio.run(
        tool.run({"command": "sh -c 'echo bad >&2; exit 7'"})
    )

    assert result.is_error is True
    assert result.error == "command exited with status 7"
    assert result.error_type == "runtime_error"
    assert "[exit 7]" in result.content
    assert "stderr:\nbad" in result.content


def test_bash_tool_times_out_noninteractive_commands(tmp_path: Path) -> None:
    tool = BashTool(cwd=tmp_path)

    result = asyncio.run(tool.run({"command": "sleep 2", "timeout": 1}))

    assert result.is_error is True
    assert result.error_type == "timeout"
    assert "timeout after 1s" in result.content


def test_bash_tool_truncates_large_output_without_deadlock(tmp_path: Path) -> None:
    tool = BashTool(cwd=tmp_path, max_output_bytes=32)

    result = asyncio.run(tool.run({"command": "yes x | head -c 4096"}))

    assert result.is_error is False
    assert "[stdout truncated at 65536 bytes]" not in result.content
    assert "[stdout truncated at 32 bytes]" in result.content


def test_bash_tool_can_be_cancelled_by_registry_timeout(tmp_path: Path) -> None:
    registry = ToolRegistry(
        [BashTool(cwd=tmp_path, default_timeout_seconds=60)],
        timeout_seconds=0.05,
    )

    async def call_with_guard() -> ToolResult:
        return await asyncio.wait_for(
            registry.call("bash", {"command": "sleep 5"}),
            timeout=1.0,
        )

    result = asyncio.run(call_with_guard())

    assert result.is_error is True
    assert result.error == "tool timed out after 0.05s"
    assert result.error_type == "timeout"
