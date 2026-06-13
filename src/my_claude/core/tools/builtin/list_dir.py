"""Built-in list-directory tool with traversal and output limits."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from my_claude.core.tools.base import ToolDefinition, ToolResult
from my_claude.core.tools.builtin.read_file import resolve_safe_path

DEFAULT_MAX_DEPTH = 4
DEFAULT_MAX_ENTRIES = 200


@dataclass(frozen=True)
class ListDirTool:
    """List a directory tree under a fixed root."""

    root: Path = Path.cwd()
    max_depth: int = DEFAULT_MAX_DEPTH
    max_entries: int = DEFAULT_MAX_ENTRIES

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_dir",
            description=(
                "List the contents of a directory under the workspace root as a tree. "
                "Hidden entries are included. Output depth and total entries are limited."
            ),
            input_schema={
                "type": "object",
                "required": [],
                "additionalProperties": False,
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to the workspace root.",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": (
                            f"How many levels deep to recurse. Max {self.max_depth}."
                        ),
                    },
                },
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        return await asyncio.to_thread(self._run_blocking, arguments)

    def _run_blocking(self, arguments: dict[str, Any]) -> ToolResult:
        requested_path = str(arguments.get("path") or ".")
        max_depth = _coerce_depth(arguments.get("max_depth"), self.max_depth)

        try:
            root = resolve_safe_path(self.root.resolve(), requested_path)
        except ValueError as error:
            return ToolResult.failure(str(error))

        if not root.exists():
            return ToolResult.failure(f"directory not found: {requested_path}")
        if not root.is_dir():
            return ToolResult.failure(f"path is not a directory: {requested_path}")

        lines = [f"{requested_path.rstrip('/') or '.'}/"]
        count = 0
        truncated = False

        def walk(directory: Path, depth: int, prefix: str) -> None:
            nonlocal count, truncated
            if depth > max_depth or truncated:
                return

            entries = sorted(directory.iterdir(), key=lambda entry: (entry.is_file(), entry.name))
            for index, entry in enumerate(entries):
                if count >= self.max_entries:
                    lines.append(f"{prefix}... (truncated)")
                    truncated = True
                    return

                connector = "`-- " if index == len(entries) - 1 else "|-- "
                suffix = "/" if entry.is_dir() else ""
                lines.append(f"{prefix}{connector}{entry.name}{suffix}")
                count += 1

                if entry.is_dir() and depth < max_depth:
                    extension = "    " if index == len(entries) - 1 else "|   "
                    walk(entry, depth + 1, prefix + extension)

        walk(root, 1, "")
        return ToolResult.success("\n".join(lines))


def _coerce_depth(value: Any, maximum: int) -> int:
    if value is None:
        return min(2, maximum)
    if not isinstance(value, int) or isinstance(value, bool):
        return min(2, maximum)
    if value <= 0:
        return 1
    return min(value, maximum)
