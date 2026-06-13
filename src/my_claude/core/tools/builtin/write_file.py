"""Built-in write-file tool with path and size safety limits."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from my_claude.core.tools.base import ToolDefinition, ToolResult
from my_claude.core.tools.builtin.read_file import resolve_safe_path

DEFAULT_MAX_BYTES = 1024 * 1024


@dataclass(frozen=True)
class WriteFileTool:
    """Write UTF-8 text under a fixed root without allowing path traversal."""

    root: Path = Path.cwd()
    max_bytes: int = DEFAULT_MAX_BYTES

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="write_file",
            description=(
                "Write UTF-8 text to a file under the workspace root. Creates parent "
                "directories as needed and overwrites existing files."
            ),
            input_schema={
                "type": "object",
                "required": ["path", "content"],
                "additionalProperties": False,
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to write.",
                    },
                },
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        return await asyncio.to_thread(self._run_blocking, arguments)

    def _run_blocking(self, arguments: dict[str, Any]) -> ToolResult:
        requested_path = str(arguments["path"])
        content = str(arguments["content"])
        encoded = content.encode("utf-8")
        if len(encoded) > self.max_bytes:
            return ToolResult.failure(
                f"content too large: {len(encoded)} bytes (limit {self.max_bytes})"
            )

        try:
            path = resolve_safe_path(self.root.resolve(), requested_path)
        except ValueError as error:
            return ToolResult.failure(str(error))

        if path.exists() and not path.is_file():
            return ToolResult.failure(f"path is not a file: {requested_path}")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult.success(f"wrote {len(encoded)} bytes to {requested_path}")
