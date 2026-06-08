"""Built-in read-file tool with path and output safety limits."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from my_claude.core.tools.base import ToolDefinition, ToolResult

DEFAULT_MAX_BYTES = 64 * 1024
DEFAULT_MAX_CHARS = 16 * 1024


@dataclass(frozen=True)
class ReadFileTool:
    """Read a text file under a fixed root without allowing path traversal."""

    root: Path = Path.cwd()
    max_bytes: int = DEFAULT_MAX_BYTES
    max_chars: int = DEFAULT_MAX_CHARS

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_file",
            description="Read a UTF-8 text file under the workspace root.",
            input_schema={
                "type": "object",
                "required": ["path"],
                "additionalProperties": False,
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace root.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum decoded characters to return.",
                    },
                },
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        return await asyncio.to_thread(self._run_blocking, arguments)

    def _run_blocking(self, arguments: dict[str, Any]) -> ToolResult:
        requested_path = str(arguments["path"])
        max_chars = _coerce_max_chars(arguments.get("max_chars"), self.max_chars)
        root = self.root.resolve()

        try:
            path = _resolve_safe_path(root, requested_path)
        except ValueError as error:
            return ToolResult.failure(str(error))

        if not path.exists():
            return ToolResult.failure(f"file not found: {requested_path}")
        if not path.is_file():
            return ToolResult.failure(f"path is not a file: {requested_path}")

        read_size = min(self.max_bytes, max_chars * 4) + 1
        raw = path.read_bytes()[:read_size]
        truncated_by_bytes = len(raw) > self.max_bytes
        if truncated_by_bytes:
            raw = raw[: self.max_bytes]

        decoded = raw.decode("utf-8", errors="replace")
        truncated_by_chars = len(decoded) > max_chars
        if truncated_by_chars:
            decoded = decoded[:max_chars]

        if truncated_by_bytes or truncated_by_chars:
            decoded += "\n\n[read_file truncated]"

        return ToolResult.success(decoded)


def _resolve_safe_path(root: Path, requested_path: str) -> Path:
    if "\x00" in requested_path:
        raise ValueError("path contains null byte")

    candidate = Path(requested_path)
    if candidate.is_absolute():
        raise ValueError("absolute paths are not allowed")

    resolved = (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("path traversal is not allowed")

    return resolved


def _coerce_max_chars(value: Any, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        return default
    if value <= 0:
        return default

    return min(value, default)
