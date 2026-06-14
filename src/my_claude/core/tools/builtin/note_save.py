"""Tool for saving durable session notes for future turns."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from my_claude.core.tools.base import ToolDefinition, ToolResult


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class NoteSaveTool:
    """Append one long-term note to the current session's notes.md file."""

    session_id: str
    notes_path: Path
    run_id: str = "manual"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="note_save",
            description=(
                "Save a concise fact or decision to this session's notes. "
                "These notes are visible in future turns of the same session."
            ),
            input_schema={
                "type": "object",
                "required": ["content"],
                "additionalProperties": False,
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The durable fact or decision to remember.",
                    },
                },
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        return await asyncio.to_thread(self._run_blocking, arguments)

    def _run_blocking(self, arguments: dict[str, Any]) -> ToolResult:
        content = str(arguments["content"]).strip()
        if not content:
            return ToolResult.failure("note content must not be empty")

        self.notes_path.parent.mkdir(parents=True, exist_ok=True)
        with self.notes_path.open("a", encoding="utf-8") as file:
            file.write(f"## Note ({_now()}, {self.run_id})\n{content}\n\n")
        return ToolResult.success("saved")
