"""Tool for adding a task to the run-local task notebook."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, ClassVar

from my_claude.core.task.manager import TaskManager
from my_claude.core.tools.base import ParamsModel, ToolDefinition, ToolResult


@dataclass(frozen=True)
class TaskCreateTool:
    """Create a task in the shared per-run task manager."""

    params_model: ClassVar[ParamsModel | None] = None

    task_manager: TaskManager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="task_create",
            description=(
                "Create a new task to track a unit of work. Use this to break a "
                "complex goal into smaller, trackable steps. Returns JSON."
            ),
            input_schema={
                "type": "object",
                "required": ["subject"],
                "additionalProperties": False,
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "Short title for the task.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional longer description of the work.",
                    },
                    "blocked_by": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Task IDs that must be completed first.",
                    },
                },
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        return await asyncio.to_thread(self._run_blocking, arguments)

    def _run_blocking(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            task = self.task_manager.create(
                subject=str(arguments["subject"]),
                description=str(arguments.get("description") or ""),
                blocked_by=_integer_list(arguments.get("blocked_by")),
            )
        except (TypeError, ValueError) as error:
            return ToolResult.failure(str(error))

        return ToolResult.success(json.dumps(task.to_dict(), ensure_ascii=False))


def _integer_list(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("blocked_by must be an array of task IDs")
    return [int(item) for item in value]
