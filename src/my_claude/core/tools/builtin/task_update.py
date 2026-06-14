"""Tool for updating the run-local task notebook."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, ClassVar, cast

from my_claude.core.task.manager import TaskManager
from my_claude.core.task.model import TaskStatus
from my_claude.core.tools.base import ParamsModel, ToolDefinition, ToolResult


@dataclass(frozen=True)
class TaskUpdateTool:
    """Update a task through the shared per-run task manager."""

    params_model: ClassVar[ParamsModel | None] = None

    task_manager: TaskManager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="task_update",
            description=(
                "Update a task's status or dependency list. Set status to "
                "'in_progress' when starting work and 'completed' when finished. "
                "Completing a task clears it from other tasks' blocked_by lists."
            ),
            input_schema={
                "type": "object",
                "required": ["task_id"],
                "additionalProperties": False,
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "ID of the task to update.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed"],
                        "description": "New task status.",
                    },
                    "add_blocked_by": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Task IDs to add to blocked_by.",
                    },
                    "remove_blocked_by": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Task IDs to remove from blocked_by.",
                    },
                },
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        return await asyncio.to_thread(self._run_blocking, arguments)

    def _run_blocking(self, arguments: dict[str, Any]) -> ToolResult:
        status = arguments.get("status")
        if status is not None and status not in ("pending", "in_progress", "completed"):
            return ToolResult.failure(f"invalid status: {status!r}")

        try:
            task = self.task_manager.update(
                int(arguments["task_id"]),
                status=cast(TaskStatus | None, status),
                add_blocked_by=_integer_list(arguments.get("add_blocked_by")),
                remove_blocked_by=_integer_list(arguments.get("remove_blocked_by")),
            )
        except (TypeError, ValueError) as error:
            return ToolResult.failure(str(error))

        return ToolResult.success(json.dumps(task.to_dict(), ensure_ascii=False))


def _integer_list(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("dependency fields must be arrays of task IDs")
    return [int(item) for item in value]
