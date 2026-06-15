"""Tool for reading one task from the run-local task notebook."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, ClassVar

from my_claude.core.task.manager import TaskManager
from my_claude.core.tools.base import ParamsModel, ToolDefinition, ToolResult


@dataclass(frozen=True)
class TaskGetTool:
    """Fetch a task by ID from the shared per-run task manager."""

    params_model: ClassVar[ParamsModel | None] = None

    task_manager: TaskManager

    @property
    def name(self) -> str:
        return "task_get"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Get full details of a task by integer ID. Returns JSON.",
            input_schema={
                "type": "object",
                "required": ["task_id"],
                "additionalProperties": False,
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "ID of the task to retrieve.",
                    },
                },
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        return await asyncio.to_thread(self._run_blocking, arguments)

    def _run_blocking(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            task = self.task_manager.get(int(arguments["task_id"]))
        except (TypeError, ValueError) as error:
            return ToolResult.failure(str(error))

        return ToolResult.success(json.dumps(task.to_dict(), ensure_ascii=False))
