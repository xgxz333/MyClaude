"""Tool for listing the run-local task notebook."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, ClassVar

from my_claude.core.task.manager import TaskManager
from my_claude.core.tools.base import ParamsModel, ToolDefinition, ToolResult


@dataclass(frozen=True)
class TaskListTool:
    """List tasks from the shared per-run task manager."""

    params_model: ClassVar[ParamsModel | None] = None

    task_manager: TaskManager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="task_list",
            description=(
                "List all tasks with current status and blocking dependencies. "
                "Use this to check what work remains and what can start next."
            ),
            input_schema={
                "type": "object",
                "required": [],
                "additionalProperties": False,
                "properties": {},
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        del arguments
        return await asyncio.to_thread(self._run_blocking)

    def _run_blocking(self) -> ToolResult:
        return ToolResult.success(self.task_manager.format_list())
