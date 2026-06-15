"""Tool for querying background subagent results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from my_claude.core.agents import BackgroundTaskRegistry
from my_claude.core.tools.base import ParamsModel, ToolDefinition, ToolResult


class AgentResultParams(BaseModel):
    """Arguments accepted by the agent_result tool."""

    model_config = ConfigDict(extra="ignore")

    run_id: Annotated[str, Field(strict=True)]


@dataclass(frozen=True)
class AgentResultTool:
    """Return the result of a background subagent."""

    params_model: ClassVar[ParamsModel] = AgentResultParams

    background_registry: BackgroundTaskRegistry

    @property
    def name(self) -> str:
        return "agent_result"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Return the result of a background subagent by run_id.",
            input_schema=AgentResultParams.model_json_schema(),
        )

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        params = AgentResultParams.model_validate(arguments)
        entry = self.background_registry.get(params.run_id)
        if entry is None:
            return ToolResult(
                content=(
                    f"Unknown run_id: {params.run_id}. "
                    "Only background subagents can be queried."
                ),
                is_error=True,
                error="background subagent not found",
                error_type="runtime_error",
            )

        task, context = entry
        if not task.done():
            return ToolResult.success("still running")
        if task.cancelled():
            return ToolResult(
                content="Subagent was cancelled.",
                is_error=True,
                error="subagent was cancelled",
                error_type="cancelled",
            )
        if exc := task.exception():
            return ToolResult(
                content=f"Subagent raised an exception: {exc}",
                is_error=True,
                error=str(exc),
                error_type="runtime_error",
            )

        result = context.result or "Subagent completed with no text result."
        return ToolResult.success(result)
