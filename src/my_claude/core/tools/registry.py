"""Tool registry and spec extraction helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from my_claude.core.bus.events import AgentEvent
from my_claude.core.events.bus import EventBus
from my_claude.core.permissions.manager import PermissionManager
from my_claude.core.tools.base import BaseTool, ToolDefinition, ToolResult
from my_claude.core.tools.invocation import ToolInvoker


class ToolRegistry:
    """In-memory registry for discovering, describing, and invoking tools by name."""

    def __init__(
        self,
        tools: Sequence[BaseTool] | None = None,
        *,
        timeout_seconds: float = 10.0,
        permission_manager: PermissionManager | None = None,
        event_bus: EventBus[AgentEvent] | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._timeout_seconds = timeout_seconds
        self._permission_manager = permission_manager
        self._event_bus = event_bus
        self._run_id = run_id
        self._session_id = session_id
        for tool in tools or ():
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.definition.name] = tool

    def definitions(self) -> list[ToolDefinition]:
        return [tool.definition for tool in self._tools.values()]

    def tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "name": definition.name,
                "description": definition.description,
                "input_schema": definition.input_schema,
            }
            for definition in self.definitions()
        ]

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        tool_use_id: str | None = None,
    ) -> ToolResult:
        invoker = ToolInvoker(
            self._tools,
            timeout_seconds=self._timeout_seconds,
            permission_manager=self._permission_manager,
            event_bus=self._event_bus,
            run_id=self._run_id,
            session_id=self._session_id,
        )
        return await invoker.invoke(name, arguments, tool_use_id=tool_use_id)
