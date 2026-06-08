"""Base tool abstractions used by the ReAct loop."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolResult:
    """Normalized result returned by every tool invocation."""

    content: str
    is_error: bool = False
    error: str | None = None

    @classmethod
    def success(cls, content: str) -> ToolResult:
        return cls(content=content)

    @classmethod
    def failure(cls, error: str, *, content: str | None = None) -> ToolResult:
        return cls(content=content if content is not None else error, is_error=True, error=error)


@dataclass(frozen=True)
class ToolDefinition:
    """Serializable description of a tool exposed to the LLM."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


class BaseTool(Protocol):
    """Abstract template for tools that always return ToolResult."""

    @property
    def definition(self) -> ToolDefinition:
        """Return the public definition shown to the LLM."""
        ...

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool and return a success or error result."""
        ...


ToolHandler = Callable[[dict[str, Any]], Awaitable[str | ToolResult]]


@dataclass(frozen=True)
class FunctionTool:
    """Adapter for simple async functions used as tools."""

    definition: ToolDefinition
    handler: ToolHandler

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            result = await self.handler(arguments)
        except Exception as error:
            return ToolResult.failure(str(error))

        if isinstance(result, ToolResult):
            return result

        return ToolResult.success(result)

