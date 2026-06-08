"""Compatibility exports for core tool abstractions."""

from __future__ import annotations

from my_claude.core.tools.base import (
    BaseTool,
    FunctionTool,
    ToolDefinition,
    ToolHandler,
    ToolResult,
)
from my_claude.core.tools.registry import ToolRegistry

Tool = FunctionTool

__all__ = [
    "BaseTool",
    "FunctionTool",
    "Tool",
    "ToolDefinition",
    "ToolHandler",
    "ToolRegistry",
    "ToolResult",
]
