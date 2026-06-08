from __future__ import annotations

import asyncio
from typing import Any

from my_claude.core.tools.base import FunctionTool, ToolDefinition, ToolResult
from my_claude.core.tools.registry import ToolRegistry


def test_function_tool_returns_success_result() -> None:
    async def handler(arguments: dict[str, Any]) -> str:
        return f"ok:{arguments['value']}"

    tool = FunctionTool(
        definition=ToolDefinition(name="echo", description="Echo a value."),
        handler=handler,
    )

    result = asyncio.run(tool.run({"value": "x"}))

    assert result == ToolResult.success("ok:x")


def test_function_tool_converts_exception_to_error_result() -> None:
    async def handler(_arguments: dict[str, Any]) -> str:
        raise RuntimeError("broken")

    tool = FunctionTool(
        definition=ToolDefinition(name="fail", description="Fail intentionally."),
        handler=handler,
    )

    result = asyncio.run(tool.run({}))

    assert result.is_error is True
    assert result.error == "broken"
    assert result.content == "broken"


def test_tool_registry_unknown_tool_returns_error_result() -> None:
    registry = ToolRegistry()

    result = asyncio.run(registry.call("missing", {}))

    assert result.is_error is True
    assert result.error == "unknown tool: missing"


def test_tool_registry_validates_required_arguments() -> None:
    async def handler(_arguments: dict[str, Any]) -> str:
        return "ok"

    registry = ToolRegistry(
        [
            FunctionTool(
                definition=ToolDefinition(
                    name="echo",
                    description="Echo a value.",
                    input_schema={
                        "type": "object",
                        "required": ["value"],
                        "properties": {"value": {"type": "string"}},
                    },
                ),
                handler=handler,
            )
        ]
    )

    result = asyncio.run(registry.call("echo", {}))

    assert result.is_error is True
    assert result.error == "missing required tool argument: value"


def test_tool_registry_validates_argument_types() -> None:
    async def handler(_arguments: dict[str, Any]) -> str:
        return "ok"

    registry = ToolRegistry(
        [
            FunctionTool(
                definition=ToolDefinition(
                    name="echo",
                    description="Echo a value.",
                    input_schema={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                ),
                handler=handler,
            )
        ]
    )

    result = asyncio.run(registry.call("echo", {"value": 1}))

    assert result.is_error is True
    assert result.error == "invalid type for tool argument value: expected string"


def test_tool_registry_rejects_unknown_arguments_when_schema_is_strict() -> None:
    async def handler(_arguments: dict[str, Any]) -> str:
        return "ok"

    registry = ToolRegistry(
        [
            FunctionTool(
                definition=ToolDefinition(
                    name="echo",
                    description="Echo a value.",
                    input_schema={
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"value": {"type": "string"}},
                    },
                ),
                handler=handler,
            )
        ]
    )

    result = asyncio.run(registry.call("echo", {"extra": "x"}))

    assert result.is_error is True
    assert result.error == "unknown tool argument: extra"


def test_tool_registry_timeout_returns_error_result() -> None:
    async def handler(_arguments: dict[str, Any]) -> str:
        await asyncio.sleep(1)
        return "ok"

    registry = ToolRegistry(
        [
            FunctionTool(
                definition=ToolDefinition(name="slow", description="Sleep."),
                handler=handler,
            )
        ],
        timeout_seconds=0.01,
    )

    result = asyncio.run(registry.call("slow", {}))

    assert result.is_error is True
    assert result.error == "tool timed out after 0.01s"


def test_tool_registry_catches_unexpected_tool_exceptions() -> None:
    class ExplodingTool:
        definition = ToolDefinition(name="boom", description="Raise outside FunctionTool.")

        async def run(self, _arguments: dict[str, Any]) -> ToolResult:
            raise RuntimeError("boom")

    registry = ToolRegistry([ExplodingTool()])

    result = asyncio.run(registry.call("boom", {}))

    assert result.is_error is True
    assert result.error == "tool failed unexpectedly: boom"


def test_tool_registry_extracts_tool_specs() -> None:
    async def handler(_arguments: dict[str, Any]) -> str:
        return "ok"

    registry = ToolRegistry(
        [
            FunctionTool(
                definition=ToolDefinition(
                    name="echo",
                    description="Echo a value.",
                    input_schema={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                ),
                handler=handler,
            )
        ]
    )

    assert registry.tool_specs() == [
        {
            "name": "echo",
            "description": "Echo a value.",
            "input_schema": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
        }
    ]
