"""Safe tool invocation with lookup, argument validation, timeout, and error capture."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from my_claude.core.tools.base import BaseTool, ToolResult


@dataclass(frozen=True)
class ToolInvoker:
    """Safely invoke tools from a registry map."""

    tools: dict[str, BaseTool]
    timeout_seconds: float = 10.0

    async def invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult.failure(f"unknown tool: {name}")

        validation_error = validate_tool_arguments(tool.definition.input_schema, arguments)
        if validation_error is not None:
            return ToolResult.failure(validation_error, error_type="schema_error")

        task = asyncio.create_task(tool.run(arguments))
        try:
            done, _pending = await asyncio.wait({task}, timeout=self.timeout_seconds)
            if task not in done:
                task.cancel()
                await _wait_for_cancelled_tool(task)
                return ToolResult.failure(
                    f"tool timed out after {self.timeout_seconds:.2f}s",
                    error_type="timeout",
                )
            return task.result()
        except asyncio.CancelledError:
            task.cancel()
            raise
        except TimeoutError:
            return ToolResult.failure(
                f"tool timed out after {self.timeout_seconds:.2f}s",
                error_type="timeout",
            )
        except Exception as error:
            return ToolResult.failure(
                f"tool failed unexpectedly: {error}",
                error_type="runtime_error",
            )


def validate_tool_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> str | None:
    """Validate arguments against the supported JSON Schema subset."""

    if not schema:
        return None

    if schema.get("type") == "object" and not isinstance(arguments, dict):
        return "tool arguments must be an object"

    required = schema.get("required", [])
    if isinstance(required, list):
        for field_name in required:
            if isinstance(field_name, str) and field_name not in arguments:
                return f"missing required tool argument: {field_name}"

    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for field_name, value in arguments.items():
            property_schema = properties.get(field_name)
            if property_schema is None:
                if schema.get("additionalProperties") is False:
                    return f"unknown tool argument: {field_name}"
                continue
            if isinstance(property_schema, dict):
                error = _validate_value_type(field_name, value, property_schema)
                if error is not None:
                    return error

    return None


def _validate_value_type(field_name: str, value: Any, schema: dict[str, Any]) -> str | None:
    expected_type = schema.get("type")
    if expected_type is None:
        return None

    expected_types = expected_type if isinstance(expected_type, list) else [expected_type]
    if not all(isinstance(item, str) for item in expected_types):
        return None

    if any(_matches_json_schema_type(value, item) for item in expected_types):
        return None

    return f"invalid type for tool argument {field_name}: expected {expected_type}"


def _matches_json_schema_type(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (isinstance(value, int | float) and not isinstance(value, bool))
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "null":
        return value is None

    return True


async def _wait_for_cancelled_tool(task: asyncio.Task[ToolResult]) -> None:
    try:
        await asyncio.wait_for(task, timeout=0.5)
    except TimeoutError:
        task.add_done_callback(_consume_tool_result)
    except asyncio.CancelledError:
        return
    except Exception:
        return


def _consume_tool_result(task: asyncio.Task[ToolResult]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception:
        return
