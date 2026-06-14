"""Safe tool invocation with lookup, argument validation, timeout, and error capture."""

from __future__ import annotations

import asyncio
import datetime
import time
from dataclasses import dataclass
from datetime import UTC
from typing import Any

from pydantic import ValidationError

from my_claude.core.bus.events import (
    AgentEvent,
    PermissionDeniedEvent,
    PermissionGrantedEvent,
    PermissionRequestedEvent,
    ToolCallFailedEvent,
)
from my_claude.core.events.bus import EventBus
from my_claude.core.permissions.manager import PermissionManager
from my_claude.core.tools.base import BaseTool, ToolResult

_MAX_RETRIES = 2
_RETRY_BASE_S = 2.0
_RETRYABLE = {"runtime_error", "rate_limited"}


class RateLimitedError(Exception):
    """Raised by tools when an invocation is rate limited."""


def _now() -> str:
    return datetime.datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ToolInvoker:
    """Safely invoke tools from a registry map."""

    tools: dict[str, BaseTool]
    timeout_seconds: float = 10.0
    permission_manager: PermissionManager | None = None
    event_bus: EventBus[AgentEvent] | None = None
    run_id: str | None = None
    session_id: str | None = None

    async def invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        tool_use_id: str | None = None,
    ) -> ToolResult:
        started_at = time.monotonic()
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult.failure(f"unknown tool: {name}")

        resolved_tool_use_id = tool_use_id or f"{name}:manual"

        params_model = getattr(tool, "params_model", None)
        if params_model is not None:
            try:
                arguments = params_model.model_validate(dict(arguments)).model_dump()
            except ValidationError as error:
                return await self._fail(
                    name,
                    "schema_error",
                    str(error),
                    attempt=1,
                    tool_use_id=resolved_tool_use_id,
                    elapsed_ms=_elapsed_ms(started_at),
                )
        else:
            validation_error = validate_tool_arguments(tool.definition.input_schema, arguments)
            if validation_error is not None:
                return await self._fail(
                    name,
                    "schema_error",
                    validation_error,
                    attempt=1,
                    tool_use_id=resolved_tool_use_id,
                    elapsed_ms=_elapsed_ms(started_at),
                )

        if self.permission_manager is not None:
            allowed, decision = await self.permission_manager.check_and_wait(
                tool_use_id=resolved_tool_use_id,
                tool_name=name,
                params=arguments,
                session_id=self.session_id or "",
                event_emitter=self._emit_permission if self.event_bus is not None else None,
            )
            if allowed and decision != "auto_allow":
                await self._emit_permission_granted(resolved_tool_use_id, decision)
            if not allowed:
                if decision != "auto_deny":
                    await self._emit_permission_denied(resolved_tool_use_id, decision)
                return await self._fail(
                    name,
                    "permission_denied",
                    "Permission denied by user. You may not execute this command. "
                    "Try an alternative approach or ask the user what to do.",
                    attempt=1,
                    tool_use_id=resolved_tool_use_id,
                    elapsed_ms=_elapsed_ms(started_at),
                )

        for attempt in range(1, _MAX_RETRIES + 2):
            error_class = "runtime_error"
            error_message = ""
            error_content: str | None = None
            try:
                task = asyncio.create_task(tool.run(arguments))
                try:
                    result = await asyncio.wait_for(
                        task,
                        timeout=self.timeout_seconds,
                    )
                except TimeoutError:
                    await _wait_for_cancelled_tool(task)
                    return await self._fail(
                        name,
                        "timeout",
                        f"tool timed out after {self.timeout_seconds:.2f}s",
                        attempt=attempt,
                        tool_use_id=resolved_tool_use_id,
                        elapsed_ms=_elapsed_ms(started_at),
                    )

                if not result.is_error:
                    return result

                error_class = result.error_type or "runtime_error"
                error_message = result.error or result.content
                error_content = result.content
            except asyncio.CancelledError:
                raise
            except RateLimitedError as error:
                error_class = "rate_limited"
                error_message = str(error) or "tool invocation rate limited"
            except Exception as error:
                error_class = "runtime_error"
                error_message = f"tool failed unexpectedly: {error}"

            if error_class in _RETRYABLE and attempt <= _MAX_RETRIES:
                await self._emit_tool_call_failed(
                    name,
                    error_class,
                    error_message,
                    attempt=attempt,
                    tool_use_id=resolved_tool_use_id,
                    elapsed_ms=_elapsed_ms(started_at),
                )
                await asyncio.sleep(_RETRY_BASE_S * (2 ** (attempt - 1)))
                continue

            return await self._fail(
                name,
                error_class,
                error_message,
                attempt=attempt,
                tool_use_id=resolved_tool_use_id,
                elapsed_ms=_elapsed_ms(started_at),
                content=error_content,
            )

        return await self._fail(
            name,
            "runtime_error",
            "tool failed unexpectedly",
            attempt=_MAX_RETRIES + 1,
            tool_use_id=resolved_tool_use_id,
            elapsed_ms=_elapsed_ms(started_at),
        )


    async def _emit_permission(self, raw: dict[str, Any]) -> None:
        if self.event_bus is None:
            return

        params = raw.get("params")
        event = PermissionRequestedEvent(
            run_id=self.run_id,
            tool_use_id=str(raw["tool_use_id"]),
            tool_name=str(raw["tool_name"]),
            params=params if isinstance(params, dict) else {},
            param_preview=str(raw.get("param_preview", "")),
            session_id=str(raw.get("session_id", "")),
            ts=str(raw.get("ts", "")) or None,
        )
        await self.event_bus.publish(event)

    async def _emit_permission_granted(self, tool_use_id: str, decision: str) -> None:
        if self.event_bus is None:
            return

        await self.event_bus.publish(
            PermissionGrantedEvent(
                run_id=self.run_id,
                tool_use_id=tool_use_id,
                decision=decision,
                ts=_now(),
            )
        )

    async def _emit_permission_denied(self, tool_use_id: str, decision: str) -> None:
        if self.event_bus is None:
            return

        await self.event_bus.publish(
            PermissionDeniedEvent(
                run_id=self.run_id,
                tool_use_id=tool_use_id,
                decision=decision,
                ts=_now(),
            )
        )

    async def _emit_tool_call_failed(
        self,
        tool_name: str,
        error_class: str,
        error_message: str,
        *,
        attempt: int,
        tool_use_id: str | None,
        elapsed_ms: int,
    ) -> None:
        if self.event_bus is None:
            return

        await self.event_bus.publish(
            ToolCallFailedEvent(
                run_id=self.run_id,
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                error_class=error_class,
                error_message=error_message,
                error=error_message,
                error_type=error_class,
                elapsed_ms=elapsed_ms,
                attempt=attempt,
                ts=_now(),
            )
        )

    async def _fail(
        self,
        tool_name: str,
        error_class: str,
        error_message: str,
        *,
        attempt: int,
        tool_use_id: str | None,
        elapsed_ms: int,
        content: str | None = None,
    ) -> ToolResult:
        await self._emit_tool_call_failed(
            tool_name,
            error_class,
            error_message,
            attempt=attempt,
            tool_use_id=tool_use_id,
            elapsed_ms=elapsed_ms,
        )
        return ToolResult.failure(error_message, content=content, error_type=error_class)


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


def _elapsed_ms(started_at: float) -> int:
    return int((time.monotonic() - started_at) * 1000)


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
