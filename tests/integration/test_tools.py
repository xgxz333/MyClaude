from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field

import my_claude.core.tools.invocation as invocation_module
from my_claude.agent.events import AgentEvent, ToolCallFailedEvent
from my_claude.core.events.bus import EventBus
from my_claude.core.tools.base import FunctionTool, ToolDefinition, ToolResult
from my_claude.core.tools.invocation import RateLimitedError
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


def test_tool_registry_validates_pydantic_params_before_tool_execution() -> None:
    class EchoParams(BaseModel):
        model_config = ConfigDict(extra="ignore")

        value: str = Field(strict=True)

    class RecordingTool:
        name = "echo"
        params_model = EchoParams
        definition = ToolDefinition(name="echo", description="Echo a value.")
        calls = 0

        async def run(self, arguments: dict[str, Any]) -> ToolResult:
            del arguments
            self.calls += 1
            return ToolResult.success("ok")

    tool = RecordingTool()
    registry = ToolRegistry([tool])

    result = asyncio.run(registry.call("echo", {"value": 1}))

    assert result.is_error is True
    assert result.error_type == "schema_error"
    assert result.error is not None
    assert "value" in result.error
    assert tool.calls == 0


def test_tool_registry_passes_pydantic_model_dump_to_tool() -> None:
    class EchoParams(BaseModel):
        model_config = ConfigDict(extra="ignore")

        value: str = Field(strict=True)
        count: int = Field(default=1, strict=True)

    class RecordingTool:
        name = "echo"
        params_model = EchoParams
        definition = ToolDefinition(name="echo", description="Echo a value.")

        def __init__(self) -> None:
            self.received: dict[str, Any] | None = None

        async def run(self, arguments: dict[str, Any]) -> ToolResult:
            self.received = arguments
            return ToolResult.success("ok")

    tool = RecordingTool()
    registry = ToolRegistry([tool])

    result = asyncio.run(registry.call("echo", {"value": "x", "extra": "ignored"}))

    assert result == ToolResult.success("ok")
    assert tool.received == {"value": "x", "count": 1}


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


def test_tool_registry_retries_retryable_tool_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(invocation_module, "_RETRY_BASE_S", 0.0)
    attempts = 0
    events: list[AgentEvent] = []
    bus: EventBus[AgentEvent] = EventBus()

    async def record_event(event: AgentEvent) -> None:
        events.append(event)

    async def handler(_arguments: dict[str, Any]) -> ToolResult:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return ToolResult.failure("try again", error_type="runtime_error")
        return ToolResult.success("ok")

    bus.subscribe(record_event)
    registry = ToolRegistry(
        [
            FunctionTool(
                definition=ToolDefinition(name="flaky", description="Flaky."),
                handler=handler,
            )
        ],
        event_bus=bus,
        run_id="run-1",
    )

    result = asyncio.run(registry.call("flaky", {}, tool_use_id="tool-1"))

    assert result == ToolResult.success("ok")
    assert attempts == 3
    failed_events = [event for event in events if isinstance(event, ToolCallFailedEvent)]
    assert [event.attempt for event in failed_events] == [1, 2]
    assert [event.error_type for event in failed_events] == ["runtime_error", "runtime_error"]


def test_tool_registry_retries_rate_limited_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(invocation_module, "_RETRY_BASE_S", 0.0)

    class RateLimitedTool:
        name = "limited"
        definition = ToolDefinition(name="limited", description="Rate limited.")
        attempts = 0

        async def run(self, _arguments: dict[str, Any]) -> ToolResult:
            self.attempts += 1
            if self.attempts == 1:
                raise RateLimitedError("slow down")
            return ToolResult.success("ok")

    tool = RateLimitedTool()
    registry = ToolRegistry([tool])

    result = asyncio.run(registry.call("limited", {}))

    assert result == ToolResult.success("ok")
    assert tool.attempts == 2


def test_tool_registry_timeout_does_not_retry() -> None:
    attempts = 0

    async def handler(_arguments: dict[str, Any]) -> str:
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(1)
        return "ok"

    registry = ToolRegistry(
        [
            FunctionTool(
                definition=ToolDefinition(name="slow_once", description="Sleep."),
                handler=handler,
            )
        ],
        timeout_seconds=0.01,
    )

    result = asyncio.run(registry.call("slow_once", {}))

    assert result.is_error is True
    assert result.error_type == "timeout"
    assert attempts == 1


def test_tool_registry_catches_unexpected_tool_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(invocation_module, "_RETRY_BASE_S", 0.0)

    class ExplodingTool:
        name = "boom"
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
