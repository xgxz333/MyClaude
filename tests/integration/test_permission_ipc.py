from __future__ import annotations

import asyncio
from typing import Any

from my_claude.agent.events import (
    AgentEvent,
    AgentEventType,
    PermissionGrantedEvent,
    PermissionRequestedEvent,
)
from my_claude.core.app import _permission_respond_handler, register_routes
from my_claude.core.bus.command import (
    PERMISSION_RESPOND_METHOD,
    PermissionRespondCommand,
    PermissionRespondResult,
    command_from_request,
    result_to_json,
)
from my_claude.core.bus.envelope import JsonRpcRequest
from my_claude.core.events.bus import EventBus
from my_claude.core.permissions.manager import PermissionManager
from my_claude.core.tools.base import FunctionTool, ToolDefinition
from my_claude.core.tools.registry import ToolRegistry
from my_claude.core.transport.socket_server import TCPServer


def test_permission_respond_command_parses_frontend_decision() -> None:
    request = JsonRpcRequest(
        id=1,
        method=PERMISSION_RESPOND_METHOD,
        params={"tool_use_id": "tool-1", "decision": "always_allow"},
    )

    command = command_from_request(request)

    assert isinstance(command, PermissionRespondCommand)
    assert command.tool_use_id == "tool-1"
    assert command.decision == "always_allow"
    assert result_to_json(PermissionRespondResult()) == {"ok": True}


def test_register_routes_exposes_permission_respond_method() -> None:
    server = TCPServer("127.0.0.1", 0, max_request_bytes=65536)

    register_routes(server)

    assert PERMISSION_RESPOND_METHOD in server.routes


def test_tool_registry_emits_permission_request_event_and_resumes_on_response() -> None:
    async def handler(arguments: dict[str, Any]) -> str:
        return f"ok:{arguments['value']}"

    async def run() -> tuple[str, list[AgentEvent]]:
        manager = PermissionManager(timeout_s=1)
        bus: EventBus[AgentEvent] = EventBus()
        events: list[AgentEvent] = []

        async def listener(event: AgentEvent) -> None:
            events.append(event)
            if isinstance(event, PermissionRequestedEvent):
                manager.respond(event.tool_use_id, "allow_once")

        bus.subscribe(listener)
        registry = ToolRegistry(
            [
                FunctionTool(
                    definition=ToolDefinition(name="echo", description="Echo."),
                    handler=handler,
                )
            ],
            permission_manager=manager,
            event_bus=bus,
            run_id="run-1",
            session_id="sess-1",
        )

        result = await registry.call("echo", {"value": "x"}, tool_use_id="tool-1")
        return result.content, events

    content, events = asyncio.run(run())

    assert content == "ok:x"
    assert len(events) == 2
    event = events[0]
    assert isinstance(event, PermissionRequestedEvent)
    assert event.type is AgentEventType.PERMISSION_REQUESTED
    assert event.run_id == "run-1"
    assert event.tool_use_id == "tool-1"
    assert event.tool_name == "echo"
    assert event.params == {"value": "x"}
    assert event.param_preview == "{'value': 'x'}"
    granted = events[1]
    assert isinstance(granted, PermissionGrantedEvent)
    assert granted.type is AgentEventType.PERMISSION_GRANTED
    assert granted.decision == "allow_once"


def test_permission_respond_handler_wakes_pending_request() -> None:
    async def run() -> bool:
        manager = PermissionManager(timeout_s=1)
        task = asyncio.create_task(
            manager.request_approval(
                tool_use_id="tool-1",
                tool_name="bash",
                params={"command": "printf 'ok'"},
                session_id="sess-1",
            )
        )

        while not manager.pending():
            await asyncio.sleep(0)

        result = await _permission_respond_handler(
            {"tool_use_id": "tool-1", "decision": "allow_once"},
            permission_manager=manager,
        )
        allowed = await task
        assert result == PermissionRespondResult()
        return allowed

    assert asyncio.run(run()) is True
