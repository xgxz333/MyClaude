from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from my_claude.agent.control import LoopController
from my_claude.agent.events import AgentEvent, AgentEventType
from my_claude.agent.tools import Tool, ToolDefinition, ToolRegistry, ToolResult
from my_claude.core.context import AnthropicMessage, RunStatus, ToolUseBlock, WorkingMemory
from my_claude.core.loop import AgentLoop
from my_claude.llm.client import LLMResponse


class ScriptedLLMClient:
    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._responses = list(responses)

    async def complete(
        self,
        messages: Sequence[AnthropicMessage],
        tools: Sequence[ToolDefinition],
    ) -> LLMResponse:
        if not self._responses:
            raise RuntimeError("no scripted llm response left")
        return self._responses.pop(0)


def test_react_loop_observes_tool_calls_acts_then_terminates() -> None:
    events: list[AgentEvent] = []

    async def handle(event: AgentEvent) -> None:
        events.append(event)

    async def echo_tool(arguments: dict[str, Any]) -> str:
        return f"echo:{arguments['value']}"

    memory = WorkingMemory.from_goal("use the echo tool", run_id="run-1", max_steps=3)
    tools = ToolRegistry(
        [
            Tool(
                definition=ToolDefinition(name="echo", description="Echo a value."),
                handler=echo_tool,
            )
        ]
    )
    loop = AgentLoop(
        llm_client=ScriptedLLMClient(
            [
                LLMResponse(
                    content="I need the echo tool.",
                    content_blocks=(
                        ToolUseBlock(
                            id="tool-1",
                            name="echo",
                            input={"value": "ok"},
                        ),
                    ),
                ),
                LLMResponse(content="Task complete."),
            ]
        ),
        tools=tools,
        working_memory=memory,
        loop_controller=LoopController(max_iterations=3),
        event_handler=handle,
    )

    result = asyncio.run(loop.run())

    assert result.final_response == "Task complete."
    assert memory.status == RunStatus.COMPLETED
    assert memory.step == 2
    assert [message.role for message in memory.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert memory.messages[2].content[0].type == "tool_result"
    assert [event.type for event in events] == [
        AgentEventType.STEP_STARTED,
        AgentEventType.LLM_REQUEST_STARTED,
        AgentEventType.LLM_RESPONSE_COMPLETED,
        AgentEventType.TOOL_CALL_STARTED,
        AgentEventType.TOOL_CALL_COMPLETED,
        AgentEventType.STEP_FINISHED,
        AgentEventType.STEP_STARTED,
        AgentEventType.LLM_REQUEST_STARTED,
        AgentEventType.LLM_RESPONSE_COMPLETED,
        AgentEventType.STEP_FINISHED,
        AgentEventType.RUN_COMPLETED,
    ]
    assert events[0].data == {"run_id": "run-1", "step": 1}


def test_react_loop_marks_memory_cancelled_when_external_cancel_is_requested() -> None:
    memory = WorkingMemory.from_goal("cancel me", run_id="run-1", max_steps=1)
    loop = AgentLoop(
        llm_client=ScriptedLLMClient([LLMResponse(content="unused")]),
        tools=ToolRegistry(),
        working_memory=memory,
        loop_controller=LoopController(max_iterations=1),
        cancel_requested=lambda: True,
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(loop.run())

    assert memory.status == RunStatus.CANCELLED
    assert memory.status_transition_reason == "cancelled before planning"


def test_react_loop_records_tool_error_without_raising() -> None:
    events: list[AgentEvent] = []

    async def handle(event: AgentEvent) -> None:
        events.append(event)

    async def failing_tool(_arguments: dict[str, Any]) -> ToolResult:
        return ToolResult.failure("tool failed")

    memory = WorkingMemory.from_goal("use a failing tool", run_id="run-1", max_steps=3)
    tools = ToolRegistry(
        [
            Tool(
                definition=ToolDefinition(name="fail", description="Fail intentionally."),
                handler=failing_tool,
            )
        ]
    )
    loop = AgentLoop(
        llm_client=ScriptedLLMClient(
            [
                LLMResponse(
                    content="I need the failing tool.",
                    content_blocks=(
                        ToolUseBlock(
                            id="tool-1",
                            name="fail",
                            input={},
                        ),
                    ),
                ),
                LLMResponse(content="I recovered from the tool error."),
            ]
        ),
        tools=tools,
        working_memory=memory,
        loop_controller=LoopController(max_iterations=3),
        event_handler=handle,
    )

    result = asyncio.run(loop.run())
    tool_result = memory.messages[2].content[0]
    completed_events = [
        event for event in events if event.type == AgentEventType.TOOL_CALL_COMPLETED
    ]

    assert result.final_response == "I recovered from the tool error."
    assert memory.status == RunStatus.COMPLETED
    assert tool_result.type == "tool_result"
    assert tool_result.is_error is True
    assert completed_events[0].data["error"] == "tool failed"
