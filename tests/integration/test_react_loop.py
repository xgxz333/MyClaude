from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

import my_claude.core.tools.invocation as invocation_module
from my_claude.agent.control import LoopController
from my_claude.agent.events import AgentEvent, AgentEventType
from my_claude.agent.tools import Tool, ToolDefinition, ToolRegistry, ToolResult
from my_claude.core.context import AnthropicMessage, RunStatus, ToolUseBlock, WorkingMemory
from my_claude.core.loop import AgentLoop
from my_claude.llm.client import LLMClient, LLMResponse, LLMUsage


class ScriptedLLMClient:
    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._responses = list(responses)
        self.steps: list[int | None] = []
        self.system_prompt_patches: list[str | None] = []

    async def complete(
        self,
        messages: Sequence[AnthropicMessage],
        tools: Sequence[ToolDefinition],
        *,
        step: int | None = None,
        system_prompt_patch: str | None = None,
    ) -> LLMResponse:
        del messages, tools
        self.steps.append(step)
        self.system_prompt_patches.append(system_prompt_patch)
        if not self._responses:
            raise RuntimeError("no scripted llm response left")
        return self._responses.pop(0)


class RecordingCompactor:
    def __init__(self) -> None:
        self.calls: list[tuple[WorkingMemory, LLMClient]] = []

    async def compact(self, context: WorkingMemory, provider: LLMClient) -> None:
        self.calls.append((context, provider))


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


def test_react_loop_auto_compacts_after_tool_use_response_above_threshold() -> None:
    compactor = RecordingCompactor()
    client = ScriptedLLMClient(
        [
            LLMResponse(
                content="I need the echo tool.",
                content_blocks=(
                    ToolUseBlock(id="tool-1", name="echo", input={"value": "ok"}),
                ),
                stop_reason="tool_use",
                usage=LLMUsage(input_tokens=80, context_pct=0.8),
            ),
            LLMResponse(content="Task complete.", stop_reason="end_turn"),
        ]
    )

    async def echo_tool(arguments: dict[str, Any]) -> str:
        return f"echo:{arguments['value']}"

    memory = WorkingMemory.from_goal("use the echo tool", run_id="run-1", max_steps=3)
    loop = AgentLoop(
        llm_client=client,
        tools=ToolRegistry(
            [
                Tool(
                    definition=ToolDefinition(name="echo", description="Echo a value."),
                    handler=echo_tool,
                )
            ]
        ),
        working_memory=memory,
        loop_controller=LoopController(max_iterations=3),
        compactor=compactor,
        compact_threshold=0.8,
    )

    asyncio.run(loop.run())

    assert compactor.calls == [(memory, client)]


def test_react_loop_auto_compact_ignores_high_watermark_without_compactor() -> None:
    async def echo_tool(_arguments: dict[str, Any]) -> str:
        return "echo:ok"

    memory = WorkingMemory.from_goal("use the echo tool", run_id="run-1", max_steps=3)
    loop = AgentLoop(
        llm_client=ScriptedLLMClient(
            [
                LLMResponse(
                    content="I need the echo tool.",
                    content_blocks=(ToolUseBlock(id="tool-1", name="echo", input={}),),
                    stop_reason="tool_use",
                    usage=LLMUsage(input_tokens=90, context_pct=0.9),
                ),
                LLMResponse(content="Task complete.", stop_reason="end_turn"),
            ]
        ),
        tools=ToolRegistry(
            [
                Tool(
                    definition=ToolDefinition(name="echo", description="Echo a value."),
                    handler=echo_tool,
                )
            ]
        ),
        working_memory=memory,
        loop_controller=LoopController(max_iterations=3),
        compact_threshold=0.8,
    )

    result = asyncio.run(loop.run())

    assert result.final_response == "Task complete."
    assert memory.status == RunStatus.COMPLETED


@pytest.mark.parametrize(
    ("response", "threshold"),
    [
        (
            LLMResponse(
                content="I need the echo tool.",
                content_blocks=(ToolUseBlock(id="tool-1", name="echo", input={}),),
                stop_reason="tool_use",
                usage=LLMUsage(input_tokens=80, context_pct=0.8),
            ),
            0.0,
        ),
        (
            LLMResponse(
                content="I need the echo tool.",
                content_blocks=(ToolUseBlock(id="tool-1", name="echo", input={}),),
                stop_reason="tool_use",
                usage=LLMUsage(input_tokens=79, context_pct=0.79),
            ),
            0.8,
        ),
        (
            LLMResponse(
                content="Done.",
                stop_reason="end_turn",
                usage=LLMUsage(input_tokens=80, context_pct=0.8),
            ),
            0.8,
        ),
        (
            LLMResponse(
                content="I need the echo tool.",
                content_blocks=(ToolUseBlock(id="tool-1", name="echo", input={}),),
                stop_reason="tool_use",
            ),
            0.8,
        ),
    ],
)
def test_react_loop_auto_compact_stays_disabled_unless_all_conditions_match(
    response: LLMResponse,
    threshold: float,
) -> None:
    compactor = RecordingCompactor()

    async def echo_tool(_arguments: dict[str, Any]) -> str:
        return "echo:ok"

    responses = [response]
    if response.tool_uses():
        responses.append(LLMResponse(content="Task complete.", stop_reason="end_turn"))

    loop = AgentLoop(
        llm_client=ScriptedLLMClient(responses),
        tools=ToolRegistry(
            [
                Tool(
                    definition=ToolDefinition(name="echo", description="Echo a value."),
                    handler=echo_tool,
                )
            ]
        ),
        working_memory=WorkingMemory.from_goal("ship it", run_id="run-1", max_steps=3),
        loop_controller=LoopController(max_iterations=3),
        compactor=compactor,
        compact_threshold=threshold,
    )

    asyncio.run(loop.run())

    assert compactor.calls == []


def test_react_loop_passes_system_prompt_patch_to_llm() -> None:
    client = ScriptedLLMClient([LLMResponse(content="done")])
    memory = WorkingMemory.from_goal("ship it", run_id="run-1", max_steps=1)
    memory.system_prompt_patch = "Long-term session notes:\n- fact: repo uses uv"
    loop = AgentLoop(
        llm_client=client,
        tools=ToolRegistry(),
        working_memory=memory,
        loop_controller=LoopController(max_iterations=1),
    )

    asyncio.run(loop.run())

    assert client.system_prompt_patches == ["Long-term session notes:\n- fact: repo uses uv"]


def test_react_loop_max_steps_fails_without_raising_and_includes_run_id() -> None:
    events: list[AgentEvent] = []

    async def handle(event: AgentEvent) -> None:
        events.append(event)

    async def echo_tool(arguments: dict[str, Any]) -> str:
        return f"echo:{arguments['value']}"

    memory = WorkingMemory.from_goal("use echo once", run_id="run-1", max_steps=1)
    loop = AgentLoop(
        llm_client=ScriptedLLMClient(
            [
                LLMResponse(
                    content="",
                    content_blocks=(
                        ToolUseBlock(
                            id="tool-1",
                            name="echo",
                            input={"value": "ok"},
                        ),
                    ),
                    stop_reason="tool_use",
                )
            ]
        ),
        tools=ToolRegistry(
            [
                Tool(
                    definition=ToolDefinition(name="echo", description="Echo a value."),
                    handler=echo_tool,
                )
            ]
        ),
        working_memory=memory,
        loop_controller=LoopController(max_iterations=1),
        event_handler=handle,
    )

    result = asyncio.run(loop.run())

    assert result.final_response == ""
    assert memory.status == RunStatus.FAILED
    assert memory.reason == "exceeded_max_steps"
    failed_events = [
        event for event in events if event.type == AgentEventType.RUN_FAILED
    ]
    assert len(failed_events) == 1
    assert failed_events[0].data == {"error": "exceeded_max_steps", "run_id": "run-1"}


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


def test_react_loop_records_tool_error_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(invocation_module, "_RETRY_BASE_S", 0.0)

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
