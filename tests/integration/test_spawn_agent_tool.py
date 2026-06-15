from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

from my_claude.agent.events import (
    AgentEvent,
    RunCompletedEvent,
    RunStartedEvent,
    SubagentFinishedEvent,
    SubagentStartedEvent,
)
from my_claude.agent.tools import ToolDefinition, ToolRegistry
from my_claude.core.agents import AgentProfile, AgentProfileLoader, BackgroundTaskRegistry
from my_claude.core.context import AnthropicMessage
from my_claude.core.events.bus import EventBus
from my_claude.core.tools.builtin.agent_result import AgentResultTool
from my_claude.core.tools.builtin.spawn_agent import SpawnAgentTool
from my_claude.llm.client import LLMResponse


class RecordingLLMClient:
    def __init__(self, response: str = "child done", *, delay: float = 0.0) -> None:
        self.response = response
        self.delay = delay
        self.messages: list[list[AnthropicMessage]] = []
        self.tools: list[list[ToolDefinition]] = []
        self.system_prompt_patches: list[str | None] = []

    async def complete(
        self,
        messages: Sequence[AnthropicMessage],
        tools: Sequence[ToolDefinition],
        *,
        step: int | None = None,
        system_prompt_patch: str | None = None,
    ) -> LLMResponse:
        del step
        if self.delay:
            await asyncio.sleep(self.delay)
        self.messages.append(list(messages))
        self.tools.append(list(tools))
        self.system_prompt_patches.append(system_prompt_patch)
        return LLMResponse(content=self.response)


def test_spawn_agent_runs_child_synchronously_with_isolated_context(
    tmp_path: Path,
) -> None:
    result, events, client, profiles = asyncio.run(_spawn_planner(tmp_path))

    assert result.content == "child done"
    assert not result.is_error
    assert isinstance(events[0], SubagentStartedEvent)
    assert events[0].run_id == "child-run-1"
    assert events[0].parent_run_id == "parent-run"
    assert events[0].description == "plan work"
    assert events[0].subagent_type == "planner"
    assert any(
        isinstance(event, RunStartedEvent) and event.run_id == "child-run-1"
        for event in events
    )
    assert any(
        isinstance(event, RunCompletedEvent) and event.run_id == "child-run-1"
        for event in events
    )
    finished = events[-1]
    assert isinstance(finished, SubagentFinishedEvent)
    assert finished.run_id == "child-run-1"
    assert finished.parent_run_id == "parent-run"
    assert finished.result == "child done"
    assert finished.is_error is False
    assert client.messages == [[AnthropicMessage.user_text("make a plan")]]
    assert "你是规划专家" in (client.system_prompt_patches[0] or "")
    assert profiles[0] is not None
    assert profiles[0].allowed_tools == [
        "read_file",
        "list_dir",
        "task_create",
        "task_update",
        "task_list",
    ]


def test_agent_profile_loader_uses_project_profile_before_builtin(
    tmp_path: Path,
) -> None:
    project_profile = tmp_path / ".myclaude" / "agents" / "planner.toml"
    project_profile.parent.mkdir(parents=True)
    project_profile.write_text(
        "\n".join(
            [
                "[agent]",
                'description = "custom planner"',
                'system_prompt = "Project planner prompt"',
                'allowed_tools = ["task_list"]',
            ]
        ),
        encoding="utf-8",
    )

    profile = AgentProfileLoader(project_dir=tmp_path).resolve("planner")

    assert profile == AgentProfile(
        name="planner",
        description="custom planner",
        system_prompt="Project planner prompt",
        allowed_tools=["task_list"],
    )


def test_agent_profile_loader_returns_none_for_unknown_profile(tmp_path: Path) -> None:
    profile = AgentProfileLoader(project_dir=tmp_path, home_dir=tmp_path).resolve("missing")

    assert profile is None


def test_spawn_agent_rejects_nested_depth_over_limit(tmp_path: Path) -> None:
    tool = SpawnAgentTool(
        llm_client=RecordingLLMClient(),
        workspace_root=tmp_path,
        max_steps=3,
        depth=2,
    )

    result = asyncio.run(
        tool.run(
            {
                "description": "too deep",
                "prompt": "try nesting",
            }
        )
    )

    assert result.is_error
    assert result.content == "Subagent nesting limit (2) reached; cannot spawn further subagents."


def test_spawn_agent_runs_in_background_and_agent_result_returns_result(
    tmp_path: Path,
) -> None:
    result, pending, completed, events = asyncio.run(_spawn_background_and_collect(tmp_path))

    assert result.content == (
        "Subagent started in background. run_id=child-run-bg. "
        "Use agent_result(run_id='child-run-bg') to retrieve result."
    )
    assert not result.is_error
    assert pending.content == "still running"
    assert not pending.is_error
    assert completed.content == "child done"
    assert not completed.is_error
    assert isinstance(events[0], SubagentStartedEvent)
    assert events[0].run_id == "child-run-bg"
    assert isinstance(events[-1], SubagentFinishedEvent)
    assert events[-1].run_id == "child-run-bg"
    assert events[-1].is_error is False


def test_agent_result_returns_error_for_missing_background_run() -> None:
    result = asyncio.run(
        AgentResultTool(BackgroundTaskRegistry()).run({"run_id": "missing-run"})
    )

    assert result.is_error
    assert result.content == (
        "Unknown run_id: missing-run. Only background subagents can be queried."
    )


def test_agent_result_reports_cancelled_task(tmp_path: Path) -> None:
    result = asyncio.run(_cancel_background_task(tmp_path))

    assert result.is_error
    assert result.content == "Subagent was cancelled."


def test_agent_result_reports_task_exception(tmp_path: Path) -> None:
    result = asyncio.run(_fail_background_task(tmp_path))

    assert result.is_error
    assert result.content == "Subagent raised an exception: boom"


async def _spawn_background_and_collect(
    tmp_path: Path,
) -> tuple[object, object, object, list[AgentEvent]]:
    registry = BackgroundTaskRegistry()
    parent_bus: EventBus[AgentEvent] = EventBus()
    events: list[AgentEvent] = []

    async def collect(event: AgentEvent) -> None:
        events.append(event)

    parent_bus.subscribe(collect)
    tool = SpawnAgentTool(
        llm_client=RecordingLLMClient(delay=0.01),
        workspace_root=tmp_path,
        max_steps=3,
        parent_event_bus=parent_bus,
        parent_run_id="parent-run",
        background_registry=registry,
        run_id_factory=lambda: "child-run-bg",
    )
    result = await tool.run(
        {
            "description": "background",
            "prompt": "do it in background",
            "run_in_background": True,
        }
    )
    pending = await AgentResultTool(registry).run({"run_id": "child-run-bg"})
    task_entry = registry.get("child-run-bg")
    assert task_entry is not None
    await task_entry[0]
    completed = await AgentResultTool(registry).run({"run_id": "child-run-bg"})
    return result, pending, completed, events


async def _cancel_background_task(tmp_path: Path) -> object:
    registry = BackgroundTaskRegistry()
    tool = SpawnAgentTool(
        llm_client=RecordingLLMClient(delay=10.0),
        workspace_root=tmp_path,
        max_steps=3,
        background_registry=registry,
        run_id_factory=lambda: "child-run-cancel",
    )
    await tool.run(
        {
            "description": "cancel",
            "prompt": "wait",
            "run_in_background": True,
        }
    )
    task_entry = registry.get("child-run-cancel")
    assert task_entry is not None
    task_entry[0].cancel()
    try:
        await task_entry[0]
    except asyncio.CancelledError:
        pass
    return await AgentResultTool(registry).run({"run_id": "child-run-cancel"})


async def _fail_background_task(tmp_path: Path) -> object:
    def build_broken_registry(
        _child_bus: EventBus[AgentEvent],
        _child_run_id: str,
        _child_depth: int,
        _profile: AgentProfile | None,
    ) -> ToolRegistry:
        raise RuntimeError("boom")

    registry = BackgroundTaskRegistry()
    tool = SpawnAgentTool(
        llm_client=RecordingLLMClient(),
        workspace_root=tmp_path,
        max_steps=3,
        background_registry=registry,
        child_registry_builder=build_broken_registry,
        run_id_factory=lambda: "child-run-fail",
    )
    await tool.run(
        {
            "description": "fail",
            "prompt": "explode",
            "run_in_background": True,
        }
    )
    task_entry = registry.get("child-run-fail")
    assert task_entry is not None
    try:
        await task_entry[0]
    except RuntimeError:
        pass
    return await AgentResultTool(registry).run({"run_id": "child-run-fail"})


async def _spawn_planner(
    tmp_path: Path,
) -> tuple[object, list[AgentEvent], RecordingLLMClient, list[AgentProfile | None]]:
    client = RecordingLLMClient()
    parent_bus: EventBus[AgentEvent] = EventBus()
    events: list[AgentEvent] = []
    profiles: list[AgentProfile | None] = []

    async def collect(event: AgentEvent) -> None:
        events.append(event)

    def build_child_registry(
        child_bus: EventBus[AgentEvent],
        child_run_id: str,
        child_depth: int,
        profile: AgentProfile | None,
    ) -> ToolRegistry:
        del child_bus, child_run_id, child_depth
        profiles.append(profile)
        return ToolRegistry()

    parent_bus.subscribe(collect)
    tool = SpawnAgentTool(
        llm_client=client,
        workspace_root=tmp_path,
        max_steps=3,
        parent_event_bus=parent_bus,
        parent_run_id="parent-run",
        child_registry_builder=build_child_registry,
        run_id_factory=lambda: "child-run-1",
    )
    result = await tool.run(
        {
            "description": "plan work",
            "prompt": "make a plan",
            "subagent_type": "planner",
        }
    )
    return result, events, client, profiles
