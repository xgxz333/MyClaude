from __future__ import annotations

import asyncio
import json
from pathlib import Path

from my_claude.core.config import AppConfig
from my_claude.core.context import BASE_SYSTEM_PROMPT, ExecutionContext, ExecutionMode
from my_claude.core.runner import AgentRunner, _build_registry, prepare_run_context
from my_claude.core.task.manager import TaskManager


def test_task_tools_share_one_task_manager_instance(tmp_path: Path) -> None:
    created, updated, fetched, listed = asyncio.run(_exercise_shared_task_tools(tmp_path))

    assert created["id"] == 1
    assert created["subject"] == "Plan implementation"
    assert updated["status"] == "completed"
    assert fetched["status"] == "completed"
    assert "[x] #1: Plan implementation" in listed


async def _exercise_shared_task_tools(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], str]:
    task_manager = TaskManager(tmp_path / ".tasks")
    registry = _build_registry(task_manager, workspace_root=tmp_path)

    created_result = await registry.call(
        "task_create",
        {
            "subject": "Plan implementation",
            "description": "Break down the work.",
        },
    )
    assert not created_result.is_error
    created = json.loads(created_result.content)

    updated_result = await registry.call(
        "task_update",
        {
            "task_id": created["id"],
            "status": "completed",
        },
    )
    assert not updated_result.is_error
    updated = json.loads(updated_result.content)

    fetched_result = await registry.call("task_get", {"task_id": created["id"]})
    assert not fetched_result.is_error
    fetched = json.loads(fetched_result.content)

    listed_result = await registry.call("task_list", {})
    assert not listed_result.is_error

    return created, updated, fetched, listed_result.content


def test_prepare_run_context_creates_run_local_task_sandbox(tmp_path: Path) -> None:
    config = AppConfig(runs_dir=tmp_path / "runs", llm_provider="local")

    first = prepare_run_context("first goal", config=config)
    second = prepare_run_context("second goal", config=config)

    assert first.workspace_dir == first.run_dir
    assert first.tasks_dir == first.run_dir / ".tasks"
    assert first.tasks_dir.exists()
    assert second.tasks_dir == second.run_dir / ".tasks"
    assert second.tasks_dir.exists()
    assert first.tasks_dir != second.tasks_dir


def test_prepare_run_context_applies_tool_whitelist(tmp_path: Path) -> None:
    config = AppConfig(runs_dir=tmp_path / "runs", llm_provider="local")
    execution_context = ExecutionContext(
        mode=ExecutionMode.ISOLATED,
        run_id="run-whitelist",
        goal="create tasks only",
        tool_whitelist=["task_create", "task_list"],
    )

    context = prepare_run_context(
        "ignored",
        config=config,
        execution_context=execution_context,
    )

    assert [definition.name for definition in context.tools.definitions()] == [
        "task_create",
        "task_list",
    ]


def test_prepare_run_context_uses_system_prompt_override_as_base(tmp_path: Path) -> None:
    config = AppConfig(runs_dir=tmp_path / "runs", llm_provider="local")
    context = prepare_run_context(
        "coordinate",
        config=config,
        run_id="run-prompt-override",
        system_prompt_override="You are the coordinator.",
    )

    assert context.working_memory.system_prompt_patch is not None
    assert context.working_memory.system_prompt_patch.startswith("You are the coordinator.")
    assert BASE_SYSTEM_PROMPT not in context.working_memory.system_prompt_patch


def test_agent_runner_run_and_capture_applies_tool_whitelist(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    outcome = asyncio.run(
        AgentRunner(AppConfig(runs_dir=runs_dir, llm_provider="local")).run_and_capture(
            "list tasks",
            run_id="run-filtered",
            tool_whitelist=["task_list"],
        )
    )

    events = [
        json.loads(line)
        for line in (runs_dir / "run-filtered" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert outcome.status == "completed"
    assert [
        event["data"]["tools"]
        for event in events
        if event["type"] == "llm.request_started"
    ] == [1]


def test_agent_runner_run_and_capture_uses_named_run_sandbox(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    outcome = asyncio.run(
        AgentRunner(
            AppConfig(runs_dir=runs_dir, llm_provider="local")
        ).run_and_capture("ship it", run_id="run-1")
    )

    assert outcome.status == "completed"
    assert outcome.result.startswith("Local LLM placeholder accepted goal: ship it")
    assert outcome.reason == "agent run completed"
    assert (runs_dir / "run-1" / ".tasks").exists()
