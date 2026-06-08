"""Assemble all dependencies and runtime state for one agent run."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from my_claude.agent.agent import Agent, AgentResult
from my_claude.agent.control import LoopController
from my_claude.agent.events import AgentEvent, EventHandler, RunCancelledEvent
from my_claude.agent.memory import RunStatus, WorkingMemory
from my_claude.agent.tools import ToolRegistry
from my_claude.core.config import AppConfig
from my_claude.core.events.bus import EventBus
from my_claude.core.events.writer import JsonlEventWriter
from my_claude.core.tools.builtin.read_file import ReadFileTool
from my_claude.llm.client import create_llm_client

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunContext:
    """Prepared runtime objects needed before the agent loop starts."""

    run_id: str
    run_dir: Path
    timeline_path: Path
    event_bus: EventBus[AgentEvent]
    working_memory: WorkingMemory
    tools: ToolRegistry
    loop_controller: LoopController
    agent: Agent


@dataclass(frozen=True)
class RunResult:
    """Final metadata and agent output produced by a completed run."""

    run_id: str
    run_dir: Path
    timeline_path: Path
    agent_result: AgentResult


async def run_goal(
    goal: str,
    *,
    config: AppConfig,
    listeners: Sequence[EventHandler] = (),
) -> RunResult:
    context = prepare_run_context(goal, config=config, listeners=listeners)

    with JsonlEventWriter(context.timeline_path) as timeline_writer:
        context.event_bus.subscribe(timeline_writer.handle)
        try:
            agent_result = await context.agent.run()
        except asyncio.CancelledError:
            await _handle_run_interruption(context)
            raise

    return RunResult(
        run_id=context.run_id,
        run_dir=context.run_dir,
        timeline_path=context.timeline_path,
        agent_result=agent_result,
    )


def prepare_run_context(
    goal: str,
    *,
    config: AppConfig,
    listeners: Sequence[EventHandler] = (),
) -> RunContext:
    run_id = new_run_id()
    run_dir = config.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    event_bus: EventBus[AgentEvent] = EventBus()
    for listener in listeners:
        event_bus.subscribe(listener)

    llm_client = create_llm_client(config)
    tools = ToolRegistry([ReadFileTool(root=Path.cwd())])
    loop_controller = LoopController(max_iterations=config.agent_max_iterations)
    working_memory = WorkingMemory.from_goal(
        goal,
        run_id=run_id,
        max_steps=config.agent_max_iterations,
    )
    timeline_path = run_dir / "timeline.jsonl"
    agent = Agent(
        llm_client=llm_client,
        tools=tools,
        working_memory=working_memory,
        loop_controller=loop_controller,
        event_handler=event_bus.publish,
    )

    return RunContext(
        run_id=run_id,
        run_dir=run_dir,
        timeline_path=timeline_path,
        event_bus=event_bus,
        working_memory=working_memory,
        tools=tools,
        loop_controller=loop_controller,
        agent=agent,
    )


async def _handle_run_interruption(context: RunContext) -> None:
    if context.working_memory.status != RunStatus.CANCELLED:
        context.working_memory.set_status(
            RunStatus.CANCELLED,
            reason="agent run cancelled",
            transition_reason="runner caught cancellation or interruption",
        )
        await context.event_bus.publish(
            RunCancelledEvent(reason=context.working_memory.status_transition_reason)
        )

    logger.info(
        "run cancelled: run_id=%s reason=%s",
        context.run_id,
        context.working_memory.status_transition_reason,
    )


def new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:12]}"
