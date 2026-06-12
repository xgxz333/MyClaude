"""Assemble all dependencies and runtime state for one agent run."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from my_claude.agent.agent import Agent, AgentResult
from my_claude.agent.control import LoopController
from my_claude.agent.events import (
    AgentEvent,
    EventHandler,
    RunCancelledEvent,
    RunFailedEvent,
    RunStartedEvent,
)
from my_claude.agent.memory import RunStatus, WorkingMemory
from my_claude.agent.tools import ToolRegistry
from my_claude.core.config import AppConfig
from my_claude.core.events.bus import EventBus
from my_claude.core.events.writer import JsonlEventWriter
from my_claude.core.tools.builtin.read_file import ReadFileTool
from my_claude.core.trace.provider import EventBusTracingProvider, TracingProvider
from my_claude.core.trace.writer import TraceWriter
from my_claude.llm.client import create_llm_client

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunContext:
    """Prepared runtime objects needed before the agent loop starts."""

    run_id: str
    run_dir: Path
    timeline_path: Path
    trace_path: Path
    config: AppConfig
    event_bus: EventBus[AgentEvent]
    working_memory: WorkingMemory
    tools: ToolRegistry
    loop_controller: LoopController


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
    trace_writer: TraceWriter | None = None,
) -> RunResult:
    context = prepare_run_context(goal, config=config)
    return await run_prepared_context(
        context,
        listeners=listeners,
        trace_writer=trace_writer,
    )


async def run_prepared_context(
    context: RunContext,
    *,
    listeners: Sequence[EventHandler] = (),
    trace_writer: TraceWriter | None = None,
) -> RunResult:
    """Run a prepared context after attaching durable and live event subscribers."""
    with JsonlEventWriter(context.timeline_path) as timeline_writer:
        async with _trace_writer_context(context, trace_writer) as active_trace_writer:
            context.event_bus.subscribe(timeline_writer.handle)
            if active_trace_writer is not None:
                tracing_provider = EventBusTracingProvider(active_trace_writer)
                context.event_bus.subscribe(tracing_provider.handle)
            for listener in listeners:
                context.event_bus.subscribe(listener)

            try:
                await context.event_bus.publish(
                    RunStartedEvent(
                        goal=context.working_memory.goal,
                        run_id=context.run_id,
                    )
                )
                try:
                    agent = _create_agent(context, trace_writer=active_trace_writer)
                except Exception as error:
                    await _handle_run_failure(context, error)
                    raise
                agent_result = await agent.run(emit_run_started=False)
            except asyncio.CancelledError:
                await _handle_run_interruption(context)
                raise

    return RunResult(
        run_id=context.run_id,
        run_dir=context.run_dir,
        timeline_path=context.timeline_path,
        agent_result=agent_result,
    )


@asynccontextmanager
async def _trace_writer_context(
    context: RunContext,
    trace_writer: TraceWriter | None,
) -> AsyncIterator[TraceWriter | None]:
    if trace_writer is not None:
        yield trace_writer
        return
    if not context.config.trace_enabled:
        yield None
        return

    async with TraceWriter(context.trace_path) as local_trace_writer:
        yield local_trace_writer


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

    tools = ToolRegistry([ReadFileTool(root=Path.cwd())])
    loop_controller = LoopController(max_iterations=config.agent_max_iterations)
    working_memory = WorkingMemory.from_goal(
        goal,
        run_id=run_id,
        max_steps=config.agent_max_iterations,
    )
    timeline_path = run_dir / "events.jsonl"
    trace_path = run_dir / "trace.jsonl"
    return RunContext(
        run_id=run_id,
        run_dir=run_dir,
        timeline_path=timeline_path,
        trace_path=trace_path,
        config=config,
        event_bus=event_bus,
        working_memory=working_memory,
        tools=tools,
        loop_controller=loop_controller,
    )


def _create_agent(context: RunContext, *, trace_writer: TraceWriter | None) -> Agent:
    llm_client = create_llm_client(
        context.config,
        event_handler=context.event_bus.publish,
        run_id=context.run_id,
    )
    if trace_writer is not None:
        llm_client = TracingProvider(
            llm_client,
            trace_writer,
            include_payload=context.config.trace_include_llm_payload,
            run_id=context.run_id,
        )
    return Agent(
        llm_client=llm_client,
        tools=context.tools,
        working_memory=context.working_memory,
        loop_controller=context.loop_controller,
        event_handler=context.event_bus.publish,
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


async def _handle_run_failure(context: RunContext, error: Exception) -> None:
    if context.working_memory.status != RunStatus.FAILED:
        context.working_memory.set_status(
            RunStatus.FAILED,
            reason=str(error),
            transition_reason="runner failed before agent loop started",
        )
    await context.event_bus.publish(
        RunFailedEvent(
            message=str(error),
            error=str(error),
            run_id=context.run_id,
        )
    )


def new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:6]}"
