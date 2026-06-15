"""Assemble all dependencies and runtime state for one agent run."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
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
from my_claude.core.agents import BackgroundTaskRegistry
from my_claude.core.compaction import Compactor
from my_claude.core.config import AppConfig
from my_claude.core.context import AnthropicMessage, ExecutionContext
from my_claude.core.events.bus import EventBus
from my_claude.core.events.writer import JsonlEventWriter
from my_claude.core.mcp import McpServerManager
from my_claude.core.memory.loader import load_context_file
from my_claude.core.permissions.manager import PermissionManager
from my_claude.core.task.manager import TaskManager
from my_claude.core.tools.base import BaseTool
from my_claude.core.tools.builtin import (
    AgentResultTool,
    BashTool,
    ListDirTool,
    NoteSaveTool,
    ReadFileTool,
    SpawnAgentTool,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskUpdateTool,
    WriteFileTool,
)
from my_claude.core.trace.provider import EventBusTracingProvider, TracingProvider
from my_claude.core.trace.writer import TraceWriter
from my_claude.llm.client import LLMClient, create_llm_client

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunContext:
    """Prepared runtime objects needed before the agent loop starts."""

    run_id: str
    run_dir: Path
    workspace_dir: Path
    tasks_dir: Path
    timeline_path: Path
    trace_path: Path
    config: AppConfig
    session_id: str | None
    event_bus: EventBus[AgentEvent]
    working_memory: WorkingMemory
    tools: ToolRegistry
    loop_controller: LoopController
    permission_manager: PermissionManager | None = None
    tool_whitelist: list[str] | None = None
    background_tasks: BackgroundTaskRegistry = field(default_factory=BackgroundTaskRegistry)
    mcp_manager: McpServerManager | None = None


@dataclass(frozen=True)
class RunResult:
    """Final metadata and agent output produced by a completed run."""

    run_id: str
    run_dir: Path
    timeline_path: Path
    agent_result: AgentResult
    messages: list[AnthropicMessage] = field(default_factory=list)


@dataclass(frozen=True)
class RunOutcome:
    """Kama-style captured run outcome returned by AgentRunner."""

    status: str
    result: str
    reason: str | None


class AgentRunner:
    """Convenience runner that prepares a run sandbox and captures the final result."""

    def __init__(
        self,
        config: AppConfig,
        *,
        listeners: Sequence[EventHandler] = (),
        trace_writer: TraceWriter | None = None,
        mcp_manager: McpServerManager | None = None,
    ) -> None:
        self._config = config
        self._listeners = tuple(listeners)
        self._trace_writer = trace_writer
        self._mcp_manager = mcp_manager

    async def run(self, goal: str, *, run_id: str | None = None) -> None:
        await self.run_and_capture(goal, run_id=run_id)

    async def run_and_capture(
        self,
        goal: str,
        *,
        run_id: str | None = None,
        system_prompt_override: str | None = None,
        tool_whitelist: list[str] | None = None,
    ) -> RunOutcome:
        global_ctx = load_context_file(Path("~/.myclaude/context.md"))
        project_ctx = load_context_file(Path(".myclaude/context.md"))
        context = prepare_run_context(
            goal,
            config=self._config,
            run_id=run_id,
            global_context=global_ctx,
            project_context=project_ctx,
            system_prompt_override=system_prompt_override,
            tool_whitelist=tool_whitelist,
            mcp_manager=self._mcp_manager,
        )
        result = await run_prepared_context(
            context,
            listeners=self._listeners,
            trace_writer=self._trace_writer,
        )
        return RunOutcome(
            status=context.working_memory.status.value,
            result=result.agent_result.final_response,
            reason=context.working_memory.reason,
        )


async def run_goal(
    goal: str,
    *,
    config: AppConfig,
    listeners: Sequence[EventHandler] = (),
    trace_writer: TraceWriter | None = None,
    mcp_manager: McpServerManager | None = None,
) -> RunResult:
    context = prepare_run_context(goal, config=config, mcp_manager=mcp_manager)
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
        messages=context.working_memory.llm_messages(),
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
    run_id: str | None = None,
    execution_context: ExecutionContext | None = None,
    permission_manager: PermissionManager | None = None,
    global_context: str | None = None,
    project_context: str | None = None,
    system_prompt_override: str | None = None,
    tool_whitelist: list[str] | None = None,
    mcp_manager: McpServerManager | None = None,
) -> RunContext:
    global_ctx = (
        load_context_file(Path("~/.myclaude/context.md"))
        if global_context is None
        else global_context
    )
    project_ctx = (
        load_context_file(Path(".myclaude/context.md"))
        if project_context is None
        else project_context
    )
    if execution_context is not None:
        run_id = execution_context.run_id
        execution_context = execution_context.model_copy(
            update={
                "global_context": global_ctx,
                "project_context": project_ctx,
                **(
                    {"system_prompt_override": system_prompt_override}
                    if system_prompt_override is not None
                    else {}
                ),
                **(
                    {"tool_whitelist": tool_whitelist}
                    if tool_whitelist is not None
                    else {}
                ),
            }
        )
    else:
        run_id = run_id or new_run_id()
        execution_context = ExecutionContext.isolated(
            goal=goal,
            run_id=run_id,
            global_context=global_ctx,
            project_context=project_ctx,
            system_prompt_override=system_prompt_override,
            tool_whitelist=tool_whitelist,
        )
    run_dir = _run_dir_for_execution_context(config, execution_context)
    run_dir.mkdir(parents=True, exist_ok=False)

    event_bus: EventBus[AgentEvent] = EventBus()
    for listener in listeners:
        event_bus.subscribe(listener)

    workspace_dir = run_dir
    tasks_dir = workspace_dir / ".tasks"
    task_manager = TaskManager(tasks_dir)
    tools = _build_registry(
        task_manager,
        workspace_root=Path.cwd(),
        notes_path=_notes_path_for_execution_context(config, execution_context),
        run_id=run_id,
        permission_manager=permission_manager,
        event_bus=event_bus,
        session_id=execution_context.session_id,
        tool_whitelist=execution_context.tool_whitelist,
        mcp_manager=mcp_manager,
    )
    loop_controller = LoopController(max_iterations=config.agent_max_iterations)
    working_memory = WorkingMemory.from_execution_context(
        execution_context,
        max_steps=config.agent_max_iterations,
    )
    timeline_path = run_dir / "events.jsonl"
    trace_path = run_dir / "trace.jsonl"
    return RunContext(
        run_id=run_id,
        run_dir=run_dir,
        workspace_dir=workspace_dir,
        tasks_dir=tasks_dir,
        timeline_path=timeline_path,
        trace_path=trace_path,
        config=config,
        session_id=execution_context.session_id,
        event_bus=event_bus,
        working_memory=working_memory,
        tools=tools,
        loop_controller=loop_controller,
        permission_manager=permission_manager,
        tool_whitelist=execution_context.tool_whitelist,
        background_tasks=BackgroundTaskRegistry(),
        mcp_manager=mcp_manager,
    )


def _build_registry(
    task_manager: TaskManager,
    *,
    workspace_root: Path,
    notes_path: Path | None = None,
    run_id: str | None = None,
    permission_manager: PermissionManager | None = None,
    event_bus: EventBus[AgentEvent] | None = None,
    session_id: str | None = None,
    tool_whitelist: Sequence[str] | None = None,
    mcp_manager: McpServerManager | None = None,
) -> ToolRegistry:
    """Build the agent toolbox with one shared task manager instance."""

    allowed: set[str] | None = set(tool_whitelist) if tool_whitelist else None

    def _ok(name: str) -> bool:
        return allowed is None or name in allowed

    candidates: list[BaseTool] = [
        ReadFileTool(root=workspace_root),
        WriteFileTool(root=workspace_root),
        ListDirTool(root=workspace_root),
        BashTool(cwd=workspace_root),
        TaskCreateTool(task_manager),
        TaskUpdateTool(task_manager),
        TaskListTool(task_manager),
        TaskGetTool(task_manager),
    ]
    tools = [tool for tool in candidates if _ok(tool.name)]
    if notes_path is not None:
        session_id = notes_path.parent.name
        note_save = NoteSaveTool(
            session_id=session_id,
            notes_path=notes_path,
            run_id=run_id or "manual",
        )
        if _ok(note_save.name):
            tools.append(note_save)
    if mcp_manager is not None:
        tools.extend(
            mcp_tool for mcp_tool in mcp_manager.get_tools() if _ok(mcp_tool.name)
        )

    return ToolRegistry(
        tools,
        timeout_seconds=130.0,
        permission_manager=permission_manager,
        event_bus=event_bus,
        run_id=run_id,
        session_id=session_id,
    )


def _notes_path_for_execution_context(
    config: AppConfig,
    execution_context: ExecutionContext,
) -> Path | None:
    if execution_context.session_id is None:
        return None
    return config.runs_dir / "sessions" / execution_context.session_id / "notes.md"


def _run_dir_for_execution_context(
    config: AppConfig,
    execution_context: ExecutionContext,
) -> Path:
    if execution_context.session_id is None:
        return config.runs_dir / execution_context.run_id
    return (
        config.runs_dir
        / "sessions"
        / execution_context.session_id
        / "runs"
        / execution_context.run_id
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
    _register_subagent_tools(context, llm_client)
    session_dir = (
        context.config.runs_dir / "sessions" / context.session_id
        if context.session_id is not None
        else context.run_dir
    )
    compactor = Compactor(
        context.event_bus,
        session_dir,
        context.session_id or "",
        llm_client_factory=lambda: create_llm_client(
            context.config,
            event_handler=None,
            run_id="compact",
        ),
    )
    return Agent(
        llm_client=llm_client,
        tools=context.tools,
        working_memory=context.working_memory,
        loop_controller=context.loop_controller,
        event_handler=context.event_bus.publish,
        compactor=compactor,
        compact_threshold=context.config.compaction.auto_threshold,
    )


def _register_subagent_tools(context: RunContext, llm_client: LLMClient) -> None:
    allowed = set(context.tool_whitelist) if context.tool_whitelist else None

    def build_child_registry(
        child_bus: EventBus[AgentEvent],
        child_run_id: str,
        child_depth: int,
        profile: object,
    ) -> ToolRegistry:
        profile_allowed_tools = getattr(profile, "allowed_tools", None)
        child_registry = _build_registry(
            TaskManager(context.config.runs_dir / child_run_id / ".tasks"),
            workspace_root=Path.cwd(),
            run_id=child_run_id,
            permission_manager=context.permission_manager,
            event_bus=child_bus,
            tool_whitelist=profile_allowed_tools,
            mcp_manager=context.mcp_manager,
        )
        child_allowed = set(profile_allowed_tools) if profile_allowed_tools else None
        if child_allowed is None or "spawn_agent" in child_allowed:
            child_registry.register(
                SpawnAgentTool(
                    llm_client=llm_client,
                    workspace_root=Path.cwd(),
                    max_steps=context.config.agent_max_iterations,
                    parent_event_bus=child_bus,
                    parent_run_id=child_run_id,
                    depth=child_depth,
                    background_registry=context.background_tasks,
                    child_registry_builder=build_child_registry,
                    run_id_factory=new_run_id,
                )
            )
        if child_allowed is None or "agent_result" in child_allowed:
            child_registry.register(
                AgentResultTool(background_registry=context.background_tasks)
            )
        return child_registry

    if allowed is None or "spawn_agent" in allowed:
        context.tools.register(
            SpawnAgentTool(
                llm_client=llm_client,
                workspace_root=Path.cwd(),
                max_steps=context.config.agent_max_iterations,
                parent_event_bus=context.event_bus,
                parent_run_id=context.run_id,
                depth=0,
                background_registry=context.background_tasks,
                child_registry_builder=build_child_registry,
                run_id_factory=new_run_id,
            )
        )
    if allowed is None or "agent_result" in allowed:
        context.tools.register(AgentResultTool(background_registry=context.background_tasks))


async def _handle_run_interruption(context: RunContext) -> None:
    if context.working_memory.status != RunStatus.CANCELLED:
        context.working_memory.set_status(
            RunStatus.CANCELLED,
            reason="agent run cancelled",
            transition_reason="runner caught cancellation or interruption",
        )
        await context.event_bus.publish(
            RunCancelledEvent(
                run_id=context.run_id,
                reason=context.working_memory.status_transition_reason,
            )
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
