"""Core server entrypoint that starts the TCP IPC runtime."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import time
from collections.abc import Sequence
from pathlib import Path

from my_claude.agent.events import AgentEvent, EventHandler
from my_claude.core.bus.command import (
    AGENT_RUN_METHOD,
    CORE_PING_METHOD,
    EVENT_SUBSCRIBE_METHOD,
    PERMISSION_RESPOND_METHOD,
    SESSION_CLOSE_METHOD,
    SESSION_COMPACT_METHOD,
    SESSION_CREATE_METHOD,
    SESSION_GET_HISTORY_METHOD,
    SESSION_MESSAGE_METHOD,
    SESSION_SEND_MESSAGE_METHOD,
    AgentRunCommand,
    AgentRunResult,
    BusResult,
    EventSubscribeCommand,
    EventSubscribeResult,
    PermissionRespondCommand,
    PermissionRespondResult,
    SessionCloseCommand,
    SessionCloseResult,
    SessionCompactCommand,
    SessionCompactResult,
    SessionCreateCommand,
    SessionCreateResult,
    SessionGetHistoryCommand,
    SessionGetHistoryResult,
    SessionMessageCommand,
    SessionMessageResult,
    SessionSendMessageCommand,
    SessionSendMessageResult,
    command_from_request,
    get_server_version,
    ping_result,
)
from my_claude.core.bus.envelope import JsonRpcRequest
from my_claude.core.config import AppConfig, load_config
from my_claude.core.events.bus import EventBus
from my_claude.core.logging_setup import setup_logging
from my_claude.core.permissions.manager import PermissionManager
from my_claude.core.runner import (
    RunContext,
    RunResult,
    new_run_id,
    prepare_run_context,
    run_prepared_context,
)
from my_claude.core.session import SessionManager, SessionMessageOutcome
from my_claude.core.trace.paths import DAEMON_TRACE_FILENAME
from my_claude.core.trace.writer import TraceWriter
from my_claude.core.transport.ipc_broadcaster import IpcEventBroadcaster
from my_claude.core.transport.socket_server import TCPServer, current_tcp_connection
from my_claude.llm.client import create_llm_client

logger = logging.getLogger(__name__)


def register_routes(
    server: TCPServer,
    *,
    config: AppConfig | None = None,
    started_at: float | None = None,
    server_version: str | None = None,
    run_tasks: set[asyncio.Task[RunResult]] | None = None,
    trace_writer: TraceWriter | None = None,
) -> None:
    start_time = started_at if started_at is not None else time.perf_counter()
    version = server_version or get_server_version()
    runtime_config = config or AppConfig()
    trace_emitter = trace_writer if runtime_config.trace_enabled else None
    event_broadcaster = IpcEventBroadcaster(
        runtime_config.runs_dir,
        trace_emitter=trace_emitter,
    )
    daemon_event_bus: EventBus[AgentEvent] = EventBus()
    daemon_event_bus.subscribe(event_broadcaster.handle)
    permission_manager = PermissionManager(
        policy_file=Path("~/.myclaude/policy.toml").expanduser(),
        timeout_s=runtime_config.permission_timeout_s,
    )
    active_run_tasks: set[asyncio.Task[RunResult]] = run_tasks if run_tasks is not None else set()
    session_manager = SessionManager(
        runtime_config.runs_dir,
        daemon_event_bus,
        compact_provider_factory=lambda: create_llm_client(
            runtime_config,
            event_handler=None,
            run_id="compact",
        ),
    )

    async def handle_core_ping(_request: JsonRpcRequest) -> BusResult:
        return ping_result(
            uptime_seconds=time.perf_counter() - start_time,
            server_version=version,
        )

    async def handle_event_subscribe(request: JsonRpcRequest) -> BusResult:
        command = command_from_request(request)
        if not isinstance(command, EventSubscribeCommand):
            raise ValueError("event.subscribe params are invalid")

        connection = current_tcp_connection()
        replayed_count = await event_broadcaster.replay(
            connection,
            replay_from=command.params.replay_from,
            replay_from_run=command.params.replay_from_run,
            topics=command.params.topics,
            scope=command.params.scope,
            event_types=command.params.event_types,
            run_id=command.params.run_id,
        )
        subscription_id = event_broadcaster.subscribe(
            connection,
            topics=command.params.topics,
            scope=command.params.scope,
            event_types=command.params.event_types,
            run_id=command.params.run_id,
        )
        return EventSubscribeResult(
            subscription_id=subscription_id,
            next_sequence=event_broadcaster.next_sequence,
            replayed_count=replayed_count,
        )

    async def handle_agent_run(request: JsonRpcRequest) -> BusResult:
        command = command_from_request(request)
        if not isinstance(command, AgentRunCommand):
            raise ValueError("agent.run params are invalid")

        goal = command.params.goal.strip()
        if not goal:
            raise ValueError("agent.run goal must not be empty")

        message_outcome, context = await _prepare_one_shot_session_run(
            goal,
            session_manager=session_manager,
            config=runtime_config,
            permission_manager=permission_manager,
        )
        run_task = _schedule_background_run(
            context,
            listeners=[event_broadcaster.handle],
            trace_writer=trace_emitter,
            run_tasks=active_run_tasks,
        )
        _persist_session_result_when_done(
            run_task,
            session_manager=session_manager,
            message_outcome=message_outcome,
        )
        return AgentRunResult(
            run_id=context.run_id,
            goal=context.working_memory.goal,
            timeline_path=str(context.timeline_path),
        )

    async def handle_session_create(request: JsonRpcRequest) -> BusResult:
        return await _session_create_handler(
            request,
            session_manager=session_manager,
        )

    async def handle_session_message(request: JsonRpcRequest) -> BusResult:
        command = command_from_request(request)
        if not isinstance(command, SessionMessageCommand):
            raise ValueError("session.message params are invalid")

        run_id = new_run_id()
        message_outcome = await session_manager.start_message(
            session_id=command.params.session_id,
            message=command.params.message,
            run_id=run_id,
        )
        context = prepare_run_context(
            message_outcome.execution_context.goal,
            config=runtime_config,
            execution_context=message_outcome.execution_context,
            permission_manager=permission_manager,
        )
        run_task = _schedule_background_run(
            context,
            listeners=[event_broadcaster.handle],
            trace_writer=trace_emitter,
            run_tasks=active_run_tasks,
        )
        _persist_session_result_when_done(
            run_task,
            session_manager=session_manager,
            message_outcome=message_outcome,
        )
        return SessionMessageResult(
            session_id=message_outcome.session.session_id,
            turn=message_outcome.turn,
            run_id=context.run_id,
            timeline_path=str(context.timeline_path),
        )

    async def handle_session_send_message(request: JsonRpcRequest) -> BusResult:
        command = command_from_request(request)
        if not isinstance(command, SessionSendMessageCommand):
            raise ValueError("session.send_message params are invalid")

        run_id = new_run_id()
        message_outcome = await session_manager.start_message(
            session_id=command.params.session_id,
            message=command.params.content,
            run_id=run_id,
        )
        context = prepare_run_context(
            message_outcome.execution_context.goal,
            config=runtime_config,
            execution_context=message_outcome.execution_context,
            permission_manager=permission_manager,
        )
        run_task = _schedule_background_run(
            context,
            listeners=[event_broadcaster.handle],
            trace_writer=trace_emitter,
            run_tasks=active_run_tasks,
        )
        _persist_session_result_when_done(
            run_task,
            session_manager=session_manager,
            message_outcome=message_outcome,
        )
        return SessionSendMessageResult(run_id=context.run_id)

    async def handle_session_get_history(request: JsonRpcRequest) -> BusResult:
        command = command_from_request(request)
        if not isinstance(command, SessionGetHistoryCommand):
            raise ValueError("session.get_history params are invalid")

        return SessionGetHistoryResult(
            messages=await session_manager.get_history(command.params.session_id)
        )

    async def handle_session_compact(request: JsonRpcRequest) -> BusResult:
        command = command_from_request(request)
        if not isinstance(command, SessionCompactCommand):
            raise ValueError("session.compact params are invalid")

        outcome = await session_manager.compact(
            command.params.session_id,
            focus=command.params.focus,
        )
        return SessionCompactResult(
            summary_tokens=outcome.summary_tokens,
            saved_tokens=outcome.saved_tokens,
        )

    async def handle_session_close(request: JsonRpcRequest) -> BusResult:
        command = command_from_request(request)
        if not isinstance(command, SessionCloseCommand):
            raise ValueError("session.close params are invalid")

        permission_manager.cancel_session(command.params.session_id)
        await session_manager.close(command.params.session_id)
        return SessionCloseResult()

    async def handle_permission_respond(request: JsonRpcRequest) -> BusResult:
        params = request.params if isinstance(request.params, dict) else {}
        return await _permission_respond_handler(
            params,
            permission_manager=permission_manager,
        )

    server.register(CORE_PING_METHOD, handle_core_ping)
    server.register(EVENT_SUBSCRIBE_METHOD, handle_event_subscribe)
    server.register(AGENT_RUN_METHOD, handle_agent_run)
    server.register(SESSION_CREATE_METHOD, handle_session_create)
    server.register(SESSION_MESSAGE_METHOD, handle_session_message)
    server.register(SESSION_SEND_MESSAGE_METHOD, handle_session_send_message)
    server.register(SESSION_GET_HISTORY_METHOD, handle_session_get_history)
    server.register(SESSION_COMPACT_METHOD, handle_session_compact)
    server.register(SESSION_CLOSE_METHOD, handle_session_close)
    server.register(PERMISSION_RESPOND_METHOD, handle_permission_respond)


async def _prepare_one_shot_session_run(
    goal: str,
    *,
    session_manager: SessionManager,
    config: AppConfig,
    permission_manager: PermissionManager | None = None,
) -> tuple[SessionMessageOutcome, RunContext]:
    """Adapt legacy `agent.run` requests onto the session-backed execution path."""

    run_id = new_run_id()
    create_outcome = await session_manager.create(mode="one_shot", title=goal[:40])
    message_outcome = await session_manager.start_message(
        session_id=create_outcome.session.session_id,
        message=goal,
        run_id=run_id,
    )
    context = prepare_run_context(
        message_outcome.execution_context.goal,
        config=config,
        execution_context=message_outcome.execution_context,
        permission_manager=permission_manager,
    )
    return message_outcome, context


async def _permission_respond_handler(
    params: dict[str, object],
    *,
    permission_manager: PermissionManager,
) -> PermissionRespondResult:
    cmd = PermissionRespondCommand.model_validate(params)
    permission_manager.respond(cmd.tool_use_id, cmd.decision)
    return PermissionRespondResult()


def _persist_session_result_when_done(
    run_task: asyncio.Task[RunResult],
    *,
    session_manager: SessionManager,
    message_outcome: SessionMessageOutcome,
) -> None:
    run_task.add_done_callback(
        lambda task: asyncio.create_task(
            session_manager.finish_message(
                session_id=message_outcome.session.session_id,
                task=task,
                prefill_messages=message_outcome.prefill_messages,
            )
        )
    )


async def _session_create_handler(
    request: JsonRpcRequest,
    *,
    session_manager: SessionManager,
) -> BusResult:
    command = command_from_request(request)
    if not isinstance(command, SessionCreateCommand):
        raise ValueError("session.create params are invalid")

    outcome = await session_manager.create(
        mode=command.params.mode,
        title=command.params.title,
    )
    session = outcome.session
    logger.info("created chat session: session_id=%s path=%s", session.session_id, outcome.path)
    return SessionCreateResult(
        session_id=session.session_id,
        status=session.status,
        title=session.title,
    )


def _schedule_background_run(
    context: RunContext,
    *,
    listeners: Sequence[EventHandler],
    trace_writer: TraceWriter | None,
    run_tasks: set[asyncio.Task[RunResult]],
) -> asyncio.Task[RunResult]:
    run_task = asyncio.create_task(
        run_prepared_context(
            context,
            listeners=listeners,
            trace_writer=trace_writer,
        ),
        name=f"myclaude-run-{context.run_id}",
    )
    run_tasks.add(run_task)
    run_task.add_done_callback(
        lambda task: _finish_background_run(
            task,
            run_tasks,
            run_id=context.run_id,
        )
    )
    logger.info("accepted agent run: run_id=%s", context.run_id)
    return run_task


def _finish_background_run(
    task: asyncio.Task[RunResult],
    run_tasks: set[asyncio.Task[RunResult]],
    *,
    run_id: str | None = None,
) -> None:
    run_tasks.discard(task)
    if task.cancelled():
        logger.info("background agent run cancelled: run_id=%s", run_id or "<unknown>")
        return

    try:
        result = task.result()
    except Exception:
        logger.exception("background agent run failed: run_id=%s", run_id or "<unknown>")
    else:
        logger.info("background agent run finished: run_id=%s", result.run_id)


async def _cancel_background_runs(
    run_tasks: set[asyncio.Task[RunResult]],
    *,
    timeout_seconds: float = 2.0,
) -> None:
    active_tasks = {task for task in run_tasks if not task.done()}
    if not active_tasks:
        return

    logger.info("cancelling %d background run task(s)", len(active_tasks))
    for task in active_tasks:
        task.cancel()

    done, pending = await asyncio.wait(active_tasks, timeout=timeout_seconds)
    for task in done:
        if task.cancelled():
            continue
        try:
            task.result()
        except Exception:
            logger.exception("background agent run failed during shutdown")

    if pending:
        logger.warning(
            "core shutdown left %d background run task(s) pending",
            len(pending),
        )


async def _run_async() -> None:
    started_at = time.perf_counter()
    config = load_config()
    setup_logging(config)
    shutdown_event = asyncio.Event()
    trace_writer: TraceWriter | None = None
    if config.trace_enabled:
        trace_writer = TraceWriter(_trace_path(config))
        await trace_writer.start()
    server = TCPServer(
        config.core_host,
        config.core_port,
        max_request_bytes=config.max_request_bytes,
        trace_emitter=trace_writer,
    )
    run_tasks: set[asyncio.Task[RunResult]] = set()
    register_routes(
        server,
        config=config,
        started_at=started_at,
        run_tasks=run_tasks,
        trace_writer=trace_writer,
    )

    logger = logging.getLogger(__name__)
    logger.info("core routes registered: %s", ", ".join(server.routes))
    logger.info("core initialized in %.2fms", (time.perf_counter() - started_at) * 1000)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown_event.set)
        except NotImplementedError:
            pass

    try:
        await server.serve_until_stopped(shutdown_event)
    finally:
        await server.shutdown()
        await _cancel_background_runs(run_tasks)
        if trace_writer is not None:
            await trace_writer.stop()


def _trace_path(config: AppConfig) -> Path:
    return config.trace_file or config.runs_dir / DAEMON_TRACE_FILENAME


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="myclaude-core")
    parser.parse_args(argv)

    try:
        asyncio.run(_run_async())
    except KeyboardInterrupt:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
