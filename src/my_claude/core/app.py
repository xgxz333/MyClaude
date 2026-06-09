"""Core server entrypoint that starts the TCP IPC runtime."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import time

from my_claude.core.bus.command import (
    AGENT_RUN_METHOD,
    CORE_PING_METHOD,
    EVENT_SUBSCRIBE_METHOD,
    AgentRunCommand,
    AgentRunResult,
    BusResult,
    EventSubscribeCommand,
    EventSubscribeResult,
    command_from_request,
    get_server_version,
    ping_result,
)
from my_claude.core.bus.envelope import JsonRpcRequest
from my_claude.core.config import AppConfig, load_config
from my_claude.core.logging_setup import setup_logging
from my_claude.core.runner import RunResult, prepare_run_context, run_prepared_context
from my_claude.core.transport.ipc_broadcaster import IpcEventBroadcaster
from my_claude.core.transport.socket_server import TCPServer, current_tcp_connection

logger = logging.getLogger(__name__)


def register_routes(
    server: TCPServer,
    *,
    config: AppConfig | None = None,
    started_at: float | None = None,
    server_version: str | None = None,
) -> None:
    start_time = started_at if started_at is not None else time.perf_counter()
    version = server_version or get_server_version()
    runtime_config = config or AppConfig()
    event_broadcaster = IpcEventBroadcaster(runtime_config.runs_dir)
    run_tasks: set[asyncio.Task[RunResult]] = set()

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
        await event_broadcaster.replay(
            connection,
            replay_from=command.params.replay_from,
            event_types=command.params.event_types,
            run_id=command.params.run_id,
        )
        subscription_id = event_broadcaster.subscribe(
            connection,
            event_types=command.params.event_types,
            run_id=command.params.run_id,
        )
        return EventSubscribeResult(
            subscription_id=subscription_id,
            next_sequence=event_broadcaster.next_sequence,
        )

    async def handle_agent_run(request: JsonRpcRequest) -> BusResult:
        command = command_from_request(request)
        if not isinstance(command, AgentRunCommand):
            raise ValueError("agent.run params are invalid")

        context = prepare_run_context(command.params.goal, config=runtime_config)
        run_task = asyncio.create_task(
            run_prepared_context(
                context,
                listeners=[event_broadcaster.handle],
            )
        )
        run_tasks.add(run_task)
        run_task.add_done_callback(lambda task: _finish_background_run(task, run_tasks))
        return AgentRunResult(
            run_id=context.run_id,
            goal=context.working_memory.goal,
            timeline_path=str(context.timeline_path),
        )

    server.register(CORE_PING_METHOD, handle_core_ping)
    server.register(EVENT_SUBSCRIBE_METHOD, handle_event_subscribe)
    server.register(AGENT_RUN_METHOD, handle_agent_run)


def _finish_background_run(
    task: asyncio.Task[RunResult],
    run_tasks: set[asyncio.Task[RunResult]],
) -> None:
    run_tasks.discard(task)
    if task.cancelled():
        return

    try:
        task.result()
    except Exception:
        logger.exception("background agent run failed")


async def _run_async() -> None:
    started_at = time.perf_counter()
    config = load_config()
    setup_logging(config)
    shutdown_event = asyncio.Event()
    server = TCPServer(
        config.core_host,
        config.core_port,
        max_request_bytes=config.max_request_bytes,
    )
    register_routes(server, config=config, started_at=started_at)

    logger = logging.getLogger(__name__)
    logger.info("core routes registered: %s", ", ".join(server.routes))
    logger.info("core initialized in %.2fms", (time.perf_counter() - started_at) * 1000)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown_event.set)
        except NotImplementedError:
            pass

    await server.serve_until_stopped(shutdown_event)


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
