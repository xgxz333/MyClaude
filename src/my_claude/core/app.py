from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import time

from my_claude.core.bus.command import CORE_PING_METHOD, BusResult, get_server_version, ping_result
from my_claude.core.bus.envelope import JsonRpcRequest
from my_claude.core.config import load_config
from my_claude.core.logging_setup import setup_logging
from my_claude.core.transport.socket_server import TCPServer


def register_routes(
    server: TCPServer,
    *,
    started_at: float | None = None,
    server_version: str | None = None,
) -> None:
    start_time = started_at if started_at is not None else time.perf_counter()
    version = server_version or get_server_version()

    async def handle_core_ping(_request: JsonRpcRequest) -> BusResult:
        return ping_result(
            uptime_seconds=time.perf_counter() - start_time,
            server_version=version,
        )

    server.register(CORE_PING_METHOD, handle_core_ping)


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
    register_routes(server, started_at=started_at)

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
