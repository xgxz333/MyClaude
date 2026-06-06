from __future__ import annotations

import asyncio
import json
import time

from my_claude.cli.commands.ping import _ping, format_ping_output
from my_claude.core.app import register_routes
from my_claude.core.bus.command import CorePingResult
from my_claude.core.transport.socket_server import TCPServer


def test_ping_roundtrip() -> None:
    result, elapsed_ms = asyncio.run(_run_ping_roundtrip())

    assert result.pong == "pong"
    assert result.uptime_seconds >= 0
    assert result.server_version == "test-version"
    assert elapsed_ms >= 0


def test_ping_output_contains_uptime_latency_pong_and_server_version() -> None:
    result = CorePingResult(
        pong="pong",
        uptime_seconds=12.345,
        server_version="test-version",
    )

    output = json.loads(format_ping_output(result, latency_ms=1.234))

    assert output == {
        "pong": "pong",
        "uptime_seconds": 12.345,
        "server_version": "test-version",
        "latency_ms": 1.23,
    }


async def _run_ping_roundtrip() -> tuple[CorePingResult, float]:
    server = TCPServer("127.0.0.1", 0, max_request_bytes=65536)
    register_routes(server, started_at=time.perf_counter(), server_version="test-version")
    await server.start()

    try:
        port = _bound_port(server)
        return await _ping("127.0.0.1", port, timeout=1.0)
    finally:
        await server.shutdown()


def _bound_port(server: TCPServer) -> int:
    if server._server is None or not server._server.sockets:
        raise RuntimeError("test server did not start")

    return int(server._server.sockets[0].getsockname()[1])
