from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from my_claude.cli.commands.ping import _ping, format_ping_output
from my_claude.core.app import register_routes
from my_claude.core.bus.command import CORE_PING_METHOD, BusResult, CorePingResult, ping_result
from my_claude.core.bus.envelope import JsonRpcRequest
from my_claude.core.transport.socket_server import TCPConnection, TCPServer, current_tcp_connection


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


def test_route_handler_can_read_current_tcp_connection() -> None:
    assert asyncio.run(_route_reads_current_tcp_connection())


def test_tcp_connection_close_callbacks_are_immediate_and_idempotent() -> None:
    calls = asyncio.run(_close_tcp_connection_twice_and_register_late_callback())

    assert calls == ["first", "late"]


async def _run_ping_roundtrip() -> tuple[CorePingResult, float]:
    server = TCPServer("127.0.0.1", 0, max_request_bytes=65536)
    register_routes(server, started_at=time.perf_counter(), server_version="test-version")
    await server.start()

    try:
        port = _bound_port(server)
        return await _ping("127.0.0.1", port, timeout=1.0)
    finally:
        await server.shutdown()


async def _close_tcp_connection_twice_and_register_late_callback() -> list[str]:
    calls: list[str] = []
    writer = FakeStreamWriter()
    connection = TCPConnection(writer)  # type: ignore[arg-type]

    connection.add_close_callback(lambda: calls.append("first"))
    await connection.close()
    await connection.close()
    connection.add_close_callback(lambda: calls.append("late"))

    return calls


async def _route_reads_current_tcp_connection() -> bool:
    server = TCPServer("127.0.0.1", 0, max_request_bytes=65536)
    saw_connection = False

    async def handle_core_ping(_request: JsonRpcRequest) -> BusResult:
        nonlocal saw_connection
        current_tcp_connection()
        saw_connection = True
        return ping_result(uptime_seconds=0, server_version="test-version")

    server.register(CORE_PING_METHOD, handle_core_ping)
    await server.start()

    try:
        await _ping("127.0.0.1", _bound_port(server), timeout=1.0)
        return saw_connection
    finally:
        await server.shutdown()


def _bound_port(server: TCPServer) -> int:
    if server._server is None or not server._server.sockets:
        raise RuntimeError("test server did not start")

    return int(server._server.sockets[0].getsockname()[1])


class FakeStreamWriter:
    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass

    def write(self, _data: bytes) -> None:
        pass

    async def drain(self) -> None:
        pass

    def get_extra_info(self, _name: str) -> Any:
        return None
