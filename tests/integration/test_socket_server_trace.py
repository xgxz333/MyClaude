from __future__ import annotations

import asyncio
from typing import Any

from my_claude.core.bus.command import CORE_PING_METHOD, BusResult, ping_result
from my_claude.core.bus.envelope import (
    EventPushEnvelope,
    JsonRpcErrorCode,
    JsonRpcRequest,
    make_error_response,
    make_notification,
    make_success_response,
    to_ndjson,
)
from my_claude.core.trace import TraceRecord
from my_claude.core.transport.socket_server import TCPConnection, TCPServer


class RecordingTraceEmitter:
    def __init__(self) -> None:
        self.records: list[TraceRecord] = []
        self.on_emit: Any = None

    def emit(self, record: TraceRecord) -> None:
        if self.on_emit is not None:
            self.on_emit(record)
        self.records.append(record)


def test_socket_server_traces_received_command_with_client_identity() -> None:
    records = asyncio.run(_dispatch_with_trace())

    assert len(records) == 1
    record = records[0]
    assert record.layer == "ipc"
    assert record.direction == "CLIENT→CORE"
    assert record.kind == "command"
    assert record.client_id == "127.0.0.1:43123"
    assert record.data["method"] == "core.ping"
    assert record.data["id"] == 7
    assert record.data["params"] == {}
    byte_count = record.data["bytes"]
    assert isinstance(byte_count, int)
    assert byte_count > 0


async def _dispatch_with_trace() -> list[TraceRecord]:
    trace = RecordingTraceEmitter()
    server = TCPServer("127.0.0.1", 0, max_request_bytes=65536, trace_emitter=trace)
    connection = TCPConnection(FakeStreamWriter(), trace_emitter=trace)  # type: ignore[arg-type]

    async def handle_core_ping(_request: JsonRpcRequest) -> BusResult:
        return ping_result(uptime_seconds=0, server_version="test-version")

    server.register(CORE_PING_METHOD, handle_core_ping)
    line = to_ndjson(JsonRpcRequest(id=7, method=CORE_PING_METHOD, params={}))
    await server.dispatch(line, connection)
    return trace.records


def test_socket_connection_traces_response_after_successful_drain() -> None:
    trace = RecordingTraceEmitter()
    writer = FakeStreamWriter()

    def assert_drained_before_trace(_record: TraceRecord) -> None:
        assert writer.drain_count == 1

    trace.on_emit = assert_drained_before_trace

    async def write_response() -> list[TraceRecord]:
        connection = TCPConnection(writer, trace_emitter=trace)  # type: ignore[arg-type]
        await connection.write_model(make_success_response(7, {"pong": "pong"}))
        return trace.records

    records = asyncio.run(write_response())

    assert len(records) == 1
    record = records[0]
    assert record.kind == "response"
    assert record.direction == "CORE→CLIENT"
    assert record.client_id == "127.0.0.1:43123"
    assert record.data["id"] == 7
    assert record.data["result"] == {"pong": "pong"}


def test_socket_connection_traces_dynamic_outbound_kinds() -> None:
    records = asyncio.run(_write_dynamic_kinds())

    assert [record.kind for record in records] == ["error"]
    assert records[0].data["id"] == 9


async def _write_dynamic_kinds() -> list[TraceRecord]:
    trace = RecordingTraceEmitter()
    connection = TCPConnection(FakeStreamWriter(), trace_emitter=trace)  # type: ignore[arg-type]

    await connection.write_model(make_error_response(9, JsonRpcErrorCode.INVALID_PARAMS))
    await connection.write_notification(make_notification("event.publish", {"sequence": 1}))
    await connection.write_event(EventPushEnvelope(event={"type": "run.started"}))
    return trace.records


class FakeStreamWriter:
    def __init__(self) -> None:
        self.values: list[bytes] = []
        self.drain_count = 0
        self.closed = False

    def write(self, data: bytes) -> None:
        self.values.append(data)

    async def drain(self) -> None:
        self.drain_count += 1

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass

    def get_extra_info(self, name: str) -> Any:
        if name == "peername":
            return ("127.0.0.1", 43123)
        return None
