from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from my_claude.core.trace import (
    TraceRecord,
    TraceWriter,
    TraceWriterBackpressure,
    serialize_trace_record,
)


def test_serialize_trace_record_writes_compact_json() -> None:
    record = _record(sequence=1, data={"goal": "ship it"})

    payload = json.loads(serialize_trace_record(record))

    assert payload["direction"] == "CORE"
    assert payload["layer"] == "event"
    assert payload["kind"] == "event"
    assert payload["step"] == 1
    assert payload["data"] == {"goal": "ship it"}


def test_trace_writer_persists_all_accepted_records_on_stop(tmp_path: Path) -> None:
    asyncio.run(_write_many_records(tmp_path / "trace.jsonl", count=500))


async def _write_many_records(path: Path, *, count: int) -> None:
    writer = TraceWriter(path, batch_size=32)
    await writer.start()

    for sequence in range(count):
        writer.emit(_record(sequence=sequence, data={"value": sequence}))

    await writer.stop()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == count
    assert [json.loads(line)["step"] for line in lines] == list(range(count))


def test_trace_writer_async_context_stops_without_losing_records(tmp_path: Path) -> None:
    asyncio.run(_write_with_context(tmp_path / "trace.jsonl"))


async def _write_with_context(path: Path) -> None:
    async with TraceWriter(path, batch_size=2) as writer:
        writer.emit(_record(sequence=1))
        writer.emit(_record(sequence=2))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["step"] for line in lines] == [1, 2]


def test_trace_writer_rejects_emit_outside_running_lifecycle(tmp_path: Path) -> None:
    asyncio.run(_reject_emit_outside_lifecycle(tmp_path / "trace.jsonl"))


async def _reject_emit_outside_lifecycle(path: Path) -> None:
    writer = TraceWriter(path)

    with pytest.raises(RuntimeError, match="not started"):
        writer.emit(_record(sequence=1))

    await writer.start()
    writer.emit(_record(sequence=2))
    await writer.stop()

    with pytest.raises(RuntimeError, match="not started"):
        writer.emit(_record(sequence=3))


def test_trace_writer_reports_backpressure_without_dropping_records(tmp_path: Path) -> None:
    asyncio.run(_report_backpressure(tmp_path / "trace.jsonl"))


async def _report_backpressure(path: Path) -> None:
    release = asyncio.Event()
    writer = PausedTraceWriter(path, release, max_queue_size=1)
    await writer.start()

    writer.emit(_record(sequence=1))
    with pytest.raises(TraceWriterBackpressure, match="queue is full"):
        writer.emit(_record(sequence=2))

    release.set()
    await writer.stop()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["step"] for line in lines] == [1]


class PausedTraceWriter(TraceWriter):
    def __init__(self, path: Path, release: asyncio.Event, *, max_queue_size: int) -> None:
        super().__init__(path, max_queue_size=max_queue_size)
        self._release = release

    async def _drain(self) -> None:
        await self._release.wait()
        await super()._drain()


def _record(sequence: int, data: dict[str, object] | None = None) -> TraceRecord:
    return TraceRecord(
        ts="2026-06-12T12:00:00+00:00",
        direction="CORE",
        layer="event",
        kind="event",
        step=sequence,
        data=data or {},
    )
