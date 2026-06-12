"""Asynchronous JSONL writer for trace records."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TextIO, cast

from my_claude.core.trace.record import TraceRecord

_STOP = object()


class TraceWriterBackpressure(RuntimeError):
    """Raised when a trace record cannot be accepted without blocking."""


class TraceWriter:
    """High-throughput producer-consumer writer for trace records.

    ``emit`` is intentionally synchronous and non-blocking: once it returns, the
    record has been accepted by the in-memory queue and ``stop`` will persist it
    before returning.
    """

    def __init__(
        self,
        path: Path,
        *,
        max_queue_size: int = 10_000,
        batch_size: int = 128,
    ) -> None:
        if max_queue_size < 0:
            raise ValueError("max_queue_size must be greater than or equal to 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        self._path = path
        self._batch_size = batch_size
        self._queue: asyncio.Queue[TraceRecord | object] = asyncio.Queue(maxsize=max_queue_size)
        self._file: TextIO | None = None
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._stop_lock = asyncio.Lock()

    async def __aenter__(self) -> TraceWriter:
        await self.start()
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.stop()

    async def start(self) -> None:
        """Open the target file and start the background drain task."""

        if self._task is not None:
            if self._task.done():
                await self._task
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")
        self._stopping = False
        self._task = asyncio.create_task(self._drain(), name=f"trace-writer:{self._path}")

    def emit(self, record: TraceRecord) -> None:
        """Accept a trace record for asynchronous persistence without awaiting I/O."""

        task = self._task
        if task is None:
            raise RuntimeError("trace writer is not started")
        if self._stopping:
            raise RuntimeError("trace writer is stopping")
        if task.done():
            self._raise_task_failure(task)

        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull as error:
            raise TraceWriterBackpressure("trace writer queue is full") from error

    async def stop(self) -> None:
        """Flush every accepted record and stop the background task."""

        async with self._stop_lock:
            task = self._task
            if task is None:
                return

            self._stopping = True
            await self._send_stop(task)

            try:
                await task
            finally:
                self._close_file()
                self._task = None
                self._stopping = False

    async def _send_stop(self, task: asyncio.Task[None]) -> None:
        while True:
            if task.done():
                await task
                return
            try:
                self._queue.put_nowait(_STOP)
                return
            except asyncio.QueueFull:
                await asyncio.sleep(0)

    async def _drain(self) -> None:
        batch: list[TraceRecord] = []

        while True:
            item = await self._queue.get()
            try:
                if item is _STOP:
                    await self._write_batch(batch)
                    return

                batch.append(cast(TraceRecord, item))
                if await self._collect_ready_records(batch):
                    return

                if len(batch) >= self._batch_size or self._queue.empty():
                    await self._write_batch(batch)
            finally:
                self._queue.task_done()

    async def _collect_ready_records(self, batch: list[TraceRecord]) -> bool:
        while len(batch) < self._batch_size:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return False

            try:
                if item is _STOP:
                    await self._write_batch(batch)
                    return True

                batch.append(cast(TraceRecord, item))
            finally:
                self._queue.task_done()

        return False

    async def _write_batch(self, batch: list[TraceRecord]) -> None:
        if not batch:
            return

        lines = [f"{serialize_trace_record(record)}\n" for record in batch]
        await asyncio.to_thread(self._write_lines, lines)
        batch.clear()

    def _write_lines(self, lines: list[str]) -> None:
        if self._file is None:
            raise RuntimeError("trace file is not open")

        self._file.writelines(lines)
        self._file.flush()

    def _close_file(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def _raise_task_failure(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            raise RuntimeError("trace writer background task was cancelled")
        error = task.exception()
        if error is not None:
            raise RuntimeError("trace writer background task failed") from error
        raise RuntimeError("trace writer is stopped")


def serialize_trace_record(record: TraceRecord) -> str:
    """Serialize one trace record as a compact JSON object."""

    return record.model_dump_json(exclude_none=True)
