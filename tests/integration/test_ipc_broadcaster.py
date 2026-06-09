from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import cast

from my_claude.agent.events import (
    AgentEvent,
    AgentEventType,
    RunCompletedEvent,
    RunStartedEvent,
    StepStartedEvent,
)
from my_claude.core.bus.envelope import EventPushEnvelope
from my_claude.core.events.writer import serialize_event
from my_claude.core.transport.ipc_broadcaster import IpcEventBroadcaster, IpcEventConnection


class RecordingConnection:
    def __init__(self, *, fail_on_write: bool = False) -> None:
        self.events: list[EventPushEnvelope] = []
        self._close_callbacks: list[Callable[[], None]] = []
        self._fail_on_write = fail_on_write
        self._closed = False
        self.write_events_count = 0

    def add_close_callback(self, callback: Callable[[], None]) -> None:
        if self._closed:
            callback()
            return

        self._close_callbacks.append(callback)

    async def write_event(self, event: EventPushEnvelope) -> None:
        if self._fail_on_write:
            raise ConnectionError("client disconnected")
        self.events.append(event)

    async def write_events(self, events: list[EventPushEnvelope]) -> None:
        if self._fail_on_write:
            raise ConnectionError("client disconnected")
        self.write_events_count += 1
        self.events.extend(events)

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        callbacks = tuple(self._close_callbacks)
        self._close_callbacks.clear()
        for callback in callbacks:
            callback()


def test_ipc_broadcaster_filters_by_event_type_and_run_id() -> None:
    topic_connection, scoped_connection = asyncio.run(_broadcast_filtered_events())

    topic_events = [_event_type(event) for event in topic_connection.events]
    scoped_events = [_event_type(event) for event in scoped_connection.events]

    assert topic_events == ["run.started", "run.started"]
    assert scoped_events == ["run.started", "step.started", "run.finished"]


async def _broadcast_filtered_events() -> tuple[RecordingConnection, RecordingConnection]:
    broadcaster = IpcEventBroadcaster()
    topic_connection = RecordingConnection()
    scoped_connection = RecordingConnection()
    broadcaster.subscribe(
        cast(IpcEventConnection, topic_connection),
        event_types=[AgentEventType.RUN_STARTED],
    )
    broadcaster.subscribe(
        cast(IpcEventConnection, scoped_connection),
        run_id="run-1",
    )

    await broadcaster.handle(RunStartedEvent(goal="first", run_id="run-1"))
    await broadcaster.handle(StepStartedEvent(run_id="run-1", step=1))
    await broadcaster.handle(RunStartedEvent(goal="second", run_id="run-2"))
    await broadcaster.handle(RunCompletedEvent(goal="first", run_id="run-1"))

    return topic_connection, scoped_connection


def test_ipc_broadcaster_replays_only_matching_history(tmp_path: Path) -> None:
    connection = asyncio.run(_replay_matching_history(tmp_path))

    assert [_event_type(event) for event in connection.events] == [
        "step.started",
        "run.finished",
    ]
    assert connection.write_events_count == 1


async def _replay_matching_history(tmp_path: Path) -> RecordingConnection:
    runs_dir = tmp_path / "runs"
    _write_history(
        runs_dir,
        "run-1",
        [
            RunStartedEvent(goal="first", run_id="run-1"),
            StepStartedEvent(run_id="run-1", step=1),
            RunCompletedEvent(goal="first", run_id="run-1"),
        ],
    )
    _write_history(
        runs_dir,
        "run-2",
        [
            RunStartedEvent(goal="second", run_id="run-2"),
        ],
    )

    broadcaster = IpcEventBroadcaster(runs_dir)
    connection = RecordingConnection()
    await broadcaster.replay(
        cast(IpcEventConnection, connection),
        replay_from=2,
        run_id="run-1",
    )

    return connection


def test_ipc_broadcaster_replay_rejects_invalid_history_line(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text('{"type":"run.started","data":[]}\n', encoding="utf-8")

    async def replay() -> None:
        broadcaster = IpcEventBroadcaster(runs_dir)
        await broadcaster.replay(
            cast(IpcEventConnection, RecordingConnection()),
            replay_from=1,
            run_id="run-1",
        )

    try:
        asyncio.run(replay())
    except ValueError as error:
        assert "invalid event history line" in str(error)
    else:
        raise AssertionError("invalid history line was accepted")


def test_ipc_broadcaster_removes_closed_or_broken_connections() -> None:
    assert asyncio.run(_remove_closed_or_broken_connections())


async def _remove_closed_or_broken_connections() -> bool:
    broadcaster = IpcEventBroadcaster()
    closed_connection = RecordingConnection()
    broken_connection = RecordingConnection(fail_on_write=True)

    broadcaster.subscribe(cast(IpcEventConnection, closed_connection))
    broadcaster.subscribe(cast(IpcEventConnection, broken_connection))
    closed_connection.close()

    assert broadcaster.subscriber_count == 1

    await broadcaster.handle(RunStartedEvent(goal="first", run_id="run-1"))

    return broadcaster.subscriber_count == 0


def test_ipc_broadcaster_immediately_reclaims_already_closed_connection() -> None:
    broadcaster = IpcEventBroadcaster()
    closed_connection = RecordingConnection()
    closed_connection.close()

    broadcaster.subscribe(cast(IpcEventConnection, closed_connection))

    assert broadcaster.subscriber_count == 0


def _event_type(envelope: EventPushEnvelope) -> str:
    assert envelope.kind == "event"
    event = envelope.event
    assert isinstance(event, dict)
    event_type = event["type"]
    assert isinstance(event_type, str)
    return event_type


def _write_history(runs_dir: Path, run_id: str, events: list[AgentEvent]) -> None:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    lines = [serialize_event(event) for event in events]
    (run_dir / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
