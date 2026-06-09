"""Network event broadcaster for subscribed IPC clients."""

from __future__ import annotations

import fnmatch
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from my_claude.agent.events import AgentEvent, AgentEventType, KnownAgentEventAdapter
from my_claude.core.bus.envelope import EventPushEnvelope


class IpcEventConnection(Protocol):
    """Connection surface required by the IPC broadcaster."""

    def add_close_callback(self, callback: IpcConnectionClosedCallback) -> None: ...

    async def write_event(self, event: EventPushEnvelope) -> None: ...

    async def write_events(self, events: list[EventPushEnvelope]) -> None: ...


type IpcConnectionClosedCallback = Callable[[], None]


@dataclass(frozen=True)
class IpcEventSubscription:
    """One client's event delivery preferences."""

    subscription_id: str
    connection: IpcEventConnection
    topics: tuple[str, ...] = ("*",)
    scope: str = "global"
    event_types: frozenset[AgentEventType] | None = None
    run_id: str | None = None


class IpcEventBroadcaster:
    """EventBus subscriber that filters and pushes agent events to IPC clients."""

    def __init__(self, runs_dir: Path | None = None) -> None:
        self._runs_dir = runs_dir
        self._subscribers: dict[str, IpcEventSubscription] = {}
        self._history: list[tuple[int, AgentEvent]] = []
        self._next_sequence = 1

    @property
    def next_sequence(self) -> int:
        return self._next_sequence

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def subscribe(
        self,
        connection: IpcEventConnection,
        *,
        topics: list[str] | None = None,
        scope: str = "global",
        event_types: list[AgentEventType] | None = None,
        run_id: str | None = None,
    ) -> str:
        subscription_id = uuid.uuid4().hex
        subscription = IpcEventSubscription(
            subscription_id=subscription_id,
            connection=connection,
            topics=_normalize_topics(topics),
            scope=scope,
            event_types=None if event_types is None else frozenset(event_types),
            run_id=run_id,
        )
        self._subscribers[subscription_id] = subscription
        connection.add_close_callback(lambda: self.unsubscribe(subscription_id))
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> None:
        self._subscribers.pop(subscription_id, None)

    async def replay(
        self,
        connection: IpcEventConnection,
        *,
        replay_from: int | None,
        replay_from_run: str | None = None,
        topics: list[str] | None = None,
        scope: str = "global",
        event_types: list[AgentEventType] | None = None,
        run_id: str | None = None,
    ) -> int:
        if replay_from is None and replay_from_run is None:
            return 0
        if self._runs_dir is None:
            return 0

        subscription = IpcEventSubscription(
            subscription_id="history-replay",
            connection=connection,
            topics=_normalize_topics(topics),
            scope=scope,
            event_types=None if event_types is None else frozenset(event_types),
            run_id=run_id,
        )
        start_sequence = replay_from or 1
        history_run_id = replay_from_run or run_id
        event_frames = [
            _event_frame(event)
            for sequence, event in self._read_history_events(run_id=history_run_id)
            if sequence >= start_sequence and self._matches(subscription, event)
        ]
        if not event_frames:
            return 0

        await connection.write_events(event_frames)
        return len(event_frames)

    async def handle(self, event: AgentEvent) -> None:
        sequence = self._next_sequence
        self._next_sequence += 1
        self._history.append((sequence, event))

        stale_subscribers: list[str] = []
        for subscription_id, subscription in tuple(self._subscribers.items()):
            if not self._matches(subscription, event):
                continue

            try:
                await self._push(subscription.connection, sequence, event)
            except (ConnectionError, OSError, RuntimeError):
                stale_subscribers.append(subscription_id)

        for subscription_id in stale_subscribers:
            self.unsubscribe(subscription_id)

    def _matches(self, subscription: IpcEventSubscription, event: AgentEvent) -> bool:
        if subscription.event_types is not None and event.type not in subscription.event_types:
            return False

        if subscription.run_id is not None and _event_run_id(event) != subscription.run_id:
            return False

        if not _matches_topic(event.type.value, subscription.topics):
            return False

        if not _matches_scope(_event_run_id(event), subscription.scope):
            return False

        return True

    async def _push(
        self,
        connection: IpcEventConnection,
        sequence: int,
        event: AgentEvent,
    ) -> None:
        del sequence
        await connection.write_event(_event_frame(event))

    def _read_history_events(self, *, run_id: str | None) -> list[tuple[int, AgentEvent]]:
        if self._runs_dir is None:
            return []

        events: list[tuple[int, AgentEvent]] = []
        sequence = 1
        for path in self._history_paths(run_id=run_id):
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    event = _deserialize_history_event(line)
                except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as error:
                    raise ValueError(
                        f"invalid event history line: {path}:{line_number}"
                    ) from error

                events.append((sequence, event))
                sequence += 1

        return events

    def _history_paths(self, *, run_id: str | None) -> list[Path]:
        if self._runs_dir is None:
            return []
        if run_id is not None:
            path = self._runs_dir / run_id / "events.jsonl"
            return [path] if path.exists() else []

        return sorted(self._runs_dir.glob("*/events.jsonl"))


def _event_run_id(event: AgentEvent) -> str | None:
    run_id = event.data.get("run_id")
    if isinstance(run_id, str):
        return run_id

    return None


def _deserialize_history_event(line: str) -> AgentEvent:
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("event history line must be a JSON object")

    event_type = payload.get("type")
    message = payload.get("message", "")
    data = payload.get("data", {})
    if not isinstance(event_type, str):
        raise ValueError("event history line must include a string type")
    if not isinstance(message, str):
        raise ValueError("event history line message must be a string")
    if not isinstance(data, dict):
        raise ValueError("event history line data must be an object")

    event_payload = {"type": event_type, "message": message, **data}
    return KnownAgentEventAdapter.validate_python(event_payload)


def _event_frame(event: AgentEvent) -> EventPushEnvelope:
    return EventPushEnvelope(event=event.model_dump(mode="json"))


def _normalize_topics(topics: list[str] | None) -> tuple[str, ...]:
    if not topics:
        return ("*",)

    return tuple(topics)


def _matches_topic(event_type: str, topics: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(event_type, topic) for topic in topics)


def _matches_scope(run_id: str | None, scope: str) -> bool:
    if scope == "global":
        return True
    if scope.startswith("run:"):
        return run_id == scope[4:]
    return False
