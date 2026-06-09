"""Network event broadcaster for subscribed IPC clients."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from my_claude.agent.events import AgentEvent, AgentEventType, KnownAgentEventAdapter
from my_claude.core.bus.command import EVENT_PUBLISH_METHOD
from my_claude.core.bus.envelope import JsonRpcNotification, make_notification


class IpcEventConnection(Protocol):
    """Connection surface required by the IPC broadcaster."""

    def add_close_callback(self, callback: IpcConnectionClosedCallback) -> None: ...

    async def write_notification(self, notification: JsonRpcNotification) -> None: ...

    async def write_notifications(self, notifications: list[JsonRpcNotification]) -> None: ...


type IpcConnectionClosedCallback = Callable[[], None]


@dataclass(frozen=True)
class IpcEventSubscription:
    """One client's event delivery preferences."""

    subscription_id: str
    connection: IpcEventConnection
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
        event_types: list[AgentEventType] | None = None,
        run_id: str | None = None,
    ) -> str:
        subscription_id = uuid.uuid4().hex
        subscription = IpcEventSubscription(
            subscription_id=subscription_id,
            connection=connection,
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
        event_types: list[AgentEventType] | None = None,
        run_id: str | None = None,
    ) -> None:
        if replay_from is None or self._runs_dir is None:
            return

        subscription = IpcEventSubscription(
            subscription_id="history-replay",
            connection=connection,
            event_types=None if event_types is None else frozenset(event_types),
            run_id=run_id,
        )
        notifications = [
            _event_notification(sequence, event)
            for sequence, event in self._read_history_events(run_id=run_id)
            if sequence >= replay_from and self._matches(subscription, event)
        ]
        if not notifications:
            return

        await connection.write_notifications(notifications)

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

        return True

    async def _push(
        self,
        connection: IpcEventConnection,
        sequence: int,
        event: AgentEvent,
    ) -> None:
        await connection.write_notification(
            _event_notification(sequence, event)
        )

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


def _event_notification(sequence: int, event: AgentEvent) -> JsonRpcNotification:
    return make_notification(
        EVENT_PUBLISH_METHOD,
        {
            "sequence": sequence,
            "event": event.model_dump(mode="json"),
        },
    )
