"""Asynchronous publish-subscribe event bus for runtime events."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

type AsyncEventHandler[EventPayload] = Callable[[EventPayload], Awaitable[None]]
type Unsubscribe = Callable[[], None]


class EventBus[EventPayload]:
    """Register async handlers and broadcast each published event to subscribers."""

    def __init__(self) -> None:
        self._subscribers: list[AsyncEventHandler[EventPayload]] = []

    def subscribe(self, handler: AsyncEventHandler[EventPayload]) -> Unsubscribe:
        self._subscribers.append(handler)

        def unsubscribe() -> None:
            self._subscribers.remove(handler)

        return unsubscribe

    async def publish(self, event: EventPayload) -> None:
        subscribers = tuple(self._subscribers)
        if not subscribers:
            return

        await asyncio.gather(*(subscriber(event) for subscriber in subscribers))

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
