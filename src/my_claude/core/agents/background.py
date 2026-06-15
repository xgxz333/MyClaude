"""Shared registry for background subagent tasks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from my_claude.core.context import ExecutionContext


@dataclass(frozen=True)
class BackgroundTaskEntry:
    """One background subagent task and its context."""

    task: asyncio.Task[str]
    context: ExecutionContext


class BackgroundTaskRegistry:
    """Track background subagent tasks by run id."""

    def __init__(self) -> None:
        self._tasks: dict[str, BackgroundTaskEntry] = {}

    def register(
        self,
        run_id: str,
        task: asyncio.Task[str],
        context: ExecutionContext,
    ) -> None:
        self._tasks[run_id] = BackgroundTaskEntry(task=task, context=context)

    def get(self, run_id: str) -> tuple[asyncio.Task[str], ExecutionContext] | None:
        entry = self._tasks.get(run_id)
        if entry is None:
            return None
        return entry.task, entry.context

    def all(self) -> list[tuple[asyncio.Task[str], ExecutionContext]]:
        return [(entry.task, entry.context) for entry in self._tasks.values()]

    def remove(self, run_id: str) -> None:
        self._tasks.pop(run_id, None)
