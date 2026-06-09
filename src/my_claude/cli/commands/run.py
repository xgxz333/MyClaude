"""Implementation of the `myclaude run` CLI command."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from typing import TextIO

from my_claude.agent.events import AgentEvent, AgentEventType
from my_claude.core.config import load_config
from my_claude.core.events.bus import EventBus
from my_claude.core.runner import run_goal


class StdoutPrinter:
    """Render agent lifecycle events to stdout and failures to stderr."""

    def __init__(
        self,
        stream: TextIO | None = None,
        error_stream: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._error_stream = error_stream if error_stream is not None else sys.stderr
        self._clock = clock
        self._started_at: float | None = None
        self._steps = 0
        self._response_had_tokens = False
        self._cursor_open = False

    def subscribe(self, event_bus: EventBus[AgentEvent]) -> None:
        event_bus.subscribe(self.handle)

    async def handle(self, event: AgentEvent) -> None:
        if event.type == AgentEventType.RUN_STARTED:
            self._started_at = self._clock()
            self._steps = 0
            self._response_had_tokens = False
            self._cursor_open = False
            run_id = event.data.get("run_id") or event.data.get("goal", "")
            print(f"[run] {run_id}", file=self._stream, flush=True)
        elif event.type == AgentEventType.STEP_STARTED:
            self._ensure_newline()
            self._response_had_tokens = False
            self._steps = int(event.data.get("step", self._steps + 1))
            print(f"[step {self._steps}] planning...", file=self._stream, flush=True)
        elif event.type == AgentEventType.STEP_FINISHED:
            self._ensure_newline()
            step = event.data.get("step", self._steps)
            print(f"[step {step}] done", file=self._stream, flush=True)
        elif event.type == AgentEventType.LLM_REQUEST_STARTED:
            self._response_had_tokens = False
        elif event.type == AgentEventType.LLM_TOKEN:
            print(event.data.get("token", ""), end="", file=self._stream, flush=True)
            self._response_had_tokens = True
            self._cursor_open = True
        elif event.type == AgentEventType.LLM_RESPONSE_COMPLETED:
            if not self._response_had_tokens:
                print(event.message, file=self._stream, flush=True)
            elif self._cursor_open:
                print(file=self._stream, flush=True)
                self._cursor_open = False
        elif event.type == AgentEventType.TOOL_CALL_STARTED:
            self._ensure_newline()
            tool_name = event.data.get("tool_name", "")
            arguments = event.data.get("arguments", {})
            arguments_text = json.dumps(arguments, ensure_ascii=False)
            print(f"[tool] {tool_name} {arguments_text}", file=self._stream, flush=True)
        elif event.type == AgentEventType.TOOL_CALL_COMPLETED:
            tool_name = event.data.get("tool_name", "")
            error = event.data.get("error")
            elapsed_ms = event.data.get("elapsed_ms", 0)
            if error:
                print(f"[tool] {tool_name} ✗  {error}", file=self._error_stream, flush=True)
            else:
                print(f"[tool] {tool_name} ✓  {elapsed_ms}ms", file=self._stream, flush=True)
        elif event.type == AgentEventType.RUN_COMPLETED:
            self._print_finished("success", self._stream)
        elif event.type == AgentEventType.RUN_CANCELLED:
            self._print_finished("cancelled", self._error_stream)
        elif event.type == AgentEventType.RUN_FAILED:
            error = event.data.get("error") or event.message
            if error:
                print(f"[run] error: {error}", file=self._error_stream, flush=True)
            self._print_finished("failed", self._error_stream)

    def _ensure_newline(self) -> None:
        if self._cursor_open:
            print(file=self._stream, flush=True)
            self._cursor_open = False

    def _print_finished(self, status: str, stream: TextIO) -> None:
        self._ensure_newline()

        elapsed_seconds = 0.0
        if self._started_at is not None:
            elapsed_seconds = self._clock() - self._started_at

        print(
            f"[run] {status}  {self._steps} steps  {elapsed_seconds:.1f}s",
            file=stream,
            flush=True,
        )


StdoutEventPrinter = StdoutPrinter


def main(args: argparse.Namespace) -> int:
    try:
        return asyncio.run(_run_async(args))
    except KeyboardInterrupt:
        print("[run] cancelled", file=sys.stderr, flush=True)
        return 130
    except RuntimeError as error:
        print(f"[run] failed: {error}", file=sys.stderr, flush=True)
        return 1


async def _run_async(args: argparse.Namespace) -> int:
    config = load_config()
    printer = StdoutPrinter()
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, shutdown_event.set)

    run_task = asyncio.create_task(run_goal(args.goal, config=config, listeners=[printer.handle]))
    shutdown_task = asyncio.create_task(shutdown_event.wait())

    try:
        done, pending = await asyncio.wait(
            {run_task, shutdown_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if shutdown_task in done:
            run_task.cancel()
            with suppress(asyncio.CancelledError):
                await run_task
            return 130

        shutdown_task.cancel()
        for task in pending:
            task.cancel()
        try:
            await run_task
        except Exception:
            return 1
        return 0
    finally:
        shutdown_task.cancel()
