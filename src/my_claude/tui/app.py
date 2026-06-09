"""Textual client for the MyClaude daemon."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.timer import Timer
from textual.widgets import Button, Footer, Header, Input, Log, Static

from my_claude.agent.events import AgentEvent, AgentEventType, KnownAgentEventAdapter
from my_claude.core.bus.command import (
    AGENT_RUN_METHOD,
    EVENT_PUBLISH_METHOD,
    EVENT_SUBSCRIBE_METHOD,
)
from my_claude.core.config import AppConfig, load_config
from my_claude.core.transport.socket_client import SocketClient, SocketClientError


class MyClaudeTui(App[None]):
    """Terminal UI that subscribes to daemon events over one long-lived socket."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #status {
        height: 1;
        padding: 0 1;
    }

    #event-log {
        height: 1fr;
        border: solid $surface;
    }

    #command-bar {
        height: 3;
        padding: 0 1;
    }

    #goal-input {
        width: 1fr;
    }

    #run-button {
        width: 12;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
    ]
    LOG_FLUSH_INTERVAL_SECONDS = 0.05

    def __init__(self, *, config: AppConfig | None = None) -> None:
        super().__init__()
        self._config = config or load_config()
        self._client: SocketClient | None = None
        self._connected = asyncio.Event()
        self._log_buffer: list[str] = []
        self._log_flush_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("disconnected", id="status")
        yield Log(id="event-log", highlight=True)
        with Horizontal(id="command-bar"):
            yield Input(placeholder="Goal", id="goal-input")
            yield Button("Run", id="run-button", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self._log_flush_timer = self.set_interval(
            self.LOG_FLUSH_INTERVAL_SECONDS,
            self._flush_log_buffer,
        )
        self.run_worker(
            self._network_loop(),
            name="daemon-network",
            exclusive=True,
            exit_on_error=False,
        )
        self.query_one("#goal-input", Input).focus()

    def on_unmount(self) -> None:
        if self._log_flush_timer is not None:
            self._log_flush_timer.stop()
            self._log_flush_timer = None
        self._flush_log_buffer()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-button":
            await self._submit_goal()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "goal-input":
            await self._submit_goal()

    async def _submit_goal(self) -> None:
        goal_input = self.query_one("#goal-input", Input)
        goal = goal_input.value.strip()
        if not goal:
            return

        client = self._client
        if client is None or not client.is_connected:
            self._write_log("[run] daemon is not connected")
            return

        try:
            result = await client.request(AGENT_RUN_METHOD, {"goal": goal})
        except SocketClientError as error:
            self._write_log(f"[run] failed to start: {error}")
            return

        goal_input.value = ""
        run_id = _result_field(result, "run_id") or "accepted"
        self._write_log(f"[run] accepted {run_id}")

    async def _network_loop(self) -> None:
        retry_seconds = 1.0

        while True:
            client: SocketClient | None = None
            try:
                self._set_status("connecting")
                client = SocketClient(
                    self._config.core_host,
                    self._config.core_port,
                    timeout_seconds=self._config.ipc_timeout_seconds,
                    max_response_bytes=self._config.max_request_bytes,
                )
                await client.connect()
                client.on(EVENT_PUBLISH_METHOD, self._handle_event_notification)

                await client.request(EVENT_SUBSCRIBE_METHOD, {})

                self._client = client
                self._connected.set()
                retry_seconds = 1.0
                self._set_status(
                    f"connected {self._config.core_host}:{self._config.core_port}"
                )
                self._write_log("[ipc] subscribed")

                await client.wait_closed()
                self._write_log("[ipc] connection closed")
            except asyncio.CancelledError:
                raise
            except SocketClientError as error:
                self._write_log(f"[ipc] {error}; retrying in {retry_seconds:.0f}s")
                await asyncio.sleep(retry_seconds)
                retry_seconds = min(retry_seconds * 2, 5.0)
            finally:
                self._flush_log_buffer()
                if client is not None:
                    with suppress(Exception):
                        await client.close()
                if self._client is client:
                    self._client = None
                self._connected.clear()
                self._set_status("disconnected")

    async def _handle_event_notification(self, params: dict[str, Any]) -> None:
        event_payload = params.get("event")
        if not isinstance(event_payload, dict):
            raise SocketClientError("event.publish params must include an event object")

        event = KnownAgentEventAdapter.validate_python(event_payload)
        self._write_log(_format_event(event))

    def _set_status(self, value: str) -> None:
        self.query_one("#status", Static).update(value)

    def _write_log(self, line: str) -> None:
        self._log_buffer.append(line)

    def _flush_log_buffer(self) -> None:
        if not self._log_buffer:
            return

        lines = self._log_buffer
        self._log_buffer = []
        self.query_one("#event-log", Log).write_lines(lines)


def _format_event(event: AgentEvent) -> str:
    if event.type == AgentEventType.RUN_STARTED:
        return f"[run] {event.data.get('run_id') or event.data.get('goal', '')}"
    if event.type == AgentEventType.STEP_STARTED:
        return f"[step {event.data.get('step', '')}] planning"
    if event.type == AgentEventType.STEP_FINISHED:
        return f"[step {event.data.get('step', '')}] done"
    if event.type == AgentEventType.LLM_TOKEN:
        return str(event.data.get("token", ""))
    if event.type == AgentEventType.LLM_RESPONSE_COMPLETED:
        return event.message
    if event.type == AgentEventType.TOOL_CALL_STARTED:
        return f"[tool] {event.data.get('tool_name', '')}"
    if event.type == AgentEventType.TOOL_CALL_COMPLETED:
        error = event.data.get("error")
        if error:
            return f"[tool] {event.data.get('tool_name', '')} failed: {error}"
        return f"[tool] {event.data.get('tool_name', '')} done"
    if event.type == AgentEventType.RUN_FAILED:
        return f"[run] failed: {event.data.get('error') or event.message}"
    if event.type == AgentEventType.RUN_CANCELLED:
        return "[run] cancelled"
    if event.type == AgentEventType.RUN_COMPLETED:
        return "[run] completed"

    return f"[event] {event.type.value}"


def _result_field(result: object, key: str) -> str | None:
    if not isinstance(result, dict):
        return None

    value = result.get(key)
    return value if isinstance(value, str) else None


def main() -> int:
    MyClaudeTui().run()
    return 0
