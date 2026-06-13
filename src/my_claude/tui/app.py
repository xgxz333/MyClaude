"""Textual client for the MyClaude daemon."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from contextlib import suppress
from typing import Any

from pydantic import ValidationError
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Input, Label, Static

from my_claude.agent.events import AgentEvent, AgentEventType, KnownAgentEventAdapter
from my_claude.core.bus.command import (
    AGENT_RUN_METHOD,
    EVENT_SUBSCRIBE_METHOD,
)
from my_claude.core.config import AppConfig, load_config
from my_claude.core.transport.socket_client import SocketClient, SocketClientError


class LLMStreamBlock(Static):
    """Accumulate streamed LLM tokens in one Textual widget."""

    DEFAULT_CSS = "LLMStreamBlock { padding: 0 2; color: $text; }"

    def __init__(self) -> None:
        super().__init__("")
        self.text = ""

    def append_token(self, token: str) -> None:
        self.text += token
        self.update(self.text)


class ToolCallBlock(Widget):
    """Collapsible tool call block matching the KamaClaude TUI format."""

    DEFAULT_CSS = """
    ToolCallBlock { height: auto; padding: 0 0; }
    ToolCallBlock > .detail { display: none; padding: 0 4; color: $text-muted; }
    ToolCallBlock.expanded > .detail { display: block; }
    """

    def __init__(self, tool_name: str, params: Mapping[str, Any]) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.params = dict(params)
        self._params_full = _params_str(self.params)
        self.output = ""
        self.elapsed_ms = 0
        self.is_error = False
        self.finished = False
        self._summary_widget: Static | None = None
        self._detail_widget: Static | None = None

    def compose(self) -> ComposeResult:
        self._summary_widget = Static(self.summary, classes="summary")
        self._detail_widget = Static("", classes="detail")
        yield self._summary_widget
        yield self._detail_widget

    @property
    def summary(self) -> str:
        params_pre = _preview(self._params_full, 60)
        icon = "[bold yellow]✎[/bold yellow]"
        line = f"  {icon} [bold]{self.tool_name}[/bold]  [dim]{params_pre}[/dim]"
        if self.finished:
            out_pre = _preview(self.output, 50)
            color = "red" if self.is_error else "dim"
            hint = "  [dim]▸ click to expand[/dim]" if len(self.output) > 50 else ""
            line += (
                f"\n  [dim]↳[/dim] [{color}]{out_pre}[/{color}]"
                f"  [dim]{self.elapsed_ms}ms[/dim]{hint}"
            )
        return line

    def set_result(self, output: str, elapsed_ms: int, *, is_error: bool = False) -> None:
        self.output = output
        self.elapsed_ms = elapsed_ms
        self.is_error = is_error
        self.finished = True
        if self._summary_widget is not None:
            self._summary_widget.update(self.summary)

    def on_click(self) -> None:
        if not self.finished:
            return
        if "expanded" in self.classes:
            self.remove_class("expanded")
            return

        if self._detail_widget is not None:
            self._detail_widget.update(
                f"[dim]params:[/dim]\n    {self._params_full}\n"
                f"[dim]output:[/dim]\n    {self.output}\n"
                f"[dim]elapsed:[/dim] {self.elapsed_ms}ms"
            )
        self.add_class("expanded")


class MyClaudeTui(App[None]):
    """Terminal UI that subscribes to daemon events over one long-lived socket."""

    TITLE = "MyClaude"
    BINDINGS = [
        Binding("q", "quit", "quit"),
        Binding("ctrl+c", "quit", "quit"),
    ]
    CSS = """
    Screen { background: $background; layout: vertical; }
    #header {
        height: 1;
        background: $primary;
        color: $text;
        padding: 0 1;
    }
    #log-view {
        height: 1fr;
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
    Static.run-header { color: cyan; padding: 1 2 0 2; }
    Static.step-divider { color: $text-muted; padding: 0 2; }
    Static.run-ok { color: green; padding: 0 2 1 2; }
    Static.run-err { color: red; padding: 0 2 1 2; }
    Static.usage { padding: 0 2; }
    Static.log-line { padding: 0 2; }
    """

    def __init__(self, *, config: AppConfig | None = None) -> None:
        super().__init__()
        self._config = config or load_config()
        self._client: SocketClient | None = None
        self._connected = asyncio.Event()
        self._current_llm: LLMStreamBlock | None = None
        self._pending_tool_blocks: dict[str, ToolCallBlock] = {}

    def compose(self) -> ComposeResult:
        yield Label("[bold]MyClaude[/bold]  [dim]connecting...[/dim]", id="header")
        yield VerticalScroll(id="log-view")
        with Horizontal(id="command-bar"):
            yield Input(placeholder="Goal", id="goal-input")
            yield Button("Run", id="run-button", variant="primary")

    def on_mount(self) -> None:
        self.run_worker(
            self._network_loop(),
            name="daemon-network",
            exclusive=True,
            exit_on_error=False,
        )
        self.query_one("#goal-input", Input).focus()

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
            self._append_log("ERROR", "tui", "daemon is not connected")
            return

        try:
            await client.request(AGENT_RUN_METHOD, {"goal": goal})
        except SocketClientError as error:
            self._append_log("ERROR", "tui", f"failed to start: {error}")
            return

        goal_input.value = ""

    async def _network_loop(self) -> None:
        retry_seconds = 1.0
        header = self.query_one("#header", Label)

        while True:
            client: SocketClient | None = None
            try:
                header.update("[bold]MyClaude[/bold]  [dim]connecting...[/dim]")
                client = SocketClient(
                    self._config.core_host,
                    self._config.core_port,
                    timeout_seconds=self._config.ipc_timeout_seconds,
                    max_response_bytes=self._config.max_request_bytes,
                )
                await client.connect()
                client.on_event(self._handle_event)

                await client.request(
                    EVENT_SUBSCRIBE_METHOD,
                    {
                        "topics": [
                            "run.*",
                            "step.*",
                            "tool.*",
                            "llm.token",
                            "llm.usage",
                            "llm.response_completed",
                            "log.*",
                        ],
                        "scope": "global",
                    },
                )

                self._client = client
                self._connected.set()
                retry_seconds = 1.0
                header.update(
                    f"[bold]MyClaude[/bold]  "
                    f"[dim]{self._config.core_host}:{self._config.core_port}[/dim]"
                )

                await client.wait_closed()
            except asyncio.CancelledError:
                raise
            except SocketClientError as error:
                header.update(
                    f"[bold]MyClaude[/bold]  "
                    f"[red]{error}; retrying in {retry_seconds:.0f}s[/red]"
                )
                await asyncio.sleep(retry_seconds)
                retry_seconds = min(retry_seconds * 2, 5.0)
            finally:
                if client is not None:
                    with suppress(Exception):
                        await client.close()
                if self._client is client:
                    self._client = None
                self._connected.clear()
                self._break_llm()
                header.update("[bold]MyClaude[/bold]  [dim]disconnected - retrying...[/dim]")

    async def _handle_event(self, event_payload: dict[str, Any]) -> None:
        try:
            event = KnownAgentEventAdapter.validate_python(event_payload)
        except ValidationError:
            self._write_event_payload(event_payload)
            return

        self._write_event(event)

    def _append(self, widget: Widget) -> None:
        log_view = self.query_one("#log-view", VerticalScroll)
        log_view.mount(widget)
        log_view.scroll_end(animate=False)

    def _break_llm(self) -> None:
        self._current_llm = None

    def _append_log(self, level: str, source: str, message: str) -> None:
        color = "bold red" if level == "ERROR" else ("yellow" if level == "WARNING" else "dim")
        self._append(
            Static(
                f"[{color}]{level}[/{color}]  [dim]{source}[/dim]  {message}",
                classes="log-line",
            )
        )

    def _write_event(self, event: AgentEvent) -> None:
        self._write_event_payload(event.model_dump(mode="json", exclude_none=True))

    def _write_event_payload(self, event: Mapping[str, Any]) -> None:
        event_type = str(event.get("type", ""))

        if event_type == AgentEventType.LLM_TOKEN:
            token = str(event.get("token", ""))
            if self._current_llm is None:
                llm_block = LLMStreamBlock()
                self._append(llm_block)
                self._current_llm = llm_block
            self._current_llm.append_token(token)
            return

        self._break_llm()

        if event_type == AgentEventType.RUN_STARTED:
            run_id = event.get("run_id", "")
            goal = event.get("goal", "")
            self._append(
                Static(
                    f"[bold cyan]▶ run[/bold cyan]  [dim]{run_id}[/dim]\n"
                    f"  [dim]goal:[/dim] {goal}",
                    classes="run-header",
                )
            )

        elif event_type == AgentEventType.STEP_STARTED:
            step = event.get("step", "")
            self._append(
                Static(
                    f"[dim]── step {step} {'─' * 48}[/dim]",
                    classes="step-divider",
                )
            )

        elif event_type == AgentEventType.TOOL_CALL_STARTED:
            tool_use_id = str(event.get("tool_use_id", ""))
            tool_name = str(event.get("tool_name", ""))
            params = _mapping_from(event.get("params")) or _mapping_from(event.get("arguments"))
            tool_block = ToolCallBlock(tool_name, params)
            self._pending_tool_blocks[tool_use_id] = tool_block
            self._append(tool_block)

        elif event_type == AgentEventType.TOOL_CALL_COMPLETED:
            tool_use_id = str(event.get("tool_use_id", ""))
            elapsed_ms = _int_from(event.get("elapsed_ms"))
            output = event.get("output")
            if output is None:
                output = event.get("result")
            error = event.get("error_message")
            if error is None:
                error = event.get("error")

            if tool_use_id in self._pending_tool_blocks:
                tool_block = self._pending_tool_blocks.pop(tool_use_id)
                tool_block.set_result(
                    str(error if error is not None else output or ""),
                    elapsed_ms,
                    is_error=error is not None,
                )

        elif event_type == AgentEventType.RUN_COMPLETED:
            status = event.get("status", "success")
            steps = event.get("steps", 0) or 0
            reason = event.get("reason") or event.get("error") or ""
            if status == "success":
                self._append(
                    Static(
                        f"[bold green]✓ completed[/bold green]  [dim]{steps} steps[/dim]",
                        classes="run-ok",
                    )
                )
            else:
                self._append_failed_run(reason=str(reason), steps=steps)

        elif event_type == AgentEventType.RUN_FAILED:
            reason = event.get("reason") or event.get("error") or ""
            self._append_failed_run(reason=str(reason), steps=event.get("steps", 0) or 0)

        elif event_type == AgentEventType.RUN_CANCELLED:
            reason = event.get("reason") or ""
            self._append_failed_run(reason=str(reason), steps=event.get("steps", 0) or 0)

        elif event_type == AgentEventType.LLM_USAGE:
            self._append(
                Static(
                    f"[dim]  tokens  "
                    f"in={event.get('input_tokens')} "
                    f"out={event.get('output_tokens')} "
                    f"cache={event.get('cache_read_input_tokens')}[/dim]",
                    classes="usage",
                )
            )

        elif event_type == AgentEventType.LLM_RESPONSE_COMPLETED:
            content = str(event.get("content") or "")
            if content:
                block = LLMStreamBlock()
                self._append(block)
                block.append_token(content)

        elif event_type == "log.line":
            self._append_log(
                str(event.get("level", "INFO")),
                str(event.get("source", "")),
                str(event.get("message", "")),
            )

    def _append_failed_run(self, *, reason: str, steps: object) -> None:
        detail = f"  [dim]{reason}[/dim]" if reason else ""
        self._append(
            Static(
                f"[bold red]✗ failed[/bold red]{detail}  [dim]{steps} steps[/dim]",
                classes="run-err",
            )
        )


def _mapping_from(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _int_from(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _preview(value: str, limit: int) -> str:
    return value[:limit] + "…" if len(value) > limit else value


def _params_str(params: Mapping[str, Any]) -> str:
    return json.dumps(params, ensure_ascii=False)


def main() -> int:
    MyClaudeTui().run()
    return 0
