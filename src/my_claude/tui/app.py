"""Textual client for the MyClaude daemon."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from contextlib import suppress
from typing import Any

from pydantic import ValidationError
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Input, Label, Static

from my_claude.agent.events import AgentEvent, AgentEventType, KnownAgentEventAdapter
from my_claude.core.bus.command import (
    EVENT_SUBSCRIBE_METHOD,
    PERMISSION_RESPOND_METHOD,
    SESSION_CLOSE_METHOD,
    SESSION_CREATE_METHOD,
    SESSION_SEND_MESSAGE_METHOD,
)
from my_claude.core.config import AppConfig, load_config
from my_claude.core.transport.socket_client import SocketClient, SocketClientError

RUN_SCOPED_EVENT_TYPES = {
    AgentEventType.RUN_STARTED.value,
    AgentEventType.STEP_STARTED.value,
    AgentEventType.STEP_FINISHED.value,
    AgentEventType.LLM_REQUEST_STARTED.value,
    AgentEventType.LLM_TOKEN.value,
    AgentEventType.LLM_USAGE.value,
    AgentEventType.LLM_MODEL_SELECTED.value,
    AgentEventType.LLM_RESPONSE_COMPLETED.value,
    AgentEventType.TOOL_CALL_STARTED.value,
    AgentEventType.TOOL_CALL_FAILED.value,
    AgentEventType.TOOL_CALL_COMPLETED.value,
    AgentEventType.PERMISSION_REQUESTED.value,
    AgentEventType.PERMISSION_GRANTED.value,
    AgentEventType.PERMISSION_DENIED.value,
    AgentEventType.RUN_COMPLETED.value,
    AgentEventType.RUN_CANCELLED.value,
    AgentEventType.RUN_FAILED.value,
}


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


class PermissionBlock(Static):
    """Static log entry shown when a tool call is waiting for approval."""

    DEFAULT_CSS = "PermissionBlock { padding: 0 2; color: yellow; }"
    _LABEL_MAP = {
        "allow_once": "allowed (once)",
        "always_allow": "always allowed",
        "deny_once": "denied",
        "always_deny": "always denied",
        "timeout": "timed out",
    }

    class Resolved(Message):
        """Message emitted when the permission log block is resolved."""

        def __init__(self, block: PermissionBlock, decision: str) -> None:
            self.block = block
            self.decision = decision
            super().__init__()

    def __init__(self, tool_use_id: str, tool_name: str, param_preview: str) -> None:
        self.tool_use_id = tool_use_id
        self.tool_name = tool_name
        self.param_preview = param_preview
        self._resolved = False
        super().__init__(self._pending_text())

    def _pending_text(self) -> str:
        preview = f"  [dim]{self.param_preview}[/dim]" if self.param_preview else ""
        return f"[bold yellow]? permission[/bold yellow]  [bold]{self.tool_name}[/bold]{preview}"

    def _resolve(self, decision: str) -> None:
        if self._resolved:
            return
        self._resolved = True
        allowed = decision in ("allow_once", "always_allow")
        icon = "[bold green]✓[/bold green]" if allowed else "[bold red]✗[/bold red]"
        label = self._LABEL_MAP.get(decision, decision)
        preview = f"  [dim]{self.param_preview}[/dim]" if self.param_preview else ""
        self.update(
            f"{icon} permission  [bold]{self.tool_name}[/bold]{preview}  [dim]{label}[/dim]"
        )
        self.post_message(self.Resolved(self, decision))


class PermissionSelect(Static):
    """Focusable inline approval control for one pending permission request."""

    DEFAULT_CSS = """
    PermissionSelect {
        padding: 0 2 1 2;
        color: $text;
    }
    PermissionSelect:focus {
        text-style: bold;
        background: $boost;
    }
    """

    can_focus = True
    _CHOICES: tuple[tuple[str, str, str], ...] = (
        ("allow_once", "Allow once", "y / 1"),
        ("always_allow", "Always allow", "a / 2"),
        ("deny_once", "Deny", "n / 3"),
        ("always_deny", "Always deny", "d / 4"),
    )
    _KEY_MAP: dict[str, str] = {
        "y": "allow_once",
        "1": "allow_once",
        "a": "always_allow",
        "2": "always_allow",
        "n": "deny_once",
        "3": "deny_once",
        "d": "always_deny",
        "4": "always_deny",
    }

    class Decided(Message):
        """Message emitted when the user picks a permission decision."""

        def __init__(self, widget: PermissionSelect, tool_use_id: str, decision: str) -> None:
            self.widget = widget
            self.tool_use_id = tool_use_id
            self.decision = decision
            super().__init__()

    def __init__(self, tool_use_id: str) -> None:
        super().__init__("")
        self.tool_use_id = tool_use_id
        self._cursor = 0

    def on_mount(self) -> None:
        self.update(self._render_ui())
        self.focus()

    def _render_ui(self) -> str:
        lines: list[str] = []
        for index, (_decision, label, key_hint) in enumerate(self._CHOICES):
            if index == self._cursor:
                lines.append(f"  [bold cyan]> {label}[/bold cyan]  [dim]{key_hint}[/dim]")
            else:
                lines.append(f"    {label}  [dim]{key_hint}[/dim]")
        lines.append("[dim]  up/down navigate   enter confirm[/dim]")
        return "\n".join(lines)

    def on_key(self, event: events.Key) -> None:
        key = event.key
        if key in ("up", "k"):
            event.stop()
            self._cursor = (self._cursor - 1) % len(self._CHOICES)
            self.update(self._render_ui())
            return
        if key in ("down", "j"):
            event.stop()
            self._cursor = (self._cursor + 1) % len(self._CHOICES)
            self.update(self._render_ui())
            return
        if key == "enter":
            event.stop()
            self._pick(self._CHOICES[self._cursor][0])
            return

        decision = self._KEY_MAP.get(key)
        if decision is not None:
            event.stop()
            self._pick(decision)

    def _pick(self, decision: str) -> None:
        self.post_message(self.Decided(self, self.tool_use_id, decision))


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
        width: 10;
    }
    Static.user-message { color: $text; padding: 1 2 0 2; }
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
        self._session_id: str | None = None
        self._active_run_id: str | None = None
        self._awaiting_message_result = False
        self._pending_events_by_run: dict[str, list[Mapping[str, Any]]] = {}
        self._current_llm: LLMStreamBlock | None = None
        self._pending_tool_blocks: dict[str, ToolCallBlock] = {}
        self._pending_permission_blocks: dict[str, PermissionBlock] = {}
        self._pending_permission_selects: dict[str, PermissionSelect] = {}

    def compose(self) -> ComposeResult:
        yield Label("[bold]MyClaude[/bold]  [dim]connecting...[/dim]", id="header")
        yield VerticalScroll(id="log-view")
        with Horizontal(id="command-bar"):
            yield Input(placeholder="Connecting...", id="goal-input", disabled=True)
            yield Button("Send", id="run-button", variant="primary", disabled=True)

    def on_mount(self) -> None:
        self.run_worker(
            self._network_loop(),
            name="daemon-network",
            exclusive=True,
            exit_on_error=False,
        )
        self._set_waiting_for_connection()
        self.query_one("#goal-input", Input).focus()

    async def on_unmount(self) -> None:
        client = self._client
        if client is None or self._session_id is None or not client.is_connected:
            return
        with suppress(SocketClientError):
            await client.request(
                SESSION_CLOSE_METHOD,
                {"session_id": self._session_id},
            )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-button":
            self._submit_message()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "goal-input":
            self._submit_message()

    def on_key(self, event: events.Key) -> None:
        if not self._pending_permission_selects:
            return
        select = next(iter(self._pending_permission_selects.values()))
        if select.has_focus:
            return
        if event.key in PermissionSelect._KEY_MAP or event.key in ("up", "down", "k", "j", "enter"):
            select.on_key(event)

    def on_chat_text_area_submitted(self, event: Any) -> None:
        value = getattr(event, "value", None)
        if value is None:
            text_area = getattr(event, "text_area", None)
            value = getattr(text_area, "text", None)
            if value is None:
                value = getattr(text_area, "value", "")
        content = str(value).strip()
        if content:
            self._submit_message(content=content)

    async def on_permission_select_decided(self, message: PermissionSelect.Decided) -> None:
        tool_use_id = message.tool_use_id
        decision = message.decision
        self._pending_permission_selects.pop(tool_use_id, None)
        with suppress(Exception):
            message.widget.remove()

        permission_block = self._pending_permission_blocks.pop(tool_use_id, None)
        if permission_block is not None:
            permission_block._resolve(decision)

        self.run_worker(
            self._do_respond_permission(tool_use_id, decision),
            name=f"permission_respond:{tool_use_id}",
            exclusive=False,
            exit_on_error=False,
        )

    async def _submit_goal(self) -> None:
        self._submit_message()

    def _submit_message(self, *, content: str | None = None) -> None:
        message_input = self.query_one("#goal-input", Input)
        message = (content if content is not None else message_input.value).strip()
        if not message:
            return

        client = self._client
        if client is None or not client.is_connected:
            self._append_log("ERROR", "tui", "daemon is not connected")
            return

        if self._session_id is None:
            self._append_log("ERROR", "tui", "session is not ready")
            return

        if self._active_run_id is not None:
            self._append_log("WARNING", "tui", f"session is busy: {self._active_run_id}")
            return

        self._awaiting_message_result = True
        message_input.value = ""
        self._append(Static(f"[bold]you[/bold]  {message}", classes="user-message"))
        self._set_input_state(
            disabled=True,
            placeholder="Sending...",
            button_label="Busy",
        )
        self.run_worker(
            self._do_send_message(message),
            name="send_message",
            exclusive=False,
            exit_on_error=False,
        )

    async def _do_send_message(self, content: str) -> None:
        client = self._client
        if client is None or not client.is_connected or self._session_id is None:
            self._awaiting_message_result = False
            self._mark_turn_waiting()
            self._append_log("ERROR", "tui", "daemon is not connected")
            return

        try:
            result = await client.request(
                SESSION_SEND_MESSAGE_METHOD,
                {"session_id": self._session_id, "content": content},
            )
        except SocketClientError as error:
            self._awaiting_message_result = False
            self._mark_turn_waiting()
            self._append_log("ERROR", "tui", f"failed to send: {error}")
            return

        self._awaiting_message_result = False
        self._mark_turn_running(str(result["run_id"]))

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
                            "session.*",
                            "permission.*",
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
                session_result = await client.request(SESSION_CREATE_METHOD, {})
                self._session_id = str(session_result["session_id"])

                self._client = client
                self._connected.set()
                retry_seconds = 1.0
                header.update(
                    f"[bold]MyClaude[/bold]  "
                    f"[dim]{self._config.core_host}:{self._config.core_port}[/dim]  "
                    f"[dim]{self._session_id}[/dim]"
                )
                self._mark_turn_waiting()

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
                self._session_id = None
                self._active_run_id = None
                self._awaiting_message_result = False
                self._pending_events_by_run.clear()
                self._break_llm()
                self._set_waiting_for_connection()
                header.update("[bold]MyClaude[/bold]  [dim]disconnected - retrying...[/dim]")

    async def _handle_event(self, event_payload: dict[str, Any]) -> None:
        try:
            event = KnownAgentEventAdapter.validate_python(event_payload)
        except ValidationError:
            self._receive_event_payload(event_payload)
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
        self._receive_event_payload(event.model_dump(mode="json", exclude_none=True))

    def _receive_event_payload(self, event: Mapping[str, Any]) -> None:
        run_id = _run_id_from(event)
        if run_id is not None and self._should_buffer_event(run_id):
            self._pending_events_by_run.setdefault(run_id, []).append(event)
            return
        if not self._should_render_event(event):
            return

        self._render_event_payload(event)

    def _render_event_payload(self, event: Mapping[str, Any]) -> None:
        event_type = str(event.get("type", ""))

        if event_type == AgentEventType.LLM_TOKEN:
            token = str(event.get("token", ""))
            if self._current_llm is None:
                llm_block = LLMStreamBlock()
                self._append(llm_block)
                self._current_llm = llm_block
            self._current_llm.append_token(token)
            return

        if event_type == AgentEventType.LLM_RESPONSE_COMPLETED:
            content = str(event.get("content") or "")
            current_llm = self._current_llm
            if current_llm is not None and current_llm.text:
                self._break_llm()
                return
            if content:
                block = current_llm or LLMStreamBlock()
                if current_llm is None:
                    self._append(block)
                block.append_token(content)
            self._break_llm()
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

        elif event_type == AgentEventType.TOOL_CALL_FAILED:
            tool_use_id = str(event.get("tool_use_id", ""))
            elapsed_ms = _int_from(event.get("elapsed_ms"))
            error = event.get("error_message")
            if error is None:
                error = event.get("error")

            if tool_use_id in self._pending_tool_blocks:
                tool_block = self._pending_tool_blocks.pop(tool_use_id)
                tool_block.set_result(str(error or ""), elapsed_ms, is_error=True)

        elif event_type == AgentEventType.PERMISSION_REQUESTED:
            tool_use_id = str(event.get("tool_use_id", ""))
            tool_name = str(event.get("tool_name", ""))
            param_preview = str(event.get("param_preview", ""))
            permission_block = PermissionBlock(tool_use_id, tool_name, param_preview)
            self._pending_permission_blocks[tool_use_id] = permission_block
            self._set_input_state(
                disabled=True,
                placeholder="Permission required...",
                button_label="Busy",
            )
            self._append(permission_block)
            select = PermissionSelect(tool_use_id)
            self._mount_permission_select(select)

        elif event_type in (
            AgentEventType.PERMISSION_GRANTED,
            AgentEventType.PERMISSION_DENIED,
        ):
            tool_use_id = str(event.get("tool_use_id", ""))
            decision = str(event.get("decision", ""))
            self._resolve_permission(tool_use_id, decision)

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
            self._finish_active_turn_if_matching(event)

        elif event_type == AgentEventType.RUN_FAILED:
            reason = event.get("reason") or event.get("error") or ""
            self._append_failed_run(reason=str(reason), steps=event.get("steps", 0) or 0)
            self._finish_active_turn_if_matching(event)

        elif event_type == AgentEventType.RUN_CANCELLED:
            reason = event.get("reason") or ""
            self._append_failed_run(reason=str(reason), steps=event.get("steps", 0) or 0)
            self._finish_active_turn_if_matching(event)

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

        elif event_type == "log.line":
            self._append_log(
                str(event.get("level", "INFO")),
                str(event.get("source", "")),
                str(event.get("message", "")),
            )

    def _mount_permission_select(self, select: PermissionSelect) -> None:
        self._pending_permission_selects[select.tool_use_id] = select
        try:
            self.mount(select, before="#command-bar")
        except Exception:
            self._append(select)

    def _respond_permission(self, tool_use_id: str, decision: str) -> None:
        self._resolve_permission(tool_use_id, decision)
        self.run_worker(
            self._do_respond_permission(tool_use_id, decision),
            name=f"permission_respond:{tool_use_id}",
            exclusive=False,
            exit_on_error=False,
        )

    async def _do_respond_permission(self, tool_use_id: str, decision: str) -> None:
        client = self._client
        if client is None or not client.is_connected:
            self._append_log("ERROR", "tui", "daemon is not connected")
            return

        try:
            await client.request(
                PERMISSION_RESPOND_METHOD,
                {"tool_use_id": tool_use_id, "decision": decision},
            )
        except SocketClientError as error:
            self._append_log("ERROR", "tui", f"failed to respond: {error}")

    def _resolve_permission(self, tool_use_id: str, decision: str) -> None:
        select = self._pending_permission_selects.pop(tool_use_id, None)
        if select is not None:
            with suppress(Exception):
                select.remove()

        permission_block = self._pending_permission_blocks.pop(tool_use_id, None)
        if permission_block is not None:
            permission_block._resolve(decision)

    def _append_failed_run(self, *, reason: str, steps: object) -> None:
        detail = f"  [dim]{reason}[/dim]" if reason else ""
        self._append(
            Static(
                f"[bold red]✗ failed[/bold red]{detail}  [dim]{steps} steps[/dim]",
                classes="run-err",
            )
        )

    def _mark_turn_running(self, run_id: str) -> None:
        self._active_run_id = run_id
        self._set_input_state(
            disabled=True,
            placeholder=f"Waiting for {run_id}...",
            button_label="Busy",
        )
        self._replay_pending_events(run_id)

    def _mark_turn_waiting(self) -> None:
        self._active_run_id = None
        self._set_input_state(
            disabled=False,
            placeholder="Message",
            button_label="Send",
        )

    def _set_waiting_for_connection(self) -> None:
        self._set_input_state(
            disabled=True,
            placeholder="Connecting...",
            button_label="Send",
        )

    def _should_buffer_event(self, run_id: str) -> bool:
        return (
            self._awaiting_message_result
            and self._active_run_id is None
        )

    def _should_render_event(self, event: Mapping[str, Any]) -> bool:
        event_type = str(event.get("type", ""))
        if event_type == "log.line":
            return True
        if event_type not in RUN_SCOPED_EVENT_TYPES:
            return False

        run_id = _run_id_from(event)
        return self._active_run_id is not None and run_id == self._active_run_id

    def _replay_pending_events(self, run_id: str) -> None:
        for event in self._pending_events_by_run.pop(run_id, []):
            if self._should_render_event(event):
                self._render_event_payload(event)

    def _finish_active_turn_if_matching(self, event: Mapping[str, Any]) -> None:
        if self._active_run_id is None:
            return
        if event.get("run_id") != self._active_run_id:
            return

        self._mark_turn_waiting()

    def _set_input_state(
        self,
        *,
        disabled: bool,
        placeholder: str,
        button_label: str,
    ) -> None:
        try:
            message_input = self.query_one("#goal-input", Input)
            send_button = self.query_one("#run-button", Button)
        except NoMatches:
            return

        message_input.disabled = disabled
        message_input.placeholder = placeholder
        send_button.disabled = disabled
        send_button.label = button_label


def _mapping_from(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _run_id_from(event: Mapping[str, Any]) -> str | None:
    run_id = event.get("run_id")
    return run_id if isinstance(run_id, str) else None


def _int_from(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _preview(value: str, limit: int) -> str:
    return value[:limit] + "…" if len(value) > limit else value


def _params_str(params: Mapping[str, Any]) -> str:
    return json.dumps(params, ensure_ascii=False)


def main() -> int:
    MyClaudeTui().run()
    return 0
