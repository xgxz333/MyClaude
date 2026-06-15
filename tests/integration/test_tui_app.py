from __future__ import annotations

from textual.widget import Widget
from textual.widgets import Static

from my_claude.agent.events import (
    ContextCompactedEvent,
    LLMResponseCompletedEvent,
    LLMTokenEvent,
    LLMUsageEvent,
    PermissionRequestedEvent,
    RunCompletedEvent,
    RunStartedEvent,
    StepStartedEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from my_claude.core.transport.socket_client import SocketClientError
from my_claude.tui.app import (
    LLMStreamBlock,
    MyClaudeTui,
    PermissionBlock,
    PermissionSelect,
    ToolCallBlock,
)


def test_tui_accumulates_streaming_llm_tokens_in_place() -> None:
    app = RenderlessTui()
    app._mark_turn_running("run-1")

    app._write_event(LLMTokenEvent(token="hello", run_id="run-1"))
    app._write_event(LLMTokenEvent(token=" world", run_id="run-1"))

    assert len(app.mounted_widgets) == 1
    block = app.mounted_widgets[0]
    assert isinstance(block, LLMStreamBlock)
    assert block.text == "hello world"


def test_tui_uses_completed_response_when_no_tokens_arrived() -> None:
    app = RenderlessTui()
    app._mark_turn_running("run-1")

    app._write_event(LLMResponseCompletedEvent(content="final answer", run_id="run-1"))

    assert len(app.mounted_widgets) == 1
    block = app.mounted_widgets[0]
    assert isinstance(block, LLMStreamBlock)
    assert block.text == "final answer"


def test_tui_does_not_duplicate_completed_response_after_streaming_tokens() -> None:
    app = RenderlessTui()
    app._mark_turn_running("run-1")

    app._write_event(LLMTokenEvent(token="final", run_id="run-1"))
    app._write_event(LLMResponseCompletedEvent(content="final answer", run_id="run-1"))

    assert len(app.mounted_widgets) == 1
    block = app.mounted_widgets[0]
    assert isinstance(block, LLMStreamBlock)
    assert block.text == "final"


def test_tui_renders_run_step_and_usage_like_kama() -> None:
    app = RenderlessTui()
    app._mark_turn_running("run-1")

    app._write_event(RunStartedEvent(goal="ship it", run_id="run-1"))
    app._write_event(StepStartedEvent(run_id="run-1", step=2))
    app._write_event(
        LLMUsageEvent(
            run_id="run-1",
            input_tokens=10,
            output_tokens=4,
            cache_read_input_tokens=1,
            context_pct=0.63,
        )
    )
    app._write_event(RunCompletedEvent(goal="ship it", run_id="run-1", steps=2))

    assert app.static_texts == [
        "[bold cyan]▶ run[/bold cyan]  [dim]run-1[/dim]\n  [dim]goal:[/dim] ship it",
        "[dim]── step 2 ────────────────────────────────────────────────[/dim]",
        "[dim]  tokens  in=10 out=4 cache=1[/dim]  [dim]ctx:63.0% ████████████░░░░░░░░[/dim]",
        "[bold green]✓ completed[/bold green]  [dim]2 steps[/dim]",
    ]
    assert app._last_context_pct == 0.63


def test_tui_handles_bad_context_pct_without_crashing() -> None:
    app = RenderlessTui()
    app._mark_turn_running("run-1")

    app._receive_event_payload(
        {
            "type": "llm.usage",
            "run_id": "run-1",
            "input_tokens": 10,
            "output_tokens": 4,
            "cache_read_input_tokens": 1,
            "context_pct": "not-a-number",
        }
    )

    assert app.static_texts == [
        "[dim]  tokens  in=10 out=4 cache=1[/dim]  [dim]ctx:0.0% ░░░░░░░░░░░░░░░░░░░░[/dim]"
    ]
    assert app._last_context_pct == 0.0


def test_tui_renders_context_compacted_event_and_resets_watermark() -> None:
    app = RenderlessTui()
    app._last_context_pct = 0.91

    app._write_event(
        ContextCompactedEvent(
            session_id="sess-1",
            run_id="run-1",
            original_tokens=12,
            summary_tokens=2,
            ts="2026-06-15T00:00:00Z",
        )
    )

    assert app._last_context_pct == 0.0
    assert app.static_texts == [
        "[bold cyan]⚡ Context compacted[/bold cyan]  "
        "[dim]original≈12 tokens → summary=2 tokens[/dim]"
    ]


def test_tui_ignores_events_from_other_runs() -> None:
    app = RenderlessTui()
    app._mark_turn_running("run-1")

    app._write_event(RunStartedEvent(goal="other", run_id="run-2"))
    app._write_event(LLMTokenEvent(token="external", run_id="run-2"))
    app._write_event(RunCompletedEvent(goal="other", run_id="run-2", steps=1))

    assert app.mounted_widgets == []
    assert app._active_run_id == "run-1"


def test_tui_replays_early_events_after_session_message_is_accepted() -> None:
    app = RenderlessTui()
    app._awaiting_message_result = True

    app._write_event(RunStartedEvent(goal="ship it", run_id="run-1"))
    app._write_event(LLMTokenEvent(token="early", run_id="run-1"))

    assert app.mounted_widgets == []

    app._awaiting_message_result = False
    app._mark_turn_running("run-1")

    assert len(app.mounted_widgets) == 2
    block = app.mounted_widgets[1]
    assert isinstance(block, LLMStreamBlock)
    assert block.text == "early"


def test_tui_unlocks_message_input_after_active_session_run_finishes() -> None:
    app = RenderlessTui()

    app._mark_turn_running("run-1")
    app._write_event(RunCompletedEvent(goal="other", run_id="run-2", steps=1))

    assert app._active_run_id == "run-1"
    assert app.input_states[-1] == (True, "Waiting for run-1...", "Busy")

    app._write_event(RunCompletedEvent(goal="ship it", run_id="run-1", steps=2))

    assert app._active_run_id is None
    assert app.input_states[-1] == (False, "Message", "Send")


def test_tui_renders_tool_calls_as_collapsible_blocks_like_kama() -> None:
    app = RenderlessTui()
    app._mark_turn_running("run-1")

    app._write_event(
        ToolCallStartedEvent(
            run_id="run-1",
            tool_use_id="tool-1",
            tool_name="bash",
            arguments={"command": "echo hi"},
        )
    )
    app._write_event(
        ToolCallCompletedEvent(
            run_id="run-1",
            tool_use_id="tool-1",
            tool_name="bash",
            result="hi",
            elapsed_ms=12,
        )
    )

    assert len(app.mounted_widgets) == 1
    block = app.mounted_widgets[0]
    assert isinstance(block, ToolCallBlock)
    assert block.tool_name == "bash"
    assert block.params == {"command": "echo hi"}
    assert block.output == "hi"
    assert block.elapsed_ms == 12
    assert block.summary == (
        '  [bold yellow]✎[/bold yellow] [bold]bash[/bold]  '
        '[dim]{"command": "echo hi"}[/dim]\n'
        "  [dim]↳[/dim] [dim]hi[/dim]  [dim]12ms[/dim]"
    )


def test_tui_renders_permission_request_with_focusable_select() -> None:
    app = RenderlessTui()
    app._mark_turn_running("run-1")

    app._write_event(
        PermissionRequestedEvent(
            run_id="run-1",
            tool_use_id="tool-1",
            tool_name="bash",
            params={"command": "cat /etc/passwd"},
            param_preview="cat /etc/passwd",
        )
    )

    assert len(app.mounted_widgets) == 2
    block = app.mounted_widgets[0]
    select = app.mounted_widgets[1]
    assert isinstance(block, PermissionBlock)
    assert isinstance(select, PermissionSelect)
    assert app._pending_permission_blocks["tool-1"] is block
    assert select.tool_use_id == "tool-1"


def test_tui_submit_message_schedules_background_worker() -> None:
    app = WorkerRecordingTui()
    app._client = ConnectedClient()
    app._session_id = "sess-1"
    app.message_input.value = "ship it"

    app._submit_message()

    assert app.message_input.value == ""
    assert app.worker_calls == [("send_message", False)]
    assert app.static_texts == ["[bold]you[/bold]  ship it"]
    assert app.input_states[-1] == (True, "Sending...", "Busy")


def test_tui_compact_command_schedules_compact_worker() -> None:
    app = WorkerRecordingTui()
    app._client = ConnectedClient()
    app._session_id = "sess-1"
    app.message_input.value = "/compact"

    app._submit_message()

    assert app.message_input.value == ""
    assert app.worker_calls == [("compact", False)]
    assert app.static_texts == []


def test_tui_do_compact_renders_result() -> None:
    app = RenderlessTui()
    app._client = CompactClient()
    app._session_id = "sess-1"
    app._config = app._config.model_copy(
        update={"ipc_timeout_seconds": 5.0, "llm_timeout_seconds": 120.0}
    )

    import asyncio

    asyncio.run(app._do_compact())

    assert app._last_context_pct == 0.0
    assert app.static_texts == [
        "[dim]⚡ compacting context...[/dim]",
        "[bold cyan]⚡ Context compacted[/bold cyan]  "
        "[dim]summary=12 tokens  saved≈34 tokens[/dim]",
    ]
    assert app._client.timeout_seconds == 120.0


def test_tui_do_compact_renders_empty_history_error() -> None:
    app = RenderlessTui()
    app._client = CompactErrorClient(-32021, "compaction failed or not beneficial")
    app._session_id = "sess-1"

    import asyncio

    asyncio.run(app._do_compact())

    assert app.static_texts == [
        "[dim]⚡ compacting context...[/dim]",
        "[red]compact error:[/red] nothing to compact yet; "
        "send a message and wait for it to finish first",
    ]


def test_tool_call_block_click_lazily_loads_details_and_toggles_state() -> None:
    block = ToolCallBlock(
        "write_file",
        {"path": "README.md", "content": "ok"},
    )
    list(block.compose())
    block.set_result("written", 5)

    block.on_click()
    assert block.has_class("expanded")

    block.on_click()
    assert not block.has_class("expanded")


def test_tool_call_block_summary_previews_long_output() -> None:
    block = ToolCallBlock(
        "bash",
        {"command": "printf long"},
    )

    block.set_result("x" * 80, 3)

    assert "[dim]▸ click to expand[/dim]" in block.summary
    assert "x" * 50 in block.summary
    assert "…" in block.summary


class RenderlessTui(MyClaudeTui):
    def __init__(self) -> None:
        super().__init__()
        self.mounted_widgets: list[Widget] = []
        self.input_states: list[tuple[bool, str, str]] = []

    @property
    def static_texts(self) -> list[str]:
        return [
            str(widget.renderable)
            for widget in self.mounted_widgets
            if isinstance(widget, Static)
        ]

    def _append(self, widget: Widget) -> None:
        self.mounted_widgets.append(widget)

    def _set_input_state(
        self,
        *,
        disabled: bool,
        placeholder: str,
        button_label: str,
    ) -> None:
        self.input_states.append((disabled, placeholder, button_label))


class FakeInput:
    def __init__(self) -> None:
        self.value = ""
        self.disabled = False
        self.placeholder = ""


class FakeButton:
    def __init__(self) -> None:
        self.disabled = False
        self.label = ""


class ConnectedClient:
    is_connected = True

    async def request(self, _method: str, _params: dict[str, str]) -> dict[str, str]:
        raise AssertionError("request should run only inside the worker")


class CompactClient:
    is_connected = True
    timeout_seconds: float | None = None

    async def request(
        self,
        method: str,
        params: dict[str, str],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, int]:
        assert method == "session.compact"
        assert params == {"session_id": "sess-1", "focus": ""}
        self.timeout_seconds = timeout_seconds
        return {"summary_tokens": 12, "saved_tokens": 34}


class CompactErrorClient:
    is_connected = True

    def __init__(self, code: int, message: str) -> None:
        self._code = code
        self._message = message

    async def request(
        self,
        method: str,
        params: dict[str, str],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, int]:
        del timeout_seconds
        assert method == "session.compact"
        assert params == {"session_id": "sess-1", "focus": ""}
        raise SocketClientError(self._message, code=self._code)


class WorkerRecordingTui(RenderlessTui):
    def __init__(self) -> None:
        super().__init__()
        self.message_input = FakeInput()
        self.send_button = FakeButton()
        self.worker_calls: list[tuple[str | None, bool]] = []

    def query_one(self, selector: str, widget_type: object = None) -> object:
        del widget_type
        if selector == "#goal-input":
            return self.message_input
        if selector == "#run-button":
            return self.send_button
        raise LookupError(selector)

    def run_worker(
        self,
        work: object,
        *,
        name: str | None = None,
        exclusive: bool = True,
        exit_on_error: bool = True,
        **kwargs: object,
    ) -> None:
        del exit_on_error, kwargs
        self.worker_calls.append((name, exclusive))
        close = getattr(work, "close", None)
        if callable(close):
            close()
