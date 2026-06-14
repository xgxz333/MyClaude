from __future__ import annotations

from textual.widget import Widget
from textual.widgets import Static

from my_claude.agent.events import (
    LLMResponseCompletedEvent,
    LLMTokenEvent,
    LLMUsageEvent,
    RunCompletedEvent,
    RunStartedEvent,
    StepStartedEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from my_claude.tui.app import LLMStreamBlock, MyClaudeTui, ToolCallBlock


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
        )
    )
    app._write_event(RunCompletedEvent(goal="ship it", run_id="run-1", steps=2))

    assert app.static_texts == [
        "[bold cyan]▶ run[/bold cyan]  [dim]run-1[/dim]\n  [dim]goal:[/dim] ship it",
        "[dim]── step 2 ────────────────────────────────────────────────[/dim]",
        "[dim]  tokens  in=10 out=4 cache=1[/dim]",
        "[bold green]✓ completed[/bold green]  [dim]2 steps[/dim]",
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
