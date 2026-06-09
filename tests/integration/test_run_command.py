from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import pytest

import my_claude.core.runner as runner_module
from my_claude.agent.agent import Agent
from my_claude.agent.events import (
    AgentEvent,
    AgentEventType,
    LLMRequestStartedEvent,
    LLMResponseCompletedEvent,
    LLMTokenEvent,
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
    StepFinishedEvent,
    StepStartedEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from my_claude.agent.memory import RunStatus, TextBlock, WorkingMemory
from my_claude.agent.tools import ToolDefinition, ToolRegistry
from my_claude.cli.commands.run import StdoutPrinter
from my_claude.cli.main import main
from my_claude.core.config import AppConfig, load_config
from my_claude.core.context import AnthropicMessage
from my_claude.core.events.bus import EventBus
from my_claude.core.runner import (
    prepare_run_context,
    run_goal,
    run_prepared_context,
)
from my_claude.llm.client import (
    LLMResponse,
    LocalLLMClient,
    _anthropic_messages_url,
    create_llm_client,
)


def test_run_command_requires_goal(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["run"])

    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert "the following arguments are required: --goal" in captured.err


def test_run_command_accepts_goal(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MYCLAUDE_LLM_PROVIDER", "local")
    monkeypatch.delenv("MYCLAUDE_LLM_API_KEY", raising=False)
    monkeypatch.setenv("MYCLAUDE_RUNS_DIR", str(tmp_path / "runs"))

    exit_code = main(["run", "--goal", "write tests"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "[run] 20" in captured.out
    assert "[step 1] planning..." in captured.out
    assert "Local LLM placeholder accepted goal: write tests" in captured.out
    assert "[step 1] done" in captured.out
    assert "[run] success  1 steps" in captured.out


def test_config_accepts_anthropic_env_aliases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
                [
                    "MYCLAUDE_LLM_PROVIDER=anthropic",
                    "ANTHROPIC_API_KEY=test-key",
                    "ANTHROPIC_BASE_URL=https://ai.prism.uno/v1",
                "ANTHROPIC_MODEL=claude-test",
                "ANTHROPIC_MAX_TOKENS=256",
            ]
        ),
        encoding="utf-8",
    )
    for key in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
            "ANTHROPIC_MAX_TOKENS",
            "MYCLAUDE_LLM_PROVIDER",
            "MYCLAUDE_LLM_API_KEY",
        "MYCLAUDE_LLM_BASE_URL",
        "MYCLAUDE_LLM_MODEL",
        "MYCLAUDE_LLM_MAX_TOKENS",
    ):
        monkeypatch.delenv(key, raising=False)

    config = load_config(config_file=tmp_path / "missing.toml", env_file=env_file)

    assert config.llm_api_key == "test-key"
    assert config.llm_base_url == "https://ai.prism.uno/v1"
    assert config.llm_model == "claude-test"
    assert config.llm_max_tokens == 256
    assert os.environ["ANTHROPIC_API_KEY"] == "test-key"
    assert os.environ["ANTHROPIC_BASE_URL"] == "https://ai.prism.uno"


def test_anthropic_provider_normalizes_base_url_to_messages_endpoint() -> None:
    assert _anthropic_messages_url("https://ai.prism.uno/v1") == (
        "https://ai.prism.uno/v1/messages"
    )
    assert _anthropic_messages_url("https://ai.prism.uno/v1/messages") == (
        "https://ai.prism.uno/v1/messages"
    )
    assert _anthropic_messages_url("https://ai.prism.uno") == (
        "https://ai.prism.uno/v1/messages"
    )


def test_anthropic_client_syncs_config_to_sdk_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    create_llm_client(
        AppConfig(
            llm_provider="anthropic",
            llm_api_key="test-key",
            llm_base_url="https://ai.prism.uno/v1/messages",
            llm_model="claude-test",
        )
    )

    assert os.environ["ANTHROPIC_API_KEY"] == "test-key"
    assert os.environ["ANTHROPIC_BASE_URL"] == "https://ai.prism.uno"


def test_prepare_run_context_assembles_runtime_before_agent_loop(tmp_path: Path) -> None:
    config = AppConfig(runs_dir=tmp_path / "runs")
    context = prepare_run_context("ship it", config=config)

    assert context.run_id
    assert context.run_dir.exists()
    assert context.run_dir.parent == tmp_path / "runs"
    assert context.timeline_path == context.run_dir / "events.jsonl"
    assert context.working_memory.run_id == context.run_id
    assert context.working_memory.goal == "ship it"
    assert context.working_memory.max_steps == config.agent_max_iterations
    assert context.working_memory.step == 0
    assert context.working_memory.status == RunStatus.PENDING
    assert context.working_memory.messages[0].role == "user"
    assert isinstance(context.working_memory.messages[0].content[0], TextBlock)
    assert context.working_memory.messages[0].content[0].text == "ship it"
    assert [definition.name for definition in context.tools.definitions()] == ["read_file"]
    assert context.loop_controller.should_continue()


def test_runner_writes_timeline_file(tmp_path: Path) -> None:
    config = AppConfig(runs_dir=tmp_path / "runs")
    result = asyncio.run(run_goal("ship it", config=config))

    lines = result.timeline_path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines]

    assert result.run_dir.exists()
    assert result.agent_result.goal == "ship it"
    assert [event["type"] for event in events] == [
        "run_started",
        "step_started",
        "llm_request_started",
        "llm_response_completed",
        "step_finished",
        "run_completed",
    ]
    assert events[0]["data"]["goal"] == "ship it"
    assert events[1]["data"]["step"] == 1


def test_prepared_runner_writes_timeline_and_broadcasts_events(tmp_path: Path) -> None:
    broadcast_events: list[AgentEventType] = []

    async def broadcaster(event: AgentEvent) -> None:
        broadcast_events.append(event.type)

    config = AppConfig(runs_dir=tmp_path / "runs")
    context = prepare_run_context("ship it", config=config)

    result = asyncio.run(run_prepared_context(context, listeners=[broadcaster]))
    timeline_events = [
        json.loads(line)["type"]
        for line in result.timeline_path.read_text(encoding="utf-8").splitlines()
    ]

    assert timeline_events == [event_type.value for event_type in broadcast_events]
    assert broadcast_events == [
        AgentEventType.RUN_STARTED,
        AgentEventType.STEP_STARTED,
        AgentEventType.LLM_REQUEST_STARTED,
        AgentEventType.LLM_RESPONSE_COMPLETED,
        AgentEventType.STEP_FINISHED,
        AgentEventType.RUN_COMPLETED,
    ]


def test_runner_publishes_run_started_before_initializing_llm_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[AgentEventType] = []
    events_seen_when_llm_initialized: list[AgentEventType] = []

    async def listener(event: AgentEvent) -> None:
        events.append(event.type)

    def create_client(_config: AppConfig, **_kwargs: object) -> LocalLLMClient:
        events_seen_when_llm_initialized.extend(events)
        return LocalLLMClient()

    monkeypatch.setattr(runner_module, "create_llm_client", create_client)
    context = prepare_run_context("ship it", config=AppConfig(runs_dir=tmp_path / "runs"))

    asyncio.run(run_prepared_context(context, listeners=[listener]))

    assert events_seen_when_llm_initialized == [AgentEventType.RUN_STARTED]
    assert events[0] == AgentEventType.RUN_STARTED


def test_runner_marks_broadcasts_logs_closes_and_reraises_on_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class SlowLLMClient:
        def __init__(self, started: asyncio.Event) -> None:
            self._started = started

        async def complete(
            self,
            messages: Sequence[AnthropicMessage],
            tools: Sequence[ToolDefinition],
        ) -> LLMResponse:
            self._started.set()
            await asyncio.sleep(60)
            return LLMResponse(content="never")

    async def run_and_cancel() -> Path:
        started = asyncio.Event()
        monkeypatch.setattr(
            runner_module,
            "create_llm_client",
            lambda _config, **_kwargs: SlowLLMClient(started),
        )
        config = AppConfig(runs_dir=tmp_path / "runs")
        task = asyncio.create_task(run_goal("cancel me", config=config))
        await started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        return next(config.runs_dir.glob("*/events.jsonl"))

    caplog.set_level(logging.INFO)

    timeline_path = asyncio.run(run_and_cancel())
    events = [
        json.loads(line)
        for line in timeline_path.read_text(encoding="utf-8").splitlines()
    ]

    with timeline_path.open("a", encoding="utf-8") as file:
        file.write("")

    assert events[-1]["type"] == "run_cancelled"
    assert "async cancellation requested" in events[-1]["data"]["reason"]
    assert "run cancelled" in caplog.text


def test_working_memory_tracks_steps_status_and_dynamic_messages() -> None:
    memory = WorkingMemory.from_goal("ship it", run_id="run-1", max_steps=3)

    first_step = memory.mark_step_started(reason="loop entered")
    memory.append_assistant_text("I can do that.")
    memory.append_tool_result(tool_use_id="tool-1", content="done")
    memory.set_status(
        RunStatus.COMPLETED,
        reason="finished",
        transition_reason="all requested work completed",
    )

    assert first_step == 1
    assert memory.step == 1
    assert memory.status == RunStatus.COMPLETED
    assert memory.reason == "finished"
    assert memory.status_transition_reason == "all requested work completed"
    assert [message.role for message in memory.messages] == ["user", "assistant", "user"]
    assert memory.messages[0].content[0].type == "text"
    assert memory.messages[1].content[0].type == "text"
    assert memory.messages[2].content[0].type == "tool_result"


def test_agent_dispatches_events_to_handler() -> None:
    events: list[AgentEvent] = []

    async def handle(event: AgentEvent) -> None:
        events.append(event)

    agent = Agent(
        llm_client=LocalLLMClient(),
        tools=ToolRegistry(),
        event_handler=handle,
    )

    result = asyncio.run(agent.run("ship it"))

    assert result.goal == "ship it"
    assert result.final_response.startswith("Local LLM placeholder accepted goal: ship it")
    assert [event.type for event in events] == [
        AgentEventType.RUN_STARTED,
        AgentEventType.STEP_STARTED,
        AgentEventType.LLM_REQUEST_STARTED,
        AgentEventType.LLM_RESPONSE_COMPLETED,
        AgentEventType.STEP_FINISHED,
        AgentEventType.RUN_COMPLETED,
    ]
    assert events[0].data["run_id"] == "local"


def test_event_bus_broadcasts_run_and_tool_events_to_async_subscribers() -> None:
    bus: EventBus[AgentEvent] = EventBus()
    first_listener_events: list[AgentEventType] = []
    second_listener_events: list[AgentEventType] = []

    async def first_listener(event: AgentEvent) -> None:
        first_listener_events.append(event.type)

    async def second_listener(event: AgentEvent) -> None:
        second_listener_events.append(event.type)

    bus.subscribe(first_listener)
    bus.subscribe(second_listener)

    async def publish_events() -> None:
        await bus.publish(RunStartedEvent(goal="ship it"))
        await bus.publish(
            ToolCallCompletedEvent(
                tool_use_id="tool-1",
                tool_name="example",
                result="ok",
            )
        )

    asyncio.run(publish_events())

    assert first_listener_events == [
        AgentEventType.RUN_STARTED,
        AgentEventType.TOOL_CALL_COMPLETED,
    ]
    assert second_listener_events == first_listener_events


def test_stdout_printer_formats_tokens_tools_and_final_summary() -> None:
    stream = io.StringIO()
    error_stream = io.StringIO()
    bus: EventBus[AgentEvent] = EventBus()
    times = iter([10.0, 12.34])
    printer = StdoutPrinter(
        stream=stream,
        error_stream=error_stream,
        clock=lambda: next(times),
    )

    printer.subscribe(bus)

    async def publish_events() -> None:
        await bus.publish(RunStartedEvent(goal="ship it"))
        await bus.publish(StepStartedEvent(run_id="run-1", step=1))
        await bus.publish(LLMRequestStartedEvent(model_input_messages=1, tools=1))
        await bus.publish(LLMTokenEvent(token="hello"))
        await bus.publish(LLMTokenEvent(token=" world"))
        await bus.publish(
            ToolCallStartedEvent(
                tool_use_id="tool-1",
                tool_name="read_file",
                arguments={"path": "README.md"},
            )
        )
        await bus.publish(
            ToolCallCompletedEvent(
                tool_use_id="tool-1",
                tool_name="read_file",
                result="ok",
                elapsed_ms=1,
            )
        )
        await bus.publish(LLMResponseCompletedEvent(message="", content=""))
        await bus.publish(StepFinishedEvent(run_id="run-1", step=1))
        await bus.publish(RunCompletedEvent(goal="ship it"))

    asyncio.run(publish_events())

    output = stream.getvalue()
    assert "[step 1] planning..." in output
    assert 'hello world\n[tool] read_file {"path": "README.md"}' in output
    assert "[tool] read_file ✓  1ms" in output
    assert "[step 1] done" in output
    assert "[run] success  1 steps  2.3s" in output
    assert error_stream.getvalue() == ""


def test_stdout_printer_formats_run_failure_error() -> None:
    stream = io.StringIO()
    error_stream = io.StringIO()
    bus: EventBus[AgentEvent] = EventBus()
    times = iter([10.0, 12.0])
    printer = StdoutPrinter(
        stream=stream,
        error_stream=error_stream,
        clock=lambda: next(times),
    )

    printer.subscribe(bus)

    async def publish_events() -> None:
        await bus.publish(RunStartedEvent(goal="ship it"))
        await bus.publish(StepStartedEvent(run_id="run-1", step=1))
        await bus.publish(LLMRequestStartedEvent(model_input_messages=1, tools=0))
        await bus.publish(RunFailedEvent(error="LLM request failed: HTTP Error 503"))

    asyncio.run(publish_events())

    output = error_stream.getvalue()
    assert "[run] error: LLM request failed: HTTP Error 503" in output
    assert "[run] failed  1 steps  2.0s" in output


def test_llm_token_event_is_a_pydantic_event_model() -> None:
    event = LLMTokenEvent(token="hello", index=1)

    assert event.type == AgentEventType.LLM_TOKEN
    assert event.data == {"token": "hello", "index": 1}
    assert event.model_dump(mode="json") == {
        "type": "llm_token",
        "message": "llm token",
        "token": "hello",
        "index": 1,
        "run_id": None,
    }
