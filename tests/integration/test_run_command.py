from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import pytest

import my_claude.core.app as app_module
import my_claude.core.runner as runner_module
from my_claude.agent.agent import Agent, AgentResult
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
from my_claude.cli.commands.run import StdoutPrinter, _run_over_socket
from my_claude.cli.main import main
from my_claude.core.app import register_routes
from my_claude.core.bus.command import AGENT_RUN_METHOD, EVENT_SUBSCRIBE_METHOD, BusResult
from my_claude.core.bus.envelope import JsonRpcRequest
from my_claude.core.config import AppConfig, load_config
from my_claude.core.context import AnthropicMessage
from my_claude.core.events.bus import EventBus
from my_claude.core.runner import (
    RunContext,
    RunResult,
    prepare_run_context,
    run_goal,
    run_prepared_context,
)
from my_claude.core.trace.writer import TraceWriter
from my_claude.core.transport.socket_client import SocketClient
from my_claude.core.transport.socket_server import RouteHandler, TCPServer
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
    tmp_path: Path,
) -> None:
    exit_code, methods = asyncio.run(_run_via_test_daemon("write tests", tmp_path))

    captured = capsys.readouterr()

    assert exit_code == 0
    assert methods[:2] == [EVENT_SUBSCRIBE_METHOD, AGENT_RUN_METHOD]
    assert "[run] 20" in captured.out
    assert "[step 1] planning..." in captured.out
    assert "Local LLM placeholder accepted goal: write tests" in captured.out
    assert "[step 1] done" in captured.out
    assert "[run] success  1 steps" in captured.out


def test_daemon_writes_all_layers_to_shared_trace_channel(tmp_path: Path) -> None:
    records = asyncio.run(_run_via_test_daemon_with_shared_trace_writer("write tests", tmp_path))

    layers = {record["layer"] for record in records}
    directions = {record["direction"] for record in records}
    kinds = {record["kind"] for record in records}

    assert "ipc" in layers
    assert "event" in layers
    assert "llm" in layers
    assert {"CLIENT→CORE", "CORE→CLIENT", "CORE", "CORE→LLM", "LLM→CORE"} <= directions
    assert {"command", "response", "push", "event", "api_call", "api_response"} <= kinds


def test_agent_run_route_returns_run_id_before_background_run_finishes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, background_started, background_finished = asyncio.run(
        _agent_run_returns_before_background_finish(monkeypatch, tmp_path)
    )

    assert result["accepted"] is True
    assert result["run_id"]
    assert result["goal"] == "write tests"
    timeline_path = result["timeline_path"]
    assert isinstance(timeline_path, str)
    assert timeline_path.endswith("events.jsonl")
    assert background_started is True
    assert background_finished is False


def test_agent_run_route_accepts_concurrent_background_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first, second, started_count, remaining_tasks = asyncio.run(
        _agent_run_accepts_concurrent_background_runs(monkeypatch, tmp_path)
    )

    assert first["accepted"] is True
    assert second["accepted"] is True
    assert first["run_id"] != second["run_id"]
    assert first["goal"] == "first goal"
    assert second["goal"] == "second goal"
    assert started_count == 2
    assert remaining_tasks == 0


async def _agent_run_returns_before_background_finish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[dict[str, object], bool, bool]:
    started = asyncio.Event()
    release = asyncio.Event()
    finished = False

    async def slow_run_prepared_context(
        context: RunContext,
        *,
        listeners: Sequence[object] = (),
        trace_writer: object | None = None,
    ) -> RunResult:
        del trace_writer
        nonlocal finished
        started.set()
        await release.wait()
        finished = True
        return RunResult(
            run_id=context.run_id,
            run_dir=context.run_dir,
            timeline_path=context.timeline_path,
            agent_result=AgentResult(
                goal=context.working_memory.goal,
                final_response="done",
            ),
        )

    monkeypatch.setattr(app_module, "run_prepared_context", slow_run_prepared_context)

    config = AppConfig(runs_dir=tmp_path / "runs", llm_provider="local")
    server = TCPServer("127.0.0.1", 0, max_request_bytes=config.max_request_bytes)
    register_routes(server, config=config, server_version="test-version")
    await server.start()

    try:
        async with SocketClient(
            "127.0.0.1",
            _bound_port(server),
            timeout_seconds=1.0,
        ) as client:
            result = await client.request(AGENT_RUN_METHOD, {"goal": "write tests"})
            await asyncio.wait_for(started.wait(), timeout=1.0)
            return result, started.is_set(), finished
    finally:
        release.set()
        await asyncio.sleep(0)
        await server.shutdown()


async def _agent_run_accepts_concurrent_background_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object], int, int]:
    started_run_ids: list[str] = []
    release = asyncio.Event()

    async def slow_run_prepared_context(
        context: RunContext,
        *,
        listeners: Sequence[object] = (),
        trace_writer: object | None = None,
    ) -> RunResult:
        del listeners, trace_writer
        started_run_ids.append(context.run_id)
        await release.wait()
        return RunResult(
            run_id=context.run_id,
            run_dir=context.run_dir,
            timeline_path=context.timeline_path,
            agent_result=AgentResult(
                goal=context.working_memory.goal,
                final_response="done",
            ),
        )

    monkeypatch.setattr(app_module, "run_prepared_context", slow_run_prepared_context)

    config = AppConfig(runs_dir=tmp_path / "runs", llm_provider="local")
    run_tasks: set[asyncio.Task[RunResult]] = set()
    server = TCPServer("127.0.0.1", 0, max_request_bytes=config.max_request_bytes)
    register_routes(
        server,
        config=config,
        server_version="test-version",
        run_tasks=run_tasks,
    )
    await server.start()

    try:
        async with SocketClient(
            "127.0.0.1",
            _bound_port(server),
            timeout_seconds=1.0,
        ) as client:
            first = await client.request(AGENT_RUN_METHOD, {"goal": " first goal "})
            second = await client.request(AGENT_RUN_METHOD, {"goal": "second goal"})
            await _wait_until(lambda: len(started_run_ids) == 2)
    finally:
        release.set()
        if run_tasks:
            await asyncio.gather(*tuple(run_tasks), return_exceptions=True)
        await server.shutdown()

    return first, second, len(started_run_ids), len(run_tasks)


def test_agent_run_route_passes_shared_trace_writer_to_background_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert asyncio.run(_agent_run_uses_shared_trace_writer(monkeypatch, tmp_path))


async def _agent_run_uses_shared_trace_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> bool:
    seen_trace_writer: object | None = None

    async def recording_run_prepared_context(
        context: RunContext,
        *,
        listeners: Sequence[object] = (),
        trace_writer: object | None = None,
    ) -> RunResult:
        del listeners
        nonlocal seen_trace_writer
        seen_trace_writer = trace_writer
        return RunResult(
            run_id=context.run_id,
            run_dir=context.run_dir,
            timeline_path=context.timeline_path,
            agent_result=AgentResult(
                goal=context.working_memory.goal,
                final_response="done",
            ),
        )

    monkeypatch.setattr(
        app_module,
        "run_prepared_context",
        recording_run_prepared_context,
    )

    config = AppConfig(runs_dir=tmp_path / "runs", llm_provider="local")
    trace_writer = TraceWriter(tmp_path / "daemon.jsonl")
    server = TCPServer(
        "127.0.0.1",
        0,
        max_request_bytes=config.max_request_bytes,
        trace_emitter=trace_writer,
    )
    register_routes(
        server,
        config=config,
        server_version="test-version",
        trace_writer=trace_writer,
    )
    await trace_writer.start()
    await server.start()

    try:
        async with SocketClient(
            "127.0.0.1",
            _bound_port(server),
            timeout_seconds=1.0,
        ) as client:
            await client.request(AGENT_RUN_METHOD, {"goal": "write tests"})
            await asyncio.sleep(0)
            return seen_trace_writer is trace_writer and server._trace_emitter is trace_writer
    finally:
        await server.shutdown()
        await trace_writer.stop()


async def _run_via_test_daemon(goal: str, tmp_path: Path) -> tuple[int, list[str]]:
    config = AppConfig(runs_dir=tmp_path / "runs", llm_provider="local")
    server = TCPServer("127.0.0.1", 0, max_request_bytes=config.max_request_bytes)
    register_routes(server, config=config, server_version="test-version")
    methods: list[str] = []

    for method, handler in tuple(server.routes.items()):

        async def recording_handler(
            request: JsonRpcRequest,
            *,
            method: str = method,
            handler: object = handler,
        ) -> BusResult:
            methods.append(method)
            typed_handler = cast(RouteHandler, handler)
            return await typed_handler(request)

        server.routes[method] = recording_handler

    await server.start()
    try:
        client_config = config.model_copy(
            update={
                "core_host": "127.0.0.1",
                "core_port": _bound_port(server),
            }
        )
        exit_code = await _run_over_socket(goal, config=client_config, printer=StdoutPrinter())
        return exit_code, methods
    finally:
        await server.shutdown()


async def _run_via_test_daemon_with_shared_trace_writer(
    goal: str,
    tmp_path: Path,
) -> list[dict[str, object]]:
    config = AppConfig(runs_dir=tmp_path / "runs", llm_provider="local")
    trace_path = tmp_path / "daemon.jsonl"
    trace_writer = TraceWriter(trace_path)
    server = TCPServer(
        "127.0.0.1",
        0,
        max_request_bytes=config.max_request_bytes,
        trace_emitter=trace_writer,
    )
    register_routes(
        server,
        config=config,
        server_version="test-version",
        trace_writer=trace_writer,
    )

    await trace_writer.start()
    await server.start()
    try:
        client_config = config.model_copy(
            update={
                "core_host": "127.0.0.1",
                "core_port": _bound_port(server),
            }
        )
        await _run_over_socket(goal, config=client_config, printer=StdoutPrinter())
    finally:
        await server.shutdown()
        await trace_writer.stop()

    return [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]


def _bound_port(server: TCPServer) -> int:
    if server._server is None or not server._server.sockets:
        raise RuntimeError("test server did not start")

    return int(server._server.sockets[0].getsockname()[1])


async def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout_seconds: float = 1.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("condition was not met before timeout")
        await asyncio.sleep(0)


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
    assert context.workspace_dir == context.run_dir
    assert context.tasks_dir == context.run_dir / ".tasks"
    assert context.tasks_dir.exists()
    assert context.timeline_path == context.run_dir / "events.jsonl"
    assert context.trace_path == context.run_dir / "trace.jsonl"
    assert context.working_memory.run_id == context.run_id
    assert context.working_memory.goal == "ship it"
    assert context.working_memory.max_steps == config.agent_max_iterations
    assert context.working_memory.step == 0
    assert context.working_memory.status == RunStatus.PENDING
    assert context.working_memory.messages[0].role == "user"
    assert isinstance(context.working_memory.messages[0].content[0], TextBlock)
    assert context.working_memory.messages[0].content[0].text == "ship it"
    assert [definition.name for definition in context.tools.definitions()] == [
        "read_file",
        "write_file",
        "list_dir",
        "bash",
        "task_create",
        "task_update",
        "task_list",
        "task_get",
    ]
    assert context.loop_controller.should_continue()


def test_runner_writes_timeline_file(tmp_path: Path) -> None:
    config = AppConfig(runs_dir=tmp_path / "runs")
    result = asyncio.run(run_goal("ship it", config=config))

    lines = result.timeline_path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines]

    assert result.run_dir.exists()
    assert result.agent_result.goal == "ship it"
    assert [event["type"] for event in events] == [
        "run.started",
        "step.started",
        "llm.request_started",
        "llm.response_completed",
        "step.finished",
        "run.finished",
    ]
    assert events[0]["data"]["goal"] == "ship it"
    assert events[1]["data"]["step"] == 1


def test_runner_writes_event_bus_trace_file(tmp_path: Path) -> None:
    config = AppConfig(runs_dir=tmp_path / "runs")
    result = asyncio.run(run_goal("ship it", config=config))
    trace_path = result.run_dir / "trace.jsonl"

    assert trace_path.exists()
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    event_records = [record for record in records if record["layer"] == "event"]
    llm_records = [record for record in records if record["layer"] == "llm"]

    assert [record["data"]["type"] for record in event_records] == [
        "run.started",
        "step.started",
        "llm.request_started",
        "llm.response_completed",
        "step.finished",
        "run.finished",
    ]
    assert all(record["direction"] == "CORE" for record in event_records)
    assert all(record["kind"] == "event" for record in event_records)
    assert event_records[0]["data"]["goal"] == "ship it"
    assert event_records[0]["data"]["run_id"] == result.run_id
    assert [record["kind"] for record in llm_records] == ["api_call", "api_response"]
    assert [record["step"] for record in llm_records] == [1, 1]
    assert "messages" in llm_records[0]["data"]
    assert "text" in llm_records[1]["data"]


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
            *,
            step: int | None = None,
        ) -> LLMResponse:
            del messages, tools, step
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

    assert events[-1]["type"] == "run.cancelled"
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
        "type": "llm.token",
        "message": "llm token",
        "token": "hello",
        "index": 1,
        "run_id": None,
    }
