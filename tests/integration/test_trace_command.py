from __future__ import annotations

import json
from pathlib import Path

import pytest

from my_claude.cli.commands.trace import format_trace_record, read_trace_records
from my_claude.cli.main import main
from my_claude.core.trace import TraceDirection, TraceLayer, TraceRecord


def test_trace_command_prints_global_timeline(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "runs" / "daemon.jsonl"
    _write_trace_records(
        trace_path,
        [
            _record(
                direction="CLIENT→CORE",
                layer="ipc",
                kind="command",
                data={"method": "agent.run", "params": {"goal": "ship it"}},
            ),
            _record(
                direction="LLM→CORE",
                layer="llm",
                kind="api_response",
                step=1,
                data={"latency_ms": 12, "usage": {"output_tokens": 5}},
            ),
        ],
    )
    monkeypatch.setenv("MYCLAUDE_RUNS_DIR", str(tmp_path / "runs"))

    exit_code = main(["trace"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "CLIENT→CORE" in captured.out
    assert "agent.run" in captured.out
    assert "LLM→CORE" in captured.out
    assert "latency=12ms" in captured.out


def test_trace_command_filters_run_id_and_raw_json(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "runs" / "daemon.jsonl"
    _write_trace_records(
        trace_path,
        [
            _record(run_id="run-2", data={"type": "run.started"}),
            _record(run_id="run-1", data={"type": "run.started"}),
        ],
    )
    monkeypatch.setenv("MYCLAUDE_RUNS_DIR", str(tmp_path / "runs"))

    exit_code = main(["trace", "run-1", "--raw"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["run_id"] == "run-1"
    assert payload["data"]["type"] == "run.started"


def test_trace_command_filters_layer_and_direction(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "runs" / "daemon.jsonl"
    _write_trace_records(
        trace_path,
        [
            _record(direction="CORE", layer="event", kind="event", data={"type": "run.started"}),
            _record(direction="CORE→LLM", layer="llm", kind="api_call", data={"message_count": 1}),
        ],
    )
    monkeypatch.setenv("MYCLAUDE_RUNS_DIR", str(tmp_path / "runs"))

    exit_code = main(["trace", "--layer", "llm", "--direction", "CORE→LLM"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "api_call" in captured.out
    assert "run.started" not in captured.out


def test_trace_command_reports_missing_file(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MYCLAUDE_RUNS_DIR", str(tmp_path / "runs"))

    exit_code = main(["trace"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "trace file not found" in captured.err
    assert "daemon.jsonl" in captured.err


def test_read_trace_records_rejects_invalid_line(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("{invalid-json}\n", encoding="utf-8")

    try:
        read_trace_records(trace_path)
    except ValueError as error:
        assert "invalid trace record" in str(error)
    else:
        raise AssertionError("invalid trace line was accepted")


def test_read_trace_records_accepts_legacy_daemon_trace_lines(tmp_path: Path) -> None:
    trace_path = tmp_path / "daemon.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace",
                        "timestamp": "2026-06-12T04:08:46.749380Z",
                        "layer": "ipc",
                        "component": "socket_server",
                        "operation": "command.received",
                        "direction": "inbound",
                        "payload": {
                            "client": "127.0.0.1:36766",
                            "method": "agent.run",
                            "request_id": 2,
                            "params_kind": "object",
                            "byte_count": 91,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "trace",
                        "run_id": "run-1",
                        "timestamp": "2026-06-12T04:08:47.019714Z",
                        "layer": "llm",
                        "component": "tracing_provider",
                        "operation": "response",
                        "direction": "inbound",
                        "payload": {
                            "kind": "response",
                            "step": 1,
                            "elapsed_ms": 12,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    command, response = read_trace_records(trace_path)

    assert command.direction == "CLIENT→CORE"
    assert command.layer == "ipc"
    assert command.kind == "command"
    assert command.client_id == "127.0.0.1:36766"
    assert command.data["method"] == "agent.run"
    assert command.data["id"] == 2

    assert response.run_id == "run-1"
    assert response.direction == "LLM→CORE"
    assert response.layer == "llm"
    assert response.kind == "api_response"
    assert response.step == 1
    assert response.data["latency_ms"] == 12


def test_format_trace_record_summarizes_event_data() -> None:
    line = format_trace_record(
        _record(
            run_id="run-1",
            data={"type": "step.started", "run_id": "run-1", "step": 1},
        )
    )

    assert "event" in line
    assert "run=run-1" in line
    assert "type=step.started" in line


def _write_trace_records(path: Path, records: list[TraceRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(record.model_dump_json(exclude_none=True) for record in records) + "\n",
        encoding="utf-8",
    )


def _record(
    *,
    direction: TraceDirection = "CORE",
    layer: TraceLayer = "event",
    kind: str = "event",
    run_id: str | None = None,
    step: int | None = None,
    client_id: str | None = None,
    data: dict[str, object] | None = None,
) -> TraceRecord:
    return TraceRecord(
        ts="2026-06-12T12:00:00+00:00",
        direction=direction,
        layer=layer,
        kind=kind,
        run_id=run_id,
        step=step,
        client_id=client_id,
        data=data or {},
    )
