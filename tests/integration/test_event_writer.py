from __future__ import annotations

import asyncio
import json
from pathlib import Path

from my_claude.agent.events import RunStartedEvent, ToolCallCompletedEvent
from my_claude.core.events.writer import JsonlEventWriter, serialize_event


class RecordingTextIO:
    def __init__(self) -> None:
        self.values: list[str] = []
        self.flush_count = 0
        self.closed = False

    def write(self, value: str) -> int:
        self.values.append(value)
        return len(value)

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.closed = True


def test_jsonl_event_writer_writes_json_line_and_flushes_per_event(tmp_path: Path) -> None:
    event_file = RecordingTextIO()
    writer = JsonlEventWriter(tmp_path / "events.jsonl")
    writer._file = event_file  # type: ignore[assignment]

    asyncio.run(
        writer.handle(
            ToolCallCompletedEvent(
                tool_use_id="tool-1",
                tool_name="example",
                result="ok",
            )
        )
    )

    assert event_file.flush_count == 1
    assert len(event_file.values) == 1
    assert event_file.values[0].endswith("\n")

    payload = json.loads(event_file.values[0])
    assert payload["type"] == "tool_call_completed"
    assert payload["message"] == "tool call completed"
    assert payload["data"] == {"tool_use_id": "tool-1", "tool_name": "example", "result": "ok"}


def test_jsonl_event_writer_context_opens_and_closes_file(tmp_path: Path) -> None:
    path = tmp_path / "events" / "timeline.jsonl"
    event = RunStartedEvent(goal="ship it")

    with JsonlEventWriter(path) as writer:
        asyncio.run(writer.handle(event))

    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["type"] == "run_started"


def test_serialize_event_redacts_secrets_from_tool_results() -> None:
    payload = json.loads(
        serialize_event(
            ToolCallCompletedEvent(
                tool_use_id="tool-1",
                tool_name="read_file",
                result=(
                    "ANTHROPIC_API_KEY=sk-secret-value\n"
                    "Authorization: Bearer secret-token\n"
                    "NORMAL=value"
                ),
            )
        )
    )

    result = payload["data"]["result"]
    assert "sk-secret-value" not in result
    assert "secret-token" not in result
    assert "ANTHROPIC_API_KEY=<redacted>" in result
    assert "Authorization: Bearer <redacted>" in result
