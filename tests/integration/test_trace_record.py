from __future__ import annotations

import pytest
from pydantic import ValidationError

from my_claude.core.trace import TraceRecord


def test_trace_record_keeps_control_flow_fields_closed_and_data_open() -> None:
    record = TraceRecord(
        ts="2026-06-12T12:00:00+00:00",
        direction="CLIENT→CORE",
        layer="ipc",
        kind="command",
        run_id="run-1",
        step=2,
        client_id="127.0.0.1:43123",
        data={
            "method": "agent.run",
            "params": {
                "goal": "ship it",
                "metadata": {"priority": 1, "dry_run": False, "tags": ["s3"]},
            },
        },
    )

    dumped = record.model_dump(mode="json")

    assert dumped["direction"] == "CLIENT→CORE"
    assert dumped["layer"] == "ipc"
    assert dumped["kind"] == "command"
    assert dumped["data"]["params"]["metadata"]["tags"] == ["s3"]


def test_trace_record_does_not_persist_unknown_top_level_business_fields() -> None:
    record = TraceRecord.model_validate(
        {
            "ts": "2026-06-12T12:00:00+00:00",
            "direction": "CORE→LLM",
            "layer": "llm",
            "kind": "api_call",
            "prompt": "business data must live in data",
            "data": {},
        }
    )

    assert "prompt" not in record.model_dump()


def test_trace_record_requires_known_control_flow_values() -> None:
    with pytest.raises(ValidationError):
        TraceRecord.model_validate(
            {
                "ts": "2026-06-12T12:00:00+00:00",
                "direction": "CORE→TOOL",
                "layer": "tool",
                "kind": "event",
                "data": {},
            }
        )
