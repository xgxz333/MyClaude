from __future__ import annotations

import asyncio
from collections.abc import Sequence

from my_claude.agent.events import RunStartedEvent, ToolCallCompletedEvent
from my_claude.agent.tools import ToolDefinition
from my_claude.core.context import AnthropicMessage, ToolUseBlock
from my_claude.core.trace import (
    EventBusTracingProvider,
    TraceRecord,
    TracingProvider,
    trace_record_from_event,
)
from my_claude.llm.client import LLMResponse


class RecordingTraceEmitter:
    def __init__(self) -> None:
        self.records: list[TraceRecord] = []

    def emit(self, record: TraceRecord) -> None:
        self.records.append(record)


def test_trace_record_from_event_marks_core_event_flow_and_keeps_full_event_payload() -> None:
    event = ToolCallCompletedEvent(
        run_id="run-1",
        tool_use_id="tool-1",
        tool_name="read_file",
        result="ok",
        elapsed_ms=12,
    )

    record = trace_record_from_event(event)

    assert record.run_id == "run-1"
    assert record.layer == "event"
    assert record.direction == "CORE"
    assert record.kind == "event"

    assert record.data == {
        "type": "tool.call_finished",
        "message": "tool call completed",
        "run_id": "run-1",
        "tool_use_id": "tool-1",
        "tool_name": "read_file",
        "result": "ok",
        "output": None,
        "error": None,
        "error_type": None,
        "elapsed_ms": 12,
        "ts": None,
    }


def test_tracing_provider_is_an_async_event_bus_subscriber() -> None:
    emitter = RecordingTraceEmitter()
    provider = EventBusTracingProvider(emitter)

    asyncio.run(provider.handle(RunStartedEvent(goal="ship it", run_id="run-1")))

    assert len(emitter.records) == 1
    record = emitter.records[0]
    assert record.direction == "CORE"
    assert record.kind == "event"
    assert record.data["type"] == "run.started"


def test_llm_tracing_provider_records_request_and_response_without_full_payload() -> None:
    emitter = RecordingTraceEmitter()
    provider = TracingProvider(
        StaticLLMClient(
            LLMResponse(
                content="done",
                raw={
                    "message": {
                        "stop_reason": "end_turn",
                        "usage": {"output_tokens": 2},
                    }
                },
            )
        ),
        emitter,
        include_payload=False,
        run_id="run-1",
    )

    response = asyncio.run(
        provider.complete(
            [AnthropicMessage.user_text("ship it")],
            [ToolDefinition(name="read_file", description="Read a file.")],
            step=2,
        )
    )

    assert response.content == "done"
    assert [record.kind for record in emitter.records] == ["api_call", "api_response"]
    request, response_record = emitter.records
    assert request.layer == "llm"
    assert request.direction == "CORE→LLM"
    assert request.run_id == "run-1"
    assert request.step == 2
    assert request.data == {
        "has_system_prompt_patch": False,
        "message_count": 1,
        "tool_count": 1,
    }
    assert response_record.layer == "llm"
    assert response_record.direction == "LLM→CORE"
    assert response_record.run_id == "run-1"
    assert response_record.step == 2
    assert response_record.data["stop_reason"] == "end_turn"
    assert response_record.data["usage"] == {"output_tokens": 2}
    assert response_record.data["content_length"] == 4
    assert response_record.data["content_block_count"] == 0
    assert response_record.data["has_raw_response"] is True
    assert "messages" not in request.data
    assert "text" not in response_record.data
    assert StaticLLMClient.last_step == 2


def test_llm_tracing_provider_can_include_full_request_and_response_payload() -> None:
    emitter = RecordingTraceEmitter()
    provider = TracingProvider(
        StaticLLMClient(
            LLMResponse(
                content="done",
                content_blocks=(
                    ToolUseBlock(id="toolu-1", name="read_file", input={"path": "README.md"}),
                ),
                raw={
                    "id": "response-1",
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            )
        ),
        emitter,
        include_payload=True,
        run_id="run-2",
    )

    asyncio.run(
        provider.complete(
            [AnthropicMessage.user_text("ship it")],
            [
                ToolDefinition(
                    name="read_file",
                    description="Read a file.",
                    input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
                )
            ],
            step=1,
        )
    )

    request, response = emitter.records
    assert request.run_id == "run-2"
    assert response.run_id == "run-2"
    assert request.data["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "ship it"}]}
    ]
    assert request.data["tool_schemas"] == [
        {
            "name": "read_file",
            "description": "Read a file.",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]
    assert response.data["stop_reason"] == "tool_use"
    assert response.data["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert response.data["text"] == "done"
    assert response.data["tool_calls"] == [
        {
            "type": "tool_use",
            "id": "toolu-1",
            "name": "read_file",
            "input": {"path": "README.md"},
        }
    ]
    assert response.data["raw"] == {
        "id": "response-1",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


class StaticLLMClient:
    last_step: int | None = None

    def __init__(self, response: LLMResponse) -> None:
        self._response = response

    async def complete(
        self,
        messages: Sequence[AnthropicMessage],
        tools: Sequence[ToolDefinition],
        *,
        step: int | None = None,
        system_prompt_patch: str | None = None,
    ) -> LLMResponse:
        del messages, tools, system_prompt_patch
        StaticLLMClient.last_step = step
        return self._response
