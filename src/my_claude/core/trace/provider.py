"""Trace providers for EventBus and LLM instrumentation."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, cast

from my_claude.agent.events import AgentEvent
from my_claude.agent.tools import ToolDefinition
from my_claude.core.context import AnthropicMessage, ToolUseBlock
from my_claude.core.trace.record import TraceRecord
from my_claude.llm.client import LLMClient, LLMResponse

CORE_FLOW = "CORE"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TraceEmitter(Protocol):
    """Minimal trace sink used by trace instrumentation."""

    def emit(self, record: TraceRecord) -> None: ...


class EventBusTracingProvider:
    """EventBus subscriber that records internal core event flow as trace records."""

    def __init__(self, emitter: TraceEmitter) -> None:
        self._emitter = emitter

    async def handle(self, event: AgentEvent) -> None:
        self._emitter.emit(trace_record_from_event(event))


class TracingProvider:
    """Decorator that records LLM request and response traces around a provider call."""

    def __init__(
        self,
        provider: LLMClient,
        emitter: TraceEmitter,
        *,
        include_payload: bool = True,
        run_id: str | None = None,
    ) -> None:
        self._provider = provider
        self._emitter = emitter
        self._include_payload = include_payload
        self._run_id = run_id

    async def complete(
        self,
        messages: Sequence[AnthropicMessage],
        tools: Sequence[ToolDefinition],
        *,
        step: int | None = None,
        system_prompt_patch: str | None = None,
    ) -> LLMResponse:
        self._emit_request(messages, tools, step=step, system_prompt_patch=system_prompt_patch)
        started_at = time.monotonic()
        response = await self._provider.complete(
            messages,
            tools,
            step=step,
            system_prompt_patch=system_prompt_patch,
        )
        latency_ms = int((time.monotonic() - started_at) * 1000)
        self._emit_response(response, step=step, latency_ms=latency_ms)
        return response

    def _emit_request(
        self,
        messages: Sequence[AnthropicMessage],
        tools: Sequence[ToolDefinition],
        *,
        step: int | None,
        system_prompt_patch: str | None,
    ) -> None:
        data: dict[str, Any]
        if self._include_payload:
            data = {
                "system_prompt_patch": system_prompt_patch,
                "messages": [_message_payload(message) for message in messages],
                "tool_schemas": [_tool_schema_payload(tool) for tool in tools],
            }
        else:
            data = {
                "has_system_prompt_patch": bool(system_prompt_patch),
                "message_count": len(messages),
                "tool_count": len(tools),
            }

        self._emitter.emit(
            TraceRecord(
                ts=_now(),
                direction="CORE→LLM",
                layer="llm",
                kind="api_call",
                run_id=self._run_id,
                step=step,
                data=data,
            )
        )

    def _emit_response(
        self,
        response: LLMResponse,
        *,
        step: int | None,
        latency_ms: int,
    ) -> None:
        data: dict[str, Any]
        if self._include_payload:
            data = {
                "stop_reason": _stop_reason(response.raw),
                "text": response.content,
                "tool_calls": _tool_calls_payload(response),
                "usage": _usage_payload(response.raw),
                "content_blocks": [
                    _json_value(block.model_dump(mode="json"))
                    for block in response.content_blocks
                ],
                "raw": _json_value(response.raw),
                "latency_ms": latency_ms,
            }
        else:
            data = {
                "stop_reason": _stop_reason(response.raw),
                "usage": _usage_payload(response.raw),
                "content_length": len(response.content),
                "content_block_count": len(response.content_blocks),
                "has_raw_response": response.raw is not None,
                "latency_ms": latency_ms,
            }

        self._emitter.emit(
            TraceRecord(
                ts=_now(),
                direction="LLM→CORE",
                layer="llm",
                kind="api_response",
                run_id=self._run_id,
                step=step,
                data=data,
            )
        )


def trace_record_from_event(event: AgentEvent) -> TraceRecord:
    """Build one event-layer trace record from a complete runtime event."""

    event_dict = event.model_dump(mode="json")
    return TraceRecord(
        ts=_now(),
        direction="CORE",
        layer="event",
        kind="event",
        run_id=_event_run_id(event),
        data=event_dict,
    )


def _event_run_id(event: AgentEvent) -> str | None:
    run_id = event.data.get("run_id")
    if isinstance(run_id, str):
        return run_id

    return None


def _message_payload(message: AnthropicMessage) -> dict[str, Any]:
    return message.model_dump(mode="json")


def _tool_schema_payload(tool: ToolDefinition) -> dict[str, Any]:
    return cast(dict[str, Any], _json_value(asdict(tool)))


def _tool_calls_payload(response: LLMResponse) -> list[dict[str, Any]]:
    return [
        block.model_dump(mode="json")
        for block in response.content_blocks
        if isinstance(block, ToolUseBlock)
    ]


def _stop_reason(raw: Mapping[str, Any] | None) -> str | None:
    if raw is None:
        return None

    value = raw.get("stop_reason")
    if isinstance(value, str):
        return value

    message = raw.get("message")
    if isinstance(message, Mapping):
        value = message.get("stop_reason")
        if isinstance(value, str):
            return value

    return None


def _usage_payload(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {}

    usage = raw.get("usage")
    if isinstance(usage, Mapping):
        return cast(dict[str, Any], _json_value(usage))

    message = raw.get("message")
    if isinstance(message, Mapping):
        usage = message.get("usage")
        if isinstance(usage, Mapping):
            return cast(dict[str, Any], _json_value(usage))

    return {}


def _json_value(value: object) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))

    return str(value)
