"""Pydantic event models emitted by the agent runtime and persisted by the bus."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter


class AgentEventType(StrEnum):
    """Known lifecycle event names emitted during an agent run."""

    RUN_STARTED = "run_started"
    LLM_REQUEST_STARTED = "llm_request_started"
    LLM_TOKEN = "llm_token"
    LLM_RESPONSE_COMPLETED = "llm_response_completed"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    RUN_COMPLETED = "run_completed"
    RUN_CANCELLED = "run_cancelled"
    RUN_FAILED = "run_failed"


class AgentEvent(BaseModel):
    """Base event shape shared by every concrete runtime event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: AgentEventType
    message: str = ""

    @property
    def data(self) -> Mapping[str, Any]:
        return self.model_dump(
            mode="json",
            exclude={"type", "message"},
            exclude_none=True,
        )


class RunStartedEvent(AgentEvent):
    """Event emitted when a run starts before the first agent loop iteration."""

    type: Literal[AgentEventType.RUN_STARTED] = AgentEventType.RUN_STARTED
    message: str = "run started"
    goal: str


class LLMRequestStartedEvent(AgentEvent):
    """Event emitted immediately before sending messages to the LLM."""

    type: Literal[AgentEventType.LLM_REQUEST_STARTED] = AgentEventType.LLM_REQUEST_STARTED
    message: str = "llm request started"
    model_input_messages: int
    tools: int


class LLMTokenEvent(AgentEvent):
    """Event emitted for one streamed token or text delta from the LLM."""

    type: Literal[AgentEventType.LLM_TOKEN] = AgentEventType.LLM_TOKEN
    message: str = "llm token"
    token: str
    index: int | None = None


class LLMResponseCompletedEvent(AgentEvent):
    """Event emitted after the LLM has returned a complete response."""

    type: Literal[AgentEventType.LLM_RESPONSE_COMPLETED] = AgentEventType.LLM_RESPONSE_COMPLETED
    content: str
    has_raw_response: bool = False


class ToolCallStartedEvent(AgentEvent):
    """Event emitted before invoking a registered tool."""

    type: Literal[AgentEventType.TOOL_CALL_STARTED] = AgentEventType.TOOL_CALL_STARTED
    message: str = "tool call started"
    tool_use_id: str
    tool_name: str
    arguments: Mapping[str, Any]


class ToolCallCompletedEvent(AgentEvent):
    """Event emitted after a tool call returns or fails."""

    type: Literal[AgentEventType.TOOL_CALL_COMPLETED] = AgentEventType.TOOL_CALL_COMPLETED
    message: str = "tool call completed"
    tool_use_id: str | None = None
    tool_name: str
    result: str | None = None
    error: str | None = None


class RunCompletedEvent(AgentEvent):
    """Event emitted after the agent loop completes successfully."""

    type: Literal[AgentEventType.RUN_COMPLETED] = AgentEventType.RUN_COMPLETED
    message: str = "run completed"
    goal: str


class RunCancelledEvent(AgentEvent):
    """Event emitted when a run is cancelled before completion."""

    type: Literal[AgentEventType.RUN_CANCELLED] = AgentEventType.RUN_CANCELLED
    message: str = "run cancelled"
    reason: str | None = None


class RunFailedEvent(AgentEvent):
    """Event emitted when the agent loop raises an unrecoverable error."""

    type: Literal[AgentEventType.RUN_FAILED] = AgentEventType.RUN_FAILED
    message: str = "run failed"
    error: str


KnownAgentEvent = (
    RunStartedEvent
    | LLMRequestStartedEvent
    | LLMTokenEvent
    | LLMResponseCompletedEvent
    | ToolCallStartedEvent
    | ToolCallCompletedEvent
    | RunCompletedEvent
    | RunCancelledEvent
    | RunFailedEvent
)

KnownAgentEventAdapter: TypeAdapter[KnownAgentEvent] = TypeAdapter(KnownAgentEvent)
EventHandler = Callable[[AgentEvent], Awaitable[None]]


async def dispatch_event(handler: EventHandler | None, event: AgentEvent) -> None:
    if handler is None:
        return

    await handler(event)
