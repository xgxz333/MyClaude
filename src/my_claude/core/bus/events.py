"""Pydantic event models emitted by the agent runtime and persisted by the bus."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter


class AgentEventType(StrEnum):
    """Known lifecycle event names emitted during an agent run."""

    RUN_STARTED = "run.started"
    STEP_STARTED = "step.started"
    STEP_FINISHED = "step.finished"
    LLM_REQUEST_STARTED = "llm.request_started"
    LLM_TOKEN = "llm.token"
    LLM_USAGE = "llm.usage"
    LLM_MODEL_SELECTED = "llm.model_selected"
    LLM_RESPONSE_COMPLETED = "llm.response_completed"
    TOOL_CALL_STARTED = "tool.call_started"
    TOOL_CALL_COMPLETED = "tool.call_finished"
    RUN_COMPLETED = "run.finished"
    RUN_CANCELLED = "run.cancelled"
    RUN_FAILED = "run.failed"
    SESSION_CREATED = "session.created"
    SESSION_MESSAGE_RECEIVED = "session.message_received"
    SESSION_WAITING_FOR_INPUT = "session.waiting_for_input"
    SESSION_RESUMED = "session.resumed"
    SESSION_CLOSED = "session.closed"


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
    run_id: str | None = None


class StepStartedEvent(AgentEvent):
    """Event emitted when one ReAct loop step begins."""

    type: Literal[AgentEventType.STEP_STARTED] = AgentEventType.STEP_STARTED
    message: str = "step started"
    run_id: str
    step: int


class StepFinishedEvent(AgentEvent):
    """Event emitted when one ReAct loop step finishes."""

    type: Literal[AgentEventType.STEP_FINISHED] = AgentEventType.STEP_FINISHED
    message: str = "step finished"
    run_id: str
    step: int


class LLMRequestStartedEvent(AgentEvent):
    """Event emitted immediately before sending messages to the LLM."""

    type: Literal[AgentEventType.LLM_REQUEST_STARTED] = AgentEventType.LLM_REQUEST_STARTED
    message: str = "llm request started"
    run_id: str | None = None
    model_input_messages: int
    tools: int


class LLMTokenEvent(AgentEvent):
    """Event emitted for one streamed token or text delta from the LLM."""

    type: Literal[AgentEventType.LLM_TOKEN] = AgentEventType.LLM_TOKEN
    message: str = "llm token"
    token: str
    index: int | None = None
    run_id: str | None = None


class LLMUsageEvent(AgentEvent):
    """Event emitted after the LLM stream reports usage statistics."""

    type: Literal[AgentEventType.LLM_USAGE] = AgentEventType.LLM_USAGE
    message: str = "llm usage"
    run_id: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class LLMModelSelectedEvent(AgentEvent):
    """Event emitted when the static model is selected for a request."""

    type: Literal[AgentEventType.LLM_MODEL_SELECTED] = AgentEventType.LLM_MODEL_SELECTED
    message: str = "llm model selected"
    run_id: str
    model: str
    strategy: str = "static"


class LLMResponseCompletedEvent(AgentEvent):
    """Event emitted after the LLM has returned a complete response."""

    type: Literal[AgentEventType.LLM_RESPONSE_COMPLETED] = AgentEventType.LLM_RESPONSE_COMPLETED
    run_id: str | None = None
    content: str
    has_raw_response: bool = False


class ToolCallStartedEvent(AgentEvent):
    """Event emitted before invoking a registered tool."""

    type: Literal[AgentEventType.TOOL_CALL_STARTED] = AgentEventType.TOOL_CALL_STARTED
    message: str = "tool call started"
    run_id: str | None = None
    tool_use_id: str
    tool_name: str
    arguments: Mapping[str, Any]


class ToolCallCompletedEvent(AgentEvent):
    """Event emitted after a tool call returns or fails."""

    type: Literal[AgentEventType.TOOL_CALL_COMPLETED] = AgentEventType.TOOL_CALL_COMPLETED
    message: str = "tool call completed"
    run_id: str | None = None
    tool_use_id: str | None = None
    tool_name: str
    result: str | None = None
    error: str | None = None
    error_type: str | None = None
    elapsed_ms: int | None = None


class RunCompletedEvent(AgentEvent):
    """Event emitted after the agent loop completes successfully."""

    type: Literal[AgentEventType.RUN_COMPLETED] = AgentEventType.RUN_COMPLETED
    message: str = "run completed"
    goal: str
    run_id: str | None = None
    status: str = "success"
    steps: int | None = None


class RunCancelledEvent(AgentEvent):
    """Event emitted when a run is cancelled before completion."""

    type: Literal[AgentEventType.RUN_CANCELLED] = AgentEventType.RUN_CANCELLED
    message: str = "run cancelled"
    run_id: str | None = None
    reason: str | None = None


class RunFailedEvent(AgentEvent):
    """Event emitted when the agent loop raises an unrecoverable error."""

    type: Literal[AgentEventType.RUN_FAILED] = AgentEventType.RUN_FAILED
    message: str = "run failed"
    error: str
    run_id: str | None = None


class SessionCreatedEvent(AgentEvent):
    """Event emitted when the daemon creates a chat session."""

    type: Literal[AgentEventType.SESSION_CREATED] = AgentEventType.SESSION_CREATED
    message: str = "session created"
    session_id: str
    mode: str = "chat"
    status: str = "active"
    title: str | None = None
    path: str


class SessionMessageReceivedEvent(AgentEvent):
    """Event emitted after a user message is appended to a session thread."""

    type: Literal[AgentEventType.SESSION_MESSAGE_RECEIVED] = (
        AgentEventType.SESSION_MESSAGE_RECEIVED
    )
    message: str = "session message received"
    session_id: str
    content: str


class SessionWaitingForInputEvent(AgentEvent):
    """Event emitted when a chat session finishes a turn and waits for input."""

    type: Literal[AgentEventType.SESSION_WAITING_FOR_INPUT] = (
        AgentEventType.SESSION_WAITING_FOR_INPUT
    )
    message: str = "session waiting for input"
    session_id: str
    last_run_id: str


class SessionResumedEvent(AgentEvent):
    """Event emitted when a waiting chat session accepts another message."""

    type: Literal[AgentEventType.SESSION_RESUMED] = AgentEventType.SESSION_RESUMED
    message: str = "session resumed"
    session_id: str


class SessionClosedEvent(AgentEvent):
    """Event emitted when a session is closed."""

    type: Literal[AgentEventType.SESSION_CLOSED] = AgentEventType.SESSION_CLOSED
    message: str = "session closed"
    session_id: str


KnownAgentEvent = (
    RunStartedEvent
    | StepStartedEvent
    | StepFinishedEvent
    | LLMRequestStartedEvent
    | LLMTokenEvent
    | LLMUsageEvent
    | LLMModelSelectedEvent
    | LLMResponseCompletedEvent
    | ToolCallStartedEvent
    | ToolCallCompletedEvent
    | RunCompletedEvent
    | RunCancelledEvent
    | RunFailedEvent
    | SessionCreatedEvent
    | SessionMessageReceivedEvent
    | SessionWaitingForInputEvent
    | SessionResumedEvent
    | SessionClosedEvent
)

KnownAgentEventAdapter: TypeAdapter[KnownAgentEvent] = TypeAdapter(KnownAgentEvent)
EventHandler = Callable[[AgentEvent], Awaitable[None]]


async def dispatch_event(handler: EventHandler | None, event: AgentEvent) -> None:
    if handler is None:
        return

    await handler(event)
