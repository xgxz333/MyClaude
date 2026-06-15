"""Typed command and result models accepted by the S0 core bus."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from my_claude.core.bus.envelope import JsonRpcRequest
from my_claude.core.bus.events import AgentEventType

CORE_PING_METHOD: Literal["core.ping"] = "core.ping"
EVENT_SUBSCRIBE_METHOD: Literal["event.subscribe"] = "event.subscribe"
EVENT_PUBLISH_METHOD: Literal["event.publish"] = "event.publish"
AGENT_RUN_METHOD: Literal["agent.run"] = "agent.run"
SESSION_CREATE_METHOD: Literal["session.create"] = "session.create"
SESSION_MESSAGE_METHOD: Literal["session.message"] = "session.message"
SESSION_SEND_MESSAGE_METHOD: Literal["session.send_message"] = "session.send_message"
SESSION_GET_HISTORY_METHOD: Literal["session.get_history"] = "session.get_history"
SESSION_COMPACT_METHOD: Literal["session.compact"] = "session.compact"
SESSION_CLOSE_METHOD: Literal["session.close"] = "session.close"
PERMISSION_RESPOND_METHOD: Literal["permission.respond"] = "permission.respond"
PACKAGE_NAME = "MyClaude"
FALLBACK_VERSION = "0.1.0"


class CorePingParams(BaseModel):
    """Parameters for the `core.ping` command."""

    model_config = ConfigDict(extra="forbid")


class CorePingCommand(BaseModel):
    """Command envelope for checking whether the core server is alive."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["core.ping"] = CORE_PING_METHOD
    params: CorePingParams = Field(default_factory=CorePingParams)


class CorePingResult(BaseModel):
    """Result returned by the core server after a successful ping."""

    model_config = ConfigDict(extra="forbid")

    pong: Literal["pong"] = "pong"
    uptime_seconds: float = Field(ge=0)
    server_version: str


class EventSubscribeParams(BaseModel):
    """Parameters for subscribing to pushed runtime events."""

    model_config = ConfigDict(extra="forbid")

    replay_from: int | None = Field(default=None, ge=1)
    event_types: list[AgentEventType] | None = None
    run_id: str | None = Field(default=None, min_length=1)
    topics: list[str] = Field(default_factory=lambda: ["*"])
    scope: str = "global"
    replay_from_run: str | None = Field(default=None, min_length=1)


class EventSubscribeCommand(BaseModel):
    """Command envelope for opening an event stream on the current connection."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["event.subscribe"] = EVENT_SUBSCRIBE_METHOD
    params: EventSubscribeParams = Field(default_factory=EventSubscribeParams)


class EventSubscribeResult(BaseModel):
    """Result returned after registering the event stream."""

    model_config = ConfigDict(extra="forbid")

    subscription_id: str
    next_sequence: int
    replayed_count: int = 0


class AgentRunParams(BaseModel):
    """Parameters for requesting an agent run from the daemon."""

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1)


class AgentRunCommand(BaseModel):
    """Command envelope for triggering an agent run in the daemon."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["agent.run"] = AGENT_RUN_METHOD
    params: AgentRunParams


class AgentRunResult(BaseModel):
    """Result returned immediately after the daemon accepts an agent run."""

    model_config = ConfigDict(extra="forbid")

    accepted: Literal[True] = True
    run_id: str
    goal: str
    timeline_path: str


class SessionCreateParams(BaseModel):
    """Parameters for creating a daemon-side chat session."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["one_shot", "chat"] = "chat"
    title: str | None = Field(default=None, min_length=1)


class SessionCreateCommand(BaseModel):
    """Command envelope for creating a daemon-side chat session."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["session.create"] = SESSION_CREATE_METHOD
    params: SessionCreateParams = Field(default_factory=SessionCreateParams)


class SessionCreateResult(BaseModel):
    """Result returned after creating a daemon-side chat session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    status: Literal["active", "waiting_for_input", "closed"] = "active"
    title: str | None = None


class SessionMessageParams(BaseModel):
    """Parameters for sending one user message to an existing chat session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class SessionMessageCommand(BaseModel):
    """Command envelope for sending one user message to a chat session."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["session.message"] = SESSION_MESSAGE_METHOD
    params: SessionMessageParams


class SessionSendMessageParams(BaseModel):
    """Kama-compatible parameters for sending one user message to a session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    content: str = Field(min_length=1)


class SessionSendMessageCommand(BaseModel):
    """Kama-compatible command envelope for sending one user message."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["session.send_message"] = SESSION_SEND_MESSAGE_METHOD
    params: SessionSendMessageParams


class SessionMessageResult(BaseModel):
    """Result returned immediately after the daemon accepts a chat message."""

    model_config = ConfigDict(extra="forbid")

    accepted: Literal[True] = True
    session_id: str
    turn: int
    run_id: str
    timeline_path: str


class SessionSendMessageResult(BaseModel):
    """Kama-compatible result returned after a session message run completes."""

    model_config = ConfigDict(extra="forbid")

    run_id: str


class SessionGetHistoryParams(BaseModel):
    """Parameters for reading a session's complete conversation thread."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)


class SessionGetHistoryCommand(BaseModel):
    """Command envelope for reading a session's complete conversation thread."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["session.get_history"] = SESSION_GET_HISTORY_METHOD
    params: SessionGetHistoryParams


class SessionGetHistoryResult(BaseModel):
    """Result returned with Anthropic-compatible session messages."""

    model_config = ConfigDict(extra="forbid")

    messages: list[dict[str, Any]]


class SessionCompactParams(BaseModel):
    """Parameters for manually compacting a session thread."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    focus: str = ""


class SessionCompactCommand(BaseModel):
    """Command envelope for manually compacting a session thread."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["session.compact"] = SESSION_COMPACT_METHOD
    params: SessionCompactParams


class SessionCompactResult(BaseModel):
    """Result returned after compacting a session thread."""

    model_config = ConfigDict(extra="forbid")

    summary_tokens: int
    saved_tokens: int


class SessionCloseParams(BaseModel):
    """Parameters for closing a session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)


class SessionCloseCommand(BaseModel):
    """Command envelope for closing a session."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["session.close"] = SESSION_CLOSE_METHOD
    params: SessionCloseParams


class SessionCloseResult(BaseModel):
    """Result returned after closing a session."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["closed"] = "closed"


class PermissionRespondCommand(BaseModel):
    """Command envelope for resolving a pending permission request."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["permission.respond"] = PERMISSION_RESPOND_METHOD
    tool_use_id: str = Field(min_length=1)
    decision: str = Field(min_length=1)


class PermissionRespondResult(BaseModel):
    """Empty result returned after accepting a permission response."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = True


class HandlerError(Exception):
    """JSON-RPC handler error with an explicit application error code."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


BusCommand = (
    CorePingCommand
    | EventSubscribeCommand
    | AgentRunCommand
    | SessionCreateCommand
    | SessionMessageCommand
    | SessionSendMessageCommand
    | SessionGetHistoryCommand
    | SessionCompactCommand
    | SessionCloseCommand
    | PermissionRespondCommand
)
BusResult = (
    CorePingResult
    | EventSubscribeResult
    | AgentRunResult
    | SessionCreateResult
    | SessionMessageResult
    | SessionSendMessageResult
    | SessionGetHistoryResult
    | SessionCompactResult
    | SessionCloseResult
    | PermissionRespondResult
)


def command_from_request(request: JsonRpcRequest) -> BusCommand:
    if request.method not in {
        CORE_PING_METHOD,
        EVENT_SUBSCRIBE_METHOD,
        AGENT_RUN_METHOD,
        SESSION_CREATE_METHOD,
        SESSION_MESSAGE_METHOD,
        SESSION_SEND_MESSAGE_METHOD,
        SESSION_GET_HISTORY_METHOD,
        SESSION_COMPACT_METHOD,
        SESSION_CLOSE_METHOD,
        PERMISSION_RESPOND_METHOD,
    }:
        raise ValueError(f"unknown command method: {request.method}")

    params = request.params if request.params is not None else {}
    if not isinstance(params, dict):
        raise ValueError(f"{request.method} params must be an object")

    payload = {"method": request.method, "params": params}
    if request.method == CORE_PING_METHOD:
        return CorePingCommand.model_validate(payload)
    if request.method == EVENT_SUBSCRIBE_METHOD:
        return EventSubscribeCommand.model_validate(payload)
    if request.method == SESSION_CREATE_METHOD:
        return SessionCreateCommand.model_validate(payload)
    if request.method == SESSION_MESSAGE_METHOD:
        return SessionMessageCommand.model_validate(payload)
    if request.method == SESSION_SEND_MESSAGE_METHOD:
        return SessionSendMessageCommand.model_validate(payload)
    if request.method == SESSION_GET_HISTORY_METHOD:
        return SessionGetHistoryCommand.model_validate(payload)
    if request.method == SESSION_COMPACT_METHOD:
        return SessionCompactCommand.model_validate(payload)
    if request.method == SESSION_CLOSE_METHOD:
        return SessionCloseCommand.model_validate(payload)
    if request.method == PERMISSION_RESPOND_METHOD:
        return PermissionRespondCommand.model_validate(params)

    return AgentRunCommand.model_validate(payload)


def result_to_json(result: BusResult) -> dict[str, Any]:
    return result.model_dump(mode="json")


def ping_result(uptime_seconds: float, server_version: str | None = None) -> CorePingResult:
    return CorePingResult(
        uptime_seconds=uptime_seconds,
        server_version=server_version or get_server_version(),
    )


def get_server_version() -> str:
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return FALLBACK_VERSION
