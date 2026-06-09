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


BusCommand = CorePingCommand | EventSubscribeCommand | AgentRunCommand
BusResult = CorePingResult | EventSubscribeResult | AgentRunResult


def command_from_request(request: JsonRpcRequest) -> BusCommand:
    if request.method not in {CORE_PING_METHOD, EVENT_SUBSCRIBE_METHOD, AGENT_RUN_METHOD}:
        raise ValueError(f"unknown command method: {request.method}")

    params = request.params if request.params is not None else {}
    if not isinstance(params, dict):
        raise ValueError(f"{request.method} params must be an object")

    payload = {"method": request.method, "params": params}
    if request.method == CORE_PING_METHOD:
        return CorePingCommand.model_validate(payload)
    if request.method == EVENT_SUBSCRIBE_METHOD:
        return EventSubscribeCommand.model_validate(payload)

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
