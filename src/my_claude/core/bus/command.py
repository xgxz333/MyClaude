"""Typed command and result models accepted by the S0 core bus."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from my_claude.core.bus.envelope import JsonRpcRequest

CORE_PING_METHOD: Literal["core.ping"] = "core.ping"
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


BusCommand = CorePingCommand
BusResult = CorePingResult

BusCommandAdapter = TypeAdapter(BusCommand)
BusResultAdapter = TypeAdapter(BusResult)


def command_from_request(request: JsonRpcRequest) -> BusCommand:
    if request.method != CORE_PING_METHOD:
        raise ValueError(f"unknown command method: {request.method}")

    params = request.params if request.params is not None else {}
    if not isinstance(params, dict):
        raise ValueError("core.ping params must be an object")

    return BusCommandAdapter.validate_python(
        {
            "method": request.method,
            "params": params,
        }
    )


def result_to_json(result: BusResult) -> dict[str, Any]:
    return BusResultAdapter.validate_python(result).model_dump()


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
