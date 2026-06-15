"""JSON-RPC envelope models and helpers for the S0 IPC protocol."""

from __future__ import annotations

import json
from enum import IntEnum
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    TypeAdapter,
    ValidationError,
)

JsonRpcId = StrictInt | StrictStr | None


class JsonRpcErrorCode(IntEnum):
    """Standard JSON-RPC error codes used by the core transport."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    SERVER_ERROR = -32000


class JsonRpcError(BaseModel):
    """Structured error object embedded in a JSON-RPC error response."""

    model_config = ConfigDict(extra="forbid")

    code: int
    message: str
    data: Any | None = None


class JsonRpcRequest(BaseModel):
    """Incoming JSON-RPC request frame received from a client."""

    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = "2.0"
    id: JsonRpcId = None
    method: str = Field(min_length=1)
    params: dict[str, Any] | list[Any] | None = None


class JsonRpcNotification(BaseModel):
    """Server-pushed JSON-RPC notification frame without a response id."""

    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = "2.0"
    method: str = Field(min_length=1)
    params: dict[str, Any] | list[Any] | None = None


class EventPushEnvelope(BaseModel):
    """Server-pushed event frame compatible with the Kama S2 IPC stream."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["event"] = "event"
    event: dict[str, Any]


class JsonRpcSuccessResponse(BaseModel):
    """JSON-RPC response frame for successful command execution."""

    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = "2.0"
    id: JsonRpcId
    result: Any


class JsonRpcErrorResponse(BaseModel):
    """JSON-RPC response frame for failed request parsing or execution."""

    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = "2.0"
    id: JsonRpcId
    error: JsonRpcError


JsonRpcResponse = JsonRpcSuccessResponse | JsonRpcErrorResponse
JsonRpcResponseAdapter: TypeAdapter[JsonRpcResponse] = TypeAdapter(JsonRpcResponse)


def parse_request(raw: bytes | str) -> JsonRpcRequest:
    return JsonRpcRequest.model_validate(_loads_json_object(raw))


def parse_notification(raw: bytes | str) -> JsonRpcNotification:
    return JsonRpcNotification.model_validate(_loads_json_object(raw))


def parse_response(raw: bytes | str) -> JsonRpcResponse:
    return JsonRpcResponseAdapter.validate_python(_loads_json_object(raw))


def make_success_response(request_id: JsonRpcId, result: Any) -> JsonRpcSuccessResponse:
    return JsonRpcSuccessResponse(id=request_id, result=result)


def make_notification(
    method: str,
    params: dict[str, Any] | list[Any] | None = None,
) -> JsonRpcNotification:
    return JsonRpcNotification(method=method, params=params)


def make_error_response(
    request_id: JsonRpcId,
    code: JsonRpcErrorCode | int,
    message: str | None = None,
    data: Any | None = None,
) -> JsonRpcErrorResponse:
    code_value = code.value if isinstance(code, JsonRpcErrorCode) else code
    default_message = (
        _default_error_message(code)
        if isinstance(code, JsonRpcErrorCode)
        else "Server error"
    )
    error = JsonRpcError(
        code=code_value,
        message=message or default_message,
        data=data,
    )
    return JsonRpcErrorResponse(id=request_id, error=error)


def to_ndjson(model: BaseModel) -> bytes:
    return (model.model_dump_json(exclude_none=True) + "\n").encode("utf-8")


def _loads_json_object(raw: bytes | str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValidationError.from_exception_data(
            "JsonRpcEnvelope",
            cast(
                Any,
                [
                    {
                        "type": "value_error",
                        "loc": ("json",),
                        "msg": "Invalid JSON",
                        "input": raw,
                        "ctx": {"error": error},
                    }
                ],
            ),
        ) from error

    if not isinstance(value, dict):
        raise ValidationError.from_exception_data(
            "JsonRpcEnvelope",
            cast(
                Any,
                [
                    {
                        "type": "value_error",
                        "loc": ("json",),
                        "msg": "JSON-RPC envelope must be an object",
                        "input": value,
                        "ctx": {"error": ValueError("JSON-RPC envelope must be an object")},
                    }
                ],
            ),
        )

    return value


def _default_error_message(code: JsonRpcErrorCode) -> str:
    messages = {
        JsonRpcErrorCode.PARSE_ERROR: "Parse error",
        JsonRpcErrorCode.INVALID_REQUEST: "Invalid Request",
        JsonRpcErrorCode.METHOD_NOT_FOUND: "Method not found",
        JsonRpcErrorCode.INVALID_PARAMS: "Invalid params",
        JsonRpcErrorCode.INTERNAL_ERROR: "Internal error",
        JsonRpcErrorCode.SERVER_ERROR: "Server error",
    }
    return messages[code]
