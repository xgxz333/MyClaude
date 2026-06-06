from __future__ import annotations

import json
from enum import IntEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, TypeAdapter, ValidationError


JsonRpcId = StrictInt | StrictStr | None


class JsonRpcErrorCode(IntEnum):
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    SERVER_ERROR = -32000


class JsonRpcError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: int
    message: str
    data: Any | None = None


class JsonRpcRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = "2.0"
    id: JsonRpcId = None
    method: str = Field(min_length=1)
    params: dict[str, Any] | list[Any] | None = None


class JsonRpcSuccessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = "2.0"
    id: JsonRpcId
    result: Any


class JsonRpcErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = "2.0"
    id: JsonRpcId
    error: JsonRpcError


JsonRpcResponse = JsonRpcSuccessResponse | JsonRpcErrorResponse
JsonRpcResponseAdapter = TypeAdapter(JsonRpcResponse)


def parse_request(raw: bytes | str) -> JsonRpcRequest:
    return JsonRpcRequest.model_validate(_loads_json_object(raw))


def parse_response(raw: bytes | str) -> JsonRpcResponse:
    return JsonRpcResponseAdapter.validate_python(_loads_json_object(raw))


def make_success_response(request_id: JsonRpcId, result: Any) -> JsonRpcSuccessResponse:
    return JsonRpcSuccessResponse(id=request_id, result=result)


def make_error_response(
    request_id: JsonRpcId,
    code: JsonRpcErrorCode,
    message: str | None = None,
    data: Any | None = None,
) -> JsonRpcErrorResponse:
    error = JsonRpcError(code=code.value, message=message or _default_error_message(code), data=data)
    return JsonRpcErrorResponse(id=request_id, error=error)


def to_ndjson(model: BaseModel) -> bytes:
    return (model.model_dump_json(exclude_none=True) + "\n").encode("utf-8")


def _loads_json_object(raw: bytes | str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValidationError.from_exception_data(
            "JsonRpcEnvelope",
            [
                {
                    "type": "value_error",
                    "loc": ("json",),
                    "msg": "Invalid JSON",
                    "input": raw,
                    "ctx": {"error": error},
                }
            ],
        ) from error

    if not isinstance(value, dict):
        raise ValidationError.from_exception_data(
            "JsonRpcEnvelope",
            [
                {
                    "type": "value_error",
                    "loc": ("json",),
                    "msg": "JSON-RPC envelope must be an object",
                    "input": value,
                    "ctx": {"error": ValueError("JSON-RPC envelope must be an object")},
                }
            ],
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
