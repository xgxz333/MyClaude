from __future__ import annotations

import json
import sys
from enum import IntEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUTPUT = ROOT / "WIRE_PROTOCOL.md"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from my_claude.core.bus.command import (  # noqa: E402
    CORE_PING_METHOD,
    CorePingCommand,
    CorePingParams,
    CorePingResult,
)
from my_claude.core.bus.envelope import (  # noqa: E402
    JsonRpcError,
    JsonRpcErrorCode,
    JsonRpcErrorResponse,
    JsonRpcRequest,
    JsonRpcSuccessResponse,
)


def main() -> int:
    OUTPUT.write_text(generate_doc(), encoding="utf-8")
    print(f"generated {OUTPUT.relative_to(ROOT)}")
    return 0


def generate_doc() -> str:
    sections = [
        "# Wire Protocol",
        "",
        "This document is generated from the Pydantic protocol models in `src/my_claude/core/bus`.",
        "",
        "## Transport",
        "",
        "- Transport: TCP",
        "- Encoding: UTF-8",
        "- Framing: NDJSON. Each request and response is one JSON value followed by `\\n`.",
        "- JSON-RPC version: `2.0`",
        "- Maximum request size: configured by `max_request_bytes`; default is `65536` bytes.",
        "",
        "## JSON-RPC Envelope",
        "",
        _schema_block("JsonRpcRequest", JsonRpcRequest),
        "",
        _schema_block("JsonRpcSuccessResponse", JsonRpcSuccessResponse),
        "",
        _schema_block("JsonRpcErrorResponse", JsonRpcErrorResponse),
        "",
        _schema_block("JsonRpcError", JsonRpcError),
        "",
        "## Error Codes",
        "",
        _error_code_table(JsonRpcErrorCode),
        "",
        "## Commands",
        "",
        f"### `{CORE_PING_METHOD}`",
        "",
        "S0 health check command.",
        "",
        "**Params**",
        "",
        _schema_block("CorePingParams", CorePingParams),
        "",
        "**Command Model**",
        "",
        _schema_block("CorePingCommand", CorePingCommand),
        "",
        "**Result**",
        "",
        _schema_block("CorePingResult", CorePingResult),
        "",
        "**Request Example**",
        "",
        _json_block(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": CORE_PING_METHOD,
                "params": {},
            }
        ),
        "",
        "**Success Response Example**",
        "",
        _json_block(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "ok": True,
                    "message": "pong",
                },
            }
        ),
        "",
        "**Error Response Example**",
        "",
        _json_block(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "error": {
                    "code": JsonRpcErrorCode.METHOD_NOT_FOUND.value,
                    "message": "method not found: unknown.method",
                },
            }
        ),
        "",
    ]
    return "\n".join(sections)


def _schema_block(name: str, model: type[BaseModel]) -> str:
    return "\n".join(
        [
            f"### `{name}`",
            "",
            "```json",
            json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2),
            "```",
        ]
    )


def _error_code_table(error_codes: type[IntEnum]) -> str:
    lines = [
        "| Name | Code |",
        "| --- | ---: |",
    ]
    for code in error_codes:
        lines.append(f"| `{code.name}` | `{code.value}` |")
    return "\n".join(lines)


def _json_block(value: dict[str, Any]) -> str:
    return "\n".join(
        [
            "```json",
            json.dumps(value, ensure_ascii=False, indent=2),
            "```",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
