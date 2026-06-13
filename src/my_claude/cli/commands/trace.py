"""Implementation of the `myclaude trace` CLI command."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TextIO

from my_claude.core.config import AppConfig, load_config
from my_claude.core.trace import TraceDirection, TraceLayer, TraceRecord
from my_claude.core.trace.paths import DAEMON_TRACE_FILENAME

_COLORS = {
    "CLIENT→CORE": "\033[36m",
    "CORE→CLIENT": "\033[33m",
    "CORE": "\033[32m",
    "CORE→LLM": "\033[35m",
    "LLM→CORE": "\033[34m",
}
_RESET = "\033[0m"
_BOLD = "\033[1m"


def main(args: argparse.Namespace) -> int:
    config = load_config()
    path = _trace_path(args, config)
    run_id = _run_id_filter(args)
    raw = bool(args.raw or args.json)

    if args.limit is not None and args.limit < 0:
        print("--limit must be greater than or equal to 0", file=sys.stderr)
        return 1

    if not path.exists():
        print(f"trace file not found: {path}", file=sys.stderr)
        return 1

    try:
        records = read_trace_records(path)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    records = filter_trace_records(
        records,
        run_id=run_id,
        layer=args.layer,
        direction=args.direction,
    )
    if args.limit is not None:
        records = records[-args.limit :] if args.limit else []

    print_trace_records(records, raw=raw)

    if args.follow:
        try:
            follow_trace_file(
                path,
                run_id=run_id,
                layer=args.layer,
                direction=args.direction,
                raw=raw,
            )
        except KeyboardInterrupt:
            return 130

    return 0


def read_trace_records(path: Path) -> list[TraceRecord]:
    records: list[TraceRecord] = []

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = parse_trace_line(line)
        if record is None:
            raise ValueError(f"invalid trace record: {path}:{line_number}")
        records.append(record)

    return records


def parse_trace_line(line: str) -> TraceRecord | None:
    try:
        payload = json.loads(line)
    except Exception:
        return None

    try:
        return TraceRecord.model_validate(payload)
    except Exception:
        return _legacy_trace_record(payload)


def filter_trace_records(
    records: list[TraceRecord],
    *,
    run_id: str | None,
    layer: str | None,
    direction: str | None,
) -> list[TraceRecord]:
    return [
        record
        for record in records
        if _matches_filters(record, run_id=run_id, layer=layer, direction=direction)
    ]


def print_trace_records(
    records: list[TraceRecord],
    *,
    raw: bool = False,
    stream: TextIO | None = None,
) -> None:
    output = stream if stream is not None else sys.stdout
    for record in records:
        if raw:
            print(record.model_dump_json(exclude_none=True), file=output)
        else:
            print(format_trace_record(record), file=output)


def follow_trace_file(
    path: Path,
    *,
    run_id: str | None,
    layer: str | None,
    direction: str | None,
    raw: bool,
) -> None:
    with path.open(encoding="utf-8") as file:
        file.seek(0, 2)
        while True:
            line = file.readline()
            if not line:
                time.sleep(0.05)
                continue

            record = parse_trace_line(line.strip())
            if record is None:
                continue
            if not _matches_filters(record, run_id=run_id, layer=layer, direction=direction):
                continue
            print_trace_records([record], raw=raw)


def format_trace_record(record: TraceRecord) -> str:
    color = _COLORS.get(record.direction, "")
    ts = record.ts[11:23] if len(record.ts) >= 23 else record.ts
    direction = f"{color}{_BOLD}{record.direction:<14}{_RESET}"
    kind = f"{record.kind:<13}"

    parts: list[str] = []
    if record.run_id:
        parts.append(f"run={record.run_id[:8]}")
    if record.step is not None:
        parts.append(f"step={record.step}")
    parts.append(_summarize(record))

    return f"{ts}  {direction}  {kind}  {'  '.join(parts)}"


def _trace_path(args: argparse.Namespace, config: AppConfig) -> Path:
    path = args.path
    if isinstance(path, str):
        return Path(path)
    return config.trace_file or config.runs_dir / DAEMON_TRACE_FILENAME


def _run_id_filter(args: argparse.Namespace) -> str | None:
    if isinstance(args.run_id_flag, str):
        return args.run_id_flag
    if isinstance(args.run_id, str):
        return args.run_id
    return None


def _matches_filters(
    record: TraceRecord,
    *,
    run_id: str | None,
    layer: str | None,
    direction: str | None,
) -> bool:
    if run_id is not None and record.run_id != run_id:
        return False
    if layer is not None and record.layer != layer:
        return False
    if direction is not None and record.direction != direction:
        return False
    return True


def _summarize(record: TraceRecord) -> str:
    data = record.data
    kind = record.kind

    if kind == "command":
        params = data.get("params", {})
        goal = params.get("goal", "") if isinstance(params, dict) else ""
        suffix = f'  goal="{str(goal)[:50]}"' if goal else ""
        return f"method={data.get('method')}{suffix}"

    if kind == "response":
        result = data.get("result", {})
        if isinstance(result, dict) and "run_id" in result:
            return f"run_id={str(result['run_id'])[:8]}"
        return str(result)[:60]

    if kind == "error":
        error = data.get("error", {})
        if isinstance(error, dict):
            return f"code={error.get('code')}  {error.get('message', '')}"
        return str(error)[:60]

    if kind == "push":
        return f"event={data.get('event_type')}  sub={data.get('sub_id')}"

    if kind == "event":
        return f"type={data.get('type')}"

    if kind == "api_call":
        messages = data.get("messages")
        message_count = (
            len(messages) if isinstance(messages, list) else data.get("message_count", "?")
        )
        tools = data.get("tool_schemas")
        tool_count = len(tools) if isinstance(tools, list) else data.get("tool_count", "?")
        return f"msgs={message_count}  tools={tool_count}"

    if kind == "api_response":
        usage = data.get("usage", {})
        output_tokens = usage.get("output_tokens", "?") if isinstance(usage, dict) else "?"
        return (
            f"stop={data.get('stop_reason')}  "
            f"latency={data.get('latency_ms')}ms  "
            f"out_tokens={output_tokens}"
        )

    return str(data)[:60]


def _legacy_trace_record(payload: object) -> TraceRecord | None:
    if not isinstance(payload, dict):
        return None

    timestamp = payload.get("timestamp")
    if not isinstance(timestamp, str):
        return None

    legacy_payload = payload.get("payload")
    data = dict(legacy_payload) if isinstance(legacy_payload, dict) else {}
    operation = str(payload.get("operation", ""))
    legacy_layer = str(payload.get("layer", ""))
    legacy_direction = str(payload.get("direction", ""))
    run_id = _legacy_run_id(payload, data)
    step = data.get("step")

    if legacy_layer == "event_bus" or operation == "event":
        event_data = data.get("event")
        if isinstance(event_data, dict):
            data = dict(event_data)
        layer: TraceLayer = "event"
        direction: TraceDirection = "CORE"
        kind = "event"
    elif legacy_layer == "llm":
        layer = "llm"
        if operation == "request":
            direction = "CORE→LLM"
            kind = "api_call"
            if "tool_schema_count" in data and "tool_count" not in data:
                data["tool_count"] = data["tool_schema_count"]
        else:
            direction = "LLM→CORE"
            kind = "api_response"
            if "elapsed_ms" in data and "latency_ms" not in data:
                data["latency_ms"] = data["elapsed_ms"]
    elif legacy_layer == "ipc":
        layer = "ipc"
        direction = "CORE→CLIENT" if legacy_direction == "outbound" else "CLIENT→CORE"
        kind = _legacy_ipc_kind(operation, data)
        data = _legacy_ipc_data(operation, kind, data)
    else:
        return None

    return TraceRecord(
        ts=timestamp,
        direction=direction,
        layer=layer,
        kind=kind,
        run_id=run_id,
        step=step if isinstance(step, int) else None,
        client_id=data.get("client") if isinstance(data.get("client"), str) else None,
        data=data,
    )


def _legacy_run_id(payload: dict[object, object], data: dict[str, object]) -> str | None:
    run_id = payload.get("run_id")
    if isinstance(run_id, str):
        return run_id

    event = data.get("event")
    if isinstance(event, dict):
        event_run_id = event.get("run_id")
        if isinstance(event_run_id, str):
            return event_run_id

    payload_run_id = data.get("run_id")
    if isinstance(payload_run_id, str):
        return payload_run_id

    return None


def _legacy_ipc_kind(operation: str, data: dict[str, object]) -> str:
    if operation == "command.received":
        return "command"
    if operation == "event.push":
        return "push"
    if operation == "response.sent":
        legacy_kind = data.get("kind")
        return "error" if legacy_kind == "jsonrpc.error" else "response"
    return operation or "event"


def _legacy_ipc_data(
    operation: str,
    kind: str,
    data: dict[str, object],
) -> dict[str, object]:
    if operation == "command.received":
        params = data.get("params")
        if not isinstance(params, dict):
            params = {"kind": data.get("params_kind")}
        return {
            "method": data.get("method"),
            "id": data.get("request_id"),
            "params": params,
            "bytes": data.get("byte_count"),
            **data,
        }

    if operation == "response.sent":
        if kind == "error":
            return {"error": data, **data}
        return {"id": data.get("request_id"), "result": data, **data}

    if operation == "event.push":
        return {
            "sub_id": data.get("sub_id"),
            "event_type": data.get("event_type"),
        }

    return data
