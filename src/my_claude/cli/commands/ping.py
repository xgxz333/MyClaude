from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

from my_claude.core.bus.command import CORE_PING_METHOD, CorePingResult
from my_claude.core.bus.envelope import (
    JsonRpcErrorResponse,
    JsonRpcRequest,
    parse_response,
    to_ndjson,
)
from my_claude.core.config import load_config


async def _ping(host: str, port: int, timeout: float) -> tuple[CorePingResult, float]:
    started_at = time.perf_counter()
    reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)

    try:
        request = JsonRpcRequest(id=1, method=CORE_PING_METHOD, params={})

        writer.write(to_ndjson(request))
        await writer.drain()

        line = await asyncio.wait_for(reader.readline(), timeout)
        elapsed_ms = (time.perf_counter() - started_at) * 1000

        if not line:
            raise RuntimeError("connection closed before ping response")

        response = parse_response(line)
        if response.id != request.id:
            raise RuntimeError("ping response id does not match request id")
        if isinstance(response, JsonRpcErrorResponse):
            raise RuntimeError(f"ping failed: {response.error.message}")

        return CorePingResult.model_validate(response.result), elapsed_ms
    finally:
        writer.close()
        await writer.wait_closed()


def main(args: argparse.Namespace) -> int:
    config = load_config()
    host = args.host if args.host is not None else config.core_host
    port = args.port if args.port is not None else config.core_port
    timeout = args.timeout if args.timeout is not None else config.ipc_timeout_seconds

    try:
        result, elapsed_ms = asyncio.run(_ping(host, port, timeout))
    except OSError as error:
        print(f"ping failed: {error}")
        return 1
    except TimeoutError:
        print(f"ping timed out after {timeout:.2f}s")
        return 1
    except RuntimeError as error:
        print(str(error))
        return 1
    except ValueError as error:
        print(f"invalid ping response: {error}")
        return 1

    print(format_ping_output(result, elapsed_ms))
    return 0


def format_ping_output(result: CorePingResult, latency_ms: float) -> str:
    result_data: dict[str, Any] = result.model_dump()
    result_data["latency_ms"] = round(latency_ms, 2)
    return json.dumps(result_data, ensure_ascii=False, separators=(",", ":"))
