from __future__ import annotations

import asyncio
from typing import Any

import pytest

from my_claude.core.bus.envelope import (
    JsonRpcErrorCode,
    make_error_response,
    make_notification,
    make_success_response,
    to_ndjson,
)
from my_claude.core.transport.socket_client import SocketClientError, SocketFrameDispatcher


def test_socket_frame_dispatcher_routes_mixed_stream_frames() -> None:
    calls, result = asyncio.run(_dispatch_mixed_stream_frames())

    assert calls == [
        ("first", 1),
        ("second", 1),
        ("first", 2),
        ("second", 2),
    ]
    assert result == {"run_id": "run-1"}


async def _dispatch_mixed_stream_frames() -> tuple[list[tuple[str, int]], Any]:
    loop = asyncio.get_running_loop()
    response_future: asyncio.Future[Any] = loop.create_future()
    pending = {1: response_future}
    calls: list[tuple[str, int]] = []

    async def first_handler(params: dict[str, Any]) -> None:
        calls.append(("first", int(params["sequence"])))
        await asyncio.sleep(0)

    async def second_handler(params: dict[str, Any]) -> None:
        calls.append(("second", int(params["sequence"])))

    dispatcher = SocketFrameDispatcher(
        pending,
        {"event.publish": [first_handler, second_handler]},
    )

    await dispatcher.dispatch_line(
        to_ndjson(make_notification("event.publish", {"sequence": 1}))
    )
    await dispatcher.dispatch_line(
        to_ndjson(make_success_response(1, {"run_id": "run-1"}))
    )
    await dispatcher.dispatch_line(
        to_ndjson(make_notification("event.publish", {"sequence": 2}))
    )

    return calls, await response_future


def test_socket_frame_dispatcher_sets_jsonrpc_error_on_pending_request() -> None:
    async def dispatch_error() -> None:
        loop = asyncio.get_running_loop()
        response_future: asyncio.Future[Any] = loop.create_future()
        dispatcher = SocketFrameDispatcher({1: response_future}, {})

        await dispatcher.dispatch_line(
            to_ndjson(
                make_error_response(
                    1,
                    JsonRpcErrorCode.INVALID_PARAMS,
                    "bad params",
                    {"field": "goal"},
                )
            )
        )

        with pytest.raises(SocketClientError, match="bad params"):
            await response_future

    asyncio.run(dispatch_error())
