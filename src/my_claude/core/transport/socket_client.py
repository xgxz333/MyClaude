"""Async JSON-RPC socket client with server-push notification callbacks."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

from my_claude.core.bus.envelope import (
    JsonRpcErrorResponse,
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcSuccessResponse,
    parse_notification,
    parse_response,
    to_ndjson,
)

NotificationHandler = Callable[[dict[str, Any]], Awaitable[None]]


class SocketClientError(RuntimeError):
    """Raised when the socket client cannot complete a request."""


class SocketFrameDispatcher:
    """Demultiplex one socket byte stream into responses and pushed notifications."""

    def __init__(
        self,
        pending: dict[int, asyncio.Future[Any]],
        notification_handlers: dict[str, list[NotificationHandler]],
    ) -> None:
        self._pending = pending
        self._notification_handlers = notification_handlers

    async def dispatch_line(self, line: bytes) -> None:
        frame = self._deserialize(line)
        if isinstance(frame, JsonRpcSuccessResponse | JsonRpcErrorResponse):
            self._dispatch_response(frame)
            return

        await self._dispatch_notification(frame)

    def _deserialize(self, line: bytes) -> JsonRpcResponse | JsonRpcNotification:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise SocketClientError("invalid JSON-RPC frame") from error

        if not isinstance(payload, dict):
            raise SocketClientError("JSON-RPC frame must be an object")

        if "id" in payload:
            return parse_response(line)

        if "method" in payload:
            return parse_notification(line)

        raise SocketClientError("JSON-RPC frame must be a response or notification")

    def _dispatch_response(self, response: JsonRpcResponse) -> None:
        if not isinstance(response.id, int):
            return

        response_future = self._pending.pop(response.id, None)
        if response_future is None or response_future.done():
            return

        if isinstance(response, JsonRpcErrorResponse):
            response_future.set_exception(
                SocketClientError(f"{response.error.message}: {response.error.data}")
            )
            return

        response_future.set_result(response.result)

    async def _dispatch_notification(self, notification: JsonRpcNotification) -> None:
        if not isinstance(notification.params, dict):
            raise SocketClientError("JSON-RPC notification params must be an object")

        handlers = tuple(self._notification_handlers.get(notification.method, ()))
        for handler in handlers:
            await handler(notification.params)


class SocketClient:
    """Maintain one TCP JSON-RPC connection and dispatch pushed notifications."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout_seconds: float,
        max_response_bytes: int = 65536,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._next_id = 1
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._notification_handlers: dict[str, list[NotificationHandler]] = {}
        self._dispatcher = SocketFrameDispatcher(self._pending, self._notification_handlers)
        self._closed = False

    async def __aenter__(self) -> SocketClient:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    async def connect(self) -> None:
        if self._writer is not None:
            return

        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(
                    self.host,
                    self.port,
                    limit=self.max_response_bytes + 1,
                ),
                self.timeout_seconds,
            )
        except TimeoutError as error:
            raise SocketClientError(
                f"connect timed out after {self.timeout_seconds:.2f}s"
            ) from error
        except OSError as error:
            raise SocketClientError(f"connect failed: {error}") from error

        self._closed = False
        self._reader_task = asyncio.create_task(self._read_loop())

    def on(self, method: str, handler: NotificationHandler) -> None:
        self._notification_handlers.setdefault(method, []).append(handler)

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> Any:
        writer = self._require_writer()
        request_id = self._next_id
        self._next_id += 1
        loop = asyncio.get_running_loop()
        response_future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = response_future

        request = JsonRpcRequest(id=request_id, method=method, params=params or {})
        try:
            writer.write(to_ndjson(request))
            await writer.drain()
        except OSError as error:
            self._pending.pop(request_id, None)
            raise SocketClientError(f"request send failed: {error}") from error

        try:
            timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
            if timeout <= 0:
                return await response_future
            return await asyncio.wait_for(response_future, timeout)
        except asyncio.CancelledError:
            self._pending.pop(request_id, None)
            raise
        except TimeoutError as error:
            self._pending.pop(request_id, None)
            raise SocketClientError(f"{method} timed out after {timeout:.2f}s") from error

    async def close(self) -> None:
        self._closed = True

        if self._reader_task is not None:
            self._reader_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._reader_task
            self._reader_task = None

        writer = self._writer
        self._reader = None
        self._writer = None
        if writer is not None:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()

        self._fail_pending(SocketClientError("connection closed"))

    async def wait_closed(self) -> None:
        if self._reader_task is None:
            return

        await self._reader_task

    @property
    def is_connected(self) -> bool:
        return self._writer is not None and not self._closed

    def _require_writer(self) -> asyncio.StreamWriter:
        if self._writer is None:
            raise SocketClientError("socket client is not connected")
        return self._writer

    async def _read_loop(self) -> None:
        reader = self._reader
        if reader is None:
            return

        try:
            while not reader.at_eof():
                try:
                    line = await reader.readuntil(b"\n")
                except asyncio.IncompleteReadError as error:
                    if error.partial:
                        self._fail_pending(SocketClientError("incomplete response frame"))
                    break
                except asyncio.LimitOverrunError as error:
                    self._fail_pending(
                        SocketClientError(
                            f"response too large: max {self.max_response_bytes} bytes"
                        )
                    )
                    raise SocketClientError("response too large") from error

                await self._dispatcher.dispatch_line(line)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._fail_pending(SocketClientError(f"socket read failed: {error}"))
        finally:
            if not self._closed:
                self._fail_pending(SocketClientError("connection closed by server"))

    def _fail_pending(self, error: Exception) -> None:
        for response_future in tuple(self._pending.values()):
            if not response_future.done():
                response_future.set_exception(error)
        self._pending.clear()
