"""TCP transport for newline-delimited JSON-RPC requests."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from contextvars import ContextVar

from pydantic import BaseModel, ValidationError

from my_claude.core.bus.command import BusResult, command_from_request, result_to_json
from my_claude.core.bus.envelope import (
    EventPushEnvelope,
    JsonRpcErrorCode,
    JsonRpcErrorResponse,
    JsonRpcId,
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcSuccessResponse,
    make_error_response,
    make_success_response,
    parse_request,
    to_ndjson,
)

RouteHandler = Callable[[JsonRpcRequest], Awaitable[BusResult]]
JsonRpcReply = JsonRpcSuccessResponse | JsonRpcErrorResponse
ConnectionClosedCallback = Callable[[], None]
_CURRENT_TCP_CONNECTION: ContextVar[TCPConnection | None] = ContextVar(
    "current_tcp_connection",
    default=None,
)


class TCPConnection:
    """Connection-scoped writer used by request handlers for server push messages."""

    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self._writer = writer
        self._write_lock = asyncio.Lock()
        self._close_callbacks: list[ConnectionClosedCallback] = []
        self._closed = False

    def add_close_callback(self, callback: ConnectionClosedCallback) -> None:
        if self._closed:
            callback()
            return

        self._close_callbacks.append(callback)

    async def write_model(self, model: BaseModel) -> None:
        async with self._write_lock:
            self._writer.write(to_ndjson(model))
            await self._writer.drain()

    async def write_notification(self, notification: JsonRpcNotification) -> None:
        await self.write_notifications([notification])

    async def write_notifications(self, notifications: list[JsonRpcNotification]) -> None:
        async with self._write_lock:
            for notification in notifications:
                self._writer.write(to_ndjson(notification))
            await self._writer.drain()

    async def write_event(self, event: EventPushEnvelope) -> None:
        await self.write_events([event])

    async def write_events(self, events: list[EventPushEnvelope]) -> None:
        async with self._write_lock:
            for event in events:
                self._writer.write(to_ndjson(event))
            await self._writer.drain()

    async def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        callbacks = tuple(self._close_callbacks)
        self._close_callbacks.clear()
        for callback in callbacks:
            with suppress(Exception):
                callback()

        self._writer.close()
        with suppress(Exception):
            await asyncio.wait_for(self._writer.wait_closed(), timeout=1.0)


def current_tcp_connection() -> TCPConnection:
    """Return the TCP connection currently dispatching this request."""
    connection = _CURRENT_TCP_CONNECTION.get()
    if connection is None:
        raise RuntimeError("no current TCP connection is active")

    return connection


class TCPServer:
    """Async TCP server that parses requests, dispatches routes, and writes JSON-RPC replies."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        max_request_bytes: int,
        logger: logging.Logger | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.max_request_bytes = max_request_bytes
        self.logger = logger or logging.getLogger(__name__)
        self.routes: dict[str, RouteHandler] = {}
        self._server: asyncio.Server | None = None
        self._connection_tasks: set[asyncio.Task[None]] = set()

    def register(self, method: str, handler: RouteHandler) -> None:
        self.routes[method] = handler

    async def start(self) -> None:
        if self.port != 0:
            try:
                _reader, writer = await asyncio.open_connection(self.host, self.port)
            except OSError:
                pass
            else:
                writer.close()
                with suppress(Exception):
                    await writer.wait_closed()
                raise SystemExit(f"core already running at {self.host}:{self.port}")

        self._server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port,
            limit=self.max_request_bytes + 1,
        )
        sockets = ", ".join(str(socket.getsockname()) for socket in self._server.sockets or [])
        self.logger.info("tcp server started on %s", sockets)

    async def serve_until_stopped(self, shutdown_event: asyncio.Event) -> None:
        await self.start()

        try:
            await shutdown_event.wait()
            self.logger.info("tcp server shutdown requested")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        if self._server is not None:
            self._server.close()
            with suppress(TimeoutError):
                await asyncio.wait_for(self._server.wait_closed(), timeout=2.0)
            self._server = None

        if self._connection_tasks:
            for task in self._connection_tasks:
                task.cancel()
            done, pending = await asyncio.wait(self._connection_tasks, timeout=2.0)
            for task in done:
                if task.cancelled():
                    continue
                with suppress(Exception):
                    task.result()
            if pending:
                self.logger.warning(
                    "tcp server shutdown left %d connection task(s) pending",
                    len(pending),
                )
            self._connection_tasks.clear()

        self.logger.info("tcp server stopped")

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._connection_tasks.add(task)

        peer = writer.get_extra_info("peername")
        connection = TCPConnection(writer)
        self.logger.debug("client connected: %s", peer)

        try:
            while not reader.at_eof():
                try:
                    line = await reader.readuntil(b"\n")
                except asyncio.IncompleteReadError as error:
                    if len(error.partial) > self.max_request_bytes:
                        await self._write_response(writer, self._request_too_large_error())
                    elif error.partial:
                        await self._write_response(writer, self._incomplete_request_error())
                    break
                except asyncio.LimitOverrunError:
                    await self._write_response(writer, self._request_too_large_error())
                    break

                if len(line) > self.max_request_bytes:
                    await self._write_response(writer, self._request_too_large_error())
                    break

                response = await self.dispatch(line, connection)
                await connection.write_model(response)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception("client handling failed: %s", peer)
            with suppress(Exception):
                await connection.write_model(
                    make_error_response(None, JsonRpcErrorCode.INTERNAL_ERROR),
                )
        finally:
            await connection.close()
            if task is not None:
                self._connection_tasks.discard(task)
            self.logger.debug("client disconnected: %s", peer)

    async def dispatch(self, line: bytes, connection: TCPConnection) -> JsonRpcReply:
        request_id: JsonRpcId = None

        try:
            request = parse_request(line)
            request_id = request.id
        except ValidationError as error:
            return make_error_response(
                request_id,
                _request_validation_error_code(error),
                data=error.errors(),
            )

        try:
            handler = self.routes.get(request.method)
            if handler is None:
                return make_error_response(
                    request_id,
                    JsonRpcErrorCode.METHOD_NOT_FOUND,
                    f"method not found: {request.method}",
                )

            command_from_request(request)
            token = _CURRENT_TCP_CONNECTION.set(connection)
            try:
                result = await handler(request)
            finally:
                _CURRENT_TCP_CONNECTION.reset(token)
            return make_success_response(request_id, result_to_json(result))
        except (ValidationError, ValueError) as error:
            return make_error_response(
                request_id,
                JsonRpcErrorCode.INVALID_PARAMS,
                str(error),
            )
        except Exception as error:
            self.logger.exception("request dispatch failed")
            return make_error_response(
                request_id,
                JsonRpcErrorCode.INTERNAL_ERROR,
                data={"error": str(error)},
            )

    async def _write_response(self, writer: asyncio.StreamWriter, response: JsonRpcReply) -> None:
        writer.write(to_ndjson(response))
        await writer.drain()

    def _request_too_large_error(self) -> JsonRpcErrorResponse:
        return make_error_response(
            None,
            JsonRpcErrorCode.INVALID_REQUEST,
            f"request too large: max {self.max_request_bytes} bytes",
        )

    def _incomplete_request_error(self) -> JsonRpcErrorResponse:
        return make_error_response(
            None,
            JsonRpcErrorCode.INVALID_REQUEST,
            "request must be newline-delimited JSON",
        )


def _request_validation_error_code(error: ValidationError) -> JsonRpcErrorCode:
    for item in error.errors():
        if item.get("loc") == ("json",) and "Invalid JSON" in str(item.get("msg", "")):
            return JsonRpcErrorCode.PARSE_ERROR
    return JsonRpcErrorCode.INVALID_REQUEST
