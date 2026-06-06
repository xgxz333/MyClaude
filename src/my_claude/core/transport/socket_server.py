from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress

from pydantic import ValidationError

from my_claude.core.bus.command import BusResult, command_from_request, result_to_json
from my_claude.core.bus.envelope import (
    JsonRpcErrorCode,
    JsonRpcErrorResponse,
    JsonRpcId,
    JsonRpcRequest,
    JsonRpcSuccessResponse,
    make_error_response,
    make_success_response,
    parse_request,
    to_ndjson,
)


RouteHandler = Callable[[JsonRpcRequest], Awaitable[BusResult]]
JsonRpcReply = JsonRpcSuccessResponse | JsonRpcErrorResponse


class TCPServer:
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
            await self._server.wait_closed()
            self._server = None

        if self._connection_tasks:
            for task in self._connection_tasks:
                task.cancel()
            await asyncio.gather(*self._connection_tasks, return_exceptions=True)
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

                response = await self.dispatch(line)
                await self._write_response(writer, response)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception("client handling failed: %s", peer)
            with suppress(Exception):
                await self._write_response(
                    writer,
                    make_error_response(None, JsonRpcErrorCode.INTERNAL_ERROR),
                )
        finally:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()
            if task is not None:
                self._connection_tasks.discard(task)
            self.logger.debug("client disconnected: %s", peer)

    async def dispatch(self, line: bytes) -> JsonRpcReply:
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
            result = await handler(request)
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
