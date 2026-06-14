from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pytest

from my_claude.cli.commands.chat import _chat_async, _readline
from my_claude.cli.main import main
from my_claude.core.app import register_routes
from my_claude.core.bus.command import (
    EVENT_SUBSCRIBE_METHOD,
    SESSION_CREATE_METHOD,
    SESSION_SEND_MESSAGE_METHOD,
)
from my_claude.core.config import AppConfig
from my_claude.core.transport.socket_server import TCPServer


def test_chat_command_is_registered(capsys) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SystemExit) as exc_info:
        main(["chat", "--help"])

    captured = capsys.readouterr()

    assert exc_info.value.code == 0
    assert "Open an interactive chat session." in captured.out
    assert "--session-id" in captured.out


def test_chat_command_creates_session_sends_message_and_streams_response(
    tmp_path: Path,
) -> None:
    exit_code, output, errors, methods = asyncio.run(_run_chat_via_test_daemon(tmp_path))

    assert exit_code == 0
    assert methods[:3] == [
        EVENT_SUBSCRIBE_METHOD,
        SESSION_CREATE_METHOD,
        SESSION_SEND_MESSAGE_METHOD,
    ]
    assert "[chat] session sess-" in output
    assert "you> assistant> Local LLM placeholder accepted goal: hello" in output
    assert "Registered tools: 9" in output
    assert errors == ""


def test_readline_uses_injected_stream_without_blocking_stdin() -> None:
    output = io.StringIO()
    line = asyncio.run(
        _readline(
            "you> ",
            input_stream=io.StringIO("hello\n"),
            output_stream=output,
        )
    )

    assert line == "hello"
    assert output.getvalue() == "you> "


async def _run_chat_via_test_daemon(tmp_path: Path) -> tuple[int, str, str, list[str]]:
    config = AppConfig(runs_dir=tmp_path / "runs", llm_provider="local")
    server = TCPServer("127.0.0.1", 0, max_request_bytes=config.max_request_bytes)
    register_routes(server, config=config, server_version="test-version")
    methods: list[str] = []

    for method, handler in tuple(server.routes.items()):

        async def recording_handler(request, *, method=method, handler=handler):  # type: ignore[no-untyped-def]
            methods.append(method)
            return await handler(request)

        server.routes[method] = recording_handler

    await server.start()
    try:
        client_config = config.model_copy(
            update={
                "core_host": "127.0.0.1",
                "core_port": _bound_port(server),
            }
        )
        output = io.StringIO()
        errors = io.StringIO()
        exit_code = await _chat_async(
            config=client_config,
            input_stream=io.StringIO("hello\n/exit\n"),
            output_stream=output,
            error_stream=errors,
        )
        return exit_code, output.getvalue(), errors.getvalue(), methods
    finally:
        await server.shutdown()


def _bound_port(server: TCPServer) -> int:
    if server._server is None or not server._server.sockets:
        raise RuntimeError("test server did not start")

    return int(server._server.sockets[0].getsockname()[1])
