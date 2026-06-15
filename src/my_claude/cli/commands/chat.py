"""Implementation of the `myclaude chat` interactive command."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import MutableMapping
from typing import Any, TextIO, cast

from my_claude.agent.events import AgentEvent, AgentEventType, KnownAgentEventAdapter
from my_claude.core.bus.command import (
    EVENT_SUBSCRIBE_METHOD,
    SESSION_CLOSE_METHOD,
    SESSION_CREATE_METHOD,
    SESSION_SEND_MESSAGE_METHOD,
)
from my_claude.core.config import AppConfig, load_config
from my_claude.core.transport.socket_client import SocketClient, SocketClientError

EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}


class ChatEventRenderer:
    """Render daemon-pushed events for the currently active chat turn."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        error_stream: TextIO | None = None,
    ) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._error_stream = error_stream if error_stream is not None else sys.stderr
        self._current_run_id: str | None = None
        self._pending_by_run: MutableMapping[str, list[AgentEvent]] = {}
        self._terminal_event = asyncio.Event()
        self._terminal_exit_code = 0
        self._response_started = False
        self._response_had_tokens = False

    async def start_run(self, run_id: str) -> None:
        self._current_run_id = run_id
        self._terminal_event = asyncio.Event()
        self._terminal_exit_code = 0
        self._response_started = False
        self._response_had_tokens = False

        for event in self._pending_by_run.pop(run_id, []):
            await self._render(event)

    async def handle_event_payload(self, event_payload: dict[str, Any]) -> None:
        event = KnownAgentEventAdapter.validate_python(event_payload)
        run_id = _event_run_id(event)
        if run_id is None:
            return
        if run_id != self._current_run_id:
            self._pending_by_run.setdefault(run_id, []).append(event)
            return

        await self._render(event)

    async def wait_for_turn(self) -> int:
        await self._terminal_event.wait()
        self._ensure_newline()
        self._current_run_id = None
        return self._terminal_exit_code

    async def _render(self, event: AgentEvent) -> None:
        if event.type == AgentEventType.LLM_TOKEN:
            self._ensure_assistant_prefix()
            print(event.data.get("token", ""), end="", file=self._stream, flush=True)
            self._response_had_tokens = True
        elif event.type == AgentEventType.LLM_RESPONSE_COMPLETED:
            if not self._response_had_tokens:
                self._ensure_assistant_prefix()
                print(event.data.get("content", ""), end="", file=self._stream, flush=True)
        elif event.type == AgentEventType.TOOL_CALL_STARTED:
            self._ensure_newline()
            tool_name = event.data.get("tool_name", "")
            arguments = json.dumps(event.data.get("arguments", {}), ensure_ascii=False)
            print(f"[tool] {tool_name} {arguments}", file=self._stream, flush=True)
        elif event.type == AgentEventType.TOOL_CALL_COMPLETED:
            self._ensure_newline()
            tool_name = event.data.get("tool_name", "")
            error = event.data.get("error")
            if error:
                print(f"[tool] {tool_name} failed: {error}", file=self._error_stream, flush=True)
            else:
                elapsed_ms = event.data.get("elapsed_ms", 0)
                print(f"[tool] {tool_name} done {elapsed_ms}ms", file=self._stream, flush=True)
        elif event.type == AgentEventType.RUN_COMPLETED:
            self._terminal_exit_code = 0
        elif event.type == AgentEventType.SESSION_WAITING_FOR_INPUT:
            self._terminal_exit_code = 0
            self._terminal_event.set()
        elif event.type == AgentEventType.RUN_CANCELLED:
            self._terminal_exit_code = 130
            self._terminal_event.set()
        elif event.type == AgentEventType.RUN_FAILED:
            self._ensure_newline()
            error = event.data.get("error") or event.message
            print(f"[chat] run failed: {error}", file=self._error_stream, flush=True)
            self._terminal_exit_code = 1
            self._terminal_event.set()

    def _ensure_assistant_prefix(self) -> None:
        if self._response_started:
            return

        print("assistant> ", end="", file=self._stream, flush=True)
        self._response_started = True

    def _ensure_newline(self) -> None:
        if not self._response_started:
            return

        print(file=self._stream, flush=True)
        self._response_started = False


def main(args: argparse.Namespace) -> int:
    try:
        return asyncio.run(_chat_async(args))
    except KeyboardInterrupt:
        print("[chat] cancelled", file=sys.stderr, flush=True)
        return 130
    except SocketClientError as error:
        print(f"[chat] failed: {error}", file=sys.stderr, flush=True)
        return 1


async def _chat_async(
    args: argparse.Namespace | None = None,
    *,
    config: AppConfig | None = None,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    error_stream: TextIO | None = None,
) -> int:
    runtime_config = config or load_config()
    stream = output_stream if output_stream is not None else sys.stdout
    errors = error_stream if error_stream is not None else sys.stderr
    host = _arg_value(args, "host") or runtime_config.core_host
    port = _arg_value(args, "port") or runtime_config.core_port
    timeout = _arg_value(args, "timeout") or runtime_config.ipc_timeout_seconds
    session_id = _arg_value(args, "session_id")
    title = _arg_value(args, "title")

    renderer = ChatEventRenderer(stream=stream, error_stream=errors)

    async with SocketClient(
        str(host),
        int(port),
        timeout_seconds=float(timeout),
        max_response_bytes=runtime_config.max_request_bytes,
    ) as client:
        client.on_event(renderer.handle_event_payload)

        # Subscribe before creating/sending so pushed lifecycle events cannot be missed.
        await client.request(
            EVENT_SUBSCRIBE_METHOD,
            {
                "topics": [
                    "session.*",
                    "run.*",
                    "tool.*",
                    "llm.token",
                    "llm.response_completed",
                ],
                "scope": "global",
            },
        )

        if session_id is None:
            params: dict[str, Any] = {}
            if title is not None:
                params["title"] = title
            create_result = await client.request(SESSION_CREATE_METHOD, params)
            session_id = str(create_result["session_id"])

        print(f"[chat] session {session_id}", file=stream, flush=True)

        while True:
            line = await _readline(
                "you> ",
                input_stream=input_stream,
                output_stream=stream,
            )
            if line is None:
                await client.request(
                    SESSION_CLOSE_METHOD,
                    {"session_id": session_id},
                    timeout_seconds=float(timeout),
                )
                print(file=stream, flush=True)
                return 0

            message = line.strip()
            if not message:
                continue
            if message in EXIT_COMMANDS:
                await client.request(
                    SESSION_CLOSE_METHOD,
                    {"session_id": session_id},
                    timeout_seconds=float(timeout),
                )
                return 0

            try:
                result = await client.request(
                    SESSION_SEND_MESSAGE_METHOD,
                    {"session_id": session_id, "content": message},
                    timeout_seconds=float(timeout),
                )
            except SocketClientError as error:
                print(f"[chat] send failed: {error}", file=errors, flush=True)
                return 1

            await renderer.start_run(str(result["run_id"]))
            exit_code = await renderer.wait_for_turn()
            if exit_code != 0:
                return exit_code


async def _readline(
    prompt: str,
    *,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> str | None:
    stream = input_stream if input_stream is not None else sys.stdin
    output = output_stream if output_stream is not None else sys.stdout

    if stream is sys.stdin:
        try:
            return cast(str, await asyncio.to_thread(input, prompt))
        except EOFError:
            return None

    output.write(prompt)
    output.flush()
    line = cast(str, await asyncio.to_thread(stream.readline))
    if line == "":
        return None
    return line.rstrip("\n")


def _arg_value(args: argparse.Namespace | None, name: str) -> Any:
    if args is None:
        return None
    return getattr(args, name, None)


def _event_run_id(event: AgentEvent) -> str | None:
    last_run_id = event.data.get("last_run_id")
    if isinstance(last_run_id, str):
        return last_run_id

    run_id = event.data.get("run_id")
    if isinstance(run_id, str):
        return run_id
    return None
