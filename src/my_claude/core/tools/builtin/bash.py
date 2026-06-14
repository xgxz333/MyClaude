"""Built-in bash tool for non-interactive terminal commands."""

from __future__ import annotations

import asyncio
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from my_claude.core.tools.base import ParamsModel, ToolDefinition, ToolResult

DEFAULT_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 120
MAX_OUTPUT_BYTES = 64 * 1024
_READ_CHUNK_BYTES = 8192


class BashParams(BaseModel):
    """Arguments accepted by the bash tool."""

    model_config = ConfigDict(extra="ignore")

    command: Annotated[str, Field(strict=True)]
    timeout: Annotated[
        int,
        Field(default=DEFAULT_TIMEOUT_SECONDS, ge=1, le=MAX_TIMEOUT_SECONDS, strict=True),
    ]


@dataclass(frozen=True)
class BashTool:
    """Execute a non-interactive shell command without blocking the event loop."""

    params_model: ClassVar[ParamsModel] = BashParams

    cwd: Path
    default_timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_timeout_seconds: int = MAX_TIMEOUT_SECONDS
    max_output_bytes: int = MAX_OUTPUT_BYTES

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="bash",
            description=(
                "Execute a non-interactive shell command from the workspace root. "
                "Returns stdout, stderr, and exit status. Commands that wait for "
                "input will time out. Output is truncated at 64 KiB."
            ),
            input_schema={
                "type": "object",
                "required": ["command"],
                "additionalProperties": False,
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": (
                            f"Maximum seconds to wait. Default "
                            f"{self.default_timeout_seconds}, max {self.max_timeout_seconds}."
                        ),
                    },
                },
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        command = str(arguments["command"])
        timeout_seconds = _coerce_timeout(
            arguments.get("timeout"),
            default=self.default_timeout_seconds,
            maximum=self.max_timeout_seconds,
        )

        try:
            return await self._run_command(command, timeout_seconds=timeout_seconds)
        except Exception as error:
            return ToolResult.failure(
                f"failed to execute command: {error}",
                error_type="runtime_error",
            )

    async def _run_command(self, command: str, *, timeout_seconds: int) -> ToolResult:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=self.cwd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
        if process.stdout is None or process.stderr is None:
            await _kill_process(process)
            return ToolResult.failure(
                "failed to capture command output",
                error_type="runtime_error",
            )

        stdout_task = asyncio.create_task(
            _read_limited(process.stdout, max_bytes=self.max_output_bytes)
        )
        stderr_task = asyncio.create_task(
            _read_limited(process.stderr, max_bytes=self.max_output_bytes)
        )

        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
        except TimeoutError:
            timed_out = True
            await _kill_process(process)
            await process.wait()
        except asyncio.CancelledError:
            await _kill_process(process)
            stdout_task.cancel()
            stderr_task.cancel()
            await _wait_for_process_exit(process)
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            raise

        stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
        if timed_out:
            return ToolResult.failure(
                _format_result(
                    command=command,
                    timeout_seconds=timeout_seconds,
                    stdout=stdout,
                    stderr=stderr,
                ),
                error_type="timeout",
            )

        returncode = process.returncode if process.returncode is not None else 1
        content = _format_result(
            command=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )
        if returncode != 0:
            return ToolResult.failure(
                f"command exited with status {returncode}",
                content=content,
                error_type="runtime_error",
            )

        return ToolResult.success(content)


@dataclass(frozen=True)
class CapturedOutput:
    text: str
    truncated: bool = False
    max_bytes: int = MAX_OUTPUT_BYTES


async def _read_limited(
    stream: asyncio.StreamReader,
    *,
    max_bytes: int,
) -> CapturedOutput:
    chunks: list[bytes] = []
    captured = 0
    truncated = False

    while True:
        chunk = await stream.read(_READ_CHUNK_BYTES)
        if not chunk:
            break

        remaining = max_bytes - captured
        if remaining > 0:
            chunks.append(chunk[:remaining])
            captured += min(len(chunk), remaining)
        if len(chunk) > remaining:
            truncated = True

    text = b"".join(chunks).decode("utf-8", errors="replace")
    return CapturedOutput(text=text, truncated=truncated, max_bytes=max_bytes)


async def _kill_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return

    if os.name == "nt":
        process.kill()
        return

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


async def _wait_for_process_exit(process: asyncio.subprocess.Process) -> None:
    try:
        await asyncio.wait_for(process.wait(), timeout=1.0)
    except (TimeoutError, ProcessLookupError):
        return


def _coerce_timeout(value: Any, *, default: int, maximum: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        return default
    if value <= 0:
        return default
    return min(value, maximum)


def _format_result(
    *,
    command: str,
    stdout: CapturedOutput,
    stderr: CapturedOutput,
    returncode: int | None = None,
    timeout_seconds: int | None = None,
) -> str:
    status = (
        f"timeout after {timeout_seconds}s"
        if timeout_seconds is not None
        else f"exit {returncode}"
    )
    sections = [
        f"$ {command}",
        f"[{status}]",
        _format_stream("stdout", stdout),
        _format_stream("stderr", stderr),
    ]
    return "\n".join(sections).rstrip()


def _format_stream(name: str, output: CapturedOutput) -> str:
    text = output.text.rstrip()
    if not text:
        text = "[empty]"
    if output.truncated:
        text = f"{text}\n[{name} truncated at {output.max_bytes} bytes]"
    return f"{name}:\n{text}"
