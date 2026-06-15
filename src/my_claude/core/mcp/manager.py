"""Manage external MCP servers and expose their tools as internal tools."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

from my_claude.core.config import McpServerConfig
from my_claude.core.tools.base import ParamsModel, ToolDefinition, ToolResult
from my_claude.core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class McpServerUnavailableError(RuntimeError):
    """Raised when an MCP server cannot service a request."""


class McpToolError(RuntimeError):
    """Raised when an MCP server returns an application-level tool error."""


class McpClient(Protocol):
    """Minimal MCP client protocol used by the manager and tool wrapper."""

    async def list_tools(self) -> list[McpToolDefinition]: ...

    async def call_tool(self, name: str, params: dict[str, Any]) -> str: ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class McpToolDefinition:
    """Tool definition discovered from an MCP server."""

    name: str
    description: str | None = None
    input_schema: dict[str, Any] | None = None


McpToolDef = McpToolDefinition


@dataclass(frozen=True)
class McpTool:
    """Internal tool wrapper around a remote MCP tool."""

    client: McpClient
    server_name: str
    tool_def: McpToolDefinition
    params_model: ParamsModel | None = None

    @property
    def name(self) -> str:
        return f"{self.server_name}__{self.tool_def.name}"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.tool_def.description or f"MCP tool from {self.server_name}",
            input_schema=self.tool_def.input_schema
            or {"type": "object", "properties": {}},
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        return await self.invoke(arguments)

    async def invoke(self, params: dict[str, Any]) -> ToolResult:
        try:
            content = await self.client.call_tool(self.tool_def.name, dict(params))
        except McpServerUnavailableError as error:
            return ToolResult(
                content=f"mcp server '{self.server_name}' unavailable: {error}",
                is_error=True,
                error=str(error),
                error_type="runtime_error",
            )
        except McpToolError as error:
            return ToolResult(
                content=f"mcp tool '{self.name}' error: {error}",
                is_error=True,
                error=str(error),
                error_type="runtime_error",
            )
        except Exception as error:
            return ToolResult(
                content=f"mcp tool '{self.name}' unexpected error: {error}",
                is_error=True,
                error=str(error),
                error_type="runtime_error",
            )
        return ToolResult(content=content)


class McpServerManager:
    """Connect to configured MCP servers and cache discovered tools."""

    def __init__(self) -> None:
        self._clients: dict[str, McpClient] = {}
        self._tools: list[McpTool] = []

    async def start_all(self, configs: list[McpServerConfig]) -> None:
        for cfg in configs:
            try:
                client = await self._connect(cfg)
                tool_defs = await client.list_tools()
            except Exception:
                logger.exception("failed to start MCP server: %s", cfg.name)
                continue
            self._clients[cfg.name] = client
            self._tools.extend(
                McpTool(client=client, server_name=cfg.name, tool_def=tool_def)
                for tool_def in tool_defs
            )

    def get_tools(self) -> list[McpTool]:
        return list(self._tools)

    def register_tools(self, registry: ToolRegistry) -> None:
        for tool in self._tools:
            registry.register(tool)

    async def shutdown(self) -> None:
        await self.stop_all()

    async def stop_all(self) -> None:
        clients = list(self._clients.values())
        self._clients.clear()
        self._tools.clear()
        for client in clients:
            try:
                await client.close()
            except Exception:
                logger.exception("failed to close MCP client")

    async def _connect(self, cfg: McpServerConfig) -> McpClient:
        if cfg.transport == "stdio":
            if not cfg.command:
                raise ValueError(f"mcp stdio server missing command: {cfg.name}")
            return await StdioMcpClient.start(cfg.command, cfg.args, cfg.env or None)
        if cfg.transport == "tcp":
            if not cfg.host or cfg.port is None:
                raise ValueError(f"mcp tcp server missing host/port: {cfg.name}")
            return await TcpMcpClient.connect(cfg.host, cfg.port)
        raise ValueError(f"unsupported mcp transport: {cfg.transport}")


class JsonRpcMcpClient:
    """Small JSON-RPC-style MCP client used for stdio and tcp transports."""

    _STREAM_LIMIT = 64 * 1024 * 1024

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        process: asyncio.subprocess.Process | None = None,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._process = process
        self._next_id = 1
        self._lock = asyncio.Lock()
        self._stderr_task: asyncio.Task[None] | None = None

    async def initialize(self) -> None:
        await self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "myclaude", "version": "0.1"},
            },
        )
        await self._notify("notifications/initialized", {})

    async def list_tools(self) -> list[McpToolDefinition]:
        payload = await self._request("tools/list", {})
        raw_tools = payload.get("tools", []) if isinstance(payload, dict) else payload
        if not isinstance(raw_tools, list):
            raw_tools = []
        if not isinstance(raw_tools, list):
            return []
        return [_tool_definition_from_payload(tool) for tool in raw_tools]

    async def call_tool(self, name: str, params: dict[str, Any]) -> str:
        payload = await self._request(
            "tools/call",
            {"name": name, "arguments": params},
        )
        return _content_from_payload(payload)

    async def close(self) -> None:
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
            self._stderr_task = None
        self._writer.close()
        try:
            await self._writer.wait_closed()
        except Exception:
            pass
        if self._process is not None and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=2.0)
            except TimeoutError:
                self._process.kill()
                await self._process.wait()

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        async with self._lock:
            request_id = self._next_id
            request_id_str = str(request_id)
            self._next_id += 1
            self._writer.write(
                (
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "method": method,
                            "params": params,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                ).encode("utf-8")
            )
            await self._writer.drain()
            while True:
                try:
                    line = await asyncio.wait_for(self._reader.readline(), timeout=30.0)
                except TimeoutError as error:
                    raise McpServerUnavailableError("MCP server read timeout") from error
                except asyncio.LimitOverrunError as error:
                    raise McpServerUnavailableError(
                        f"MCP response too large (>{self._STREAM_LIMIT // 1024 // 1024}MB): {error}"
                    ) from error
                if not line:
                    raise McpServerUnavailableError("mcp server closed connection")
                try:
                    response = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    logger.debug("mcp: ignoring non-JSON line: %r", line[:200])
                    continue
                response_id = response.get("id")
                if response_id is None:
                    logger.debug("mcp: received server notification: %s", response.get("method"))
                    continue
                if str(response_id) != request_id_str:
                    continue
                if response.get("error") is not None:
                    error = response["error"]
                    if isinstance(error, dict):
                        raise McpToolError(
                            f"{error.get('message', str(error))} (code={error.get('code')})"
                        )
                    raise McpToolError(str(error))
                return response.get("result")

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._writer.write(
            (
                json.dumps(
                    {"jsonrpc": "2.0", "method": method, "params": params},
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8")
        )
        await self._writer.drain()


class StdioMcpClient(JsonRpcMcpClient):
    """MCP client backed by a stdio subprocess."""

    @classmethod
    async def start(
        cls,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
    ) -> StdioMcpClient:
        process = await asyncio.create_subprocess_exec(
            command,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **(env or {})},
            limit=cls._STREAM_LIMIT,
        )
        if process.stdin is None or process.stdout is None:
            raise McpServerUnavailableError("mcp stdio pipes unavailable")
        client = cls(process.stdout, process.stdin, process=process)
        client._stderr_task = asyncio.create_task(client._drain_stderr())
        await client.initialize()
        return client

    async def _drain_stderr(self) -> None:
        if self._process is None or self._process.stderr is None:
            return
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    return
                decoded = line.decode(errors="replace").rstrip()
                if decoded:
                    logger.debug("mcp stderr: %s", decoded)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug("mcp stderr drain stopped", exc_info=True)


class TcpMcpClient(JsonRpcMcpClient):
    """MCP client backed by a TCP stream."""

    @classmethod
    async def connect(cls, host: str, port: int) -> TcpMcpClient:
        reader, writer = await asyncio.open_connection(
            host,
            port,
            limit=cls._STREAM_LIMIT,
        )
        client = cls(reader, writer)
        await client.initialize()
        return client


def _tool_definition_from_payload(payload: object) -> McpToolDefinition:
    if not isinstance(payload, dict):
        return McpToolDefinition(name="unknown")
    name = str(payload.get("name", "unknown"))
    description = payload.get("description")
    input_schema = payload.get("input_schema", payload.get("inputSchema"))
    return McpToolDefinition(
        name=name,
        description=description if isinstance(description, str) else None,
        input_schema=input_schema if isinstance(input_schema, dict) else None,
    )


def _content_from_payload(payload: object) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        content = payload.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(_content_from_payload(item) for item in content)
        text = payload.get("text")
        if isinstance(text, str):
            return text
    return json.dumps(payload, ensure_ascii=False)
