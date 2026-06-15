from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from my_claude.core.config import AppConfig, McpServerConfig, load_config
from my_claude.core.mcp import (
    McpServerManager,
    McpServerUnavailableError,
    McpTool,
    McpToolDefinition,
)
from my_claude.core.runner import prepare_run_context


class FakeMcpClient:
    def __init__(
        self,
        tools: list[McpToolDefinition] | None = None,
        *,
        unavailable: bool = False,
    ) -> None:
        self.tools = tools or []
        self.unavailable = unavailable
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    async def list_tools(self) -> list[McpToolDefinition]:
        return self.tools

    async def call_tool(self, name: str, params: dict[str, Any]) -> str:
        if self.unavailable:
            raise McpServerUnavailableError("down")
        self.calls.append((name, params))
        return f"{name}:{params['value']}"

    async def close(self) -> None:
        self.closed = True


class FakeMcpManager(McpServerManager):
    def __init__(self, client: FakeMcpClient) -> None:
        super().__init__()
        self.client = client

    async def _connect(self, _cfg: McpServerConfig) -> FakeMcpClient:
        return self.client


def test_load_config_reads_mcp_servers(tmp_path: Path) -> None:
    config_file = tmp_path / "myclaude.toml"
    config_file.write_text(
        "\n".join(
            [
                "[[mcp.servers]]",
                'name = "filesystem"',
                'transport = "stdio"',
                'command = "mcp-fs"',
                'args = ["--root", "."]',
                "",
                "[[mcp.servers]]",
                'name = "internal-kb"',
                'transport = "tcp"',
                'host = "10.0.0.5"',
                "port = 3000",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_file=config_file, env_file=tmp_path / ".env")

    assert len(config.mcp_servers) == 2
    assert config.mcp_servers[0].name == "filesystem"
    assert config.mcp_servers[0].transport == "stdio"
    assert config.mcp_servers[0].args == ["--root", "."]
    assert config.mcp_servers[1].name == "internal-kb"
    assert config.mcp_servers[1].transport == "tcp"
    assert config.mcp_servers[1].host == "10.0.0.5"
    assert config.mcp_servers[1].port == 3000


def test_mcp_tool_wraps_definition_and_calls_remote_tool() -> None:
    client = FakeMcpClient()
    tool = McpTool(
        client=client,
        server_name="filesystem",
        tool_def=McpToolDefinition(
            name="read_file",
            description="Read a file.",
            input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
        ),
    )

    result = asyncio.run(tool.run({"value": "README.md"}))

    assert tool.name == "filesystem__read_file"
    assert tool.definition.name == "filesystem__read_file"
    assert tool.definition.description == "Read a file."
    assert result.content == "read_file:README.md"
    assert not result.is_error
    assert client.calls == [("read_file", {"value": "README.md"})]


def test_mcp_tool_returns_error_when_server_unavailable() -> None:
    tool = McpTool(
        client=FakeMcpClient(unavailable=True),
        server_name="filesystem",
        tool_def=McpToolDefinition(name="read_file"),
    )

    result = asyncio.run(tool.run({"value": "README.md"}))

    assert result.is_error
    assert result.error_type == "runtime_error"
    assert result.content == "mcp server 'filesystem' unavailable: down"


def test_mcp_manager_discovers_tools_and_shutdown_closes_clients() -> None:
    client = FakeMcpClient([McpToolDefinition(name="search")])
    manager = FakeMcpManager(client)

    asyncio.run(
        manager.start_all(
            [McpServerConfig(name="internal-kb", transport="tcp", host="127.0.0.1", port=1)]
        )
    )

    assert [tool.name for tool in manager.get_tools()] == ["internal-kb__search"]
    asyncio.run(manager.shutdown())
    assert client.closed is True


def test_prepare_run_context_registers_mcp_tools_with_whitelist(tmp_path: Path) -> None:
    manager = McpServerManager()
    manager._tools = [
        McpTool(
            client=FakeMcpClient(),
            server_name="filesystem",
            tool_def=McpToolDefinition(name="read_file"),
        ),
        McpTool(
            client=FakeMcpClient(),
            server_name="internal-kb",
            tool_def=McpToolDefinition(name="search"),
        ),
    ]

    context = prepare_run_context(
        "use mcp",
        config=AppConfig(runs_dir=tmp_path / "runs"),
        tool_whitelist=["filesystem__read_file"],
        mcp_manager=manager,
    )

    assert [definition.name for definition in context.tools.definitions()] == [
        "filesystem__read_file"
    ]
