from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from my_claude.agent.events import AgentEvent
from my_claude.core.app import register_routes
from my_claude.core.bus.command import (
    EVENT_SUBSCRIBE_METHOD,
    SESSION_CLOSE_METHOD,
    SESSION_CREATE_METHOD,
    SESSION_GET_HISTORY_METHOD,
    SESSION_SEND_MESSAGE_METHOD,
)
from my_claude.core.config import AppConfig
from my_claude.core.context import ExecutionContext, ExecutionMode, SemanticMemoryKind
from my_claude.core.events.bus import EventBus
from my_claude.core.session.manager import (
    SessionBusyError,
    SessionManager,
)
from my_claude.core.session.store import (
    Session,
    SessionStore,
    SessionTurn,
)
from my_claude.core.tools.builtin.note_save import NoteSaveTool
from my_claude.core.transport.socket_client import SocketClient
from my_claude.core.transport.socket_server import TCPServer


def test_session_manager_create_initializes_memory_lock_disk_and_event(
    tmp_path: Path,
) -> None:
    session, session_file, events = asyncio.run(_create_session_with_manager(tmp_path))

    assert session.session_id.startswith("sess-")
    assert len(session.session_id.removeprefix("sess-")) == 12
    assert session.status == "active"
    assert session.run_ids == []
    assert session.active_run_id is None
    assert session.turns == []
    assert session_file.exists()

    payload = json.loads(session_file.read_text(encoding="utf-8"))
    assert payload["id"] == session.session_id
    assert payload["mode"] == "chat"
    assert payload["status"] == "active"
    assert payload["run_ids"] == []
    assert events == ["session.created"]


def test_session_create_handler_persists_indexes_and_publishes_event(
    tmp_path: Path,
) -> None:
    result, pushed_events, session_file = asyncio.run(_create_session_over_socket(tmp_path))

    assert result["session_id"].startswith("sess-")
    assert result["title"] == "stage s4"
    assert session_file.exists()

    payload = json.loads(session_file.read_text(encoding="utf-8"))
    assert payload["id"] == result["session_id"]
    assert payload["mode"] == "chat"
    assert payload["status"] == "active"
    assert payload["title"] == "stage s4"
    assert payload["run_ids"] == []

    assert pushed_events == [
        {
            "type": "session.created",
            "message": "session created",
            "session_id": result["session_id"],
            "mode": "chat",
            "status": "active",
            "title": "stage s4",
            "path": str(session_file),
        }
    ]


def test_kama_session_ipc_create_send_history_close(tmp_path: Path) -> None:
    result = asyncio.run(_kama_session_ipc_roundtrip(tmp_path))

    assert result["created"]["status"] == "active"
    assert result["send"]["run_id"]
    assert [message["role"] for message in result["history"]["messages"]] == [
        "user",
        "assistant",
    ]
    assert result["history"]["messages"][0]["content"] == "hello"
    assert result["closed"] == {"status": "closed"}


def test_session_message_writes_context_before_run_and_busy_fails_fast(
    tmp_path: Path,
) -> None:
    session_file = asyncio.run(_start_message_and_assert_busy_semantics(tmp_path))

    payload = json.loads(session_file.read_text(encoding="utf-8"))
    assert payload["active_run_id"] == "run-1"
    assert payload["run_ids"] == ["run-1"]
    thread_file = session_file.with_name("thread.jsonl")
    turns = [json.loads(line) for line in thread_file.read_text(encoding="utf-8").splitlines()]
    assert turns[0]["role"] == "user"
    assert turns[0]["content"] == "hello"
    assert turns[0]["run_id"] == "run-1"


def test_session_message_restores_execution_context_from_persistent_session(
    tmp_path: Path,
) -> None:
    execution_context = asyncio.run(_restore_session_and_build_execution_context(tmp_path))

    assert execution_context.mode == ExecutionMode.SESSION
    assert execution_context.session_id is not None
    assert execution_context.goal == "new question"
    assert [message.role for message in execution_context.episodic_messages] == [
        "user",
        "assistant",
        "user",
    ]
    assert [message.text for message in execution_context.episodic_messages] == [
        "old question",
        "old answer",
        "new question",
    ]
    assert [
        (item.kind, item.content)
        for item in execution_context.semantic_memory
    ] == [
        (SemanticMemoryKind.FACT, "repo uses uv"),
        (SemanticMemoryKind.DECISION, "busy sessions fail fast"),
    ]
    system_prompt_patch = execution_context.system_prompt_patch()
    assert system_prompt_patch is not None
    assert system_prompt_patch.startswith("## Session Notes")
    assert "fact: repo uses uv" in system_prompt_patch
    llm_messages = execution_context.llm_messages()
    assert llm_messages[0].text == "old question"
    assert llm_messages[-1].text == "new question"


def test_session_store_reads_full_history_without_sliding_window(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "runs")
    session = Session(session_id="sess-full-history")
    session.turns.extend(
        SessionTurn(
            role="user" if index % 2 == 0 else "assistant",
            content=f"message-{index}",
        )
        for index in range(30)
    )
    store.write_meta(session)

    messages = store.read_messages(session.session_id)

    assert [message.content for message in messages] == [
        f"message-{index}"
        for index in range(30)
    ]


def test_session_store_append_messages_preserves_existing_history(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "runs")
    session = Session(
        session_id="sess-append",
        turns=[SessionTurn(role="user", content="first")],
    )
    store.write_meta(session)

    path = store.append_messages(
        session,
        [
            SessionTurn(role="assistant", content="second"),
            SessionTurn(role="user", content="third"),
        ],
    )

    assert path == store.path_for(session.session_id)
    assert [message.content for message in store.read_messages(session.session_id)] == [
        "first",
        "second",
        "third",
    ]


def test_note_save_tool_writes_notes_md_for_future_session_context(
    tmp_path: Path,
) -> None:
    execution_context, notes_path = asyncio.run(_save_note_and_restore_context(tmp_path))

    assert notes_path.exists()
    assert "repo uses uv" in notes_path.read_text(encoding="utf-8")
    assert execution_context.semantic_memory[0].kind == SemanticMemoryKind.NOTE
    assert "## Note" in execution_context.semantic_memory[0].content
    assert "repo uses uv" in execution_context.semantic_memory[0].content


async def _create_session_with_manager(
    tmp_path: Path,
) -> tuple[Session, Path, list[str]]:
    bus: EventBus[AgentEvent] = EventBus()
    events: list[str] = []
    manager = SessionManager(tmp_path / "runs", bus)

    async def collect(event: AgentEvent) -> None:
        events.append(event.type.value)

    bus.subscribe(collect)
    outcome = await manager.create(title="direct")

    assert manager._sessions[outcome.session.session_id] is outcome.session
    assert outcome.session.session_id in manager._locks

    return outcome.session, outcome.path, events


async def _start_message_and_assert_busy_semantics(tmp_path: Path) -> Path:
    manager = SessionManager(tmp_path / "runs", EventBus())
    create_outcome = await manager.create(title="busy")
    session_id = create_outcome.session.session_id
    lock = manager._locks[session_id]

    await lock.acquire()
    try:
        with pytest.raises(SessionBusyError, match="session busy"):
            await manager.start_message(
                session_id=session_id,
                message="blocked",
                run_id="run-blocked",
            )
    finally:
        lock.release()

    message_outcome = await manager.start_message(
        session_id=session_id,
        message="hello",
        run_id="run-1",
    )
    with pytest.raises(SessionBusyError, match="active run run-1"):
        await manager.start_message(
            session_id=session_id,
            message="second",
            run_id="run-2",
        )

    return message_outcome.path


async def _restore_session_and_build_execution_context(tmp_path: Path) -> ExecutionContext:
    first_manager = SessionManager(tmp_path / "runs", EventBus())
    create_outcome = await first_manager.create(title="restore")
    session = create_outcome.session
    session.turns.extend(
        [
            SessionTurn(role="user", content="old question"),
            SessionTurn(role="assistant", content="old answer"),
        ]
    )
    first_manager._store.write_meta(session)
    first_manager._store.notes_path_for(session.session_id).write_text(
        "\n".join(
            [
                "- 2026-01-01T00:00:00+00:00 [fact] repo uses uv",
                "- 2026-01-01T00:00:01+00:00 [decision] busy sessions fail fast",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    restored_manager = SessionManager(tmp_path / "runs", EventBus())
    message_outcome = await restored_manager.start_message(
        session_id=session.session_id,
        message="new question",
        run_id="run-restored",
    )
    return message_outcome.execution_context


async def _save_note_and_restore_context(tmp_path: Path) -> tuple[ExecutionContext, Path]:
    manager = SessionManager(tmp_path / "runs", EventBus())
    create_outcome = await manager.create(title="notes")
    session_id = create_outcome.session.session_id
    notes_path = tmp_path / "runs" / "sessions" / session_id / "notes.md"
    tool = NoteSaveTool(session_id=session_id, notes_path=notes_path)

    result = await tool.run({"kind": "fact", "content": "repo uses uv"})
    assert not result.is_error

    message_outcome = await manager.start_message(
        session_id=session_id,
        message="what should you remember?",
        run_id="run-with-note",
    )
    return message_outcome.execution_context, notes_path


async def _create_session_over_socket(
    tmp_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    config = AppConfig(runs_dir=tmp_path / "runs", llm_provider="local")
    server = TCPServer("127.0.0.1", 0, max_request_bytes=config.max_request_bytes)
    register_routes(server, config=config, server_version="test-version")
    pushed_events: list[dict[str, Any]] = []

    async def collect_event(event: dict[str, Any]) -> None:
        pushed_events.append(event)

    await server.start()
    try:
        async with SocketClient(
            "127.0.0.1",
            _bound_port(server),
            timeout_seconds=1.0,
        ) as client:
            client.on_event(collect_event)
            await client.request(
                EVENT_SUBSCRIBE_METHOD,
                {"topics": ["session.*"], "scope": "global"},
            )
            result = await client.request(SESSION_CREATE_METHOD, {"title": "stage s4"})
            session_id = str(result["session_id"])
            session_file = config.runs_dir / "sessions" / session_id / "meta.json"
            await _wait_until(lambda: len(pushed_events) == 1)
            return result, pushed_events, session_file
    finally:
        await server.shutdown()


async def _kama_session_ipc_roundtrip(tmp_path: Path) -> dict[str, Any]:
    config = AppConfig(runs_dir=tmp_path / "runs", llm_provider="local")
    server = TCPServer("127.0.0.1", 0, max_request_bytes=config.max_request_bytes)
    register_routes(server, config=config, server_version="test-version")

    await server.start()
    try:
        async with SocketClient(
            "127.0.0.1",
            _bound_port(server),
            timeout_seconds=1.0,
        ) as client:
            created = await client.request(
                SESSION_CREATE_METHOD,
                {"mode": "chat", "title": "ipc test"},
            )
            session_id = str(created["session_id"])
            send = await client.request(
                SESSION_SEND_MESSAGE_METHOD,
                {"session_id": session_id, "content": "hello"},
            )
            history = await client.request(
                SESSION_GET_HISTORY_METHOD,
                {"session_id": session_id},
            )
            closed = await client.request(
                SESSION_CLOSE_METHOD,
                {"session_id": session_id},
            )
            return {
                "created": created,
                "send": send,
                "history": history,
                "closed": closed,
            }
    finally:
        await server.shutdown()


def _bound_port(server: TCPServer) -> int:
    if server._server is None or not server._server.sockets:
        raise RuntimeError("test server did not start")

    return int(server._server.sockets[0].getsockname()[1])


async def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout_seconds: float = 1.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("condition was not met before timeout")
        await asyncio.sleep(0)
