"""Daemon-side chat session manager."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from my_claude.agent.events import (
    AgentEvent,
    SessionClosedEvent,
    SessionCreatedEvent,
    SessionMessageReceivedEvent,
    SessionResumedEvent,
    SessionWaitingForInputEvent,
)
from my_claude.core.context import (
    AnthropicMessage,
    ExecutionContext,
    ExecutionMode,
    SemanticMemoryItem,
)
from my_claude.core.events.bus import EventBus
from my_claude.core.runner import RunResult
from my_claude.core.session.store import (
    Session,
    SessionMode,
    SessionNote,
    SessionStore,
    SessionTurn,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class SessionCreateOutcome:
    """Metadata returned after a session is created."""

    session: Session
    path: Path


@dataclass(frozen=True)
class SessionMessageOutcome:
    """Metadata returned after a user message is accepted."""

    session: Session
    turn: int
    prefill_messages: int
    execution_context: ExecutionContext
    path: Path


class SessionBusyError(ValueError):
    """Raised when a session is already handling another operation."""


class SessionManager:
    """Maintain daemon-side chat sessions in memory, on disk, and on the event bus."""

    def __init__(self, runs_dir: Path, bus: EventBus[AgentEvent]) -> None:
        self._store = SessionStore(runs_dir)
        self._bus = bus
        self._sessions: dict[str, Session] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def create(
        self,
        *,
        mode: SessionMode = "chat",
        title: str | None = None,
    ) -> SessionCreateOutcome:
        sid = f"sess-{uuid.uuid4().hex[:12]}"
        normalized_title = title.strip() if title is not None else None
        session = Session(session_id=sid, mode=mode, title=normalized_title)

        self._sessions[sid] = session
        self._locks[sid] = asyncio.Lock()

        path = self._store.write_meta(session)
        await self._bus.publish(
            SessionCreatedEvent(
                session_id=session.session_id,
                mode=session.mode,
                status=session.status,
                title=session.title,
                path=str(path),
            )
        )
        return SessionCreateOutcome(session=session, path=path)

    async def start_message(
        self,
        *,
        session_id: str,
        message: str,
        run_id: str,
    ) -> SessionMessageOutcome:
        lock = self._require_lock(session_id)
        if lock.locked():
            raise SessionBusyError(f"session busy: {session_id}")

        await lock.acquire()
        try:
            session = self._require(session_id)
            if session.active_run_id is not None:
                raise SessionBusyError(
                    f"session busy: {session_id} active run {session.active_run_id}"
                )
            if session.status == "closed":
                raise ValueError(f"session already closed: {session_id}")

            normalized_message = message.strip()
            if not normalized_message:
                raise ValueError("session.message message must not be empty")

            if session.status == "waiting_for_input":
                await self._bus.publish(SessionResumedEvent(session_id=session.session_id))

            turn_number = session.next_turn
            user_turn = SessionTurn(
                role="user",
                content=normalized_message,
                run_id=run_id,
            )
            session.run_ids.append(run_id)
            session.active_run_id = run_id
            session.status = "active"
            if not session.title:
                session.title = normalized_message[:40]
            session.updated_at = _now()
            path = self._store.append_messages(session, [user_turn])
            await self._bus.publish(
                SessionMessageReceivedEvent(
                    session_id=session.session_id,
                    content=normalized_message,
                )
            )
            full_timeline = self._store.read_messages(session.session_id)
            notes = [*session.notes, *self._store.read_notes(session.session_id)]
            execution_context = _build_execution_context(
                session,
                goal=normalized_message,
                run_id=run_id,
                turns=full_timeline,
                notes=notes,
            )
            return SessionMessageOutcome(
                session=session,
                turn=turn_number,
                prefill_messages=len(full_timeline),
                execution_context=execution_context,
                path=path,
            )
        finally:
            lock.release()

    async def finish_message(
        self,
        *,
        session_id: str,
        task: asyncio.Task[RunResult],
        prefill_messages: int,
    ) -> None:
        lock = self._locks.get(session_id)
        if lock is None:
            return

        async with lock:
            session = self._sessions.get(session_id)
            if session is None:
                return

            try:
                result = task.result()
            except asyncio.CancelledError:
                session.active_run_id = None
                session.updated_at = _now()
                self._store.write_meta(session)
                return
            except Exception:
                session.active_run_id = None
                session.updated_at = _now()
                self._store.write_meta(session)
                return

            new_turns = [
                _message_to_turn(message, run_id=result.run_id)
                for message in result.messages[prefill_messages:]
            ]
            session.active_run_id = None
            if session.mode == "one_shot":
                session.status = "closed"
            else:
                session.status = "waiting_for_input"
            session.updated_at = _now()
            self._store.append_messages(session, new_turns)
            if session.status == "closed":
                await self._bus.publish(SessionClosedEvent(session_id=session.session_id))
            else:
                await self._bus.publish(
                    SessionWaitingForInputEvent(
                        session_id=session.session_id,
                        last_run_id=result.run_id,
                    )
                )

    def read_messages(self, session_id: str) -> list[SessionTurn]:
        """Return the complete persisted timeline for a session."""

        return self._store.read_messages(session_id)

    async def get_history(self, session_id: str) -> list[dict[str, Any]]:
        """Return the complete Anthropic-compatible message history."""

        self._require(session_id)
        return [
            _turn_to_anthropic_dict(turn)
            for turn in self._store.read_messages(session_id)
        ]

    async def close(self, session_id: str) -> None:
        """Close a session and publish the Kama-compatible closed event."""

        lock = self._require_lock(session_id)
        if lock.locked():
            raise SessionBusyError(f"session busy: {session_id}")

        await lock.acquire()
        try:
            session = self._require(session_id)
            session.status = "closed"
            session.active_run_id = None
            session.updated_at = _now()
            self._store.write_meta(session)
            await self._bus.publish(SessionClosedEvent(session_id=session.session_id))
        finally:
            lock.release()

    async def append_messages(
        self,
        session_id: str,
        messages: list[SessionTurn],
    ) -> None:
        """Append exactly the supplied messages without truncating prior history."""

        lock = self._require_lock(session_id)
        if lock.locked():
            raise SessionBusyError(f"session busy: {session_id}")

        await lock.acquire()
        try:
            session = self._require(session_id)
            self._store.append_messages(session, messages)
        finally:
            lock.release()

    def _require(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is not None:
            return session

        session = self._store.read_meta(session_id)
        if session is None:
            raise ValueError(f"unknown session: {session_id}")

        self._sessions[session_id] = session
        self._locks.setdefault(session_id, asyncio.Lock())
        return session

    def _require_lock(self, session_id: str) -> asyncio.Lock:
        lock = self._locks.get(session_id)
        if lock is not None:
            return lock

        session = self._store.read_meta(session_id)
        if session is None:
            raise ValueError(f"unknown session: {session_id}")

        self._sessions[session_id] = session
        lock = asyncio.Lock()
        self._locks[session_id] = lock
        return lock


def _build_execution_context(
    session: Session,
    *,
    goal: str,
    run_id: str,
    turns: list[SessionTurn] | None = None,
    notes: list[SessionNote] | None = None,
) -> ExecutionContext:
    return ExecutionContext(
        mode=ExecutionMode.SESSION,
        run_id=run_id,
        goal=goal,
        session_id=session.session_id,
        episodic_messages=[
            _turn_to_message(turn)
            for turn in (turns if turns is not None else session.turns)
        ],
        semantic_memory=[
            SemanticMemoryItem(kind=note.kind, content=note.content)
            for note in (notes if notes is not None else session.notes)
        ],
    )


def _turn_to_message(turn: SessionTurn) -> AnthropicMessage:
    if isinstance(turn.content, list):
        return AnthropicMessage.model_validate(
            {"role": turn.role, "content": turn.content}
        )
    if turn.role == "assistant":
        return AnthropicMessage.assistant_text(turn.content)
    return AnthropicMessage.user_text(turn.content)


def _message_to_turn(message: AnthropicMessage, *, run_id: str) -> SessionTurn:
    return SessionTurn(
        role=message.role,
        content=[
            block.model_dump(mode="json", exclude_none=True)
            for block in message.content
        ],
        run_id=run_id,
    )


def _turn_to_anthropic_dict(turn: SessionTurn) -> dict[str, Any]:
    return {"role": turn.role, "content": turn.content}
