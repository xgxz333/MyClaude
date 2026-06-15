"""Durable session metadata and full conversation history storage."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from my_claude.core.context import SemanticMemoryKind

TOOL_RESULT_LIMIT = 8_000
TOOL_RESULT_KEEP = 4_000


def _now() -> str:
    return datetime.now(UTC).isoformat()


MessageContent = str | list[dict[str, Any]]
SessionMode = Literal["one_shot", "chat"]
SessionStatus = Literal["active", "waiting_for_input", "closed"]


class SessionTurn(BaseModel):
    """One persisted user or assistant turn in a chat session."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: MessageContent
    run_id: str | None = None
    created_at: str = Field(default_factory=_now)


class SessionNote(BaseModel):
    """One durable fact, decision, or note remembered by a session."""

    model_config = ConfigDict(extra="forbid")

    kind: SemanticMemoryKind = SemanticMemoryKind.NOTE
    content: str
    created_at: str = Field(default_factory=_now)


class Session(BaseModel):
    """Persisted daemon-side chat session state."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    mode: SessionMode = "chat"
    status: SessionStatus = "active"
    title: str | None = None
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    run_ids: list[str] = Field(default_factory=list)
    active_run_id: str | None = None
    turns: list[SessionTurn] = Field(default_factory=list)
    notes: list[SessionNote] = Field(default_factory=list)

    @property
    def next_turn(self) -> int:
        return len([turn for turn in self.turns if turn.role == "user"]) + 1

    @property
    def id(self) -> str:
        """Kama-compatible session identifier alias."""

        return self.session_id


class SessionStore:
    """Persist session metadata under the daemon runs directory."""

    def __init__(self, runs_dir: Path) -> None:
        self._root = runs_dir / "sessions"

    def session_dir(self, session_id: str) -> Path:
        return self._root / session_id

    def runs_dir(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "runs"

    def path_for(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "meta.json"

    def legacy_path_for(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "session.json"

    def thread_path_for(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "thread.jsonl"

    def notes_path_for(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "notes.md"

    def write_meta(self, session: Session) -> Path:
        path = self.path_for(session.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                _session_meta_dict(session),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if session.turns:
            self._write_thread(session.session_id, session.turns)
        return path

    def read_meta(self, session_id: str) -> Session | None:
        path = self.path_for(session_id)
        if not path.exists():
            path = self.legacy_path_for(session_id)
            if not path.exists():
                return None

        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"session meta must be an object: {path}")
        session = _session_from_meta_dict(data)
        session.turns = self.read_messages(session.session_id)
        return session

    def read_messages(
        self,
        session_id: str,
        *,
        truncate_tools: bool = True,
    ) -> list[SessionTurn]:
        """Read and validate the full persisted conversation timeline."""

        thread_path = self.thread_path_for(session_id)
        if not thread_path.exists():
            legacy_path = self.legacy_path_for(session_id)
            if not legacy_path.exists() and not self.path_for(session_id).exists():
                raise ValueError(f"unknown session: {session_id}")
            if legacy_path.exists():
                data = json.loads(legacy_path.read_text(encoding="utf-8"))
                return [
                    SessionTurn.model_validate(turn)
                    for turn in data.get("turns", [])
                    if isinstance(turn, dict)
                ]
            return []

        turns: list[SessionTurn] = []
        thread_lines = thread_path.read_text(encoding="utf-8").splitlines()
        for line_number, raw_line in enumerate(thread_lines, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid thread row: {thread_path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"thread row must be an object: {thread_path}:{line_number}")
            role = row.get("role")
            if role not in {"user", "assistant"}:
                raise ValueError(f"invalid thread role: {thread_path}:{line_number}")
            turns.append(
                SessionTurn(
                    role=role,
                    content=row.get("content", ""),
                    run_id=row.get("run_id") if isinstance(row.get("run_id"), str) else None,
                    created_at=str(row.get("ts") or row.get("created_at") or _now()),
                )
            )
        messages = _trim_orphan_tool_use(turns)
        if not truncate_tools:
            return messages
        return truncate_tool_results(messages)

    def append_messages(
        self,
        session: Session,
        messages: list[SessionTurn],
    ) -> Path:
        """Append exactly the supplied messages without truncating prior history."""

        if messages:
            session.turns.extend(messages)
            session.updated_at = _now()
        return self.write_meta(session)

    def write_compacted(
        self,
        session_id: str,
        compacted_messages: list[SessionTurn],
    ) -> tuple[Path, Path]:
        """Backup and replace a session thread with compacted messages."""

        thread_path = self.thread_path_for(session_id)
        if not thread_path.exists():
            raise ValueError(f"session thread does not exist: {session_id}")

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backup_path = self.session_dir(session_id) / f"thread_{timestamp}.jsonl.bak"
        thread_path.rename(backup_path)
        self._write_thread(session_id, compacted_messages)
        return thread_path, backup_path

    def append_note(
        self,
        session_id: str,
        *,
        content: str,
        run_id: str = "manual",
        kind: SemanticMemoryKind = SemanticMemoryKind.NOTE,
    ) -> Path:
        del kind
        path = self.notes_path_for(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(f"## Note ({_now()}, {run_id})\n{content.strip()}\n\n")
        return path

    def read_notes(self, session_id: str) -> list[SessionNote]:
        path = self.notes_path_for(session_id)
        if not path.exists():
            return []

        raw_text = path.read_text(encoding="utf-8")
        if raw_text.lstrip().startswith("## Note"):
            return [SessionNote(content=raw_text.strip())]

        notes: list[SessionNote] = []
        for line_number, raw_line in enumerate(raw_text.splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                notes.append(_parse_note_line(line))
            except ValueError as error:
                raise ValueError(f"invalid note line: {path}:{line_number}") from error
        return notes

    def _write_thread(self, session_id: str, turns: list[SessionTurn]) -> None:
        path = self.thread_path_for(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            json.dumps(
                {
                    "ts": turn.created_at,
                    "role": turn.role,
                    "content": turn.content,
                    **({"run_id": turn.run_id} if turn.run_id is not None else {}),
                },
                ensure_ascii=False,
            )
            for turn in turns
        ]
        path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _parse_note_line(line: str) -> SessionNote:
    if not line.startswith("- "):
        raise ValueError("note line must be a markdown list item")

    body = line[2:].strip()
    marker_start = body.find("[")
    marker_end = body.find("]", marker_start + 1)
    if marker_start == -1 or marker_end == -1:
        return SessionNote(content=body)

    kind_value = body[marker_start + 1:marker_end]
    content = body[marker_end + 1:].strip()
    if not content:
        raise ValueError("note content must not be empty")

    try:
        kind = SemanticMemoryKind(kind_value)
    except ValueError:
        kind = SemanticMemoryKind.NOTE

    return SessionNote(kind=kind, content=content)


def _session_meta_dict(session: Session) -> dict[str, Any]:
    return {
        "id": session.session_id,
        "mode": session.mode,
        "status": session.status,
        "title": session.title or "",
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "run_ids": list(session.run_ids),
        **({"active_run_id": session.active_run_id} if session.active_run_id else {}),
    }


def _session_from_meta_dict(data: dict[str, Any]) -> Session:
    session_id = str(data.get("id") or data.get("session_id"))
    return Session(
        session_id=session_id,
        mode=data.get("mode", "chat"),
        status=data.get("status", "active"),
        title=str(data.get("title") or "") or None,
        created_at=str(data.get("created_at") or _now()),
        updated_at=str(data.get("updated_at") or _now()),
        run_ids=[str(run_id) for run_id in data.get("run_ids", [])],
        active_run_id=(
            str(data["active_run_id"])
            if isinstance(data.get("active_run_id"), str)
            else None
        ),
        turns=[
            SessionTurn.model_validate(turn)
            for turn in data.get("turns", [])
            if isinstance(turn, dict)
        ],
        notes=[
            SessionNote.model_validate(note)
            for note in data.get("notes", [])
            if isinstance(note, dict)
        ],
    )


def truncate_tool_results(
    messages: list[Any],
    limit: int = TOOL_RESULT_LIMIT,
    keep: int = TOOL_RESULT_KEEP,
) -> list[Any]:
    """Return copies with oversized tool_result text shortened for LLM input only."""

    result: list[Any] = []
    for message in messages:
        content = (
            message.content
            if isinstance(message, SessionTurn)
            else message.get("content")
            if isinstance(message, dict)
            else None
        )
        if not isinstance(content, list):
            result.append(message)
            continue

        new_blocks, changed = _truncate_tool_result_blocks(
            content,
            limit=limit,
            keep=keep,
        )

        if not changed:
            result.append(message)
        elif isinstance(message, SessionTurn):
            result.append(message.model_copy(update={"content": new_blocks}))
        elif isinstance(message, dict):
            result.append({**message, "content": new_blocks})
        else:
            result.append(message)

    return result


def _truncate_tool_result_blocks(
    blocks: list[dict[str, Any]],
    *,
    limit: int,
    keep: int,
) -> tuple[list[dict[str, Any]], bool]:
    new_blocks: list[dict[str, Any]] = []
    changed = False
    for block in blocks:
        new_block = block
        if (
            block.get("type") == "tool_result"
            and isinstance(block.get("content"), str)
        ):
            text = block["content"]
            if len(text) > limit:
                omitted = len(text) - keep
                new_block = dict(block)
                new_block["content"] = (
                    text[:keep]
                    + f"\n\n[... {omitted} chars omitted. Full output in run events.]"
                )
                changed = True
        new_blocks.append(new_block)
    return new_blocks, changed


def _trim_orphan_tool_use(turns: list[SessionTurn]) -> list[SessionTurn]:
    pending: set[str] = set()
    last_balanced = 0
    for index, turn in enumerate(turns, 1):
        content = turn.content
        if isinstance(content, list):
            if turn.role == "assistant":
                for block in content:
                    if block.get("type") == "tool_use":
                        pending.add(str(block.get("id", "")))
            elif turn.role == "user":
                for block in content:
                    if block.get("type") == "tool_result":
                        pending.discard(str(block.get("tool_use_id", "")))
        if not pending:
            last_balanced = index
    if pending:
        return turns[:last_balanced]
    return turns
