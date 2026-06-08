"""Persist agent events to disk as flushed JSON Lines."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from my_claude.agent.events import AgentEvent

SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"\b([A-Z0-9_]*(?:API_KEY|TOKEN|SECRET)[A-Z0-9_]*=)[^\s\\\"]+",
    re.IGNORECASE,
)
BEARER_TOKEN_PATTERN = re.compile(r"\b(Bearer\s+)[^\s\\\"]+", re.IGNORECASE)
SK_TOKEN_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")


class JsonlEventWriter:
    """Append events to a JSONL file and flush after every written line."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._file: TextIO | None = None

    def __enter__(self) -> JsonlEventWriter:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    async def handle(self, event: AgentEvent) -> None:
        if self._file is None:
            raise RuntimeError("event file is not open")

        self._file.write(serialize_event(event) + "\n")
        self._file.flush()


def serialize_event(event: AgentEvent) -> str:
    line = {
        "timestamp": datetime.now(UTC).isoformat(),
        "type": event.type.value,
        "message": event.message,
        "data": redact_secrets(dict(event.data)),
    }
    return json.dumps(line, ensure_ascii=False, separators=(",", ":"))


def redact_secrets(value: Any) -> Any:
    if isinstance(value, str):
        redacted = SECRET_ASSIGNMENT_PATTERN.sub(r"\1<redacted>", value)
        redacted = BEARER_TOKEN_PATTERN.sub(r"\1<redacted>", redacted)
        return SK_TOKEN_PATTERN.sub("sk-<redacted>", redacted)

    if isinstance(value, Mapping):
        return {key: redact_secrets(item) for key, item in value.items()}

    if isinstance(value, list):
        return [redact_secrets(item) for item in value]

    if isinstance(value, tuple):
        return [redact_secrets(item) for item in value]

    return value
