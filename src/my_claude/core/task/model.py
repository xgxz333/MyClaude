"""Persistent task records used by the agent task notebook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

TaskStatus = Literal["pending", "in_progress", "completed"]


@dataclass
class Task:
    """One planned work item in a run-local task notebook."""

    id: int
    subject: str
    description: str
    status: TaskStatus
    blocked_by: list[int]
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "description": self.description,
            "status": self.status,
            "blocked_by": self.blocked_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        status = data.get("status", "pending")
        if status not in ("pending", "in_progress", "completed"):
            status = "pending"

        return cls(
            id=int(data["id"]),
            subject=str(data["subject"]),
            description=str(data.get("description", "")),
            status=cast(TaskStatus, status),
            blocked_by=[int(item) for item in data.get("blocked_by", [])],
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
        )
