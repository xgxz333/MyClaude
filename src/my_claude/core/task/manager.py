"""File-backed task notebook for one agent run."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from my_claude.core.task.model import Task, TaskStatus


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TaskManager:
    """Synchronous file-backed CRUD for run-local task records."""

    def __init__(self, tasks_dir: Path) -> None:
        self._dir = tasks_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._next_id = self._max_id() + 1

    @property
    def tasks_dir(self) -> Path:
        return self._dir

    def create(
        self,
        subject: str,
        description: str = "",
        blocked_by: list[int] | None = None,
    ) -> Task:
        subject = subject.strip()
        if not subject:
            raise ValueError("task subject must not be empty")

        active_blockers = self._active_dependency_ids(blocked_by or [])

        now = _now()
        task = Task(
            id=self._next_id,
            subject=subject,
            description=description,
            status="pending",
            blocked_by=active_blockers,
            created_at=now,
            updated_at=now,
        )
        self._save(task)
        self._next_id += 1
        return task

    def get(self, task_id: int) -> Task:
        return self._load(task_id)

    def update(
        self,
        task_id: int,
        *,
        status: TaskStatus | None = None,
        add_blocked_by: list[int] | None = None,
        remove_blocked_by: list[int] | None = None,
    ) -> Task:
        task = self._load(task_id)
        if status is not None:
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"invalid status: {status!r}")
            task.status = status

        if add_blocked_by and task.status != "completed":
            active_blockers = self._active_dependency_ids(add_blocked_by, task_id=task_id)
            task.blocked_by = sorted(set(task.blocked_by + active_blockers))

        if remove_blocked_by:
            remove_set = set(remove_blocked_by)
            task.blocked_by = [item for item in task.blocked_by if item not in remove_set]

        if task.status == "completed":
            task.blocked_by = []
            self._cascade_completed_dependency(task_id)

        task.updated_at = _now()
        self._save(task)
        return task

    def list_all(self) -> list[Task]:
        tasks: list[Task] = []
        for path in sorted(self._dir.glob("task_*.json"), key=_task_file_sort_key):
            try:
                tasks.append(Task.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return tasks

    def format_list(self) -> str:
        tasks = self.list_all()
        if not tasks:
            return "No tasks."

        marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
        lines: list[str] = []
        for task in tasks:
            blocked = f" (blocked by: {task.blocked_by})" if task.blocked_by else ""
            lines.append(
                f"{marker.get(task.status, '[?]')} #{task.id}: {task.subject}{blocked}"
            )
        return "\n".join(lines)

    def _max_id(self) -> int:
        ids = [_task_file_sort_key(path) for path in self._dir.glob("task_*.json")]
        return max(ids) if ids else 0

    def _load(self, task_id: int) -> Task:
        path = self._task_path(task_id)
        if not path.exists():
            raise ValueError(f"task {task_id} not found")
        return Task.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _save(self, task: Task) -> None:
        self._task_path(task.id).write_text(
            json.dumps(task.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _cascade_completed_dependency(self, completed_id: int) -> None:
        for task in self.list_all():
            if completed_id not in task.blocked_by:
                continue
            task.blocked_by = [item for item in task.blocked_by if item != completed_id]
            task.updated_at = _now()
            self._save(task)

    def _active_dependency_ids(
        self,
        task_ids: list[int],
        *,
        task_id: int | None = None,
    ) -> list[int]:
        active_ids: list[int] = []
        for dependency_id in task_ids:
            if task_id is not None and dependency_id == task_id:
                raise ValueError("task cannot block itself")

            dependency = self._load(dependency_id)
            if dependency.status == "completed":
                continue
            active_ids.append(dependency_id)

        return sorted(set(active_ids))

    def _task_path(self, task_id: int) -> Path:
        return self._dir / f"task_{task_id}.json"


def _task_file_sort_key(path: Path) -> int:
    parts = path.stem.split("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return 0
    return int(parts[1])
