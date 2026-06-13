from __future__ import annotations

import json
from pathlib import Path

import pytest

from my_claude.core.task.manager import TaskManager


def test_task_manager_init_creates_directory_and_resumes_next_id(tmp_path: Path) -> None:
    tasks_dir = tmp_path / ".tasks"
    tasks_dir.mkdir()
    (tasks_dir / "task_7.json").write_text(
        json.dumps(
            {
                "id": 7,
                "subject": "Existing task",
                "description": "",
                "status": "pending",
                "blocked_by": [],
                "created_at": "2026-06-12T00:00:00+00:00",
                "updated_at": "2026-06-12T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    manager = TaskManager(tasks_dir)
    task = manager.create("Next task")

    assert tasks_dir.exists()
    assert task.id == 8
    assert (tasks_dir / "task_8.json").exists()


def test_task_manager_create_writes_task_json_file(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path / ".tasks")

    task = manager.create(
        " Plan work ",
        description="Break the goal down.",
    )

    assert task.id == 1
    assert task.subject == "Plan work"
    assert task.description == "Break the goal down."
    assert task.status == "pending"
    assert task.blocked_by == []
    assert task.created_at
    assert task.updated_at == task.created_at

    payload = json.loads((tmp_path / ".tasks" / "task_1.json").read_text())
    assert payload == task.to_dict()


def test_task_manager_create_rejects_empty_subject(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path / ".tasks")

    with pytest.raises(ValueError, match="subject"):
        manager.create("   ")


def test_task_manager_update_changes_status_and_dependencies(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path / ".tasks")
    dependency = manager.create("Dependency")
    task = manager.create("Blocked task", blocked_by=[dependency.id])

    updated = manager.update(
        task.id,
        status="in_progress",
        remove_blocked_by=[dependency.id],
    )

    assert updated.status == "in_progress"
    assert updated.blocked_by == []

    payload = json.loads((tmp_path / ".tasks" / "task_2.json").read_text())
    assert payload["status"] == "in_progress"
    assert payload["blocked_by"] == []
    assert payload["updated_at"] >= payload["created_at"]


def test_task_manager_completed_task_clears_blocked_by_from_other_tasks(
    tmp_path: Path,
) -> None:
    manager = TaskManager(tmp_path / ".tasks")
    dependency = manager.create("Dependency")
    other_dependency = manager.create("Other dependency")
    blocked = manager.create("Blocked task", blocked_by=[dependency.id])
    still_blocked = manager.create(
        "Still blocked",
        blocked_by=[dependency.id, other_dependency.id],
    )

    completed = manager.update(dependency.id, status="completed")

    assert completed.status == "completed"
    assert manager.get(blocked.id).blocked_by == []
    assert manager.get(still_blocked.id).blocked_by == [other_dependency.id]


def test_task_manager_completed_task_clears_its_own_blocked_by(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path / ".tasks")
    dependency = manager.create("Dependency")
    blocked = manager.create("Blocked task", blocked_by=[dependency.id])

    completed = manager.update(
        blocked.id,
        status="completed",
        add_blocked_by=[dependency.id],
    )

    assert completed.status == "completed"
    assert completed.blocked_by == []
    assert manager.get(blocked.id).blocked_by == []


def test_task_manager_does_not_store_completed_blocked_by_dependencies(
    tmp_path: Path,
) -> None:
    manager = TaskManager(tmp_path / ".tasks")
    completed_dependency = manager.create("Completed dependency")
    active_dependency = manager.create("Active dependency")
    manager.update(completed_dependency.id, status="completed")

    created = manager.create(
        "Follow-up task",
        blocked_by=[completed_dependency.id, active_dependency.id],
    )
    updated = manager.update(
        active_dependency.id,
        add_blocked_by=[completed_dependency.id],
    )

    assert created.blocked_by == [active_dependency.id]
    assert updated.blocked_by == []


def test_task_manager_update_rejects_self_dependency(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path / ".tasks")
    task = manager.create("Task")

    with pytest.raises(ValueError, match="task cannot block itself"):
        manager.update(task.id, add_blocked_by=[task.id])


def test_task_manager_update_rejects_missing_dependencies(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path / ".tasks")
    task = manager.create("Task")

    with pytest.raises(ValueError, match="task 99 not found"):
        manager.update(task.id, add_blocked_by=[99])
