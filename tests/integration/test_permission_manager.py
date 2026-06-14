from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from my_claude.core.permissions.manager import PermissionManager
from my_claude.core.permissions.policy import PermissionDecision, ToolPolicy
from my_claude.core.permissions.storage import load_policy_file


def test_permission_manager_allows_static_allow_without_emitting_event() -> None:
    events: list[dict[str, Any]] = []

    async def run() -> PermissionDecision:
        manager = PermissionManager(
            event_emitter=events.append,
            policies={"read_file": ToolPolicy(default=PermissionDecision.ALLOW)},
        )
        return await manager.authorize(
            tool_use_id="tool-1",
            tool_name="read_file",
            params={"path": "README.md"},
            session_id="sess-1",
        )

    assert asyncio.run(run()) is PermissionDecision.ALLOW
    assert events == []


def test_permission_manager_denies_static_deny_without_emitting_event() -> None:
    events: list[dict[str, Any]] = []

    async def run() -> PermissionDecision:
        manager = PermissionManager(
            event_emitter=events.append,
            policies={
                "bash": ToolPolicy(
                    deny_patterns=[r"\brm\b"],
                    default=PermissionDecision.ASK,
                )
            },
        )
        return await manager.authorize(
            tool_use_id="tool-1",
            tool_name="bash",
            params={"command": "rm file.txt"},
            session_id="sess-1",
        )

    assert asyncio.run(run()) is PermissionDecision.DENY
    assert events == []


def test_permission_manager_requests_approval_and_resumes_on_allow() -> None:
    events: list[dict[str, Any]] = []

    async def run() -> PermissionDecision:
        manager = PermissionManager(event_emitter=events.append, timeout_s=1)
        task = asyncio.create_task(
            manager.authorize(
                tool_use_id="tool-1",
                tool_name="bash",
                params={"command": "printf 'ok'"},
                session_id="sess-1",
            )
        )

        while not manager.pending():
            await asyncio.sleep(0)

        assert "tool-1" in manager.pending()
        manager.respond("tool-1", "ALLOW")
        return await task

    assert asyncio.run(run()) is PermissionDecision.ALLOW
    assert len(events) == 1
    event = events[0]
    assert event["type"] == "permission.requested"
    assert event["tool_use_id"] == "tool-1"
    assert event["tool_name"] == "bash"
    assert event["session_id"] == "sess-1"
    assert event["params"] == {"command": "printf 'ok'"}
    assert event["param_preview"] == 'command="printf \'ok\'"'
    assert isinstance(event["ts"], str)


def test_permission_manager_requests_approval_and_resumes_on_deny() -> None:
    async def run() -> tuple[bool, bool]:
        manager = PermissionManager(timeout_s=1)
        task = asyncio.create_task(
            manager.request_approval(
                tool_use_id="tool-1",
                tool_name="write_file",
                params={"path": "out.txt", "content": "data"},
                session_id="sess-1",
            )
        )

        while not manager.pending():
            await asyncio.sleep(0)

        manager.respond("tool-1", "deny")
        allowed = await task
        return allowed, bool(manager.pending())

    allowed, has_pending = asyncio.run(run())

    assert allowed is False
    assert has_pending is False


def test_permission_manager_times_out_and_removes_pending_request() -> None:
    async def run() -> tuple[bool, bool]:
        manager = PermissionManager(timeout_s=0.01)
        allowed = await manager.request_approval(
            tool_use_id="tool-1",
            tool_name="bash",
            params={"command": "printf 'ok'"},
            session_id="sess-1",
        )
        return allowed, bool(manager.pending())

    allowed, has_pending = asyncio.run(run())

    assert allowed is False
    assert has_pending is False


def test_permission_manager_warns_for_unknown_response(caplog: Any) -> None:
    manager = PermissionManager()

    with caplog.at_level(logging.WARNING):
        manager.respond("missing", "ALLOW")

    assert "unknown tool_use_id" in caplog.text


def test_permission_manager_invalid_response_denies_request(caplog: Any) -> None:
    async def run() -> bool:
        manager = PermissionManager(timeout_s=1)
        task = asyncio.create_task(
            manager.request_approval(
                tool_use_id="tool-1",
                tool_name="bash",
                params={"command": "printf 'ok'"},
            )
        )

        while not manager.pending():
            await asyncio.sleep(0)

        manager.respond("tool-1", "maybe")
        return await task

    with caplog.at_level(logging.WARNING):
        allowed = asyncio.run(run())

    assert allowed is False
    assert "invalid permission decision" in caplog.text


def test_permission_manager_always_allow_updates_caches_and_policy_file(
    tmp_path: Path,
) -> None:
    events: list[dict[str, Any]] = []
    policy_file = tmp_path / "policy.toml"

    async def run() -> tuple[tuple[bool, str], tuple[bool, str]]:
        manager = PermissionManager(
            event_emitter=events.append,
            timeout_s=1,
            policy_file=policy_file,
        )

        first = asyncio.create_task(
            manager.check_and_wait(
                tool_use_id="tool-1",
                tool_name="bash",
                params={"command": "printf 'ok'"},
                session_id="sess-1",
            )
        )
        while not manager.pending():
            await asyncio.sleep(0)
        manager.respond("tool-1", "always_allow")
        first_allowed = await first

        second_allowed = await manager.check_and_wait(
            tool_use_id="tool-2",
            tool_name="bash",
            params={"command": "printf 'again'"},
            session_id="sess-1",
        )
        return first_allowed, second_allowed

    first_allowed, second_allowed = asyncio.run(run())

    assert first_allowed == (True, "always_allow")
    assert second_allowed == (True, "auto_allow")
    assert len(events) == 1
    assert load_policy_file(policy_file) == {"bash": "allow"}


def test_permission_manager_loads_persistent_always_decision(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.toml"
    policy_file.write_text('[always]\nwrite_file = "deny"\n', encoding="utf-8")
    events: list[dict[str, Any]] = []

    async def run() -> tuple[bool, str]:
        manager = PermissionManager(event_emitter=events.append, policy_file=policy_file)
        return await manager.check_and_wait(
            tool_use_id="tool-1",
            tool_name="write_file",
            params={"path": "out.txt", "content": "data"},
            session_id="sess-1",
        )

    assert asyncio.run(run()) == (False, "auto_deny")
    assert events == []


def test_permission_manager_always_deny_updates_caches_and_policy_file(
    tmp_path: Path,
) -> None:
    policy_file = tmp_path / "policy.toml"

    async def run() -> tuple[tuple[bool, str], tuple[bool, str]]:
        manager = PermissionManager(timeout_s=1, policy_file=policy_file)
        task = asyncio.create_task(
            manager.check_and_wait(
                tool_use_id="tool-1",
                tool_name="write_file",
                params={"path": "out.txt", "content": "data"},
                session_id="sess-1",
            )
        )

        while not manager.pending():
            await asyncio.sleep(0)

        manager.respond("tool-1", "always_deny")
        first_allowed = await task
        second_allowed = await manager.check_and_wait(
            tool_use_id="tool-2",
            tool_name="write_file",
            params={"path": "again.txt", "content": "data"},
            session_id="sess-1",
        )
        return first_allowed, second_allowed

    assert asyncio.run(run()) == ((False, "always_deny"), (False, "auto_deny"))
    assert load_policy_file(policy_file) == {"write_file": "deny"}


def test_permission_manager_unknown_tool_can_use_persistent_cache() -> None:
    events: list[dict[str, Any]] = []
    manager = PermissionManager(event_emitter=events.append)
    manager._persistent_always["custom_tool"] = "allow"

    async def run() -> tuple[bool, str]:
        return await manager.check_and_wait(
            tool_use_id="tool-1",
            tool_name="custom_tool",
            params={"value": "x"},
            session_id="sess-1",
        )

    assert asyncio.run(run()) == (True, "auto_allow")
    assert events == []


def test_permission_manager_session_cache_takes_precedence_over_persistent_cache(
    tmp_path: Path,
) -> None:
    policy_file = tmp_path / "policy.toml"
    policy_file.write_text('[always]\nbash = "deny"\n', encoding="utf-8")
    manager = PermissionManager(policy_file=policy_file)
    manager._session_always[("sess-1", "bash")] = "allow"

    async def run() -> tuple[tuple[bool, str], tuple[bool, str]]:
        same_session = await manager.check_and_wait(
            tool_use_id="tool-1",
            tool_name="bash",
            params={"command": "printf 'ok'"},
            session_id="sess-1",
        )
        other_session = await manager.check_and_wait(
            tool_use_id="tool-2",
            tool_name="bash",
            params={"command": "printf 'ok'"},
            session_id="sess-2",
        )
        return same_session, other_session

    assert asyncio.run(run()) == ((True, "auto_allow"), (False, "auto_deny"))


def test_permission_manager_outside_cwd_ignores_always_allow_cache() -> None:
    events: list[dict[str, Any]] = []

    async def run() -> tuple[bool, str]:
        manager = PermissionManager(event_emitter=events.append, timeout_s=1)
        manager._session_always[("sess-1", "bash")] = "allow"
        manager._persistent_always["bash"] = "allow"

        task = asyncio.create_task(
            manager.check_and_wait(
                tool_use_id="tool-1",
                tool_name="bash",
                params={"command": "cat /etc/passwd"},
                session_id="sess-1",
            )
        )

        while not manager.pending():
            await asyncio.sleep(0)

        manager.respond("tool-1", "deny_once")
        return await task

    assert asyncio.run(run()) == (False, "deny_once")
    assert len(events) == 1
    assert events[0]["type"] == "permission.requested"


def test_permission_manager_deny_patterns_override_always_allow_cache() -> None:
    manager = PermissionManager(
        policies={
            "bash": ToolPolicy(
                deny_patterns=[r"\brm\b"],
                default=PermissionDecision.ASK,
            )
        }
    )
    manager._persistent_always["bash"] = "allow"

    async def run() -> tuple[bool, str]:
        return await manager.check_and_wait(
            tool_use_id="tool-1",
            tool_name="bash",
            params={"command": "rm file.txt"},
            session_id="sess-1",
        )

    assert asyncio.run(run()) == (False, "auto_deny")
