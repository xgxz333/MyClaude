from __future__ import annotations

import pytest

from my_claude.core.permissions.policy import (
    DEFAULT_POLICIES,
    PermissionDecision,
    ToolPolicy,
    evaluate,
    matches_outside_cwd,
)


def test_default_policies_match_tool_safety_profile() -> None:
    assert DEFAULT_POLICIES["bash"].default is PermissionDecision.ASK
    assert DEFAULT_POLICIES["write_file"].default is PermissionDecision.ASK
    assert DEFAULT_POLICIES["read_file"].default is PermissionDecision.ALLOW
    assert DEFAULT_POLICIES["list_dir"].default is PermissionDecision.ALLOW
    assert DEFAULT_POLICIES["note_save"].default is PermissionDecision.ALLOW


def test_unknown_tool_defaults_to_ask() -> None:
    assert evaluate("unknown", {}) is PermissionDecision.ASK


@pytest.mark.parametrize(
    "command",
    [
        "cat /etc/passwd",
        "printf hi > /tmp/out",
        "cat ~/notes.txt",
        "ls ../other",
        "echo $HOME",
        "printf ${PWD}",
        "cd subdir && ls",
        "printf hi; cd ..",
    ],
)
def test_matches_outside_cwd_detects_boundary_crossing_patterns(command: str) -> None:
    assert matches_outside_cwd(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "printf 'hello'",
        "ls src/my_claude",
        "mkdir -p build/output",
        "python -m pytest tests/integration/test_tools.py",
    ],
)
def test_matches_outside_cwd_allows_workspace_relative_commands(command: str) -> None:
    assert matches_outside_cwd(command) is False


def test_bash_policy_denies_before_outside_cwd_check() -> None:
    policy = ToolPolicy(
        deny_patterns=[r"\brm\s+-rf\b"],
        allow_patterns=[r".*"],
        default=PermissionDecision.ALLOW,
    )

    assert (
        evaluate("bash", {"command": "rm -rf /tmp/project"}, policy)
        is PermissionDecision.DENY
    )


def test_bash_outside_cwd_forces_ask_before_allow_patterns() -> None:
    policy = ToolPolicy(
        allow_patterns=[r".*"],
        default=PermissionDecision.ALLOW,
    )

    assert (
        evaluate("bash", {"command": "cat /etc/passwd"}, policy)
        is PermissionDecision.ASK
    )


def test_bash_allow_patterns_apply_after_safety_floor() -> None:
    policy = ToolPolicy(
        allow_patterns=[r"^printf "],
        default=PermissionDecision.ASK,
    )

    assert evaluate("bash", {"command": "printf 'ok'"}, policy) is PermissionDecision.ALLOW


def test_bash_falls_back_to_policy_default() -> None:
    policy = ToolPolicy(default=PermissionDecision.DENY)

    assert evaluate("bash", {"command": "make test"}, policy) is PermissionDecision.DENY


def test_non_bash_policy_ignores_patterns_and_uses_default() -> None:
    policy = ToolPolicy(
        deny_patterns=[r"secret"],
        allow_patterns=[r"README\.md"],
        default=PermissionDecision.ASK,
    )

    assert (
        evaluate("read_file", {"path": "secrets/secret.txt"}, policy)
        is PermissionDecision.ASK
    )
    assert (
        evaluate("read_file", {"path": "README.md"}, policy)
        is PermissionDecision.ASK
    )
    assert evaluate("read_file", {"path": "src/main.py"}, policy) is PermissionDecision.ASK
