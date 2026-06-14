"""Static permission policy evaluation for tool calls.

This module is intentionally synchronous. It runs after tool parameter
validation has produced a typed argument dictionary, and before any async
user-approval workflow is started.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final


class PermissionDecision(StrEnum):
    """Static permission decision for a tool call."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(frozen=True)
class ToolPolicy:
    """Regex-based static policy for one tool."""

    default: PermissionDecision = PermissionDecision.ASK
    allow_patterns: list[str] = field(default_factory=list)
    deny_patterns: list[str] = field(default_factory=list)


DEFAULT_POLICIES: Final[dict[str, ToolPolicy]] = {
    "bash": ToolPolicy(default=PermissionDecision.ASK),
    "write_file": ToolPolicy(default=PermissionDecision.ASK),
    "read_file": ToolPolicy(default=PermissionDecision.ALLOW),
    "list_dir": ToolPolicy(default=PermissionDecision.ALLOW),
    "note_save": ToolPolicy(default=PermissionDecision.ALLOW),
}

# Heuristics that force user approval for bash commands that may escape the
# current working directory. These are intentionally conservative and mirror
# KamaClaude's stage/s5 static permission boundary.
OUTSIDE_CWD_HEURISTICS: Final[tuple[str, ...]] = (
    r"(^|\s)/[^\s]",  # absolute path token
    r"(^|\s)~",  # home directory expansion
    r"(^|\s)\.\.(/|$|\s)",  # parent traversal as a shell token
    r"\$\{?HOME\b",  # home directory variable
    r"\$\{?PWD\b",  # current directory variable, potentially reassigned
    r"(^|\s|;|&&|\|\|)cd(\s|$)",  # directory switching changes path meaning
)
_OUTSIDE_CWD_RE: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(pattern) for pattern in OUTSIDE_CWD_HEURISTICS
)
_UNKNOWN_TOOL_DEFAULT: Final[PermissionDecision] = PermissionDecision.ASK


def matches_outside_cwd(command: str) -> bool:
    """Return whether a bash command may access outside the current CWD.

    The check is heuristic by design. It does not parse shell syntax or expand
    variables; it only identifies common boundary-crossing patterns that should
    force a user approval prompt instead of being silently allowed.
    """

    return any(pattern.search(command) is not None for pattern in _OUTSIDE_CWD_RE)


def evaluate(
    tool_name: str,
    params: dict[str, Any],
    policy: ToolPolicy | None = None,
) -> PermissionDecision:
    """Evaluate a tool call against static permission policy."""

    resolved_policy = policy or DEFAULT_POLICIES.get(tool_name)
    if resolved_policy is None:
        return _UNKNOWN_TOOL_DEFAULT

    if tool_name == "bash":
        command = params.get("command")
        command_text = command if isinstance(command, str) else ""

        if _matches_any(command_text, resolved_policy.deny_patterns):
            return PermissionDecision.DENY
        if matches_outside_cwd(command_text):
            return PermissionDecision.ASK
        if _matches_any(command_text, resolved_policy.allow_patterns):
            return PermissionDecision.ALLOW
        return resolved_policy.default

    # Kama stage/s5 only applies regex allow/deny patterns to bash commands;
    # other tools use their declared default decision.
    return resolved_policy.default


def _matches_any(value: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, value) is not None for pattern in patterns)


def param_preview(tool_name: str, params: dict[str, Any]) -> str:
    """Return a compact, stable preview for permission UI cards."""

    keys_by_tool = {
        "bash": ("command",),
        "read_file": ("path",),
        "write_file": ("path",),
        "list_dir": ("path", "max_depth"),
        "note_save": ("content",),
    }
    keys = keys_by_tool.get(tool_name, ())
    parts = [f"{key}={params[key]!r}" for key in keys if key in params]
    text = ", ".join(parts) if parts else str(params)
    return text[:60] + "..." if len(text) > 60 else text


__all__ = [
    "DEFAULT_POLICIES",
    "PermissionDecision",
    "ToolPolicy",
    "evaluate",
    "matches_outside_cwd",
    "param_preview",
]
