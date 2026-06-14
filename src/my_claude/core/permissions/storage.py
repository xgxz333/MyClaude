"""Persistence helpers for permission "always" decisions."""

from __future__ import annotations

from pathlib import Path

AlwaysDecision = str
_DEFAULT_POLICY_PATH = Path("~/.myclaude/policy.toml")


def load_policy_file(path: str | Path | None = None) -> dict[str, AlwaysDecision]:
    """Load the `[always]` permission section as `{tool_name: allow|deny}`."""

    policy_path = (Path(path) if path is not None else _DEFAULT_POLICY_PATH).expanduser()
    if not policy_path.exists():
        return {}

    result: dict[str, AlwaysDecision] = {}
    in_always = False
    for line in policy_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "[always]":
            in_always = True
            continue
        if stripped.startswith("["):
            in_always = False
            continue
        if not in_always or "=" not in stripped or stripped.startswith("#"):
            continue

        key, _, raw_value = stripped.partition("=")
        tool_name = key.strip()
        decision = raw_value.strip().strip('"')
        if tool_name and decision in ("allow", "deny"):
            result[tool_name] = decision

    return result


def save_policy_file(
    always: dict[str, AlwaysDecision],
    path: str | Path | None = None,
) -> None:
    """Write the `[always]` permission section, replacing previous contents."""

    policy_path = (Path(path) if path is not None else _DEFAULT_POLICY_PATH).expanduser()
    policy_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# ~/.myclaude/policy.toml",
        "# Managed by myclaude-core; manual edits take effect if this format is kept.",
        "",
        "[always]",
    ]
    for tool_name, decision in sorted(always.items()):
        if decision in ("allow", "deny"):
            lines.append(f'{tool_name} = "{decision}"')

    policy_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["AlwaysDecision", "load_policy_file", "save_policy_file"]
