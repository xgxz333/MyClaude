"""Load user-maintained memory/context files."""

from __future__ import annotations

from pathlib import Path


def load_context_file(path: Path) -> str:
    """Read a Markdown context file, returning empty text when it is absent."""

    expanded_path = path.expanduser()
    if not expanded_path.exists():
        return ""
    return expanded_path.read_text(encoding="utf-8").strip()
