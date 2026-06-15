"""Load slash-command skills from Markdown files with YAML frontmatter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Skill:
    """A resolved slash-command skill definition."""

    name: str
    description: str | None
    allowed_tools: list[str]
    system_prompt_template: str


class SkillLoader:
    """Resolve and render Markdown-backed slash command skills."""

    def __init__(
        self,
        *,
        project_dir: Path | None = None,
        home_dir: Path | None = None,
        builtin_dir: Path | None = None,
    ) -> None:
        self._project_dir = project_dir or Path.cwd()
        self._home_dir = home_dir or Path.home()
        self._builtin_dir = builtin_dir or Path(__file__).resolve().parent / "builtin"

    def resolve(self, name: str) -> Skill | None:
        """Return the first matching skill by configured priority."""

        for path in self._candidate_paths(name):
            if path.is_file():
                return self._load(path, fallback_name=name)
        return None

    def render_prompt(self, skill: Skill, arguments: str) -> str:
        """Render the skill prompt body with slash command arguments."""

        return skill.system_prompt_template.replace("$ARGUMENTS", arguments)

    def _candidate_paths(self, name: str) -> list[Path]:
        dirs = [
            self._project_dir / ".myclaude" / "skills",
            self._home_dir / ".kammyclaude" / "skills",
            self._home_dir / ".myclaude" / "skills",
            self._builtin_dir,
        ]
        paths: list[Path] = []
        for directory in dirs:
            paths.append(directory / f"{name}.md")
            paths.append(directory / name / "SKILL.md")
        return paths

    def list_all(self) -> list[str]:
        seen: dict[str, None] = {}
        for directory in [
            self._builtin_dir,
            self._home_dir / ".myclaude" / "skills",
            self._home_dir / ".kammyclaude" / "skills",
            self._project_dir / ".myclaude" / "skills",
        ]:
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.md")):
                seen[path.stem] = None
            for path in sorted(directory.glob("*/SKILL.md")):
                seen[path.parent.name] = None
        return list(seen)

    def list_all_skills(self) -> list[Skill]:
        skills: dict[str, Skill] = {}
        for name in self.list_all():
            skill = self.resolve(name)
            if skill is not None:
                skills[skill.name] = skill
        return list(skills.values())

    def _load(self, path: Path, *, fallback_name: str) -> Skill:
        frontmatter, body = _split_frontmatter(path.read_text(encoding="utf-8"))
        metadata = _parse_frontmatter(frontmatter)
        name = metadata.get("name") or fallback_name
        description = metadata.get("description")
        allowed_tools = metadata.get("allowed_tools") or []
        if not isinstance(name, str):
            name = fallback_name
        if description is not None and not isinstance(description, str):
            description = str(description)
        if not isinstance(allowed_tools, list):
            allowed_tools = []
        return Skill(
            name=name,
            description=description,
            allowed_tools=[str(tool) for tool in allowed_tools],
            system_prompt_template=body.strip(),
        )


def _split_frontmatter(content: str) -> tuple[str, str]:
    if not content.startswith("---\n"):
        return "", content
    try:
        _, frontmatter, body = content.split("---\n", 2)
    except ValueError:
        return "", content
    return frontmatter, body


def _parse_frontmatter(frontmatter: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in frontmatter.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if current_list_key is not None and stripped.startswith("- "):
            value = stripped[2:].strip()
            metadata[current_list_key].append(_parse_scalar(value))
            continue
        current_list_key = None
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value:
            metadata[key] = _parse_scalar(value)
        else:
            metadata[key] = []
            current_list_key = key
    return metadata


def _parse_scalar(value: str) -> str | list[str] | None:
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_strip_quotes(part.strip()) for part in inner.split(",")]
    return _strip_quotes(value)


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
