"""Load subagent role profiles from TOML files."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentProfile:
    """Configuration for a subagent role."""

    name: str
    description: str | None
    system_prompt: str
    allowed_tools: list[str]
    model: str = ""


class AgentProfileLoader:
    """Resolve subagent role profiles by priority."""

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

    def resolve(self, name: str) -> AgentProfile | None:
        for path in self._candidate_paths(name):
            if path.is_file():
                try:
                    return self._load(path, fallback_name=name)
                except Exception:
                    return None
        return None

    def load(self, name: str) -> AgentProfile | None:
        return self.resolve(name)

    def _candidate_paths(self, name: str) -> list[Path]:
        return [
            self._project_dir / ".myclaude" / "agents" / f"{name}.toml",
            self._home_dir / ".myclaude" / "agents" / f"{name}.toml",
            self._builtin_dir / f"{name}.toml",
        ]

    def _load(self, path: Path, *, fallback_name: str) -> AgentProfile:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        agent = data.get("agent", {})
        if not isinstance(agent, dict):
            agent = {}

        description = agent.get("description")
        system_prompt = agent.get("system_prompt")
        allowed_tools = agent.get("allowed_tools", [])
        model = agent.get("model", "")

        if description is not None and not isinstance(description, str):
            description = str(description)
        if not isinstance(system_prompt, str):
            system_prompt = ""
        if not isinstance(allowed_tools, list):
            allowed_tools = []
        if not isinstance(model, str):
            model = ""

        return AgentProfile(
            name=fallback_name,
            description=description,
            system_prompt=system_prompt.strip(),
            allowed_tools=[str(tool) for tool in allowed_tools],
            model=model,
        )
