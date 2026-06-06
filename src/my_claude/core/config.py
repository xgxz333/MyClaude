from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

try:
    from dotenv import dotenv_values
except ModuleNotFoundError:
    dotenv_values = None


DEFAULT_CONFIG_FILE = Path("myclaude.toml")
DEFAULT_ENV_FILE = Path(".env")
ENV_PREFIX = "MYCLAUDE_"


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    app_name: str = "MyClaude"
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"
    debug: bool = False
    ipc_timeout_seconds: float = Field(default=5.0, gt=0)
    core_host: str = "127.0.0.1"
    core_port: int = Field(default=8765, gt=0, le=65535)
    max_request_bytes: int = Field(default=65536, gt=0)


def load_config(
    config_file: Path | str = DEFAULT_CONFIG_FILE,
    env_file: Path | str = DEFAULT_ENV_FILE,
) -> AppConfig:
    values: dict[str, Any] = {}
    values.update(_load_toml_config(Path(config_file)))
    values.update(_load_env_file(Path(env_file)))
    values.update(_load_system_env())
    return AppConfig.model_validate(values)


def _load_toml_config(config_file: Path) -> dict[str, Any]:
    if not config_file.exists():
        return {}

    with config_file.open("rb") as file:
        data = tomllib.load(file)

    section = data.get("myclaude", data)
    if not isinstance(section, dict):
        return {}

    return dict(section)


def _load_env_file(env_file: Path) -> dict[str, Any]:
    if not env_file.exists():
        return {}

    if dotenv_values is not None:
        return _extract_prefixed_values(dotenv_values(env_file))

    return _extract_prefixed_values(_read_simple_env_file(env_file))


def _load_system_env() -> dict[str, Any]:
    return _extract_prefixed_values(os.environ)


def _extract_prefixed_values(values: Mapping[str, str | None]) -> dict[str, str]:
    config: dict[str, str] = {}

    for key, value in values.items():
        if value is None or not key.startswith(ENV_PREFIX):
            continue
        config_key = key.removeprefix(ENV_PREFIX).lower()
        config[config_key] = value

    return config


def _read_simple_env_file(env_file: Path) -> dict[str, str]:
    values: dict[str, str] = {}

    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")

    return values
