"""Application configuration loaded from TOML, .env files, and environment variables."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG_FILE = Path("myclaude.toml")
DEFAULT_ENV_FILE = Path(".env")
ENV_PREFIX = "MYCLAUDE_"


class AppConfig(BaseModel):
    """Validated runtime settings shared by CLI, core server, and agent runner."""

    model_config = ConfigDict(extra="ignore")

    app_name: str = "MyClaude"
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"
    debug: bool = False
    ipc_timeout_seconds: float = Field(default=5.0, gt=0)
    core_host: str = "127.0.0.1"
    core_port: int = Field(default=7437, gt=0, le=65535)
    max_request_bytes: int = Field(default=65536, gt=0)
    runs_dir: Path = Path("runs")
    agent_max_iterations: int = Field(default=20, gt=0)
    llm_provider: Literal["local", "openai-compatible", "anthropic"] = "local"
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.openai.com/v1/responses"
    llm_model: str = "claude-sonnet-4-6"
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_tokens: int = Field(default=1024, gt=0)
    trace_enabled: bool = True
    trace_file: Path | None = None
    trace_include_llm_payload: bool = True


def load_config(
    config_file: Path | str = DEFAULT_CONFIG_FILE,
    env_file: Path | str = DEFAULT_ENV_FILE,
) -> AppConfig:
    values: dict[str, Any] = {}
    values.update(_load_toml_config(Path(config_file)))
    values.update(_load_env_file(Path(env_file)))
    values.update(_load_system_env())
    config = AppConfig.model_validate(values)
    _sync_provider_env(config)
    return config


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

    dotenv_values = _load_dotenv_values(env_file)
    if dotenv_values is not None:
        return _extract_prefixed_values(dotenv_values)

    return _extract_prefixed_values(_read_simple_env_file(env_file))


def _load_system_env() -> dict[str, Any]:
    return _extract_prefixed_values(os.environ)


def _load_dotenv_values(env_file: Path) -> Mapping[str, str | None] | None:
    try:
        from dotenv import dotenv_values
    except ModuleNotFoundError:
        return None

    return dotenv_values(env_file)


def _extract_prefixed_values(values: Mapping[str, str | None]) -> dict[str, str]:
    config: dict[str, str] = {}

    for key, value in values.items():
        if value is None:
            continue
        config_key = _config_key_from_env_key(key)
        if config_key is None:
            continue
        config[config_key] = value

    return config


def _config_key_from_env_key(key: str) -> str | None:
    aliases = {
        "ANTHROPIC_API_KEY": "llm_api_key",
        "ANTHROPIC_BASE_URL": "llm_base_url",
        "ANTHROPIC_MAX_TOKENS": "llm_max_tokens",
        "ANTHROPIC_MODEL": "llm_model",
        "MYCLAUDE_LLM_DEFAULT_MODEL": "llm_model",
        "MYCLAUDE_MAX_STEPS": "agent_max_iterations",
    }
    if key in aliases:
        return aliases[key]
    if not key.startswith(ENV_PREFIX):
        return None

    return key.removeprefix(ENV_PREFIX).lower()


def _read_simple_env_file(env_file: Path) -> dict[str, str]:
    values: dict[str, str] = {}

    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")

    return values


def _sync_provider_env(config: AppConfig) -> None:
    if config.llm_provider != "anthropic":
        return
    if config.llm_api_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = config.llm_api_key
    os.environ["ANTHROPIC_BASE_URL"] = _anthropic_sdk_base_url(config.llm_base_url)


def _anthropic_sdk_base_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped == "https://api.openai.com/v1/responses":
        return "https://api.anthropic.com"
    if stripped.endswith("/v1/messages"):
        return stripped.removesuffix("/v1/messages")
    if stripped.endswith("/v1"):
        return stripped.removesuffix("/v1")
    return stripped
