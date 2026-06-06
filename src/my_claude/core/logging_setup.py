from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from my_claude.core.config import AppConfig


TEXT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def setup_logging(config: AppConfig) -> None:
    level = logging.getLevelNamesMapping().get(config.log_level.upper(), logging.INFO)
    formatter: logging.Formatter

    if config.log_format == "json":
        formatter = JsonLogFormatter()
    else:
        formatter = logging.Formatter(TEXT_LOG_FORMAT)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
