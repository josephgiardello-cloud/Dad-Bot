from __future__ import annotations

import logging
from typing import Any


class Logger:
    """Structured logging shim for orchestration/services/infrastructure."""

    def __init__(self, name: str = "dadbot.telemetry"):
        self._logger = logging.getLogger(name)

    def info(self, message: str, **fields: Any) -> None:
        self._logger.info(self._format(message, fields))

    def warning(self, message: str, **fields: Any) -> None:
        self._logger.warning(self._format(message, fields))

    def error(self, message: str, **fields: Any) -> None:
        self._logger.error(self._format(message, fields))

    def metric(self, name: str, value: Any, **fields: Any) -> None:
        self._logger.info(self._format(f"metric={name} value={value}", fields))

    def trace(self, name: str, **fields: Any) -> None:
        self._logger.debug(self._format(f"trace={name}", fields))

    @staticmethod
    def _format(message: str, fields: dict[str, Any]) -> str:
        if not fields:
            return message
        pairs = " ".join(f"{key}={value!r}" for key, value in sorted(fields.items()))
        return f"{message} | {pairs}"
