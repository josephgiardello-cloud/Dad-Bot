from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _progress_instrumentation_enabled() -> bool:
    raw = str(os.environ.get("DADBOT_PROGRESS_INSTRUMENTATION", "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _progress_log_path() -> Path:
    raw = str(os.environ.get("DADBOT_PROGRESS_LOG_PATH", "session_logs/progress_instrumentation.ndjson")).strip()
    if not raw:
        raw = "session_logs/progress_instrumentation.ndjson"
    return Path(raw)


def _write_progress_event(*, component: str, phase: str, payload: dict[str, Any]) -> None:
    if not _progress_instrumentation_enabled():
        return
    event = {
        "timestamp": time.time(),
        "component": str(component or "control_plane"),
        "phase": str(phase or "unknown"),
        "payload": dict(payload or {}),
    }
    try:
        path = _progress_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
    except Exception as exc:
        logger.debug("progress instrumentation write failed: %s", exc)


__all__ = [
    "_progress_instrumentation_enabled",
    "_progress_log_path",
    "_write_progress_event",
]