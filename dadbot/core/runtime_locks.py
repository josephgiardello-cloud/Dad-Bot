from __future__ import annotations

import threading
from typing import Any


def runtime_turn_lock(bot: Any) -> Any:
    # Canonical sync turns bridge into a worker that also needs _session_lock.
    with bot._session_lock:
        lock = getattr(bot, "_streamlit_turn_lock", None)
        if lock is None:
            lock = threading.RLock()
            bot._streamlit_turn_lock = lock
        return lock
