from __future__ import annotations

from typing import Tuple

try:
    from notifypy import Notify  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    Notify = None

try:
    from plyer import notification as plyer_notification  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    plyer_notification = None


def send_local_notification(
    title: str,
    message: str,
    *,
    app_name: str = "Dad Bot",
    backend: str = "auto",
    timeout_seconds: int = 8,
) -> Tuple[bool, str]:
    """Best-effort local desktop notification using optional local backends."""
    normalized_backend = str(backend or "auto").strip().lower() or "auto"
    title_text = str(title or app_name).strip() or app_name
    body_text = str(message or "").strip()
    if not body_text:
        return False, "empty"

    candidates = []
    if normalized_backend == "auto":
        candidates = ["notifypy", "plyer"]
    elif normalized_backend in {"notifypy", "plyer"}:
        candidates = [normalized_backend]

    for candidate in candidates:
        if candidate == "notifypy" and Notify is not None:
            try:
                notice = Notify()
                notice.application_name = app_name
                notice.title = title_text
                notice.message = body_text
                notice.send(block=False)
                return True, "notifypy"
            except Exception:
                continue

        if candidate == "plyer" and plyer_notification is not None:
            try:
                plyer_notification.notify(
                    title=title_text,
                    message=body_text,
                    app_name=app_name,
                    timeout=max(2, int(timeout_seconds or 8)),
                )
                return True, "plyer"
            except Exception:
                continue

    return False, "none"


__all__ = ["send_local_notification"]
