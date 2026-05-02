from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

POST_COMMIT_READY = "dadbot.post_commit.ready"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(slots=True)
class PostCommitEvent:
    session_id: str
    trace_id: str
    tenant_id: str = "default"
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=_utc_now_iso)
    event_type: str = field(default=POST_COMMIT_READY, init=False)
