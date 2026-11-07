from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone
from typing import Any


def create_reviewed_evidence(transcript: str) -> dict[str, str]:
    return {
        "id": uuid.uuid4().hex,
        "kind": "reviewed_transcript",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "text": transcript,
        "sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
    }


def verified_reviewed_evidence(memory: dict[str, Any]) -> dict[str, str] | None:
    evidence = memory.get("oral_history_evidence")
    if not isinstance(evidence, dict):
        return None
    identifier = evidence.get("id")
    reviewed_at = evidence.get("reviewed_at")
    text = evidence.get("text")
    digest = evidence.get("sha256")
    if (
        not isinstance(identifier, str)
        or not re.fullmatch(r"[a-f0-9]{32}", identifier)
        or evidence.get("kind") != "reviewed_transcript"
        or not isinstance(reviewed_at, str)
        or not isinstance(text, str)
        or not text
        or text != memory.get("summary")
        or not isinstance(digest, str)
        or digest != hashlib.sha256(text.encode("utf-8")).hexdigest()
    ):
        return None
    try:
        timestamp = datetime.fromisoformat(reviewed_at)
    except ValueError:
        return None
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        return None
    return {"id": identifier, "kind": "reviewed_transcript", "reviewed_at": reviewed_at, "sha256": digest}
