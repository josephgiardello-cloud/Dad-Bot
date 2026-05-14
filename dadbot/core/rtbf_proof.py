from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode(
            "utf-8",
        ),
    ).hexdigest()


def record_rtbf_receipt(
    *,
    receipt_path: Path,
    actor: str,
    reason: str,
    before_snapshot: dict[str, Any],
    after_snapshot: dict[str, Any],
) -> dict[str, Any]:
    marker = {
        "receipt_id": uuid.uuid4().hex,
        "event_type": "rtbf_delete_marker",
        "occurred_at": _utc_now_iso(),
        "actor": str(actor or "system"),
        "reason": str(reason or "user_request"),
        "before_hash": _stable_hash(before_snapshot),
        "after_hash": _stable_hash(after_snapshot),
    }
    marker["proof_hash"] = _stable_hash(marker)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(marker, ensure_ascii=True, default=str) + "\n")
    return marker


__all__ = ["record_rtbf_receipt"]
