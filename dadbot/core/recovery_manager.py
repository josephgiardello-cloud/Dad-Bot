from __future__ import annotations

import hashlib
import json
from typing import Any

from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.session_store import SessionStore


class RecoveryManager:
    """Reconstruct pending execution state from an append-only ledger."""

    def __init__(self, ledger: ExecutionLedger):
        self.ledger = ledger

    def recover(self, session_store: SessionStore) -> dict[str, Any]:
        events = self.ledger.read()
        pending = list(session_store.pending_jobs())

        digest = hashlib.sha256(
            json.dumps(events, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

        return {
            "pending_jobs": pending,
            "ledger_events": len(events),
            "replay_hash": digest,
        }
