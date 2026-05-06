from __future__ import annotations

from typing import Any

from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.session_store import SessionStore


class StartupReconciliationError(RuntimeError):
    """Raised when recovery cannot reconcile startup state from ledger."""


class RecoveryManager:
    """Ledger-backed startup/session recovery helper for control-plane wiring."""

    def __init__(self, *, ledger: ExecutionLedger):
        self.ledger = ledger

    def recover(self, *, session_store: SessionStore | None = None) -> dict[str, Any]:
        store = session_store or SessionStore(ledger=self.ledger, projection_only=True)
        if store.ledger is None:
            store.ledger = self.ledger

        try:
            events = self.ledger.read()
            store.rebuild_from_ledger(events)
            snapshot = store.snapshot()
            pending = list(store.pending_jobs())
        except Exception as exc:  # noqa: BLE001 - recovery boundary
            raise StartupReconciliationError(str(exc)) from exc

        return {
            "pending_jobs": pending,
            "ledger_events": len(events),
            "replay_hash": self.ledger.replay_hash(),
            "session_count": len(dict(snapshot.get("sessions") or {})),
            "session_snapshot_version": int(snapshot.get("version") or 0),
            "ok": True,
        }


__all__ = ["RecoveryManager", "StartupReconciliationError"]
