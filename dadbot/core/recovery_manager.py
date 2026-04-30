from __future__ import annotations

from typing import Any

from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.session_store import SessionStore


class StartupReconciliationError(RuntimeError):
    """Raised when boot reconciliation detects a corrupted durable state."""


class RecoveryManager:
    """Reconstruct pending execution state from an append-only ledger."""

    def __init__(self, ledger: ExecutionLedger):
        self.ledger = ledger
        self.boot_complete = False

    def recover(self, session_store: SessionStore) -> dict[str, Any]:
        events = self.ledger.read()
        if session_store.ledger is None:
            session_store.ledger = self.ledger
        session_store.rebuild_from_ledger(events)
        snap = session_store.snapshot()
        pending = list(session_store.pending_jobs())
        return {
            "pending_jobs": pending,
            "ledger_events": len(events),
            "replay_hash": self.ledger.replay_hash(),
            "session_count": len(dict(snap.get("sessions") or {})),
            "session_snapshot_version": int(snap.get("version") or 0),
        }

    def boot_reconcile(
        self,
        session_store: SessionStore,
        checkpoint: Any | None = None,
    ) -> dict[str, Any]:
        if self.boot_complete:
            report = self.recover(session_store)
            report.update({"ok": True, "boot_complete": True})
            return report
        if checkpoint is not None:
            try:
                checkpoint.assert_resume_at_head()
            except Exception as exc:
                raise StartupReconciliationError(str(exc)) from exc
        report = self.recover(session_store)
        if checkpoint is not None and checkpoint.latest() is None:
            checkpoint.save(label="boot_reconcile")
        self.boot_complete = True
        report.update({"ok": True, "boot_complete": True})
        return report
