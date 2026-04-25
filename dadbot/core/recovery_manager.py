from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.session_store import SessionStore

if TYPE_CHECKING:
    from dadbot.core.durable_checkpoint import DurableCheckpoint


class RecoveryError(RuntimeError):
    pass


class StartupReconciliationError(RecoveryError):
    """Raised when boot-time reconciliation detects an unrecoverable inconsistency.

    The runtime must NOT accept new turns after this error.
    """
    pass


class RecoveryManager:
    """Runtime recovery helpers backed by ledger truth."""

    def __init__(self, *, ledger: ExecutionLedger) -> None:
        self._ledger = ledger
        self._boot_complete: bool = False

    @property
    def boot_complete(self) -> bool:
        """True once boot_reconcile() has passed without error."""
        return self._boot_complete

    def _pending_jobs_from_events(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        queued: dict[str, dict[str, Any]] = {}
        terminal: set[str] = set()

        for event in events:
            event_type = str(event.get("type") or "")
            payload = dict(event.get("payload") or {})
            job_id = str(payload.get("job_id") or "").strip()
            if not job_id:
                continue

            if event_type in {"JOB_QUEUED", "JOB_SUBMITTED"}:
                queued[job_id] = {
                    "job_id": job_id,
                    "session_id": str(event.get("session_id") or "default"),
                    "request_id": str(payload.get("request_id") or "").strip(),
                    "user_input": str(payload.get("user_input") or ""),
                    "attachments": list(payload.get("attachments") or []),
                    "metadata": dict(payload.get("metadata") or {}),
                    "priority": int(payload.get("priority") or 0),
                    "submitted_at": float(payload.get("submitted_at") or 0.0),
                    "sequence": int(event.get("sequence") or 0),
                }
            elif event_type in {"JOB_COMPLETED", "JOB_FAILED"}:
                terminal.add(job_id)

        pending = [descriptor for job_id, descriptor in queued.items() if job_id not in terminal]
        pending.sort(key=lambda item: int(item.get("sequence") or 0))
        return pending

    def recover(self, *, session_store: SessionStore) -> dict[str, Any]:
        events = self._ledger.read()
        session_store.rebuild_from_ledger(events)
        pending_jobs = self._pending_jobs_from_events(events)

        replay_hash = self._ledger.replay_hash() if hasattr(self._ledger, "replay_hash") else ""
        snapshot = session_store.snapshot()

        return {
            "ledger_events": len(events),
            "pending_jobs": deepcopy(pending_jobs),
            "replay_hash": str(replay_hash or ""),
            "session_snapshot_version": int(snapshot.get("version") or 0),
            "session_count": len(dict(snapshot.get("sessions") or {})),
        }

    def assert_recoverable(self, *, session_store: SessionStore) -> dict[str, Any]:
        result = self.recover(session_store=session_store)
        if int(result.get("ledger_events") or 0) and int(result.get("session_snapshot_version") or 0) <= 0:
            raise RecoveryError("Recovery failed: non-empty ledger produced empty session projection")
        return result

    def boot_reconcile(
        self,
        *,
        session_store: SessionStore,
        checkpoint: "DurableCheckpoint | None" = None,
    ) -> dict[str, Any]:
        """Hard-fail startup gate.

        Must be called before the runtime accepts any new turns.  Raises
        StartupReconciliationError if *any* inconsistency is found:
        - Ledger replay hash mismatches saved checkpoint (if one exists).
        - Non-empty ledger produces empty session projection.

        On clean success, saves a fresh checkpoint and marks `boot_complete = True`.
        """
        if self._boot_complete:
            # Already reconciled â€” return fast.
            return {"ok": True, "reason": "already reconciled", "boot_complete": True}

        # Step 1 â€” checkpoint integrity first (catches truncation / corruption).
        checkpoint_report: dict[str, Any] = {}
        if checkpoint is not None:
            try:
                checkpoint_report = checkpoint.assert_resume_at_head()
            except Exception as exc:
                raise StartupReconciliationError(
                    f"Boot-time checkpoint integrity check failed: {exc}"
                ) from exc

        # Step 2 â€” ledger â†’ projection rebuild + pending job extraction.
        try:
            recovery_result = self.assert_recoverable(session_store=session_store)
        except RecoveryError as exc:
            raise StartupReconciliationError(
                f"Boot-time ledger reconciliation failed: {exc}"
            ) from exc

        # Step 3 â€” save a fresh post-boot checkpoint.
        if checkpoint is not None:
            checkpoint.save(label="boot_reconcile")

        self._boot_complete = True
        return {
            "ok": True,
            "boot_complete": True,
            "checkpoint_report": checkpoint_report,
            **recovery_result,
        }
