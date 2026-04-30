from __future__ import annotations

from dadbot.core.execution_ledger import ExecutionLedger


class LedgerReader:
    """Read-only helper for querying execution lifecycle state."""

    def __init__(self, ledger: ExecutionLedger):
        self._ledger = ledger

    def events(self) -> list[dict[str, object]]:
        return self._ledger.read()

    def events_for_job(self, job_id: str) -> list[dict[str, object]]:
        jid = str(job_id or "")
        return [
            e
            for e in self._ledger.read()
            if str((e.get("payload") or {}).get("job_id") or e.get("job_id") or "") == jid
        ]

    def is_terminal(self, job_id: str) -> bool:
        for event in reversed(self.events_for_job(job_id)):
            event_type = str(event.get("type") or "")
            if event_type in {"JOB_COMPLETED", "JOB_FAILED"}:
                return True
            if event_type in {"JOB_STARTED", "JOB_QUEUED", "JOB_SUBMITTED"}:
                return False
        return False
