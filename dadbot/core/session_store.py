from __future__ import annotations

from typing import Any

from dadbot.core.execution_ledger import ExecutionLedger


class SessionStore:
    """Projection helper over ledger events for recovery operations."""

    def __init__(self, ledger: ExecutionLedger, projection_only: bool = True):
        self.ledger = ledger
        self.projection_only = bool(projection_only)

    def pending_jobs(self) -> list[dict[str, Any]]:
        events = self.ledger.read()
        latest_by_job: dict[str, dict[str, Any]] = {}
        for event in events:
            job_id = str(event.get("job_id") or "")
            if not job_id:
                continue
            latest_by_job[job_id] = event

        pending: list[dict[str, Any]] = []
        for job_id, event in latest_by_job.items():
            event_type = str(event.get("type") or "")
            if event_type in {"JOB_SUBMITTED", "JOB_QUEUED", "JOB_STARTED"}:
                pending.append(
                    {
                        "job_id": job_id,
                        "session_id": str(event.get("session_id") or "default"),
                        "status": event_type,
                    }
                )
        return pending
