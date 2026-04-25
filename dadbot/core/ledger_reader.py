from __future__ import annotations

from dadbot.core.execution_ledger import ExecutionLedger


class LedgerReader:
    """Read boundary over authoritative execution ledger."""

    def __init__(self, ledger: ExecutionLedger):
        self._ledger = ledger
        self._cursor = 0

    def next_unprocessed_event(self, *, event_type: str = "", job_type: str = "") -> dict | None:
        wanted_type = str(event_type or job_type or "")
        events, next_cursor = self._ledger.events_since(self._cursor)
        self._cursor = next_cursor
        for event in events:
            if not wanted_type or str(event.get("type") or "") == wanted_type:
                return event
        return None

    def get_pending_jobs(self) -> list[str]:
        events = self._ledger.read()
        queued: list[str] = []
        started_or_finished: set[str] = set()

        for event in events:
            event_type = str(event.get("type") or "")
            payload = dict(event.get("payload") or {})
            job_id = str(payload.get("job_id") or "").strip()
            if not job_id:
                continue
            if event_type == "JOB_QUEUED":
                queued.append(job_id)
            if event_type in {"JOB_STARTED", "JOB_COMPLETED", "JOB_FAILED"}:
                started_or_finished.add(job_id)

        return [job_id for job_id in queued if job_id not in started_or_finished]

    def replay_session(self, session_id: str) -> list[dict]:
        return self._ledger.filter(session_id=session_id)
