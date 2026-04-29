from __future__ import annotations

import time
from typing import Any

from dadbot.core.execution_ledger import ExecutionLedger


class LedgerWriter:
    """Canonical event writer for execution lifecycle events."""

    def __init__(self, ledger: ExecutionLedger):
        self._ledger = ledger

    @staticmethod
    def _job_payload(job: Any) -> dict[str, Any]:
        metadata = dict(getattr(job, "metadata", {}) or {})
        return {
            "job_id": str(getattr(job, "job_id", "") or ""),
            "session_id": str(getattr(job, "session_id", "") or "default"),
            "user_input": str(getattr(job, "user_input", "") or ""),
            "attachments": getattr(job, "attachments", None),
            "metadata": metadata,
        }

    def write_event(
        self,
        event_type: str,
        *,
        session_id: str,
        trace_id: str,
        kernel_step_id: str,
        payload: dict[str, Any] | None = None,
        committed: bool = False,
    ) -> dict[str, Any]:
        return self._ledger.append(
            {
                "type": str(event_type or "EVENT"),
                "session_id": str(session_id or "default"),
                "trace_id": str(trace_id or ""),
                "kernel_step_id": str(kernel_step_id or ""),
                "payload": dict(payload or {}),
                "committed": bool(committed),
                "timestamp": time.time(),
            }
        )

    def append_session_bound(self, session_id: str) -> dict[str, Any]:
        return self._ledger.append(
            {
                "type": "SESSION_BOUND",
                "session_id": str(session_id or "default"),
                "timestamp": time.time(),
            }
        )

    def append_job_submitted(self, job: Any) -> dict[str, Any]:
        return self._ledger.append({"type": "JOB_SUBMITTED", **self._job_payload(job), "timestamp": time.time()})

    def append_job_queued(self, job: Any) -> dict[str, Any]:
        return self._ledger.append({"type": "JOB_QUEUED", **self._job_payload(job), "timestamp": time.time()})

    def append_job_started(self, job: Any) -> dict[str, Any]:
        return self._ledger.append({"type": "JOB_STARTED", **self._job_payload(job), "timestamp": time.time()})

    def append_job_completed(self, job: Any, result: Any) -> dict[str, Any]:
        return self._ledger.append(
            {
                "type": "JOB_COMPLETED",
                **self._job_payload(job),
                "result": result,
                "timestamp": time.time(),
            }
        )

    def append_job_failed(self, job: Any, error: str) -> dict[str, Any]:
        return self._ledger.append(
            {
                "type": "JOB_FAILED",
                **self._job_payload(job),
                "error": str(error or ""),
                "timestamp": time.time(),
            }
        )

    def append_runtime_witness(self, component: str, trace_id: str = "", session_id: str = "") -> dict[str, Any]:
        return self._ledger.append(
            {
                "type": "RUNTIME_WITNESS",
                "component": str(component or ""),
                "trace_id": str(trace_id or ""),
                "session_id": str(session_id or ""),
                "timestamp": time.time(),
            }
        )
