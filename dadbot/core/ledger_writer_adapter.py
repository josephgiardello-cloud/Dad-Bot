from __future__ import annotations

from typing import Any

from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_writer import LedgerWriter


class LedgerWriterAdapter:
    """Single gateway for writing execution lifecycle events to the ledger."""

    def __init__(self, ledger: ExecutionLedger) -> None:
        self._writer = LedgerWriter(ledger)

    def append_job_submitted(self, job: Any) -> dict[str, Any]:
        return self._writer.append_job_submitted(job)

    def append_job_queued(self, job: Any) -> dict[str, Any]:
        return self._writer.append_job_queued(job)

    def append_job_started(self, job: Any) -> dict[str, Any]:
        return self._writer.append_job_started(job)

    def append_job_completed(self, job: Any, result: Any) -> dict[str, Any]:
        return self._writer.append_job_completed(job, result)

    def append_job_failed(self, job: Any, error: str) -> dict[str, Any]:
        return self._writer.append_job_failed(job, error)

    def append_session_bound(
        self,
        session_id: str,
        job_id: str = "",
        *,
        trace_id: str = "",
        kernel_step_id: str = "control_plane.bind_session",
    ) -> dict[str, Any]:
        return self._writer.append_session_bound(
            session_id,
            job_id,
            trace_id=trace_id,
            kernel_step_id=kernel_step_id,
        )

    def append_runtime_witness(
        self,
        component: str,
        trace_id: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        return self._writer.append_runtime_witness(
            component,
            trace_id=trace_id,
            session_id=session_id,
        )

    def write_event(self, **kwargs: Any) -> dict[str, Any]:
        return self._writer.write_event(**kwargs)


__all__ = ["LedgerWriterAdapter"]
