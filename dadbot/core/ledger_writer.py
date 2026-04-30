from __future__ import annotations

import time
from typing import Any

from dadbot.core.execution_ledger import ExecutionLedger, WriteBoundaryGuard
from dadbot.core.invariant_gate import InvariantGate
from dadbot.core.observability import get_metrics


class LedgerWriter:
    """Canonical event writer for execution lifecycle events."""

    _COMMITTED_EVENT_TYPES = {
        "JOB_QUEUED",
        "JOB_COMPLETED",
        "JOB_FAILED",
        "SESSION_STATE_UPDATED",
    }

    def __init__(self, ledger: ExecutionLedger):
        self._ledger = ledger
        self._invariants = InvariantGate()

    @staticmethod
    def _trace_id_for_job(job: Any) -> str:
        trace_id = str(
            getattr(job, "trace_id", "")
            or dict(getattr(job, "metadata", {}) or {}).get("trace_id")
            or "",
        ).strip()
        if not trace_id:
            raise ValueError("trace_id required")
        return trace_id

    def _infer_trace_id_for_job(self, session_id: str, job_id: str) -> str:
        sid = str(session_id or "default")
        jid = str(job_id or "").strip()
        if not jid:
            raise ValueError("trace_id required")
        for event in reversed(self._ledger.read()):
            if str(event.get("session_id") or "") != sid:
                continue
            payload = dict(event.get("payload") or {})
            if str(payload.get("job_id") or "") != jid:
                continue
            trace_id = str(event.get("trace_id") or "").strip()
            if trace_id:
                return trace_id
        raise ValueError("trace_id required")

    @staticmethod
    def _job_payload(job: Any) -> dict[str, Any]:
        metadata = dict(getattr(job, "metadata", {}) or {})
        return {
            "job_id": str(getattr(job, "job_id", "") or ""),
            "request_id": str(metadata.get("request_id") or ""),
            "user_input": str(getattr(job, "user_input", "") or ""),
            "attachments": getattr(job, "attachments", None),
            "metadata": metadata,
        }

    def _write(self, payload: dict[str, Any]) -> dict[str, Any]:
        committed = bool(
            payload.get("committed", False) or str(payload.get("type") or "") in self._COMMITTED_EVENT_TYPES,
        )
        payload["committed"] = committed
        self._invariants.validate_event(payload)
        with WriteBoundaryGuard(self._ledger):
            event = self._ledger.write(payload)
        metrics = get_metrics()
        metrics.increment(f"ledger.write.{str(event.get('type') or '').lower()}")
        if committed:
            metrics.increment("ledger.committed_writes")
        return event

    def write_event(
        self,
        event_type: str,
        *,
        session_id: str,
        kernel_step_id: str,
        trace_id: str = "",
        payload: dict[str, Any] | None = None,
        committed: bool = False,
    ) -> dict[str, Any]:
        trace = str(trace_id or "").strip()
        step = str(kernel_step_id or "").strip()
        if not trace:
            raise ValueError("trace_id required")
        if not step:
            raise ValueError("kernel_step_id required")
        return self._write(
            {
                "type": str(event_type),
                "session_id": str(session_id or "default"),
                "trace_id": trace,
                "kernel_step_id": step,
                "payload": dict(payload or {}),
                "committed": bool(committed),
                "timestamp": time.time(),
            },
        )

    def append_session_bound(
        self,
        session_id: str,
        job_id: str = "",
        *,
        trace_id: str = "",
        kernel_step_id: str = "control_plane.bind_session",
    ) -> dict[str, Any]:
        resolved_trace = str(trace_id or "").strip() or self._infer_trace_id_for_job(
            session_id,
            job_id,
        )
        resolved_step = str(kernel_step_id or "").strip()
        if not resolved_step:
            raise ValueError("kernel_step_id required")
        return self._write(
            {
                "type": "SESSION_BOUND",
                "session_id": str(session_id or "default"),
                "trace_id": resolved_trace,
                "kernel_step_id": resolved_step,
                "payload": {"job_id": str(job_id or "")},
                "timestamp": time.time(),
            },
        )

    def append_job_submitted(self, job: Any) -> dict[str, Any]:
        return self._write(
            {
                "type": "JOB_SUBMITTED",
                "session_id": str(getattr(job, "session_id", "") or "default"),
                "trace_id": self._trace_id_for_job(job),
                "kernel_step_id": "control_plane.submit_turn",
                "payload": self._job_payload(job),
                "timestamp": time.time(),
            },
        )

    def append_job_queued(self, job: Any) -> dict[str, Any]:
        return self._write(
            {
                "type": "JOB_QUEUED",
                "session_id": str(getattr(job, "session_id", "") or "default"),
                "trace_id": self._trace_id_for_job(job),
                "kernel_step_id": "control_plane.enqueue",
                "payload": self._job_payload(job),
                "timestamp": time.time(),
            },
        )

    def append_job_started(self, job: Any) -> dict[str, Any]:
        return self._write(
            {
                "type": "JOB_STARTED",
                "session_id": str(getattr(job, "session_id", "") or "default"),
                "trace_id": self._trace_id_for_job(job),
                "kernel_step_id": "scheduler.execute.start",
                "payload": self._job_payload(job),
                "timestamp": time.time(),
            },
        )

    def append_job_completed(self, job: Any, result: Any) -> dict[str, Any]:
        payload = self._job_payload(job)
        payload["result"] = result
        return self._write(
            {
                "type": "JOB_COMPLETED",
                "session_id": str(getattr(job, "session_id", "") or "default"),
                "trace_id": self._trace_id_for_job(job),
                "kernel_step_id": "scheduler.execute.complete",
                "payload": payload,
                "timestamp": time.time(),
            },
        )

    def append_job_failed(self, job: Any, error: str) -> dict[str, Any]:
        payload = self._job_payload(job)
        payload["error"] = str(error or "")
        return self._write(
            {
                "type": "JOB_FAILED",
                "session_id": str(getattr(job, "session_id", "") or "default"),
                "trace_id": self._trace_id_for_job(job),
                "kernel_step_id": "scheduler.execute.failed",
                "payload": payload,
                "timestamp": time.time(),
            },
        )

    def append_runtime_witness(
        self,
        component: str,
        trace_id: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        trace = str(trace_id or "").strip()
        if not trace:
            raise ValueError("trace_id required")
        return self._write(
            {
                "type": "RUNTIME_WITNESS",
                "session_id": str(session_id or ""),
                "trace_id": trace,
                "kernel_step_id": "control_plane.runtime_witness",
                "payload": {"component": str(component or "")},
                "timestamp": time.time(),
            },
        )
