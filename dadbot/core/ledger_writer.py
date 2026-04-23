from __future__ import annotations

import time
from typing import Any

from dadbot.core.execution_ledger import ExecutionLedger, WriteBoundaryGuard
from dadbot.core.invariant_gate import InvariantGate
from dadbot.core.observability import CorrelationContext, get_metrics, get_exporter


# Event types that must be synchronously durable before control returns.
_COMMITTED_EVENT_TYPES: frozenset[str] = frozenset({
    "JOB_QUEUED",
    "JOB_STARTED",
    "JOB_COMPLETED",
    "JOB_FAILED",
    "SESSION_STATE_UPDATED",
})


class LedgerWriter:
    """Single write contract for control-plane and session projection events.

    All writes flow through this class — it is the only authorised path into
    the ledger.  Integration points added in this wave:
      - InvariantGate: hard-fails on any invariant violation before writing.
      - Committed writes: critical event types set committed=True so the backend
        performs fsync before returning.
      - Metrics: increments job-lifecycle counters and records write latency.
      - WriteBoundaryGuard: activates the guard so strict-mode ledgers accept
        the write.
    """

    def __init__(
        self,
        ledger: ExecutionLedger,
        *,
        gate: InvariantGate | None = None,
    ) -> None:
        self._ledger = ledger
        self._gate = gate or InvariantGate()

    def append_execution_witness(
        self,
        *,
        component: str,
        session_id: str,
        trace_id: str = "",
        correlation_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        component_name = str(component or "").strip()
        if not component_name:
            raise ValueError("Execution witness component must be non-empty")
        witness_payload = {
            "component": component_name,
            **dict(payload or {}),
        }
        if correlation_id:
            witness_payload.setdefault("correlation_id", str(correlation_id))
        return self.write_event(
            event_type="EXECUTION_WITNESS",
            session_id=str(session_id or "default"),
            trace_id=str(trace_id or ""),
            kernel_step_id="runtime.execution_witness",
            payload=witness_payload,
            committed=False,
        )

    def _append_write_boundary_witness(
        self,
        *,
        session_id: str,
        trace_id: str,
        correlation_id: str,
        observed_event_type: str,
        observed_kernel_step_id: str,
    ) -> dict[str, Any]:
        envelope = {
            "type": "EXECUTION_WITNESS",
            "session_id": str(session_id or "default"),
            "trace_id": str(trace_id or ""),
            "correlation_id": str(correlation_id or ""),
            "timestamp": time.time(),
            "kernel_step_id": "ledger_writer.write_event",
            "payload": {
                "component": "ledger_writer.write_event",
                "observed_event_type": str(observed_event_type or ""),
                "observed_kernel_step_id": str(observed_kernel_step_id or ""),
                "trace_id": str(trace_id or ""),
                "correlation_id": str(correlation_id or ""),
            },
        }
        self._gate.validate_event(envelope)
        with WriteBoundaryGuard(self._ledger):
            return self._ledger.write(envelope, committed=False)

    def write_event(
        self,
        *,
        event_type: str,
        session_id: str,
        trace_id: str = "",
        kernel_step_id: str,
        payload: dict[str, Any] | None = None,
        committed: bool | None = None,
    ) -> dict[str, Any]:
        # InvariantGate: validate raw inputs BEFORE any default-substitution so
        # callers cannot sneak past the gate by relying on "UNKNOWN" fallbacks.
        from dadbot.core.invariant_gate import InvariantViolationError as _IVE
        if not str(event_type or "").strip():
            raise _IVE("Event 'type' must be a non-empty string")
        if not str(session_id or "").strip():
            raise _IVE("Event 'session_id' must be non-empty")
        if not str(kernel_step_id or "").strip():
            raise _IVE("Event 'kernel_step_id' must be non-empty — kernel lineage required")

        payload_data = dict(payload or {})
        correlation_id = str(
            payload_data.get("correlation_id")
            or CorrelationContext.current()
            or CorrelationContext.ensure()
            or ""
        ).strip()
        normalized_trace_id = str(trace_id or payload_data.get("trace_id") or correlation_id).strip()
        if not normalized_trace_id:
            raise _IVE("Event 'trace_id' must be non-empty")
        if not correlation_id:
            raise _IVE("Event 'correlation_id' must be non-empty")

        payload_data.setdefault("trace_id", normalized_trace_id)
        payload_data.setdefault("correlation_id", correlation_id)

        envelope = {
            "type": str(event_type),
            "session_id": str(session_id),
            "trace_id": normalized_trace_id,
            "correlation_id": correlation_id,
            "timestamp": time.time(),
            "kernel_step_id": str(kernel_step_id),
            "payload": payload_data,
        }

        # InvariantGate: hard-fail on violation — never log-and-continue.
        self._gate.validate_event(envelope)

        # Determine committed flag: critical types are always committed.
        should_commit = bool(committed) if committed is not None else (
            str(event_type or "") in _COMMITTED_EVENT_TYPES
        )

        t0 = time.monotonic()
        with WriteBoundaryGuard(self._ledger):
            written = self._ledger.write(envelope, committed=should_commit)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        # Metrics.
        metrics = get_metrics()
        metrics.increment(f"ledger.write.{str(event_type or 'UNKNOWN').lower()}")
        metrics.observe("ledger.write_latency_ms", elapsed_ms)
        if should_commit:
            metrics.increment("ledger.committed_writes")

        # Event stream export.
        get_exporter().export({
            "event": "ledger.write",
            "event_type": str(event_type or "UNKNOWN"),
            "session_id": str(session_id or "default"),
            "trace_id": normalized_trace_id,
            "correlation_id": correlation_id,
            "committed": should_commit,
            "latency_ms": elapsed_ms,
        })

        if str(event_type or "") != "EXECUTION_WITNESS":
            self._append_write_boundary_witness(
                session_id=str(session_id or "default"),
                trace_id=normalized_trace_id,
                correlation_id=correlation_id,
                observed_event_type=str(event_type or ""),
                observed_kernel_step_id=str(kernel_step_id or ""),
            )

        return written

    def append_job_submitted(self, job) -> dict[str, Any]:
        return self.write_event(
            event_type="JOB_SUBMITTED",
            session_id=str(job.session_id),
            trace_id=str(getattr(job, "metadata", {}).get("trace_id") or ""),
            kernel_step_id="control_plane.submit_turn",
            payload={
                "job_id": str(job.job_id),
                "request_id": str(getattr(job, "request_id", "") or ""),
                "user_input": str(getattr(job, "user_input", "") or ""),
                "attachments": list(getattr(job, "attachments", []) or []),
                "metadata": dict(getattr(job, "metadata", {}) or {}),
                "priority": int(job.priority),
                "submitted_at": float(job.submitted_at),
            },
        )

    def append_session_bound(self, session_id: str, job_id: str) -> dict[str, Any]:
        return self.write_event(
            event_type="SESSION_BOUND",
            session_id=session_id,
            kernel_step_id="control_plane.bind_session",
            payload={"job_id": str(job_id or "")},
        )

    def append_job_queued(self, job) -> dict[str, Any]:
        return self.write_event(
            event_type="JOB_QUEUED",
            session_id=str(job.session_id),
            trace_id=str(getattr(job, "metadata", {}).get("trace_id") or ""),
            kernel_step_id="control_plane.enqueue",
            payload={
                "job_id": str(job.job_id),
                "request_id": str(getattr(job, "request_id", "") or ""),
                "user_input": str(getattr(job, "user_input", "") or ""),
                "attachments": list(getattr(job, "attachments", []) or []),
                "metadata": dict(getattr(job, "metadata", {}) or {}),
                "priority": int(job.priority),
                "submitted_at": float(job.submitted_at),
            },
        )

    def append_job_started(self, job) -> dict[str, Any]:
        return self.write_event(
            event_type="JOB_STARTED",
            session_id=str(job.session_id),
            trace_id=str(getattr(job, "metadata", {}).get("trace_id") or ""),
            kernel_step_id="scheduler.execute.start",
            payload={"job_id": str(job.job_id)},
        )

    def append_job_completed(self, job, result) -> dict[str, Any]:
        return self.write_event(
            event_type="JOB_COMPLETED",
            session_id=str(job.session_id),
            trace_id=str(getattr(job, "metadata", {}).get("trace_id") or ""),
            kernel_step_id="scheduler.execute.complete",
            payload={"job_id": str(job.job_id), "result": result},
        )

    def append_job_failed(self, job, error: str) -> dict[str, Any]:
        return self.write_event(
            event_type="JOB_FAILED",
            session_id=str(job.session_id),
            trace_id=str(getattr(job, "metadata", {}).get("trace_id") or ""),
            kernel_step_id="scheduler.execute.fail",
            payload={"job_id": str(job.job_id), "error": str(error or "")},
        )
