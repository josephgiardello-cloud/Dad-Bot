from __future__ import annotations

import time
from typing import Any

from dadbot.core.contracts.lifecycle_events import LifecycleEvent
from dadbot.core.execution_ledger import ExecutionLedger, WriteBoundaryGuard
from dadbot.core.execution_result_unified import ensure_unified_execution_result
from dadbot.core.invariant_gate import InvariantGate, InvariantViolationError
from dadbot.core.kernel_signals import get_metrics
from dadbot.core.runtime_contracts import validate_ledger_entry_contract


class LedgerWriter:
    """Canonical event writer for execution lifecycle events.

    Contract anchor: ledger_entry payloads are normalized through write_event/_write.
    """

    _COMMITTED_EVENT_TYPES = {
        "JOB_QUEUED",
        "JOB_COMPLETED",
        "JOB_FAILED",
        "JOB_RECONCILED",
        "EFFECT_COMMIT",
        "EFFECT_RECONCILED",
        "SESSION_STATE_UPDATED",
    }

    _SEMANTIC_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
        "EXECUTION_LIFECYCLE": ("event_type", "execution_id", "occurred_at"),
        "EFFECT_BEGIN": ("effect_id",),
        "EFFECT_COMMIT": ("effect_id",),
        "JOB_QUEUED": ("job_id",),
        "JOB_STARTED": ("job_id",),
        "JOB_COMPLETED": ("job_id",),
        "JOB_FAILED": ("job_id",),
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

    def _validate_semantic_payload(self, *, event_type: str, step_key: str, payload: dict[str, Any]) -> None:
        step = str(step_key or "").strip()
        if not step:
            raise ValueError("kernel_step_id required")
        required_fields = self._SEMANTIC_REQUIRED_FIELDS.get(str(event_type or "").strip().upper(), ())
        if not required_fields:
            return
        missing = [field for field in required_fields if not str(payload.get(field) or "").strip()]
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(
                f"semantic payload incomplete for {event_type}: missing required field(s): {missing_str}",
            )

    def _write(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(validate_ledger_entry_contract(payload))
        self._validate_semantic_payload(
            event_type=str(payload.get("type") or ""),
            step_key=str(payload.get("kernel_step_id") or ""),
            payload=dict(payload.get("payload") or {}),
        )
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
        step_key: str = "",
        trace_token: str = "",
        payload: dict[str, Any] | None = None,
        committed: bool = False,
        **legacy_kwargs: Any,
    ) -> dict[str, Any]:
        # Structural integrity: empty event_type is an invariant violation.
        # This check must fire BEFORE trace_id so InvariantViolationError
        # is raised for invalid structure rather than ValueError for lineage.
        if not str(event_type or "").strip():
            raise InvariantViolationError(
                "Event 'type' must be non-empty — structural invariant violation",
            )
        legacy_trace = legacy_kwargs.pop("trace_id", "")
        legacy_step = legacy_kwargs.pop("kernel_step_id", "")
        trace = str(trace_token or legacy_trace or "").strip()
        step = str(step_key or legacy_step or "").strip()
        if legacy_kwargs:
            unknown = ", ".join(sorted(str(name) for name in legacy_kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")
        if not trace:
            raise ValueError("trace_id required")
        self._validate_semantic_payload(
            event_type=str(event_type),
            step_key=step,
            payload=dict(payload or {}),
        )
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
        trace_token: str = "",
        step_key: str = "control_plane.bind_session",
        **legacy_kwargs: Any,
    ) -> dict[str, Any]:
        legacy_trace = legacy_kwargs.pop("trace_id", "")
        legacy_step = legacy_kwargs.pop("kernel_step_id", "")
        if legacy_kwargs:
            unknown = ", ".join(sorted(str(name) for name in legacy_kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")
        resolved_trace = str(trace_token or legacy_trace or "").strip() or self._infer_trace_id_for_job(
            session_id,
            job_id,
        )
        resolved_step = str(step_key or legacy_step or "").strip()
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

    @staticmethod
    def _execution_result_failure(payload: dict[str, Any]) -> dict[str, Any]:
        execution_result = ensure_unified_execution_result(
            dict(payload.get("metadata", {}).get("execution_result") or {}),
        )
        return dict(execution_result.get("failure") or {})

    @staticmethod
    def _has_failure_details(failure_view: dict[str, Any]) -> bool:
        return bool(
            str(failure_view.get("class") or "")
            or str(failure_view.get("type") or "")
            or str(failure_view.get("message") or ""),
        )

    @staticmethod
    def _normalized_failure_payload(failure_view: dict[str, Any]) -> dict[str, Any]:
        return {
            "failure_class": str(failure_view.get("class") or ""),
            "failure_source": str(failure_view.get("source") or ""),
            "retryable": bool(failure_view.get("retryable", False)),
            "error_type": str(failure_view.get("type") or ""),
            "message": str(failure_view.get("message") or ""),
            "class": str(failure_view.get("class") or ""),
            "source": str(failure_view.get("source") or ""),
            "type": str(failure_view.get("type") or ""),
        }

    @staticmethod
    def _legacy_failure_payload(failure: dict[str, Any] | None) -> dict[str, Any] | None:
        if isinstance(failure, dict) and bool(failure):
            return dict(failure)
        return None

    def append_job_failed(
        self,
        job: Any,
        error: Any,
        *,
        failure: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._job_payload(job)
        payload["error"] = str(error or "")
        payload["error_type"] = type(error).__name__ if isinstance(error, BaseException) else ""
        failure_view = self._execution_result_failure(payload)
        if self._has_failure_details(failure_view):
            payload["failure"] = self._normalized_failure_payload(failure_view)
        else:
            legacy_failure = self._legacy_failure_payload(failure)
            if legacy_failure is None:
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
            # Legacy fallback for older callers; unified execution_result is authoritative.
            payload["failure"] = legacy_failure
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
        trace_token: str = "",
        session_id: str = "",
        **legacy_kwargs: Any,
    ) -> dict[str, Any]:
        legacy_trace = legacy_kwargs.pop("trace_id", "")
        if legacy_kwargs:
            unknown = ", ".join(sorted(str(name) for name in legacy_kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")
        trace = str(trace_token or legacy_trace or "").strip()
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

    def append_execution_lifecycle(
        self,
        event: LifecycleEvent,
        *,
        session_id: str,
        trace_token: str = "",
        step_key: str = "",
        committed: bool = False,
        **legacy_kwargs: Any,
    ) -> dict[str, Any]:
        legacy_trace = legacy_kwargs.pop("trace_id", "")
        legacy_step = legacy_kwargs.pop("kernel_step_id", "")
        if legacy_kwargs:
            unknown = ", ".join(sorted(str(name) for name in legacy_kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")
        return self._write(
            {
                "type": "EXECUTION_LIFECYCLE",
                "session_id": str(session_id or "default"),
                "trace_id": str(trace_token or legacy_trace or "").strip(),
                "kernel_step_id": str(step_key or legacy_step or "control_plane.lifecycle"),
                "payload": event.to_payload(),
                "committed": bool(committed),
                "timestamp": time.time(),
            },
        )

    def append_effect_begin(
        self,
        *,
        session_id: str,
        trace_token: str = "",
        effect_id: str,
        request_id: str = "",
        step_key: str = "scheduler.execute.effect.begin",
        **legacy_kwargs: Any,
    ) -> dict[str, Any]:
        legacy_trace = legacy_kwargs.pop("trace_id", "")
        legacy_step = legacy_kwargs.pop("kernel_step_id", "")
        if legacy_kwargs:
            unknown = ", ".join(sorted(str(name) for name in legacy_kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")
        return self._write(
            {
                "type": "EFFECT_BEGIN",
                "session_id": str(session_id or "default"),
                "trace_id": str(trace_token or legacy_trace or "").strip(),
                "kernel_step_id": str(step_key or legacy_step or "scheduler.execute.effect.begin"),
                "payload": {
                    "effect_id": str(effect_id or "").strip(),
                    "request_id": str(request_id or "").strip(),
                },
                "committed": False,
                "timestamp": time.time(),
            },
        )

    def append_effect_commit(
        self,
        *,
        session_id: str,
        trace_token: str = "",
        effect_id: str,
        request_id: str = "",
        step_key: str = "scheduler.execute.effect.commit",
        **legacy_kwargs: Any,
    ) -> dict[str, Any]:
        legacy_trace = legacy_kwargs.pop("trace_id", "")
        legacy_step = legacy_kwargs.pop("kernel_step_id", "")
        if legacy_kwargs:
            unknown = ", ".join(sorted(str(name) for name in legacy_kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")
        return self._write(
            {
                "type": "EFFECT_COMMIT",
                "session_id": str(session_id or "default"),
                "trace_id": str(trace_token or legacy_trace or "").strip(),
                "kernel_step_id": str(step_key or legacy_step or "scheduler.execute.effect.commit"),
                "payload": {
                    "effect_id": str(effect_id or "").strip(),
                    "request_id": str(request_id or "").strip(),
                },
                "committed": True,
                "timestamp": time.time(),
            },
        )
