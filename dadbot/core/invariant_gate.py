"""InvariantGate â€” runtime enforcement hook.

Integrates at three points:
  1. LedgerWriter.write_event()  â€” before every ledger write
  2. Scheduler.drain_once()      â€” before every job execution
  3. Kernel.execute_step()       â€” before every kernel step (optional)

Rule: invariant failures HARD FAIL execution â€” they never log-and-continue.
"""

from __future__ import annotations

import time
from typing import Any


class InvariantViolationError(RuntimeError):
    """Raised when a runtime invariant is violated.  Never caught silently."""


# ---------------------------------------------------------------------------
# Built-in invariant checks
# ---------------------------------------------------------------------------


def _invariant_required_envelope_fields(event: dict[str, Any]) -> str | None:
    required = {"type", "session_id", "kernel_step_id"}
    missing = required - set(event.keys())
    if missing:
        return f"Event missing required envelope fields: {sorted(missing)}"
    return None


def _invariant_non_empty_type(event: dict[str, Any]) -> str | None:
    if not str(event.get("type") or "").strip():
        return "Event 'type' must be a non-empty string"
    return None


def _invariant_non_empty_session_id(event: dict[str, Any]) -> str | None:
    if not str(event.get("session_id") or "").strip():
        return "Event 'session_id' must be non-empty"
    return None


def _invariant_non_future_timestamp(event: dict[str, Any]) -> str | None:
    ts = event.get("timestamp")
    if ts is None:
        return None  # Missing timestamp is handled elsewhere
    try:
        ts_float = float(ts)
    except (TypeError, ValueError):
        return f"Event 'timestamp' is not a valid float: {ts!r}"
    drift_tolerance = 60.0  # seconds
    if ts_float > time.time() + drift_tolerance:
        return (
            f"Event timestamp {ts_float} is more than {drift_tolerance}s in the future â€” "
            f"clock skew or fabricated timestamp"
        )
    return None


def _invariant_payload_is_dict_or_absent(event: dict[str, Any]) -> str | None:
    payload = event.get("payload")
    if payload is not None and not isinstance(payload, dict):
        return f"Event 'payload' must be a dict or absent, got {type(payload).__name__}"
    return None


def _invariant_kernel_lineage_non_empty(event: dict[str, Any]) -> str | None:
    if not str(event.get("kernel_step_id") or "").strip():
        return "Event 'kernel_step_id' must be non-empty â€” kernel lineage required"
    return None


# ---------------------------------------------------------------------------
# Job execution invariants
# ---------------------------------------------------------------------------


def _job_invariant_session_not_terminated(
    session: dict[str, Any],
    job: Any,
) -> str | None:
    if str(session.get("status") or "active") == "terminated":
        return (
            f"Cannot execute job {getattr(job, 'job_id', '?')!r} â€” "
            f"session {getattr(job, 'session_id', '?')!r} is terminated"
        )
    return None


def _job_invariant_job_id_non_empty(session: dict[str, Any], job: Any) -> str | None:
    if not str(getattr(job, "job_id", "") or "").strip():
        return "job_id must be non-empty before execution"
    return None


def _job_invariant_session_id_non_empty(
    session: dict[str, Any],
    job: Any,
) -> str | None:
    if not str(getattr(job, "session_id", "") or "").strip():
        return "job.session_id must be non-empty before execution"
    return None


# ---------------------------------------------------------------------------
# InvariantGate
# ---------------------------------------------------------------------------


class InvariantGate:
    """Runtime invariant enforcement gate.

    Call validate_event() before every ledger write.
    Call validate_job() before every job execution.

    Both raise InvariantViolationError on any failure â€” never log and continue.
    """

    _DEFAULT_EVENT_CHECKS = [
        _invariant_required_envelope_fields,
        _invariant_non_empty_type,
        _invariant_non_empty_session_id,
        _invariant_non_future_timestamp,
        _invariant_payload_is_dict_or_absent,
        _invariant_kernel_lineage_non_empty,
    ]

    _DEFAULT_JOB_CHECKS = [
        _job_invariant_session_not_terminated,
        _job_invariant_job_id_non_empty,
        _job_invariant_session_id_non_empty,
    ]

    def __init__(
        self,
        *,
        extra_event_checks: list | None = None,
        extra_job_checks: list | None = None,
    ) -> None:
        self._event_checks = list(self._DEFAULT_EVENT_CHECKS) + list(
            extra_event_checks or [],
        )
        self._job_checks = list(self._DEFAULT_JOB_CHECKS) + list(extra_job_checks or [])
        self._violations_observed: int = 0

    def validate_event(self, event: dict[str, Any]) -> None:
        """Validate a ledger event envelope.  Raises InvariantViolationError on failure."""
        violations: list[str] = []
        for check in self._event_checks:
            result = check(event)
            if result is not None:
                violations.append(result)
        if violations:
            self._violations_observed += 1
            raise InvariantViolationError(
                f"Ledger write blocked by {len(violations)} invariant violation(s): " + "; ".join(violations),
            )

    def validate_job(self, session: dict[str, Any], job: Any) -> None:
        """Validate a job before execution.  Raises InvariantViolationError on failure."""
        violations: list[str] = []
        for check in self._job_checks:
            result = check(session, job)
            if result is not None:
                violations.append(result)
        if violations:
            self._violations_observed += 1
            raise InvariantViolationError(
                f"Job execution blocked by {len(violations)} invariant violation(s): " + "; ".join(violations),
            )

    @property
    def violations_observed(self) -> int:
        return self._violations_observed
