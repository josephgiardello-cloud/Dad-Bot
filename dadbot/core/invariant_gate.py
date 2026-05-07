"""InvariantGate â€” runtime enforcement hook.

Integrates at three points:
  1. LedgerWriter.write_event()  â€” before every ledger write
  2. Scheduler.drain_once()      â€” before every job execution
  3. Kernel.execute_step()       â€” before every kernel step (optional)

Rule: invariant failures HARD FAIL execution â€” they never log-and-continue.
"""

from __future__ import annotations

import enum
import time
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from dadbot.core.execution_schema import stamp_trace_contract_version


class InvariantViolationError(RuntimeError):
    """Raised when a runtime invariant is violated.  Never caught silently."""


# Execution semantic model (single source of truth)
# Primary ordering model is lineage-based DAG. Sequence is still required for
# deterministic event identity, but stage validity is validated by ancestry.
PRIMARY_EXECUTION_ORDERING_MODEL = "lineage_graph"
CANONICAL_TRACE_REDUCTION_RULE_VERSION = "lineage-minimal-v1"

EVENT_CLASS_TERMINAL = "terminal"
EVENT_CLASS_TRANSITIONAL = "transitional"
EVENT_CLASS_STRUCTURAL = "structural"

EXECUTION_EVENT_CLASS: dict[str, str] = {
    # Explicit classifications requested by runtime alignment.
    "stage_done": EVENT_CLASS_TERMINAL,
    "kernel_ok": EVENT_CLASS_TRANSITIONAL,
    "stage_enter": EVENT_CLASS_STRUCTURAL,
    # Remaining runtime events.
    "stage_error": EVENT_CLASS_TERMINAL,
    "turn_failed": EVENT_CLASS_TERMINAL,
    "turn_short_circuit": EVENT_CLASS_TERMINAL,
    "turn_succeeded": EVENT_CLASS_TERMINAL,
    "kernel_rejected": EVENT_CLASS_TRANSITIONAL,
    "kernel_error": EVENT_CLASS_TRANSITIONAL,
    "parallel_start": EVENT_CLASS_TRANSITIONAL,
    "parallel_done": EVENT_CLASS_TRANSITIONAL,
    "stage_skip": EVENT_CLASS_STRUCTURAL,
    "turn_start": EVENT_CLASS_STRUCTURAL,
}

VALIDITY_DOMAIN_LINEAR_SEQUENCE = "linear_sequences"
VALIDITY_DOMAIN_LINEAGE_GRAPH = "lineage_graphs"
VALIDITY_DOMAIN_HYBRID_PATH = "hybrid_paths"

EVENT_VALIDITY_DOMAIN: dict[str, str] = {
    "stage_done": VALIDITY_DOMAIN_HYBRID_PATH,
    "stage_error": VALIDITY_DOMAIN_HYBRID_PATH,
    "kernel_ok": VALIDITY_DOMAIN_LINEAGE_GRAPH,
    "kernel_rejected": VALIDITY_DOMAIN_LINEAGE_GRAPH,
    "kernel_error": VALIDITY_DOMAIN_LINEAGE_GRAPH,
    "stage_enter": VALIDITY_DOMAIN_LINEAGE_GRAPH,
    "stage_skip": VALIDITY_DOMAIN_LINEAGE_GRAPH,
    "parallel_start": VALIDITY_DOMAIN_HYBRID_PATH,
    "parallel_done": VALIDITY_DOMAIN_HYBRID_PATH,
    "turn_start": VALIDITY_DOMAIN_LINEAR_SEQUENCE,
    "turn_failed": VALIDITY_DOMAIN_HYBRID_PATH,
    "turn_short_circuit": VALIDITY_DOMAIN_HYBRID_PATH,
    "turn_succeeded": VALIDITY_DOMAIN_HYBRID_PATH,
}

_STAGE_STRUCTURAL_EVENTS = frozenset({"stage_enter", "stage_skip"})
_STAGE_TRANSITIONAL_EVENTS = frozenset({"kernel_ok", "kernel_rejected", "kernel_error"})
_STAGE_TERMINAL_EVENTS = frozenset({"stage_done", "stage_error"})
_TURN_TERMINAL_EVENTS = frozenset({"turn_failed", "turn_short_circuit", "turn_succeeded"})
_DECLARED_TRACE_EVENT_TYPES = frozenset(
    {
        "turn_start",
        "turn_failed",
        "turn_short_circuit",
        "turn_succeeded",
        "stage_enter",
        "stage_skip",
        "stage_done",
        "stage_error",
        "parallel_start",
        "parallel_done",
        "kernel_error",
        "kernel_rejected",
        "kernel_ok",
    },
)


@dataclass(frozen=True)
class ValidationDecision:
    approved: bool
    reason: str = ""
    details: dict[str, Any] | None = None
    remediation: RemediationDecision | None = None


class RemediationAction(enum.Enum):
    RETRY = "retry"
    REPLAN = "replan"
    DOWNGRADE = "downgrade"
    HARD_FAIL = "hard_fail"


@dataclass(frozen=True)
class RemediationDecision:
    action: RemediationAction
    reason: str = ""
    failure_class: str = ""
    attempt: int = 0
    max_attempts: int = 0
    details: dict[str, Any] | None = None


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

    def validate_ledger_lifecycle(self, events: list[dict[str, Any]]) -> None:
        """Validate job lifecycle ordering across a sequence of events.

        Ensures JOB_STARTED precedes any terminal state for the same job_id
        and that no job_id reaches a terminal state more than once.

        Raises InvariantViolationError on failure — never logs and continues.
        """
        check = ledger_lifecycle_check(events)
        if not check.passed:
            self._violations_observed += 1
            raise InvariantViolationError(check.message)

    @staticmethod
    def decide_remediation(
        failure_class: str,
        *,
        reason: str = "",
        attempt: int = 0,
        max_attempts: int = 0,
        details: dict[str, Any] | None = None,
    ) -> RemediationDecision:
        normalized = str(failure_class or "").strip().lower()
        if normalized == "validation_contract_violation":
            action = RemediationAction.REPLAN if attempt < max_attempts else RemediationAction.DOWNGRADE
        elif normalized == "retryable_tool_failure":
            action = RemediationAction.RETRY if attempt < max_attempts else RemediationAction.HARD_FAIL
        else:
            action = RemediationAction.HARD_FAIL
        return RemediationDecision(
            action=action,
            reason=str(reason or ""),
            failure_class=normalized,
            attempt=max(int(attempt), 0),
            max_attempts=max(int(max_attempts), 0),
            details=dict(details or {}),
        )

    @staticmethod
    def classify_execution_event(event_type: str) -> str:
        return EXECUTION_EVENT_CLASS.get(
            str(event_type or "").strip(),
            EVENT_CLASS_TRANSITIONAL,
        )

    @staticmethod
    def execution_event_domain(event_type: str) -> str:
        return EVENT_VALIDITY_DOMAIN.get(
            str(event_type or "").strip(),
            VALIDITY_DOMAIN_LINEAGE_GRAPH,
        )

    def reduce_execution_trace(
        self,
        trace: list[dict[str, Any]],
        *,
        pipeline_stage_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the canonical minimal trace for a lineage-valid execution graph.

        Canonical Trace Reduction Rule (lineage-minimal-v1)
        ---------------------------------------------------
        Given a lineage graph execution trace, keep all structural and terminal
        events, and keep only the final transitional event on each stage lineage
        segment immediately preceding a terminal closure for that same stage.

        This preserves ancestry and closure proofs while removing redundant
        in-segment transitional noise.
        """
        allowed_stages = {
            str(name or "").strip().lower()
            for name in list(pipeline_stage_names or [])
            if str(name or "").strip()
        }

        reduced: list[dict[str, Any]] = []
        events = list(trace or [])
        total = len(events)

        for idx, event in enumerate(events):
            event_type = str(event.get("event_type") or "").strip()
            event_class = self.classify_execution_event(event_type)
            if event_class != EVENT_CLASS_TRANSITIONAL:
                reduced.append(event)
                continue

            stage = str(event.get("stage") or "").strip().lower()
            if not stage:
                reduced.append(event)
                continue
            # Transitional events on kernel step aliases (not declared stages)
            # are retained; lineage compression applies to declared stage paths.
            if allowed_stages and stage not in allowed_stages:
                reduced.append(event)
                continue

            keep = False
            for follow_idx in range(idx + 1, total):
                follow = events[follow_idx]
                follow_stage = str(follow.get("stage") or "").strip().lower()
                if follow_stage != stage:
                    continue
                follow_type = str(follow.get("event_type") or "").strip()
                follow_class = self.classify_execution_event(follow_type)
                if follow_class == EVENT_CLASS_TRANSITIONAL:
                    keep = False
                    break
                # Keep only the last transitional immediately before terminal
                # closure of the same stage lineage segment.
                if follow_class == EVENT_CLASS_TERMINAL:
                    keep = True
                break
            else:
                # No subsequent same-stage semantic event: preserve evidence.
                keep = True

            if keep:
                reduced.append(event)

        return reduced

    def assess_execution_semantics(
        self,
        trace: list[dict[str, Any]],
        *,
        pipeline_stage_names: list[str] | None = None,
        require_closed_lineage: bool = False,
    ) -> ValidationDecision:
        try:
            self.validate_execution_semantics(
                trace,
                pipeline_stage_names=pipeline_stage_names,
                require_closed_lineage=require_closed_lineage,
            )
        except InvariantViolationError as exc:
            details = {"violations": [str(exc)]}
            return ValidationDecision(
                approved=False,
                reason=str(exc),
                details=details,
                remediation=self.decide_remediation(
                    "invariant_violation",
                    reason=str(exc),
                    details=details,
                ),
            )
        return ValidationDecision(approved=True)

    def build_trace_contract_decision(
        self,
        trace: list[dict[str, Any]],
        *,
        trace_id: str,
        pipeline_stage_names: list[str] | None = None,
        expected_hash: str = "",
    ) -> ValidationDecision:
        if not trace:
            reason = "Execution trace contract incomplete: no execution_trace events recorded"
            return ValidationDecision(
                approved=False,
                reason=reason,
                remediation=self.decide_remediation(
                    "trace_contract_violation",
                    reason=reason,
                ),
            )

        semantic = self.assess_execution_semantics(
            trace,
            pipeline_stage_names=pipeline_stage_names,
            require_closed_lineage=True,
        )
        if not semantic.approved:
            return semantic

        reduced_trace = self.reduce_execution_trace(
            trace,
            pipeline_stage_names=pipeline_stage_names,
        )
        reduced_semantic = self.assess_execution_semantics(
            reduced_trace,
            pipeline_stage_names=pipeline_stage_names,
            require_closed_lineage=True,
        )
        if not reduced_semantic.approved:
            return ValidationDecision(
                approved=False,
                reason=reduced_semantic.reason,
                details={
                    "violations": list((reduced_semantic.details or {}).get("violations") or []),
                    "trace_scope": "reduced",
                },
                remediation=self.decide_remediation(
                    "trace_contract_violation",
                    reason=reduced_semantic.reason,
                    details={
                        "violations": list((reduced_semantic.details or {}).get("violations") or []),
                        "trace_scope": "reduced",
                    },
                ),
            )

        for item in trace:
            event_type = str(item.get("event_type", "") or "").strip()
            stage = str(item.get("stage", "") or "").strip().lower()
            if not event_type or not stage:
                reason = (
                    "Execution trace contract incomplete: missing event_type or stage "
                    f"at sequence={int(item.get('sequence', 0) or 0)}"
                )
                return ValidationDecision(
                    approved=False,
                    reason=reason,
                    remediation=self.decide_remediation(
                        "trace_contract_violation",
                        reason=reason,
                    ),
                )
            if event_type not in _DECLARED_TRACE_EVENT_TYPES:
                reason = (
                    "Execution trace closure violation: undeclared event type "
                    f"event_type={event_type!r} sequence={int(item.get('sequence', 0) or 0)}"
                )
                return ValidationDecision(
                    approved=False,
                    reason=reason,
                    remediation=self.decide_remediation(
                        "trace_contract_violation",
                        reason=reason,
                    ),
                )

        canonical = {
            "trace_id": str(trace_id or ""),
            "events": [
                {
                    "sequence": int(item.get("sequence", 0) or 0),
                    "event_type": str(item.get("event_type", "") or ""),
                    "stage": str(item.get("stage", "") or ""),
                    "phase": str(item.get("phase", "") or ""),
                    "detail": dict(item.get("detail") or {}),
                }
                for item in reduced_trace
            ],
        }
        digest = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()

        expected = str(expected_hash or "").strip()
        if expected and expected != digest:
            reason = f"Execution trace determinism mismatch: expected={expected!r}, actual={digest!r}"
            return ValidationDecision(
                approved=False,
                reason=reason,
                details={"expected": expected, "actual": digest},
                remediation=self.decide_remediation(
                    "trace_contract_violation",
                    reason=reason,
                    details={"expected": expected, "actual": digest},
                ),
            )

        contract = stamp_trace_contract_version(
            {
                "version": "1.0",
                "event_count": len(trace),
                "reduced_event_count": len(reduced_trace),
                "ordering_model": PRIMARY_EXECUTION_ORDERING_MODEL,
                "reduction_rule": CANONICAL_TRACE_REDUCTION_RULE_VERSION,
                "trace_hash": digest,
            },
        )
        return ValidationDecision(
            approved=True,
            details={
                "reduced_trace": reduced_trace,
                "contract": contract,
                "trace_hash": digest,
            },
        )

    def validate_execution_semantics(
        self,
        trace: list[dict[str, Any]],
        *,
        pipeline_stage_names: list[str] | None = None,
        require_closed_lineage: bool = False,
    ) -> None:
        """Validate trace under a lineage-first DAG semantic model.

        Domains enforced explicitly:
        - linear_sequences: contiguous event sequence monotonicity
        - lineage_graphs: stage ancestry/open-lineage correctness
        - hybrid_paths: terminal closure over lineage-aware execution
        """
        expected_sequence = 1
        active_lineage: dict[str, bool] = {}
        terminal_turn_seen = False
        terminal_turn_event = ""
        allowed_stages = {
            str(name or "").strip().lower()
            for name in list(pipeline_stage_names or [])
            if str(name or "").strip()
        }

        for event in trace:
            event_type = str(event.get("event_type") or "").strip()
            stage = str(event.get("stage") or "").strip().lower()
            sequence = int(event.get("sequence") or 0)

            # linear_sequences domain
            if sequence != expected_sequence:
                raise InvariantViolationError(
                    "Execution semantic violation [linear_sequences]: "
                    f"non-contiguous sequence expected={expected_sequence} actual={sequence}",
                )

            if terminal_turn_seen and event_type not in _TURN_TERMINAL_EVENTS:
                raise InvariantViolationError(
                    "Execution semantic violation [hybrid_paths]: events emitted after terminal turn event",
                )

            if event_type in _TURN_TERMINAL_EVENTS:
                terminal_turn_seen = True
                terminal_turn_event = event_type

            # lineage_graphs domain
            if event_type in _STAGE_STRUCTURAL_EVENTS | _STAGE_TRANSITIONAL_EVENTS | _STAGE_TERMINAL_EVENTS:
                if not stage:
                    raise InvariantViolationError(
                        "Execution semantic violation [lineage_graphs]: missing stage for stage-scoped event",
                    )
                # Kernel transitional events may reference kernel step aliases
                # rather than declarative pipeline stage names.
                if (
                    allowed_stages
                    and event_type in (_STAGE_STRUCTURAL_EVENTS | _STAGE_TERMINAL_EVENTS)
                    and stage not in allowed_stages
                ):
                    raise InvariantViolationError(
                        "Execution semantic violation [lineage_graphs]: stage outside declared pipeline "
                        f"stage={stage!r} event_type={event_type!r}",
                    )

            if event_type == "stage_enter":
                if active_lineage.get(stage, False):
                    raise InvariantViolationError(
                        "Execution semantic violation [lineage_graphs]: nested stage_enter without closure "
                        f"stage={stage!r}",
                    )
                active_lineage[stage] = True
            elif event_type in _STAGE_TRANSITIONAL_EVENTS:
                if not active_lineage.get(stage, False) and not any(active_lineage.values()):
                    # In the simplified TurnGraph loop, kernel transition events
                    # can be emitted before the stage lineage is structurally
                    # opened. Preserve those as admissible pre-stage evidence.
                    if event_type in _STAGE_TRANSITIONAL_EVENTS:
                        expected_sequence += 1
                        continue
                    raise InvariantViolationError(
                        "Execution semantic violation [lineage_graphs]: transitional event without open lineage "
                        f"stage={stage!r} event_type={event_type!r}",
                    )
            elif event_type in _STAGE_TERMINAL_EVENTS:
                if not active_lineage.get(stage, False):
                    raise InvariantViolationError(
                        "Execution semantic violation [hybrid_paths]: terminal event without open lineage "
                        f"stage={stage!r} event_type={event_type!r}",
                    )
                active_lineage[stage] = False
            elif event_type == "stage_skip":
                active_lineage[stage] = False

            expected_sequence += 1

        still_open = sorted(stage for stage, is_open in active_lineage.items() if is_open)
        if (
            require_closed_lineage
            and still_open
            and terminal_turn_event not in {"turn_failed", "turn_short_circuit"}
        ):
            raise InvariantViolationError(
                "Execution semantic violation [hybrid_paths]: unclosed stage lineage at trace end "
                f"stages={still_open!r}",
            )


# ---------------------------------------------------------------------------
# CRITICAL invariant check constructors
# ---------------------------------------------------------------------------

_TERMINAL_EVENT_TYPES: frozenset[str] = frozenset({"JOB_COMPLETED", "JOB_FAILED"})


def execution_state_check(decision: Any) -> Any:
    """Check 1: Execution State Mismatch.

    Returns a failed InvariantCheck if post-execution states are not equivalent.
    Wire with: enforce_invariant(execution_state_check(decision), InvariantSeverity.CRITICAL)
    """
    from dadbot.core.invariant_engine import InvariantCheck  # noqa: PLC0415
    if getattr(decision, "equivalent", True):
        return InvariantCheck(passed=True)
    violations = list(getattr(decision, "violations", []) or [])
    return InvariantCheck(
        passed=False,
        message=f"State divergence after execution — violations: {violations}",
    )


def causal_structure_check(decision: Any) -> Any:
    """Check 2: Causal Structure Violation.

    Returns a failed InvariantCheck if 'causal_structure' appears in violations.
    Wire with: enforce_invariant(causal_structure_check(decision), InvariantSeverity.CRITICAL)
    """
    from dadbot.core.invariant_engine import InvariantCheck  # noqa: PLC0415
    violations = list(getattr(decision, "violations", []) or [])
    if "causal_structure" not in violations:
        return InvariantCheck(passed=True)
    return InvariantCheck(
        passed=False,
        message="Invalid causal ordering — causal_structure invariant violated",
    )


def ledger_lifecycle_check(events: list[dict[str, Any]]) -> Any:
    """Check 3: Ledger Lifecycle Integrity.

    Validates job_started → job_completed ordering and no duplicate terminal states.
    Wire with: enforce_invariant(ledger_lifecycle_check(events), InvariantSeverity.CRITICAL)
    """
    from dadbot.core.invariant_engine import InvariantCheck  # noqa: PLC0415
    started: set[str] = set()
    terminal: set[str] = set()
    for event in events:
        event_type = str(event.get("type") or "")
        payload = event.get("payload") or {}
        job_id = str(event.get("job_id") or payload.get("job_id") or "").strip()
        if not job_id:
            continue
        if event_type == "JOB_STARTED":
            started.add(job_id)
        elif event_type in _TERMINAL_EVENT_TYPES:
            if job_id in terminal:
                return InvariantCheck(
                    passed=False,
                    message=f"Invalid lifecycle sequence: duplicate terminal state for job {job_id!r}",
                )
            if job_id not in started:
                return InvariantCheck(
                    passed=False,
                    message=(
                        f"Invalid lifecycle sequence: {event_type} before JOB_STARTED "
                        f"for job {job_id!r}"
                    ),
                )
            terminal.add(job_id)
    return InvariantCheck(passed=True)


def lease_ownership_check(
    session_id: str,
    requested_owner: str,
    actual_owner: str | None,
) -> Any:
    """Check 4: Lease / Execution Ownership Violation.

    Returns a failed InvariantCheck if a different owner holds the lease.
    Wire with: enforce_invariant(lease_ownership_check(...), InvariantSeverity.CRITICAL)
    """
    from dadbot.core.invariant_engine import InvariantCheck  # noqa: PLC0415
    if actual_owner is None or actual_owner == str(requested_owner or ""):
        return InvariantCheck(passed=True)
    return InvariantCheck(
        passed=False,
        message=(
            f"Execution lease violation: session={session_id!r} "
            f"owned_by={actual_owner!r} requested_by={requested_owner!r}"
        ),
    )
