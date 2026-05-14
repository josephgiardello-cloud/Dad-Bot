from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dadbot.core.runtime_errors import InvariantViolation
from dadbot.core.system_invariants import build_default_invariant_set
from dadbot.core.system_state_model import SystemHealthStatus, SystemStateSnapshot


@dataclass(frozen=True)
class TransitionBoundaryView:
    session_id: str
    trace_id: str
    before_state: str
    after_state: str
    before_causal_step_count: int
    after_causal_step_count: int
    turn_truth_ok: bool | None = None
    policy_posture: str = "moderate"
    active_fault_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class GlobalTransitionInvariantEnforcer:
    """Hard transition-boundary guard.

    This enforcer must run at every mutation boundary that can change runtime
    execution state (control-plane lifecycle transitions and SaveNode commit).
    """

    _VALID_EXECUTION_STATES = {
        "submitted",
        "queued",
        "recovery_pending",
        "running",
        "completed",
        "failed",
    }

    def __init__(self) -> None:
        self._system_invariants = build_default_invariant_set()

    def _snapshot_for_view(self, view: TransitionBoundaryView) -> SystemStateSnapshot:
        # Tool profile topology may be unavailable at boundary call-sites; we still
        # enforce global snapshot invariants over posture/fault count/health shape.
        health = SystemHealthStatus.HEALTHY
        if view.active_fault_count > 0:
            health = SystemHealthStatus.DEGRADED
        if view.active_fault_count >= 10:
            health = SystemHealthStatus.CRITICAL
        return SystemStateSnapshot(
            timestamp_ms=int(view.metadata.get("timestamp_ms") or 0),
            tool_profiles={},
            overall_health=health,
            active_fault_count=int(view.active_fault_count),
            policy_posture=str(view.policy_posture or "moderate"),
            metadata={
                "session_id": view.session_id,
                "trace_id": view.trace_id,
                "before_state": view.before_state,
                "after_state": view.after_state,
                **dict(view.metadata or {}),
            },
        )

    def enforce(self, view: TransitionBoundaryView) -> None:
        if view.after_causal_step_count < view.before_causal_step_count:
            raise InvariantViolation(
                "Transition invariant violated: causal_step_count must be monotonic",
                context={
                    "before": view.before_causal_step_count,
                    "after": view.after_causal_step_count,
                    "trace_id": view.trace_id,
                },
            )

        if view.before_state not in self._VALID_EXECUTION_STATES:
            raise InvariantViolation(
                "Transition invariant violated: unknown before_state",
                context={"before_state": view.before_state, "trace_id": view.trace_id},
            )

        if view.after_state not in self._VALID_EXECUTION_STATES:
            raise InvariantViolation(
                "Transition invariant violated: unknown after_state",
                context={"after_state": view.after_state, "trace_id": view.trace_id},
            )

        if view.after_state == "completed" and view.turn_truth_ok is None:
            raise InvariantViolation(
                "Transition invariant violated: completed transition requires turn_truth_ok",
                context={"trace_id": view.trace_id, "after_state": view.after_state},
            )

        snapshot = self._snapshot_for_view(view)
        violations = self._system_invariants.error_or_above(snapshot)
        if violations:
            first = violations[0]
            raise InvariantViolation(
                f"Global transition invariant violation: {first.name}",
                context={
                    "detail": first.detail,
                    "severity": first.severity.value,
                    "trace_id": view.trace_id,
                    "session_id": view.session_id,
                },
            )


_DEFAULT_ENFORCER = GlobalTransitionInvariantEnforcer()


def enforce_global_transition_invariants(view: TransitionBoundaryView) -> None:
    _DEFAULT_ENFORCER.enforce(view)


__all__ = [
    "TransitionBoundaryView",
    "GlobalTransitionInvariantEnforcer",
    "enforce_global_transition_invariants",
]
