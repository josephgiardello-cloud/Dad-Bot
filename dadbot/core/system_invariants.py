"""System-Wide Invariants — Formal Constraint Checking.

Gap 2 of the remaining architecture: the system is now coherent at subsystem
level, but lacks system-wide invariants that *guarantee* no pathological state
is possible.

This module moves the system from "well-controlled" to "provably constrained"
by expressing constraints as executable predicates that can be checked at any
decision point.

Core types
----------
SystemInvariant      — A named, runnable predicate on SystemStateSnapshot
InvariantViolation   — A recorded constraint breach with severity and detail
SystemInvariantSet   — A composable collection of invariants
build_default_invariant_set() — The standard Dad-Bot constraint set (7 invariants)

Standard invariants
-------------------
  TOOL_HEALTH_ABORT_CONSISTENCY   — healthy tool must have abort_rate < 0.5
  CRITICAL_HEALTH_FAULT_FLOOR     — CRITICAL health requires active_fault_count > 0
  HEALTHY_NO_DEGRADED_TOOLS       — HEALTHY state allows no degraded tools
  POLICY_POSTURE_VALID            — policy_posture must be a known value
  CAUSAL_GRAPH_NO_CYCLES          — causal graph must be a proper DAG
  NO_UNKNOWN_HEALTH_WITH_TOOLS    — UNKNOWN health is only valid with no profiles
  ACTIVE_FAULT_COUNT_NON_NEGATIVE — active_fault_count must not be negative

Usage
-----
    inv_set = build_default_invariant_set()
    violations = inv_set.check(snapshot)
    if violations:
        for v in violations:
            print(v.name, v.severity.value, v.detail)
    assert inv_set.is_valid(snapshot)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from dadbot.core.system_state_model import SystemHealthStatus, SystemStateSnapshot


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------


class InvariantSeverity(Enum):
    WARNING = "warning"
    """Violation is notable but does not block operation."""

    ERROR = "error"
    """Violation indicates a real inconsistency that should be corrected."""

    FATAL = "fatal"
    """Violation indicates a state that must never be allowed to persist."""


# ---------------------------------------------------------------------------
# Violation record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InvariantViolation:
    """Records a single constraint violation.

    Attributes
    ----------
    name:
        The invariant name that was violated.
    description:
        What the invariant was supposed to enforce.
    severity:
        How serious the violation is.
    snapshot_timestamp_ms:
        Timestamp of the snapshot when the violation was detected.
    detail:
        Additional context about why the predicate failed.
    """

    name: str
    description: str
    severity: InvariantSeverity
    snapshot_timestamp_ms: int
    detail: str = ""

    @property
    def is_fatal(self) -> bool:
        return self.severity == InvariantSeverity.FATAL

    @property
    def is_error_or_above(self) -> bool:
        return self.severity in {InvariantSeverity.ERROR, InvariantSeverity.FATAL}


# ---------------------------------------------------------------------------
# Invariant
# ---------------------------------------------------------------------------


@dataclass
class SystemInvariant:
    """A named, executable constraint on SystemStateSnapshot.

    The predicate must return (is_valid: bool, detail: str).
    If is_valid is False, a violation is recorded with the returned detail.

    Predicate exceptions are caught and reported as ERROR violations so the
    invariant check itself can never crash the system.
    """

    name: str
    description: str
    severity: InvariantSeverity
    predicate: Callable[[SystemStateSnapshot], tuple[bool, str]]

    def check(self, snapshot: SystemStateSnapshot) -> InvariantViolation | None:
        """Check this invariant against a snapshot.

        Returns an InvariantViolation if the predicate fails, else None.
        """
        try:
            is_valid, detail = self.predicate(snapshot)
        except Exception as exc:
            return InvariantViolation(
                name=self.name,
                description=self.description,
                severity=InvariantSeverity.ERROR,
                snapshot_timestamp_ms=snapshot.timestamp_ms,
                detail=f"predicate raised exception: {exc}",
            )
        if is_valid:
            return None
        return InvariantViolation(
            name=self.name,
            description=self.description,
            severity=self.severity,
            snapshot_timestamp_ms=snapshot.timestamp_ms,
            detail=detail,
        )


# ---------------------------------------------------------------------------
# Invariant set
# ---------------------------------------------------------------------------


class SystemInvariantSet:
    """A composable, batch-checkable collection of SystemInvariant objects."""

    def __init__(self) -> None:
        self._invariants: list[SystemInvariant] = []

    def add(self, invariant: SystemInvariant) -> None:
        """Register an invariant."""
        self._invariants.append(invariant)

    def check(self, snapshot: SystemStateSnapshot) -> list[InvariantViolation]:
        """Run all registered invariants against the snapshot.

        Returns all violations found (empty list = all constraints satisfied).
        """
        return [
            v for inv in self._invariants
            if (v := inv.check(snapshot)) is not None
        ]

    def is_valid(self, snapshot: SystemStateSnapshot) -> bool:
        """True when no invariants are violated."""
        return not self.check(snapshot)

    def fatal_violations(self, snapshot: SystemStateSnapshot) -> list[InvariantViolation]:
        """Return only FATAL violations."""
        return [v for v in self.check(snapshot) if v.is_fatal]

    def error_or_above(self, snapshot: SystemStateSnapshot) -> list[InvariantViolation]:
        """Return ERROR and FATAL violations."""
        return [v for v in self.check(snapshot) if v.is_error_or_above]

    def __len__(self) -> int:
        return len(self._invariants)


# ---------------------------------------------------------------------------
# Standard invariant predicates
# ---------------------------------------------------------------------------

_VALID_POLICY_POSTURES: frozenset[str] = frozenset({"aggressive", "moderate", "lenient"})


def _inv_tool_health_abort_consistency(snapshot: SystemStateSnapshot) -> tuple[bool, str]:
    """A tool classified as healthy must not have abort_rate >= 0.5."""
    violations: list[str] = []
    for name, profile in snapshot.tool_profiles.items():
        if profile.is_healthy and profile.total_executions > 0:
            abort_rate = profile.abort_count / profile.total_executions
            if abort_rate >= 0.5:
                violations.append(
                    f"{name}: is_healthy=True but abort_rate={abort_rate:.2f} >= 0.5"
                )
    return (not violations), "; ".join(violations)


def _inv_critical_health_fault_floor(snapshot: SystemStateSnapshot) -> tuple[bool, str]:
    """CRITICAL health must be accompanied by at least one active fault."""
    if (
        snapshot.overall_health == SystemHealthStatus.CRITICAL
        and snapshot.active_fault_count == 0
        and snapshot.total_tool_count() > 0
    ):
        return False, "health=CRITICAL but active_fault_count=0 with tool profiles present"
    return True, ""


def _inv_healthy_no_degraded_tools(snapshot: SystemStateSnapshot) -> tuple[bool, str]:
    """HEALTHY overall health must mean no individual tool is degraded."""
    if snapshot.overall_health == SystemHealthStatus.HEALTHY:
        degraded = snapshot.degraded_tools()
        if degraded:
            return False, f"health=HEALTHY but degraded tools exist: {degraded}"
    return True, ""


def _inv_policy_posture_valid(snapshot: SystemStateSnapshot) -> tuple[bool, str]:
    """policy_posture must be one of the known values."""
    if snapshot.policy_posture not in _VALID_POLICY_POSTURES:
        return (
            False,
            f"policy_posture={snapshot.policy_posture!r}; "
            f"valid values: {sorted(_VALID_POLICY_POSTURES)}",
        )
    return True, ""


def _inv_causal_graph_no_cycles(snapshot: SystemStateSnapshot) -> tuple[bool, str]:
    """The causal graph (if present) must be a proper DAG — no cycles."""
    if snapshot.causal_graph is None:
        return True, ""
    cycles = snapshot.causal_graph.detect_cycles()
    if cycles:
        sample = "; ".join(str(c) for c in cycles[:3])
        return False, f"{len(cycles)} cycle(s) detected: {sample}"
    return True, ""


def _inv_no_unknown_health_with_tools(snapshot: SystemStateSnapshot) -> tuple[bool, str]:
    """UNKNOWN health is only valid when no tool profiles exist."""
    if (
        snapshot.overall_health == SystemHealthStatus.UNKNOWN
        and snapshot.total_tool_count() > 0
    ):
        return (
            False,
            f"health=UNKNOWN but {snapshot.total_tool_count()} tool profile(s) exist; "
            "health should be computable",
        )
    return True, ""


def _inv_active_fault_count_non_negative(snapshot: SystemStateSnapshot) -> tuple[bool, str]:
    """active_fault_count must not be negative."""
    if snapshot.active_fault_count < 0:
        return False, f"active_fault_count={snapshot.active_fault_count} is negative"
    return True, ""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_default_invariant_set() -> SystemInvariantSet:
    """Build the standard Dad-Bot system invariant set.

    These constraints must hold at every decision epoch.
    """
    inv_set = SystemInvariantSet()

    inv_set.add(SystemInvariant(
        name="TOOL_HEALTH_ABORT_CONSISTENCY",
        description="A tool marked healthy must not have abort_rate >= 0.5",
        severity=InvariantSeverity.ERROR,
        predicate=_inv_tool_health_abort_consistency,
    ))
    inv_set.add(SystemInvariant(
        name="CRITICAL_HEALTH_FAULT_FLOOR",
        description="CRITICAL system health requires at least one active fault",
        severity=InvariantSeverity.WARNING,
        predicate=_inv_critical_health_fault_floor,
    ))
    inv_set.add(SystemInvariant(
        name="HEALTHY_NO_DEGRADED_TOOLS",
        description="HEALTHY system state allows no degraded tools",
        severity=InvariantSeverity.ERROR,
        predicate=_inv_healthy_no_degraded_tools,
    ))
    inv_set.add(SystemInvariant(
        name="POLICY_POSTURE_VALID",
        description="policy_posture must be one of: aggressive, moderate, lenient",
        severity=InvariantSeverity.ERROR,
        predicate=_inv_policy_posture_valid,
    ))
    inv_set.add(SystemInvariant(
        name="CAUSAL_GRAPH_NO_CYCLES",
        description="The causal dependency graph must be a proper DAG",
        severity=InvariantSeverity.FATAL,
        predicate=_inv_causal_graph_no_cycles,
    ))
    inv_set.add(SystemInvariant(
        name="NO_UNKNOWN_HEALTH_WITH_TOOLS",
        description="UNKNOWN health is only valid when no tool profiles exist",
        severity=InvariantSeverity.WARNING,
        predicate=_inv_no_unknown_health_with_tools,
    ))
    inv_set.add(SystemInvariant(
        name="ACTIVE_FAULT_COUNT_NON_NEGATIVE",
        description="active_fault_count must not be negative",
        severity=InvariantSeverity.FATAL,
        predicate=_inv_active_fault_count_non_negative,
    ))

    return inv_set


__all__ = [
    "InvariantSeverity",
    "InvariantViolation",
    "SystemInvariant",
    "SystemInvariantSet",
    "build_default_invariant_set",
]
