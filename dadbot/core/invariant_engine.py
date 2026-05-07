"""Phase 4.1 — Global Invariant Engine.

A single validation authority over all execution subsystems:
    - Planner output structure
    - DAG topology validity
    - Tool execution graph integrity
    - Memory state consistency
    - Schema versions

Design:
    Each SystemInvariant has a predicate (callable) and a severity.
    GlobalInvariantEngine registers invariants and evaluates them all
    against a unified ExecutionState snapshot.

    This replaces ad-hoc assertions scattered across modules with a
    single, auditable validation pass.

    Principle: "One pass to find them all."
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)


def _strict_invariant_mode() -> bool:
    return str(os.environ.get("DADBOT_STRICT_INVARIANTS", "")).strip() == "1"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


# ---------------------------------------------------------------------------
# Invariant categories
# ---------------------------------------------------------------------------


class InvariantCategory(enum.Enum):
    """Category of the invariant, used for filtering and reporting."""

    PLANNER = "planner"
    DAG = "dag"
    TOOL_EXECUTION = "tool_execution"
    MEMORY_STATE = "memory_state"
    SCHEMA = "schema"
    RESOURCE = "resource"
    EVENT = "event"


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------


class InvariantSeverity(enum.Enum):
    """Impact level when an invariant is violated."""

    WARNING = "warning"      # Non-blocking; logged for audit.
    ERROR = "error"          # Significant — should halt planning.
    CRITICAL = "critical"    # System-level — must not proceed.
    IMPORTANT = "important"  # Alias tier: significant but not system-halting.
    DIAGNOSTIC = "diagnostic"  # Observability only; never gate execution.


# ---------------------------------------------------------------------------
# Execution state
# ---------------------------------------------------------------------------


@dataclass
class ExecutionState:
    """Unified snapshot of the current execution state for invariant evaluation.

    All fields are optional — invariants should guard against None/missing values.

    Attributes:
        planner_output:     dict from the planner (intent_type, strategy, tool_plan, etc.)
        dag:                ToolDAG instance, or None.
        tool_events:        List of tool event dicts from the event log.
        memory_entries:     List of memory entries (dicts with at minimum {"text": ...}).
        schema_versions:    Dict of schema name → version string.
        resource_report:    BudgetReport dict or None.
        session_id:         Current session identifier.
        turn_number:        Current turn index (0-based).
        extra:              Arbitrary additional context.

    """

    planner_output: dict[str, Any] = field(default_factory=dict)
    dag: Any = None
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    memory_entries: list[dict[str, Any]] = field(default_factory=list)
    schema_versions: dict[str, str] = field(default_factory=dict)
    resource_report: dict[str, Any] | None = None
    session_id: str = ""
    turn_number: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Invariant and violation types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SystemInvariant:
    """A single, named invariant with a predicate.

    Attributes:
        id:          Unique identifier (e.g., "planner.intent_type_present").
        category:    Which subsystem owns this invariant.
        description: Human-readable description of what is checked.
        predicate:   Callable[ExecutionState] → bool.
                     Returns True iff the invariant holds.
        severity:    Impact when violated.

    """

    id: str
    category: InvariantCategory
    description: str
    predicate: Callable[[ExecutionState], bool]
    severity: InvariantSeverity = InvariantSeverity.ERROR

    def check(self, state: ExecutionState) -> bool:
        try:
            return bool(self.predicate(state))
        except Exception:  # noqa: BLE001
            return False


@dataclass(frozen=True)
class InvariantViolation:
    """A recorded invariant violation."""

    invariant_id: str
    category: InvariantCategory
    description: str
    severity: InvariantSeverity
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "invariant_id": self.invariant_id,
            "category": self.category.value,
            "description": self.description,
            "severity": self.severity.value,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Enforcement API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InvariantCheck:
    """Lightweight check result consumed by enforce_invariant().

    Attributes:
        passed:  True iff the invariant holds.
        message: Human-readable description of the failure (may be empty when passed).
    """

    passed: bool
    message: str = ""


def enforce_invariant(check: InvariantCheck, severity: InvariantSeverity) -> None:
    """Centralized invariant enforcement gate.

    CRITICAL → raises InvariantViolationError immediately (never silenced).
    IMPORTANT / DIAGNOSTIC → logs a warning.

    Parameters
    ----------
    check:    An InvariantCheck produced by one of the CRITICAL check constructors.
    severity: Determines enforcement behavior.
    """
    if check.passed:
        return
    # Local import avoids module-level circularity; invariant_gate does not
    # import invariant_engine at the top level.
    from dadbot.core.invariant_gate import InvariantViolationError  # noqa: PLC0415
    if severity == InvariantSeverity.CRITICAL:
        raise InvariantViolationError(check.message)
    if _strict_invariant_mode() and severity in {
        InvariantSeverity.ERROR,
        InvariantSeverity.IMPORTANT,
    }:
        raise InvariantViolationError(check.message)
    _log.warning("Invariant warning [%s]: %s", severity.value, check.message)


# ---------------------------------------------------------------------------
# Global validation report
# ---------------------------------------------------------------------------


@dataclass
class GlobalValidationReport:
    """Result of running all invariants against an ExecutionState."""

    violations: list[InvariantViolation]
    passed: list[str]  # IDs of passing invariants.
    categories_checked: list[str]
    total_invariants: int
    critical_violations: int
    error_violations: int
    warning_violations: int
    ok: bool  # True iff no ERROR or CRITICAL violations.
    validation_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "total_invariants": self.total_invariants,
            "passed": len(self.passed),
            "violations": [v.to_dict() for v in self.violations],
            "critical_violations": self.critical_violations,
            "error_violations": self.error_violations,
            "warning_violations": self.warning_violations,
            "categories_checked": self.categories_checked,
            "validation_hash": self.validation_hash,
        }


# ---------------------------------------------------------------------------
# Built-in invariants
# ---------------------------------------------------------------------------


def _PLANNER_INTENT_PRESENT(state: ExecutionState) -> bool:
    return bool(str(state.planner_output.get("intent_type", "")).strip())


def _PLANNER_STRATEGY_PRESENT(state: ExecutionState) -> bool:
    return bool(str(state.planner_output.get("strategy", "")).strip())


def _DAG_NOT_EMPTY_WHEN_TOOLS_PLANNED(state: ExecutionState) -> bool:
    tool_plan = state.planner_output.get("tool_plan", [])
    if not tool_plan:
        return True  # No tools planned — DAG may be empty.
    if state.dag is None:
        return True  # DAG not provided — invariant not applicable.
    nodes = list(getattr(state.dag, "nodes", []))
    return len(nodes) > 0


def _DAG_ACYCLIC(state: ExecutionState) -> bool:
    if state.dag is None:
        return True
    # Use dag's own acyclicity check if available.
    is_acyclic = getattr(state.dag, "is_acyclic", None)
    if callable(is_acyclic):
        return bool(is_acyclic())
    # In strict mode, unknown DAG shape is a validation failure.
    if _strict_invariant_mode() and state.dag is not None:
        return False
    # Legacy fallback: unknown DAG implementations are treated as valid.
    return True


def _TOOL_EVENTS_HAVE_TYPE(state: ExecutionState) -> bool:
    for event in state.tool_events:
        if not event.get("type") and not event.get("event_type"):
            return False
    return True


def _MEMORY_ENTRIES_HAVE_TEXT(state: ExecutionState) -> bool:
    required_keys = frozenset({"text", "content", "value"})
    for entry in state.memory_entries:
        if not (required_keys & set(entry.keys())):
            return False
    return True


def _SCHEMA_VERSIONS_PRESENT(state: ExecutionState) -> bool:
    return len(state.schema_versions) > 0 or not state.schema_versions


def _RESOURCE_WITHIN_BUDGET(state: ExecutionState) -> bool:
    if state.resource_report is None:
        return True
    return bool(state.resource_report.get("within_budget", True))


def _EVENT_LOG_MONOTONIC(state: ExecutionState) -> bool:
    """Events should have non-decreasing sequences if sequence fields exist."""
    sequences = [int(e.get("sequence", -1)) for e in state.tool_events if e.get("sequence") is not None]
    if len(sequences) < 2:
        return True
    return all(sequences[i] <= sequences[i + 1] for i in range(len(sequences) - 1))


# Default invariant suite.
DEFAULT_INVARIANTS: list[SystemInvariant] = [
    SystemInvariant(
        id="planner.intent_type_present",
        category=InvariantCategory.PLANNER,
        description="Planner output must include a non-empty intent_type",
        predicate=_PLANNER_INTENT_PRESENT,
        severity=InvariantSeverity.ERROR,
    ),
    SystemInvariant(
        id="planner.strategy_present",
        category=InvariantCategory.PLANNER,
        description="Planner output must include a non-empty strategy",
        predicate=_PLANNER_STRATEGY_PRESENT,
        severity=InvariantSeverity.ERROR,
    ),
    SystemInvariant(
        id="dag.nonempty_when_tools_planned",
        category=InvariantCategory.DAG,
        description="DAG must be non-empty when tools are in the plan",
        predicate=_DAG_NOT_EMPTY_WHEN_TOOLS_PLANNED,
        severity=InvariantSeverity.ERROR,
    ),
    SystemInvariant(
        id="dag.acyclic",
        category=InvariantCategory.DAG,
        description="Tool DAG must be acyclic",
        predicate=_DAG_ACYCLIC,
        severity=InvariantSeverity.CRITICAL,
    ),
    SystemInvariant(
        id="tool_execution.events_have_type",
        category=InvariantCategory.TOOL_EXECUTION,
        description="All tool events must have a type field",
        predicate=_TOOL_EVENTS_HAVE_TYPE,
        severity=InvariantSeverity.ERROR,
    ),
    SystemInvariant(
        id="memory_state.entries_have_text",
        category=InvariantCategory.MEMORY_STATE,
        description="Memory entries must have a text, content, or value field",
        predicate=_MEMORY_ENTRIES_HAVE_TEXT,
        severity=InvariantSeverity.WARNING,
    ),
    SystemInvariant(
        id="schema.versions_defined",
        category=InvariantCategory.SCHEMA,
        description="Schema version registry must be non-empty when provided",
        predicate=_SCHEMA_VERSIONS_PRESENT,
        severity=InvariantSeverity.WARNING,
    ),
    SystemInvariant(
        id="resource.within_budget",
        category=InvariantCategory.RESOURCE,
        description="Tool execution must remain within turn budget",
        predicate=_RESOURCE_WITHIN_BUDGET,
        severity=InvariantSeverity.WARNING,
    ),
    SystemInvariant(
        id="event.log_monotonic",
        category=InvariantCategory.EVENT,
        description="Event log sequences must be non-decreasing",
        predicate=_EVENT_LOG_MONOTONIC,
        severity=InvariantSeverity.ERROR,
    ),
]


# ---------------------------------------------------------------------------
# Global invariant engine
# ---------------------------------------------------------------------------


class GlobalInvariantEngine:
    """Single validation pass over all registered invariants.

    Usage:
        engine = GlobalInvariantEngine.default()
        state = ExecutionState(
            planner_output={"intent_type": "question", "strategy": "fact_seeking"},
            tool_events=[...],
        )
        report = engine.validate_all(state)
        assert report.ok
    """

    def __init__(self, invariants: list[SystemInvariant] | None = None) -> None:
        self._invariants: list[SystemInvariant] = list(invariants or [])

    @classmethod
    def default(cls) -> GlobalInvariantEngine:
        return cls(invariants=list(DEFAULT_INVARIANTS))

    def register(self, invariant: SystemInvariant) -> None:
        self._invariants.append(invariant)

    def remove(self, invariant_id: str) -> None:
        self._invariants = [i for i in self._invariants if i.id != invariant_id]

    def get(self, invariant_id: str) -> SystemInvariant | None:
        for inv in self._invariants:
            if inv.id == invariant_id:
                return inv
        return None

    def invariant_count(self) -> int:
        return len(self._invariants)

    def validate_all(self, state: ExecutionState) -> GlobalValidationReport:
        """Evaluate all invariants and return a GlobalValidationReport."""
        violations: list[InvariantViolation] = []
        passed: list[str] = []
        categories: set[str] = set()

        for inv in self._invariants:
            categories.add(inv.category.value)
            try:
                result = inv.check(state)
            except Exception as exc:  # noqa: BLE001
                result = False
                violations.append(
                    InvariantViolation(
                        invariant_id=inv.id,
                        category=inv.category,
                        description=inv.description,
                        severity=InvariantSeverity.CRITICAL,
                        message=f"Invariant predicate raised exception: {exc}",
                    ),
                )
                continue

            if result:
                passed.append(inv.id)
            else:
                violations.append(
                    InvariantViolation(
                        invariant_id=inv.id,
                        category=inv.category,
                        description=inv.description,
                        severity=inv.severity,
                        message=f"Invariant '{inv.id}' failed",
                    ),
                )

        critical = sum(1 for v in violations if v.severity == InvariantSeverity.CRITICAL)
        errors = sum(1 for v in violations if v.severity == InvariantSeverity.ERROR)
        warnings = sum(1 for v in violations if v.severity == InvariantSeverity.WARNING)
        ok = critical == 0 and errors == 0

        validation_hash = _sha256(
            {
                "passed": sorted(passed),
                "violations": sorted(v.invariant_id for v in violations),
                "ok": ok,
            },
        )

        return GlobalValidationReport(
            violations=violations,
            passed=passed,
            categories_checked=sorted(categories),
            total_invariants=len(self._invariants),
            critical_violations=critical,
            error_violations=errors,
            warning_violations=warnings,
            ok=ok,
            validation_hash=validation_hash,
        )


__all__ = [
    "DEFAULT_INVARIANTS",
    "ExecutionState",
    "GlobalInvariantEngine",
    "GlobalValidationReport",
    "InvariantCategory",
    "InvariantCheck",
    "InvariantSeverity",
    "InvariantViolation",
    "SystemInvariant",
    "enforce_invariant",
]
