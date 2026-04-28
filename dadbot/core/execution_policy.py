"""Execution policy module — declarative turn execution decisions.

All policy logic for turn execution lives here. TurnGraph imports from this
module and uses the policy objects as decision delegates; the graph itself
contains no policy logic.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal


class FatalTurnError(RuntimeError):
    """Unrecoverable turn invariant violation.

    Raised when core pipeline guarantees are violated (e.g., mutation queue
    cannot be fully drained or required stages did not execute).
    """


class TurnFailureSeverity(StrEnum):
    """Severity for standardized failure taxonomy."""

    RECOVERABLE = "recoverable"
    PARTIAL = "partial"
    FATAL = "fatal"
    COMPENSATABLE = "compensatable"


@dataclass(frozen=True)
class KernelRejectionSemantics:
    """Formal runtime contract for a kernel step rejection.

    Defaults are backward-compatible: rejected steps are skipped, produce no
    step-state mutation, and the pipeline continues.
    """

    retryable: bool = False
    state_mutation_allowed: bool = False
    invalidate_downstream: bool = False
    persistence_behavior: Literal["persist_rejection_event", "no_persist"] = "persist_rejection_event"
    action: Literal["skip_stage", "abort_turn"] = "skip_stage"


@dataclass(frozen=True)
class PersistenceServiceContract:
    """Versioned persistence surface expected by TurnGraph."""

    version: str = "1.0"
    save_turn: str = "save_turn"
    save_graph_checkpoint: str = "save_graph_checkpoint"
    save_turn_event: str = "save_turn_event"


class StagePhaseMappingPolicy:
    """Declarative stage-to-phase mapping.

    Returns canonical phase name strings (PLAN / ACT / OBSERVE / RESPOND).
    Callers convert the string to their domain-specific Phase enum.  All stage
    classification logic lives here — none in the graph executor.
    """

    _PLAN_STAGES: frozenset[str] = frozenset({"preflight", "health", "memory", "context", "plan", "temporal"})
    _ACT_STAGES: frozenset[str] = frozenset({"inference", "agent", "tool", "act"})
    _OBSERVE_STAGES: frozenset[str] = frozenset({"safety", "guard", "observe", "moderate", "moderation"})
    _RESPOND_STAGES: frozenset[str] = frozenset({"save", "respond", "final", "finalize", "persist"})

    @classmethod
    def phase_name_for_stage(cls, stage: str) -> str:
        """Return canonical phase name string, or empty string if stage is unknown."""
        lowered = str(stage or "").strip().lower()
        if lowered in cls._PLAN_STAGES:
            return "PLAN"
        if lowered in cls._ACT_STAGES:
            return "ACT"
        if lowered in cls._OBSERVE_STAGES:
            return "OBSERVE"
        if lowered in cls._RESPOND_STAGES:
            return "RESPOND"
        return ""

    @classmethod
    def all_stage_phase_map(cls) -> dict[str, str]:
        """Return complete stage→phase mapping for inspection/audit."""
        result: dict[str, str] = {}
        for stage in cls._PLAN_STAGES:
            result[stage] = "PLAN"
        for stage in cls._ACT_STAGES:
            result[stage] = "ACT"
        for stage in cls._OBSERVE_STAGES:
            result[stage] = "OBSERVE"
        for stage in cls._RESPOND_STAGES:
            result[stage] = "RESPOND"
        return result


@dataclass(frozen=True)
class ResumabilityPolicy:
    """Policy governing crash-safe turn resumption.

    Controls whether TurnGraph will skip already-completed stages on re-entry
    and how long resume records are considered valid.

    Attributes
    ----------
    enabled:
        Master switch.  When False, no resume points are written or read and
        every turn starts from scratch.  Default True.
    max_age_seconds:
        Resume records older than this are considered stale and discarded.
        Default 3600 (1 hour).  Set to 0 to disable age filtering.
    skip_completed_stages:
        When True (default), the graph skips stages listed in an active
        resume record.  Set False to disable stage-skipping while still
        writing records (e.g. for audit-only mode).
    """

    enabled: bool = True
    max_age_seconds: float = 3600.0
    skip_completed_stages: bool = True


class ExecutionPolicyEngine:
    """Centralized policy decisions for turn execution behavior.

    This keeps policy semantics entirely separate from DAG orchestration.
    """

    def __init__(self, *, persistence_contract: PersistenceServiceContract) -> None:
        self._persistence_contract = persistence_contract
        self._kernel_rejection_semantics: dict[str, KernelRejectionSemantics] = {
            "*": KernelRejectionSemantics(),
        }

    @staticmethod
    def _validate_callable_arity(callable_obj: Any, *, attr_name: str, required_arity: int) -> str | None:
        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return None

        parameters = list(signature.parameters.values())
        if any(
            parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for parameter in parameters
        ):
            return None

        positional_capacity = sum(
            1
            for parameter in parameters
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
        if positional_capacity < int(required_arity):
            return (
                f"{attr_name}: callable arity too small "
                f"(expected >= {required_arity} positional args, got {positional_capacity})"
            )
        return None

    def set_kernel_rejection_semantics(self, stage: str, semantics: KernelRejectionSemantics) -> None:
        key = str(stage or "*").strip().lower() or "*"
        self._kernel_rejection_semantics[key] = semantics

    def rejection_semantics_for_stage(self, stage: str) -> KernelRejectionSemantics:
        key = str(stage or "").strip().lower()
        if key in self._kernel_rejection_semantics:
            return self._kernel_rejection_semantics[key]
        return self._kernel_rejection_semantics.get("*", KernelRejectionSemantics())

    @staticmethod
    def classify_failure(error: Exception) -> TurnFailureSeverity:
        if isinstance(error, FatalTurnError):
            return TurnFailureSeverity.FATAL
        message = str(error or "").lower()
        if "timeout" in message:
            return TurnFailureSeverity.RECOVERABLE
        if "compensat" in message or "rollback" in message:
            return TurnFailureSeverity.COMPENSATABLE
        if "degraded" in message or "partial" in message:
            return TurnFailureSeverity.PARTIAL
        return TurnFailureSeverity.FATAL

    def validate_persistence_service_contract(self, service: Any, *, strict_mode: bool) -> dict[str, Any]:
        expected_version = str(self._persistence_contract.version or "")
        if service is None:
            payload: dict[str, Any] = {
                "version": expected_version,
                "ok": False,
                "missing": [
                    self._persistence_contract.save_turn,
                    self._persistence_contract.save_graph_checkpoint,
                    self._persistence_contract.save_turn_event,
                ],
                "backend_version": "",
                "compatible": False,
            }
            if strict_mode:
                raise RuntimeError("Persistence service unavailable")
            return payload

        missing: list[str] = []
        signature_issues: list[str] = []
        for attr in (
            self._persistence_contract.save_turn,
            self._persistence_contract.save_graph_checkpoint,
            self._persistence_contract.save_turn_event,
        ):
            candidate = getattr(service, attr, None)
            if not callable(candidate):
                missing.append(attr)
                continue

            required_arity = {
                self._persistence_contract.save_turn: 2,
                self._persistence_contract.save_graph_checkpoint: 1,
                self._persistence_contract.save_turn_event: 1,
            }.get(attr, 0)
            issue = self._validate_callable_arity(
                candidate,
                attr_name=attr,
                required_arity=required_arity,
            )
            if issue:
                signature_issues.append(issue)

        backend_version = str(
            getattr(service, "contract_version", None)
            or getattr(service, "version", None)
            or ""
        ).strip()

        compatible = True
        if backend_version:
            expected_major = expected_version.split(".", 1)[0]
            backend_major = backend_version.split(".", 1)[0]
            compatible = expected_major == backend_major

        payload = {
            "version": expected_version,
            "ok": len(missing) == 0 and len(signature_issues) == 0,
            "missing": list(missing),
            "signature_issues": list(signature_issues),
            "backend_version": backend_version,
            "compatible": bool(compatible),
        }

        if strict_mode and (missing or signature_issues or not compatible):
            raise RuntimeError(
                "Persistence service contract violation: "
                f"missing callables={missing!r}, signature_issues={signature_issues!r}, "
                f"contract_version={expected_version}, "
                f"backend_version={backend_version!r}, compatible={compatible}"
            )
        return payload
