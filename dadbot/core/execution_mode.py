"""Unified execution mode resolution and recovery state management.

Consolidates execution mode handling from multiple fragmented code paths into a
single canonical resolver. Provides mode-aware branching at well-defined decision points.

Execution Modes:
- LIVE: Normal first attempt execution
- RECOVERY: Redelivery after failure, restores state from checkpoint
- REPLAY: Historical replay (deterministic, read-only)
- DEGRADED: Graph degradation fallback to legacy processing

This module ensures:
1. Execution mode resolved exactly once per turn, at entry point
2. Recovery restoration called exactly once, with full state hydration
3. Mode-aware branching only at canonical decision points
4. No dual resolution paths or inconsistent recovery handling
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ExecutionMode(Enum):
    """Canonical execution mode enumeration."""

    LIVE = "live"
    RECOVERY = "recovery"
    REPLAY = "replay"
    DEGRADED = "degraded"

    def __str__(self) -> str:
        return self.value


@dataclass
class ExecutionModeContext:
    """Resolved execution mode with supporting context."""

    mode: ExecutionMode
    is_redelivery: bool
    checkpoint_available: bool
    redelivery_count: int


class ExecutionModeResolver:
    """Unified execution mode resolver and recovery state manager.

    Called exactly once per turn execution at the orchestration entry point.
    Centralizes mode resolution from previously fragmented code paths.
    """

    @staticmethod
    def resolve(
        job: Any,
        checkpoint: dict[str, Any] | None,
    ) -> ExecutionModeContext:
        """Resolve execution mode and supporting context for a job.

        Args:
            job: ExecutionJob with metadata and execution_state
            checkpoint: Optional checkpoint dict with execution history

        Returns:
            ExecutionModeContext with resolved mode and context flags

        Contract:
        - Called exactly once at execution entry (orchestrator._prepare_execution_mode_from_checkpoint)
        - Explicit mode in metadata takes precedence if not "live"
        - Recovery mode triggered by: redelivery_count > 0 or lifecycle_state == "recovery_pending"
        - Degraded mode never set here; only set by graph exception handler
        """
        metadata = dict(getattr(job, "metadata", None) or {})
        execution_state = dict(metadata.get("execution_state") or {})

        # Extract explicit mode (highest priority)
        explicit_mode_str = str(metadata.get("execution_mode") or "").strip().lower()
        explicit_mode = _coerce_execution_mode_str(explicit_mode_str) if explicit_mode_str else None

        # If explicit mode set (and not live), use it
        if explicit_mode and explicit_mode != ExecutionMode.LIVE:
            return ExecutionModeContext(
                mode=explicit_mode,
                is_redelivery=False,
                checkpoint_available=isinstance(checkpoint, dict) and bool(checkpoint),
                redelivery_count=0,
            )

        # Check recovery conditions
        # Runtime invariant (Correctness Matrix #1): lifecycle state reads that influence decisions
        # must be projection-derived from ledger events.
        lifecycle_state = ""
        if bool(execution_state.get("_derived_from_ledger", False)):
            lifecycle_state = str(execution_state.get("lifecycle_state") or "").strip().lower()
        redelivery_count = int(execution_state.get("redelivery_count") or 0)

        if redelivery_count > 0 or lifecycle_state == "recovery_pending":
            return ExecutionModeContext(
                mode=ExecutionMode.RECOVERY,
                is_redelivery=True,
                checkpoint_available=isinstance(checkpoint, dict) and bool(checkpoint),
                redelivery_count=redelivery_count,
            )

        # Default to live
        return ExecutionModeContext(
            mode=ExecutionMode.LIVE,
            is_redelivery=False,
            checkpoint_available=isinstance(checkpoint, dict) and bool(checkpoint),
            redelivery_count=0,
        )

    @staticmethod
    def resolve_for_logging(job: Any) -> str:
        """Simplified resolve for logging/telemetry (no checkpoint available).

        Used in control plane lifecycle events where checkpoint isn't accessible.
        Returns string representation for backwards compatibility.
        """
        context = ExecutionModeResolver.resolve(job, checkpoint=None)
        return str(context.mode)

    @staticmethod
    def mark_degraded(metadata: dict[str, Any]) -> None:
        """Mark execution as degraded after graph exception.

        Called by graph exception handlers after recovery attempts fail.
        Sets execution_mode to "degraded" in metadata for telemetry and UI.

        Args:
            metadata: Job metadata dict to update (mutated in-place)
        """
        metadata["execution_mode"] = ExecutionMode.DEGRADED.value


def _coerce_execution_mode_str(value: str) -> ExecutionMode | None:
    """Coerce string to ExecutionMode, returning None if invalid."""
    if not value:
        return None
    value_lower = str(value).strip().lower()
    try:
        return ExecutionMode(value_lower)
    except ValueError:
        return None


# Backwards compatibility exports (for gradual migration)
def _resolved_execution_mode(job: Any) -> str:
    """Backwards compat: equivalent to old control_plane._resolved_execution_mode."""
    return ExecutionModeResolver.resolve_for_logging(job)


def resolve_execution_mode(job: Any, checkpoint: dict[str, Any] | None) -> str:
    """Backwards compat: equivalent to old orchestrator resolve_execution_mode."""
    context = ExecutionModeResolver.resolve(job, checkpoint)
    return str(context.mode)
