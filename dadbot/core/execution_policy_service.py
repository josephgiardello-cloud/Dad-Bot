"""
StageEntryGate — stage-entry policy evaluation extracted from TurnGraph._execute_stage.

This module owns the decision logic for whether a pipeline stage is allowed to
enter execution.  It is pure policy: no I/O, no persistence, no LangGraph
coupling.  TurnGraph delegates to it from _execute_stage.

Responsibilities
----------------
- Idempotency / crash-recovery skip gate
- Capability enforcement skip gate
- Stage ordering + temporal pre-condition invariants
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dadbot.core.graph import TurnContext
    from dadbot.core.execution_recovery import ExecutionRecovery
    from dadbot.core.capability_registry import CapabilityRegistry

from dadbot.core.capability_registry import EnforcementMode, enforce_node_entry

logger = logging.getLogger(__name__)


class StageEntryGate:
    """Evaluates whether a pipeline stage is allowed to enter execution.

    Returns a skip flag for idempotency/capability decisions and raises
    RuntimeError for hard ordering violations.

    No state mutations beyond what is strictly necessary to track stage
    ordering (written on context.state, same contract as the original
    TurnGraph._mark_stage_enter staticmethod).
    """

    # Canonical pipeline ordering constraint map.
    # Only stages listed here are subject to ordering enforcement.
    _EXPECTED_NEXT: dict[str, set[str]] = {
        "": {"preflight", "health", "temporal", "context_builder"},
        "preflight": {"inference"},
        "health": {"context_builder"},
        "temporal": {"preflight", "health", "context_builder"},
        "context_builder": {"inference"},
        "inference": {"safety"},
        "safety": {"reflection"},
        "reflection": {"save"},
        "save": set(),
    }

    # Stages that require temporal to have already run.
    _REQUIRES_TEMPORAL: frozenset[str] = frozenset({"inference", "safety", "reflection", "save"})

    def check_recovery_skip(
        self,
        stage_name: str,
        context: "TurnContext",
        recovery: "ExecutionRecovery | None",
    ) -> bool:
        """Return True if the stage should be skipped (already completed in a prior run).

        No mutations.
        """
        if recovery is not None and recovery.is_already_completed(stage_name, context):
            logger.debug(
                "Idempotency guard: skipping already-completed stage %r for turn %r",
                stage_name,
                context.trace_id,
            )
            return True
        return False

    def check_capability_skip(
        self,
        stage_name: str,
        context: "TurnContext",
        registry: "CapabilityRegistry | None",
        policy: Any,
        session_id: str,
    ) -> bool:
        """Return True if capability enforcement says SKIP for this stage.

        No mutations.
        """
        if registry is None or policy is None:
            return False
        caps = policy.caps_for(session_id)
        enforcement = enforce_node_entry(
            stage_name,
            registry=registry,
            caps=caps,
            session_id=session_id,
        )
        return enforcement == EnforcementMode.SKIP

    def enforce_stage_ordering(self, context: "TurnContext", stage_name: str) -> None:
        """Enforce canonical stage ordering and temporal pre-conditions.

        Mutates context.state to track executed stages (same contract as the
        original TurnGraph._mark_stage_enter staticmethod).  Raises RuntimeError
        on violations.
        """
        executed = context.state.setdefault("_graph_executed_stages", set())
        if not isinstance(executed, set):
            executed = set(executed) if isinstance(executed, (list, tuple)) else set()
            context.state["_graph_executed_stages"] = executed

        if stage_name in executed:
            raise RuntimeError(
                f"TurnGraph execution violation: stage {stage_name!r} executed more than once "
                f"in trace {context.trace_id!r}"
            )

        last_stage = str(context.state.get("_graph_last_stage") or "").strip()
        _canonical = (
            set(self._EXPECTED_NEXT.keys())
            | {s for vals in self._EXPECTED_NEXT.values() for s in vals}
        )
        if stage_name in _canonical:
            allowed = self._EXPECTED_NEXT.get(last_stage)
            if allowed is not None and allowed and stage_name not in allowed:
                raise RuntimeError(
                    "TurnGraph order violation: "
                    f"stage {stage_name!r} cannot execute after {last_stage!r}; "
                    f"expected one of {sorted(allowed)!r}"
                )

        executed.add(stage_name)
        context.state["_graph_last_stage"] = stage_name
        context.state["_active_graph_stage"] = stage_name

        if stage_name in self._REQUIRES_TEMPORAL:
            if not context.state.get("temporal"):
                raise RuntimeError(
                    f"TemporalNode not initialized before {stage_name!r} — "
                    "deterministic execution violated: temporal must be first in pipeline"
                )
