from __future__ import annotations

from typing import Any

from dadbot.core.memory_set_invariants import MemorySetInvariantViolation
from dadbot.core.system_state_algebra import (
    evaluate_system_state_algebra,
    persist_system_state_algebra,
)


class InvariantEnforcer:
    """Enforces commit-time invariant gates for terminal turn truth."""

    @staticmethod
    def _resolve_terminal_turn_truth(*, state: dict[str, Any], execution_result: dict[str, Any], trace_id: str) -> dict[str, Any]:
        algebra = evaluate_system_state_algebra(
            state=state,
            execution_result_payload=dict(execution_result or {}),
            trace_token=str(trace_id or ""),
            context="control_plane_commit",
        )
        return dict(algebra)

    def enforce_global_turn_invariant_gate(
        self,
        *,
        session: dict[str, Any],
        execution_result: dict[str, Any],
        trace_id: str,
    ) -> None:
        state = session.get("state")
        if not isinstance(state, dict):
            return

        algebra = self._resolve_terminal_turn_truth(
            state=state,
            execution_result=dict(execution_result or {}),
            trace_id=str(trace_id or ""),
        )
        persist_system_state_algebra(
            state=state,
            algebra=algebra,
            trace_context="control_plane_commit",
            persist_legacy_projections=True,
            terminal_snapshot=True,
        )
        gate = dict(algebra.get("projections", {}).get("control_plane_gate") or {})

        if not bool(gate.get("ok", False)):
            raise MemorySetInvariantViolation(
                "Global invariant gate violation: "
                + "; ".join(str(item) for item in list(gate.get("violations") or [])[:3]),
            )
