from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dadbot.core.execution_firewall import ExecutionBlockedInvariantError

INVARIANTS: dict[str, bool] = {
    "temporal_required": True,
    "save_node_required": True,
    "no_legacy_paths": True,
    "mutation_only_via_save": True,
}


@dataclass
class InvariantCheckResult:
    ok: bool
    reason: str = ""


def _has_temporal(turn_context: Any) -> bool:
    temporal = getattr(turn_context, "temporal", None)
    if temporal is None:
        return False
    wall_time = str(getattr(temporal, "wall_time", "") or "").strip()
    wall_date = str(getattr(temporal, "wall_date", "") or "").strip()
    return bool(wall_time and wall_date)


def _stage_name(stage: str | None) -> str:
    return str(stage or "").strip().lower()


class InvariantRegistry:
    """Single authority for structural runtime invariants.

    This registry validates correctness rules. Firewall/quarantine decisions are
    enforced separately by the execution firewall layer.
    """

    def __init__(self, invariants: dict[str, bool] | None = None) -> None:
        self.rules = dict(INVARIANTS if invariants is None else invariants)

    def check_invariants(
        self,
        turn_context: Any,
        *,
        stage: str = "",
        operation: str = "",
        mutation_outside_save_node: bool = False,
    ) -> InvariantCheckResult:
        """Centralized invariant enforcement gate for Phase 4 strict runtime.

        Raises:
            ExecutionBlockedInvariantError when any enforced invariant is violated.

        """
        stage_name = _stage_name(stage)
        temporal_missing = self.rules.get(
            "temporal_required",
            True,
        ) and not _has_temporal(turn_context)

        if temporal_missing:
            raise ExecutionBlockedInvariantError(
                f"TemporalNode invariant failed at stage={stage_name!r} operation={operation!r}",
            )

        if self.rules.get("save_node_required", True) and stage_name == "post_execute":
            traces = list(getattr(turn_context, "stage_traces", []) or [])
            stage_order = [str(getattr(item, "stage", "") or "").strip().lower() for item in traces]
            if "save" not in stage_order:
                raise ExecutionBlockedInvariantError(
                    "SaveNode stage missing at post_execute invariant gate",
                )

        if self.rules.get("mutation_only_via_save", True) and bool(
            mutation_outside_save_node,
        ):
            raise ExecutionBlockedInvariantError(
                f"Mutation outside SaveNode invariant failed at stage={stage_name!r} operation={operation!r}",
            )

        return InvariantCheckResult(ok=True)


_DEFAULT_REGISTRY = InvariantRegistry()


def check_invariants(
    turn_context: Any,
    *,
    stage: str,
    call_site: str = "",
    mutation_outside_save_node: bool = False,
) -> InvariantCheckResult:
    """Backward-compatible function wrapper around the default registry."""
    return _DEFAULT_REGISTRY.check_invariants(
        turn_context,
        stage=stage,
        operation=call_site,
        mutation_outside_save_node=mutation_outside_save_node,
    )


__all__ = [
    "INVARIANTS",
    "InvariantCheckResult",
    "InvariantRegistry",
    "check_invariants",
]
