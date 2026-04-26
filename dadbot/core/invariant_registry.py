from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dadbot.core.execution_firewall import ExecutionBlockedInvariantError, ExecutionFirewall, FirewallContext


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


def check_invariants(
    turn_context: Any,
    *,
    stage: str,
    call_site: str = "",
    firewall: ExecutionFirewall | None = None,
    mutation_outside_save_node: bool = False,
) -> InvariantCheckResult:
    """Centralized invariant enforcement gate for Phase 4 strict runtime.

    Raises:
        ExecutionBlockedInvariantError when any enforced invariant is violated.
    """

    stage_name = _stage_name(stage)
    fw = firewall or ExecutionFirewall()
    temporal_missing = INVARIANTS.get("temporal_required", True) and not _has_temporal(turn_context)

    if INVARIANTS.get("save_node_required", True) and stage_name == "post_execute":
        traces = list(getattr(turn_context, "stage_traces", []) or [])
        stage_order = [str(getattr(item, "stage", "") or "").strip().lower() for item in traces]
        if "save" not in stage_order:
            raise ExecutionBlockedInvariantError("SaveNode stage missing at post_execute invariant gate")

    fw_context = FirewallContext(
        trace_id=str(getattr(turn_context, "trace_id", "") or ""),
        stage=stage_name,
        mutation_outside_save_node=bool(
            INVARIANTS.get("mutation_only_via_save", True) and mutation_outside_save_node
        ),
        temporal_missing=bool(temporal_missing),
        metadata=dict(getattr(turn_context, "metadata", {}) or {}),
    )

    if INVARIANTS.get("no_legacy_paths", True) and call_site:
        fw.enforce_execution_firewall(call_site, fw_context)
    elif fw_context.mutation_outside_save_node or fw_context.temporal_missing:
        fw.enforce_execution_firewall(call_site or stage_name or "invariant_gate", fw_context)

    return InvariantCheckResult(ok=True)


__all__ = ["INVARIANTS", "InvariantCheckResult", "check_invariants"]
