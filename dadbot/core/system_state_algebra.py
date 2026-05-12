from __future__ import annotations

import copy
import hashlib
import json
import time
from typing import Any

from dadbot.core.execution_result_unified import ensure_unified_execution_result
from dadbot.core.memory_set_invariants import (
    MemorySetInvariantViolation,
    assert_causal_order,
    assert_memory_set_invariants,
)


SYSTEM_STATE_ALGEBRA_SCHEMA_VERSION = "1.0.0"
SYSTEM_STATE_ALGEBRA_TRACE_SCHEMA_VERSION = "1.0.0"
SYSTEM_STATE_ALGEBRA_SNAPSHOT_SCHEMA_VERSION = "1.0.0"


class _ReadOnlyProjection(dict):  # type: ignore[type-arg]
    """Dict projection that blocks in-place mutation to preserve purity."""

    def __setitem__(self, key: Any, value: Any) -> None:
        raise RuntimeError(f"Read-only system-state projection; cannot set key {key!r}")

    def __delitem__(self, key: Any) -> None:
        raise RuntimeError(f"Read-only system-state projection; cannot delete key {key!r}")

    def clear(self) -> None:
        raise RuntimeError("Read-only system-state projection; clear() is forbidden")

    def pop(self, key: Any, default: Any = None) -> Any:
        raise RuntimeError(f"Read-only system-state projection; pop({key!r}) is forbidden")

    def popitem(self) -> tuple[Any, Any]:
        raise RuntimeError("Read-only system-state projection; popitem() is forbidden")

    def setdefault(self, key: Any, default: Any = None) -> Any:
        raise RuntimeError(f"Read-only system-state projection; setdefault({key!r}) is forbidden")

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Read-only system-state projection; update() is forbidden")


def _as_memory_set(raw: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in list(raw or []) if isinstance(item, dict)]


def _stable_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _projection(payload: dict[str, Any]) -> dict[str, Any]:
    return _ReadOnlyProjection(dict(payload))


def _assert_authority_surface_integrity(state: dict[str, Any]) -> None:
    """Enforce that legacy truth keys are read-only projections of canonical algebra."""
    canonical = state.get("system_state_algebra")
    if not isinstance(canonical, dict):
        return
    projections = dict(canonical.get("projections") or {})
    expected_turn_truth = dict(projections.get("turn_truth") or {})
    expected_gate = dict(projections.get("control_plane_gate") or {})

    for key in ("turn_truth", "control_plane_invariant_gate", "turn_truth_terminal"):
        value = state.get(key)
        if value is None:
            continue
        if not isinstance(value, _ReadOnlyProjection):
            raise RuntimeError(
                f"No-side-authority-writes violation: state[{key!r}] must be a read-only projection",
            )

    live_turn_truth = state.get("turn_truth")
    if isinstance(live_turn_truth, dict) and dict(live_turn_truth) != expected_turn_truth:
        raise RuntimeError("Parallel truth derivation detected for 'turn_truth'")

    live_gate = state.get("control_plane_invariant_gate")
    if isinstance(live_gate, dict) and dict(live_gate) != expected_gate:
        raise RuntimeError("Parallel truth derivation detected for 'control_plane_invariant_gate'")

    live_terminal = state.get("turn_truth_terminal")
    if isinstance(live_terminal, dict) and dict(live_terminal) != expected_turn_truth:
        raise RuntimeError("Parallel truth derivation detected for 'turn_truth_terminal'")


def _persist_legacy_projections_if_needed(
    *,
    state: dict[str, Any],
    turn_truth: dict[str, Any],
    gate: dict[str, Any],
    persist_legacy_projections: bool,
    terminal_snapshot: bool,
) -> None:
    if persist_legacy_projections:
        state["turn_truth"] = _projection(turn_truth)
        state["control_plane_invariant_gate"] = _projection(gate)
        if terminal_snapshot:
            state["turn_truth_terminal"] = _projection(turn_truth)


def _create_trace_log_entry(
    *,
    sequence: int,
    trace_context: str,
    turn_truth: dict[str, Any],
    gate: dict[str, Any],
    canonical: dict[str, Any],
    algebra_hash: str,
    projection_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "format": "system_state_algebra_trace",
        "schema_version": SYSTEM_STATE_ALGEBRA_TRACE_SCHEMA_VERSION,
        "sequence": sequence,
        "timestamp_ms": int(time.time() * 1000),
        "context": str(trace_context or "runtime"),
        "trace_id": str(turn_truth.get("trace_id") or gate.get("trace_id") or ""),
        "algebra_version": str(canonical.get("algebra_version") or ""),
        "algebra_hash": algebra_hash,
        "projection_hashes": projection_hashes,
        "overall_consistent": bool(canonical.get("overall_consistent", False)),
        "violation_count": int(len(list(canonical.get("violations") or []))),
    }


def _append_frozen_snapshot_if_needed(
    *,
    state: dict[str, Any],
    terminal_snapshot: bool,
    turn_truth: dict[str, Any],
    gate: dict[str, Any],
    canonical: dict[str, Any],
    algebra_hash: str,
) -> None:
    if terminal_snapshot:
        snapshots = list(state.get("system_state_algebra_frozen_snapshots") or [])
        snapshots.append(
            {
                "format": "system_state_algebra_frozen_snapshot",
                "schema_version": SYSTEM_STATE_ALGEBRA_SNAPSHOT_SCHEMA_VERSION,
                "snapshot_id": f"ssa-{len(snapshots) + 1}",
                "timestamp_ms": int(time.time() * 1000),
                "trace_id": str(turn_truth.get("trace_id") or ""),
                "algebra_hash": algebra_hash,
                "algebra": canonical,
                "projections": {
                    "turn_truth": turn_truth,
                    "control_plane_gate": gate,
                },
                "benchmark_contract": {
                    "status": str(turn_truth.get("execution_status") or "pending"),
                    "overall_consistent": bool(turn_truth.get("overall_consistent", False)),
                    "violation_count": int(turn_truth.get("violation_count") or 0),
                },
            }
        )
        state["system_state_algebra_frozen_snapshots"] = snapshots


def persist_system_state_algebra(
    *,
    state: dict[str, Any],
    algebra: dict[str, Any],
    trace_context: str,
    persist_legacy_projections: bool = True,
    terminal_snapshot: bool = False,
) -> None:
    """Persist canonical algebra and derived projection artifacts.

    This is the only writer for system-state authority surfaces.
    """
    _assert_authority_surface_integrity(state)
    canonical = copy.deepcopy(dict(algebra))
    canonical["schema_version"] = SYSTEM_STATE_ALGEBRA_SCHEMA_VERSION
    state["system_state_algebra"] = canonical

    projections = dict(canonical.get("projections") or {})
    turn_truth = dict(projections.get("turn_truth") or {})
    gate = dict(projections.get("control_plane_gate") or {})

    _persist_legacy_projections_if_needed(
        state=state,
        turn_truth=turn_truth,
        gate=gate,
        persist_legacy_projections=persist_legacy_projections,
        terminal_snapshot=terminal_snapshot,
    )

    projection_hashes = {
        "turn_truth": _stable_hash(turn_truth) if turn_truth else "",
        "control_plane_gate": _stable_hash(gate) if gate else "",
    }
    algebra_hash = _stable_hash(canonical)
    trace_log = list(state.get("system_state_algebra_trace_log") or [])
    entry = _create_trace_log_entry(
        sequence=len(trace_log) + 1,
        trace_context=trace_context,
        turn_truth=turn_truth,
        gate=gate,
        canonical=canonical,
        algebra_hash=algebra_hash,
        projection_hashes=projection_hashes,
    )
    trace_log.append(entry)
    state["system_state_algebra_trace_log"] = trace_log

    _append_frozen_snapshot_if_needed(
        state=state,
        terminal_snapshot=terminal_snapshot,
        turn_truth=turn_truth,
        gate=gate,
        canonical=canonical,
        algebra_hash=algebra_hash,
    )


def _collect_execution_violations(
    *,
    state: dict[str, Any],
    execution_result_payload: dict[str, Any] | None,
) -> tuple[dict[str, Any], str, list[str], dict[str, Any]]:
    execution_result = ensure_unified_execution_result(execution_result_payload)
    execution_status = str(execution_result.get("status") or "pending").strip().lower()
    timeout_info = dict(execution_result.get("timeout") or {})
    violations: list[str] = []
    if bool(timeout_info.get("timed_out", False)) and execution_status != "failed":
        violations.append("timeout_consistency:timed_out requires failed terminal status")

    execution_contract = dict(state.get("execution_truth_contract") or {})
    if execution_contract and not bool(execution_contract.get("consistent", True)):
        violations.append(
            f"execution_truth:{execution_contract.get('failure_code') or 'inconsistent'}",
        )
    return execution_result, execution_status, violations, execution_contract


def _collect_memory_axis(
    *,
    state: dict[str, Any],
    tag: str,
) -> tuple[dict[str, Any], list[str]]:
    memory_baseline = _as_memory_set(state.get("_retrieval_baseline"))
    memory_current = _as_memory_set(state.get("memory_retrieval_set"))
    inline_memory_violation = str(state.get("memory_invariant_violation") or "").strip()
    violations: list[str] = []
    if inline_memory_violation:
        violations.append(f"memory_invariant:{inline_memory_violation}")
    elif memory_baseline:
        try:
            assert_memory_set_invariants(
                memory_baseline,
                memory_current,
                context=f"{tag}.memory",
            )
        except MemorySetInvariantViolation as exc:
            violations.append(str(exc))
    return {
        "baseline_count": len(memory_baseline),
        "current_count": len(memory_current),
        "inline_violation": inline_memory_violation,
    }, violations


def _collect_causal_axis(*, state: dict[str, Any], tag: str) -> tuple[dict[str, Any], list[str]]:
    causal_steps = [str(step) for step in list(state.get("_causal_step_log") or [])]
    violations: list[str] = []
    if causal_steps:
        try:
            assert_causal_order(causal_steps, context=f"{tag}.causal")
        except MemorySetInvariantViolation as exc:
            violations.append(str(exc))
    return {
        "steps": list(causal_steps),
        "step_count": len(causal_steps),
    }, violations


def _lifecycle_axis(state: dict[str, Any]) -> dict[str, Any]:
    phase = str(state.get("phase") or "")
    phase_history = list(state.get("phase_history") or [])
    return {
        "phase": phase,
        "phase_history_count": len(phase_history),
    }


def _authority_boundary_axis() -> dict[str, Any]:
    return {
        "memory": {
            "owns": [
                "retrieval_selection",
                "retrieval_invariants",
                "veto_constraints",
                "lifecycle_constraints",
            ],
            "does_not_own": ["final_tool_choice", "final_response_wording"],
        },
        "planner": {
            "owns": [
                "tool_choice",
                "parameter_shape",
                "execution_plan",
                "response_strategy",
            ],
            "constrained_by": ["memory.veto_constraints", "memory.lifecycle_constraints"],
        },
    }


def _algebra_projections(*, trace_token: str, overall_consistent: bool, violations: list[str]) -> dict[str, Any]:
    turn_truth_projection = {
        "trace_id": str(trace_token or ""),
        "overall_consistent": overall_consistent,
        "violations": list(violations),
        "violation_count": len(violations),
        "execution_status": "",
        "authority": "system_state_algebra",
    }
    control_plane_gate_projection = {
        "ok": overall_consistent,
        "trace_id": str(trace_token or ""),
        "violation_count": len(violations),
        "violations": list(violations),
        "authority": "system_state_algebra",
    }
    return {
        "turn_truth": turn_truth_projection,
        "control_plane_gate": control_plane_gate_projection,
    }


def evaluate_system_state_algebra(
    *,
    state: dict[str, Any],
    execution_result_payload: dict[str, Any] | None = None,
    trace_token: str = "",
    context: str = "",
    **legacy_kwargs: Any,
) -> dict[str, Any]:
    """Evaluate canonical runtime truth and emit projection-ready views.

    This consolidates all state-authority checks into a single algebra object.
    Other subsystems should consume projections from this payload instead of
    recomputing consistency independently.
    """
    _assert_authority_surface_integrity(state)
    tag = str(context or "runtime").strip() or "runtime"
    execution_result, execution_status, execution_violations, execution_contract = _collect_execution_violations(
        state=state,
        execution_result_payload=execution_result_payload,
    )
    memory_axis, memory_violations = _collect_memory_axis(state=state, tag=tag)
    causal_axis, causal_violations = _collect_causal_axis(state=state, tag=tag)
    lifecycle_axis = _lifecycle_axis(state)

    violations: list[str] = []
    violations.extend(execution_violations)
    violations.extend(memory_violations)
    violations.extend(causal_violations)

    overall_consistent = len(violations) == 0
    projections = _algebra_projections(
        trace_token=str(trace_token or legacy_kwargs.get("trace_id") or ""),
        overall_consistent=overall_consistent,
        violations=violations,
    )
    projections["turn_truth"]["execution_status"] = execution_status

    return {
        "algebra_version": "system-state-v1",
        "schema_version": SYSTEM_STATE_ALGEBRA_SCHEMA_VERSION,
        "authority": "system_state_algebra",
        "overall_consistent": overall_consistent,
        "violations": list(violations),
        "axes": {
            "execution_result": execution_result,
            "execution_contract": execution_contract,
            "memory": memory_axis,
            "causal": causal_axis,
            "lifecycle": lifecycle_axis,
            "cognitive_authority_boundary": _authority_boundary_axis(),
        },
        "projections": projections,
    }
