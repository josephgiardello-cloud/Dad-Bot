from __future__ import annotations

import pytest

from dadbot.core.system_state_algebra import (
    SYSTEM_STATE_ALGEBRA_SCHEMA_VERSION,
    SYSTEM_STATE_ALGEBRA_SNAPSHOT_SCHEMA_VERSION,
    SYSTEM_STATE_ALGEBRA_TRACE_SCHEMA_VERSION,
    evaluate_system_state_algebra,
    persist_system_state_algebra,
)


pytestmark = pytest.mark.unit


def _memory_entry(memory_id: str, score: float = 0.5) -> dict:
    return {
        "memory_id": memory_id,
        "summary": memory_id,
        "importance_score": score,
    }


def test_system_state_algebra_happy_path_projections() -> None:
    state = {
        "execution_truth_contract": {"consistent": True},
        "_retrieval_baseline": [_memory_entry("m1", 0.8)],
        "memory_retrieval_set": [_memory_entry("m1", 0.8)],
        "_causal_step_log": ["retrieval", "planning", "tool_execution", "post_tool_refresh", "persistence"],
        "phase": "save",
        "phase_history": [{"from": "plan", "to": "save", "reason": "stage-enter"}],
    }
    execution_result = {
        "status": "ok",
        "failure": {"class": "", "type": "", "message": "", "source": "", "retryable": False},
        "timeout": {"seconds": 0.0, "timed_out": False},
        "degradation": {"count": 0, "items": []},
        "outputs": {"response": "ok", "should_end": False, "semantic_eval_input_hash": ""},
    }

    algebra = evaluate_system_state_algebra(
        state=state,
        execution_result_payload=execution_result,
        trace_id="tr-1",
        context="unit",
    )

    assert algebra.get("overall_consistent") is True
    assert dict(algebra.get("projections", {}).get("turn_truth") or {}).get("overall_consistent") is True
    assert dict(algebra.get("projections", {}).get("control_plane_gate") or {}).get("ok") is True


def test_system_state_algebra_flags_timeout_status_mismatch() -> None:
    state: dict = {}
    execution_result = {
        "status": "ok",
        "failure": {"class": "", "type": "", "message": "", "source": "", "retryable": False},
        "timeout": {"seconds": 1.0, "timed_out": True},
        "degradation": {"count": 0, "items": []},
        "outputs": {"response": "", "should_end": False, "semantic_eval_input_hash": ""},
    }

    algebra = evaluate_system_state_algebra(
        state=state,
        execution_result_payload=execution_result,
        trace_id="tr-timeout",
        context="unit",
    )

    assert algebra.get("overall_consistent") is False
    violations = list(algebra.get("violations") or [])
    assert any("timeout_consistency" in str(item) for item in violations)


def test_system_state_algebra_exposes_cognitive_authority_boundary() -> None:
    algebra = evaluate_system_state_algebra(
        state={},
        execution_result_payload=None,
        trace_id="tr-boundary",
        context="unit",
    )

    boundary = dict(algebra.get("axes", {}).get("cognitive_authority_boundary") or {})
    memory = dict(boundary.get("memory") or {})
    planner = dict(boundary.get("planner") or {})
    assert "veto_constraints" in list(memory.get("owns") or [])
    assert "tool_choice" in list(planner.get("owns") or [])


def test_persist_system_state_algebra_writes_immutable_legacy_projections() -> None:
    state: dict = {}
    algebra = evaluate_system_state_algebra(
        state=state,
        execution_result_payload=None,
        trace_id="tr-purity",
        context="unit",
    )
    persist_system_state_algebra(
        state=state,
        algebra=algebra,
        trace_context="unit.persist",
        persist_legacy_projections=True,
        terminal_snapshot=False,
    )

    assert dict(state.get("system_state_algebra") or {}).get("schema_version") == SYSTEM_STATE_ALGEBRA_SCHEMA_VERSION
    with pytest.raises(RuntimeError):
        state["turn_truth"]["overall_consistent"] = False
    with pytest.raises(RuntimeError):
        state["control_plane_invariant_gate"]["ok"] = False


def test_persist_system_state_algebra_trace_and_snapshot_schema_versions() -> None:
    state: dict = {}
    algebra = evaluate_system_state_algebra(
        state=state,
        execution_result_payload=None,
        trace_id="tr-snapshot",
        context="unit",
    )
    persist_system_state_algebra(
        state=state,
        algebra=algebra,
        trace_context="unit.persist",
        persist_legacy_projections=True,
        terminal_snapshot=True,
    )

    trace_log = list(state.get("system_state_algebra_trace_log") or [])
    assert trace_log
    assert trace_log[-1].get("schema_version") == SYSTEM_STATE_ALGEBRA_TRACE_SCHEMA_VERSION
    assert trace_log[-1].get("format") == "system_state_algebra_trace"

    snapshots = list(state.get("system_state_algebra_frozen_snapshots") or [])
    assert snapshots
    assert snapshots[-1].get("schema_version") == SYSTEM_STATE_ALGEBRA_SNAPSHOT_SCHEMA_VERSION
    assert snapshots[-1].get("format") == "system_state_algebra_frozen_snapshot"


def test_no_side_authority_writes_guard_raises_on_plain_dict_override() -> None:
    state: dict = {}
    algebra = evaluate_system_state_algebra(
        state=state,
        execution_result_payload=None,
        trace_id="tr-guard",
        context="unit",
    )
    persist_system_state_algebra(
        state=state,
        algebra=algebra,
        trace_context="unit.persist",
        persist_legacy_projections=True,
        terminal_snapshot=False,
    )

    # Simulate an illegal side-authority write bypassing projection integrity.
    state["turn_truth"] = {"overall_consistent": False}
    with pytest.raises(RuntimeError, match="No-side-authority-writes violation"):
        evaluate_system_state_algebra(
            state=state,
            execution_result_payload=None,
            trace_id="tr-guard",
            context="unit",
        )
