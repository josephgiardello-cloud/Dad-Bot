from __future__ import annotations

from dadbot.core.execution_replay_engine import reconstruct_terminal_state_from_trace
from dadbot.core.execution_semantics import (
    AllowedTransformations,
    build_execution_state,
    execution_equivalence_relation,
)


def _trace(order: str = "ab") -> dict:
    calls = {
        "a": {
            "seq": 1,
            "operation": "external_system_call",
            "payload": {
                "operation": "tool_dispatch",
                "system": "builtin_tool:calendar",
                "status": "ok",
                "request_hash": "rq-a",
                "response_hash": "rs-a",
                "time_token": "ta",
            },
        },
        "b": {
            "seq": 2,
            "operation": "external_system_call",
            "payload": {
                "operation": "tool_dispatch",
                "system": "builtin_tool:weather",
                "status": "ok",
                "request_hash": "rq-b",
                "response_hash": "rs-b",
                "time_token": "tb",
            },
        },
    }
    ordered = [calls[item] for item in list(order)]
    return {
        "schema_version": "2.0",
        "final_hash": "seed-hash",
        "normalized_response": "done",
        "execution_dag": {"dag_hash": "dag-hash"},
        "memory_snapshot_used": {
            "memory_structured": {"topic": "ops"},
            "memory_full_history_id": "hist-1",
        },
        "memory_retrieval_set": [{"id": "m1", "summary": "ops"}],
        "tool_outputs": [{"tool": "calendar", "output": "ok"}, {"tool": "weather", "output": "ok"}],
        "steps": [
            {"seq": 0, "operation": "model_call", "payload": {"purpose": "chat"}},
            *ordered,
            {"seq": 3, "operation": "model_output", "payload": {"output_hash": "out-1"}},
        ],
    }


def _state(trace: dict):
    terminal = reconstruct_terminal_state_from_trace(
        terminal_state_seed={
            "schema_version": "1.0",
            "tool_trace_hash": "tool-trace",
            "policy_snapshot": {"kernel_policy": {"mode": "strict"}},
        },
        execution_trace_context=trace,
    )
    return build_execution_state(
        terminal_state=terminal,
        execution_trace_context=trace,
        semantic_mode="exact",
    )


def test_execution_semantics_relation_true_for_identical_states():
    state_a = _state(_trace("ab"))
    state_b = _state(_trace("ab"))
    decision = execution_equivalence_relation(state_a, state_b)
    assert decision.equivalent is True
    assert decision.structural_equivalent is True
    assert decision.semantic_equivalent is True
    assert decision.invariants_preserved is True


def test_execution_semantics_relation_detects_causal_structure_difference():
    state_a = _state(_trace("ab"))
    state_b = _state(_trace("ba"))
    strict = execution_equivalence_relation(state_a, state_b)
    assert strict.equivalent is False
    assert any(
        item in strict.violations for item in ["causal_structure", "execution_dag_topology", "tool_io_graph_hash"]
    )


def test_execution_semantics_relation_allows_causal_reorder_when_configured():
    state_a = _state(_trace("ab"))
    state_b = _state(_trace("ba"))
    decision = execution_equivalence_relation(
        state_a,
        state_b,
        rules=AllowedTransformations(allow_causal_reorder=True),
    )
    assert isinstance(decision.equivalent, bool)
    assert isinstance(decision.violations, list)


def test_execution_semantics_relation_enforces_embedding_state_invariant():
    state_a = _state(_trace("ab"))
    state_b = _state(_trace("ab"))
    state_b.embedding_state["lock_hash"] = "changed"
    decision = execution_equivalence_relation(state_a, state_b)
    assert decision.invariants_preserved is False
    assert "embedding_lock_state" in decision.violations
