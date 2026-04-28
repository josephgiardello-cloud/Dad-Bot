from __future__ import annotations

from dadbot.core.execution_replay_engine import (
    reconstruct_terminal_state_from_trace,
    verify_terminal_state_replay_equivalence,
)



def _trace_context() -> dict:
    return {
        "final_hash": "trace-final-hash-001",
        "normalized_response": "hello from deterministic replay",
        "execution_dag": {"dag_hash": "dag-hash-001"},
        "memory_snapshot_used": {
            "memory_structured": {"mood": "steady", "topic": "tools"},
            "memory_full_history_id": "hist-001",
        },
        "memory_retrieval_set": [
            {"id": "r1", "text": "fact-a"},
            {"id": "r2", "text": "fact-b"},
        ],
        "steps": [
            {"operation": "model_output", "payload": {"output_hash": "out-1"}},
            {"operation": "tool_call", "payload": {"name": "noop"}},
            {"operation": "model_output", "payload": {"output_hash": "out-2"}},
        ],
    }



def _seed(trace_context: dict) -> dict:
    return reconstruct_terminal_state_from_trace(
        terminal_state_seed={
            "schema_version": "1.0",
            "tool_trace_hash": "tool-trace-001",
            "policy_snapshot": {
                "kernel_policy": {"mode": "strict"},
                "kernel_rejections": [],
                "capability_audit_report": {"ok": True},
                "safety_check_result": {"allowed": True},
                "tony_level": "low",
                "tony_score": 0,
            },
        },
        execution_trace_context=trace_context,
    )



def test_phase2_replay_equivalence_passes_for_identical_seed_and_trace():
    trace_context = _trace_context()
    seed = _seed(trace_context)

    report = verify_terminal_state_replay_equivalence(
        terminal_state_seed=seed,
        execution_trace_context=trace_context,
        enforce_dag_equivalence=True,
    )

    assert report["equivalent"] is True
    assert report["violations"] == []



def test_phase2_replay_equivalence_detects_memory_linearization_divergence():
    trace_context = _trace_context()
    seed = _seed(trace_context)
    seed["memory_retrieval_hash"] = "tampered-memory-hash"

    report = verify_terminal_state_replay_equivalence(
        terminal_state_seed=seed,
        execution_trace_context=trace_context,
        enforce_dag_equivalence=True,
    )

    assert report["equivalent"] is False
    assert "memory_retrieval_hash" in report["violations"]



def test_phase2_replay_equivalence_detects_policy_snapshot_divergence():
    trace_context = _trace_context()
    seed = _seed(trace_context)
    seed["policy_hash"] = "tampered-policy-hash"

    report = verify_terminal_state_replay_equivalence(
        terminal_state_seed=seed,
        execution_trace_context=trace_context,
        enforce_dag_equivalence=True,
    )

    assert report["equivalent"] is False
    assert "policy_hash" in report["violations"]



def test_phase2_replay_equivalence_requires_dag_hash_match():
    trace_context = _trace_context()
    seed = _seed(trace_context)
    seed["execution_dag_hash"] = "dag-hash-tampered"

    report = verify_terminal_state_replay_equivalence(
        terminal_state_seed=seed,
        execution_trace_context=trace_context,
        enforce_dag_equivalence=True,
    )

    assert report["equivalent"] is False
    assert "execution_dag_hash" in report["violations"]
