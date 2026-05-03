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


def test_phase2_replay_equivalence_ignores_declared_nondeterministic_tool_fields():
    trace_context = _trace_context()
    trace_context["steps"][1] = {
        "operation": "external_system_call",
        "payload": {
            "system": "builtin_tool:calendar",
            "status": "ok",
            "tool_call_record": {
                "canonicalized_input_payload": {"args": {"day": "monday"}},
                "canonicalized_input_hash": "",
                "raw_output_payload": {
                    "result": "ok",
                    "timing": {"latency_ms": 10},
                },
                "raw_output_hash": "",
                "response_schema_version": "1.0",
                "determinism_contract": {
                    "ignore_response_fields": ["timing.latency_ms"],
                },
                "stable_time_token": "tok-1",
            },
        },
    }
    seed = _seed(trace_context)

    trace_context_t2 = _trace_context()
    trace_context_t2["steps"][1] = {
        "operation": "external_system_call",
        "payload": {
            "system": "builtin_tool:calendar",
            "status": "ok",
            "tool_call_record": {
                "canonicalized_input_payload": {"args": {"day": "monday"}},
                "canonicalized_input_hash": "",
                "raw_output_payload": {
                    "result": "ok",
                    "timing": {"latency_ms": 9999},
                },
                "raw_output_hash": "",
                "response_schema_version": "1.0",
                "determinism_contract": {
                    "ignore_response_fields": ["timing.latency_ms"],
                },
                "stable_time_token": "tok-1",
            },
        },
    }

    report = verify_terminal_state_replay_equivalence(
        terminal_state_seed=seed,
        execution_trace_context=trace_context_t2,
        enforce_dag_equivalence=True,
    )

    assert report["equivalent"] is True


def test_phase2_replay_equivalence_detects_non_ignored_tool_field_drift():
    trace_context = _trace_context()
    trace_context["steps"][1] = {
        "operation": "external_system_call",
        "payload": {
            "system": "builtin_tool:calendar",
            "status": "ok",
            "tool_call_record": {
                "canonicalized_input_payload": {"args": {"day": "monday"}},
                "canonicalized_input_hash": "",
                "raw_output_payload": {"result": "ok"},
                "raw_output_hash": "",
                "response_schema_version": "1.0",
                "determinism_contract": {
                    "ignore_response_fields": [],
                },
                "stable_time_token": "tok-1",
            },
        },
    }
    seed = _seed(trace_context)

    tampered = _trace_context()
    tampered["steps"][1] = {
        "operation": "external_system_call",
        "payload": {
            "system": "builtin_tool:calendar",
            "status": "ok",
            "tool_call_record": {
                "canonicalized_input_payload": {"args": {"day": "monday"}},
                "canonicalized_input_hash": "",
                "raw_output_payload": {"result": "changed"},
                "raw_output_hash": "",
                "response_schema_version": "1.0",
                "determinism_contract": {
                    "ignore_response_fields": [],
                },
                "stable_time_token": "tok-1",
            },
        },
    }

    report = verify_terminal_state_replay_equivalence(
        terminal_state_seed=seed,
        execution_trace_context=tampered,
        enforce_dag_equivalence=True,
    )

    assert report["equivalent"] is False
    assert "tool_invocation_sequence_hash" in report["violations"]
