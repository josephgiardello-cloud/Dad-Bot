from __future__ import annotations

import copy

import pytest

from dadbot.core.execution_context import derive_execution_trace_hash
from dadbot.core.execution_equivalence_oracle import ExecutionEquivalenceOracle
from dadbot.core.execution_replay_engine import (
    reconstruct_terminal_state_from_trace,
    verify_terminal_state_replay_equivalence,
)
from dadbot.core.execution_semantics import (
    build_execution_state,
    execution_equivalence_relation,
)

pytestmark = pytest.mark.unit


def _trace_context() -> dict:
    return {
        "schema_version": "2.0",
        "prompt": "kernel closure proof",
        "final_hash": "trace-final-hash-001",
        "normalized_response": "deterministic output",
        "execution_dag": {"dag_hash": "dag-hash-001"},
        "memory_snapshot_used": {
            "memory_structured": {"mood": "steady", "topic": "tools"},
            "memory_full_history_id": "hist-001",
        },
        "memory_retrieval_set": [
            {"id": "r1", "text": "fact-a"},
            {"id": "r2", "text": "fact-b"},
        ],
        "operations": ["model_call", "external_system_call", "model_output"],
        "steps": [
            {
                "seq": 1,
                "operation": "model_call",
                "payload": {
                    "purpose": "inference",
                    "status": "ok",
                    "input_hash": "in-1",
                },
            },
            {
                "seq": 2,
                "operation": "external_system_call",
                "payload": {
                    "system": "builtin_tool:calendar",
                    "status": "ok",
                    "request_hash": "rq-1",
                    "response_hash": "rs-1",
                },
            },
            {
                "seq": 3,
                "operation": "model_output",
                "payload": {
                    "output_hash": "out-1",
                    "status": "ok",
                    "passed": True,
                    "issue_count": 0,
                },
            },
        ],
        "execution_snapshot": {
            "outputs_per_step": [
                {"seq": 1, "operation": "model_call", "output": {"status": "ok"}},
                {
                    "seq": 2,
                    "operation": "external_system_call",
                    "output": {"status": "ok"},
                },
                {"seq": 3, "operation": "model_output", "output": {"status": "ok"}},
            ],
            "final_output": "deterministic output",
        },
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


def _state_from(trace_context: dict) -> object:
    seed = _seed(trace_context)
    return build_execution_state(
        terminal_state=seed,
        execution_trace_context=trace_context,
        semantic_mode="exact",
    )


class TestDeterminismOfExecutionFunction:
    def test_same_input_same_seed_produces_identical_trace_hash(self):
        t1 = _trace_context()
        t2 = _trace_context()
        assert derive_execution_trace_hash(t1) == derive_execution_trace_hash(t2)

    def test_unordered_dict_representation_does_not_change_trace_hash(self):
        t1 = _trace_context()
        t2 = _trace_context()
        # same semantic payload, different insertion order
        t2["steps"][1]["payload"] = {
            "response_hash": "rs-1",
            "system": "builtin_tool:calendar",
            "status": "ok",
            "request_hash": "rq-1",
        }
        assert derive_execution_trace_hash(t1) == derive_execution_trace_hash(t2)


class TestSnapshotCompletenessDimensions:
    def test_terminal_snapshot_includes_required_dimensions(self):
        trace = _trace_context()
        seed = _seed(trace)
        required = {
            "final_trace_hash",
            "memory_retrieval_hash",
            "execution_dag_hash",
            "execution_order_hash",
            "node_decision_sequence_hash",
            "failure_recovery_transition_hash",
            "tool_invocation_sequence_hash",
            "post_commit_mutation_effects_hash",
            "determinism_closure_hash",
        }
        assert required.issubset(set(seed.keys()))


class TestEquivalenceFunctionCompleteness:
    def test_reflexivity(self):
        a = _state_from(_trace_context())
        result = execution_equivalence_relation(a, a)
        assert result.equivalent is True

    def test_symmetry(self):
        a = _state_from(_trace_context())
        b = _state_from(_trace_context())
        ab = execution_equivalence_relation(a, b)
        ba = execution_equivalence_relation(b, a)
        assert ab.equivalent == ba.equivalent
        assert sorted(ab.violations) == sorted(ba.violations)

    def test_transitivity_for_identical_states(self):
        a = _state_from(_trace_context())
        b = _state_from(_trace_context())
        c = _state_from(_trace_context())
        assert execution_equivalence_relation(a, b).equivalent is True
        assert execution_equivalence_relation(b, c).equivalent is True
        assert execution_equivalence_relation(a, c).equivalent is True

    def test_oracle_path_requires_equivalence_relation(self, monkeypatch: pytest.MonkeyPatch):
        trace = _trace_context()
        seed = _seed(trace)

        def _fake_relation(*_args, **_kwargs):
            class _Decision:
                equivalent = False
                semantic_equivalent = False
                structural_equivalent = False
                invariants_preserved = False
                violations = ["forced-check"]

            return _Decision()

        monkeypatch.setattr(
            "dadbot.core.execution_equivalence_oracle.execution_equivalence_relation",
            _fake_relation,
        )

        result = ExecutionEquivalenceOracle.evaluate(
            input_seed="same-input",
            trace_seed="same-trace",
            memory_state_id="same-memory",
            terminal_state_seed=seed,
            execution_trace_context=trace,
        )
        assert result.equivalent is False
        assert "forced-check" in result.violations


class TestExecutionPathClosure:
    def test_oracle_detects_real_tamper_without_monkeypatch(self):
        trace = _trace_context()
        seed = _seed(trace)
        tampered = dict(seed)
        tampered["tool_invocation_sequence_hash"] = "tampered-tool-seq"

        result = ExecutionEquivalenceOracle.evaluate(
            input_seed="same-input",
            trace_seed="same-trace",
            memory_state_id="same-memory",
            terminal_state_seed=tampered,
            execution_trace_context=trace,
        )
        assert result.equivalent is False
        assert "tool_invocation_sequence_hash" in list(result.replay_report.get("violations") or [])

    def test_oracle_requires_both_replay_and_semantic_equivalence(self, monkeypatch: pytest.MonkeyPatch):
        trace = _trace_context()
        seed = _seed(trace)

        def _fake_verify(**_kwargs):
            return {
                "equivalent": True,
                "violations": [],
                "expected_terminal_state": seed,
                "replayed_terminal_state": seed,
            }

        def _fake_relation(*_args, **_kwargs):
            class _Decision:
                equivalent = False
                semantic_equivalent = False
                structural_equivalent = True
                invariants_preserved = True
                violations = ["semantic_actions"]

            return _Decision()

        monkeypatch.setattr(
            "dadbot.core.execution_equivalence_oracle.verify_terminal_state_replay_equivalence",
            _fake_verify,
        )
        monkeypatch.setattr(
            "dadbot.core.execution_equivalence_oracle.execution_equivalence_relation",
            _fake_relation,
        )

        result = ExecutionEquivalenceOracle.evaluate(
            input_seed="same-input",
            trace_seed="same-trace",
            memory_state_id="same-memory",
            terminal_state_seed=seed,
            execution_trace_context=trace,
        )
        assert result.replay_report.get("equivalent") is True
        assert result.equivalent is False


class TestTemporalStabilityReplayInvariance:
    def test_replay_equivalence_holds_for_identical_replay_seed(self):
        trace = _trace_context()
        seed = _seed(trace)
        report = verify_terminal_state_replay_equivalence(
            terminal_state_seed=seed,
            execution_trace_context=copy.deepcopy(trace),
            enforce_dag_equivalence=True,
        )
        assert report["equivalent"] is True
        assert report["violations"] == []

    def test_time_shift_invariance_for_temporal_only_payload_fields(self):
        trace_t = _trace_context()
        trace_t_plus = copy.deepcopy(trace_t)

        trace_t["steps"][0]["payload"]["completed_at"] = 1000.0
        trace_t_plus["steps"][0]["payload"]["completed_at"] = 999999.0
        trace_t["steps"][1]["payload"]["timestamp"] = "2026-05-02T10:00:00"
        trace_t_plus["steps"][1]["payload"]["timestamp"] = "2030-01-01T00:00:00"

        assert derive_execution_trace_hash(trace_t) == derive_execution_trace_hash(trace_t_plus)

    def test_failure_replay_transition_hash_is_stable_for_same_failure_chain(self):
        trace_a = _trace_context()
        trace_b = copy.deepcopy(trace_a)
        for trace in (trace_a, trace_b):
            trace["steps"].append(
                {
                    "seq": 4,
                    "operation": "external_system_call",
                    "payload": {
                        "system": "builtin_tool:calendar",
                        "status": "error",
                        "request_hash": "rq-fail",
                        "response_hash": "rs-fail",
                        "error": "timeout",
                    },
                },
            )

        seed_a = _seed(trace_a)
        seed_b = _seed(trace_b)
        assert (
            seed_a["failure_recovery_transition_hash"]
            == seed_b["failure_recovery_transition_hash"]
        )
