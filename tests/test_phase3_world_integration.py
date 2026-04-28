from __future__ import annotations

from types import SimpleNamespace

import pytest

from dadbot.core.execution_equivalence_oracle import ExecutionEquivalenceOracle
from dadbot.core.execution_recovery import ExecutionRecovery
from dadbot.core.execution_replay_engine import reconstruct_terminal_state_from_trace
from dadbot.core.execution_trace_context import (
    ExecutionTraceRecorder,
    bind_execution_trace,
    build_execution_trace_context,
    record_external_system_call,
)
from dadbot.core.truth_system import TruthSystemViolation, enforce_authoritative_truth_system



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


class TestPhase3ExternalIOModel:
    def test_trace_context_includes_external_system_call_graph(self):
        recorder = ExecutionTraceRecorder(trace_id="trace-1", prompt="hello")
        recorder.record("iteration_start", {"iteration": 0})
        recorder.record("model_call", {"iteration": 0, "input_hash": "in-1"})
        recorder.record("model_output", {"iteration": 0, "output_hash": "out-1"})
        recorder.record(
            "external_system_call",
            {
                "operation": "tool_dispatch",
                "system": "builtin_tool:weather",
                "request_hash": "rq-1",
                "response_hash": "rs-1",
                "status": "ok",
                "time_token": "t1",
            },
        )
        ctx = SimpleNamespace(
            user_input="hello",
            trace_id="trace-1",
            state={"memory_structured": {}, "tool_results": []},
            metadata={"determinism": {}, "control_plane": {"session_id": "s1"}},
        )
        trace = build_execution_trace_context(context=ctx, result=("done", True), recorder=recorder)
        ext = dict(trace.get("external_system_calls") or {})
        assert ext.get("graph_hash")
        assert len(list(ext.get("nodes") or [])) == 1
        assert trace["execution_dag"].get("external_system_call_graph_hash") == ext.get("graph_hash")


class TestPhase3EmbeddingLock:
    def test_embedding_version_lock_detects_model_drift(self, bot):
        manager = bot.memory_manager.semantic
        first = manager._lock_embedding_version(model_name="m1", vector_size=12)
        second = manager._lock_embedding_version(model_name="m2", vector_size=12)
        assert first["drift_detected"] is False
        assert second["drift_detected"] is True

    def test_semantic_retrieval_signature_stable_across_model_switch(self, bot, monkeypatch):
        manager = bot.memory_manager.semantic

        def _stable_embed(texts, purpose="semantic retrieval"):
            values = [texts] if isinstance(texts, str) else list(texts)
            out = []
            for value in values:
                token = str(value).lower()
                out.append([1.0 if "money" in token else 0.0] * 12)
            return out

        monkeypatch.setattr(manager, "embed_texts", _stable_embed)
        memories = [
            {"summary": "saving money this month", "category": "finance", "mood": "calm", "updated_at": "2025-01-01"},
            {"summary": "watching movies", "category": "fun", "mood": "happy", "updated_at": "2025-01-02"},
        ]

        recorder = ExecutionTraceRecorder(trace_id="phase3-semantic", prompt="money")
        with bind_execution_trace(recorder, required=False):
            manager._lock_embedding_version(model_name="model-a", vector_size=12)
            sig_a = manager.semantic_retrieval_signature("money", memories, limit=1)
            manager._lock_embedding_version(model_name="model-b", vector_size=12)
            sig_b = manager.semantic_retrieval_signature("money", memories, limit=1)
        assert sig_a == sig_b


class TestPhase3ExecutionEquivalenceOracle:
    def test_oracle_same_seed_same_trace_same_memory_is_equivalent(self):
        trace = _trace_context()
        seed = _seed(trace)
        report = ExecutionEquivalenceOracle.evaluate(
            input_seed="seed-42",
            trace_seed="trace-seed-42",
            memory_state_id=str(seed.get("memory_view_state_id") or ""),
            terminal_state_seed=seed,
            execution_trace_context=trace,
        )
        assert report.equivalent is True
        assert report.violations == []
        assert report.invariance_hash


class TestPhase3FailureRecovery:
    def test_partial_trace_repair_and_safe_fallback(self):
        broken_trace = {
            "steps": [
                {"operation": "model_call", "payload": {"input_hash": "x"}},
                "bad-step",
                {"operation": "model_output", "payload": {"output_hash": "y"}},
            ]
        }
        repaired = ExecutionRecovery.repair_partial_trace_context(broken_trace)
        assert len(repaired["steps"]) == 2
        fallback = ExecutionRecovery.safe_fallback_reconstruction(
            checkpoint={"state": {"k": "v"}, "metadata": {"m": 1}},
            trace_context=broken_trace,
        )
        assert fallback["fallback"]["mode"] == "safe_reconstruction"
        assert fallback["state"]["k"] == "v"


class TestPhase3TruthSystem:
    def test_truth_system_rejects_experimental_paths(self):
        with pytest.raises(TruthSystemViolation):
            enforce_authoritative_truth_system(
                metadata={"experimental_execution_kernel_enabled": True},
                state={},
            )


class TestTraceExternalCallRecorder:
    def test_record_external_system_call_returns_payload(self):
        recorder = ExecutionTraceRecorder(trace_id="trace-2", prompt="hello")
        from dadbot.core.execution_trace_context import bind_execution_trace

        with bind_execution_trace(recorder, required=False):
            step = record_external_system_call(
                operation="tool_dispatch",
                system="builtin_tool:calendar",
                request_payload={"args": {"day": "monday"}},
                response_payload={"status": "ok"},
                status="ok",
                source="test",
            )
        assert isinstance(step, dict)
        assert str((step or {}).get("operation") or "") == "external_system_call"
