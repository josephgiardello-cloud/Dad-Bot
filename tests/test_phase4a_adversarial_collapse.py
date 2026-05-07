from __future__ import annotations

from copy import deepcopy

import pytest

from dadbot.core.execution_replay_engine import (
    reconstruct_terminal_state_from_trace,
    verify_terminal_state_replay_equivalence,
)
from dadbot.core.graph import FatalTurnError, MutationIntent, MutationKind, MutationQueue
from dadbot.core.session_store import SessionStore


def _baseline_trace() -> dict:
    return {
        "schema_version": "2.0",
        "final_hash": "trace-final-hash-4a",
        "normalized_response": "stable response",
        "execution_dag": {"dag_hash": "dag-4a"},
        "memory_retrieval_set": [{"id": "m1", "summary": "fact"}],
        "steps": [
            {"seq": 0, "operation": "model_output", "payload": {"output_hash": "out-1"}},
            {
                "seq": 1,
                "operation": "external_system_call",
                "payload": {
                    "operation": "tool_dispatch",
                    "system": "builtin_tool:calendar",
                    "request_hash": "rq-a",
                    "response_hash": "rs-a",
                    "status": "ok",
                    "time_token": "ta",
                },
            },
            {
                "seq": 2,
                "operation": "external_system_call",
                "payload": {
                    "operation": "tool_dispatch",
                    "system": "builtin_tool:weather",
                    "request_hash": "rq-b",
                    "response_hash": "rs-b",
                    "status": "ok",
                    "time_token": "tb",
                },
            },
            {"seq": 3, "operation": "model_output", "payload": {"output_hash": "out-2"}},
        ],
    }


def _baseline_seed(trace: dict) -> dict:
    return reconstruct_terminal_state_from_trace(
        terminal_state_seed={
            "schema_version": "1.0",
            "tool_trace_hash": "tool-trace-4a",
            "policy_snapshot": {
                "kernel_policy": {"mode": "strict"},
                "kernel_rejections": [],
            },
        },
        execution_trace_context=trace,
    )


class TestPhase4ATraceCorruption:
    def test_broken_ordering_in_replay_is_flagged(self):
        baseline = _baseline_trace()
        seed = _baseline_seed(baseline)
        broken = deepcopy(baseline)
        broken["steps"] = list(reversed(broken["steps"]))
        report = verify_terminal_state_replay_equivalence(
            terminal_state_seed=seed,
            execution_trace_context=broken,
            enforce_dag_equivalence=True,
        )
        assert report["equivalent"] is False
        assert report["violations"]


class TestPhase4AExternalIONondeterminism:
    def test_out_of_order_external_calls_flag_replay_divergence(self):
        baseline = _baseline_trace()
        seed = _baseline_seed(baseline)
        nondeterministic = deepcopy(baseline)
        nondeterministic["steps"][1], nondeterministic["steps"][2] = (
            nondeterministic["steps"][2],
            nondeterministic["steps"][1],
        )
        report = verify_terminal_state_replay_equivalence(
            terminal_state_seed=seed,
            execution_trace_context=nondeterministic,
            enforce_dag_equivalence=True,
        )
        assert report["equivalent"] is False
        assert "execution_dag_hash" in report["violations"] or "final_trace_hash" in report["violations"]

    def test_duplicate_external_call_is_treated_as_divergence(self):
        baseline = _baseline_trace()
        seed = _baseline_seed(baseline)
        duplicated = deepcopy(baseline)
        duplicated["steps"].insert(2, deepcopy(duplicated["steps"][1]))
        report = verify_terminal_state_replay_equivalence(
            terminal_state_seed=seed,
            execution_trace_context=duplicated,
            enforce_dag_equivalence=True,
        )
        assert report["equivalent"] is False


class TestPhase4AMemoryMutationRaceSimulation:
    def test_out_of_order_persistence_commits_resolve_deterministically(self):
        store = SessionStore()
        events = [
            {
                "sequence": 2,
                "type": "SESSION_STATE_UPDATED",
                "session_id": "s-4a",
                "payload": {"state": {"counter": 2}},
            },
            {
                "sequence": 1,
                "type": "SESSION_STATE_UPDATED",
                "session_id": "s-4a",
                "payload": {"state": {"counter": 1}},
            },
        ]
        store.rebuild_from_ledger(events)
        assert store.get("s-4a")["counter"] == 2

    def test_partial_rollback_scenario_fails_safe(self):
        queue = MutationQueue()
        queue.bind_owner("trace-4a-race")

        def _compensator():
            return None

        for idx in range(2):
            queue.queue(
                MutationIntent(
                    type=MutationKind.GOAL,
                    payload={"idx": idx},
                    requires_temporal=False,
                    source="phase4a",
                    compensator=_compensator,
                )
            )

        calls = {"n": 0}

        def _executor(_intent):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("forced second failure")
            return _compensator

        with pytest.raises(FatalTurnError, match="MutationQueue drain failed"):
            queue.drain(_executor, hard_fail_on_error=True, transactional=True)

        snapshot = queue.snapshot()
        assert int(snapshot.get("pending") or 0) + int(snapshot.get("ledger_pending") or 0) >= 2
        latest_tx = dict(snapshot.get("latest_transaction") or {})
        assert latest_tx.get("status") in {"rolled_back", "rollback_failed"}


class TestPhase4AReplayDegradation:
    def test_checkpoint_missing_fields_rejects_fallback_reconstruction(self):
        with pytest.raises(RuntimeError, match="replay-only"):
            ExecutionRecovery.safe_fallback_reconstruction(
                checkpoint={"state": {"x": 1}},
                trace_context={"steps": [{"operation": "model_output", "payload": {"output_hash": "x"}}]},
            )

    def test_truncated_execution_log_triggers_deterministic_mismatch(self):
        baseline = _baseline_trace()
        seed = _baseline_seed(baseline)
        truncated = deepcopy(baseline)
        truncated["steps"] = truncated["steps"][:1]
        report = verify_terminal_state_replay_equivalence(
            terminal_state_seed=seed,
            execution_trace_context=truncated,
            enforce_dag_equivalence=True,
        )
        assert report["equivalent"] is False


class TestPhase4AEmbeddingDriftStress:
    def test_embedding_version_mismatch_mid_session_is_detected(self, bot):
        manager = bot.memory_manager.semantic
        first = manager._lock_embedding_version(model_name="embed-v1", vector_size=12)
        second = manager._lock_embedding_version(model_name="embed-v2", vector_size=12)
        assert first["drift_detected"] is False
        assert second["drift_detected"] is True

    def test_partial_lock_failure_does_not_silently_update_boundary(self, bot, monkeypatch):
        manager = bot.memory_manager.semantic
        manager._lock_embedding_version(model_name="embed-stable", vector_size=12)
        before = manager.embedding_version_lock()
        original = manager.with_embedding_cache_db

        def _faulty(operation, write=False):
            if write:
                raise OSError("lock storage unavailable")
            return original(operation, write=write)

        monkeypatch.setattr(manager, "with_embedding_cache_db", _faulty)

        with pytest.raises(OSError, match="lock storage unavailable"):
            manager._lock_embedding_version(model_name="embed-drifted", vector_size=12)

        monkeypatch.setattr(manager, "with_embedding_cache_db", original)
        after = manager.embedding_version_lock()
        assert after.get("model_name") == before.get("model_name")
