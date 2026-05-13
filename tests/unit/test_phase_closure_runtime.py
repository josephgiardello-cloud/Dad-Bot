from __future__ import annotations

import time

import pytest

from dadbot.core.phase_closure_runtime import PhaseClosureRuntime
from dadbot.core.tool_memory_causal_contract import CausalMemoryEntry
from dadbot.core.memory_feedback_policy import ToolMemoryProfile


def test_kernel_closure_contract_lock_and_side_effect_registry() -> None:
    runtime = PhaseClosureRuntime()
    request = runtime.kernel.lock_execution_contract(
        {
            "input": {"text": "hello", "attachments": []},
            "session_id": "s1",
        }
    )
    assert request.session_id == "s1"

    record = runtime.kernel.register_side_effect(
        effect_type="tool_write",
        subject="calendar",
        payload={"op": "set_reminder"},
        trace_id="tr-1",
    )
    assert record.effect_type == "tool_write"
    assert runtime.kernel.side_effect_registry.records(trace_id="tr-1")


def test_kernel_closure_uncertainty_and_drift_detection() -> None:
    runtime = PhaseClosureRuntime()
    uncertainty = runtime.kernel.model_tool_uncertainty(
        tool_name="calendar",
        status="partial",
        partial_confidence=0.5,
        historical_reliability=0.8,
        data_age_seconds=300,
    )
    assert 0.0 <= float(uncertainty["confidence"]["aggregate"]) <= 1.0

    first = runtime.kernel.detect_drift(slot="memory.turn_summary", value={"a": 1})
    second = runtime.kernel.detect_drift(slot="memory.turn_summary", value={"a": 2})
    assert first["drifted"] is False
    assert second["drifted"] is True


def test_memory_completion_field_model_and_consistency_checker() -> None:
    runtime = PhaseClosureRuntime()
    entry = runtime.memory.normalize_memory_field_model(
        {
            "summary": "Call mom tonight",
            "category": "family",
            "mood": "neutral",
        }
    )
    assert entry["summary"] == "Call mom tonight"

    runtime.memory.enforce_memory_lifecycle(
        from_state="active",
        to_state="reinforced",
        context="test",
    )

    before = [{"summary": "A", "importance_score": 0.9}]
    after = []
    with pytest.raises(Exception):
        runtime.memory.check_memory_consistency(before=before, after=after, context="test")


def test_world_model_snapshot_includes_temporal_and_causal_views() -> None:
    runtime = PhaseClosureRuntime()
    runtime.world.ingest_causal_entries(
        entries=[
            CausalMemoryEntry(
                tool_name="calendar",
                contract_version="v1",
                attempt=1,
                status="ok",
                causal_key="c1",
                timestamp_ms=1,
                latency_ms=10.0,
            ),
            CausalMemoryEntry(
                tool_name="email",
                contract_version="v1",
                attempt=1,
                status="ok",
                causal_key="c2",
                timestamp_ms=2,
                latency_ms=12.0,
            ),
        ],
        edges=[("c1", "c2", "calendar_reminder_drives_email")],
    )

    snapshot = runtime.world.snapshot(
        tool_profiles={"calendar": ToolMemoryProfile(tool_name="calendar", total_executions=1, success_count=1)}
    )
    assert snapshot.causal_nodes == 2
    assert snapshot.causal_edges == 1
    assert snapshot.temporal_axis["turn_started_at"]


def test_self_maintenance_health_score_and_ux_layer() -> None:
    runtime = PhaseClosureRuntime()
    report = runtime.maintenance.score_health(
        tool_execution_stats={"total": 20, "errors": 1, "p95_latency_ms": 120.0}
    )
    assert 0 <= report.overall_score <= 100

    plan = runtime.ux.dialogue_policy(
        session_id="s1",
        trace_id="tr1",
        user_input="Can you help me plan dinner and shopping?",
        memory_hits=1,
        tool_candidates=2,
    )
    assert plan["plan_id"].startswith("plan:s1:tr1")

    continuity = runtime.ux.persona_continuity(
        persona_history=[{"trait": "more patient", "applied_at": "2026-05-10T10:00:00"}],
        relationship_state={"trust_score": 0.8},
    )
    assert continuity["trait_count"] == 1

    pacing = runtime.ux.interaction_pacing(uncertainty_score=0.7, intent_type="request")
    assert pacing["response_delay_seconds"] > 0.0


def test_world_model_freshness_contract_enforces_staleness_bound() -> None:
    runtime = PhaseClosureRuntime()
    stale_execution_timestamp_ms = int((time.time() * 1000) - 60_000)
    with pytest.raises(RuntimeError, match="World Model Freshness Contract violation"):
        runtime.world.snapshot(
            tool_profiles={},
            metadata={
                "execution_timestamp_ms": stale_execution_timestamp_ms,
                "max_staleness_ms": 5,
            },
        )


def test_world_model_entity_binding_requires_existing_memory_or_derived() -> None:
    runtime = PhaseClosureRuntime()
    runtime.world.snapshot(
        tool_profiles={},
        metadata={
                "execution_timestamp_ms": int(time.time() * 1000),
            "memory_entries": [{"id": "m1"}],
            "entity_bindings": [
                {"entity_id": "e1", "memory_id": "m1", "source": "memory"},
                {"entity_id": "e2", "source": "derived"},
            ],
        },
    )

    with pytest.raises(RuntimeError, match="non-existent memory entry"):
        runtime.world.snapshot(
            tool_profiles={},
            metadata={
                "execution_timestamp_ms": int(time.time() * 1000),
                "memory_entries": [{"id": "m1"}],
                "entity_bindings": [
                    {"entity_id": "e1", "memory_id": "missing", "source": "memory"},
                ],
            },
        )


def test_kernel_closure_semantic_drift_scope_detects_behavioral_shift() -> None:
    runtime = PhaseClosureRuntime()
    first = runtime.kernel.detect_semantic_drift(
        domain="planner_consistency",
        metrics={"tool_match_rate": 0.9, "plan_depth": 4.0},
    )
    assert first["drifted"] is False
    second = runtime.kernel.detect_semantic_drift(
        domain="planner_consistency",
        metrics={"tool_match_rate": 0.5, "plan_depth": 1.0},
        relative_tolerance=0.2,
    )
    assert second["drifted"] is True


def test_self_repair_scope_forbids_core_mutation_actions() -> None:
    runtime = PhaseClosureRuntime()
    with pytest.raises(RuntimeError, match="SelfRepairScope violation"):
        runtime.maintenance.enforce_repair_scope(actions=["schema_mutation"])


def test_ux_layer_non_interference_blocks_core_decision_overrides() -> None:
    runtime = PhaseClosureRuntime()
    core_decision = {
        "strategy": "task_execution",
        "decision_outcome": "executed_tool",
        "response": "I will handle that now.",
    }
    projection = runtime.ux.apply_output_policy(
        core_decision=core_decision,
        output_projection={"style": "concise", "response": "Done."},
    )
    assert projection["response"] == "Done."

    with pytest.raises(RuntimeError, match="UX non-interference violation"):
        runtime.ux.apply_output_policy(
            core_decision=core_decision,
            output_projection={"strategy": "direct_answer", "response": "No"},
        )
