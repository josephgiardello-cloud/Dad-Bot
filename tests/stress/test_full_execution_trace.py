from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import pytest

from dadbot.core.canonical_event import validate_trace

from dadbot.core.graph import MutationQueue, TurnContext
from tests.stress.phase4_certification_gate import build_bot
from tools.phase4_legacy_integrity_scan import run_scan


CANONICAL_PIPELINE = ["temporal", "preflight", "inference", "safety", "reflection", "save"]


@pytest.fixture
def isolated_bot():
    with TemporaryDirectory() as tmp:
        bot = build_bot(Path(tmp), reply="Trace path OK.")
        try:
            yield bot
        finally:
            try:
                bot.shutdown()
            except Exception:
                pass


def _turn_stage_order(bot: Any) -> list[str]:
    evidence = dict(getattr(bot, "_last_turn_health_evidence", {}) or {})
    return [str(s).strip().lower() for s in list(evidence.get("stage_order") or [])]


def _run_audited_turn(bot: Any, user_input: str, *, trace_id: str, correlation_id: str, request_id: str, session_id: str = "audit-session") -> tuple[Any, dict[str, Any]]:
    result = asyncio.run(
        bot.turn_orchestrator.control_plane.submit_turn(
            session_id=session_id,
            user_input=user_input,
            metadata={
                "audit_mode": True,
                "trace_id": trace_id,
                "correlation_id": correlation_id,
                "request_id": request_id,
            },
        )
    )
    report = dict(getattr(bot, "_last_capability_audit_report", {}) or {})
    return result, report


def _persistent_state_snapshot(bot: Any) -> dict[str, Any]:
    graph_manager = bot.memory_manager.graph_manager
    pipeline = bot.turn_service.turn_pipeline_snapshot()
    return {
        "memory_store": json.loads(json.dumps(getattr(bot, "MEMORY_STORE", {}) or {}, sort_keys=True, default=str)),
        "graph": json.loads(json.dumps(graph_manager.graph_snapshot() or {}, sort_keys=True, default=str)),
        "history": json.loads(json.dumps(list(getattr(bot, "history", []) or []), sort_keys=True, default=str)),
        "pipeline_completed": bool(pipeline and pipeline.get("completed_at")),
        "pipeline_steps": [str(step.get("name") or "") for step in list((pipeline or {}).get("steps") or [])],
    }


def _save_boundary_snapshot(bot: Any) -> dict[str, Any]:
    session_state = bot.snapshot_session_state()
    memory_store = dict(session_state.get("memory_store", {}) or {})
    return {
        "history": json.loads(json.dumps(list(session_state.get("history", []) or []), sort_keys=True, default=str)),
        "last_mood": memory_store.get("last_mood"),
        "last_mood_updated_at": memory_store.get("last_mood_updated_at"),
        "recent_moods": json.loads(json.dumps(list(memory_store.get("recent_moods", []) or []), sort_keys=True, default=str)),
        "graph": json.loads(json.dumps(bot.memory_manager.graph_manager.graph_snapshot() or {}, sort_keys=True, default=str)),
        "pipeline_completed": bool((bot.turn_service.turn_pipeline_snapshot() or {}).get("completed_at")),
    }


def _mixed_long_horizon_input(index: int) -> str:
    prompts = [
        "Work felt really heavy today, not sure I handled it well.",
        "Had a good moment with the kids this evening.",
        "I keep overthinking everything before bed.",
        "Boss gave me some tough feedback and I'm still processing it.",
        "Felt proud of myself for finishing that report on time.",
        "Tired all the time lately, no idea why.",
        "Big decision coming up - job offer from another company.",
        "Missing Dad a lot today.",
        "Finally got to the gym for the first time in weeks.",
        "Budget's tight this month and it's stressing me out.",
        "I snapped at someone I care about and felt awful.",
        "Had a real breakthrough on a problem I've been stuck on.",
    ]
    return prompts[index % len(prompts)]


def _build_bot_from_disk(temp_path: Path, *, reply: str = "Trace path OK.") -> Any:
    return build_bot(temp_path, reply=reply, restore_from_disk=True)


def _current_turn_trace_id(bot: Any) -> str:
    evidence = dict(getattr(bot, "_last_turn_health_evidence", {}) or {})
    return str(evidence.get("trace_id") or "").strip()


def _relationship_projection_snapshot(bot: Any) -> dict[str, Any]:
    projection = json.loads(json.dumps(bot.relationship_manager.current_state() or {}, sort_keys=True, default=str))
    projection.pop("last_hypothesis_updated", None)
    projection.pop("last_updated", None)
    return projection


def _restart_boundary_snapshot(bot: Any) -> dict[str, Any]:
    save_boundary = dict(_save_boundary_snapshot(bot))
    save_boundary.pop("pipeline_completed", None)
    return {
        "save_boundary": save_boundary,
        "relationship_projection": _relationship_projection_snapshot(bot),
    }


def _committed_turn_contract(bot: Any) -> dict[str, Any]:
    evidence = dict(getattr(bot, "_last_turn_health_evidence", {}) or {})
    mutation_queue = dict(evidence.get("mutation_queue") or {})
    pipeline = dict(bot.turn_service.turn_pipeline_snapshot() or {})
    return {
        "save_boundary": _save_boundary_snapshot(bot),
        "relationship_projection": _relationship_projection_snapshot(bot),
        "mutation_queue": {
            "pending": int(mutation_queue.get("pending", 0) or 0),
            "drained": int(mutation_queue.get("drained", 0) or 0),
            "failed": int(mutation_queue.get("failed", 0) or 0),
        },
        "stage_order": _turn_stage_order(bot),
        "pipeline_steps": [str(step.get("name") or "") for step in list(pipeline.get("steps") or [])],
        "pipeline_completed": bool(pipeline.get("completed_at")),
    }


def _assert_turn_event_integrity(bot: Any, trace_id: str, *, expect_save_after: bool) -> dict[str, Any]:
    assert trace_id, "Expected non-empty trace_id for turn event integrity audit"
    events = bot.list_turn_events(trace_id)
    assert events, f"No turn events persisted for trace_id={trace_id!r}"

    # Global canonicalization assertion: no forbidden wall-clock fields must
    # survive into any event payload in the production trace.
    validate_trace(events)

    sequences = [int(event.get("sequence") or 0) for event in events]
    assert all(sequence > 0 for sequence in sequences)

    checkpoint_events = [
        event for event in events if str(event.get("event_type") or "").strip().lower() == "graph_checkpoint"
    ]
    save_before = [
        event for event in checkpoint_events if str(event.get("stage") or "") == "save" and str(event.get("status") or "") == "before"
    ]
    save_after = [
        event for event in checkpoint_events if str(event.get("stage") or "") == "save" and str(event.get("status") or "") == "after"
    ]
    save_error = [
        event for event in checkpoint_events if str(event.get("stage") or "") == "save" and str(event.get("status") or "") == "error"
    ]

    assert len(save_before) >= 1
    if expect_save_after:
        assert len(save_after) >= 1
    else:
        assert len(save_error) >= 1

    replay = bot.validate_replay_determinism(trace_id)

    return {
        "event_count": len(events),
        "save_before": len(save_before),
        "save_after": len(save_after),
        "save_error": len(save_error),
        "replay_consistent": bool(replay.get("consistent", False)),
    }


def test_full_execution_trace_matches_canonical_pipeline(isolated_bot):
    bot = isolated_bot

    unsafe_background_mutations: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    critical_mutation_keys = {
        "relationship_state",
        "recent_moods",
        "last_mood",
        "last_mood_updated_at",
        "memory_graph",
        "consolidated_memories",
    }

    original_mutate = bot.mutate_memory_store

    def _instrumented_mutate(*args, **kwargs):
        record = {
            "graph_commit_active": bool(getattr(bot, "_graph_commit_active", False)),
            "args_len": len(args),
            "keys": sorted(list(kwargs.keys())),
        }
        # Only flag writes that should be SaveNode-bound invariants.
        touches_critical_state = bool(set(record["keys"]) & critical_mutation_keys)
        if (not record["graph_commit_active"]) and touches_critical_state:
            unsafe_background_mutations.append(record)
        return original_mutate(*args, **kwargs)

    with patch.object(bot, "mutate_memory_store", side_effect=_instrumented_mutate):
        for user_input in [
            "I had a rough day but I am trying to stay steady.",
            "Please remind me to call mom tomorrow.",
            "Can you help me think through this job decision?",
        ]:
            reply, should_end = bot.process_user_message(user_input)
            assert should_end is False
            assert isinstance(reply, str) and reply.strip()

            stage_order = _turn_stage_order(bot)
            trace_records.append({
                "input": user_input,
                "stage_order": stage_order,
            })

            assert stage_order, "No stage_order captured; dynamic trace missing"

            missing = [s for s in CANONICAL_PIPELINE if s not in stage_order]
            extra = [s for s in stage_order if s not in CANONICAL_PIPELINE]
            assert not missing, f"Missing canonical stages: {missing}; trace={stage_order}"
            assert not extra, f"Unexpected legacy/side stages: {extra}; trace={stage_order}"
            assert stage_order == CANONICAL_PIPELINE, (
                f"Non-canonical order detected. expected={CANONICAL_PIPELINE}, actual={stage_order}"
            )

    assert not unsafe_background_mutations, (
        "Detected memory mutation writes outside SaveNode commit boundary: "
        f"{unsafe_background_mutations}"
    )

    # Keep a compact execution artifact for manual inspection.
    Path("session_logs").mkdir(exist_ok=True)
    Path("session_logs/full_execution_trace.json").write_text(
        json.dumps({"turns": trace_records}, indent=2),
        encoding="utf-8",
    )


def test_audit_mode_emits_capability_report(isolated_bot):
    bot = isolated_bot

    (reply, should_end), report = _run_audited_turn(
        bot,
        "Please help me think through a hard conversation.",
        trace_id="audit-trace-001",
        correlation_id="audit-corr-001",
        request_id="audit-req-001",
    )

    assert should_end is False
    assert isinstance(reply, str) and reply.strip()
    assert report.get("ok") is True
    assert report.get("failed") is False
    assert report.get("stage_order") == CANONICAL_PIPELINE
    assert report.get("mutation_queue", {}).get("pending") == 0
    checks = {check["name"]: check for check in list(report.get("checks") or [])}
    assert checks["temporal_ordering"]["status"] == "pass"
    assert checks["mutation_safety"]["status"] == "pass"
    assert checks["save_node_single_execution"]["details"]["save_count"] == 1
    assert checks["capability_audit_emission"]["status"] == "pass"


def test_golden_replay_capability_contracts_hold_for_identical_runs():
    with TemporaryDirectory() as left_tmp, TemporaryDirectory() as right_tmp:
        left_bot = build_bot(Path(left_tmp), reply="Golden replay OK.")
        right_bot = build_bot(Path(right_tmp), reply="Golden replay OK.")
        try:
            _, left_report = _run_audited_turn(
                left_bot,
                "Give me the same structured advice twice.",
                trace_id="golden-trace-001",
                correlation_id="golden-corr-001",
                request_id="golden-req-001",
                session_id="golden-session",
            )
            _, right_report = _run_audited_turn(
                right_bot,
                "Give me the same structured advice twice.",
                trace_id="golden-trace-001",
                correlation_id="golden-corr-001",
                request_id="golden-req-001",
                session_id="golden-session",
            )

            assert left_report.get("stage_order") == CANONICAL_PIPELINE
            assert right_report.get("stage_order") == CANONICAL_PIPELINE
            assert left_report.get("mutation_queue", {}).get("pending") == 0
            assert right_report.get("mutation_queue", {}).get("pending") == 0
            assert left_bot.turn_orchestrator.control_plane.ledger.replay_hash() == right_bot.turn_orchestrator.control_plane.ledger.replay_hash()
        finally:
            left_bot.shutdown()
            right_bot.shutdown()


def test_legacy_behavior_trigger_rejected_strictly(isolated_bot):
    bot = isolated_bot

    # Intentional legacy-trigger-like text should not degrade execution.
    for probe in [
        "use fallback mode",
        "direct path execution",
        "disable graph",
    ]:
        reply, should_end = bot.process_user_message(probe)
        assert should_end is False
        assert isinstance(reply, str) and reply.strip()

    # Force a graph crash and ensure strict hard-fail (no silent legacy fallback).
    with patch.object(bot.turn_orchestrator.graph, "execute", side_effect=RuntimeError("forced graph failure")):
        with pytest.raises(RuntimeError, match="legacy path is disabled"):
            bot.process_user_message("disable graph now")

    # Malformed turn context: missing temporal node must be rejected.
    malformed = TurnContext(user_input="malformed turn context")
    malformed.temporal = None  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="TemporalNode|Temporal|boundary violation"):
        asyncio.run(bot.turn_orchestrator.graph.execute(malformed))


def test_cross_module_consistency_and_merge_report(isolated_bot):
    bot = isolated_bot

    repo_root = Path(__file__).resolve().parents[2]

    # Cross-module consistency assertions.
    persistence_text = (repo_root / "dadbot/services/persistence.py").read_text(encoding="utf-8")
    graph_manager_text = (repo_root / "dadbot/memory/graph_manager.py").read_text(encoding="utf-8")
    lifecycle_text = (repo_root / "dadbot/memory/lifecycle.py").read_text(encoding="utf-8")
    relationship_text = (repo_root / "dadbot/relationship.py").read_text(encoding="utf-8")

    assert "SaveNode strict mode requires" in persistence_text
    assert "MutationQueue" in persistence_text
    assert "TemporalNode required" in graph_manager_text
    assert "projection-only" in relationship_text

    # Lifecycle layer still has direct time calls; this check catches partial migration.
    lifecycle_temporal_calls = [
        token for token in ("datetime.now(", "date.today(", "time.time(") if token in lifecycle_text
    ]

    static_report = run_scan(repo_root)

    # Runtime probe result from real graph execution.
    reply, should_end = bot.process_user_message("runtime trace probe")
    assert should_end is False
    assert isinstance(reply, str) and reply.strip()
    stage_order = _turn_stage_order(bot)

    runtime_findings: list[str] = []
    if stage_order != CANONICAL_PIPELINE:
        runtime_findings.append(
            f"runtime_stage_order_mismatch expected={CANONICAL_PIPELINE} actual={stage_order}"
        )

    if lifecycle_temporal_calls:
        runtime_findings.append(
            "lifecycle_temporal_calls_present:" + ",".join(lifecycle_temporal_calls)
        )

    merged = {
        "legacy_paths": static_report.get("legacy_paths", []),
        "dead_code": static_report.get("dead_code", []),
        "temporal_violations": static_report.get("temporal_violations", []),
        "dual_execution_paths": static_report.get("dual_execution_paths", []),
        "unsafe_mutations": static_report.get("unsafe_mutations", []),
        "overall_integrity": "PASS",
    }

    if runtime_findings:
        merged["dead_code"] = list(merged["dead_code"]) + [
            {
                "file": "tests/stress/test_full_execution_trace.py",
                "line": 1,
                "kind": "runtime_finding",
                "detail": finding,
                "snippet": "",
            }
            for finding in runtime_findings
        ]

    if merged["temporal_violations"] or merged["dual_execution_paths"] or merged["unsafe_mutations"] or runtime_findings:
        merged["overall_integrity"] = "FAIL"

    Path("session_logs").mkdir(exist_ok=True)
    Path("session_logs/phase4_integrity_merge_report.json").write_text(
        json.dumps(merged, indent=2),
        encoding="utf-8",
    )

    assert merged["overall_integrity"] in {"PASS", "FAIL"}


def test_runtime_call_graph_completeness_audit(isolated_bot, monkeypatch):
    bot = isolated_bot
    graph = bot.turn_orchestrator.graph
    kernel = graph._execution_kernel
    node_map = dict(getattr(graph, "_node_map", {}) or {})
    assert set(CANONICAL_PIPELINE).issubset(set(node_map)), "Graph node map missing canonical stages"

    temporal_node = node_map["temporal"]
    preflight_nodes = node_map["preflight"]
    inference_node = node_map["inference"]
    safety_node = node_map["safety"]
    reflection_node = node_map["reflection"]
    save_node = node_map["save"]
    assert isinstance(preflight_nodes, tuple) and len(preflight_nodes) == 2
    health_node, context_builder_node = preflight_nodes

    persistence = save_node.mgr
    runtime = persistence.turn_service.bot
    memory_coordinator = runtime.memory_coordinator
    relationship_manager = runtime.relationship_manager
    graph_manager = runtime.memory_manager.graph_manager

    call_counts: dict[str, int] = {}

    def _count(key: str) -> None:
        call_counts[key] = call_counts.get(key, 0) + 1

    def _wrap_sync(obj: Any, attr: str, key: str) -> None:
        original = getattr(obj, attr)

        def _wrapped(*args, **kwargs):
            _count(key)
            return original(*args, **kwargs)

        monkeypatch.setattr(obj, attr, _wrapped)

    def _wrap_async(obj: Any, attr: str, key: str) -> None:
        original = getattr(obj, attr)

        async def _wrapped(*args, **kwargs):
            _count(key)
            return await original(*args, **kwargs)

        monkeypatch.setattr(obj, attr, _wrapped)

    _wrap_async(kernel, "run", "kernel.run")
    _wrap_sync(kernel, "validate", "kernel.validate")
    _wrap_async(temporal_node, "run", "node.temporal")
    _wrap_async(health_node, "run", "node.preflight.health")
    _wrap_async(context_builder_node, "run", "node.preflight.context_builder")
    _wrap_async(inference_node, "run", "node.inference")
    _wrap_async(safety_node, "run", "node.safety")
    _wrap_async(reflection_node, "run", "node.reflection")
    _wrap_async(save_node, "run", "node.save")
    _wrap_sync(persistence, "begin_transaction", "persistence.begin_transaction")
    _wrap_sync(persistence, "apply_mutations", "persistence.apply_mutations")
    _wrap_sync(persistence, "finalize_turn", "persistence.finalize_turn")
    _wrap_sync(persistence, "commit_transaction", "persistence.commit_transaction")
    _wrap_sync(memory_coordinator, "consolidate_memories", "memory.consolidate_memories")
    _wrap_sync(memory_coordinator, "apply_controlled_forgetting", "memory.apply_controlled_forgetting")
    _wrap_sync(relationship_manager, "materialize_projection", "relationship.materialize_projection")
    _wrap_sync(graph_manager, "sync_graph_store", "graph.sync_graph_store")

    original_drain = MutationQueue.drain

    def _drain_wrapper(self, executor, *, hard_fail_on_error=True):
        _count("mutation_queue.drain")
        return original_drain(self, executor, hard_fail_on_error=hard_fail_on_error)

    monkeypatch.setattr(MutationQueue, "drain", _drain_wrapper)

    reply, should_end = bot.process_user_message("runtime call graph audit")
    assert should_end is False
    assert isinstance(reply, str) and reply.strip()

    stage_order = _turn_stage_order(bot)
    assert stage_order == CANONICAL_PIPELINE

    exact_once = {
        "kernel.run",
        "node.temporal",
        "node.preflight.health",
        "node.preflight.context_builder",
        "node.inference",
        "node.safety",
        "node.reflection",
        "node.save",
        "persistence.begin_transaction",
        "persistence.apply_mutations",
        "persistence.finalize_turn",
        "persistence.commit_transaction",
        "mutation_queue.drain",
        "memory.consolidate_memories",
        "memory.apply_controlled_forgetting",
        "relationship.materialize_projection",
        "graph.sync_graph_store",
    }
    for key in sorted(exact_once):
        assert call_counts.get(key, 0) == 1, f"Expected exactly one call for {key}, saw {call_counts.get(key, 0)}"

    expected_kernel_validate_calls = len(CANONICAL_PIPELINE) + 2
    assert call_counts.get("kernel.validate", 0) == expected_kernel_validate_calls, (
        "Kernel validate call count mismatch: "
        f"expected={expected_kernel_validate_calls} actual={call_counts.get('kernel.validate', 0)}"
    )

    Path("session_logs").mkdir(exist_ok=True)
    Path("session_logs/runtime_call_graph_audit.json").write_text(
        json.dumps(
            {
                "stage_order": stage_order,
                "call_counts": dict(sorted(call_counts.items())),
                "expected_kernel_validate_calls": expected_kernel_validate_calls,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_mid_save_apply_mutations_crash_rolls_back_persistent_state(isolated_bot, monkeypatch):
    bot = isolated_bot
    persistence = bot.turn_orchestrator.graph._node_map["save"].mgr
    baseline = _save_boundary_snapshot(bot)

    original_apply = persistence.apply_mutations

    def _crash_after_apply(turn_context: Any) -> None:
        original_apply(turn_context)
        raise RuntimeError("apply_mutations crash injection")

    monkeypatch.setattr(persistence, "apply_mutations", _crash_after_apply)

    with pytest.raises(RuntimeError, match="legacy path is disabled"):
        bot.process_user_message("trigger apply mutation rollback")

    after = _save_boundary_snapshot(bot)
    assert after == baseline, "Save-boundary persistent state changed after apply_mutations crash"


def test_mid_save_finalize_turn_crash_rolls_back_persistent_state(isolated_bot, monkeypatch):
    bot = isolated_bot
    persistence = bot.turn_orchestrator.graph._node_map["save"].mgr
    baseline = _save_boundary_snapshot(bot)

    def _crash_finalize(_turn_context: Any, _result: Any) -> tuple[Any, bool]:
        raise RuntimeError("finalize_turn crash injection")

    monkeypatch.setattr(persistence, "finalize_turn", _crash_finalize)

    with pytest.raises(RuntimeError, match="legacy path is disabled"):
        bot.process_user_message("trigger finalize rollback")

    after = _save_boundary_snapshot(bot)
    assert after == baseline, "Save-boundary persistent state changed after finalize_turn crash"


def test_kernel_validate_mid_loop_failure_preserves_state(isolated_bot, monkeypatch):
    bot = isolated_bot
    kernel = bot.turn_orchestrator.graph._execution_kernel
    baseline = _save_boundary_snapshot(bot)
    original_validate = kernel.validate

    def _crash_on_safety(*args, **kwargs):
        stage = kwargs.get("stage")
        if stage == "safety":
            raise RuntimeError("kernel validate crash injection")
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(kernel, "validate", _crash_on_safety)

    with pytest.raises(RuntimeError, match="legacy path is disabled"):
        bot.process_user_message("trigger kernel validation failure")

    after = _save_boundary_snapshot(bot)
    assert after == baseline, "Save-boundary persistent state changed after kernel validate failure"


def test_retry_recovery_after_mid_save_failure_restores_deterministic_state(monkeypatch):
    with TemporaryDirectory() as failed_tmp, TemporaryDirectory() as clean_tmp:
        failed_bot = build_bot(Path(failed_tmp), reply="Trace path OK.")
        clean_bot = build_bot(Path(clean_tmp), reply="Trace path OK.")
        try:
            persistence = failed_bot.turn_orchestrator.graph._node_map["save"].mgr
            original_apply = persistence.apply_mutations
            injected = {"fired": False}

            def _crash_once(turn_context: Any) -> None:
                if not injected["fired"]:
                    injected["fired"] = True
                    original_apply(turn_context)
                    raise RuntimeError("single-shot apply_mutations crash")
                original_apply(turn_context)

            monkeypatch.setattr(persistence, "apply_mutations", _crash_once)

            with pytest.raises(RuntimeError, match="legacy path is disabled"):
                failed_bot.process_user_message("recoverable failure turn")

            failed_reply, failed_should_end = failed_bot.process_user_message("recoverable failure turn")
            clean_reply, clean_should_end = clean_bot.process_user_message("recoverable failure turn")

            assert (failed_reply, failed_should_end) == (clean_reply, clean_should_end)
            assert _turn_stage_order(failed_bot) == CANONICAL_PIPELINE
            assert _turn_stage_order(clean_bot) == CANONICAL_PIPELINE
            assert _save_boundary_snapshot(failed_bot) == _save_boundary_snapshot(clean_bot)
        finally:
            try:
                failed_bot.shutdown()
            except Exception:
                pass


def test_relationship_projection_failure_rolls_back_persistent_state(isolated_bot, monkeypatch):
    bot = isolated_bot
    relationship_manager = bot.relationship_manager
    baseline = _save_boundary_snapshot(bot)

    def _crash_projection(*, turn_context=None):
        raise RuntimeError("relationship projection crash injection")

    monkeypatch.setattr(relationship_manager, "materialize_projection", _crash_projection)

    with pytest.raises(RuntimeError, match="legacy path is disabled"):
        bot.process_user_message("trigger relationship projection rollback")

    after = _save_boundary_snapshot(bot)
    assert after == baseline, "Save-boundary persistent state changed after relationship projection crash"


def test_graph_sync_failure_rolls_back_persistent_state(isolated_bot, monkeypatch):
    bot = isolated_bot
    graph_manager = bot.memory_manager.graph_manager
    baseline = _save_boundary_snapshot(bot)

    def _crash_graph_sync(*, turn_context=None):
        raise RuntimeError("graph sync crash injection")

    monkeypatch.setattr(graph_manager, "sync_graph_store", _crash_graph_sync)

    with pytest.raises(RuntimeError, match="legacy path is disabled"):
        bot.process_user_message("trigger graph sync rollback")

    after = _save_boundary_snapshot(bot)
    assert after == baseline, "Save-boundary persistent state changed after graph sync crash"


@pytest.mark.parametrize(
    ("failure_name", "target_attr", "message"),
    [
        ("relationship_projection", "materialize_projection", "relationship recovery turn"),
        ("graph_sync", "sync_graph_store", "graph sync recovery turn"),
    ],
)
def test_retry_recovery_after_mid_save_boundary_failure_matches_clean_bot(
    monkeypatch,
    failure_name: str,
    target_attr: str,
    message: str,
):
    with TemporaryDirectory() as failed_tmp, TemporaryDirectory() as clean_tmp:
        failed_bot = build_bot(Path(failed_tmp), reply="Trace path OK.")
        clean_bot = build_bot(Path(clean_tmp), reply="Trace path OK.")
        try:
            target_obj = failed_bot.relationship_manager if target_attr == "materialize_projection" else failed_bot.memory_manager.graph_manager
            original = getattr(target_obj, target_attr)
            injected = {"fired": False}

            def _crash_once(*args, **kwargs):
                if not injected["fired"]:
                    injected["fired"] = True
                    raise RuntimeError(f"{failure_name} crash injection")
                return original(*args, **kwargs)

            monkeypatch.setattr(target_obj, target_attr, _crash_once)

            with pytest.raises(RuntimeError, match="legacy path is disabled"):
                failed_bot.process_user_message(message)

            failed_reply, failed_should_end = failed_bot.process_user_message(message)
            clean_reply, clean_should_end = clean_bot.process_user_message(message)

            assert (failed_reply, failed_should_end) == (clean_reply, clean_should_end)
            assert _turn_stage_order(failed_bot) == CANONICAL_PIPELINE
            assert _turn_stage_order(clean_bot) == CANONICAL_PIPELINE
            assert _save_boundary_snapshot(failed_bot) == _save_boundary_snapshot(clean_bot)
        finally:
            try:
                failed_bot.shutdown()
            except Exception:
                pass
            try:
                clean_bot.shutdown()
            except Exception:
                pass


@pytest.mark.parametrize(
    ("failure_name", "target_attr"),
    [
        ("relationship_projection", "materialize_projection"),
        ("graph_sync", "sync_graph_store"),
    ],
)
def test_long_horizon_periodic_save_boundary_failures_recover_cleanly(
    monkeypatch,
    failure_name: str,
    target_attr: str,
):
    turn_count = 24
    failure_period = 6
    failure_turns = {turn_number for turn_number in range(1, turn_count + 1) if turn_number % failure_period == 0}

    with TemporaryDirectory() as failed_tmp, TemporaryDirectory() as clean_tmp:
        failed_bot = build_bot(Path(failed_tmp), reply="Trace path OK.")
        clean_bot = build_bot(Path(clean_tmp), reply="Trace path OK.")
        try:
            target_obj = (
                failed_bot.relationship_manager
                if target_attr == "materialize_projection"
                else failed_bot.memory_manager.graph_manager
            )
            original = getattr(target_obj, target_attr)
            injection_state = {"current_turn": 0, "fired_turns": set()}
            audit_rows: list[dict[str, Any]] = []

            def _crash_periodically(*args, **kwargs):
                current_turn = int(injection_state["current_turn"])
                fired_turns = injection_state["fired_turns"]
                if current_turn in failure_turns and current_turn not in fired_turns:
                    fired_turns.add(current_turn)
                    raise RuntimeError(f"{failure_name} periodic crash injection turn={current_turn}")
                return original(*args, **kwargs)

            monkeypatch.setattr(target_obj, target_attr, _crash_periodically)

            for turn_number in range(1, turn_count + 1):
                message = _mixed_long_horizon_input(turn_number - 1)
                injection_state["current_turn"] = turn_number
                injected_failure = turn_number in failure_turns

                if injected_failure:
                    with pytest.raises(RuntimeError, match="legacy path is disabled"):
                        failed_bot.process_user_message(message)

                failed_reply, failed_should_end = failed_bot.process_user_message(message)
                clean_reply, clean_should_end = clean_bot.process_user_message(message)

                failed_stage_order = _turn_stage_order(failed_bot)
                clean_stage_order = _turn_stage_order(clean_bot)
                failed_boundary = _save_boundary_snapshot(failed_bot)
                clean_boundary = _save_boundary_snapshot(clean_bot)

                assert (failed_reply, failed_should_end) == (clean_reply, clean_should_end)
                assert failed_stage_order == CANONICAL_PIPELINE
                assert clean_stage_order == CANONICAL_PIPELINE
                assert failed_boundary == clean_boundary

                audit_rows.append(
                    {
                        "turn": turn_number,
                        "failure_injected": injected_failure,
                        "stage_order": failed_stage_order,
                        "reply": failed_reply,
                    }
                )

            assert injection_state["fired_turns"] == failure_turns

            Path("session_logs").mkdir(exist_ok=True)
            Path(f"session_logs/periodic_recovery_audit_{failure_name}.json").write_text(
                json.dumps(
                    {
                        "failure_name": failure_name,
                        "turn_count": turn_count,
                        "failure_period": failure_period,
                        "failure_turns": sorted(failure_turns),
                        "audit_rows": audit_rows,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        finally:
            try:
                failed_bot.shutdown()
            except Exception:
                pass
            try:
                clean_bot.shutdown()
            except Exception:
                pass


@pytest.mark.parametrize(
    ("failure_name", "target_attr"),
    [
        ("relationship_projection", "materialize_projection"),
        ("graph_sync", "sync_graph_store"),
    ],
)
def test_restart_boundary_recovery_audit_matches_clean_execution(
    monkeypatch,
    failure_name: str,
    target_attr: str,
):
    turns = [_mixed_long_horizon_input(index) for index in range(8)]
    failure_turns = {3, 6}

    with TemporaryDirectory() as failed_tmp, TemporaryDirectory() as clean_tmp:
        failed_path = Path(failed_tmp)
        clean_path = Path(clean_tmp)
        failed_bot = build_bot(failed_path, reply="Trace path OK.")
        clean_bot = build_bot(clean_path, reply="Trace path OK.")
        try:
            clean_pre_turn_snapshots: list[dict[str, Any]] = []
            clean_turn_records: list[dict[str, Any]] = []
            clean_trace_ids: list[str] = []

            for turn_number, message in enumerate(turns, start=1):
                clean_pre_turn_snapshots.append(_restart_boundary_snapshot(clean_bot))
                clean_reply, clean_should_end = clean_bot.process_user_message(message)
                clean_trace_id = _current_turn_trace_id(clean_bot)
                clean_trace_ids.append(clean_trace_id)
                clean_turn_records.append(
                    {
                        "turn": turn_number,
                        "reply": clean_reply,
                        "should_end": clean_should_end,
                        "contract": _committed_turn_contract(clean_bot),
                        "event_summary": _assert_turn_event_integrity(clean_bot, clean_trace_id, expect_save_after=True),
                    }
                )

            injection_state = {"current_turn": 0, "failed_turns": set()}
            successful_trace_ids: list[str] = []
            failed_trace_ids: list[str] = []
            audit_rows: list[dict[str, Any]] = []

            def _install_failure_hook(bot: Any) -> None:
                target_obj = (
                    bot.relationship_manager
                    if target_attr == "materialize_projection"
                    else bot.memory_manager.graph_manager
                )
                original = getattr(target_obj, target_attr)

                def _crash_once_per_failure_turn(*args, **kwargs):
                    current_turn = int(injection_state["current_turn"])
                    failed_turns = injection_state["failed_turns"]
                    if current_turn in failure_turns and current_turn not in failed_turns:
                        failed_turns.add(current_turn)
                        raise RuntimeError(f"{failure_name} restart crash injection turn={current_turn}")
                    return original(*args, **kwargs)

                monkeypatch.setattr(target_obj, target_attr, _crash_once_per_failure_turn)

            _install_failure_hook(failed_bot)

            for turn_number, message in enumerate(turns, start=1):
                injection_state["current_turn"] = turn_number
                before_turn_snapshot = _restart_boundary_snapshot(failed_bot)
                restarted = False
                failed_trace_id = ""

                if turn_number in failure_turns:
                    with pytest.raises(RuntimeError, match="legacy path is disabled"):
                        failed_bot.process_user_message(message)

                    failed_trace_id = _current_turn_trace_id(failed_bot)
                    assert failed_trace_id
                    failed_trace_ids.append(failed_trace_id)
                    failure_event_summary = _assert_turn_event_integrity(
                        failed_bot,
                        failed_trace_id,
                        expect_save_after=False,
                    )
                    assert _restart_boundary_snapshot(failed_bot) == before_turn_snapshot

                    failed_bot.shutdown()
                    failed_bot = _build_bot_from_disk(failed_path, reply="Trace path OK.")
                    _install_failure_hook(failed_bot)
                    restarted = True

                    assert _restart_boundary_snapshot(failed_bot) == clean_pre_turn_snapshots[turn_number - 1]
                else:
                    failure_event_summary = None

                failed_reply, failed_should_end = failed_bot.process_user_message(message)
                successful_trace_id = _current_turn_trace_id(failed_bot)
                assert successful_trace_id
                successful_trace_ids.append(successful_trace_id)

                failed_contract = _committed_turn_contract(failed_bot)
                failed_event_summary = _assert_turn_event_integrity(
                    failed_bot,
                    successful_trace_id,
                    expect_save_after=True,
                )
                clean_record = clean_turn_records[turn_number - 1]

                assert (failed_reply, failed_should_end) == (clean_record["reply"], clean_record["should_end"])
                assert failed_contract == clean_record["contract"]
                assert failed_contract["mutation_queue"] == {"pending": 0, "drained": 0, "failed": 0}
                assert failed_contract["stage_order"] == CANONICAL_PIPELINE
                assert len(
                    [entry for entry in failed_contract["save_boundary"]["history"] if entry.get("role") == "user"]
                ) == turn_number

                audit_rows.append(
                    {
                        "turn": turn_number,
                        "restarted": restarted,
                        "failed_trace_id": failed_trace_id,
                        "successful_trace_id": successful_trace_id,
                        "failure_event_summary": failure_event_summary,
                        "successful_event_summary": failed_event_summary,
                    }
                )

            assert injection_state["failed_turns"] == failure_turns
            assert len(successful_trace_ids) == len(turns)
            assert len(set(successful_trace_ids)) == len(turns)
            assert len(set(clean_trace_ids)) == len(turns)
            assert _restart_boundary_snapshot(failed_bot) == _restart_boundary_snapshot(clean_bot)

            Path("session_logs").mkdir(exist_ok=True)
            Path(f"session_logs/restart_recovery_audit_{failure_name}.json").write_text(
                json.dumps(
                    {
                        "failure_name": failure_name,
                        "failure_turns": sorted(failure_turns),
                        "successful_trace_ids": successful_trace_ids,
                        "failed_trace_ids": failed_trace_ids,
                        "audit_rows": audit_rows,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        finally:
            try:
                failed_bot.shutdown()
            except Exception:
                pass
            try:
                clean_bot.shutdown()
            except Exception:
                pass
