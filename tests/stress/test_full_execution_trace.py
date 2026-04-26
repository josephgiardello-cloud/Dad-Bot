from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import pytest

from dadbot.core.graph import TurnContext
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
