from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dadbot.core.dadbot import DadBot

RUN_DIR = ROOT / "session_logs" / "phase4_launch_readiness"
REPORT_PATH = ROOT / "phase4_launch_readiness_report.json"


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


@contextmanager
def isolated_runtime_paths(base_dir: Path):
    base_dir.mkdir(parents=True, exist_ok=True)
    profile_path = base_dir / "dad_profile.json"
    memory_path = base_dir / "dad_memory.json"
    if not profile_path.exists():
        template_profile = ROOT / "dad_profile.template.json"
        source_profile = ROOT / "dad_profile.json"
        if template_profile.exists():
            profile_path.write_text(template_profile.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        elif source_profile.exists():
            profile_path.write_text(source_profile.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        else:
            profile_path.write_text("{}", encoding="utf-8")
    if not memory_path.exists():
        source_memory = ROOT / "dad_memory.json"
        if source_memory.exists():
            memory_path.write_text(source_memory.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        else:
            memory_path.write_text("{}", encoding="utf-8")

    env_overrides = {
        "DADBOT_PROFILE_PATH": str(base_dir / "dad_profile.json"),
        "DADBOT_MEMORY_PATH": str(base_dir / "dad_memory.json"),
        "DADBOT_SEMANTIC_DB_PATH": str(base_dir / "dad_memory_semantic.sqlite3"),
        "DADBOT_GRAPH_DB_PATH": str(base_dir / "dad_memory_graph.sqlite3"),
        "DADBOT_DISABLE_GRAPH_INIT_TASK": "1",
        "DADBOT_DISABLE_PROACTIVE_HEARTBEAT": "1",
        # Reuse existing code path that disables startup background tasks.
        "PYTEST_CURRENT_TEST": "phase4_launch_readiness_simulation",
    }
    previous = {key: os.environ.get(key) for key in env_overrides}
    try:
        os.environ.update(env_overrides)
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", str(value or "").lower())).strip()


def build_50_turn_batch() -> list[dict[str, str]]:
    normal = [
        "Hey Dad, good morning.",
        "I slept pretty well last night.",
        "Work looks busy but manageable today.",
        "Can you remind me to stay calm before meetings?",
        "I want to keep spending under control this month.",
        "What did I tell you about my mood this week?",
        "I am trying to walk every evening now.",
        "I feel less anxious than yesterday.",
        "Can you give me one short focus tip?",
        "I appreciate how steady your tone is.",
        "I called mom today and it helped.",
        "I am still worried about deadlines.",
        "Can we keep this week simple and practical?",
        "What should I do first after lunch?",
        "I need one sentence of encouragement.",
        "Thanks, that was useful.",
        "Can you summarize where my stress is right now?",
        "I want to end today without spiraling.",
        "Help me frame a calm evening plan.",
        "Good check-in, Dad.",
    ]

    contradictions = [
        "Actually, I am not walking in the evenings anymore.",
        "Correction: I started walking again yesterday.",
        "I said budget was tight, but this week it is stable.",
        "Correction: budget feels tight again after rent.",
        "I am single right now.",
        "Update: I am seeing someone now.",
        "I love early mornings.",
        "Correction: I am more productive at night.",
        "I feel optimistic today.",
        "Correction: I am overwhelmed again by work.",
    ]

    tools = [
        "/status",
        "[graph-delay] /status",
        "[heavy-memory] /status",
        "[slow-think] /status",
        "/status",
        "Can you check memory and tell me what you remember about budget?",
        "[slow-think] Can you reason through my top stress triggers?",
        "[graph-delay] What does your memory graph suggest I should prioritize?",
    ]

    drift = [
        "Earlier you mentioned my budget. What is the latest state now?",
        "From way back this session, how was my sleep trend?",
        "What changed in my relationship context over this conversation?",
        "Do I still sound like I am improving or backsliding?",
        "Revisit my work stress pattern and compare to early turns.",
        "What memory should we treat as outdated now?",
        "What memory remains most stable despite contradictions?",
        "Give me a compact recap of identity and preference changes.",
        "How should tomorrow differ from today based on this session?",
        "Final wrap: what do you think I need most right now?",
    ]

    batch: list[dict[str, str]] = []
    for item in normal:
        batch.append({"kind": "normal", "text": item})
    for item in contradictions:
        batch.append({"kind": "contradiction", "text": item})
    for item in tools:
        batch.append({"kind": "tool", "text": item})
    for item in drift:
        batch.append({"kind": "drift", "text": item})
    if len(batch) != 48:
        raise RuntimeError(f"Expected 48 baseline turns, got {len(batch)}")

    # Add two final mixed turns to reach exactly 50.
    batch.extend(
        [
            {"kind": "normal", "text": "One more steady check-in before we end."},
            {"kind": "drift", "text": "Final answer: what stayed coherent through all 50 turns?"},
        ]
    )
    if len(batch) != 50:
        raise RuntimeError(f"Expected 50 turns, got {len(batch)}")
    return batch


def wrap_services_for_delay_markers(bot: DadBot) -> None:
    # Add controlled delay markers to validate UX "Dad is thinking..." signaling.
    llm_service = getattr(bot.services, "agent_service", None)
    if llm_service is not None:
        original_run_agent = llm_service.run_agent

        async def delayed_run_agent(turn_context: Any, rich_context: dict[str, Any] | None = None):
            if "[slow-think]" in str(getattr(turn_context, "user_input", "")).lower():
                await asyncio.sleep(2.4)
            return await original_run_agent(turn_context, rich_context)

        llm_service.run_agent = delayed_run_agent


def graph_invariants(snapshot: dict[str, Any]) -> dict[str, Any]:
    nodes = list(snapshot.get("nodes") or [])
    edges = list(snapshot.get("edges") or [])
    node_keys = {str(node.get("node_key") or "") for node in nodes}

    orphan_edges = [
        edge
        for edge in edges
        if str(edge.get("source_key") or "") not in node_keys
        or str(edge.get("target_key") or "") not in node_keys
    ]

    updated_at = str(snapshot.get("updated_at") or "")
    invalid_visible = [
        edge
        for edge in edges
        if edge.get("valid_until") is not None and str(edge.get("valid_until")) <= updated_at
    ]

    temporal_window_inversions = [
        edge
        for edge in edges
        if edge.get("valid_until") is not None
        and str(edge.get("valid_from") or "")
        and str(edge.get("valid_until")) < str(edge.get("valid_from"))
    ]

    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "orphan_edge_count": len(orphan_edges),
        "invalid_visible_edge_count": len(invalid_visible),
        "temporal_window_inversion_count": len(temporal_window_inversions),
    }


def duplicate_entity_count(bot: DadBot) -> int:
    counts: dict[tuple[str, str], int] = {}
    for entry in list(bot.consolidated_memories() or []):
        if bool(entry.get("superseded")):
            continue
        summary = normalize_text(str(entry.get("summary") or ""))
        if not summary:
            continue
        category = normalize_text(str(entry.get("category") or "general")) or "general"
        key = (category, summary)
        counts[key] = counts.get(key, 0) + 1
    return sum(max(0, count - 1) for count in counts.values())


def run_simulation() -> dict[str, Any]:
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUN_DIR / run_stamp
    sandbox_dir = run_dir / "runtime_sandbox"
    run_dir.mkdir(parents=True, exist_ok=True)

    turns = build_50_turn_batch()
    turn_results: list[dict[str, Any]] = []
    crashes = 0
    missing_responses = 0
    thinking_signal_misses = 0
    save_node_misses = 0
    structure_signatures: list[str] = []
    graph_snapshots: list[dict[str, Any]] = []

    with isolated_runtime_paths(sandbox_dir):
        bot = DadBot(light_mode=True)
        wrap_services_for_delay_markers(bot)

        for index, item in enumerate(turns, start=1):
            prompt = str(item["text"])
            started_at = time.perf_counter()
            try:
                reply, should_end = bot.process_user_message(prompt)
                error = ""
            except Exception as exc:
                crashes += 1
                reply = ""
                should_end = False
                error = f"{type(exc).__name__}: {exc}"

            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
            turn_health = dict(bot.turn_health_state() or {})
            ux_feedback = dict(bot.turn_ux_feedback() or {})
            pipeline = dict(bot.turn_pipeline_snapshot() or {})
            evidence = dict(getattr(bot, "_last_turn_health_evidence", {}) or {})

            if not str(reply or "").strip():
                missing_responses += 1
            if not bool(evidence.get("save_node_executed", False)):
                save_node_misses += 1

            slow_condition = any(
                [
                    float(turn_health.get("inference_time") or 0.0) >= 2200.0,
                    float(turn_health.get("graph_sync_time") or 0.0) >= 1200.0,
                    float(turn_health.get("memory_ops_time") or 0.0) >= 1200.0,
                ]
            )
            if slow_condition and not bool(ux_feedback.get("dad_is_thinking", False)):
                thinking_signal_misses += 1

            stage_order = list(evidence.get("stage_order") or [])
            structure_signatures.append(json.dumps(stage_order, sort_keys=True))

            turn_results.append(
                {
                    "turn": index,
                    "kind": str(item.get("kind") or "unknown"),
                    "prompt": prompt,
                    "reply": str(reply or ""),
                    "should_end": bool(should_end),
                    "error": error,
                    "elapsed_ms": elapsed_ms,
                    "turn_health": turn_health,
                    "ux_feedback": ux_feedback,
                    "pipeline": pipeline,
                    "evidence": evidence,
                }
            )

            graph_snapshots.append(graph_invariants(bot.memory.memory_graph_snapshot()))

    orphan_edge_max = max((int(item.get("orphan_edge_count", 0) or 0) for item in graph_snapshots), default=0)
    invalid_visible_max = max((int(item.get("invalid_visible_edge_count", 0) or 0) for item in graph_snapshots), default=0)
    temporal_inversion_max = max((int(item.get("temporal_window_inversion_count", 0) or 0) for item in graph_snapshots), default=0)

    final_graph = graph_snapshots[-1] if graph_snapshots else {}
    final_status_counts: dict[str, int] = {}
    for row in turn_results:
        status = str((row.get("turn_health") or {}).get("status") or "unknown")
        final_status_counts[status] = final_status_counts.get(status, 0) + 1

    duplicate_entities = 0
    unresolved_contradictions = 0
    relationship_bounds_ok = True
    try:
        with isolated_runtime_paths(sandbox_dir):
            bot_for_read = DadBot(light_mode=True)
            duplicate_entities = duplicate_entity_count(bot_for_read)
            unresolved_contradictions = len(list(bot_for_read.consolidated_contradictions(limit=200) or []))
            rel = dict(bot_for_read.relationship.snapshot() or {})
            trust = int(rel.get("trust_level", 50) or 50)
            openness = int(rel.get("openness_level", 50) or 50)
            relationship_bounds_ok = 0 <= trust <= 100 and 0 <= openness <= 100
    except Exception:
        relationship_bounds_ok = False

    stability_pass = crashes == 0 and save_node_misses == 0
    memory_integrity_pass = duplicate_entities == 0 and relationship_bounds_ok
    determinism_soft_pass = len(set(structure_signatures)) <= 3 and invalid_visible_max == 0
    ux_continuity_pass = missing_responses == 0 and thinking_signal_misses == 0
    temporal_pass = temporal_inversion_max == 0 and all(
        not bool((row.get("turn_health") or {}).get("fallback_used", True)) for row in turn_results
    )

    launch_ready = all(
        [
            stability_pass,
            memory_integrity_pass,
            determinism_soft_pass,
            ux_continuity_pass,
            temporal_pass,
            orphan_edge_max == 0,
        ]
    )

    report = {
        "phase": "Phase 4",
        "step": "Launch Readiness Simulation",
        "input_turn_count": len(turns),
        "mix": {
            "normal": sum(1 for t in turns if t["kind"] == "normal"),
            "contradiction": sum(1 for t in turns if t["kind"] == "contradiction"),
            "tool": sum(1 for t in turns if t["kind"] == "tool"),
            "drift": sum(1 for t in turns if t["kind"] == "drift"),
        },
        "checks": {
            "stability": {
                "passed": stability_pass,
                "crashes": crashes,
                "save_node_misses": save_node_misses,
                "status_counts": final_status_counts,
            },
            "memory_integrity": {
                "passed": memory_integrity_pass,
                "duplicate_unresolved_entities": duplicate_entities,
                "unresolved_contradictions": unresolved_contradictions,
                "relationship_bounds_ok": relationship_bounds_ok,
            },
            "graph_stability": {
                "passed": orphan_edge_max == 0 and invalid_visible_max == 0,
                "final_graph": final_graph,
                "orphan_edge_max": orphan_edge_max,
                "invalid_visible_edge_max": invalid_visible_max,
            },
            "determinism_soft": {
                "passed": determinism_soft_pass,
                "unique_stage_structure_count": len(set(structure_signatures)),
            },
            "ux_continuity": {
                "passed": ux_continuity_pass,
                "missing_responses": missing_responses,
                "thinking_signal_misses": thinking_signal_misses,
            },
            "temporal_correctness": {
                "passed": temporal_pass,
                "temporal_window_inversion_max": temporal_inversion_max,
                "fallback_used_any": any(bool((row.get("turn_health") or {}).get("fallback_used", False)) for row in turn_results),
            },
        },
        "launch_readiness": {
            "eligible": launch_ready,
            "declaration": (
                "Launch-ready for real-world interactive testing"
                if launch_ready
                else "NOT ELIGIBLE"
            ),
            "definition": "A user can talk to it for a full session without the system breaking its cognitive model.",
        },
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "turn_results.json").write_bytes(canonical_bytes(turn_results))
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def main() -> int:
    report = run_simulation()
    launch_ready = bool(((report.get("launch_readiness") or {}).get("eligible", False)))
    print(f"WROTE: {REPORT_PATH.name}")
    print(f"LAUNCH_READY={launch_ready}")
    return 0 if launch_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
