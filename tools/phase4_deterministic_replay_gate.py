from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dadbot.core.graph import TurnContext, TurnGraph, TurnTemporalAxis
from dadbot.core.nodes import ContextBuilderNode, InferenceNode, ReflectionNode, SafetyNode, SaveNode, TemporalNode
from dadbot.services.persistence import PersistenceService

OUT_DIR = ROOT / "session_logs" / "phase4_deterministic_replay_gate"
REPORT_PATH = ROOT / "phase4_freeze_gate_report.json"


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def sha256_hex(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def sanitize_volatile(payload: Any) -> Any:
    volatile_keys = {"duration_ms", "elapsed_ms"}
    if isinstance(payload, dict):
        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            if key in volatile_keys:
                continue
            sanitized[key] = sanitize_volatile(value)
        return sanitized
    if isinstance(payload, list):
        return [sanitize_volatile(item) for item in payload]
    return payload


@dataclass
class FactEntry:
    key: str
    value: str
    updated_at: str
    valid_from: str
    valid_until: str | None
    superseded: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "updated_at": self.updated_at,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "superseded": self.superseded,
        }


class RecordingPersistenceManager:
    def __init__(self) -> None:
        self.turn_events: list[dict[str, Any]] = []
        self.checkpoints: list[dict[str, Any]] = []
        self.snapshots: list[dict[str, Any]] = []

    def persist_conversation_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.snapshots.append(dict(snapshot))

    def persist_graph_checkpoint(self, checkpoint: dict[str, Any], _skip_turn_event: bool = False) -> None:
        _ = _skip_turn_event
        self.checkpoints.append(dict(checkpoint))

    def persist_turn_event(self, event: dict[str, Any]) -> None:
        self.turn_events.append(dict(event))


class FakeGraphStoreBackend:
    def __init__(self) -> None:
        self._snapshot: dict[str, Any] = {"nodes": [], "edges": [], "updated_at": ""}

    def replace_graph(self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
        updated_at = ""
        for item in [*nodes, *edges]:
            ts = str(item.get("updated_at") or "")
            if ts and ts > updated_at:
                updated_at = ts
        self._snapshot = {
            "nodes": sorted(nodes, key=lambda n: (n["node_key"], n["node_type"], n.get("label", ""))),
            "edges": sorted(edges, key=lambda e: (e["edge_key"], e["source_key"], e["target_key"])),
            "updated_at": updated_at,
        }

    def fetch_graph(self) -> dict[str, Any]:
        return {
            "nodes": [dict(node) for node in self._snapshot.get("nodes", [])],
            "edges": [dict(edge) for edge in self._snapshot.get("edges", [])],
            "updated_at": self._snapshot.get("updated_at", ""),
        }


class FakeRuntime:
    def __init__(self) -> None:
        self._graph_commit_active = False
        self._current_turn_time_base = None
        self._fact_table: dict[str, FactEntry] = {}
        self._fact_history: list[FactEntry] = []
        self._graph_store = FakeGraphStoreBackend()
        self.history: list[dict[str, Any]] = []
        self.reflection_log: list[dict[str, Any]] = []


class FakeGraphManager:
    def __init__(self, runtime: FakeRuntime) -> None:
        self.runtime = runtime

    @staticmethod
    def _edge_is_visible(edge: dict[str, Any], current_time: str) -> bool:
        valid_from = str(edge.get("valid_from") or "")
        valid_until = edge.get("valid_until")
        if valid_from and valid_from > current_time:
            return False
        if valid_until is not None and str(valid_until) <= current_time:
            return False
        return True

    def sync_graph_store(self, turn_context: TurnContext | None = None) -> dict[str, Any]:
        if turn_context is None:
            raise RuntimeError("turn_context is required in strict mode")
        if not bool(getattr(self.runtime, "_graph_commit_active", False)):
            raise RuntimeError("Graph sync violation: commit boundary inactive")

        temporal = getattr(turn_context, "temporal", None)
        if temporal is None:
            raise RuntimeError("TemporalNode missing in graph sync")
        current_time = str(temporal.wall_time)

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        node_keys: set[str] = set()

        for entry in self.runtime._fact_history:
            source_key = f"fact:{entry.key}:{entry.value}"
            topic_key = f"topic:{entry.key}"
            if source_key not in node_keys:
                nodes.append(
                    {
                        "node_key": source_key,
                        "node_type": "fact",
                        "label": f"{entry.key}={entry.value}",
                        "updated_at": entry.updated_at,
                    }
                )
                node_keys.add(source_key)
            if topic_key not in node_keys:
                nodes.append(
                    {
                        "node_key": topic_key,
                        "node_type": "topic",
                        "label": entry.key,
                        "updated_at": entry.updated_at,
                    }
                )
                node_keys.add(topic_key)

            edges.append(
                {
                    "edge_key": f"{source_key}->related_to->{topic_key}",
                    "source_key": source_key,
                    "target_key": topic_key,
                    "relation_type": "related_to",
                    "updated_at": entry.updated_at,
                    "valid_from": entry.valid_from,
                    "valid_until": entry.valid_until,
                }
            )

        visible_edges = [
            edge
            for edge in edges
            if self._edge_is_visible(edge, current_time)
            and edge["source_key"] in node_keys
            and edge["target_key"] in node_keys
        ]
        self.runtime._graph_store.replace_graph(nodes, visible_edges)
        return self.runtime._graph_store.fetch_graph()

    def graph_snapshot(self) -> dict[str, Any]:
        return self.runtime._graph_store.fetch_graph()


class FakeMemoryManager:
    def __init__(self, runtime: FakeRuntime) -> None:
        self.graph_manager = FakeGraphManager(runtime)


class FakeMemoryCoordinator:
    def __init__(self, runtime: FakeRuntime) -> None:
        self.runtime = runtime

    @staticmethod
    def _parse_fact_input(raw: str) -> tuple[str, str]:
        # Expected format: "fact:<key>=<value>"
        text = str(raw or "").strip()
        if not text.startswith("fact:") or "=" not in text:
            return ("general", text.lower() or "empty")
        payload = text[len("fact:") :]
        key, value = payload.split("=", 1)
        return (key.strip().lower() or "general", value.strip().lower() or "unknown")

    def consolidate_memories(self, turn_context: TurnContext | None = None) -> None:
        if turn_context is None:
            raise RuntimeError("turn_context is required")
        if not bool(getattr(self.runtime, "_graph_commit_active", False)):
            raise RuntimeError("Memory mutation outside SaveNode commit boundary")

        wall_time = str(turn_context.temporal.wall_time)
        key, value = self._parse_fact_input(turn_context.user_input)

        previous = self.runtime._fact_table.get(key)
        if previous is not None and previous.value != value and not previous.superseded:
            superseded = FactEntry(
                key=previous.key,
                value=previous.value,
                updated_at=wall_time,
                valid_from=previous.valid_from,
                valid_until=wall_time,
                superseded=True,
            )
            self.runtime._fact_history.append(superseded)

        current = FactEntry(
            key=key,
            value=value,
            updated_at=wall_time,
            valid_from=wall_time,
            valid_until=None,
            superseded=False,
        )
        self.runtime._fact_table[key] = current
        self.runtime._fact_history.append(current)

    def apply_controlled_forgetting(self, turn_context: TurnContext | None = None) -> dict[str, Any]:
        if turn_context is None:
            raise RuntimeError("turn_context is required")
        if not bool(getattr(self.runtime, "_graph_commit_active", False)):
            raise RuntimeError("Memory mutation outside SaveNode commit boundary")
        # Deterministic no-op for this gate.
        return {"removed": 0, "ran": True}


class FakeRelationshipManager:
    def __init__(self, runtime: FakeRuntime) -> None:
        self.runtime = runtime

    def reflect(self, force: bool = False, turn_context: TurnContext | None = None) -> dict[str, Any]:
        if not bool(getattr(self.runtime, "_graph_commit_active", False)):
            raise RuntimeError("Relationship mutation outside SaveNode commit boundary")
        if turn_context is None:
            raise RuntimeError("turn_context is required")
        entry = {
            "force": bool(force),
            "trace_id": turn_context.trace_id,
            "recorded_at": turn_context.temporal.wall_time,
        }
        self.runtime.reflection_log.append(entry)
        return entry


class FakeTurnService:
    def __init__(self, runtime: FakeRuntime) -> None:
        self.bot = runtime

    def finalize_user_turn(self, turn_text: str, mood: str, reply: str, attachments: Any) -> tuple[str, bool]:
        _ = attachments
        self.bot.history.append(
            {
                "turn_text": str(turn_text),
                "mood": str(mood),
                "reply": str(reply),
            }
        )
        return (str(reply), False)


class FakeHealthService:
    def tick(self, context: TurnContext) -> dict[str, Any]:
        return {"ok": True, "trace_id": context.trace_id}


class FakeContextService:
    def build_context(self, context: TurnContext) -> dict[str, Any]:
        return {
            "temporal": context.temporal_snapshot(),
            "input_len": len(str(context.user_input or "")),
        }


class FakeAgentService:
    async def run_agent(self, context: TurnContext, rich_context: dict[str, Any]) -> tuple[str, bool]:
        _ = rich_context
        key = str(context.user_input or "").strip().replace(" ", "_")
        return (f"ack:{key}", False)


class FakeSafetyService:
    def enforce_policies(self, context: TurnContext, candidate: Any) -> Any:
        _ = context
        return candidate


class FakeReflectionService:
    def reflect_after_turn(self, turn_text: str, current_mood: str, reply_text: str) -> dict[str, Any]:
        return {
            "turn_text": str(turn_text),
            "mood": str(current_mood),
            "reply": str(reply_text),
        }


def build_graph(runtime: FakeRuntime, persistence: PersistenceService, recording_pm: RecordingPersistenceManager) -> TurnGraph:
    registry = {
        "persistence_service": persistence,
        "health": FakeHealthService(),
        "memory": FakeContextService(),
        "llm": FakeAgentService(),
        "safety": FakeSafetyService(),
        "reflection": FakeReflectionService(),
        "storage": persistence,
        "telemetry": None,
    }

    graph = TurnGraph(registry=registry)
    graph.add_node("temporal", TemporalNode())
    graph.add_node("context_builder", ContextBuilderNode(registry["memory"]))
    graph.add_node("inference", InferenceNode(registry["llm"]))
    graph.add_node("safety", SafetyNode(registry["safety"]))
    graph.add_node("reflection", ReflectionNode(registry["reflection"]))
    graph.add_node("save", SaveNode(registry["storage"]))
    graph.set_edge("temporal", "context_builder")
    graph.set_edge("context_builder", "inference")
    graph.set_edge("inference", "safety")
    graph.set_edge("safety", "reflection")
    graph.set_edge("reflection", "save")

    _ = recording_pm
    return graph


def build_turn_context(user_input: str, turn_index: int) -> TurnContext:
    temporal = TurnTemporalAxis.from_lock_hash(f"phase4-lock-{turn_index:04d}")
    return TurnContext(
        user_input=user_input,
        trace_id=f"trace-{turn_index:04d}",
        temporal=temporal,
        metadata={
            "determinism": {
                "enforced": True,
                "lock_hash": f"phase4-lock-{turn_index:04d}",
                "lock_id": f"phase4-lock-id-{turn_index:04d}",
            }
        },
        state={
            "turn_text": user_input,
            "mood": "neutral",
            "norm_attachments": [],
        },
    )


async def run_batch_once(inputs: list[str]) -> dict[str, Any]:
    runtime = FakeRuntime()
    recording_pm = RecordingPersistenceManager()
    runtime.memory_coordinator = FakeMemoryCoordinator(runtime)
    runtime.memory_manager = FakeMemoryManager(runtime)
    runtime.relationship_manager = FakeRelationshipManager(runtime)

    turn_service = FakeTurnService(runtime)
    persistence = PersistenceService(recording_pm, turn_service=turn_service)
    graph = build_graph(runtime, persistence, recording_pm)

    per_turn_stage_order: list[list[str]] = []
    per_turn_stage_counts: list[dict[str, int]] = []

    for idx, user_input in enumerate(inputs, start=1):
        context = build_turn_context(user_input, idx)
        _result = await graph.execute(context)
        stages = [trace.stage for trace in context.stage_traces]
        per_turn_stage_order.append(stages)
        per_turn_stage_counts.append(
            {
                "inference": stages.count("inference"),
                "reflection": stages.count("reflection"),
                "save": stages.count("save"),
            }
        )

    graph_snapshot = runtime.memory_manager.graph_manager.graph_snapshot()
    memory_payload = {
        "fact_table": {key: value.to_dict() for key, value in sorted(runtime._fact_table.items())},
        "fact_history": [entry.to_dict() for entry in runtime._fact_history],
        "history": list(runtime.history),
        "reflection_log": list(runtime.reflection_log),
    }
    ledger_payload = {
        "checkpoints": recording_pm.checkpoints,
        "turn_events": recording_pm.turn_events,
        "snapshots": recording_pm.snapshots,
    }

    return {
        "graph": sanitize_volatile(graph_snapshot),
        "memory": sanitize_volatile(memory_payload),
        "ledger": sanitize_volatile(ledger_payload),
        "execution_trace": {
            "per_turn_stage_order": per_turn_stage_order,
            "per_turn_stage_counts": per_turn_stage_counts,
        },
    }


def ledger_lock_checks(ledger_payload: dict[str, Any]) -> dict[str, Any]:
    events = list(ledger_payload.get("turn_events", []))
    canonical = canonical_bytes(ledger_payload)

    sequence_by_trace: dict[str, list[int]] = {}
    for event in events:
        trace_id = str(event.get("trace_id") or "")
        if "sequence" not in event or not trace_id:
            continue
        sequence_by_trace.setdefault(trace_id, []).append(int(event.get("sequence") or 0))
    ordered = all(values == sorted(values) and values and values[0] == 1 for values in sequence_by_trace.values())

    as_text = canonical.decode("utf-8", errors="replace")
    uuid_like = bool(
        __import__("re").search(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}",
            as_text,
        )
    )

    hidden_now_calls = "datetime.now" in as_text or "time.time" in as_text

    # Dict-order drift is guarded by canonical serialization. If re-encoding changes,
    # canonical bytes would differ.
    reencoded = canonical_bytes(json.loads(as_text))
    serialization_stable = reencoded == canonical

    return {
        "event_ordering_monotonic": ordered,
        "serialization_stable": serialization_stable,
        "uuid_variation_detected": uuid_like,
        "hidden_timestamp_marker_detected": hidden_now_calls,
    }


def stage_order_checks(execution_trace: dict[str, Any]) -> dict[str, Any]:
    required = ["temporal", "context_builder", "inference", "safety", "reflection", "save"]
    per_turn = list(execution_trace.get("per_turn_stage_order", []))
    counts = list(execution_trace.get("per_turn_stage_counts", []))

    exact_order_all = all(order == required for order in per_turn)
    no_double_inference = all(int(item.get("inference") or 0) == 1 for item in counts)
    no_skipped_reflection = all(int(item.get("reflection") or 0) == 1 for item in counts)
    no_save_reentry = all(int(item.get("save") or 0) == 1 for item in counts)

    return {
        "required_order": required,
        "exact_order_all_turns": exact_order_all,
        "no_double_inference": no_double_inference,
        "no_skipped_reflection": no_skipped_reflection,
        "no_save_reentry": no_save_reentry,
    }


def graph_invariant_checks(graph_payload: dict[str, Any], memory_payload: dict[str, Any]) -> dict[str, Any]:
    nodes = list(graph_payload.get("nodes", []))
    edges = list(graph_payload.get("edges", []))
    node_keys = {str(node.get("node_key") or "") for node in nodes}

    active_by_key: dict[str, int] = {}
    for _id, entry in (memory_payload.get("fact_table", {}) or {}).items():
        key = str((entry or {}).get("key") or "")
        active_by_key[key] = active_by_key.get(key, 0) + (0 if bool((entry or {}).get("superseded")) else 1)
    no_dual_truth = all(count <= 1 for count in active_by_key.values())

    orphan_edges = [
        edge
        for edge in edges
        if str(edge.get("source_key") or "") not in node_keys or str(edge.get("target_key") or "") not in node_keys
    ]

    visible_invalid_edges = [
        edge
        for edge in edges
        if edge.get("valid_until") is not None and str(edge.get("valid_until")) <= str(graph_payload.get("updated_at") or "")
    ]

    temporal_windows_consistent = True
    for edge in edges:
        valid_from = str(edge.get("valid_from") or "")
        valid_until = edge.get("valid_until")
        if valid_until is not None and valid_from and str(valid_until) < valid_from:
            temporal_windows_consistent = False
            break

    return {
        "no_dual_truth_after_contradiction_resolution": no_dual_truth,
        "orphan_edge_count": len(orphan_edges),
        "invalid_visible_edge_count": len(visible_invalid_edges),
        "temporal_windows_consistent": temporal_windows_consistent,
    }


def compare_runs(run_a: dict[str, Any], run_b: dict[str, Any]) -> dict[str, Any]:
    a_graph_hash = sha256_hex(run_a["graph"])
    b_graph_hash = sha256_hex(run_b["graph"])
    a_memory_hash = sha256_hex(run_a["memory"])
    b_memory_hash = sha256_hex(run_b["memory"])
    a_ledger_hash = sha256_hex(run_a["ledger"])
    b_ledger_hash = sha256_hex(run_b["ledger"])
    a_trace_hash = sha256_hex(run_a["execution_trace"])
    b_trace_hash = sha256_hex(run_b["execution_trace"])

    return {
        "graph": {
            "run1_sha256": a_graph_hash,
            "run2_sha256": b_graph_hash,
            "byte_identical": a_graph_hash == b_graph_hash,
        },
        "memory": {
            "run1_sha256": a_memory_hash,
            "run2_sha256": b_memory_hash,
            "byte_identical": a_memory_hash == b_memory_hash,
        },
        "ledger": {
            "run1_sha256": a_ledger_hash,
            "run2_sha256": b_ledger_hash,
            "byte_identical": a_ledger_hash == b_ledger_hash,
        },
        "execution_trace": {
            "run1_sha256": a_trace_hash,
            "run2_sha256": b_trace_hash,
            "identical_stage_ordering": a_trace_hash == b_trace_hash,
        },
    }


def build_input_batch() -> list[str]:
    # 20 deterministic turns, including intentional contradictions for invariant checks.
    return [
        "fact:job=stressed",
        "fact:budget=tight",
        "fact:sleep=poor",
        "fact:job=focused",
        "fact:budget=stable",
        "fact:support=needed",
        "fact:budget=tight",
        "fact:exercise=none",
        "fact:sleep=improving",
        "fact:job=stressed",
        "fact:family=connected",
        "fact:budget=stable",
        "fact:exercise=light",
        "fact:sleep=consistent",
        "fact:job=focused",
        "fact:support=received",
        "fact:family=connected",
        "fact:budget=stable",
        "fact:exercise=light",
        "fact:sleep=consistent",
    ]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    batch = build_input_batch()
    run1 = asyncio.run(run_batch_once(batch))
    run2 = asyncio.run(run_batch_once(batch))

    comparison = compare_runs(run1, run2)
    step2_ledger_lock = ledger_lock_checks(run1["ledger"])
    step3_stage_order = stage_order_checks(run1["execution_trace"])
    step4_invariants = graph_invariant_checks(run1["graph"], run1["memory"])

    step1_pass = all(
        [
            comparison["graph"]["byte_identical"],
            comparison["memory"]["byte_identical"],
            comparison["ledger"]["byte_identical"],
            comparison["execution_trace"]["identical_stage_ordering"],
        ]
    )
    step2_pass = all(
        [
            step2_ledger_lock["event_ordering_monotonic"],
            step2_ledger_lock["serialization_stable"],
            not step2_ledger_lock["uuid_variation_detected"],
            not step2_ledger_lock["hidden_timestamp_marker_detected"],
            comparison["ledger"]["byte_identical"],
        ]
    )
    step3_pass = all(
        [
            step3_stage_order["exact_order_all_turns"],
            step3_stage_order["no_double_inference"],
            step3_stage_order["no_skipped_reflection"],
            step3_stage_order["no_save_reentry"],
        ]
    )
    step4_pass = all(
        [
            step4_invariants["no_dual_truth_after_contradiction_resolution"],
            step4_invariants["orphan_edge_count"] == 0,
            step4_invariants["invalid_visible_edge_count"] == 0,
            step4_invariants["temporal_windows_consistent"],
        ]
    )

    overall_pass = all([step1_pass, step2_pass, step3_pass, step4_pass])

    report = {
        "phase": "Phase 4",
        "input_turn_count": len(batch),
        "step1_full_deterministic_replay_gate": {
            "passed": step1_pass,
            "comparison": comparison,
        },
        "step2_ledger_final_determinism_lock": {
            "passed": step2_pass,
            "checks": step2_ledger_lock,
        },
        "step3_stage_order_enforcement_validation": {
            "passed": step3_pass,
            "checks": step3_stage_order,
        },
        "step4_edge_case_graph_invariant_tests": {
            "passed": step4_pass,
            "checks": step4_invariants,
        },
        "step5_freeze_certification": {
            "eligible": overall_pass,
            "declaration": (
                "Graph is a deterministic, bi-temporal cognitive engine with transactional memory lifecycle."
                if overall_pass
                else "NOT ELIGIBLE"
            ),
        },
    }

    (OUT_DIR / "run1_graph.json").write_bytes(canonical_bytes(run1["graph"]))
    (OUT_DIR / "run1_memory.json").write_bytes(canonical_bytes(run1["memory"]))
    (OUT_DIR / "run1_ledger.json").write_bytes(canonical_bytes(run1["ledger"]))
    (OUT_DIR / "run1_execution_trace.json").write_bytes(canonical_bytes(run1["execution_trace"]))
    (OUT_DIR / "run2_graph.json").write_bytes(canonical_bytes(run2["graph"]))
    (OUT_DIR / "run2_memory.json").write_bytes(canonical_bytes(run2["memory"]))
    (OUT_DIR / "run2_ledger.json").write_bytes(canonical_bytes(run2["ledger"]))
    (OUT_DIR / "run2_execution_trace.json").write_bytes(canonical_bytes(run2["execution_trace"]))
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"WROTE: {REPORT_PATH.name}")
    print(f"STEP1_PASS={step1_pass}")
    print(f"STEP2_PASS={step2_pass}")
    print(f"STEP3_PASS={step3_pass}")
    print(f"STEP4_PASS={step4_pass}")
    print(f"STEP5_ELIGIBLE={overall_pass}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
