from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dadbot.core.graph import TurnContext, TurnGraph
from dadbot.core.nodes import (
    ContextBuilderNode,
    HealthNode,
    InferenceNode,
    ReflectionNode,
    SafetyNode,
    SaveNode,
    TemporalNode,
)
from dadbot.services.persistence import PersistenceService

TURN_TRACE_PATH = ROOT / "turn_execution_trace.log"
TEMPORAL_AUDIT_PATH = ROOT / "temporal_audit_report.json"
SAVE_ENFORCEMENT_PATH = ROOT / "save_node_enforcement_report.json"
ENTROPY_PATH = ROOT / "graph_legacy_entropy_report.json"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def find_def_line(text: str, fn_name: str) -> int | None:
    pattern = re.compile(rf"^\s*(?:async\s+)?def\s+{re.escape(fn_name)}\s*\(", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        return None
    return text.count("\n", 0, m.start()) + 1


def build_static_chain() -> dict[str, Any]:
    dadbot_file = ROOT / "dadbot" / "core" / "dadbot.py"
    orchestrator_file = ROOT / "dadbot" / "core" / "orchestrator.py"
    graph_file = ROOT / "dadbot" / "core" / "graph.py"

    dadbot_src = read_text(dadbot_file)
    orch_src = read_text(orchestrator_file)
    graph_src = read_text(graph_file)

    return {
        "entry_file": "dadbot/core/dadbot.py",
        "entry_symbol": "DadBot.process_user_message",
        "static_chain": [
            {
                "symbol": "DadBot.process_user_message",
                "file": "dadbot/core/dadbot.py",
                "line": find_def_line(dadbot_src, "process_user_message"),
                "next": "DadBot._run_graph_turn_sync",
            },
            {
                "symbol": "DadBot._run_graph_turn_sync",
                "file": "dadbot/core/dadbot.py",
                "line": find_def_line(dadbot_src, "_run_graph_turn_sync"),
                "next": "DadBotOrchestrator.handle_turn",
            },
            {
                "symbol": "DadBotOrchestrator.handle_turn",
                "file": "dadbot/core/orchestrator.py",
                "line": find_def_line(orch_src, "handle_turn"),
                "next": "TurnGraph.execute",
            },
            {
                "symbol": "TurnGraph.execute",
                "file": "dadbot/core/graph.py",
                "line": find_def_line(graph_src, "execute"),
                "next": "node chain preflight->inference->safety->reflection->save",
            },
        ],
    }


class DictRegistry(dict):
    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key, default)


class FakeHealthService:
    def tick(self, _context: Any) -> dict[str, Any]:
        return {"ok": True}


class FakeMemoryService:
    async def query(self, _user_input: str) -> list[dict[str, Any]]:
        return [{"summary": "demo"}]

    def build_context(self, _context: Any) -> dict[str, Any]:
        return {"source": "fake"}


class FakeLLMService:
    async def run_agent(self, _context: Any, _rich_context: dict[str, Any]) -> tuple[str, bool]:
        return ("ok", False)


class FakeSafetyService:
    def enforce_policies(self, _context: Any, candidate: Any) -> Any:
        return candidate


class FakeReflectionService:
    def reflect_after_turn(self, _turn_text: str, _mood: str, _reply: str) -> dict[str, Any]:
        return {"reflected": True}


class FakePersistenceManager:
    def persist_conversation_snapshot(self, _snapshot: dict[str, Any]) -> None:
        return None

    def persist_graph_checkpoint(self, _checkpoint: dict[str, Any], _skip_turn_event: bool = False) -> None:
        return None

    def persist_turn_event(self, _event: dict[str, Any]) -> None:
        return None


class FakeGraphManager:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def sync_graph_store(self, turn_context: Any = None) -> None:
        _ = turn_context
        self._calls.append("sync_graph_store")


class FakeMemoryManager:
    def __init__(self, calls: list[str]) -> None:
        self.graph_manager = FakeGraphManager(calls)


class FakeMemoryCoordinator:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def consolidate_memories(self, turn_context: Any = None) -> None:
        _ = turn_context
        self._calls.append("consolidate_memories")

    def apply_controlled_forgetting(self, turn_context: Any = None) -> None:
        _ = turn_context
        self._calls.append("apply_controlled_forgetting")


class FakeRelationshipManager:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def reflect(self, force: bool = False) -> None:
        _ = force
        self._calls.append("relationship.reflect")


class FakeRuntime:
    def __init__(self, calls: list[str]) -> None:
        self._graph_commit_active = False
        self._current_turn_time_base = None
        self.memory_coordinator = FakeMemoryCoordinator(calls)
        self.memory_manager = FakeMemoryManager(calls)
        self.relationship_manager = FakeRelationshipManager(calls)


class FakeTurnService:
    def __init__(self, runtime: FakeRuntime) -> None:
        self.bot = runtime

    def finalize_user_turn(self, _turn_text: str, _mood: str, reply: str, _attachments: Any) -> tuple[str, bool]:
        return (str(reply or ""), False)


async def run_controlled_turn() -> dict[str, Any]:
    call_order: list[str] = []

    runtime = FakeRuntime(call_order)
    turn_service = FakeTurnService(runtime)
    persistence_service = PersistenceService(FakePersistenceManager(), turn_service=turn_service)

    registry = DictRegistry(
        {
            "persistence_service": persistence_service,
            "health": FakeHealthService(),
            "memory": FakeMemoryService(),
            "llm": FakeLLMService(),
            "safety": FakeSafetyService(),
            "reflection": FakeReflectionService(),
            "storage": persistence_service,
            "telemetry": None,
        }
    )

    graph = TurnGraph(registry=registry)
    graph.add_node(
        "preflight", (TemporalNode(), HealthNode(registry["health"]), ContextBuilderNode(registry["memory"]))
    )
    graph.add_node("inference", InferenceNode(registry["llm"]))
    graph.add_node("safety", SafetyNode(registry["safety"]))
    graph.add_node("reflection", ReflectionNode(registry["reflection"]))
    graph.add_node("save", SaveNode(registry["storage"]))
    graph.set_edge("preflight", "inference")
    graph.set_edge("inference", "safety")
    graph.set_edge("safety", "reflection")
    graph.set_edge("reflection", "save")

    context = TurnContext(user_input="hello")
    result = await graph.execute(context)

    stage_sequence = [trace.stage for trace in context.stage_traces]
    save_count = sum(1 for stage in stage_sequence if stage == "save")

    return {
        "runtime_stage_sequence": stage_sequence,
        "save_node_count": save_count,
        "graph_result": result,
        "call_order": call_order,
    }


def temporal_audit() -> dict[str, Any]:
    targets = [
        ROOT / "dadbot" / "memory" / "graph_manager.py",
        ROOT / "dadbot" / "managers" / "memory_coordination.py",
        ROOT / "dadbot" / "relationship.py",
        ROOT / "dadbot" / "services" / "context_service.py",
        ROOT / "dadbot" / "core" / "nodes.py",
    ]

    checks = {
        "datetime.now": re.compile(r"\bdatetime\.now\s*\("),
        "date.today": re.compile(r"\bdate\.today\s*\("),
        "_projection_fallback": re.compile(r"_projection_fallback"),
        "any_.now()": re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.now\s*\("),
    }

    file_reports: list[dict[str, Any]] = []
    total_hits = 0

    for path in targets:
        text = read_text(path)
        lines = text.splitlines()
        hits: list[dict[str, Any]] = []
        for i, line in enumerate(lines, start=1):
            for label, rx in checks.items():
                if rx.search(line):
                    hits.append({"line": i, "check": label, "text": line.strip()})
        total_hits += len(hits)
        file_reports.append(
            {
                "file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "hit_count": len(hits),
                "hits": hits,
            }
        )

    return {
        "scope": "active runtime path targeted files only",
        "checks": list(checks.keys()),
        "status": "clean" if total_hits == 0 else "violations_found",
        "total_hits": total_hits,
        "files": file_reports,
    }


def entropy_check() -> dict[str, Any]:
    targets = [
        ROOT / "dadbot" / "core" / "dadbot.py",
        ROOT / "dadbot" / "core" / "orchestrator.py",
        ROOT / "dadbot" / "core" / "graph.py",
        ROOT / "dadbot" / "core" / "nodes.py",
        ROOT / "dadbot" / "services" / "agent_service.py",
        ROOT / "dadbot" / "services" / "persistence.py",
        ROOT / "dadbot" / "registry.py",
    ]

    patterns = {
        "turn_service.process_user_message": re.compile(r"turn_service\.process_user_message\s*\("),
        "turn_service.process_user_message_async": re.compile(r"turn_service\.process_user_message_async\s*\("),
        "turn_service.process_user_message_stream": re.compile(r"turn_service\.process_user_message_stream\s*\("),
        "_record_graph_fallback": re.compile(r"_record_graph_fallback"),
        "graph_turns_enabled": re.compile(r"graph_turns_enabled"),
        "getattr(...turn_service": re.compile(r"getattr\([^\n]*turn_service"),
        "legacy fallback": re.compile(r"legacy fallback", re.IGNORECASE),
    }

    matches: list[dict[str, Any]] = []
    for path in targets:
        text = read_text(path)
        for line_no, line in enumerate(text.splitlines(), start=1):
            for label, rx in patterns.items():
                if rx.search(line):
                    matches.append(
                        {
                            "file": str(path.relative_to(ROOT)).replace("\\", "/"),
                            "line": line_no,
                            "pattern": label,
                            "text": line.strip(),
                        }
                    )

    return {
        "scope": [str(p.relative_to(ROOT)).replace("\\", "/") for p in targets],
        "patterns": list(patterns.keys()),
        "entropy_detected": len(matches) > 0,
        "match_count": len(matches),
        "matches": matches,
    }


def main() -> int:
    static_chain = build_static_chain()
    runtime_result = asyncio.run(run_controlled_turn())
    temporal_report = temporal_audit()
    entropy_report = entropy_check()

    required_order = [
        "consolidate_memories",
        "apply_controlled_forgetting",
        "sync_graph_store",
        "relationship.reflect",
    ]
    observed_order = runtime_result["call_order"]

    order_valid = observed_order == required_order
    deviations: list[str] = []
    if not order_valid:
        deviations.append(f"observed={observed_order}")

    save_report = {
        "save_node_executed_exactly_once": runtime_result["save_node_count"] == 1,
        "observed_call_order": observed_order,
        "required_call_order": required_order,
        "order_valid": order_valid,
        "deviations": deviations,
    }

    fallback_refs = [
        {
            "file": m["file"],
            "line": m["line"],
            "pattern": m["pattern"],
            "text": m["text"],
        }
        for m in entropy_report["matches"]
    ]

    trace_payload = {
        "entry_file": static_chain["entry_file"],
        "entry_symbol": static_chain["entry_symbol"],
        "static_chain": static_chain["static_chain"],
        "runtime_stage_sequence": runtime_result["runtime_stage_sequence"],
        "save_node_count": runtime_result["save_node_count"],
        "save_node_executed_exactly_once": runtime_result["save_node_count"] == 1,
        "graph_result": runtime_result["graph_result"],
        "fallback_references_in_active_path": fallback_refs,
    }

    TURN_TRACE_PATH.write_text(json.dumps(trace_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    TEMPORAL_AUDIT_PATH.write_text(json.dumps(temporal_report, indent=2, ensure_ascii=True), encoding="utf-8")
    SAVE_ENFORCEMENT_PATH.write_text(json.dumps(save_report, indent=2, ensure_ascii=True), encoding="utf-8")
    ENTROPY_PATH.write_text(json.dumps(entropy_report, indent=2, ensure_ascii=True), encoding="utf-8")

    print("WROTE:", TURN_TRACE_PATH.name)
    print("WROTE:", TEMPORAL_AUDIT_PATH.name, "status=", temporal_report["status"])
    print("WROTE:", SAVE_ENFORCEMENT_PATH.name, "order_valid=", save_report["order_valid"])
    print(
        "WROTE:",
        ENTROPY_PATH.name,
        "entropy_detected=",
        entropy_report["entropy_detected"],
        "matches=",
        entropy_report["match_count"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
