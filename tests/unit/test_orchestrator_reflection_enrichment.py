from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from dadbot.core.graph_context import TurnContext
from dadbot.core.orchestrator import DadBotOrchestrator
from dadbot.core.reflection_ir import DriftReflectionEngine


pytestmark = pytest.mark.unit


def _build_sample_ledger() -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as handle:
        base_time = datetime(2026, 1, 7, 22, 30, 0)

        entries = [
            {
                "turn_index": 0,
                "recorded_at": base_time.isoformat(),
                "session_goals": ["finish tests"],
                "relational_state": {"is_aligned": True},
                "goal_alignment_diversion_streak": 0,
                "user_input_excerpt": "finish persistence unit tests",
                "decision": "followed_intent",
            },
            {
                "turn_index": 1,
                "recorded_at": base_time.replace(minute=31).isoformat(),
                "session_goals": ["finish tests"],
                "relational_state": {"is_aligned": False},
                "goal_alignment_diversion_streak": 1,
                "user_input_excerpt": "search fishing rods online",
                "decision": "diverted_from_intent",
                "goal_alignment_mandatory_halt": True,
            },
            {
                "turn_index": 2,
                "recorded_at": base_time.replace(minute=32).isoformat(),
                "session_goals": ["finish tests"],
                "relational_state": {"is_aligned": True},
                "goal_alignment_diversion_streak": 0,
                "user_input_excerpt": "back to persistence assertions",
                "decision": "followed_intent",
            },
        ]
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
        return handle.name


def test_reflection_summary_includes_evidence_graph_payload() -> None:
    ledger_path = _build_sample_ledger()
    try:
        orchestrator = DadBotOrchestrator.__new__(DadBotOrchestrator)
        orchestrator._reflection_engine = DriftReflectionEngine()
        orchestrator._resolve_ledger_path = lambda _session_id: ledger_path  # type: ignore[method-assign]

        context = TurnContext(user_input="continue test work")
        orchestrator._analyze_behavioral_reflection("session-1", context)

        reflection = dict(context.state.get("reflection_summary") or {})
        assert reflection
        evidence_graph = dict(reflection.get("evidence_graph") or {})
        assert isinstance(evidence_graph.get("edge_count"), int)
        assert isinstance(evidence_graph.get("node_count"), int)
        assert isinstance(evidence_graph.get("edges"), list)
        assert evidence_graph.get("edge_count", 0) >= 1
    finally:
        Path(ledger_path).unlink(missing_ok=True)
