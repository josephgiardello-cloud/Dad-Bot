from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dad import DadBot
from dadbot.core.coherence_metrics import OutputCoherenceTracker


def _bind_bot_snapshot_scope(bot: DadBot, temp_path: Path) -> None:
    """Controlled mutator for temporary runtime storage bindings.

    Keep runtime-path rebinding inside a single boundary so observability tools
    do not leak ad-hoc mutable state assignments.
    """
    bindings = {
        "MEMORY_PATH": temp_path / "dad_memory.json",
        "SEMANTIC_MEMORY_DB_PATH": temp_path / "dad_memory_semantic.sqlite3",
        "GRAPH_STORE_DB_PATH": temp_path / "dad_memory_graph.sqlite3",
        "SESSION_LOG_DIR": temp_path / "session_logs",
    }
    for key, value in bindings.items():
        setattr(bot, key, value)


def _make_bot(temp_path: Path) -> DadBot:
    bot = DadBot(light_mode=True)
    _bind_bot_snapshot_scope(bot, temp_path)
    setattr(bot, "MEMORY_STORE", bot.default_memory_store())
    bot.save_memory_store()
    setattr(bot.memory, "export_memory_store", lambda _path: None)
    return bot


def _turn_context(day_index: int) -> SimpleNamespace:
    base = datetime(2026, 5, 1, 12, 0, 0)
    dt = base + timedelta(days=day_index)
    return SimpleNamespace(
        temporal=SimpleNamespace(
            wall_time=dt.isoformat(timespec="seconds"),
            wall_date=dt.date().isoformat(),
        )
    )


def _scenario_memory(turn: int, *, conflict_heavy: bool) -> tuple[str, str, float, int, int, str, list[str]]:
    turn_date = (datetime(2026, 5, 1) + timedelta(days=turn)).date()

    if conflict_heavy:
        if turn % 15 == 0:
            summary = "Tony says he never uses checklists anymore."
            category = "general"
            importance = 0.20
            access = 0
            high_hits = 0
            entry_date = (turn_date - timedelta(days=420)).isoformat()
            contradictions = ["Tony prefers direct checklist style."]
            return summary, category, importance, access, high_hits, entry_date, contradictions
        if turn % 10 == 0:
            summary = "Tony prefers direct checklist style for planning."
            category = "preferences"
            importance = 0.78
            access = 5
            high_hits = 3
            entry_date = (turn_date - timedelta(days=10)).isoformat()
            contradictions = ["Tony says he never uses checklists anymore."]
            return summary, category, importance, access, high_hits, entry_date, contradictions

    if turn % 13 == 0:
        summary = f"Tony mentioned passing thought turn {turn}."
        category = "general"
        importance = 0.09
        access = 0
        high_hits = 0
        entry_date = (turn_date - timedelta(days=420)).isoformat()
        contradictions = []
    elif turn % 9 == 0:
        summary = f"Tony emergency fund milestone turn {turn}."
        category = "finance"
        importance = 0.86
        access = 6
        high_hits = 4
        entry_date = (turn_date - timedelta(days=14)).isoformat()
        contradictions = []
    elif turn % 7 == 0:
        summary = f"Tony prefers direct checklist style turn {turn}."
        category = "preferences"
        importance = 0.78
        access = 5
        high_hits = 3
        entry_date = (turn_date - timedelta(days=10)).isoformat()
        contradictions = []
    else:
        summary = f"Tony is tracking family-work balance plan turn {turn}."
        category = "work"
        importance = 0.74
        access = 4
        high_hits = 2
        entry_date = (turn_date - timedelta(days=7)).isoformat()
        contradictions = []

    return summary, category, importance, access, high_hits, entry_date, contradictions


def run_phase45_stress(turns: int = 180, *, conflict_heavy: bool = False) -> dict:
    with TemporaryDirectory() as temp_dir:
        bot = _make_bot(Path(temp_dir))
        coherence = OutputCoherenceTracker(window_size=8)

        ingested = 0
        archived_total = 0
        forgetting_events = []

        try:
            for turn in range(1, turns + 1):
                # High-density ingestion with controlled signal quality.
                summary, category, importance, access, high_hits, entry_date, contradictions = _scenario_memory(
                    turn,
                    conflict_heavy=conflict_heavy,
                )

                entry = bot.normalize_memory_entry(
                    {
                        "summary": summary,
                        "category": category,
                        "mood": "neutral",
                        "created_at": entry_date,
                        "updated_at": entry_date,
                        "importance_score": importance,
                        "access_count": access,
                        "confidence_history": {"high": high_hits, "medium": 1, "low": 0},
                        "high_confidence_hits": high_hits,
                        "contradictions": contradictions,
                    }
                )
                catalog = list(bot.memory_catalog())
                catalog.append(entry)
                bot.save_memory_catalog(catalog)
                ingested += 1

                # Simple coherence tracking over deterministic mood-cycle replies.
                reply = f"I hear you, buddy. Step {turn} and we move forward together."
                coherence.record_reply(reply)

                # Periodic forgetting pass.
                if turn % 10 == 0:
                    result = bot.memory_coordinator.apply_controlled_forgetting(
                        turn_context=_turn_context(turn),
                    )
                    archived = int(result.get("archived", result.get("removed", 0)) or 0)
                    archived_total += archived
                    forgetting_events.append(
                        {
                            "turn": turn,
                            "archived": archived,
                            "retained": len(bot.memory_catalog()),
                            "threshold": float(result.get("threshold", 0.0) or 0.0),
                        }
                    )

            active = len(bot.memory_catalog())
            retention_ratio = active / float(max(1, ingested))
            effective_retention_ratio = active / float(max(1, active + archived_total))
            drift = coherence.detect_personality_drift(threshold=0.72)

            return {
                "turns": turns,
                "ingested": ingested,
                "active": active,
                "archived_total": archived_total,
                "retention_ratio": round(retention_ratio, 4),
                "effective_retention_ratio": round(effective_retention_ratio, 4),
                "scenario": "conflict-heavy" if conflict_heavy else "baseline-high-density",
                "drift": drift,
                "forgetting_events": forgetting_events,
            }
        finally:
            bot.shutdown()


if __name__ == "__main__":
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "profiles": [
            run_phase45_stress(180, conflict_heavy=False),
            run_phase45_stress(200, conflict_heavy=True),
        ],
    }
    out_path = Path("_phase45_stress_report.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"REPORT_PATH={out_path}")
