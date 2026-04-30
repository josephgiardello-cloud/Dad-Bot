"""Integration coverage for memory persistence PII scrubbing."""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

from dadbot.managers.memory_coordination import MemoryCoordinator


class DummyBot:
    def __init__(self):
        self.saved_catalog = None

    def memory_catalog(self):
        return []

    def normalize_memory_text(self, text):
        return str(text or "").strip().lower()

    def normalize_memory_entry(self, entry):
        return entry

    def days_since_iso_date(self, value):
        from datetime import date, datetime

        if not value:
            return None
        try:
            parsed = datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
            return max(0, (date.today() - parsed).days)
        except ValueError:
            return None

    def normalize_mood(self, mood):
        _MOOD_MAP = {
            "happy": "positive",
            "great": "positive",
            "good": "positive",
            "sad": "low",
            "depressed": "low",
            "tired": "tired",
            "anxious": "anxious",
            "stressed": "anxious",
        }
        return _MOOD_MAP.get(str(mood or "").lower().strip(), "neutral")

    def save_memory_catalog(self, catalog):
        self.saved_catalog = catalog

    def record_runtime_issue(self, purpose, fallback, exc=None, **kwargs):
        pass


def test_update_memory_store_scrubs_pii_before_saving(monkeypatch):
    bot = DummyBot()
    bot._graph_commit_active = True
    coordinator = MemoryCoordinator(bot)
    turn_context = SimpleNamespace(
        temporal=SimpleNamespace(
            wall_time=datetime.now().isoformat(timespec="seconds"),
            wall_date=date.today().isoformat(),
        ),
        state={"_active_graph_stage": "save"},
    )

    def fake_extract_session_memories(history):
        return [
            {
                "summary": "My email is test@example.com and my phone is 555-123-4567.",
                "category": "personal",
                "mood": "neutral",
            }
        ]

    monkeypatch.setattr(coordinator, "extract_session_memories", fake_extract_session_memories)
    coordinator.update_memory_store([], turn_context=turn_context)

    assert bot.saved_catalog is not None
    assert len(bot.saved_catalog) == 1
    saved = bot.saved_catalog[0]
    assert "test@example.com" not in saved["summary"]
    assert "555-123-4567" not in saved["summary"]
    assert "[EMAIL REDACTED]" in saved["summary"]
    assert "[PHONE REDACTED]" in saved["summary"]
    assert "_pii_scrubbed" in saved
    assert "email" in saved["_pii_scrubbed"]
    assert "phone" in saved["_pii_scrubbed"]
