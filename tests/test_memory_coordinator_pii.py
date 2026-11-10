"""Integration coverage for memory persistence PII scrubbing."""

from __future__ import annotations

import json
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from dadbot.managers.memory_coordination import MemoryCoordinator
from dadbot.services.memory_service import extract_facts

pytestmark = pytest.mark.unit


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
        trace_id="trace-memory-1",
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
                "provenance": {
                    "source": "user_message",
                    "trust_level": 0.9,
                    "agent": "ollama_memory_extractor",
                    "evidence": "My email is test@example.com and my phone is 555-123-4567.",
                },
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
    assert saved["provenance"]["source"] == "user_message"
    assert saved["provenance"]["trust_level"] == 0.9
    assert saved["provenance"]["trace_id"] == "trace-memory-1"
    assert "test@example.com" not in saved["provenance"]["evidence"]
    assert "555-123-4567" not in saved["provenance"]["evidence"]
    assert "[EMAIL REDACTED]" in saved["provenance"]["evidence"]
    assert "[PHONE REDACTED]" in saved["provenance"]["evidence"]
    assert len(turn_context.state["_validated_memory_fact_refs"]) == 1


def test_extract_facts_requires_valid_schema_and_exact_source_evidence():
    source = "I am saving for a bike and feel excited."

    class Runtime:
        content = json.dumps(
            [
                {
                    "fact": "saving for a bike",
                    "category": "finance",
                    "mood": "excited",
                },
                {
                    "fact": "The user has a dog.",
                    "category": "personal",
                    "mood": "neutral",
                },
            ]
        )

        def call_ollama_chat(self, **kwargs):
            assert kwargs["response_format"] == "json"
            assert kwargs["purpose"] == "memory fact extraction"
            return self.content

        @staticmethod
        def extract_ollama_message_content(response):
            return response

        @staticmethod
        def parse_model_json_content(content):
            return json.loads(content)

    facts = extract_facts(Runtime(), source)

    assert len(facts) == 1
    assert facts[0].fact == "saving for a bike"

    Runtime.content = json.dumps(
        [{"summary": "legacy untyped memory", "category": "personal", "mood": "neutral"}]
    )
    with pytest.raises(ValueError):
        extract_facts(Runtime(), source)


def test_memory_coordinator_builds_provenance_from_extracted_user_fact():
    bot = DummyBot()
    source = "I am saving for a bike."
    bot.build_memory_transcript = lambda history: source
    bot.call_ollama_chat = lambda **kwargs: json.dumps(
        [
            {
                "fact": "I am saving for a bike.",
                "category": "finance",
                "mood": "neutral",
            }
        ]
    )
    bot.extract_ollama_message_content = lambda response: response
    bot.parse_model_json_content = json.loads

    memories = MemoryCoordinator(bot).extract_session_memories([{"role": "user", "content": source}])

    assert memories == [
        {
            "summary": "I am saving for a bike.",
            "category": "finance",
            "mood": "neutral",
            "provenance": {
                "source": "user_message",
                "trust_level": 0.9,
                "agent": "ollama_memory_extractor",
                "evidence": "I am saving for a bike.",
            },
        }
    ]


def test_update_memory_store_skips_without_turn_context_even_if_commit_active(monkeypatch):
    bot = DummyBot()
    bot._graph_commit_active = True
    coordinator = MemoryCoordinator(bot)

    called = {"extract": 0}

    def fake_extract_session_memories(history):
        called["extract"] += 1
        return [
            {
                "summary": "This should never be processed without turn context.",
                "category": "test",
                "mood": "neutral",
            }
        ]

    monkeypatch.setattr(coordinator, "extract_session_memories", fake_extract_session_memories)
    coordinator.update_memory_store([{"role": "user", "content": "remember this"}], turn_context=None)

    assert called["extract"] == 0
    assert bot.saved_catalog is None
