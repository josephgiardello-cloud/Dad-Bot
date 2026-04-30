from datetime import date, datetime, timedelta
from types import SimpleNamespace

from dadbot.services.context_service import ContextService


class StubContextBuilder:
    def build_core_persona_prompt(self):
        return "core"

    def build_dynamic_profile_context(self):
        return "profile"

    def build_relationship_context(self):
        return "relationship"

    def build_session_summary_context(self):
        return "summary"

    def build_memory_context(self, _user_input):
        return []

    def build_relevant_context(self, _user_input):
        return "relevant"

    def build_cross_session_context(self, _user_input):
        return "cross-session"


def test_context_service_passes_query_embedding_to_semantic_index():
    captured = {}

    class StubMemoryManager:
        def embed_text(self, text):
            captured["embedded_text"] = text
            return [0.1, 0.9]

    class StubSemanticIndex:
        def fetch_candidates(self, **kwargs):
            captured.update(kwargs)
            return []

    temporal = SimpleNamespace(
        wall_time=datetime.now().isoformat(timespec="seconds"),
        wall_date=date.today().isoformat(),
    )
    turn_ctx = SimpleNamespace(
        user_input="Tell me about last summer",
        temporal=temporal,
        temporal_snapshot=lambda: {
            "wall_time": temporal.wall_time,
            "wall_date": temporal.wall_date,
        },
    )
    service = ContextService(StubContextBuilder(), StubMemoryManager(), semantic_index=StubSemanticIndex())
    service.build_context(turn_ctx)

    assert captured["embedded_text"] == "Tell me about last summer"
    assert captured["query_embedding"] == [0.1, 0.9]
    assert captured["query_category"] == "general"
    assert captured["query_mood"] == "neutral"


def test_proactive_heartbeat_queues_daily_checkin_once(bot, monkeypatch):
    bot.MEMORY_STORE["pending_proactive_messages"] = []
    bot.MEMORY_STORE["last_mood_updated_at"] = (date.today() - timedelta(days=1)).isoformat()
    bot.MEMORY_STORE["last_daily_checkin_at"] = None

    monkeypatch.setattr(
        bot,
        "current_runtime_health_snapshot",
        lambda **_kwargs: {"level": "green", "updated_at": datetime.now().isoformat(timespec="seconds")},
    )

    morning = datetime.combine(date.today(), datetime.min.time()).replace(hour=9)
    later = morning.replace(hour=10)

    first = bot.maintenance_scheduler.run_proactive_heartbeat(reference_time=morning)
    second = bot.maintenance_scheduler.run_proactive_heartbeat(reference_time=later)

    queued = [entry for entry in bot.pending_proactive_messages() if entry.get("source") == "daily-checkin"]
    assert first["queued_daily_checkin"] == 1
    assert second["queued_daily_checkin"] == 0
    assert len(queued) == 1
