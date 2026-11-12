from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

from dadbot.core.boot_mixin import DadBotBootMixin
from dadbot.services.context_service import ContextService

pytestmark = pytest.mark.unit


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
    bot.MEMORY_STORE = {
        **bot.MEMORY_STORE,
        "pending_proactive_messages": [],
        "last_mood_updated_at": (date.today() - timedelta(days=1)).isoformat(),
        "last_daily_checkin_at": None,
    }

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


def test_background_heartbeat_uses_streamlit_turn_lock_for_event_dispatch(monkeypatch):
    calls = []
    lock_state = {"held": False}

    class _TrackedLock:
        def __enter__(self):
            lock_state["held"] = True

        def __exit__(self, *_args):
            lock_state["held"] = False

    monkeypatch.setattr(
        "dadbot.core.runtime_locks.runtime_turn_lock",
        lambda _bot: _TrackedLock(),
    )

    def _run_heartbeat(**kwargs):
        assert lock_state["held"]
        calls.append(kwargs)

    worker = SimpleNamespace(
        _proactive_heartbeat_interval_seconds=300,
        assistant=SimpleNamespace(run_heartbeat=_run_heartbeat),
        background_tasks=SimpleNamespace(wait_for_shutdown=lambda _timeout: True),
    )
    worker._run_proactive_heartbeat_tick = lambda: DadBotBootMixin._run_proactive_heartbeat_tick(worker)

    DadBotBootMixin._run_proactive_heartbeat_loop(worker)

    assert calls == [{"force": False}]
