import asyncio
from types import SimpleNamespace

import pytest

from dadbot.core.graph_context import TurnContext
from dadbot.core.graph_pipeline_nodes import HealthNode
from dadbot.managers.long_term import LongTermSignalsManager
from dadbot.managers.maintenance import MaintenanceScheduler
from dadbot.services.maintenance_service import MaintenanceService

pytestmark = pytest.mark.unit


@pytest.fixture
def service():
    bot = SimpleNamespace()
    return MaintenanceService(MaintenanceScheduler(bot), LongTermSignalsManager(bot))


def test_health_tick_runs_proactive_jobs_without_inline_durable_synthesis(service, monkeypatch):
    calls = []

    def run_jobs():
        calls.append("proactive")
        return {"queued_reminders": 1}

    monkeypatch.setattr(service.maintenance, "run_scheduled_proactive_jobs", run_jobs)
    context = TurnContext("Synthetic health check")
    registry = SimpleNamespace(get=lambda *_args, **_kwargs: service)
    asyncio.run(HealthNode().execute(registry, context))
    assert calls == ["proactive"]
    assert context.state["health"] == {
        "status": "ok", "proactive": {"queued_reminders": 1},
        "ran": False, "archived": False, "consolidated_count": 0,
        "pattern_count": 0, "persona_evolved": False,
    }


def test_proactive_failure_is_logged_and_reported_as_degraded(service, monkeypatch, caplog):
    def unavailable():
        raise OSError("Synthetic proactive outage")

    monkeypatch.setattr(service.maintenance, "run_scheduled_proactive_jobs", unavailable)
    result = service.tick(None)
    assert result["status"] == "degraded"
    assert result["error"] == "Synthetic proactive outage"
    assert result["ran"] is False
    assert "Synthetic proactive outage" in caplog.text


def test_source_mode_skips_proactive_side_effects():
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Source mode consulted maintenance")

    registry = SimpleNamespace(get=forbidden)
    context = TurnContext("Synthetic source check", metadata={"source_mode": True})
    asyncio.run(HealthNode().execute(registry, context))
    assert context.state["health"] == {"skipped": True, "reason": "source_mode"}
