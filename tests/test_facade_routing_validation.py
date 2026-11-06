"""Verify DadBot facade access delegates to its live service registry."""

from dadbot.core.dadbot import DadBot


def test_facade_delegates_registered_services(bot):
    registry = bot.service_registry
    for service_name in ("health_manager", "mood_manager", "tool_registry"):
        service = registry.get(service_name, optional=True)
        assert service is not None, f"{service_name} is not registered"
        assert getattr(bot, service_name) is service


def test_legacy_unified_routing_table_is_retired():
    assert not hasattr(DadBot, "_UNIFIED_ROUTING")
