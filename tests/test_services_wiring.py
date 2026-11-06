"""Verify that the current service registry exposes the runtime services."""


def test_core_services_are_registered_and_delegated(bot):
    registry = bot.service_registry
    assert registry is bot.services.registry

    for service_name in ("tool_registry", "mood_manager", "health_manager", "tts_manager"):
        service = registry.get(service_name, optional=True)
        assert service is not None, f"{service_name} is not registered"
        assert getattr(bot, service_name) is service
