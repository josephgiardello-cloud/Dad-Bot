import pytest

from dadbot.core.services import ToolRegistry, build_services

pytestmark = pytest.mark.unit


def test_production_service_factory_rejects_local_stubs(monkeypatch):
    monkeypatch.setenv("DADBOT_ENVIRONMENT", "production")

    with pytest.raises(RuntimeError, match="tool_registry, planner"):
        build_services()


def test_production_service_factory_accepts_explicit_dependencies(monkeypatch):
    monkeypatch.setenv("DADBOT_ENVIRONMENT", "production")
    tool_registry = object()
    planner = object()

    services = build_services(tool_registry=tool_registry, planner=planner)

    assert services["tool_registry"] is tool_registry
    assert services["planner"] is planner


def test_development_service_factory_keeps_local_defaults(monkeypatch):
    monkeypatch.setenv("DADBOT_ENVIRONMENT", "development")

    services = build_services()

    assert isinstance(services["tool_registry"], ToolRegistry)
    assert services["tool_registry"].get("reminder") is not None
    assert services["planner"] is not None
