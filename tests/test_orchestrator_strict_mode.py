from __future__ import annotations

from types import SimpleNamespace

import pytest

from dadbot.core.orchestrator import DadBotOrchestrator

pytestmark = pytest.mark.unit


def test_production_orchestrator_rejects_non_strict_composition(monkeypatch) -> None:
    monkeypatch.setenv("DADBOT_ENVIRONMENT", "production")

    with pytest.raises(RuntimeError, match="Production composition requires strict mode"):
        DadBotOrchestrator(strict=False)


def test_orchestrator_propagates_strict_mode_to_persistence(monkeypatch) -> None:
    monkeypatch.setenv("DADBOT_ENVIRONMENT", "production")
    persistence_service = SimpleNamespace(strict_mode=False)

    class _Registry:
        def get(self, name: str, *, optional: bool = False):
            assert optional is True
            return persistence_service if name == "persistence_service" else None

    control_plane = SimpleNamespace(
        execution_token="",
        kernel_gateway=SimpleNamespace(submit_turn=lambda: None),
        submit_turn=lambda: None,
        registry=object(),
    )
    monkeypatch.setattr(DadBotOrchestrator, "_build_turn_graph", lambda self: object())
    monkeypatch.setattr(DadBotOrchestrator, "_build_control_plane", lambda self: control_plane)

    orchestrator = DadBotOrchestrator(registry=_Registry(), strict=True)

    assert orchestrator.strict is True
    assert persistence_service.strict_mode is True
