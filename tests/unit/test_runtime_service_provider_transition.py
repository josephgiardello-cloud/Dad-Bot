import inspect
import os

import pytest

from dadbot.core.dadbot import DadBot
from dadbot.core.boot_mixin import DadBotBootMixin
from dadbot.core.policy_store import DadPolicyStore, InMemoryAsyncPolicyPersistence
from dadbot.core.runtime_errors import ConfigurationError
from dadbot.core.turn_mixin import DadBotTurnMixin
from dadbot.core.world_model import InMemoryAsyncWorldModelPersistence, WorldModelStore
from dadbot.memory.ledger import InMemoryAsyncMemoryLedgerPersistence, MemoryLedger

pytestmark = pytest.mark.unit


class _RuntimeServicesStub:
    def __init__(self) -> None:
        self.policy_store = DadPolicyStore(InMemoryAsyncPolicyPersistence())
        self.memory_ledger = MemoryLedger(InMemoryAsyncMemoryLedgerPersistence())
        self.world_model_store = WorldModelStore(InMemoryAsyncWorldModelPersistence())

    def get_policy_store(self) -> DadPolicyStore:
        return self.policy_store

    def get_memory_ledger(self) -> MemoryLedger:
        return self.memory_ledger

    def get_world_model_store(self) -> WorldModelStore:
        return self.world_model_store


class _InvalidRuntimeServicesStub:
    def get_policy_store(self):
        return object()

    def get_memory_ledger(self):
        return object()

    def get_world_model_store(self):
        return object()


class _TurnMixinHarness(DadBotTurnMixin):
    pass


def test_turn_mixin_uses_injected_runtime_services_provider():
    bot = _TurnMixinHarness()
    runtime_services = _RuntimeServicesStub()
    bot.runtime_services = runtime_services

    assert bot._get_policy_store() is runtime_services.policy_store
    assert bot._get_memory_ledger() is runtime_services.memory_ledger
    assert bot._get_world_model_store() is runtime_services.world_model_store


def test_turn_mixin_rejects_invalid_runtime_service_provider_payloads():
    bot = _TurnMixinHarness()
    bot.runtime_services = _InvalidRuntimeServicesStub()

    with pytest.raises(ConfigurationError, match="policy store"):
        bot._get_policy_store()
    with pytest.raises(ConfigurationError, match="memory ledger"):
        bot._get_memory_ledger()
    with pytest.raises(ConfigurationError, match="world model store"):
        bot._get_world_model_store()


def test_boot_mixin_wires_core_runtime_services_dependency():
    source = inspect.getsource(DadBotBootMixin._initialize_services)
    assert "core_runtime_services" in source
    assert "DefaultCoreRuntimeServices" in source


def test_dadbot_prefers_explicit_runtime_services_constructor_injection(monkeypatch, tmp_path):
    monkeypatch.setenv("DADBOT_MEMORY_PATH", str(tmp_path / "memory.json"))
    monkeypatch.setenv("DADBOT_SEMANTIC_DB_PATH", str(tmp_path / "semantic.sqlite3"))
    monkeypatch.setenv("DADBOT_GRAPH_DB_PATH", str(tmp_path / "graph.sqlite3"))
    monkeypatch.setenv("DADBOT_SESSION_LOG_DIR", str(tmp_path / "session_logs"))

    runtime_services = _RuntimeServicesStub()
    bot = DadBot(runtime_services=runtime_services)
    try:
        assert bot.runtime_services is runtime_services
        assert bot._get_policy_store() is runtime_services.policy_store
        assert bot._get_memory_ledger() is runtime_services.memory_ledger
        assert bot._get_world_model_store() is runtime_services.world_model_store
    finally:
        bot.shutdown()


def test_resolve_dependency_prefers_explicit_service_overrides_over_registry():
    bot = object.__new__(DadBot)
    bot._explicit_dependencies = {"runtime_interface": "explicit"}
    bot._dependency_registry = {"runtime_interface": "registry"}

    resolved = DadBot._resolve_dependency(bot, "runtime_interface", lambda: "factory")

    assert resolved == "explicit"
