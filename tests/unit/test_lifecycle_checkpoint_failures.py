from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from dadbot.core.boot_mixin import DadBotBootMixin
from dadbot.core.dadbot import DadBot
from dadbot.core.turn_mixin import DadBotTurnMixin


def test_snapshot_error_is_not_converted_to_empty_state():
    class Runtime(DadBotTurnMixin):
        def snapshot_session_state(self):
            raise OSError("snapshot unavailable")

    with pytest.raises(OSError, match="snapshot unavailable"):
        Runtime()._snapshot_kernel_state()


def test_missing_snapshot_provider_is_rejected():
    with pytest.raises(RuntimeError, match="snapshot provider"):
        DadBotTurnMixin()._snapshot_kernel_state()


def test_non_mapping_snapshot_is_rejected():
    class Runtime(DadBotTurnMixin):
        def snapshot_session_state(self):
            return None

    with pytest.raises(TypeError, match="mapping"):
        Runtime()._snapshot_kernel_state()


def test_missing_checkpoint_tap_is_reported(caplog):
    runtime = DadBotTurnMixin()
    with caplog.at_level(logging.WARNING):
        runtime._checkpoint_turn_state()
        runtime._checkpoint_turn_state()
    assert sum("checkpoint" in record.message.lower() for record in caplog.records) == 1


def test_configured_checkpoint_requires_checkpoint_api():
    runtime = DadBotTurnMixin()
    runtime.event_tap = object()
    with pytest.raises(RuntimeError, match="maybe_checkpoint"):
        runtime._checkpoint_turn_state()


def test_configured_checkpoint_receives_valid_snapshot():
    snapshots = []

    class Runtime(DadBotTurnMixin):
        event_tap = SimpleNamespace(maybe_checkpoint=lambda **kwargs: snapshots.append(kwargs))

        def snapshot_session_state(self):
            return {"turn": 1}

    Runtime()._checkpoint_turn_state(state_hash="hash")
    assert snapshots == [{"state_snapshot": {"kernel_state": {"turn": 1}}, "state_hash": "hash"}]


def test_shutdown_surfaces_failed_memory_flush():
    class Runtime(DadBotBootMixin):
        background_tasks = SimpleNamespace(shutdown=lambda **kwargs: None)

        def __init__(self):
            pass

        def _shutdown_ollama_async_client(self):
            pass

        def save_memory_store(self):
            raise OSError("flush failed")

    with pytest.raises(OSError, match="flush failed"):
        Runtime().shutdown()


def test_facade_rejects_missing_memory_manager():
    bot = object.__new__(DadBot)
    bot._memory_manager = None
    with pytest.raises(RuntimeError, match="persistence manager is unavailable"):
        bot.save_memory_store()


def test_boot_validates_managers_and_hydrates_before_ready(make_test_dadbot, monkeypatch):
    observed = []
    original = DadBot._validate_managers
    monkeypatch.setenv("DADBOT_VALIDATE_MANAGERS_SMOKE", "0")

    def validate(bot, *, smoke=False):
        observed.append((bot.lifecycle_state, bool(bot.MEMORY_STORE)))
        return original(bot, smoke=smoke)

    monkeypatch.setattr(DadBot, "_validate_managers", validate)
    bot = make_test_dadbot()
    assert observed == [("orchestration", True)]
    assert bot.lifecycle_state == "ready"
