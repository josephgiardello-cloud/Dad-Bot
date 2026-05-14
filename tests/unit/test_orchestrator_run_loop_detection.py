from __future__ import annotations

import asyncio

import pytest

from dadbot.core.orchestrator import DadBotOrchestrator


pytestmark = pytest.mark.unit


async def _fake_handle_turn(*_args, **_kwargs):
    return ("ok", False)


def _make_orchestrator_stub() -> DadBotOrchestrator:
    orchestrator = DadBotOrchestrator.__new__(DadBotOrchestrator)
    orchestrator.handle_turn = _fake_handle_turn  # type: ignore[assignment]
    return orchestrator


def test_run_accepts_no_running_event_loop_runtime_error(monkeypatch: pytest.MonkeyPatch):
    orchestrator = _make_orchestrator_stub()

    def _raise_no_loop() -> asyncio.AbstractEventLoop:
        raise RuntimeError("no running event loop")

    monkeypatch.setattr(asyncio, "get_running_loop", _raise_no_loop)
    result = orchestrator.run("hello")

    assert result == ("ok", False)


def test_run_does_not_swallow_unrelated_runtime_error(monkeypatch: pytest.MonkeyPatch):
    orchestrator = _make_orchestrator_stub()

    def _raise_unexpected() -> asyncio.AbstractEventLoop:
        raise RuntimeError("unexpected loop failure")

    monkeypatch.setattr(asyncio, "get_running_loop", _raise_unexpected)

    with pytest.raises(RuntimeError, match="unexpected loop failure"):
        orchestrator.run("hello")
