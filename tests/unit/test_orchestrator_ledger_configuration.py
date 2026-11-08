from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from dadbot.core.control_plane import ExecutionJob
from dadbot.core.ledger_backend import FileWALLedgerBackend, InMemoryLedgerBackend
from dadbot.core.ledger_writer import LedgerWriter
from dadbot.core.orchestrator import DadBotOrchestrator


def _orchestrator(*, bot: object | None = None) -> DadBotOrchestrator:
    orchestrator = DadBotOrchestrator.__new__(DadBotOrchestrator)
    orchestrator.bot = bot
    orchestrator.graph = None
    orchestrator._enable_observability = True
    return orchestrator


def test_execution_ledger_remains_in_memory_by_default(monkeypatch) -> None:
    monkeypatch.delenv("DADBOT_DURABLE_EXECUTION_LEDGER", raising=False)

    control_plane = _orchestrator()._build_control_plane()

    assert isinstance(control_plane.ledger._backend, InMemoryLedgerBackend)


def test_durable_execution_ledger_uses_session_log_and_reloads(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    control_plane = _orchestrator(bot=SimpleNamespace(SESSION_LOG_DIR=tmp_path))._build_control_plane()

    assert isinstance(control_plane.ledger._backend, FileWALLedgerBackend)
    assert control_plane.ledger._backend.path() == tmp_path / "execution_ledger.jsonl"

    LedgerWriter(control_plane.ledger).append_job_submitted(
        ExecutionJob(session_id="test-session", user_input="hello"),
    )
    restarted_control_plane = _orchestrator(bot=SimpleNamespace(SESSION_LOG_DIR=tmp_path))._build_control_plane()

    assert len(restarted_control_plane.ledger.read()) == 1


def test_durable_execution_ledger_uses_configured_log_dir_without_bot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    monkeypatch.setenv("DADBOT_SESSION_LOG_DIR", str(tmp_path))

    control_plane = _orchestrator()._build_control_plane()

    assert isinstance(control_plane.ledger._backend, FileWALLedgerBackend)
    assert control_plane.ledger._backend.path() == tmp_path / "execution_ledger.jsonl"
