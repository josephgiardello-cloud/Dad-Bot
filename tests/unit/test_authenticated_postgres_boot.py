from types import SimpleNamespace

import pytest

from dadbot.core.boot_mixin import DadBotBootMixin


def test_authenticated_boot_rejects_postgres_execution_ledger(monkeypatch):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    monkeypatch.setenv("DADBOT_EXECUTION_LEDGER_BACKEND", "postgres")
    bot = object.__new__(DadBotBootMixin)
    bot._dependency_overrides = {"storage_security_mode": "AUTHENTICATED"}
    bot.config = SimpleNamespace(tenant_id="t1_test", session_log_dir=None)

    with pytest.raises(RuntimeError, match="not supported for authenticated production"):
        bot._initialize_execution_ledger()
