import pytest

from dadbot.config import DadBotConfig
from dadbot_system.contracts import PersistenceSettings, ServiceConfig

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("backend", [None, "auto", "sqlite", " SQLITE "])
@pytest.mark.parametrize("dsn", ["", "postgresql://configured-snapshot-service"])
def test_memory_indexes_can_use_sqlite_without_changing_postgres_snapshots(monkeypatch, backend, dsn):
    if backend is None:
        monkeypatch.delenv("DADBOT_MEMORY_INDEX_BACKEND", raising=False)
    else:
        monkeypatch.setenv("DADBOT_MEMORY_INDEX_BACKEND", backend)
    config = DadBotConfig(
        tenant_id="config-test",
        service_config=ServiceConfig(persistence=PersistenceSettings(postgres_dsn=dsn)),
    )
    assert config.postgres_dsn == dsn
    expected = "" if backend is not None and backend.strip().lower() == "sqlite" else dsn
    assert config.graph_store.postgres_dsn == expected
    assert config.semantic_index.postgres_dsn == expected


@pytest.mark.parametrize("backend", ["", "postgres", "invalid"])
def test_invalid_index_backend_fails_configuration(monkeypatch, backend):
    monkeypatch.setenv("DADBOT_MEMORY_INDEX_BACKEND", backend)
    with pytest.raises(ValueError, match="DADBOT_MEMORY_INDEX_BACKEND"):
        DadBotConfig(tenant_id="config-test", service_config=ServiceConfig())
