from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from dadbot.core.persistence import (
    CheckpointError,
    CheckpointIntegrityError,
    PostgresCheckpointer,
    RedisCheckpointer,
    RedisPostgresCheckpointer,
)


def _manifest() -> dict:
    return {
        "python_version": "3.13",
        "env_hash": "checkpoint-test",
        "dependency_versions": {},
        "timezone": "UTC",
    }


def _checkpoint(turn: int, checkpoint_hash: str, previous_hash: str = "") -> dict:
    return {
        "checkpoint_hash": checkpoint_hash,
        "prev_checkpoint_hash": previous_hash,
        "status": "completed",
        "state": {"turn": turn},
        "metadata": {},
    }


def _assert_fresh_process_recovery(
    backend: str,
    endpoint: str,
    namespace_or_prefix: str,
    tenant_id: str,
    expected_hash: str,
) -> None:
    script = r"""
import sys
from dadbot.core.persistence import PostgresCheckpointer, RedisCheckpointer

backend, endpoint, namespace_or_prefix, tenant_id, expected_hash = sys.argv[1:]
if backend == "postgres":
    checkpointer = PostgresCheckpointer(
        endpoint,
        default_tenant_id="tenant-default",
        table_prefix=namespace_or_prefix,
        trusted_local=True,
    )
else:
    checkpointer = RedisCheckpointer(
        endpoint,
        namespace=namespace_or_prefix,
        default_tenant_id="tenant-default",
        trusted_local=True,
    )
loaded = checkpointer.load_checkpoint("shared-session", tenant_id=tenant_id)
assert loaded["checkpoint_hash"] == expected_hash
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            backend,
            endpoint,
            namespace_or_prefix,
            tenant_id,
            expected_hash,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_remote_checkpointers_require_trusted_local_opt_in() -> None:
    with pytest.raises(CheckpointError, match="trusted-local opt-in"):
        PostgresCheckpointer("unused", trusted_local=False)
    with pytest.raises(CheckpointError, match="trusted-local opt-in"):
        RedisCheckpointer("unused", trusted_local=False)


@pytest.fixture
def postgres_checkpointer(postgres_test_dsn):
    import psycopg
    from psycopg import sql

    prefix = f"cp_{uuid.uuid4().hex[:16]}"
    checkpointer = PostgresCheckpointer(
        postgres_test_dsn,
        default_tenant_id="tenant-default",
        table_prefix=prefix,
        trusted_local=True,
    )
    try:
        yield checkpointer
    finally:
        with psycopg.connect(postgres_test_dsn, autocommit=True) as connection:
            for suffix in ("meta", "writes", "records"):
                connection.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(f"{prefix}_{suffix}")))


@pytest.fixture
def redis_checkpointer():
    redis_url = str(os.environ.get("DADBOT_TEST_REDIS_URL") or "").strip()
    if not redis_url:
        pytest.skip("Set DADBOT_TEST_REDIS_URL to run Redis checkpoint integration tests.")
    namespace = f"test-checkpoint-{uuid.uuid4().hex}"
    checkpointer = RedisCheckpointer(
        redis_url,
        namespace=namespace,
        default_tenant_id="tenant-default",
        trusted_local=True,
    )
    try:
        yield checkpointer
    finally:
        keys = list(checkpointer._client.scan_iter(match=f"{namespace}:*"))
        if keys:
            checkpointer._client.delete(*keys)
        checkpointer._client.close()


def _assert_remote_checkpoint_recovery(checkpointer, reloaded, backend: str) -> None:
    checkpointer.save_checkpoint(
        "shared-session",
        "trace-one",
        _checkpoint(1, f"{backend}-tenant-a-one"),
        _manifest(),
        tenant_id="tenant-a",
    )
    checkpointer.save_checkpoint(
        "shared-session",
        "trace-two",
        _checkpoint(2, f"{backend}-tenant-a-two", f"{backend}-tenant-a-one"),
        _manifest(),
        tenant_id="tenant-a",
    )
    checkpointer.save_checkpoint(
        "shared-session",
        "trace-two",
        _checkpoint(3, f"{backend}-tenant-a-two-replaced", f"{backend}-tenant-a-one"),
        _manifest(),
        tenant_id="tenant-a",
    )
    checkpointer.save_checkpoint(
        "shared-session",
        "trace-one",
        _checkpoint(1, f"{backend}-tenant-b-one"),
        _manifest(),
        tenant_id="tenant-b",
    )
    assert reloaded.load_checkpoint("shared-session", tenant_id="tenant-a")["checkpoint_hash"] == (
        f"{backend}-tenant-a-two-replaced"
    )
    assert reloaded.load_checkpoint(
        "shared-session",
        "trace-one",
        tenant_id="tenant-b",
    )["checkpoint_hash"] == f"{backend}-tenant-b-one"
    assert reloaded.checkpoint_count("shared-session", tenant_id="tenant-a") == 2
    assert reloaded.checkpoint_count("shared-session", tenant_id="tenant-b") == 1
    assert reloaded.load_checkpoint(
        "shared-session",
        "trace-one",
        current_manifest={**_manifest(), "env_hash": "different"},
        strict=False,
        tenant_id="tenant-a",
    )["checkpoint_hash"] == f"{backend}-tenant-a-one"
    with pytest.raises(CheckpointIntegrityError, match="Manifest drift"):
        reloaded.load_checkpoint(
            "shared-session",
            "trace-one",
            current_manifest={**_manifest(), "env_hash": "different"},
            strict=True,
            tenant_id="tenant-a",
        )
    with pytest.raises(CheckpointError):
        reloaded.load_checkpoint("shared-session", tenant_id="tenant-c")
    assert reloaded.validate_consistency("shared-session", tenant_id="tenant-a")["ok"]
    assert reloaded.prune_old_checkpoints(
        "shared-session",
        keep_count=1,
        tenant_id="tenant-a",
    ) == 1
    assert reloaded.checkpoint_count("shared-session", tenant_id="tenant-a") == 1
    assert reloaded.delete_session("shared-session", tenant_id="tenant-b") == 1
    assert reloaded.checkpoint_count("shared-session", tenant_id="tenant-b") == 0


def test_postgres_checkpoint_recovery_is_tenant_qualified_and_restart_safe(postgres_checkpointer) -> None:
    reloaded = PostgresCheckpointer(
        postgres_checkpointer._dsn,
        default_tenant_id="tenant-default",
        table_prefix=postgres_checkpointer._prefix,
        trusted_local=True,
    )
    _assert_remote_checkpoint_recovery(postgres_checkpointer, reloaded, "postgres")
    _assert_fresh_process_recovery(
        "postgres",
        postgres_checkpointer._dsn,
        postgres_checkpointer._prefix,
        "tenant-a",
        "postgres-tenant-a-two-replaced",
    )


def test_redis_checkpoint_recovery_is_tenant_qualified_and_restart_safe(redis_checkpointer) -> None:
    reloaded = RedisCheckpointer(
        os.environ["DADBOT_TEST_REDIS_URL"],
        namespace=redis_checkpointer._namespace,
        default_tenant_id="tenant-default",
        trusted_local=True,
    )
    try:
        _assert_remote_checkpoint_recovery(redis_checkpointer, reloaded, "redis")
        _assert_fresh_process_recovery(
            "redis",
            os.environ["DADBOT_TEST_REDIS_URL"],
            redis_checkpointer._namespace,
            "tenant-a",
            "redis-tenant-a-two-replaced",
        )
    finally:
        reloaded._client.close()


def test_postgres_checkpoint_write_and_audit_log_are_transactional(postgres_checkpointer) -> None:
    import psycopg
    from psycopg import sql

    with psycopg.connect(postgres_checkpointer._dsn) as connection:
        connection.execute(
            sql.SQL("ALTER TABLE {} ADD CONSTRAINT {} CHECK (status <> 'ok')").format(
                sql.Identifier(postgres_checkpointer._write_table),
                sql.Identifier("reject_successful_checkpoint"),
            ),
        )

    with pytest.raises(CheckpointError, match="Failed to save PostgreSQL checkpoint"):
        postgres_checkpointer.save_checkpoint(
            "interrupted-session",
            "trace",
            _checkpoint(1, "interrupted"),
            _manifest(),
            tenant_id="tenant-a",
        )
    assert postgres_checkpointer.checkpoint_count("interrupted-session", tenant_id="tenant-a") == 0
    with psycopg.connect(postgres_checkpointer._dsn) as connection:
        rows = connection.execute(
            sql.SQL(
                "SELECT status, error FROM {} WHERE tenant_id = %s AND session_id = %s ORDER BY id"
            ).format(sql.Identifier(postgres_checkpointer._write_table)),
            ("tenant-a", "interrupted-session"),
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "error"
    assert "reject_successful_checkpoint" in rows[0][1]


def test_redis_checkpoint_script_rejects_partial_write_before_mutation(redis_checkpointer) -> None:
    index_key, registry_key, writes_key = redis_checkpointer._session_keys("tenant-a", "interrupted-session")
    redis_checkpointer._client.set(registry_key, "unexpected-type")

    with pytest.raises(CheckpointError, match="Failed to save Redis checkpoint"):
        redis_checkpointer.save_checkpoint(
            "interrupted-session",
            "trace",
            _checkpoint(1, "interrupted"),
            _manifest(),
            tenant_id="tenant-a",
        )
    assert redis_checkpointer._client.get(
        redis_checkpointer._checkpoint_key("tenant-a", "interrupted-session", "dHJhY2U"),
    ) is None
    assert redis_checkpointer._client.zcard(index_key) == 0
    assert json.loads(redis_checkpointer._client.lindex(writes_key, 0))["status"] == "error"


def test_postgres_checkpoint_rejects_tampered_owner_metadata(postgres_checkpointer) -> None:
    import psycopg
    from psycopg import sql

    postgres_checkpointer.save_checkpoint(
        "owner-session",
        "owner-trace",
        _checkpoint(1, "owner-hash"),
        _manifest(),
        tenant_id="tenant-a",
    )
    with psycopg.connect(postgres_checkpointer._dsn) as connection:
        connection.execute(
            sql.SQL(
                "UPDATE {} SET payload = jsonb_set(payload, %s, to_jsonb(%s::text), true) "
                "WHERE tenant_id = %s AND session_id = %s AND trace_id = %s"
            ).format(sql.Identifier(postgres_checkpointer._checkpoint_table)),
            (
                ["checkpoint", "metadata", "tenant_id"],
                "tenant-b",
                "tenant-a",
                "owner-session",
                "owner-trace",
            ),
        )

    with pytest.raises(CheckpointIntegrityError, match="metadata tenant_id"):
        postgres_checkpointer.load_checkpoint(
            "owner-session",
            "owner-trace",
            tenant_id="tenant-a",
        )


def test_redis_checkpoint_rejects_tampered_owner_metadata(redis_checkpointer) -> None:
    redis_checkpointer.save_checkpoint(
        "owner-session",
        "owner-trace",
        _checkpoint(1, "owner-hash"),
        _manifest(),
        tenant_id="tenant-a",
    )
    encoded_trace = redis_checkpointer._encode("owner-trace")
    key = redis_checkpointer._checkpoint_key("tenant-a", "owner-session", encoded_trace)
    record = redis_checkpointer._get_record("tenant-a", "owner-session", "owner-trace")
    assert record is not None
    tampered = dict(record)
    tampered_payload = dict(tampered["payload"])
    tampered_checkpoint = dict(tampered_payload["checkpoint"])
    tampered_metadata = dict(tampered_checkpoint["metadata"])
    tampered_metadata["tenant_id"] = "tenant-b"
    tampered_checkpoint["metadata"] = tampered_metadata
    tampered_payload["checkpoint"] = tampered_checkpoint
    tampered["payload"] = tampered_payload
    redis_checkpointer._client.set(key, json.dumps(tampered))

    with pytest.raises(CheckpointIntegrityError, match="metadata tenant_id"):
        redis_checkpointer.load_checkpoint(
            "owner-session",
            "owner-trace",
            tenant_id="tenant-a",
        )


def test_postgres_is_authoritative_over_a_conflicting_redis_checkpoint(
    postgres_checkpointer,
    redis_checkpointer,
) -> None:
    durable_hash = "durable-checkpoint"
    postgres_checkpointer.save_checkpoint(
        "shared-session",
        "same-trace",
        _checkpoint(1, durable_hash),
        _manifest(),
        tenant_id="tenant-a",
    )
    redis_checkpointer.save_checkpoint(
        "shared-session",
        "same-trace",
        _checkpoint(2, "stale-cache-checkpoint"),
        _manifest(),
        tenant_id="tenant-a",
    )

    composite = RedisPostgresCheckpointer(postgres_checkpointer, redis_checkpointer)
    loaded = composite.load_checkpoint(
        "shared-session",
        "same-trace",
        tenant_id="tenant-a",
    )

    assert loaded["checkpoint_hash"] == durable_hash
    assert redis_checkpointer.load_checkpoint(
        "shared-session",
        "same-trace",
        tenant_id="tenant-a",
    )["checkpoint_hash"] == durable_hash
    assert composite.last_cache_error is None


def test_concurrent_same_trace_replacements_remain_complete(
    postgres_checkpointer,
    redis_checkpointer,
) -> None:
    for backend in (postgres_checkpointer, redis_checkpointer):
        hashes = ["concurrent-one", "concurrent-two"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    backend.save_checkpoint,
                    "concurrent-session",
                    "same-trace",
                    _checkpoint(index, checkpoint_hash),
                    _manifest(),
                    tenant_id="tenant-a",
                )
                for index, checkpoint_hash in enumerate(hashes, start=1)
            ]
            assert all(future.result(timeout=10) for future in futures)

        assert backend.checkpoint_count("concurrent-session", tenant_id="tenant-a") == 1
        assert backend.load_checkpoint(
            "concurrent-session",
            "same-trace",
            tenant_id="tenant-a",
        )["checkpoint_hash"] in hashes
