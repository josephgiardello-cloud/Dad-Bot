from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from dadbot_system.state import PostgresStateStore

TENANT_A = "t1_" + "a" * 26
TENANT_B = "t1_" + "b" * 26


@pytest.fixture
def postgres_session_store(postgres_test_dsn):
    import psycopg
    from psycopg import sql

    prefix = f"dadbot_session_test_{uuid.uuid4().hex}"
    names = (f"{prefix}_session", f"{prefix}_task", f"{prefix}_event")
    store = PostgresStateStore(
        postgres_test_dsn,
        session_table=names[0],
        task_table=names[1],
        event_table=names[2],
    )
    try:
        yield store
    finally:
        with psycopg.connect(postgres_test_dsn, autocommit=True) as connection:
            for name in (
                store.owned_sessions.session_table,
                store.owned_sessions.event_table,
                *names,
            ):
                connection.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(name)))


def test_real_postgres_session_ownership_listing_termination_and_recovery(postgres_session_store):
    from psycopg import sql

    store = postgres_session_store
    sessions = store.owned_sessions
    session_id = "shared-session"

    with pytest.raises(TypeError):
        store.load_session_state(session_id)
    with pytest.raises(TypeError):
        sessions.load_session_state(session_id)

    sessions.save_session_state(TENANT_A, session_id, {"owner": TENANT_A})
    sessions.save_session_state(TENANT_B, session_id, {"owner": TENANT_B})
    sessions.save_session_state(TENANT_B, "tenant-b-only", {"owner": TENANT_B})
    sessions.append_event(
        TENANT_A,
        session_id,
        {"tenant_id": TENANT_A, "session_id": session_id, "kind": "created"},
    )
    sessions.append_event(
        TENANT_B,
        session_id,
        {"tenant_id": TENANT_B, "session_id": session_id, "kind": "created"},
    )

    assert sessions.load_session_state(TENANT_A, session_id) == {"owner": TENANT_A}
    assert sessions.load_session_state(TENANT_B, session_id) == {"owner": TENANT_B}
    assert sessions.list_sessions(TENANT_A) == [session_id]
    assert sessions.list_sessions(TENANT_B) == [session_id, "tenant-b-only"]
    assert sessions.list_events(TENANT_A, session_id)[0]["tenant_id"] == TENANT_A
    assert sessions.list_events(TENANT_B, session_id)[0]["tenant_id"] == TENANT_B
    with pytest.raises(ValueError, match="ownership identity mismatch"):
        sessions.append_event(
            TENANT_A,
            session_id,
            {"tenant_id": TENANT_B, "session_id": session_id},
        )

    with sessions._tenant_connection(TENANT_A) as connection:
        connection.execute(
            sql.SQL("UPDATE {} SET payload = %s WHERE tenant_id = %s AND session_id = %s").format(
                sql.Identifier(sessions.session_table)
            ),
            (
                json.dumps({"tenant_id": TENANT_B, "session_id": session_id, "state": {}}),
                TENANT_A,
                session_id,
            ),
        )
    with pytest.raises(ValueError, match="ownership identity mismatch"):
        sessions.load_session_state(TENANT_A, session_id)
    with pytest.raises(ValueError, match="ownership identity mismatch"):
        sessions.save_session_state(TENANT_A, session_id, {"owner": TENANT_A})
    with sessions._tenant_connection(TENANT_A) as connection:
        connection.execute(
            sql.SQL("UPDATE {} SET payload = %s WHERE tenant_id = %s AND session_id = %s").format(
                sql.Identifier(sessions.session_table)
            ),
            (
                json.dumps({"tenant_id": TENANT_A, "session_id": session_id, "state": {"owner": TENANT_A}}),
                TENANT_A,
                session_id,
            ),
        )

    script = """
import json, os, sys
from dadbot_system.state import PostgresStateStore
store = PostgresStateStore(
    os.environ["DADBOT_TEST_POSTGRES_DSN"],
    session_table=sys.argv[1], task_table=sys.argv[2], event_table=sys.argv[3])
print(json.dumps({
    "tenant_a": store.owned_sessions.load_session_state(sys.argv[4], "shared-session"),
    "tenant_b": store.owned_sessions.load_session_state(sys.argv[5], "shared-session"),
}))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            store._session_table,
            store._task_table,
            store._event_table,
            TENANT_A,
            TENANT_B,
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    recovered = json.loads(result.stdout.strip().splitlines()[-1])
    assert recovered == {
        "tenant_a": {"owner": TENANT_A},
        "tenant_b": {"owner": TENANT_B},
    }

    def write_owned_session(tenant_id: str) -> None:
        sessions.save_session_state(tenant_id, "concurrent", {"owner": tenant_id})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write_owned_session, [TENANT_A, TENANT_B] * 8))
    assert sessions.load_session_state(TENANT_A, "concurrent") == {"owner": TENANT_A}
    assert sessions.load_session_state(TENANT_B, "concurrent") == {"owner": TENANT_B}

    assert sessions.delete_session(TENANT_A, session_id)
    assert sessions.load_session_state(TENANT_A, session_id) is None
    assert sessions.list_events(TENANT_A, session_id) == []
    assert sessions.load_session_state(TENANT_B, session_id) == {"owner": TENANT_B}
    assert len(sessions.list_events(TENANT_B, session_id)) == 1


def test_real_postgres_rls_tenant_binding_is_transaction_local(postgres_session_store):
    import psycopg
    from psycopg import sql

    store = postgres_session_store
    sessions = store.owned_sessions
    session_id = "rls-session"
    sessions.save_session_state(TENANT_A, session_id, {"owner": TENANT_A})
    sessions.save_session_state(TENANT_B, session_id, {"owner": TENANT_B})

    with psycopg.connect(store._dsn) as connection:
        assert connection.execute(
            "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
        ).fetchone() == (False, False)
        connection.execute("SELECT set_config('dadbot.tenant_id', %s, true)", (TENANT_A,))
        select = sql.SQL("SELECT tenant_id FROM {} WHERE session_id = %s").format(
            sql.Identifier(sessions.session_table)
        )
        assert connection.execute(select, (session_id,)).fetchall() == [(TENANT_A,)]
        update = sql.SQL("UPDATE {} SET payload = payload WHERE tenant_id = %s AND session_id = %s").format(
            sql.Identifier(sessions.session_table)
        )
        assert connection.execute(update, (TENANT_B, session_id)).rowcount == 0

        connection.commit()
        current_tenant = "SELECT current_setting('dadbot.tenant_id', true)"
        assert connection.execute(current_tenant).fetchone()[0] in (None, "")
        assert connection.execute(select, (session_id,)).fetchall() == []

        connection.execute("SELECT set_config('dadbot.tenant_id', %s, true)", (TENANT_B,))
        assert connection.execute(select, (session_id,)).fetchall() == [(TENANT_B,)]
        connection.rollback()
        assert connection.execute(current_tenant).fetchone()[0] in (None, "")


def test_real_postgres_rejects_mismatched_metadata_and_unowned_legacy_records(postgres_session_store):
    import psycopg
    from psycopg import sql

    store = postgres_session_store
    sessions = store.owned_sessions
    legacy_id = "legacy-session"
    with psycopg.connect(store._dsn) as connection:
        connection.execute(
            sql.SQL("INSERT INTO {} (session_id, payload) VALUES (%s, %s)").format(
                sql.Identifier(store._session_table)
            ),
            (legacy_id, json.dumps({"history": ["unowned"]})),
        )
    with pytest.raises(RuntimeError, match="Unowned legacy PostgreSQL session data"):
        sessions.load_session_state(TENANT_A, legacy_id)
    with pytest.raises(RuntimeError, match="Unowned legacy PostgreSQL session data"):
        sessions.save_session_state(TENANT_A, legacy_id, {"history": ["new"]})
