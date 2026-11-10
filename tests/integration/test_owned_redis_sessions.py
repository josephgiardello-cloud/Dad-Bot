from __future__ import annotations

import multiprocessing
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from dadbot_system.state import RedisStateStore

TENANT_A = "t1_" + "a" * 26
TENANT_B = "t1_" + "b" * 26


@pytest.fixture
def redis_session_store():
    url = str(os.environ.get("DADBOT_TEST_REDIS_URL") or "").strip()
    if not url:
        pytest.skip("Set DADBOT_TEST_REDIS_URL to run Redis session integration tests.")
    namespace = f"dadbot-session-test-{uuid.uuid4().hex}"
    store = RedisStateStore(url, namespace=namespace)
    store._client.ping()
    try:
        yield store
    finally:
        keys = list(store._client.scan_iter(match=f"{namespace}:*"))
        if keys:
            store._client.delete(*keys)
        store._client.close()


def _read_redis_session(url: str, namespace: str, tenant_id: str, output) -> None:
    store = RedisStateStore(url, namespace=namespace)
    try:
        output.put(store.owned_sessions.load_session_state(tenant_id, "shared-session"))
    finally:
        store._client.close()


def test_real_redis_session_ownership_listing_termination_and_recovery(redis_session_store):
    store = redis_session_store
    sessions = store.owned_sessions
    namespace = store._namespace
    url = os.environ["DADBOT_TEST_REDIS_URL"]

    with pytest.raises(TypeError):
        store.load_session_state("shared-session")
    with pytest.raises(TypeError):
        sessions.load_session_state("shared-session")

    sessions.save_session_state(TENANT_A, "shared-session", {"owner": TENANT_A})
    sessions.save_session_state(TENANT_B, "shared-session", {"owner": TENANT_B})
    sessions.save_session_state(TENANT_B, "tenant-b-only", {"owner": TENANT_B})
    sessions.append_event(
        TENANT_A,
        "shared-session",
        {"tenant_id": TENANT_A, "session_id": "shared-session", "kind": "created"},
    )
    sessions.append_event(
        TENANT_B,
        "shared-session",
        {"tenant_id": TENANT_B, "session_id": "shared-session", "kind": "created"},
    )

    assert sessions.load_session_state(TENANT_A, "shared-session") == {"owner": TENANT_A}
    assert sessions.load_session_state(TENANT_B, "shared-session") == {"owner": TENANT_B}
    assert sessions.list_sessions(TENANT_A) == ["shared-session"]
    assert sessions.list_sessions(TENANT_B) == ["shared-session", "tenant-b-only"]
    assert sessions.list_events(TENANT_A, "shared-session")[0]["tenant_id"] == TENANT_A
    assert sessions.list_events(TENANT_B, "shared-session")[0]["tenant_id"] == TENANT_B
    with pytest.raises(ValueError, match="ownership identity mismatch"):
        sessions.append_event(
            TENANT_A,
            "shared-session",
            {"tenant_id": TENANT_B, "session_id": "shared-session"},
        )

    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(
        target=_read_redis_session,
        args=(url, namespace, TENANT_B, output),
    )
    process.start()
    process.join(20)
    try:
        assert process.exitcode == 0
        assert output.get(timeout=5) == {"owner": TENANT_B}
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        output.close()
        output.join_thread()

    def write_owned_session(tenant_id: str) -> None:
        sessions.save_session_state(tenant_id, "concurrent", {"owner": tenant_id})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write_owned_session, [TENANT_A, TENANT_B] * 8))
    assert sessions.load_session_state(TENANT_A, "concurrent") == {"owner": TENANT_A}
    assert sessions.load_session_state(TENANT_B, "concurrent") == {"owner": TENANT_B}

    assert sessions.delete_session(TENANT_A, "shared-session")
    assert sessions.load_session_state(TENANT_A, "shared-session") is None
    assert sessions.list_events(TENANT_A, "shared-session") == []
    assert sessions.load_session_state(TENANT_B, "shared-session") == {"owner": TENANT_B}
    assert len(sessions.list_events(TENANT_B, "shared-session")) == 1


def test_real_redis_rejects_mismatched_metadata_and_unowned_legacy_records(redis_session_store):
    store = redis_session_store
    sessions = store.owned_sessions
    legacy_id = "legacy-session"
    store._client.set(f"{store._namespace}:session:{legacy_id}", '{"history":["unowned"]}')
    with pytest.raises(RuntimeError, match="Unowned legacy Redis session data"):
        sessions.load_session_state(TENANT_A, legacy_id)
    with pytest.raises(RuntimeError, match="Unowned legacy Redis session data"):
        sessions.save_session_state(TENANT_A, legacy_id, {"history": ["new"]})

    mismatched_id = "mismatched-session"
    store._client.set(
        sessions._keys(TENANT_A, mismatched_id)[0],
        '{"tenant_id":"%s","session_id":"%s","state":{}}' % (TENANT_B, mismatched_id),
    )
    with pytest.raises(ValueError, match="ownership identity mismatch"):
        sessions.load_session_state(TENANT_A, mismatched_id)
