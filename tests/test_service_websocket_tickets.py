from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from dadbot_system import DadBotOrchestrator, InMemoryStateStore, ServiceConfig, create_api_app
from dadbot_system.delegation import SQLiteDelegationStore
from dadbot_system.security import ServiceTokenManager

pytestmark = pytest.mark.integration
SECRET = "websocket-ticket-test-secret"


class SharedRedis:
    def __init__(self):
        self.values = {}
        self.lock = threading.Lock()

    def set(self, name, value, *, ex, nx):
        with self.lock:
            assert nx is True
            assert 0 < ex <= 60
            if name in self.values:
                return False
            self.values[name] = (value, time.time() + ex)
            return True

    def getdel(self, name):
        with self.lock:
            entry = self.values.pop(name, None)
            return entry[0] if entry and entry[1] > time.time() else None

    def close(self):
        pass


def headers(*, tenant="t1_aaaaaaaaaaaaaaaaaaaaaaaaaa", scopes=("read", "write"), ttl=3600):
    token = ServiceTokenManager(SECRET).issue(subject="browser", tenant_id=tenant, scopes=set(scopes), ttl_seconds=ttl)
    return {"Authorization": f"Bearer {token}"}


def make_app(*, redis_url="redis://ticket-test.invalid/0", secret=SECRET, auth_required=True):
    config = ServiceConfig.from_dict(
        {
            "security": {
                "token_secret": secret,
                "auth_required": auth_required,
                "authority_sqlite_path": make_app.authority_path,
            },
            "persistence": {"redis_url": redis_url},
        }
    )
    broker = SimpleNamespace(enqueue=lambda task: None, get_result_nowait=lambda: None)
    orchestrator = DadBotOrchestrator(broker, state_store=InMemoryStateStore(), planner_debug_factory=dict)
    return create_api_app(orchestrator, config=config, runtime_bot=SimpleNamespace())


@pytest.fixture(autouse=True)
def configured_authority(tmp_path, monkeypatch):
    path = str(tmp_path / "authority.sqlite")
    SQLiteDelegationStore.initialize(path)
    SQLiteDelegationStore(path, SECRET).grant_membership(
        issuer="dadbot",
        subject="browser",
        tenant_id="t1_aaaaaaaaaaaaaaaaaaaaaaaaaa",
        scopes={"read", "write"},
    )
    monkeypatch.setattr(make_app, "authority_path", path, raising=False)


@pytest.fixture
def shared_redis(monkeypatch):
    redis = pytest.importorskip("redis")
    backend = SharedRedis()
    monkeypatch.setattr(redis, "from_url", lambda *args, **kwargs: backend)
    return backend


def issue(client, *, channel="pulse", session="first", tenant="t1_aaaaaaaaaaaaaaaaaaaaaaaaaa", credentials=None):
    return client.post(
        "/v1/websocket/ticket",
        headers=credentials if credentials is not None else headers(),
        json={"channel": channel, "session_id": session, "tenant_id": tenant},
    )


def pulse_path(ticket, session="first", tenant="t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"):
    return f"/v1/sessions/{session}/pulse/stream?ticket={ticket}&tenant_id={tenant}"


def test_ticket_is_consumed_once_across_api_workers(shared_redis):
    with TestClient(make_app()) as first, TestClient(make_app()) as second:
        response = issue(first)
        assert response.status_code == 200
        payload = response.json()
        assert response.headers["cache-control"] == "no-store"
        assert 0 < payload["expires_in"] <= 60
        token = payload["ticket"]
        assert all(token not in key for key in shared_redis.values)
        with second.websocket_connect(pulse_path(token)) as websocket:
            assert websocket.receive_json()["tenant_id"] == "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"
        with pytest.raises(WebSocketDisconnect) as error:
            with first.websocket_connect(pulse_path(token)):
                pass
        assert error.value.code == 4401


@pytest.mark.parametrize(
    ("channel", "path"),
    [
        ("events", "/v1/sessions/first/events/stream"),
        ("spatial", "/v1/spatial/ws"),
    ],
)
def test_other_websocket_channels_accept_their_own_tickets(shared_redis, channel, path):
    with TestClient(make_app()) as client:
        response = issue(client, channel=channel)
        assert response.status_code == 200
        token = response.json()["ticket"]
        with client.websocket_connect(f"{path}?ticket={token}"):
            pass


@pytest.mark.parametrize(
    ("auth_required", "secret_key", "allowed"),
    [(False, "", True), (False, "server-key", False), (True, "", False)],
)
def test_no_auth_development_mode_cannot_bypass_server_authentication(monkeypatch, auth_required, secret_key, allowed):
    monkeypatch.setenv("DADBOT_SECRET_KEY", secret_key)
    monkeypatch.setenv("DADBOT_ALLOW_LEGACY_DEFAULT_TENANT", "true")
    with TestClient(make_app(redis_url="", auth_required=auth_required)) as client:
        path = "/v1/sessions/first/pulse/stream"
        if allowed:
            with client.websocket_connect(path) as websocket:
                assert websocket.receive_json()["tenant_id"] == "default"
        else:
            with pytest.raises(WebSocketDisconnect) as error:
                with client.websocket_connect(path):
                    pass
            assert error.value.code == 4401


@pytest.mark.parametrize(
    ("payload", "credentials", "status"),
    [
        ({"channel": "pulse", "session_id": "first"}, {}, 401),
        ({"channel": "spatial"}, headers(scopes=("read",)), 403),
        ({"channel": "pulse", "session_id": "first", "tenant_id": "t1_bbbbbbbbbbbbbbbbbbbbbbbbbb"}, headers(), 403),
        ({"channel": "invalid"}, headers(), 422),
        ({"channel": "pulse"}, headers(), 422),
        ({"channel": "pulse", "session_id": "a/b"}, headers(), 422),
    ],
)
def test_ticket_issuance_validates_identity_scope_and_target(shared_redis, payload, credentials, status):
    with TestClient(make_app()) as client:
        response = client.post("/v1/websocket/ticket", json=payload, headers=credentials)
    assert response.status_code == status
    assert not shared_redis.values


@pytest.mark.parametrize(
    "wrong_path",
    [
        "/v1/sessions/second/pulse/stream?tenant_id=family-a",
        "/v1/sessions/first/events/stream?tenant_id=family-a",
        "/v1/spatial/ws?tenant_id=family-a",
        "/v1/sessions/first/pulse/stream?tenant_id=family-b",
    ],
)
def test_ticket_is_bound_to_session_channel_and_tenant(shared_redis, wrong_path):
    with TestClient(make_app()) as client:
        token = issue(client).json()["ticket"]
        with pytest.raises(WebSocketDisconnect) as error:
            with client.websocket_connect(f"{wrong_path}&ticket={token}"):
                pass
        assert error.value.code == 4401
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(pulse_path(token)):
                pass


def test_ticket_cannot_cross_service_secret_boundaries(shared_redis):
    with TestClient(make_app()) as first, TestClient(make_app(secret="other-service")) as second:
        token = issue(first).json()["ticket"]
        with pytest.raises(WebSocketDisconnect):
            with second.websocket_connect(pulse_path(token)):
                pass
        with first.websocket_connect(pulse_path(token)) as websocket:
            assert websocket.receive_json()["tenant_id"] == "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"


def test_ticket_expires_no_later_than_the_original_principal(shared_redis, monkeypatch):
    from dadbot_system import security

    with TestClient(make_app()) as client:
        payload = issue(client, credentials=headers(ttl=2)).json()
        assert 0 < payload["expires_in"] <= 2
        monkeypatch.setattr(security, "time", SimpleNamespace(time=lambda: payload["expires_at"] + 1))
        with pytest.raises(WebSocketDisconnect) as error:
            with client.websocket_connect(pulse_path(payload["ticket"])):
                pass
        assert error.value.code == 4401


def test_ticket_store_is_required_and_fails_closed(shared_redis, monkeypatch, caplog):
    import redis

    with TestClient(make_app(redis_url="")) as client:
        assert issue(client).status_code == 503

    def unavailable(*args, **kwargs):
        raise redis.exceptions.ConnectionError("simulated outage")

    with TestClient(make_app()) as client:
        token = issue(client).json()["ticket"]
        monkeypatch.setattr(shared_redis, "set", unavailable)
        assert issue(client).status_code == 503
        monkeypatch.setattr(shared_redis, "getdel", unavailable)
        with pytest.raises(WebSocketDisconnect) as error:
            with client.websocket_connect(pulse_path(token)):
                pass
        assert error.value.code == 1013
    assert "ticket store unavailable" in caplog.text
    assert token not in caplog.text


def test_ticket_exchange_checks_secret_key_without_putting_it_in_websocket_url(shared_redis, monkeypatch):
    monkeypatch.setenv("DADBOT_SECRET_KEY", "ticket-secret-key")
    with TestClient(make_app()) as client:
        assert issue(client).status_code == 401
        credentials = {**headers(), "X-DADBOT-KEY": "ticket-secret-key"}
        response = issue(client, credentials=credentials)
        assert response.status_code == 200
        with client.websocket_connect(pulse_path(response.json()["ticket"])) as websocket:
            assert websocket.receive_json()["tenant_id"] == "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"


def test_secret_protected_ticket_exchange_allows_only_configured_cors_preflights(shared_redis, monkeypatch):
    monkeypatch.setenv("DADBOT_SECRET_KEY", "ticket-secret-key")
    with TestClient(make_app()) as client:
        preflight = {
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type,x-dadbot-key",
        }
        response = client.options("/v1/websocket/ticket", headers=preflight)
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost:8501"
        rejected = client.options("/v1/websocket/ticket", headers={**preflight, "Origin": "https://untrusted.invalid"})
        assert rejected.status_code == 400
        assert "access-control-allow-origin" not in rejected.headers
        assert issue(client).status_code == 401
    assert not shared_redis.values


@pytest.mark.skipif(not os.environ.get("DADBOT_TEST_REDIS_URL"), reason="requires an isolated Redis test instance")
def test_real_redis_cross_worker_exchange_and_atomic_consumption():
    from dadbot_system.security import AuthenticationError, RedisWebSocketTicketStore

    url = os.environ["DADBOT_TEST_REDIS_URL"]
    with TestClient(make_app(redis_url=url)) as first, TestClient(make_app(redis_url=url)) as second:
        token = issue(first).json()["ticket"]
        with second.websocket_connect(pulse_path(token)) as websocket:
            assert websocket.receive_json()["tenant_id"] == "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"
        with pytest.raises(WebSocketDisconnect):
            with first.websocket_connect(pulse_path(token)):
                pass

    store = RedisWebSocketTicketStore(url, namespace="isolated-atomic-test")
    principal = ServiceTokenManager(SECRET).verify(headers()["Authorization"].removeprefix("Bearer "))
    token, _ = store.issue(principal, path="/spatial/ws")

    def consume(_):
        try:
            return store.consume(token, path="/spatial/ws")
        except AuthenticationError:
            return None

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(consume, range(8)))
        assert sum(result is not None for result in results) == 1
        assert next(result for result in results if result is not None) == principal
    finally:
        store.close()


@pytest.mark.parametrize("credential", ["access_token", "dadbot_key", "x_dadbot_key"])
@pytest.mark.parametrize(
    "path", ["/v1/spatial/ws", "/v1/sessions/first/pulse/stream", "/v1/sessions/first/events/stream"]
)
def test_long_lived_query_credentials_are_rejected(credential, path):
    with TestClient(make_app(redis_url="")) as client:
        with pytest.raises(WebSocketDisconnect) as error:
            with client.websocket_connect(f"{path}?{credential}=forbidden", headers=headers()):
                pass
        assert error.value.code == 4401
