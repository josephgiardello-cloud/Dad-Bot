import base64
import hashlib
import hmac
import json
import time

import pytest

from dadbot_system import DadBotOrchestrator, InMemoryStateStore, ServiceConfig, create_api_app
from dadbot_system.delegation import SQLiteDelegationStore
from dadbot_system.security import ServiceTokenManager
from dadbot_system.state import SQLiteStateStore

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

SECRET = "service-principal-regression-key"
TENANT_A = "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"
TENANT_B = "t1_bbbbbbbbbbbbbbbbbbbbbbbbbb"


class Broker:
    def __init__(self):
        self.enqueued = []

    def enqueue(self, task):
        self.enqueued.append(task)

    def get_result_nowait(self):
        return None


def signed_token(payload, key=SECRET):
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    encoded = base64.urlsafe_b64encode(body).decode().rstrip("=")
    signature = hmac.new(key.encode(), body, hashlib.sha256).hexdigest()
    return f"{encoded}.{signature}"


def valid_claims():
    return {
        "v": 1,
        "sub": "principal",
        "tenant_id": TENANT_A,
        "scopes": ["write"],
        "allowed_tools": [],
        "iat": time.time() - 1,
        "exp": time.time() + 3600,
        "iss": "dadbot",
        "jti": "principal-1",
    }


@pytest.fixture
def configured_authority(tmp_path):
    config = ServiceConfig()
    config.security.token_secret = SECRET
    config.security.authority_sqlite_path = str(tmp_path / "authority.sqlite")
    SQLiteDelegationStore.initialize(config.security.authority_sqlite_path)
    SQLiteDelegationStore(config.security.authority_sqlite_path, SECRET).grant_membership(
        issuer="dadbot", subject="principal", tenant_id=TENANT_A, scopes={"write"}
    )
    return config


@pytest.mark.parametrize(
    ("claim", "value"),
    [
        ("sub", ""),
        ("sub", "anonymous"),
        ("tenant_id", ""),
        ("jti", ""),
        ("exp", float("nan")),
        ("exp", float("inf")),
        ("exp", 10**400),
        ("iat", float("nan")),
        ("v", "not-a-version"),
        ("scopes", "write"),
        ("allowed_tools", None),
    ],
)
def test_api_rejects_signed_incomplete_or_invalid_principal_before_submission(claim, value, configured_authority):
    claims = valid_claims()
    claims[claim] = value
    broker = Broker()
    config = configured_authority
    app = create_api_app(DadBotOrchestrator(broker, state_store=InMemoryStateStore()), config=config)

    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {signed_token(claims)}"},
            json={"tenant_id": claims["tenant_id"] or TENANT_A, "user_input": "hello"},
        )

    assert response.status_code == 401
    assert not broker.enqueued


def test_api_rejects_forged_principal_before_submission(configured_authority):
    broker = Broker()
    config = configured_authority
    app = create_api_app(DadBotOrchestrator(broker, state_store=InMemoryStateStore()), config=config)

    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {signed_token(valid_claims(), key='different-key')}"},
            json={"tenant_id": TENANT_A, "user_input": "hello"},
        )

    assert response.status_code == 401
    assert not broker.enqueued


@pytest.mark.parametrize(("subject", "tenant"), [("", TENANT_A), ("anonymous", TENANT_A), ("user", "")])
def test_token_issuer_rejects_missing_or_anonymous_identity(subject, tenant):
    with pytest.raises(ValueError):
        ServiceTokenManager(SECRET).issue(subject=subject, tenant_id=tenant, scopes={"write"})


def test_authorized_principal_can_submit_same_tenant_task(configured_authority):
    broker = Broker()
    config = configured_authority
    app = create_api_app(DadBotOrchestrator(broker, state_store=InMemoryStateStore()), config=config)
    token = ServiceTokenManager(SECRET).issue(subject="principal", tenant_id=TENANT_A, scopes={"write"})

    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": TENANT_A, "user_input": "hello"},
        )

    assert response.status_code == 200
    assert broker.enqueued[0].tenant_id == TENANT_A


def test_revoked_membership_cannot_read_session_events(configured_authority, tmp_path):
    config = configured_authority
    authority = SQLiteDelegationStore(config.security.authority_sqlite_path, SECRET)
    authority.grant_membership(
        issuer="dadbot", subject="principal", tenant_id=TENANT_A, scopes={"read"}
    )
    config.persistence.sqlite_path = str(tmp_path / "state.sqlite")
    state = SQLiteStateStore(config.persistence.sqlite_path)
    state.owned_sessions.append_event(
        TENANT_A,
        "same-session",
        {"tenant_id": TENANT_A, "session_id": "same-session", "marker": "synthetic-private"},
    )
    app = create_api_app(DadBotOrchestrator(Broker(), state_store=state), config=config)
    token = ServiceTokenManager(SECRET).issue(subject="principal", tenant_id=TENANT_A, scopes={"read"})
    headers = {"Authorization": f"Bearer {token}"}

    with TestClient(app) as client:
        before = client.get(f"/v1/sessions/same-session/events?tenant_id={TENANT_A}", headers=headers)
        assert before.status_code == 200
        assert before.json()[0]["marker"] == "synthetic-private"
        authority.revoke_membership(issuer="dadbot", subject="principal", tenant_id=TENANT_A)
        after = client.get(f"/v1/sessions/same-session/events?tenant_id={TENANT_A}", headers=headers)

    assert after.status_code in {401, 403}


def test_read_scope_requires_active_tenant_membership(configured_authority, tmp_path):
    config = configured_authority
    config.persistence.sqlite_path = str(tmp_path / "state.sqlite")
    state = SQLiteStateStore(config.persistence.sqlite_path)
    app = create_api_app(DadBotOrchestrator(Broker(), state_store=state), config=config)
    token = ServiceTokenManager(SECRET).issue(subject="principal", tenant_id=TENANT_A, scopes={"read"})

    with TestClient(app) as client:
        response = client.get(
            f"/v1/sessions/session-without-read/events?tenant_id={TENANT_A}",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 403


@pytest.mark.parametrize("failure", ["revoked", "unavailable"])
def test_invalid_read_authority_is_rejected_before_event_sql(configured_authority, tmp_path, failure):
    class TracedSQLiteStateStore(SQLiteStateStore):
        def __init__(self, path):
            self.statements = []
            super().__init__(path)

        def _connect(self):
            connection = super()._connect()
            connection.set_trace_callback(self.statements.append)
            return connection

    config = configured_authority
    authority_path = tmp_path / "authority.sqlite"
    authority = SQLiteDelegationStore(str(authority_path), SECRET)
    authority.grant_membership(
        issuer="dadbot", subject="principal", tenant_id=TENANT_A, scopes={"read"}
    )
    config.security.authority_sqlite_path = str(authority_path)
    config.persistence.sqlite_path = str(tmp_path / "state.sqlite")
    state = TracedSQLiteStateStore(config.persistence.sqlite_path)
    state.owned_sessions.append_event(
        TENANT_A,
        "same-session",
        {"tenant_id": TENANT_A, "session_id": "same-session", "marker": "synthetic-private"},
    )
    app = create_api_app(DadBotOrchestrator(Broker(), state_store=state), config=config)
    token = ServiceTokenManager(SECRET).issue(subject="principal", tenant_id=TENANT_A, scopes={"read"})
    headers = {"Authorization": f"Bearer {token}"}

    with TestClient(app) as client:
        before = client.get(f"/v1/sessions/same-session/events?tenant_id={TENANT_A}", headers=headers)
        assert before.status_code == 200
        assert before.json()[0]["marker"] == "synthetic-private"
        state.statements.clear()
        if failure == "revoked":
            authority.revoke_membership(issuer="dadbot", subject="principal", tenant_id=TENANT_A)
        else:
            authority_path.rename(tmp_path / "authority-unavailable.sqlite")
        response = client.get(f"/v1/sessions/same-session/events?tenant_id={TENANT_A}", headers=headers)

    event_reads = [
        statement for statement in state.statements
        if statement.strip().upper().startswith("SELECT") and "FROM events" in statement
    ]
    assert not event_reads, "Protected event SQL ran after authority became invalid"
    assert response.status_code in ({401, 403} if failure == "revoked" else {503})


@pytest.mark.parametrize("kind", ["missing", "expired", "wrong-tenant-admin", "invalid-signature"])
def test_api_denies_invalid_authority_without_queuing(kind, configured_authority):
    broker = Broker()
    config = configured_authority
    app = create_api_app(DadBotOrchestrator(broker, state_store=InMemoryStateStore()), config=config)
    claims = valid_claims()
    if kind == "expired":
        claims["exp"] = time.time() - 0.1
        claims["iat"] = time.time() - 10
    if kind == "wrong-tenant-admin":
        claims["scopes"] = ["admin"]
    token = signed_token(claims)
    if kind == "invalid-signature":
        token = token.rsplit(".", 1)[0] + ".invalid"
    headers = {} if kind == "missing" else {"Authorization": f"Bearer {token}"}
    tenant = TENANT_B if kind == "wrong-tenant-admin" else TENANT_A

    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers=headers,
            json={
                "tenant_id": tenant,
                "user_input": "hello",
                "metadata": {"auth": {"tenant_id": tenant, "scopes": ["admin"]}},
            },
        )

    assert response.status_code == (403 if kind == "wrong-tenant-admin" else 401)
    assert not broker.enqueued
