from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from dadbot_system import DadBotOrchestrator, InMemoryStateStore, ServiceConfig, create_api_app
from dadbot_system.delegation import SQLiteDelegationStore
from dadbot_system.mcp_security import (
    MCP_CONNECT_SCOPE,
    MCP_RESOURCE_URL_DEFAULT,
    mcp_tool_scope,
)
from dadbot_system.security import ServiceTokenManager

TENANT = "t1_" + "b" * 26
SECRET = "mcp-token-exchange-test-secret"


class Broker:
    def enqueue(self, _task):
        raise AssertionError("MCP token exchange must not submit tasks")

    def get_result_nowait(self):
        return None


def _configured_app(tmp_path):
    config = ServiceConfig()
    config.security.auth_required = True
    config.security.token_secret = SECRET
    config.security.authority_sqlite_path = str(tmp_path / "authority.sqlite")
    SQLiteDelegationStore.initialize(config.security.authority_sqlite_path)
    authority = SQLiteDelegationStore(config.security.authority_sqlite_path, SECRET)
    authority.grant_membership(
        issuer="dadbot",
        subject="mcp-exchange-user",
        tenant_id=TENANT,
        scopes={"read", "write"},
    )
    app = create_api_app(
        DadBotOrchestrator(Broker(), state_store=InMemoryStateStore()),
        config=config,
    )
    token = ServiceTokenManager(SECRET).issue(
        subject="mcp-exchange-user",
        tenant_id=TENANT,
        scopes={"read", "write"},
        allowed_tools={"add_reminder", "list_reminders"},
        ttl_seconds=3600,
    )
    return app, authority, token


def test_mcp_exchange_issues_audience_bound_short_lived_tool_token(tmp_path):
    app, _authority, token = _configured_app(tmp_path)
    token_manager = ServiceTokenManager(SECRET)

    with TestClient(app) as client:
        response = client.post(
            "/v1/mcp/token",
            headers={"Authorization": f"Bearer {token}"},
            json={"tools": ["add_reminder"], "ttl_seconds": 90},
        )
        assert response.status_code == 200
        payload = response.json()

        principal = token_manager.verify(
            payload["access_token"],
            expected_audience=MCP_RESOURCE_URL_DEFAULT,
        )
        assert payload["token_type"] == "Bearer"
        assert payload["audience"] == MCP_RESOURCE_URL_DEFAULT
        assert payload["allowed_tools"] == ["add_reminder"]
        assert principal.allowed_tools == frozenset({"add_reminder"})
        assert principal.scopes == frozenset(
            {MCP_CONNECT_SCOPE, "write", mcp_tool_scope("add_reminder")},
        )
        assert 0 < principal.expires_at - principal.issued_at <= 90
        assert principal.expires_at <= time.time() + 90

        wrong_audience = client.get(
            "/v1/tasks/not-a-real-task",
            headers={"Authorization": f"Bearer {payload['access_token']}"},
        )
        assert wrong_audience.status_code == 401


def test_mcp_exchange_cannot_escalate_tool_or_scope_grants(tmp_path):
    app, _authority, token = _configured_app(tmp_path)
    with TestClient(app) as client:
        oversized_ttl = client.post(
            "/v1/mcp/token",
            headers={"Authorization": f"Bearer {token}"},
            json={"tools": ["add_reminder"], "ttl_seconds": 301},
        )
        assert oversized_ttl.status_code == 422

        denied_tool = client.post(
            "/v1/mcp/token",
            headers={"Authorization": f"Bearer {token}"},
            json={"tools": ["computer_write_text_file"]},
        )
        assert denied_tool.status_code == 403

        invalid_audience = ServiceTokenManager(SECRET).issue(
            subject="mcp-exchange-user",
            tenant_id=TENANT,
            scopes={"read", "write"},
            allowed_tools={"add_reminder"},
            audience=MCP_RESOURCE_URL_DEFAULT,
        )
        rejected = client.post(
            "/v1/mcp/token",
            headers={"Authorization": f"Bearer {invalid_audience}"},
            json={"tools": ["add_reminder"]},
        )
        assert rejected.status_code == 401
