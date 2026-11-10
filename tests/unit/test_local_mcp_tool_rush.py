from __future__ import annotations

import asyncio
import pathlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dadbot_system import local_mcp_server as lms
from dadbot_system.delegation import SQLiteDelegationStore
from dadbot_system.mcp_security import (
    MCP_RESOURCE_URL_DEFAULT,
    MCP_TOOL_BASE_SCOPES,
    MCPToolManifestError,
    mcp_tool_scope,
    verify_mcp_tool_manifest,
)
from dadbot_system.security import AuthorizationError, ServiceTokenManager

pytestmark = pytest.mark.unit
TENANT = "t1_" + "a" * 26
SECRET = "local-mcp-authority-test-secret"


def _authority(tmp_path: pathlib.Path):
    SQLiteDelegationStore.initialize(str(tmp_path / "authority.sqlite"))
    store = SQLiteDelegationStore(str(tmp_path / "authority.sqlite"), SECRET)
    manager = ServiceTokenManager(SECRET)
    return store, manager, lms.MCPToolAuthority(manager, store, resource_url=MCP_RESOURCE_URL_DEFAULT)


def _access_context(token: str):
    from mcp.server.auth.middleware.auth_context import auth_context_var
    from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
    from mcp.server.auth.provider import AccessToken

    principal = ServiceTokenManager(SECRET).verify(token, expected_audience=MCP_RESOURCE_URL_DEFAULT)
    access = AccessToken(
        token=token,
        client_id=principal.subject,
        scopes=sorted(principal.scopes),
        expires_at=int(principal.expires_at),
        resource=MCP_RESOURCE_URL_DEFAULT,
        subject=principal.subject,
    )
    return auth_context_var, auth_context_var.set(AuthenticatedUser(access))


def _mcp_token(
    manager: ServiceTokenManager,
    *,
    tools: set[str],
    extra_scopes: frozenset[str] = frozenset(),
    ttl: int = 120,
):
    scopes = {lms.MCP_CONNECT_SCOPE, *extra_scopes}
    for tool_name in tools:
        scopes.add(mcp_tool_scope(tool_name))
        scopes.add(MCP_TOOL_BASE_SCOPES[tool_name])
    return manager.issue(
        subject="mcp-test-user",
        tenant_id=TENANT,
        scopes=scopes,
        allowed_tools=tools,
        ttl_seconds=ttl,
        audience=MCP_RESOURCE_URL_DEFAULT,
    )


def test_registered_reminder_uses_canonical_facade(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    pytest.importorskip("mcp.server.mcpserver")
    store, manager, authority = _authority(tmp_path)
    store.grant_membership(issuer="dadbot", subject="mcp-test-user", tenant_id=TENANT, scopes={"write"})
    reminder = {"title": "Call home", "due_at": "2026-06-01T10:00:00"}
    bot = SimpleNamespace(
        config=SimpleNamespace(tenant_id=TENANT),
        add_reminder=Mock(return_value=reminder),
    )
    server = lms.build_server(bot, authority=authority)
    token = _mcp_token(manager, tools={"add_reminder"})
    context_var, context_token = _access_context(token)
    try:
        result = server._tool_manager._tools["add_reminder"].fn("Call home", "tomorrow")
    finally:
        context_var.reset(context_token)
    assert result == reminder
    bot.add_reminder.assert_called_once_with("Call home", "tomorrow")


def test_mcp_scope_is_tool_specific_and_revocation_blocks_subsequent_calls(tmp_path: pathlib.Path):
    pytest.importorskip("mcp.server.mcpserver")
    store, manager, authority = _authority(tmp_path)
    store.grant_membership(issuer="dadbot", subject="mcp-test-user", tenant_id=TENANT, scopes={"write"})
    bot = SimpleNamespace(config=SimpleNamespace(tenant_id=TENANT), add_reminder=Mock(return_value={"ok": True}))
    server = lms.build_server(bot, authority=authority)
    tool = server._tool_manager._tools["add_reminder"].fn

    token = _mcp_token(manager, tools={"add_reminder"})
    context_var, context_token = _access_context(token)
    try:
        assert tool("Call home", "tomorrow") == {"ok": True}
        bot.config.tenant_id = "t1_" + "c" * 26
        with pytest.raises(AuthorizationError, match="configured server tenant"):
            tool("Call home", "tomorrow")
        bot.config.tenant_id = TENANT
        store.revoke_membership(issuer="dadbot", subject="mcp-test-user", tenant_id=TENANT)
        with pytest.raises(AuthorizationError, match="active tenant membership"):
            tool("Call home", "tomorrow")
    finally:
        context_var.reset(context_token)
    assert bot.add_reminder.call_count == 1


def test_mcp_token_verifier_enforces_audience_ttl_and_named_tool_grant(tmp_path: pathlib.Path):
    pytest.importorskip("mcp.server.mcpserver")
    _store, manager, authority = _authority(tmp_path)
    valid_token = _mcp_token(manager, tools={"add_reminder"})
    assert asyncio.run(authority.verify_token(valid_token)) is not None

    wrong_audience = manager.issue(
        subject="mcp-test-user",
        tenant_id=TENANT,
        scopes={lms.MCP_CONNECT_SCOPE, "write", mcp_tool_scope("add_reminder")},
        allowed_tools={"add_reminder"},
        ttl_seconds=120,
        audience="https://other.example/mcp",
    )
    missing_tool_scope = manager.issue(
        subject="mcp-test-user",
        tenant_id=TENANT,
        scopes={lms.MCP_CONNECT_SCOPE, "write"},
        allowed_tools={"add_reminder"},
        ttl_seconds=120,
        audience=MCP_RESOURCE_URL_DEFAULT,
    )
    long_lived = _mcp_token(manager, tools={"add_reminder"}, ttl=301)
    assert asyncio.run(authority.verify_token(wrong_audience)) is None
    assert asyncio.run(authority.verify_token(missing_tool_scope)) is None
    assert asyncio.run(authority.verify_token(long_lived)) is None


def test_mcp_tool_manifest_fails_closed_on_schema_drift(tmp_path: pathlib.Path):
    pytest.importorskip("mcp.server.mcpserver")
    _store, _manager, authority = _authority(tmp_path)
    server = lms.build_server(authority=authority)
    tools = asyncio.run(server.list_tools())
    verify_mcp_tool_manifest(tools)

    changed = [tool.model_dump(mode="json", by_alias=True, exclude_none=True) for tool in tools]
    changed[0]["description"] = "Unapproved description change."
    with pytest.raises(MCPToolManifestError, match="schema drift"):
        verify_mcp_tool_manifest(changed)


def test_streamable_http_requires_auth_and_only_lists_granted_tools(tmp_path: pathlib.Path):
    pytest.importorskip("mcp.server.mcpserver")
    from fastapi.testclient import TestClient

    store, manager, authority = _authority(tmp_path)
    store.grant_membership(issuer="dadbot", subject="mcp-test-user", tenant_id=TENANT, scopes={"read"})
    bot = SimpleNamespace(config=SimpleNamespace(tenant_id=TENANT))
    server = lms.build_server(bot, authority=authority)
    app = server.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
    )
    token = _mcp_token(manager, tools={"memory_snapshot"})
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    with TestClient(app, base_url=MCP_RESOURCE_URL_DEFAULT.rsplit("/mcp", 1)[0]) as client:
        unauthenticated = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1"},
                },
            },
        )
        assert unauthenticated.status_code == 401

        initialized = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1"},
                },
            },
        )
        assert initialized.status_code == 200
        listing = client.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}},
        )
        assert listing.status_code == 200
        tools = listing.json()["result"]["tools"]
        assert [tool["name"] for tool in tools] == ["memory_snapshot"]

        resource_listing = client.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 4, "method": "resources/list", "params": {}},
        )
        assert resource_listing.status_code == 200
        resources = resource_listing.json()["result"]["resources"]
        assert [resource["uri"] for resource in resources] == ["dadbot://memory/narratives"]

        denied_resource = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 5,
                "method": "resources/read",
                "params": {"uri": "dadbot://memory/heritage"},
            },
        )
        assert denied_resource.status_code == 200
        assert "error" in denied_resource.json()

        denied = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {"name": "add_reminder", "arguments": {"title": "Call home"}},
            },
        )
        assert denied.status_code == 200
        assert denied.json()["result"]["isError"] is True


def test_markdown_task_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    monkeypatch.setattr(lms, "_project_root", lambda: tmp_path)

    path = lms._task_file_path()
    tasks = [
        {"id": "tsk1", "title": "Ship release notes", "due": "2026-05-20T10:00:00", "priority": "high", "done": False},
        {"id": "tsk2", "title": "Archive old logs", "due": "", "priority": "low", "done": True},
    ]

    lms._write_markdown_tasks(path, tasks)
    loaded = lms._load_markdown_tasks(path)

    assert len(loaded) == 2
    assert loaded[0]["id"] == "tsk1"
    assert loaded[0]["done"] is False
    assert loaded[1]["id"] == "tsk2"
    assert loaded[1]["done"] is True


def test_python_function_refactor_rename_and_docstring(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    monkeypatch.setattr(lms, "_project_root", lambda: tmp_path)

    target = tmp_path / "module.py"
    target.write_text(
        "def sample(value):\n"
        "    return value * 2\n",
        encoding="utf-8",
    )

    result = lms._python_function_refactor(
        target,
        function_name="sample",
        new_name="sample_v2",
        prepend_docstring="Doubles the input.",
    )

    updated = target.read_text(encoding="utf-8")
    assert result["changed"] is True
    assert "def sample_v2(value):" in updated
    assert '"""Doubles the input."""' in updated


def test_python_function_refactor_missing_function(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    monkeypatch.setattr(lms, "_project_root", lambda: tmp_path)

    target = tmp_path / "module.py"
    target.write_text("def alpha():\n    return 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not found"):
        lms._python_function_refactor(target, function_name="beta")


def test_workspace_file_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    monkeypatch.setattr(lms, "_project_root", lambda: tmp_path)

    (tmp_path / "docs").mkdir()
    file_a = tmp_path / "docs" / "notes.md"
    file_b = tmp_path / "main.py"
    file_a.write_text("alpha beta gamma", encoding="utf-8")
    file_b.write_text("print('hello')", encoding="utf-8")

    files = lms._iter_workspace_files(tmp_path)
    names = sorted(path.name for path in files)

    assert names == ["main.py", "notes.md"]
    assert "alpha beta" in lms._safe_read_text(file_a)


def test_get_pending_executive_tasks_filters_and_limits(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    monkeypatch.setattr(lms, "_project_root", lambda: tmp_path)

    path = lms._task_file_path()
    lms._write_markdown_tasks(
        path,
        [
            {"id": "a", "title": "Open 1", "due": "2026-05-20", "priority": "normal", "done": False},
            {"id": "b", "title": "Done", "due": "", "priority": "low", "done": True},
            {"id": "c", "title": "Open 2", "due": "2026-05-19", "priority": "high", "done": False},
        ],
    )

    pending = lms.get_pending_executive_tasks(limit=1)

    assert len(pending) == 1
    assert pending[0]["done"] is False


def test_persist_research_memory_syncs_and_writes_event():
    calls = {"sync": 0, "event": 0}

    class _Writer:
        def write_event(self, *_args, **_kwargs):
            calls["event"] += 1
            return {"event_id": "evt-1"}

    class _Bot:
        ledger_writer = _Writer()

        @staticmethod
        def sync_semantic_memory_index(_rows):
            calls["sync"] += 1

    result = lms._persist_research_memory(_Bot(), url="https://example.com", preview="important finding")

    assert result["memory_indexed"] is True
    assert result["memory_event_written"] is True
    assert calls["sync"] == 1
    assert calls["event"] == 1
