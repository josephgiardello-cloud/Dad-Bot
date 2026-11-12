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
    tenant_id: str = TENANT,
):
    scopes = {lms.MCP_CONNECT_SCOPE, *extra_scopes}
    for tool_name in tools:
        scopes.add(mcp_tool_scope(tool_name))
        scopes.add(MCP_TOOL_BASE_SCOPES[tool_name])
    return manager.issue(
        subject="mcp-test-user",
        tenant_id=tenant_id,
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
    import json
    import subprocess
    import sys

    monkeypatch.setattr(lms, "_project_root", lambda: tmp_path)

    path = lms._task_file_path(TENANT)
    tasks = [
        {"id": "tsk1", "title": "Ship release notes", "due": "2026-05-20T10:00:00", "priority": "high", "done": False},
        {"id": "tsk2", "title": "Archive old logs", "due": "", "priority": "low", "done": True},
    ]

    lms._write_markdown_tasks(path, tasks, tenant_id=TENANT)
    loaded = lms._load_markdown_tasks(path, tenant_id=TENANT)

    assert len(loaded) == 2
    assert loaded[0]["id"] == "tsk1"
    assert loaded[0]["done"] is False
    assert loaded[1]["id"] == "tsk2"
    assert loaded[1]["done"] is True
    script = (
        "import json, pathlib, sys; from dadbot_system import local_mcp_server as lms; "
        "lms._project_root = lambda: pathlib.Path(sys.argv[1]); "
        "print(json.dumps(lms.get_pending_executive_tasks(tenant_id=sys.argv[2])))"
    )
    restored = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), TENANT],
        capture_output=True, text=True, check=True, timeout=30,
    )
    assert json.loads(restored.stdout) == [loaded[0]]


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

    path = lms._task_file_path(TENANT)
    lms._write_markdown_tasks(
        path,
        [
            {"id": "a", "title": "Open 1", "due": "2026-05-20", "priority": "normal", "done": False},
            {"id": "b", "title": "Done", "due": "", "priority": "low", "done": True},
            {"id": "c", "title": "Open 2", "due": "2026-05-19", "priority": "high", "done": False},
        ],
        tenant_id=TENANT,
    )

    pending = lms.get_pending_executive_tasks(tenant_id=TENANT, limit=1)

    assert len(pending) == 1
    assert pending[0]["done"] is False


@pytest.mark.parametrize("operation", ["list", "complete"])
def test_executive_tools_do_not_expose_or_complete_another_tenants_tasks(
    monkeypatch, tmp_path, operation, make_test_dadbot,
):
    pytest.importorskip("mcp.server.mcpserver")
    monkeypatch.setattr(lms, "_project_root", lambda: tmp_path)
    tenant_b = "t1_" + "b" * 26
    store, manager, authority = _authority(tmp_path)
    tools = {"executive_add_task", "executive_list_tasks", "executive_complete_task"}
    servers = {}
    for tenant in (TENANT, tenant_b):
        store.grant_membership(issuer="dadbot", subject="mcp-test-user", tenant_id=tenant, scopes={"read", "write"})
        servers[tenant] = lms.build_server(
            make_test_dadbot(tenant_id=tenant), authority=authority,
        )

    context_var, context_token = _access_context(_mcp_token(manager, tools=tools))
    try:
        added = servers[TENANT]._tool_manager._tools["executive_add_task"].fn("Tenant A private task")
    finally:
        context_var.reset(context_token)

    context_var, context_token = _access_context(_mcp_token(manager, tools=tools, tenant_id=tenant_b))
    try:
        if operation == "list":
            result = servers[tenant_b]._tool_manager._tools["executive_list_tasks"].fn()
            assert result["tasks"] == []
        else:
            result = servers[tenant_b]._tool_manager._tools["executive_complete_task"].fn(added["task"]["id"])
            assert result["completed"] is False
    finally:
        context_var.reset(context_token)

    context_var, context_token = _access_context(_mcp_token(manager, tools=tools))
    try:
        result = servers[TENANT]._tool_manager._tools["executive_list_tasks"].fn()
        assert [task["title"] for task in result["tasks"]] == ["Tenant A private task"]
    finally:
        context_var.reset(context_token)


def test_executive_task_reads_require_explicit_tenant_without_creating_files(monkeypatch, tmp_path):
    monkeypatch.setattr(lms, "_project_root", lambda: tmp_path)
    with pytest.raises((TypeError, ValueError)):
        lms.get_pending_executive_tasks()  # pyright: ignore[reportCallIssue]
    assert lms.get_pending_executive_tasks(tenant_id=TENANT) == []
    assert not (tmp_path / "session_logs").exists()


def test_legacy_unowned_tasks_are_preserved_but_not_adopted(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(lms, "_project_root", lambda: tmp_path)
    legacy = tmp_path / "session_logs" / "executive_tasks.md"
    legacy.parent.mkdir()
    content = "# Dad-Bot Executive Tasks\n\n- [ ] Unassigned {id:legacy, due:, pri:normal}\n"
    legacy.write_text(content, encoding="utf-8")

    assert lms.get_pending_executive_tasks(tenant_id=TENANT) == []
    assert legacy.read_text(encoding="utf-8") == content
    assert list(legacy.parent.iterdir()) == [legacy]
    assert "explicit tenant assignment" in caplog.text


@pytest.mark.parametrize("damage", ["foreign_owner", "missing_owner", "malformed_row", "duplicate_ids", "invalid_utf8"])
def test_executive_boards_fail_explicitly_on_invalid_authority(monkeypatch, tmp_path, damage):
    monkeypatch.setattr(lms, "_project_root", lambda: tmp_path)
    path = lms._task_file_path(TENANT)
    path.parent.mkdir()
    if damage == "invalid_utf8":
        path.write_bytes(b"\xff")
    else:
        header = f"# Dad-Bot Executive Tasks\n<!-- tenant_id:{TENANT} -->\n"
        row = "- [ ] Task {id:duplicate, due:, pri:normal}\n"
        contents = {
            "foreign_owner": f"# Dad-Bot Executive Tasks\n<!-- tenant_id:{'t1_' + 'b' * 26} -->\n",
            "missing_owner": "# Dad-Bot Executive Tasks\n\n",
            "malformed_row": header + "malformed row\n",
            "duplicate_ids": header + row * 2,
        }
        path.write_text(contents[damage], encoding="utf-8")
    before = path.read_bytes()
    error = AuthorizationError if damage in {"foreign_owner", "missing_owner"} else ValueError
    with pytest.raises(error):
        lms.get_pending_executive_tasks(tenant_id=TENANT)
    with pytest.raises(error):
        lms._write_markdown_tasks(path, [], tenant_id=TENANT)
    assert path.read_bytes() == before


def test_failed_executive_publication_preserves_board_and_cleans_temporary_file(monkeypatch, tmp_path):
    monkeypatch.setattr(lms, "_project_root", lambda: tmp_path)
    path = lms._task_file_path(TENANT)
    tasks = [{"id": "first", "title": "Original", "due": "", "priority": "normal", "done": False}]
    lms._write_markdown_tasks(path, tasks, tenant_id=TENANT)
    before = path.read_bytes()

    def reject(_source, _destination):
        raise OSError("Executive publication failed")

    monkeypatch.setattr(lms.os, "replace", reject)
    with pytest.raises(OSError, match="Executive publication failed"):
        lms._write_markdown_tasks(path, [], tenant_id=TENANT)
    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path]


def test_executive_task_fields_cannot_inject_additional_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(lms, "_project_root", lambda: tmp_path)
    path = lms._task_file_path(TENANT)
    lms._write_markdown_tasks(path, [], tenant_id=TENANT)
    before = path.read_bytes()
    task = {
        "id": "second", "title": "First {id:first, due:, pri:normal}\n- [ ] Second",
        "due": "", "priority": "normal", "done": False,
    }
    with pytest.raises(ValueError, match="single-line"):
        lms._write_markdown_tasks(path, [task], tenant_id=TENANT)
    assert path.read_bytes() == before


def test_executive_updates_serialize_the_complete_read_modify_write(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    monkeypatch.setattr(lms, "_project_root", lambda: tmp_path)
    first_entered, release, second_attempted, second_entered = Event(), Event(), Event(), Event()

    def add(task_id, *, pause=False):
        if not pause:
            second_attempted.set()
        with lms._executive_task_update(TENANT) as (path, tasks):
            if pause:
                first_entered.set()
                assert release.wait(5)
            else:
                second_entered.set()
            tasks.append({"id": task_id, "title": task_id, "due": "", "priority": "normal", "done": False})
            lms._write_markdown_tasks(path, tasks, tenant_id=TENANT)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(add, "first", pause=True)
        try:
            assert first_entered.wait(5)
            second = executor.submit(add, "second")
            assert second_attempted.wait(5)
            assert not second_entered.wait(0.2)
        finally:
            release.set()
        first.result(timeout=5)
        second.result(timeout=5)
    assert {task["id"] for task in lms.get_pending_executive_tasks(tenant_id=TENANT)} == {"first", "second"}


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
