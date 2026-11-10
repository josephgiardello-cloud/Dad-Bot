from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from dadbot.core._tool_sandbox import ToolExecutionRecord
from dadbot.memory.working_memory import (
    WorkingMemoryStore,
    new_memory_fact_reference,
    record_successful_tool_step,
)
from dadbot.services import _tool_pipeline_mixin
from dadbot.services.persistence import PersistenceService
from dadbot.services.turn_service import TurnService

pytestmark = pytest.mark.unit


def _runtime(tmp_path: Path, tenant_id: str = "tenant-a") -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            tenant_id=tenant_id,
            session_log_dir=tmp_path,
        ),
    )


def _context(
    *,
    session_id: str = "session-a",
    trace_id: str = "trace-a",
    records: list | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        trace_id=trace_id,
        state={},
        metadata={"control_plane": {"session_id": session_id}},
        _tool_io_ledger=SimpleNamespace(records=records or [object()]),
    )


def test_working_memory_survives_store_reconstruction_and_expires(tmp_path, monkeypatch):
    from dadbot.memory import working_memory

    now = [1000.0]
    monkeypatch.setattr(working_memory.time, "time", lambda: now[0])
    path = tmp_path / "working.sqlite3"
    first_store = WorkingMemoryStore(path)
    first_store.write(
        tenant_id="tenant-a",
        session_id="session-a",
        trace_id="trace-a",
        key="1:tool-call",
        value={"tool_name": "web_search", "result": "A result"},
        ttl_seconds=10,
    )

    reopened_store = WorkingMemoryStore(path)
    assert len(
        reopened_store.list_steps(
            tenant_id="tenant-a",
            session_id="session-a",
            trace_id="trace-a",
        )
    ) == 1

    now[0] += 11
    assert reopened_store.list_steps(
        tenant_id="tenant-a",
        session_id="session-a",
        trace_id="trace-a",
    ) == []


def test_working_memory_isolated_by_tenant_session_and_trace(tmp_path):
    store = WorkingMemoryStore(tmp_path / "working.sqlite3")
    for tenant_id, session_id, trace_id in (
        ("tenant-a", "session-a", "trace-a"),
        ("tenant-b", "session-a", "trace-a"),
        ("tenant-a", "session-b", "trace-a"),
        ("tenant-a", "session-a", "trace-b"),
    ):
        store.write(
            tenant_id=tenant_id,
            session_id=session_id,
            trace_id=trace_id,
            key="1:step",
            value={"tenant": tenant_id},
        )

    assert store.list_steps(
        tenant_id="tenant-a",
        session_id="session-a",
        trace_id="trace-a",
    )[0]["payload"]["tenant"] == "tenant-a"
    assert len(
        store.list_steps(
            tenant_id="tenant-b",
            session_id="session-a",
            trace_id="trace-a",
        )
    ) == 1


def test_successful_tool_record_is_scrubbed_and_failed_denied_or_replayed_steps_are_skipped(tmp_path):
    runtime = _runtime(tmp_path)
    context = _context()
    result = ToolExecutionRecord(
        tool_name="web_search",
        idempotency_key="call-1",
        status="succeeded",
        result={"email": "person@example.com", "summary": "Found a page"},
        error="",
    )

    assert record_successful_tool_step(
        runtime,
        context,
        tool_name="web_search",
        parameters={"query": "example"},
        record=result,
    )
    store = WorkingMemoryStore(tmp_path / "working_memory.sqlite3")
    steps = store.list_steps(
        tenant_id="tenant-a",
        session_id="session-a",
        trace_id="trace-a",
    )
    assert len(steps) == 1
    assert steps[0]["payload"]["result"]["email"] == "[EMAIL REDACTED]"

    for status, value in (
        ("failed", {"status": "success"}),
        ("cached", {"status": "success"}),
        ("replayed", {"status": "success"}),
        ("succeeded", {"status": "denied"}),
        ("succeeded", {"status": "contract_violation"}),
        ("succeeded", {"authorized": False}),
    ):
        rejected = ToolExecutionRecord(
            tool_name="set_reminder",
            idempotency_key=f"call-{status}",
            status=status,
            result=value,
            error="",
        )
        assert not record_successful_tool_step(
            runtime,
            context,
            tool_name="set_reminder",
            parameters={},
            record=rejected,
        )
    assert len(
        store.list_steps(
            tenant_id="tenant-a",
            session_id="session-a",
            trace_id="trace-a",
        )
    ) == 1


def test_working_memory_is_promoted_only_after_save_commit(tmp_path):
    runtime = _runtime(tmp_path)
    context = _context()
    fact_ref = new_memory_fact_reference()
    context.state["_validated_memory_fact_refs"] = [fact_ref]
    context.state["_save_transaction"] = {
        "commit_id": "commit-1",
        "status": "active",
    }
    context.temporal = SimpleNamespace(wall_time=datetime.now().isoformat())
    store = WorkingMemoryStore(tmp_path / "working_memory.sqlite3")
    store.write(
        tenant_id="tenant-a",
        session_id="session-a",
        trace_id="trace-a",
        key="1:tool-call",
        value={"tool_name": "web_search", "result": "temporary"},
    )
    row = store.list_steps(
        tenant_id="tenant-a",
        session_id="session-a",
        trace_id="trace-a",
    )[0]
    assert row["promoted_at"] is None

    service = PersistenceService(object(), turn_service=SimpleNamespace(bot=runtime))
    service._persist_hierarchical_memory_commit = lambda *_args, **_kwargs: None
    service._record_merkle_anchor = lambda *_args, **_kwargs: None
    service.commit_transaction(context)

    promoted = store.list_steps(
        tenant_id="tenant-a",
        session_id="session-a",
        trace_id="trace-a",
    )[0]
    assert promoted["promoted_at"] is not None
    assert promoted["promoted_fact_refs"] == [fact_ref]
    assert context.state["working_memory_promoted_steps"] == 1


def test_sync_and_async_tool_paths_write_successful_steps(bot, monkeypatch):
    service = TurnService(bot)
    monkeypatch.setattr(service, "_reminder_confirmation_reply", lambda _result: "Reminder saved.")
    monkeypatch.setattr(
        service,
        "_reflect_on_web_observation",
        lambda _input, _query, observation, _settings: observation,
    )

    async def reflect_async(_input, _query, observation, _settings):
        return observation

    monkeypatch.setattr(service, "_reflect_on_web_observation_async", reflect_async)
    executions = []

    def fake_execute_tool(*, tool_name, **_kwargs):
        result = (
            {"id": "reminder-1", "title": "Buy apples"}
            if tool_name == "set_reminder"
            else {
                "heading": "Search result",
                "summary": "A useful result",
                "source_label": "Example",
            }
        )
        record = ToolExecutionRecord(
            tool_name=tool_name,
            idempotency_key=f"call-{len(executions) + 1}",
            status="succeeded",
            result=result,
            error="",
        )
        executions.append(record)
        return record

    monkeypatch.setattr(_tool_pipeline_mixin, "execute_tool", fake_execute_tool)

    def context(index: int):
        return _context(trace_id=f"trace-{index}")

    reminder_params = {"title": "Buy apples", "due_text": "tomorrow"}
    service._execute_set_reminder_tool_sync(
        params=reminder_params,
        stripped_input="Buy apples",
        current_mood="neutral",
        plan_reason="explicit request",
        turn_context=context(1),
    )
    asyncio.run(
        service._execute_set_reminder_tool_async(
            params=reminder_params,
            stripped_input="Buy apples",
            current_mood="neutral",
            plan_reason="explicit request",
            turn_context=context(2),
        )
    )
    web_params = {"query": "weather"}
    service._execute_web_search_tool_sync(
        params=web_params,
        stripped_input="weather",
        settings={},
        plan_reason="explicit request",
        turn_context=context(3),
    )
    asyncio.run(
        service._execute_web_search_tool_async(
            params=web_params,
            stripped_input="weather",
            settings={},
            plan_reason="explicit request",
            turn_context=context(4),
        )
    )

    store = WorkingMemoryStore(bot.config.session_log_dir / "working_memory.sqlite3")
    for index, tool_name in enumerate(
        ("set_reminder", "set_reminder", "web_search", "web_search"),
        start=1,
    ):
        steps = store.list_steps(
            tenant_id=bot.config.tenant_id,
            session_id="session-a",
            trace_id=f"trace-{index}",
        )
        assert len(steps) == 1
        assert steps[0]["payload"]["tool_name"] == tool_name
