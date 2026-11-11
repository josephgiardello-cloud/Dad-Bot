from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_backend import FileWALLedgerBackend
from dadbot.core.state_lineage import canonical_state_hash
from dadbot.core.tool_approval_journal import ApprovedToolCall, ToolApprovalJournal

pytestmark = pytest.mark.unit


def _call(**changes):
    return replace(ApprovedToolCall(
        session_id="thread-a", trace_id="trace-a", request_id="approval-a", tool_name="light_control",
        input_hash=canonical_state_hash({"tool_name": "light_control", "arguments": {"entity_id": "light.kitchen"}}),
    ), **changes)


def _execute(journal, call, executor, **changes):
    return journal.execute(
        call, tool_name="light_control", arguments={"entity_id": "light.kitchen"}, executor=executor, **changes,
    )


def _ledger(path):
    ledger = ExecutionLedger()
    ledger._backend = FileWALLedgerBackend(path)
    ledger.load_from_backend()
    return ledger


def _uncertain_call(journal, call=None):
    call = call or _call()
    journal.propose(call)
    journal.decide(call, approved=True)
    asyncio.run(_execute(journal, call, AsyncMock(return_value={"is_error": True, "result": "unknown"})))
    return call


def _review_context(call, *, tenant_id="tenant-a"):
    from dadbot.core.graph_context import TurnContext

    return TurnContext(
        user_input="I verified that the Home Assistant action happened.",
        trace_id="review-trace",
        metadata={"tenant_id": tenant_id, "session_id": call.session_id, "request_id": "review-request"},
    )


@pytest.mark.parametrize("outcome", ["performed", "not_performed"])
def test_manual_review_is_durable_but_does_not_certify_success_or_authorize_retry(tmp_path, outcome):
    path = tmp_path / "review.jsonl"
    journal = ToolApprovalJournal(_ledger(path), tenant_id="tenant-a")
    call = _uncertain_call(journal)
    action = journal.actions("thread-a")[0]
    assert action.review_outcome == ""
    assert action.retry_policy == "manual_review_required"
    event = journal.review_action(
        call, outcome_event_id=action.outcome_event_id, outcome=outcome, turn_context=_review_context(call),
    )
    assert event["type"] == "TOOL_ACTION_REVIEWED" and event["committed"] is True
    assert event["payload"]["review_source"] == "user_confirmation"
    assert "arguments" not in event["payload"] and "result" not in event["payload"]
    assert event["trace_id"] == "review-trace"
    restored = ToolApprovalJournal(_ledger(path), tenant_id="tenant-a")
    restored.expire_recovered()
    action = restored.actions("thread-a")[0]
    assert action.status == "uncertain"
    assert action.review_outcome == outcome
    assert action.retry_policy == ("do_not_repeat" if outcome == "performed" else "new_proposal_required")
    with pytest.raises(RuntimeError, match="successful"):
        asyncio.run(restored.replay(call, arguments={"entity_id": "light.kitchen"}))
    executor = AsyncMock(return_value={"is_error": False})
    with pytest.raises(RuntimeError, match="not permitted"):
        asyncio.run(_execute(restored, call, executor))
    executor.assert_not_called()


def test_manual_review_repeats_are_idempotent_and_conflicts_fail():
    journal = ToolApprovalJournal(ExecutionLedger(), tenant_id="tenant-a")
    call = _uncertain_call(journal)
    event_id = journal.actions("thread-a")[0].outcome_event_id
    first = journal.review_action(
        call, outcome_event_id=event_id, outcome="performed", turn_context=_review_context(call),
    )
    count = len(journal.ledger.read())
    assert journal.review_action(
        call, outcome_event_id=event_id, outcome="performed", turn_context=_review_context(call),
    ) == first
    assert len(journal.ledger.read()) == count
    with pytest.raises(RuntimeError, match="different review"):
        journal.review_action(
            call, outcome_event_id=event_id, outcome="not_performed", turn_context=_review_context(call),
        )
    assert len(journal.ledger.read()) == count


def test_manual_review_stale_identity_and_wrong_tenant_fail():
    journal = ToolApprovalJournal(ExecutionLedger(), tenant_id="tenant-a")
    call = _uncertain_call(journal)
    event_id = journal.actions("thread-a")[0].outcome_event_id
    with pytest.raises(ValueError, match="outcome event"):
        journal.review_action(
            call, outcome_event_id="stale-event", outcome="performed", turn_context=_review_context(call),
        )
    with pytest.raises(ValueError, match="tenant/thread"):
        journal.review_action(
            call, outcome_event_id=event_id, outcome="performed",
            turn_context=_review_context(call, tenant_id="tenant-b"),
        )
    with pytest.raises(ValueError, match="identity"):
        journal.review_action(
            replace(call, trace_id="another-trace"), outcome_event_id=event_id, outcome="performed",
            turn_context=_review_context(call),
        )
    assert journal.actions("thread-a")[0].review_outcome == ""


@pytest.mark.parametrize("status", ["pending", "approved", "started", "rejected", "expired", "succeeded"])
def test_only_uncertain_outcomes_can_be_manually_reviewed(status):
    journal = ToolApprovalJournal(ExecutionLedger(), tenant_id="tenant-a")
    call = _call()
    journal.propose(call)
    if status == "rejected":
        journal.decide(call, approved=False)
    elif status == "expired":
        journal.expire(call)
    elif status != "pending":
        journal.decide(call, approved=True)
        if status == "started":
            journal._transition(call, "started", expected={"approved"})
        elif status == "succeeded":
            asyncio.run(_execute(journal, call, AsyncMock(return_value={"is_error": False})))
    action = journal.actions("thread-a")[0]
    with pytest.raises(RuntimeError, match="uncertain"):
        journal.review_action(
            call, outcome_event_id=action.outcome_event_id, outcome="performed", turn_context=_review_context(call),
        )


def test_action_review_queue_is_tenant_and_thread_scoped():
    ledger = ExecutionLedger()
    journal = ToolApprovalJournal(ledger, tenant_id="tenant-a")
    _uncertain_call(journal)
    other = _call(session_id="thread-b", request_id="approval-b")
    _uncertain_call(journal, other)
    another_tenant = ToolApprovalJournal(ledger, tenant_id="tenant-b")
    _uncertain_call(another_tenant, _call(tool_name="light_control"))
    assert [action.call.session_id for action in journal.actions("thread-a")] == ["thread-a"]
    assert len(another_tenant.actions("thread-a")) == 1
    from dadbot.core.tenant_session import tenant_session_scope

    with tenant_session_scope("tenant-a", "thread-a"):
        with pytest.raises(ValueError, match="active session"):
            journal.actions("thread-b")
        with pytest.raises(ValueError, match="active session"):
            another_tenant.actions("thread-a")


def test_review_fsync_failure_blocks_reads_until_explicit_backend_recovery(tmp_path, monkeypatch):
    from dadbot.core import ledger_backend

    journal = ToolApprovalJournal(_ledger(tmp_path / "review.jsonl"), tenant_id="tenant-a")
    call = _uncertain_call(journal)
    event_id = journal.actions("thread-a")[0].outcome_event_id
    monkeypatch.setattr(ledger_backend.os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("review fsync failed")))
    with pytest.raises(OSError, match="review fsync"):
        journal.review_action(
            call, outcome_event_id=event_id, outcome="performed", turn_context=_review_context(call),
        )
    with pytest.raises(RuntimeError, match="backend state is uncertain"):
        journal.actions("thread-a")
    monkeypatch.undo()
    restored = ToolApprovalJournal(_ledger(tmp_path / "review.jsonl"), tenant_id="tenant-a")
    restored.expire_recovered()
    assert restored.status(call) == "uncertain"
    with pytest.raises(RuntimeError, match="successful"):
        asyncio.run(restored.replay(call, arguments={"entity_id": "light.kitchen"}))


def test_review_write_failure_does_not_remove_item_from_queue(monkeypatch):
    journal = ToolApprovalJournal(ExecutionLedger(), tenant_id="tenant-a")
    call = _uncertain_call(journal)
    event_id = journal.actions("thread-a")[0].outcome_event_id
    def unavailable(*_args, **_kwargs):
        raise OSError("review store unavailable")
    monkeypatch.setattr(journal.writer, "write_event", unavailable)
    with pytest.raises(OSError, match="review store unavailable"):
        journal.review_action(
            call, outcome_event_id=event_id, outcome="performed", turn_context=_review_context(call),
        )
    assert journal.actions("thread-a")[0].review_outcome == ""


def test_action_review_revoked_authority_cannot_commit(monkeypatch):
    from dadbot.core import tool_approval_journal

    journal = ToolApprovalJournal(ExecutionLedger(), tenant_id="tenant-a")
    call = _uncertain_call(journal)
    event_id = journal.actions("thread-a")[0].outcome_event_id
    def revoked():
        raise PermissionError("revoked")
    monkeypatch.setattr(tool_approval_journal, "verify_live_authority", revoked)
    with pytest.raises(PermissionError, match="revoked"):
        journal.review_action(
            call, outcome_event_id=event_id, outcome="performed", turn_context=_review_context(call),
        )
    assert journal.actions("thread-a")[0].review_outcome == ""


@pytest.mark.parametrize("field,value", [
    ("review_source", "provider_verified"),
    ("outcome_event_id", "another-outcome"),
    ("input_hash", "another-input"),
    ("review_scope", {"request_id": 42}),
])
def test_recorded_action_review_corruption_is_not_projected_as_success(field, value):
    journal = ToolApprovalJournal(ExecutionLedger(), tenant_id="tenant-a")
    call = _uncertain_call(journal)
    action = journal.actions("thread-a")[0]
    payload = {
        "request_id": call.request_id, "trace_id": call.trace_id, "tool_name": call.tool_name,
        "input_hash": call.input_hash, "outcome_event_id": action.outcome_event_id,
        "outcome": "performed", "review_source": "user_confirmation",
        "review_scope": {
            "tenant_id": "tenant-a", "session_id": "thread-a", "trace_id": "review-trace",
            "request_id": "review-request",
        },
        field: value,
    }
    journal.writer.write_event(
        "TOOL_ACTION_REVIEWED", tenant_id="tenant-a", session_id="thread-a", trace_id="review-trace",
        kernel_step_id="tools.action_review", committed=True, payload=payload,
    )
    with pytest.raises(RuntimeError, match="Recorded action review"):
        journal.actions("thread-a")
    assert journal.status(call) == "uncertain"


def test_committed_manual_review_survives_abrupt_process_exit(tmp_path):
    path = tmp_path / "crash-review.jsonl"
    script = """
import os, sys, asyncio
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_backend import FileWALLedgerBackend
from dadbot.core.graph_context import TurnContext
from dadbot.core.tool_approval_journal import ToolApprovalJournal, ApprovedToolCall
from dadbot.core.state_lineage import canonical_state_hash
ledger = ExecutionLedger()
ledger._backend = FileWALLedgerBackend(sys.argv[1])
ledger.load_from_backend()
journal = ToolApprovalJournal(ledger, tenant_id="tenant-a")
call = ApprovedToolCall(
    session_id="thread-a", trace_id="trace-a", request_id="approval-a", tool_name="light_control",
    input_hash=canonical_state_hash({"tool_name":"light_control","arguments":{"entity_id":"light.kitchen"}}),
)
journal.propose(call)
journal.decide(call, approved=True)
async def remote():
    return {"is_error": True}
asyncio.run(journal.execute(
    call, tool_name="light_control", arguments={"entity_id":"light.kitchen"}, executor=remote,
))
context = TurnContext(
    user_input="", trace_id="review-trace",
    metadata={"tenant_id":"tenant-a","session_id":"thread-a","request_id":"review-request"},
)
journal.review_action(
    call, outcome_event_id=journal.actions("thread-a")[0].outcome_event_id,
    outcome="not_performed", turn_context=context,
)
os._exit(31)
"""
    result = subprocess.run([sys.executable, "-c", script, str(path)], check=False, timeout=20)
    assert result.returncode == 31
    journal = ToolApprovalJournal(_ledger(path), tenant_id="tenant-a")
    journal.expire_recovered()
    assert journal.actions("thread-a")[0].review_outcome == "not_performed"
    assert journal.actions("thread-a")[0].status == "uncertain"
    assert journal.actions("thread-a")[0].retry_policy == "new_proposal_required"
    assert sum(event["type"] == "TOOL_ACTION_REVIEWED" for event in journal.ledger.read()) == 1


def test_approval_audit_is_scoped_and_does_not_store_arguments():
    ledger = ExecutionLedger()
    journal = ToolApprovalJournal(ledger, tenant_id="tenant-a")
    call = _call()
    journal.propose(call)
    event = ledger.read()[-1]
    assert event["tenant_id"] == "tenant-a"
    assert event["session_id"] == call.session_id
    assert event["trace_id"] == call.trace_id
    assert event["payload"]["input_hash"] == call.input_hash
    assert "arguments" not in event["payload"]
    assert event["committed"] is True


def test_approved_execution_commits_dispatch_before_effect_and_never_runs_twice():
    ledger = ExecutionLedger()
    journal = ToolApprovalJournal(ledger, tenant_id="tenant-a")
    call = _call()
    journal.propose(call)
    journal.decide(call, approved=True)

    async def effect():
        assert ledger.read()[-1]["payload"]["status"] == "started"
        return {"is_error": False, "content": "completed"}

    execute = AsyncMock(side_effect=effect)
    assert asyncio.run(_execute(journal, call, execute))["content"] == "completed"
    assert journal.status(call) == "succeeded"
    with pytest.raises(RuntimeError, match="not permitted"):
        asyncio.run(_execute(journal, call, execute))
    execute.assert_awaited_once()


@pytest.mark.parametrize("approved", [False, True])
def test_restart_expires_pending_or_approved_proposals(tmp_path, approved):
    path = tmp_path / "approvals.jsonl"
    first = ToolApprovalJournal(_ledger(path), tenant_id="tenant-a")
    call = _call()
    first.propose(call)
    if approved:
        first.decide(call, approved=True)
    restored = ToolApprovalJournal(_ledger(path), tenant_id="tenant-a")
    restored.expire_recovered()
    assert restored.status(call) == "expired"
    executor = AsyncMock(return_value={})
    with pytest.raises(RuntimeError, match="not permitted"):
        asyncio.run(_execute(restored, call, executor))
    executor.assert_not_called()


def test_restart_marks_started_call_uncertain_without_retry(tmp_path):
    path = tmp_path / "approvals.jsonl"
    first = ToolApprovalJournal(_ledger(path), tenant_id="tenant-a")
    call = _call()
    first.propose(call)
    first.decide(call, approved=True)
    first._transition(call, "started", expected={"approved"})
    restored = ToolApprovalJournal(_ledger(path), tenant_id="tenant-a")
    restored.expire_recovered()
    assert restored.status(call) == "uncertain"
    executor = AsyncMock(return_value={})
    with pytest.raises(RuntimeError, match="not permitted"):
        asyncio.run(_execute(restored, call, executor))
    executor.assert_not_called()


@pytest.mark.parametrize("exception", [TimeoutError("remote timeout"), asyncio.CancelledError()])
def test_failure_or_cancellation_records_uncertainty(exception):
    journal = ToolApprovalJournal(ExecutionLedger(), tenant_id="tenant-a")
    call = _call()
    journal.propose(call)
    journal.decide(call, approved=True)
    executor = AsyncMock(side_effect=exception)
    with pytest.raises(type(exception)):
        asyncio.run(_execute(journal, call, executor))
    assert journal.status(call) == "uncertain"
    with pytest.raises(RuntimeError, match="not permitted"):
        asyncio.run(_execute(journal, call, executor))
    executor.assert_awaited_once()


def test_denied_or_changed_approval_cannot_execute():
    journal = ToolApprovalJournal(ExecutionLedger(), tenant_id="tenant-a")
    call = _call()
    journal.propose(call)
    journal.decide(call, approved=False)
    executor = AsyncMock(return_value={})
    with pytest.raises(RuntimeError, match="not permitted"):
        asyncio.run(_execute(journal, call, executor))
    with pytest.raises(ValueError, match="arguments changed"):
        asyncio.run(journal.execute(
            call, tool_name="light_control", arguments={"entity_id": "light.other"}, executor=executor,
        ))
    executor.assert_not_called()


def test_failed_dispatch_record_blocks_effect(monkeypatch):
    journal = ToolApprovalJournal(ExecutionLedger(), tenant_id="tenant-a")
    call = _call()
    journal.propose(call)
    journal.decide(call, approved=True)

    def unavailable(*_args, **_kwargs):
        raise OSError("execution WAL unavailable")

    monkeypatch.setattr(journal.writer, "write_event", unavailable)
    executor = AsyncMock(return_value={})
    with pytest.raises(OSError, match="WAL unavailable"):
        asyncio.run(_execute(journal, call, executor))
    executor.assert_not_called()
    assert journal.status(call) == "approved"


def test_failed_success_record_keeps_started_outcome_uncertain_on_recovery(monkeypatch):
    journal = ToolApprovalJournal(ExecutionLedger(), tenant_id="tenant-a")
    call = _call()
    journal.propose(call)
    journal.decide(call, approved=True)
    write = journal.writer.write_event

    def fail_success(*args, **kwargs):
        if kwargs["payload"].get("status") == "succeeded":
            raise OSError("completion WAL unavailable")
        return write(*args, **kwargs)

    monkeypatch.setattr(journal.writer, "write_event", fail_success)
    executor = AsyncMock(return_value={})
    with pytest.raises(OSError, match="completion WAL unavailable"):
        asyncio.run(_execute(journal, call, executor))
    assert journal.status(call) == "started"
    journal.expire_recovered()
    assert journal.status(call) == "uncertain"
    executor.assert_awaited_once()


def test_approval_identity_is_tenant_and_session_scoped():
    ledger = ExecutionLedger()
    first = ToolApprovalJournal(ledger, tenant_id="tenant-a")
    second = ToolApprovalJournal(ledger, tenant_id="tenant-b")
    call = _call()
    other = _call(session_id="thread-b")
    first.propose(call)
    first.propose(other)
    second.propose(call)
    first.decide(call, approved=True)
    assert first.status(other) == second.status(call) == "pending"
    with pytest.raises(ValueError, match="identity"):
        first.decide(_call(tool_name="other"), approved=True)


@pytest.mark.parametrize("field", ["trace_id", "tool_name", "input_hash"])
def test_approval_status_rejects_changed_identity(field):
    journal = ToolApprovalJournal(ExecutionLedger(), tenant_id="tenant-a")
    journal.propose(_call())
    with pytest.raises(ValueError, match="identity"):
        journal.status(_call(**{field: "changed"}))


def test_failed_recovery_is_not_cached_by_service(monkeypatch):
    from dadbot.runtime.mcp.approval_service import MCPApprovalService

    bot = SimpleNamespace(config=SimpleNamespace(tenant_id="tenant-a"))
    service = MCPApprovalService(bot)
    bot._execution_ledger = ExecutionLedger()
    recovery = ToolApprovalJournal.expire_recovered
    attempts = []

    def recover(journal):
        attempts.append(journal)
        if len(attempts) == 1:
            raise OSError("recovery unavailable")
        recovery(journal)

    monkeypatch.setattr(ToolApprovalJournal, "expire_recovered", recover)
    with pytest.raises(OSError, match="recovery unavailable"):
        service._approval_journal()
    assert service._journal is None
    assert service._approval_journal() is attempts[1]


def test_real_wal_fsync_failure_blocks_dispatch_and_recovery_never_retries(tmp_path, monkeypatch):
    from dadbot.core import ledger_backend

    path = tmp_path / "approvals.jsonl"
    journal = ToolApprovalJournal(_ledger(path), tenant_id="tenant-a")
    call = _call()
    journal.propose(call)
    journal.decide(call, approved=True)
    executor = AsyncMock(return_value={})

    def unavailable(_descriptor):
        raise OSError("fsync unavailable")

    with monkeypatch.context() as failing:
        failing.setattr(ledger_backend.os, "fsync", unavailable)
        with pytest.raises(OSError, match="fsync unavailable"):
            asyncio.run(_execute(journal, call, executor))
    executor.assert_not_called()
    restored = ToolApprovalJournal(_ledger(path), tenant_id="tenant-a")
    restored.expire_recovered()
    assert restored.status(call) == "uncertain"
    with pytest.raises(RuntimeError, match="not permitted"):
        asyncio.run(_execute(restored, call, executor))
    executor.assert_not_called()


@pytest.mark.parametrize("phase", ["pending", "started", "io_recorded"])
def test_process_exit_recovers_without_dispatching(tmp_path, phase):
    path = tmp_path / "approvals.jsonl"
    script = """
import asyncio, os, sys
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_backend import FileWALLedgerBackend
from dadbot.core.state_lineage import canonical_state_hash
from dadbot.core.tool_approval_journal import ApprovedToolCall, ToolApprovalJournal
ledger = ExecutionLedger()
ledger._backend = FileWALLedgerBackend(sys.argv[1])
ledger.load_from_backend()
journal = ToolApprovalJournal(ledger, tenant_id="tenant-a")
arguments = {"entity_id": "light.kitchen"}
call = ApprovedToolCall("thread-a", "trace-a", "approval-a", "light_control",
    canonical_state_hash({"tool_name": "light_control", "arguments": arguments}))
journal.propose(call)
if sys.argv[2] != "pending":
    journal.decide(call, approved=True)
    if sys.argv[2] == "io_recorded":
        write = journal.writer.write_event
        def exit_after_io(event_type, **kwargs):
            event = write(event_type, **kwargs)
            if event_type == "TOOL_IO_RECORDED":
                os._exit(29)
            return event
        journal.writer.write_event = exit_after_io
    async def effect():
        if sys.argv[2] == "started":
            os._exit(29)
        return {"result": "done"}
    asyncio.run(journal.execute(call, tool_name="light_control", arguments=arguments, executor=effect))
os._exit(29)
"""
    child = subprocess.run([sys.executable, "-c", script, str(path), phase], check=False, timeout=20)
    assert child.returncode == 29
    restored = ToolApprovalJournal(_ledger(path), tenant_id="tenant-a")
    restored.expire_recovered()
    assert restored.status(_call()) == ("expired" if phase == "pending" else "uncertain")
    executor = AsyncMock(return_value={})
    with pytest.raises(RuntimeError, match="not permitted"):
        asyncio.run(_execute(restored, _call(), executor))
    executor.assert_not_called()
    with pytest.raises(RuntimeError, match="successful"):
        asyncio.run(restored.replay(_call(), arguments={"entity_id": "light.kitchen"}))
    if phase == "io_recorded":
        assert any(event["type"] == "TOOL_IO_RECORDED" for event in restored.ledger.read())


def test_live_authority_is_rechecked_before_dispatch(monkeypatch):
    from dadbot.core import tool_approval_journal

    journal = ToolApprovalJournal(ExecutionLedger(), tenant_id="tenant-a")
    call = _call()
    journal.propose(call)
    journal.decide(call, approved=True)

    def revoked():
        raise PermissionError("authority revoked")

    monkeypatch.setattr(tool_approval_journal, "verify_live_authority", revoked)
    executor = AsyncMock(return_value={})
    with pytest.raises(PermissionError, match="revoked"):
        asyncio.run(_execute(journal, call, executor))
    assert journal.status(call) == "approved"
    executor.assert_not_called()


def _mcp_service(monkeypatch, *, result=None, follow_up=False, ledger=None):
    from dadbot.runtime.mcp import approval_service
    from dadbot.runtime.mcp.tool_client import MCPApprovalRequest, run_mcp_tool_loop

    tools = [{
        "name": "light_control",
        "input_schema": {
            "type": "object", "properties": {"entity_id": {"type": "string"}},
            "required": ["entity_id"], "additionalProperties": False,
        },
    }]
    plan = json.dumps({
        "should_continue": True, "action_input": "Review this action", "reasoning": "Explicit request",
        "tool_calls": [{"name": "light_control", "arguments": {"entity_id": "light.kitchen"}}],
        "confidence": 1.0,
    })
    request = asyncio.run(run_mcp_tool_loop(
        [{"role": "user", "content": "Turn off light.kitchen"}],
        tools, model_call=lambda *_args: plan,
    ))
    assert isinstance(request, MCPApprovalRequest)
    ledger = ledger if ledger is not None else ExecutionLedger()
    bot = SimpleNamespace(
        _execution_ledger=ledger, config=SimpleNamespace(tenant_id="tenant-a"),
        active_thread_id="thread-a",
    )
    service = approval_service.MCPApprovalService(bot)
    bot.decision_contexts = []

    async def execute_turn(turn_request):
        from dadbot.core.execution_contract import TurnResponse
        from dadbot.core.graph_context import TurnContext
        from dadbot.core.graph_pipeline_nodes import MCPToolNode

        context = TurnContext(
            user_input=turn_request.input.text, trace_id=f"decision-{turn_request.context.request_id}",
            metadata={
                **turn_request.metadata, "tenant_id": turn_request.context.tenant_id,
                "session_id": turn_request.session_id, "request_id": turn_request.context.request_id,
            },
        )
        bot.decision_contexts.append(context)
        registry = SimpleNamespace(get=lambda *_args, **_kwargs: service)
        await MCPToolNode().execute(registry, context)
        return TurnResponse(reply=context.state["candidate"][0])

    bot.execute_turn = execute_turn
    connections = []
    actions = []

    class Client:
        @asynccontextmanager
        async def connect(self):
            connections.append(True)
            yield object()

        async def list_allowed_tools(self, _session, allowed_names):
            return [tool for tool in tools if tool["name"] in allowed_names]

        async def call_tool(self, _session, name, arguments):
            assert ledger.read()[-1]["payload"]["status"] == "started"
            actions.append((name, arguments))
            if isinstance(result, Exception):
                raise result
            return result if result is not None else {"is_error": False, "content": "completed"}

    monkeypatch.setenv("DADBOT_HA_MCP_URL", "http://127.0.0.1:9999/mcp")
    monkeypatch.setenv("DADBOT_HA_MCP_ALLOWED_TOOLS", "light_control")
    monkeypatch.setattr(approval_service, "MCPToolClient", lambda *_args, **_kwargs: Client())
    summary = json.dumps({
        "should_continue": False, "action_input": "The approved action completed.",
        "reasoning": "Confirmed result", "tool_calls": [], "confidence": 1.0,
    })
    model_outputs = iter([plan, summary] if follow_up else [summary])
    monkeypatch.setattr(service, "_model_call", lambda *_args, **_kwargs: next(model_outputs))
    service._queue_approval(request, [], "http://127.0.0.1:9999/mcp", thread_id="thread-a", trace_id="trace-a")
    return service, ledger, request, actions, connections


def test_mcp_approved_call_uses_durable_journal_and_is_not_repeated(monkeypatch):
    service, ledger, _request, actions, _connections = _mcp_service(monkeypatch)
    assert "completed" in asyncio.run(service.resolve_mcp_approval("thread-a", approved=True))
    assert len(service.bot.decision_contexts) == 1
    context = service.bot.decision_contexts[0]
    assert context.user_input == "Approve this Home Assistant action."
    assert context.metadata["mcp_approval_resolution"] == {"request_id": _request.request_id, "approved": True}
    record = context._tool_io_ledger.records[0]
    assert record.metadata["scope"]["trace_id"] == context.trace_id
    assert record.metadata["scope"]["request_id"] == context.metadata["request_id"]
    assert [event["payload"]["status"] for event in ledger.read() if event["type"] == "TOOL_APPROVAL_UPDATED"] == [
        "pending", "approved", "started", "succeeded",
    ]
    assert all(event["session_id"] == "thread-a" and event["tenant_id"] == "tenant-a" for event in ledger.read())
    assert "No additional action" in asyncio.run(service.resolve_mcp_approval("thread-a", approved=True))
    assert len(actions) == 1


def test_mcp_rejection_is_audited_without_connecting(monkeypatch):
    service, ledger, _request, actions, connections = _mcp_service(monkeypatch)
    assert "not approved" in asyncio.run(service.resolve_mcp_approval("thread-a", approved=False))
    assert ledger.read()[-1]["payload"]["status"] == "rejected"
    assert not actions and not connections


def test_mcp_restart_expires_proposal_without_restoring_action(monkeypatch):
    from dadbot.runtime.mcp.approval_service import MCPApprovalService

    service, ledger, _request, actions, connections = _mcp_service(monkeypatch)
    restored = MCPApprovalService(service.bot)
    assert restored.get_pending_mcp_approval("thread-a") is None
    assert ledger.read()[-1]["payload"]["status"] == "expired"
    asyncio.run(restored.resolve_mcp_approval("thread-a", approved=True))
    assert not actions and not connections


def test_mcp_uncertain_outcome_is_audited_and_not_retried(monkeypatch):
    service, ledger, _request, actions, _connections = _mcp_service(monkeypatch, result=TimeoutError("remote timeout"))
    reply = asyncio.run(service.resolve_mcp_approval("thread-a", approved=True))
    assert "uncertain" in reply
    assert ledger.read()[-1]["payload"]["status"] == "uncertain"
    asyncio.run(service.resolve_mcp_approval("thread-a", approved=True))
    assert len(actions) == 1


@pytest.mark.parametrize("outcome", ["performed", "not_performed"])
def test_mcp_manual_review_is_a_canonical_no_dispatch_continuation(monkeypatch, outcome):
    service, _ledger, request, actions, connections = _mcp_service(monkeypatch, result=TimeoutError("remote timeout"))
    asyncio.run(service.resolve_mcp_approval("thread-a", approved=True))
    action = service.get_mcp_action_history("thread-a")[0]
    count = len(connections)
    reply = asyncio.run(service.review_mcp_action(
        "thread-a", request_id=request.request_id, outcome_event_id=action.outcome_event_id, outcome=outcome,
    ))
    assert "your confirmation" in reply and "not" in reply
    assert len(service.bot.decision_contexts) == 2
    context = service.bot.decision_contexts[-1]
    assert context.state["decision_outcome"] == "no_tool_needed"
    assert context._tool_io_ledger is None
    assert context.metadata["mcp_approval_resolution"] == {
        "request_id": request.request_id, "outcome_event_id": action.outcome_event_id, "review_outcome": outcome,
    }
    assert context.user_input == service.review_text(outcome=outcome, request_id=request.request_id)
    assert len(actions) == 1 and len(connections) == count
    assert service.get_pending_mcp_approval("thread-a") is None
    restored = type(service)(service.bot)
    assert restored.get_mcp_action_history("thread-a")[0].review_outcome == outcome
    assert restored.get_mcp_action_history("thread-a")[0].status == "uncertain"
    assert not restored.get_pending_mcp_approval("thread-a")


def test_mcp_manual_review_stale_selection_is_rejected_before_graph_submission(monkeypatch):
    service, _ledger, request, actions, connections = _mcp_service(monkeypatch, result=TimeoutError("remote timeout"))
    asyncio.run(service.resolve_mcp_approval("thread-a", approved=True))
    with pytest.raises(ValueError, match="no longer matches"):
        asyncio.run(service.review_mcp_action(
            "thread-a", request_id=request.request_id, outcome_event_id="stale-event", outcome="performed",
        ))
    assert len(actions) == len(connections) == len(service.bot.decision_contexts) == 1


def test_mcp_manual_review_commit_failure_propagates_without_success_reply(monkeypatch):
    service, _ledger, request, actions, connections = _mcp_service(monkeypatch, result=TimeoutError("remote timeout"))
    asyncio.run(service.resolve_mcp_approval("thread-a", approved=True))
    action = service.get_mcp_action_history("thread-a")[0]
    def unavailable(*_args, **_kwargs):
        raise OSError("action review unavailable")
    monkeypatch.setattr(service._approval_journal().writer, "write_event", unavailable)
    with pytest.raises(OSError, match="action review unavailable"):
        asyncio.run(service.review_mcp_action(
            "thread-a", request_id=request.request_id, outcome_event_id=action.outcome_event_id, outcome="performed",
        ))
    assert len(actions) == len(connections) == 1
    assert service.get_mcp_action_history("thread-a")[0].review_outcome == ""


def test_mcp_follow_up_requires_a_separate_recorded_approval(monkeypatch):
    service, ledger, _request, actions, _connections = _mcp_service(monkeypatch, follow_up=True)
    assert "Another" in asyncio.run(service.resolve_mcp_approval("thread-a", approved=True))
    follow_up = service.get_pending_mcp_approval("thread-a")
    assert follow_up is not None
    assert ledger.read()[-1]["payload"]["status"] == "pending"
    assert "completed" in asyncio.run(service.resolve_mcp_approval("thread-a", approved=True))
    assert len(actions) == 2
    assert len({event["payload"]["request_id"] for event in ledger.read()}) == 2


def test_mcp_withdrawn_allowlist_expires_approval_without_dispatch(monkeypatch):
    service, ledger, _request, actions, _connections = _mcp_service(monkeypatch)
    monkeypatch.setenv("DADBOT_HA_MCP_ALLOWED_TOOLS", "")
    assert "no longer available" in asyncio.run(service.resolve_mcp_approval("thread-a", approved=True))
    assert ledger.read()[-1]["payload"]["status"] == "expired"
    assert not actions


def test_mcp_approval_audit_failure_blocks_dispatch_and_retains_proposal(monkeypatch):
    service, ledger, request, actions, connections = _mcp_service(monkeypatch)

    def unavailable(*_args, **_kwargs):
        raise OSError("approval journal unavailable")

    monkeypatch.setattr(service._approval_journal().writer, "write_event", unavailable)
    with pytest.raises(OSError, match="journal unavailable"):
        asyncio.run(service.resolve_mcp_approval("thread-a", approved=True))
    assert service.get_pending_mcp_approval("thread-a") is request
    assert ledger.read()[-1]["payload"]["status"] == "pending"
    assert not actions and not connections


def test_mcp_stale_button_does_not_decide_a_different_proposal(monkeypatch):
    service, ledger, request, actions, connections = _mcp_service(monkeypatch)
    with pytest.raises(ValueError, match="no longer matches"):
        asyncio.run(service.resolve_mcp_approval("thread-a", approved=True, request_id="old-request"))
    assert service.get_pending_mcp_approval("thread-a") is request
    assert ledger.read()[-1]["payload"]["status"] == "pending"
    assert not actions and not connections and not service.bot.decision_contexts


def test_mcp_removed_configuration_is_handled_inside_graph(monkeypatch):
    service, ledger, _request, actions, connections = _mcp_service(monkeypatch)
    monkeypatch.delenv("DADBOT_HA_MCP_URL")
    assert "configuration changed" in asyncio.run(service.resolve_mcp_approval("thread-a", approved=True))
    assert len(service.bot.decision_contexts) == 1
    assert ledger.read()[-1]["payload"]["status"] == "expired"
    assert not actions and not connections


@pytest.mark.parametrize("approved", ["true", 1, None])
def test_mcp_decision_requires_explicit_boolean(monkeypatch, approved):
    service, ledger, _request, actions, connections = _mcp_service(monkeypatch)
    with pytest.raises(ValueError, match="boolean"):
        asyncio.run(service.resolve_mcp_approval("thread-a", approved=approved))
    assert ledger.read()[-1]["payload"]["status"] == "pending"
    assert not actions and not connections


def test_mcp_decision_cannot_target_another_active_thread(monkeypatch):
    service, ledger, _request, actions, connections = _mcp_service(monkeypatch)
    service.bot.active_thread_id = "thread-b"
    with pytest.raises(ValueError, match="active thread"):
        asyncio.run(service.resolve_mcp_approval("thread-a", approved=True))
    assert ledger.read()[-1]["payload"]["status"] == "pending"
    assert not actions and not connections


def test_completed_tool_io_survives_restart_and_replays_without_dispatch(tmp_path):
    path = tmp_path / "approvals.jsonl"
    journal = ToolApprovalJournal(_ledger(path), tenant_id="tenant-a")
    call = _call()
    journal.propose(call)
    journal.decide(call, approved=True)
    output = {"is_error": False, "result": {"state": "off", "receipt": "receipt-a"}}
    executor = AsyncMock(return_value=output)
    asyncio.run(_execute(journal, call, executor))
    restored = ToolApprovalJournal(_ledger(path), tenant_id="tenant-a")
    restored.expire_recovered()
    replay = asyncio.run(restored.replay(call, arguments={"entity_id": "light.kitchen"}))
    assert replay == output
    executor.assert_awaited_once()
    approvals = [event for event in restored.ledger.read() if event["type"] == "TOOL_APPROVAL_UPDATED"]
    assert "receipt-a" not in json.dumps(approvals)
    io = [event for event in restored.ledger.read() if event["type"] == "TOOL_IO_RECORDED"]
    assert len(io) == 1
    assert io[0]["payload"]["tool_io"]["successful_count"] == 1
    assert io[0]["payload"]["tool_io"]["records"][0]["output_payload"] == output
    assert approvals[-1]["payload"]["io_event_id"] == io[0]["event_id"]


def test_same_tool_arguments_with_separate_approvals_are_not_globally_cached():
    journal = ToolApprovalJournal(ExecutionLedger(), tenant_id="tenant-a")
    executor = AsyncMock(return_value={"is_error": False, "result": "done"})
    for call in (_call(), _call(request_id="approval-b")):
        journal.propose(call)
        journal.decide(call, approved=True)
        asyncio.run(_execute(journal, call, executor))
    assert executor.await_count == 2


@pytest.mark.parametrize("status", ["pending", "approved", "started", "uncertain", "rejected", "expired"])
def test_replay_requires_durably_successful_approval(status):
    journal = ToolApprovalJournal(ExecutionLedger(), tenant_id="tenant-a")
    call = _call()
    journal.propose(call)
    if status in {"approved", "started", "uncertain"}:
        journal.decide(call, approved=True)
    if status in {"started", "uncertain"}:
        journal._transition(call, "started", expected={"approved"})
    if status == "uncertain":
        journal._transition(call, "uncertain", expected={"started"})
    if status == "rejected":
        journal.decide(call, approved=False)
    if status == "expired":
        journal.expire(call)
    with pytest.raises(RuntimeError, match="successful"):
        asyncio.run(journal.replay(call, arguments={"entity_id": "light.kitchen"}))


def test_replay_rejects_changed_arguments_or_identity():
    journal = ToolApprovalJournal(ExecutionLedger(), tenant_id="tenant-a")
    call = _call()
    journal.propose(call)
    journal.decide(call, approved=True)
    asyncio.run(_execute(journal, call, AsyncMock(return_value={"result": "done"})))
    with pytest.raises(ValueError, match="arguments changed"):
        asyncio.run(journal.replay(call, arguments={"entity_id": "light.other"}))
    with pytest.raises(ValueError, match="identity"):
        asyncio.run(journal.replay(replace(call, trace_id="other"), arguments={"entity_id": "light.kitchen"}))


def test_failed_tool_io_append_prevents_success_and_never_replays(monkeypatch):
    journal = ToolApprovalJournal(ExecutionLedger(), tenant_id="tenant-a")
    call = _call()
    journal.propose(call)
    journal.decide(call, approved=True)
    write = journal.writer.write_event

    def fail_io(event_type, **kwargs):
        if event_type == "TOOL_IO_RECORDED":
            raise OSError("tool IO unavailable")
        return write(event_type, **kwargs)

    monkeypatch.setattr(journal.writer, "write_event", fail_io)
    executor = AsyncMock(return_value={"result": "completed"})
    with pytest.raises(OSError, match="tool IO unavailable"):
        asyncio.run(_execute(journal, call, executor))
    assert journal.status(call) == "started"
    journal.expire_recovered()
    assert journal.status(call) == "uncertain"
    with pytest.raises(RuntimeError, match="successful"):
        asyncio.run(journal.replay(call, arguments={"entity_id": "light.kitchen"}))
    executor.assert_awaited_once()


def test_actual_mcp_service_records_kernel_io_and_restart_replay_never_connects(tmp_path, monkeypatch):
    service, ledger, _request, actions, connections = _mcp_service(
        monkeypatch, ledger=_ledger(tmp_path / "mcp.jsonl"),
    )
    call = service._pending["thread-a"][3]
    asyncio.run(service.resolve_mcp_approval("thread-a", approved=True))
    assert len(actions) == len(connections) == 1
    restored = ToolApprovalJournal(_ledger(tmp_path / "mcp.jsonl"), tenant_id="tenant-a")
    restored.expire_recovered()
    from dadbot.core.graph_context import TurnContext

    executed = service.bot.decision_contexts[0]
    context = TurnContext(user_input="", trace_id=executed.trace_id, metadata=dict(executed.metadata))
    assert asyncio.run(restored.replay(
        call, arguments={"entity_id": "light.kitchen"}, turn_context=context,
    ))["content"] == "completed"
    assert len(actions) == len(connections) == 1
    event = next(event for event in ledger.read() if event["type"] == "TOOL_IO_RECORDED")
    assert event["session_id"] == "thread-a" and event["tenant_id"] == "tenant-a"
    assert (
        event["payload"]["tool_io"]["records"][0]["metadata"]["scope"]["request_id"]
        == context.metadata["request_id"]
    )
    assert event["trace_id"] == call.trace_id
    assert event["payload"]["execution_scope"]["trace_id"] == context.trace_id


def test_tool_replay_rejects_environment_drift(monkeypatch):
    from dadbot.core import tool_executor
    from dadbot.core.runtime_errors import ReplayInvariantViolation

    journal = ToolApprovalJournal(ExecutionLedger(), tenant_id="tenant-a")
    call = _call()
    journal.propose(call)
    journal.decide(call, approved=True)
    asyncio.run(_execute(journal, call, AsyncMock(return_value={"result": "done"})))
    monkeypatch.setattr(tool_executor, "_environment_fingerprint", lambda _metadata: "changed-env")
    with pytest.raises(ReplayInvariantViolation, match="ledger hit"):
        asyncio.run(journal.replay(call, arguments={"entity_id": "light.kitchen"}))


@pytest.mark.parametrize("result", [{"result": object()}, {"result": float("nan")}])
def test_non_json_tool_output_cannot_be_certified_for_exact_replay(result):
    journal = ToolApprovalJournal(ExecutionLedger(), tenant_id="tenant-a")
    call = _call()
    journal.propose(call)
    journal.decide(call, approved=True)
    executor = AsyncMock(return_value=result)
    with pytest.raises((TypeError, ValueError)):
        asyncio.run(_execute(journal, call, executor))
    assert journal.status(call) == "started"
    journal.expire_recovered()
    assert journal.status(call) == "uncertain"
    executor.assert_awaited_once()


def test_completed_io_receipts_cannot_be_swapped_between_approvals(monkeypatch):
    from dadbot.core.runtime_errors import ReplayInvariantViolation

    journal = ToolApprovalJournal(ExecutionLedger(), tenant_id="tenant-a")
    first, second = _call(), _call(request_id="approval-b")
    for call in (first, second):
        journal.propose(call)
        journal.decide(call, approved=True)
        asyncio.run(_execute(journal, call, AsyncMock(return_value={"result": call.request_id})))
    events = deepcopy(journal.ledger.read())
    completions = [
        event for event in events if event["type"] == "TOOL_APPROVAL_UPDATED"
        and event["payload"]["status"] == "succeeded"
    ]
    completions[0]["payload"]["io_event_id"] = completions[1]["payload"]["io_event_id"]
    monkeypatch.setattr(journal.ledger, "read", lambda: events)
    with pytest.raises(ReplayInvariantViolation, match="scope"):
        asyncio.run(journal.replay(first, arguments={"entity_id": "light.kitchen"}))
