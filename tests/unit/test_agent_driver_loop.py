from __future__ import annotations

import hashlib
from typing import Literal

import pytest

from dadbot.core.agent_loop_recovery import AgentLoopCursor
from dadbot.core.execution_contract import TurnDelivery, TurnResponse
from dadbot.core.execution_receipt import ReceiptSigner
from dadbot.runtime.agent_driver_loop import (
    AgentDriverLoop,
    AgentLoopOutcomeVerification,
    DriverLoopPolicy,
)
from dadbot.runtime_core.bus import EventBus

pytestmark = pytest.mark.unit


class _KernelStub:
    def __init__(self, replies: list[tuple[str, bool]]) -> None:
        self.replies = list(replies)
        self.calls: list[str] = []
        self.requests = []

    def execute_turn(self, request, **_kwargs):
        assert request.delivery == TurnDelivery.SYNC
        self.requests.append(request)
        self.calls.append(str(request.input.text or ""))
        if not self.replies:
            return TurnResponse(reply="", should_end=False)
        reply, should_end = self.replies.pop(0)
        return TurnResponse(reply=reply, should_end=should_end)


class _FailingKernelStub:
    def __init__(self, fail_count: int) -> None:
        self.fail_count = int(fail_count)

    def execute_turn(self, request, **_kwargs):
        _ = request
        if self.fail_count > 0:
            self.fail_count -= 1
            raise RuntimeError("synthetic failure")
        return TurnResponse(reply="recovered", should_end=False)


def _completion_for_request(kernel, signer, request_id, session_id, tenant_id):
    request = next(item for item in kernel.requests if item.metadata["request_id"] == request_id)
    trace_id = f"trace-{request_id}"
    receipt = signer.sign(
        turn_id=trace_id,
        stage="save",
        sequence=1,
        stage_call_id=f"call-{request_id}",
        checkpoint_hash="a" * 64,
        prev_receipt_sig="",
    )
    return {
        "tenant_id": tenant_id,
        "session_id": session_id,
        "request_id": request_id,
        "job_id": f"job-{request_id}",
        "trace_id": trace_id,
        "metadata": {
            "agent_loop": request.metadata["agent_loop"],
            "tool_outcome_evidence": {
                "schema_version": 1,
                "trace_id": trace_id,
                "outcomes": [],
                "truncated": False,
            },
            "execution_receipt_status": "verified",
            "execution_receipt_evidence": {
                "schema_version": 1,
                "trace_id": trace_id,
                "key_id": signer.key_id,
                "stages": ["save"],
                "receipts": [receipt.to_dict()],
            },
        },
    }


def test_driver_loop_stops_on_kernel_should_end():
    kernel = _KernelStub(replies=[("reflecting", False), ("final", True), ("unused", False)])
    loop = AgentDriverLoop(kernel, policy=DriverLoopPolicy(max_turns=5, max_failures=1, max_consecutive_noop=2))

    result = loop.run("start")

    assert result.stop_reason == "kernel_should_end"
    assert result.completed_turns == 2
    assert [record.commit_status for record in result.records] == ["committed", "committed"]
    assert kernel.calls == ["start", "reflecting"]


def test_driver_loop_stops_on_noop_threshold():
    kernel = _KernelStub(replies=[("", False), ("", False), ("never", False)])
    loop = AgentDriverLoop(kernel, policy=DriverLoopPolicy(max_turns=6, max_failures=1, max_consecutive_noop=2))

    result = loop.run("start")

    assert result.stop_reason == "noop_threshold"
    assert result.completed_turns == 2
    assert result.consecutive_noop == 2


def test_driver_loop_respects_failure_budget():
    kernel = _FailingKernelStub(fail_count=3)
    loop = AgentDriverLoop(kernel, policy=DriverLoopPolicy(max_turns=5, max_failures=1, max_consecutive_noop=2))

    result = loop.run("start")

    assert result.stop_reason == "failure_budget"
    assert result.failures == 2
    assert all(record.commit_status == "failed" for record in result.records)


def test_driver_loop_emits_telemetry_events():
    """EventBus receives loop_started, loop_turn_completed, and loop_stopped."""
    bus = EventBus()
    kernel = _KernelStub(replies=[("hi", False), ("bye", True)])
    loop = AgentDriverLoop(kernel, policy=DriverLoopPolicy(max_turns=5), event_bus=bus)

    loop.run("hello", session_id="sess-1")

    events = bus.drain()
    types = [e.type for e in events]
    assert types[0] == "loop_started"
    assert types.count("loop_turn_completed") == 2
    assert types[-1] == "loop_stopped"
    # all events carry the session thread_id
    assert all(e.thread_id == "sess-1" for e in events)
    turn_event = next(e for e in events if e.type == "loop_turn_completed")
    assert turn_event.payload["turn_index"] == 1
    assert turn_event.payload["action_type"] == "kernel_turn"
    assert turn_event.payload["commit_status"] == "committed"
    assert turn_event.payload["duration_ms"] >= 0.0
    # loop_stopped payload has stop_reason
    stopped = next(e for e in events if e.type == "loop_stopped")
    assert stopped.payload["stop_reason"] == "kernel_should_end"


def test_driver_loop_retries_after_system_observation_without_noop_penalty():
    kernel = _KernelStub(replies=[("ok", True)])
    loop = AgentDriverLoop(kernel, policy=DriverLoopPolicy(max_turns=3, max_failures=1, max_consecutive_noop=2))

    calls = {"n": 0}

    def reflection_hook(_ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "action_input": "",
                "should_continue": True,
                "system_observation": "System observation: invalid tool attempted, retry.",
            }
        return {"action_input": "retry with valid tool", "should_continue": True}

    result = loop.run("start", reflection_hook=reflection_hook)

    assert result.stop_reason == "kernel_should_end"
    assert result.completed_turns == 1
    assert result.records[0].commit_status == "skipped"
    assert result.records[1].commit_status == "committed"
    assert kernel.calls == ["retry with valid tool"]


def test_driver_loop_reflects_on_observation_and_ledger_records_goal_board():
    kernel = _KernelStub(replies=[("progress", False), ("done", True)])
    loop = AgentDriverLoop(kernel, policy=DriverLoopPolicy(max_turns=3))
    contexts = []

    def reflection_hook(ctx):
        contexts.append(ctx)
        if ctx["turn_index"] == 1:
            return {"action_input": "start task", "goal_updates": {"status": "in progress"}}
        return {"action_input": "finish task"}

    result = loop.run("initial task", tenant_id="tenant-a", reflection_hook=reflection_hook)

    assert contexts[0]["observation"] == "initial task"
    assert contexts[1]["goal_blackboard"] == {"status": "in progress"}
    assert result.goal_blackboard == {"status": "in progress"}
    assert kernel.requests[0].context.tenant_id == "tenant-a"
    assert kernel.requests[0].context.request_id == kernel.requests[0].metadata["request_id"]
    assert kernel.requests[0].metadata["agent_loop"]["turn_index"] == 1
    assert kernel.requests[0].metadata["agent_loop"]["goal_blackboard"] == {}
    assert kernel.requests[0].metadata["agent_loop"]["plan_verification"] == "passed"
    assert kernel.requests[0].metadata["agent_loop"]["trigger_type"] == "interactive"
    assert kernel.requests[0].metadata["agent_loop"]["schema_version"] == 1
    assert kernel.requests[0].metadata["agent_loop"]["loop_run_id"]


def test_driver_loop_passes_canonical_hash_only_outcomes_to_next_reflection():
    kernel = _KernelStub(replies=[("tool ran", False), ("finished", True)])
    contexts = []
    evidence = {
        "schema_version": 1,
        "trace_id": "trace-1",
        "outcomes": [
            {
                "event_id": "00000000-0000-4000-8000-000000000001",
                "event_checksum": f"evtchk-{'a' * 64}",
                "previous_checksum": "",
                "trace_id": "trace-1",
                "tool_name": "safe_tool",
                "status": "succeeded",
                "authorization_decision": "allowed",
                "authority_mode": "trusted_local",
                "input_hash": "b" * 64,
                "output_hash": "c" * 64,
            },
        ],
        "truncated": False,
    }

    def completion_reader(request_id, session_id, tenant_id):
        request = next(item for item in kernel.requests if item.metadata["request_id"] == request_id)
        return {
            "tenant_id": tenant_id,
            "session_id": session_id,
            "request_id": request_id,
            "trace_id": "trace-1",
            "metadata": {
                "agent_loop": request.metadata["agent_loop"],
                "tool_outcome_evidence": evidence,
            },
        }

    loop = AgentDriverLoop(
        kernel,
        policy=DriverLoopPolicy(max_turns=2),
        commit_verifier=completion_reader,
    )
    result = loop.run(
        "run the tool",
        reflection_hook=lambda ctx: (
            contexts.append(ctx) or {"action_input": "continue"}
        ),
    )

    assert contexts[1]["tool_outcome_evidence"] == evidence
    assert result.tool_outcome_evidence == evidence
    assert result.completed_turns == 2


def test_driver_loop_rejects_cross_scope_canonical_completion_metadata():
    kernel = _KernelStub(replies=[("untrusted", True)])
    loop = AgentDriverLoop(
        kernel,
        commit_verifier=lambda request_id, session_id, _tenant_id: {
            "tenant_id": "other-tenant",
            "session_id": session_id,
            "request_id": request_id,
            "trace_id": "trace-1",
            "metadata": {},
        },
    )

    with pytest.raises(ValueError, match="scope"):
        loop.run(
            "start",
            tenant_id="tenant-a",
            reflection_hook=lambda _ctx: {
                "action_input": "continue",
                "goal_updates": {"status": "advanced"},
            },
        )


def test_driver_loop_does_not_advance_without_canonical_completion():
    kernel = _KernelStub(replies=[("unverified", True)])
    loop = AgentDriverLoop(
        kernel,
        policy=DriverLoopPolicy(max_turns=3),
        commit_verifier=lambda _request_id, _session_id, _tenant_id: False,
    )

    result = loop.run(
        "start",
        reflection_hook=lambda _ctx: {
            "action_input": "continue",
            "goal_updates": {"status": "advanced"},
        },
        initial_goal_blackboard={"status": "pending"},
    )

    assert result.stop_reason == "commit_unverified"
    assert result.completed_turns == 0
    assert result.goal_blackboard == {"status": "pending"}
    assert result.records[0].commit_status == "uncertain"
    assert result.records[0].error == "Canonical JOB_COMPLETED evidence is missing for the request"


def test_driver_loop_stops_as_uncertain_when_dispatch_raises_with_verifier():
    kernel = _FailingKernelStub(fail_count=1)
    loop = AgentDriverLoop(
        kernel,
        policy=DriverLoopPolicy(max_turns=3, max_failures=2),
        commit_verifier=lambda _request_id, _session_id, _tenant_id: False,
    )

    result = loop.run(
        "start",
        reflection_hook=lambda _ctx: {
            "action_input": "possibly side-effecting action",
            "goal_updates": {"status": "advanced"},
        },
    )

    assert result.stop_reason == "commit_unverified"
    assert result.failures == 0
    assert result.completed_turns == 0
    assert result.goal_blackboard == {}
    assert result.records[0].commit_status == "uncertain"


def test_driver_loop_rejects_invalid_plan_before_kernel_dispatch():
    kernel = _KernelStub(replies=[("unused", True)])
    loop = AgentDriverLoop(
        kernel,
        policy=DriverLoopPolicy(max_turns=3, max_failures=0),
        allowed_tools=["safe_tool"],
    )

    result = loop.run(
        "start",
        reflection_hook=lambda _ctx: {
            "action_input": "run a tool",
            "tool_calls": [{"name": "unsafe_tool", "arguments": {}}],
        },
    )

    assert result.stop_reason == "failure_budget"
    assert result.failures == 1
    assert kernel.calls == []
    assert result.records[0].commit_status == "rejected"
    assert "not in allowed list" in result.records[0].error


def test_driver_loop_rejects_non_string_action_before_kernel_dispatch():
    kernel = _KernelStub(replies=[("unused", True)])
    loop = AgentDriverLoop(kernel, policy=DriverLoopPolicy(max_failures=0))

    result = loop.run("start", reflection_hook=lambda _ctx: {"action_input": 42})

    assert result.stop_reason == "failure_budget"
    assert kernel.calls == []
    assert result.records[0].commit_status == "rejected"


def test_deterministic_outcome_verifier_stops_only_when_postconditions_are_satisfied():
    kernel = _KernelStub(replies=[("first", True), ("finished", False), ("unused", False)])
    signer = ReceiptSigner(b"test-goal-verifier-key-material-0001")
    statuses: list[Literal["not_satisfied", "satisfied"]] = ["not_satisfied", "satisfied"]

    def completion_reader(request_id, session_id, tenant_id):
        return _completion_for_request(kernel, signer, request_id, session_id, tenant_id)

    def outcome_verifier(evidence) -> Literal["not_satisfied", "satisfied"]:
        assert evidence.tenant_id == "tenant-a"
        assert evidence.goal_blackboard == {}
        assert evidence.proposed_goal_updates == {}
        assert evidence.execution_receipt_evidence["key_id"] == signer.key_id
        assert evidence.tool_outcome_evidence["trace_id"] == evidence.trace_id
        return statuses.pop(0)

    loop = AgentDriverLoop(
        kernel,
        policy=DriverLoopPolicy(max_turns=3, outcome_verifier_id="test.goal.v1"),
        commit_verifier=completion_reader,
        outcome_verification=AgentLoopOutcomeVerification(outcome_verifier, signer),
    )

    result = loop.run(
        "start",
        tenant_id="tenant-a",
        reflection_hook=lambda _ctx: {"action_input": "continue"},
    )

    assert result.stop_reason == "goal_satisfied"
    assert result.goal_status == "satisfied"
    assert result.completed_turns == 2
    assert [record.outcome_status for record in result.records] == ["not_satisfied", "satisfied"]
    assert len(kernel.calls) == 2


def test_unknown_goal_outcome_stops_without_claiming_success():
    kernel = _KernelStub(replies=[("maybe", False), ("must not run", False)])
    signer = ReceiptSigner(b"test-goal-verifier-key-material-0001")

    def completion_reader(request_id, session_id, tenant_id):
        return _completion_for_request(kernel, signer, request_id, session_id, tenant_id)

    loop = AgentDriverLoop(
        kernel,
        policy=DriverLoopPolicy(max_turns=2, outcome_verifier_id="test.goal.v1"),
        commit_verifier=completion_reader,
        outcome_verification=AgentLoopOutcomeVerification(lambda _evidence: "unknown", signer),
    )
    result = loop.run("start", reflection_hook=lambda _ctx: {"action_input": "possibly side-effecting"})

    assert result.stop_reason == "outcome_unverified"
    assert result.goal_status == "unknown"
    assert result.completed_turns == 1
    assert len(kernel.calls) == 1
    assert result.records[0].outcome_status == "unknown"


def test_goal_verification_requires_a_stable_contract_and_canonical_receipts():
    kernel = _KernelStub(replies=[("unused", True)])

    with pytest.raises(ValueError, match="matching policy ID"):
        AgentDriverLoop(
            kernel,
            outcome_verification=AgentLoopOutcomeVerification(
                lambda _evidence: "satisfied",
                ReceiptSigner(b"test-goal-verifier-key-material-0001"),
            ),
            commit_verifier=lambda *_args: None,
        )

    with pytest.raises(RuntimeError, match="stable receipt signer"):
        AgentDriverLoop(
            kernel,
            policy=DriverLoopPolicy(outcome_verifier_id="test.goal.v1"),
            outcome_verification=AgentLoopOutcomeVerification(
                lambda _evidence: "satisfied",
                ReceiptSigner(),
            ),
            commit_verifier=lambda *_args: None,
        )


def test_legacy_cursor_policy_loads_without_an_outcome_verifier_id():
    legacy_policy = {
        "max_turns": 2,
        "max_failures": 1,
        "max_consecutive_noop": 2,
        "allowed_tools": None,
        "trigger_type": "interactive",
    }
    cursor = AgentLoopCursor(
        tenant_id="tenant-a",
        session_id="session-a",
        loop_run_id="run-a",
        input_hash=hashlib.sha256(b"goal").hexdigest(),
        policy=legacy_policy,
    )

    loaded = AgentLoopCursor.read(
        cursor.to_dict(),
        tenant_id="tenant-a",
        session_id="session-a",
        initial_input="goal",
        policy={**legacy_policy, "outcome_verifier_id": ""},
    )

    assert loaded.policy["outcome_verifier_id"] == ""


def test_invalid_receipts_never_reach_the_goal_verifier():
    kernel = _KernelStub(replies=[("done", False)])
    signer = ReceiptSigner(b"test-goal-verifier-key-material-0001")
    verifier_calls = []

    def completion_reader(request_id, session_id, tenant_id):
        completion = _completion_for_request(kernel, signer, request_id, session_id, tenant_id)
        completion["metadata"]["execution_receipt_evidence"]["receipts"][0]["signature"] = "0" * 64
        return completion

    loop = AgentDriverLoop(
        kernel,
        policy=DriverLoopPolicy(max_turns=1, outcome_verifier_id="test.goal.v1"),
        commit_verifier=completion_reader,
        outcome_verification=AgentLoopOutcomeVerification(
            lambda evidence: verifier_calls.append(evidence) or "satisfied",
            signer,
        ),
    )

    with pytest.raises(ValueError, match="receipt verification failed"):
        loop.run("start", reflection_hook=lambda _ctx: {"action_input": "do work"})

    assert verifier_calls == []
