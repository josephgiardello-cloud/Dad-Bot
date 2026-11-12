from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from dadbot.assistant_runtime import AssistantRuntime
from dadbot.core.execution_contract import TurnDelivery, TurnResponse
from dadbot.runtime.agent_driver_loop import AgentDriverLoop, DriverLoopPolicy

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _isolated_task_workspace(monkeypatch, tmp_path):
    monkeypatch.setattr("dadbot_system.local_mcp_server._project_root", lambda: tmp_path)


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------


class _CyclingKernel:
    """Cycles through (reply, should_end) pairs, then returns empty."""

    def __init__(self, turns: list[tuple[str, bool]]) -> None:
        self.config = SimpleNamespace(tenant_id="t1_" + "a" * 26)
        self._turns = list(turns)
        self._idx = 0
        self.calls: list[str] = []
        self.requests = []
        self.reflection_payloads = []

    def execute_turn(self, request, **_kwargs):
        assert request.delivery == TurnDelivery.SYNC
        self.requests.append(request)
        self.calls.append(str(request.input.text or ""))
        if self._idx < len(self._turns):
            reply, should_end = self._turns[self._idx]
            self._idx += 1
            return TurnResponse(reply=reply, should_end=should_end)
        return TurnResponse(reply="", should_end=False)

    def call_llm(self, messages, **_kwargs):
        payload = json.loads(messages[1]["content"])
        self.reflection_payloads.append(payload)
        updates = {"status": "in progress"} if payload["turn_index"] == 1 else {}
        return json.dumps(
            {
                "action_input": payload["observation"],
                "goal_updates": updates,
            }
        )


class _FailAfterNKernel:
    """Succeeds for first n calls then raises."""

    def __init__(self, succeed: int, reply: str = "ok") -> None:
        self._succeed = succeed
        self._reply = reply
        self._count = 0

    def execute_turn(self, request, **_kwargs):
        self._count += 1
        if self._count > self._succeed:
            raise RuntimeError("simulated kernel failure")
        return TurnResponse(reply=self._reply, should_end=False)


class _FailFirstKernel:
    def __init__(self) -> None:
        self._count = 0

    def execute_turn(self, request, **_kwargs):
        self._count += 1
        if self._count == 1:
            raise RuntimeError("simulated first-turn failure")
        return TurnResponse(reply="ok", should_end=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_loop_happy_path_multi_turn():
    """Loop runs multiple turns and records each one."""
    kernel = _CyclingKernel([("hello", False), ("world", False), ("done", True)])
    loop = AgentDriverLoop(kernel, policy=DriverLoopPolicy(max_turns=5))
    result = loop.run("start")

    assert result.stop_reason == "kernel_should_end"
    assert result.completed_turns == 3
    assert result.failures == 0
    assert len(result.records) == 3
    assert result.records[0].reply == "hello"
    assert result.records[2].reply == "done"
    assert result.records[2].should_end is True


def test_loop_respects_max_turns():
    """Loop stops at max_turns when kernel never signals end."""
    kernel = _CyclingKernel([("a", False)] * 20)
    loop = AgentDriverLoop(kernel, policy=DriverLoopPolicy(max_turns=3))
    result = loop.run("go")

    assert result.stop_reason == "max_turns"
    assert result.completed_turns == 3


def test_loop_failure_budget_terminates_cleanly():
    """Loop stops after exceeding failure budget and result is well-formed."""
    kernel = _FailAfterNKernel(succeed=1)
    loop = AgentDriverLoop(kernel, policy=DriverLoopPolicy(max_turns=10, max_failures=2))
    result = loop.run("start")

    assert result.stop_reason == "failure_budget"
    assert result.failures > 0
    # At least the first successful turn should be recorded
    assert result.completed_turns >= 1
    # All failed records should carry error text
    failed = [r for r in result.records if r.commit_status == "failed"]
    assert all(r.error for r in failed)


def test_goal_updates_are_applied_only_after_a_committed_turn():
    kernel = _FailFirstKernel()
    observed_blackboards = []

    def reflect(ctx):
        observed_blackboards.append(dict(ctx["goal_blackboard"]))
        return {"action_input": "continue", "goal_updates": {"status": "advanced"}}

    result = AgentDriverLoop(
        kernel,
        policy=DriverLoopPolicy(max_turns=2, max_failures=1),
    ).run("start", reflection_hook=reflect)

    assert observed_blackboards == [{}, {}]
    assert [record.commit_status for record in result.records] == ["failed", "committed"]
    assert result.goal_blackboard == {"status": "advanced"}


def test_loop_records_are_ordered():
    """turn_index in records increments monotonically."""
    kernel = _CyclingKernel([("r1", False), ("r2", False), ("r3", True)])
    loop = AgentDriverLoop(kernel, policy=DriverLoopPolicy(max_turns=5))
    result = loop.run("input")

    indices = [r.turn_index for r in result.records]
    assert indices == sorted(indices)
    assert indices == list(range(1, len(indices) + 1))


def test_loop_single_turn_exits():
    """max_turns=1 stops after one committed turn."""
    kernel = _CyclingKernel([("reply", False)])
    loop = AgentDriverLoop(kernel, policy=DriverLoopPolicy(max_turns=1))
    result = loop.run("hi")

    assert result.completed_turns == 1
    assert result.stop_reason == "max_turns"


# ---------------------------------------------------------------------------
# AssistantRuntime surface tests
# ---------------------------------------------------------------------------


def test_run_agent_loop_on_facade_returns_result():
    """AssistantRuntime.run_agent_loop delegates to AgentDriverLoop correctly."""
    kernel = _CyclingKernel([("hello", False), ("bye", True)])
    rt = AssistantRuntime(kernel)
    result = rt.run_agent_loop("hi", policy=DriverLoopPolicy(max_turns=5))

    assert result.completed_turns == 2
    assert result.stop_reason == "kernel_should_end"
    assert result.goal_blackboard == {"status": "in progress"}
    assert kernel.requests[0].metadata["agent_loop"]["goal_blackboard"] == {}
    assert kernel.requests[0].metadata["agent_loop"]["trigger_type"] == "interactive"
    assert kernel.reflection_payloads[1]["goal_blackboard"] == {"status": "in progress"}
    assert kernel.reflection_payloads[0]["memory_signals"] == {
        "conflicts": [],
        "temporal": [],
    }


def test_facade_restores_tenant_session_goal_progress_and_verifies_completion():
    tenant_id = _CyclingKernel([]).config.tenant_id
    persisted_progress = {
        "schema_version": 1,
        "goal_blackboard": {"status": "resumed"},
        "last_committed": {
            "tenant_id": tenant_id,
            "session_id": "session-resume",
            "trace_id": "trace-1",
            "job_id": "job-1",
            "request_id": "request-1",
            "loop_run_id": "run-1",
            "turn_index": 1,
        },
        "last_tool_outcome_evidence": {
            "schema_version": 1,
            "trace_id": "trace-1",
            "outcomes": [],
            "truncated": False,
        },
    }

    class _Registry:
        def get_or_create(self, session_id, *, tenant_id):
            assert session_id == "session-resume"
            assert tenant_id == tenant_id_expected
            return {"state": {"agent_loop_progress": persisted_progress}}

    class _LedgerIndex:
        def completed_agent_loop_entries(self, *, session_id, tenant_id):
            assert session_id == "session-resume"
            assert tenant_id == tenant_id_expected
            return [{
                "tenant_id": tenant_id,
                "session_id": session_id,
                "request_id": "request-1",
                "job_id": "job-1",
                "trace_id": "trace-1",
                "metadata": {
                    "agent_loop": {
                        "schema_version": 1,
                        "loop_run_id": "run-1",
                        "turn_index": 1,
                        "goal_blackboard": {},
                        "goal_updates": {"status": "resumed"},
                    },
                    "tool_outcome_evidence": persisted_progress["last_tool_outcome_evidence"],
                },
            }, {
                "tenant_id": tenant_id,
                "session_id": session_id,
                "request_id": "request-2",
                "job_id": "job-2",
                "trace_id": "trace-2",
                "metadata": {
                    "agent_loop": {
                        "schema_version": 1,
                        "loop_run_id": "run-1",
                        "turn_index": 2,
                        "goal_blackboard": {"status": "resumed"},
                        "goal_updates": {"status": "recovered"},
                    },
                    "tool_outcome_evidence": {
                        "schema_version": 1,
                        "trace_id": "trace-2",
                        "outcomes": [],
                        "truncated": False,
                    },
                },
            }]

        def completed_metadata(self, *, session_id, request_id, tenant_id):
            verified_scopes.append((session_id, request_id, tenant_id))
            request = next(
                item for item in kernel.requests
                if item.metadata["request_id"] == request_id
            )
            return {
                "tenant_id": tenant_id,
                "session_id": session_id,
                "request_id": request_id,
                "trace_id": "trace-current",
                "metadata": {
                    "agent_loop": request.metadata["agent_loop"],
                    "tool_outcome_evidence": {
                        "schema_version": 1,
                        "trace_id": "trace-current",
                        "outcomes": [],
                        "truncated": False,
                    },
                },
            }

    tenant_id_expected = tenant_id
    verified_scopes = []
    reconciled_patches = []
    kernel = _CyclingKernel([("done", True)])
    kernel.control_plane = SimpleNamespace(
        registry=_Registry(),
        _ledger_index=_LedgerIndex(),
        _merge_session_state=lambda **kwargs: reconciled_patches.append(kwargs),
    )
    observed_boards = []

    def reflection_hook(ctx):
        observed_boards.append(dict(ctx["goal_blackboard"]))
        observed_tool_outcomes.append(dict(ctx["tool_outcome_evidence"]))
        return {"action_input": "continue"}

    observed_tool_outcomes = []
    result = AssistantRuntime(kernel).run_agent_loop(
        "resume",
        policy=DriverLoopPolicy(max_turns=1),
        session_id="session-resume",
        reflection_hook=reflection_hook,
        enable_semantic_memory=False,
    )

    assert observed_boards == [{"status": "recovered"}]
    assert observed_tool_outcomes[0]["trace_id"] == "trace-2"
    assert result.tool_outcome_evidence["trace_id"] == "trace-current"
    assert result.goal_blackboard == {"status": "recovered"}
    assert result.completed_turns == 1
    assert reconciled_patches[0]["state_patch"]["agent_loop_progress"]["last_committed"]["request_id"] == "request-2"
    assert reconciled_patches[0]["tenant_id"] == tenant_id
    request = kernel.requests[0]
    assert verified_scopes == [
        ("session-resume", request.metadata["request_id"], tenant_id),
    ]


def test_run_agent_loop_default_policy():
    """Facade works without explicitly passing a policy."""
    kernel = _CyclingKernel([("ok", True)])
    rt = AssistantRuntime(kernel)
    result = rt.run_agent_loop("test")

    assert result.completed_turns == 1
    assert result.stop_reason == "kernel_should_end"
