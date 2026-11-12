from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dadbot.core.agent_loop_progress import (
    AgentLoopCommitIdentity,
    apply_committed_agent_loop_progress,
    reconcile_committed_agent_loop_progress,
    read_agent_loop_goal_blackboard,
    read_agent_loop_tool_outcome_evidence,
)
from dadbot.core.control_plane_post_turn import _apply_submit_success_postprocessing_impl

pytestmark = pytest.mark.unit


def _progress(*, tenant_id: str = "tenant-a", session_id: str = "session-a") -> dict:
    return {
        "schema_version": 1,
        "goal_blackboard": {"status": "in progress"},
        "last_committed": {
            "tenant_id": tenant_id,
            "session_id": session_id,
            "trace_id": "trace-1",
            "job_id": "job-1",
            "request_id": "request-1",
            "loop_run_id": "run-1",
            "turn_index": 1,
        },
    }


def _tool_outcome_evidence(trace_id: str = "trace-2") -> dict:
    return {
        "schema_version": 1,
        "trace_id": trace_id,
        "outcomes": [
            {
                "event_id": "00000000-0000-4000-8000-000000000001",
                "event_checksum": f"evtchk-{'a' * 64}",
                "previous_checksum": "",
                "trace_id": trace_id,
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


def test_reads_goal_blackboard_for_matching_tenant_and_session():
    state = {"agent_loop_progress": _progress()}

    assert read_agent_loop_goal_blackboard(
        state,
        tenant_id="tenant-a",
        session_id="session-a",
    ) == {"status": "in progress"}


@pytest.mark.parametrize(
    ("tenant_id", "session_id"),
    [("tenant-b", "session-a"), ("tenant-a", "session-b")],
)
def test_rejects_goal_blackboard_from_another_scope(tenant_id: str, session_id: str):
    with pytest.raises(ValueError, match="different"):
        read_agent_loop_goal_blackboard(
            {"agent_loop_progress": _progress()},
            tenant_id=tenant_id,
            session_id=session_id,
        )


def test_rejects_malformed_persisted_goal_progress():
    with pytest.raises(ValueError, match="Invalid persisted agent-loop"):
        read_agent_loop_goal_blackboard({"agent_loop_progress": {"schema_version": 1}})


def test_applies_progress_only_from_well_formed_committed_metadata():
    state: dict = {}
    applied = apply_committed_agent_loop_progress(
        state,
        metadata={
            "request_id": "request-2",
            "agent_loop": {
                "schema_version": 1,
                "loop_run_id": "run-2",
                "turn_index": 2,
                "goal_updates": {"status": "complete"},
            },
        },
        identity=AgentLoopCommitIdentity(
            tenant_id="tenant-a",
            session_id="session-a",
            trace_id="trace-2",
            job_id="job-2",
        ),
    )

    assert applied is True
    assert read_agent_loop_goal_blackboard(state) == {"status": "complete"}
    assert state["agent_loop_progress"]["last_committed"]["request_id"] == "request-2"


def test_persists_and_reads_hash_only_tool_outcomes_with_scope_validation():
    state: dict = {}
    apply_committed_agent_loop_progress(
        state,
        metadata={
            "request_id": "request-2",
            "agent_loop": {
                "schema_version": 1,
                "loop_run_id": "run-2",
                "turn_index": 1,
                "goal_updates": {},
            },
            "tool_outcome_evidence": _tool_outcome_evidence(),
        },
        identity=AgentLoopCommitIdentity(
            tenant_id="tenant-a",
            session_id="session-a",
            trace_id="trace-2",
            job_id="job-2",
        ),
    )

    assert read_agent_loop_tool_outcome_evidence(
        state,
        tenant_id="tenant-a",
        session_id="session-a",
    ) == _tool_outcome_evidence()
    with pytest.raises(ValueError, match="different tenant"):
        read_agent_loop_tool_outcome_evidence(state, tenant_id="tenant-b", session_id="session-a")


def test_rejects_tool_outcome_evidence_from_another_trace():
    with pytest.raises(ValueError, match="different trace"):
        apply_committed_agent_loop_progress(
            {},
            metadata={
                "request_id": "request-2",
                "agent_loop": {
                    "schema_version": 1,
                    "loop_run_id": "run-2",
                    "turn_index": 1,
                    "goal_updates": {},
                },
                "tool_outcome_evidence": _tool_outcome_evidence("trace-other"),
            },
            identity=AgentLoopCommitIdentity(
                tenant_id="tenant-a",
                session_id="session-a",
                trace_id="trace-2",
                job_id="job-2",
            ),
        )


def test_rejects_raw_tool_outputs_in_hash_only_evidence():
    evidence = _tool_outcome_evidence()
    evidence["outcomes"][0]["raw_output"] = "must not be persisted"
    with pytest.raises(ValueError, match="Invalid hash-only tool outcome"):
        apply_committed_agent_loop_progress(
            {},
            metadata={
                "request_id": "request-2",
                "agent_loop": {
                    "schema_version": 1,
                    "loop_run_id": "run-2",
                    "turn_index": 1,
                    "goal_updates": {},
                },
                "tool_outcome_evidence": evidence,
            },
            identity=AgentLoopCommitIdentity(
                tenant_id="tenant-a",
                session_id="session-a",
                trace_id="trace-2",
                job_id="job-2",
            ),
        )


def test_reconciles_committed_completion_missing_session_state():
    state: dict = {}
    reconciled = reconcile_committed_agent_loop_progress(
        state,
        completed_entries=[
            {
                "tenant_id": "tenant-a",
                "session_id": "session-a",
                "request_id": "request-2",
                "job_id": "job-2",
                "trace_id": "trace-2",
                "metadata": {
                    "agent_loop": {
                        "schema_version": 1,
                        "loop_run_id": "run-2",
                        "turn_index": 1,
                        "goal_blackboard": {},
                        "goal_updates": {"status": "complete"},
                    },
                    "tool_outcome_evidence": _tool_outcome_evidence(),
                },
            },
        ],
        tenant_id="tenant-a",
        session_id="session-a",
    )

    assert reconciled is True
    assert read_agent_loop_goal_blackboard(state) == {"status": "complete"}
    assert state["agent_loop_progress"]["last_committed"]["request_id"] == "request-2"


def test_refuses_to_reconcile_from_stale_goal_snapshot():
    state = {"agent_loop_progress": _progress()}
    with pytest.raises(ValueError, match="stale goal snapshot"):
        reconcile_committed_agent_loop_progress(
            state,
            completed_entries=[
                {
                    "tenant_id": "tenant-a",
                    "session_id": "session-a",
                    "request_id": "request-1",
                    "job_id": "job-1",
                    "trace_id": "trace-1",
                    "metadata": {
                        "agent_loop": {
                            "schema_version": 1,
                            "loop_run_id": "run-1",
                            "turn_index": 1,
                            "goal_blackboard": {},
                            "goal_updates": {"status": "in progress"},
                        },
                    },
                },
                {
                    "tenant_id": "tenant-a",
                    "session_id": "session-a",
                    "request_id": "request-2",
                    "job_id": "job-2",
                    "trace_id": "trace-2",
                    "metadata": {
                        "agent_loop": {
                            "schema_version": 1,
                            "loop_run_id": "run-2",
                            "turn_index": 2,
                            "goal_blackboard": {"status": "stale"},
                            "goal_updates": {"status": "complete"},
                        },
                    },
                },
            ],
            tenant_id="tenant-a",
            session_id="session-a",
        )


def test_rejects_malformed_committed_goal_updates():
    with pytest.raises(ValueError, match="Invalid agent-loop goal state"):
        apply_committed_agent_loop_progress(
            {},
            metadata={
                "request_id": "request-2",
                "agent_loop": {
                    "schema_version": 1,
                    "loop_run_id": "run-2",
                    "turn_index": 1,
                    "goal_updates": ["not", "a", "mapping"],
                },
            },
            identity=AgentLoopCommitIdentity(
                tenant_id="tenant-a",
                session_id="session-a",
                trace_id="trace-2",
                job_id="job-2",
            ),
        )


def test_success_post_turn_persists_agent_loop_progress_through_session_state_merge():
    session = {"state": {}}
    memory_hierarchy = SimpleNamespace(
        promote_turn=Mock(),
        lifecycle_maintenance=Mock(),
    )
    control_plane = SimpleNamespace(
        registry=SimpleNamespace(get_or_create=Mock(return_value=session)),
        _promote_semantic_memory=Mock(),
        _semantic_memory_graph=SimpleNamespace(update_from_turn=Mock()),
        _adaptation_engine=SimpleNamespace(record_outcome=Mock()),
        _belief_state_engine=SimpleNamespace(update_from_turn=Mock()),
        _update_identity_state=Mock(return_value={}),
        _update_felt_persona_stream=Mock(return_value={}),
        _update_conversation_trajectory=Mock(return_value={}),
        _memory_hierarchy_manager=memory_hierarchy,
        _planning_optimizer=SimpleNamespace(record_plan_outcome=Mock()),
        _interactive_cognition_ui=None,
        _merge_session_state=Mock(),
        _emit_runtime_stream_event=Mock(),
        _topology_record_node=Mock(),
    )
    job = SimpleNamespace(
        tenant_id="tenant-a",
        session_id="session-a",
        trace_id="trace-1",
        job_id="job-1",
        user_input="continue",
        metadata={
            "request_id": "request-1",
            "agent_loop": {
                "schema_version": 1,
                "loop_run_id": "run-1",
                "turn_index": 1,
                "goal_updates": {"status": "complete"},
            },
            "tool_outcome_evidence": _tool_outcome_evidence("trace-1"),
        },
    )

    _apply_submit_success_postprocessing_impl(
        control_plane,
        job=job,
        result=("done", True),
        session_key="session-a",
        trace_token="trace-1",
        loop_iterations=1,
    )

    merged_state = control_plane._merge_session_state.call_args.kwargs["state_patch"]
    assert read_agent_loop_goal_blackboard(merged_state) == {"status": "complete"}
    assert read_agent_loop_tool_outcome_evidence(merged_state) == _tool_outcome_evidence("trace-1")
    assert control_plane._merge_session_state.call_args.kwargs["tenant_id"] == "tenant-a"
