from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, cast

from dadbot.core.agent_loop_progress import (
    normalize_goal_blackboard,
    normalize_goal_updates,
    normalize_tool_outcome_evidence,
)

AgentLoopOutcomeStatus = Literal["satisfied", "not_satisfied", "unknown"]
AgentLoopOutcomeVerifier = Callable[["AgentLoopOutcomeEvidence"], AgentLoopOutcomeStatus]


@dataclass(frozen=True, slots=True)
class AgentLoopOutcomeEvidence:
    """Canonical, bounded evidence supplied to a deterministic goal verifier."""

    tenant_id: str
    session_id: str
    request_id: str
    job_id: str
    trace_id: str
    loop_run_id: str
    turn_index: int
    outcome_verifier_id: str
    should_end: bool
    goal_blackboard: dict[str, str]
    proposed_goal_updates: dict[str, str]
    tool_outcome_evidence: dict[str, Any]
    execution_receipt_evidence: dict[str, Any]


def build_agent_loop_outcome_evidence(
    completion: dict[str, Any],
    *,
    should_end: bool,
) -> AgentLoopOutcomeEvidence:
    if type(completion) is not dict:
        raise ValueError("Goal verification requires canonical completion metadata")
    tenant_id = completion.get("tenant_id")
    session_id = completion.get("session_id")
    request_id = completion.get("request_id")
    trace_id = completion.get("trace_id")
    job_id = completion.get("job_id")
    metadata = completion.get("metadata")
    if type(metadata) is not dict:
        raise ValueError("Goal verification completion metadata is malformed")
    loop = metadata.get("agent_loop")
    loop_run_id = loop.get("loop_run_id") if type(loop) is dict else None
    turn_index = loop.get("turn_index") if type(loop) is dict else None
    outcome_verifier_id = loop.get("outcome_verifier_id") if type(loop) is dict else None
    if (
        any(type(value) is not str or not value.strip() for value in (
            tenant_id, session_id, request_id, trace_id, job_id, loop_run_id, outcome_verifier_id,
        ))
        or type(loop) is not dict
        or type(loop.get("schema_version")) is not int or loop["schema_version"] != 1
        or type(turn_index) is not int or turn_index < 1
        or type(should_end) is not bool
    ):
        raise ValueError("Goal verification completion identity is invalid")
    receipt_evidence = metadata.get("execution_receipt_evidence")
    if type(receipt_evidence) is not dict:
        raise ValueError("Goal verification requires canonical execution receipt evidence")
    return AgentLoopOutcomeEvidence(
        tenant_id=tenant_id,
        session_id=session_id,
        request_id=request_id,
        job_id=job_id,
        trace_id=trace_id,
        loop_run_id=loop_run_id,
        turn_index=turn_index,
        outcome_verifier_id=outcome_verifier_id,
        should_end=should_end,
        goal_blackboard=normalize_goal_blackboard(loop.get("goal_blackboard")),
        proposed_goal_updates=normalize_goal_updates(loop.get("goal_updates")),
        tool_outcome_evidence=normalize_tool_outcome_evidence(
            metadata.get("tool_outcome_evidence"),
            expected_trace_id=trace_id,
        ),
        execution_receipt_evidence=deepcopy(receipt_evidence),
    )


def verify_agent_loop_outcome(
    verifier: AgentLoopOutcomeVerifier,
    evidence: AgentLoopOutcomeEvidence,
) -> AgentLoopOutcomeStatus:
    status = verifier(evidence)
    if type(status) is not str or status not in {"satisfied", "not_satisfied", "unknown"}:
        raise ValueError("Goal verifier must return satisfied, not_satisfied, or unknown")
    return cast(AgentLoopOutcomeStatus, status)


__all__ = [
    "AgentLoopOutcomeEvidence",
    "AgentLoopOutcomeStatus",
    "AgentLoopOutcomeVerifier",
    "build_agent_loop_outcome_evidence",
    "verify_agent_loop_outcome",
]
