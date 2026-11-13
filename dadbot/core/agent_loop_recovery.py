from __future__ import annotations

import hashlib
import math
import re
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from dadbot.core.durability import FileLockMutex
from dadbot.core.execution_receipt import ExecutionReceipt, ReceiptChain, ReceiptSigner
from dadbot.core.ledger_backend import FileWALLedgerBackend
from dadbot.core.session_store import SessionStore
from dadbot.core.system_state_algebra import restore_system_state_projections
from dadbot_system.delegation import require_tenant_authority_if_active


def validate_agent_loop_policy(policy: Any) -> None:
    if type(policy) is not dict or set(policy) != {
        "max_turns", "max_failures", "max_consecutive_noop", "allowed_tools", "trigger_type",
    }:
        raise ValueError("Invalid canonical loop cursor policy")
    for name, minimum in (("max_turns", 1), ("max_failures", 0), ("max_consecutive_noop", 1)):
        if type(policy[name]) is not int or policy[name] < minimum:
            raise ValueError("Invalid canonical loop cursor policy budget")
    tools = policy["allowed_tools"]
    if (
        (tools is not None and (
            type(tools) is not list or any(type(tool) is not str or not tool.strip() for tool in tools)
        ))
        or type(policy["trigger_type"]) is not str or not policy["trigger_type"].strip()
    ):
        raise ValueError("Invalid canonical loop cursor policy tools or trigger")


def project_agent_loop_session(control_plane: Any, *, tenant_id: str, session_id: str) -> dict[str, Any]:
    state = SessionStore.projected_from_ledger(
        ledger=control_plane.ledger,
    ).get(session_id, tenant_id=tenant_id) or {}
    restore_system_state_projections(state)
    return state


def supports_agent_loop_recovery(backend: Any) -> bool:
    return type(backend) is FileWALLedgerBackend


@dataclass(slots=True)
class AgentLoopCursor:
    tenant_id: str
    session_id: str
    loop_run_id: str
    input_hash: str
    policy: dict[str, Any]
    next_turn: int = 1
    completed_turns: int = 0
    failures: int = 0
    consecutive_noop: int = 0
    last_request_id: str = ""
    pending_request_id: str = ""
    last_action_input: str = ""
    system_observation: str = ""
    stop_reason: str = ""
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def read(
        cls, value: Any, *, tenant_id: str, session_id: str,
        initial_input: str | None, policy: dict[str, Any] | None,
    ) -> AgentLoopCursor:
        if (
            type(value) is not dict or set(value) != set(cls.__dataclass_fields__)
            or not isinstance(policy, dict)
        ):
            raise ValueError("Invalid canonical agent-loop cursor")
        cursor = cls(**deepcopy(value))
        validate_agent_loop_policy(cursor.policy)
        if (
            type(cursor.schema_version) is not int or cursor.schema_version != 1
            or cursor.tenant_id != tenant_id or cursor.session_id != session_id
            or (initial_input is not None and (
                cursor.input_hash != hashlib.sha256(initial_input.encode("utf-8")).hexdigest()
            ))
            or cursor.policy != policy
        ):
            raise ValueError("Canonical loop cursor scope, input, or policy mismatch")
        for name in (
            "loop_run_id", "input_hash", "last_request_id", "pending_request_id",
            "last_action_input", "system_observation", "stop_reason",
        ):
            if type(getattr(cursor, name)) is not str or len(getattr(cursor, name)) > 16384:
                raise ValueError("Invalid canonical loop cursor text")
        if not cursor.loop_run_id or re.fullmatch("[0-9a-f]{64}", cursor.input_hash) is None:
            raise ValueError("Canonical loop cursor has no run identity")
        for name in ("next_turn", "completed_turns", "failures", "consecutive_noop"):
            if type(getattr(cursor, name)) is not int or getattr(cursor, name) < 0:
                raise ValueError("Invalid canonical loop cursor budget")
        if (
            not 1 <= cursor.next_turn <= policy["max_turns"] + 1
            or cursor.completed_turns + cursor.failures > cursor.next_turn - 1
            or cursor.consecutive_noop > cursor.next_turn - 1
            or (bool(cursor.last_request_id) != bool(cursor.completed_turns))
            or (cursor.pending_request_id and (
                cursor.stop_reason or cursor.next_turn > policy["max_turns"]
            ))
            or cursor.stop_reason not in {
                "", "kernel_should_end", "noop_threshold", "max_turns", "reflection_stop", "failure_budget",
            }
        ):
            raise ValueError("Inconsistent canonical loop cursor")
        return cursor


def recover_agent_loop_cursor(cursor: AgentLoopCursor, index: Any, signer: ReceiptSigner) -> str:
    request_id = cursor.pending_request_id or cursor.last_request_id
    if not request_id:
        return ""
    completion = index.completed_metadata(
        session_id=cursor.session_id, request_id=request_id, tenant_id=cursor.tenant_id,
    )
    if completion is None:
        raise RuntimeError("Interrupted agent-loop dispatch is uncertain; automatic resend is forbidden")
    verify_completion_receipts(completion, signer)
    loop = completion["metadata"].get("agent_loop")
    if (
        type(loop) is not dict or loop.get("loop_run_id") != cursor.loop_run_id
        or type(loop.get("turn_index")) is not int
        or (cursor.pending_request_id and loop["turn_index"] != cursor.next_turn)
        or (not cursor.pending_request_id and loop["turn_index"] >= cursor.next_turn)
    ):
        raise ValueError("Canonical loop cursor does not match its completion")
    outcome = index.completed_result(
        session_id=cursor.session_id, request_id=request_id, tenant_id=cursor.tenant_id,
    )
    if (
        not isinstance(outcome, tuple) or len(outcome) != 2
        or type(outcome[0]) is not str or type(outcome[1]) is not bool
    ):
        raise ValueError("Canonical loop completion has no valid recorded result")
    initial_reply, should_end = outcome
    if cursor.pending_request_id:
        cursor.last_request_id = cursor.pending_request_id
        cursor.pending_request_id = ""
        cursor.next_turn += 1
        cursor.completed_turns += 1
        cursor.consecutive_noop = 0 if initial_reply.strip() else cursor.consecutive_noop + 1
        cursor.system_observation = ""
        if should_end:
            cursor.stop_reason = "kernel_should_end"
        elif cursor.consecutive_noop >= cursor.policy["max_consecutive_noop"]:
            cursor.stop_reason = "noop_threshold"
        elif cursor.next_turn > cursor.policy["max_turns"]:
            cursor.stop_reason = "max_turns"
    return initial_reply


def verify_completion_receipts(completion: dict[str, Any], signer: ReceiptSigner) -> None:
    metadata = completion.get("metadata")
    evidence = metadata.get("execution_receipt_evidence") if type(metadata) is dict else None
    if not signer.is_stable:
        raise RuntimeError("Loop resume requires a stable execution receipt key")
    if (
        type(metadata) is not dict or metadata.get("execution_receipt_status") != "verified"
        or type(evidence) is not dict
        or set(evidence) != {"schema_version", "trace_id", "key_id", "stages", "receipts"}
        or type(evidence["schema_version"]) is not int or evidence["schema_version"] != 1
        or evidence["trace_id"] != completion.get("trace_id")
        or evidence["key_id"] != signer.key_id
        or type(evidence["stages"]) is not list or not 1 <= len(evidence["stages"]) <= 64
        or any(type(stage) is not str or not stage or len(stage) > 120 for stage in evidence["stages"])
        or type(evidence["receipts"]) is not list
        or len(evidence["receipts"]) != len(evidence["stages"])
    ):
        raise ValueError("Canonical loop completion receipt envelope is unavailable or invalid")
    receipts = []
    fields = {
        "turn_id", "stage", "sequence", "stage_call_id", "checkpoint_hash",
        "prev_receipt_sig", "completed_at", "signature",
    }
    for value in evidence["receipts"]:
        if (
            type(value) is not dict or set(value) != fields
            or any(type(value[name]) is not str for name in fields - {"sequence", "completed_at"})
            or type(value["sequence"]) is not int
            or type(value["completed_at"]) not in (int, float)
            or not math.isfinite(value["completed_at"])
        ):
            raise ValueError("Invalid canonical execution receipt")
        receipts.append(ExecutionReceipt.from_dict(value))
    chain = ReceiptChain(receipts)
    if (
        [receipt.stage for receipt in receipts] != evidence["stages"]
        or chain.verify_continuity(
            signer, expected_stages=evidence["stages"], expected_turn_id=completion["trace_id"],
        )
    ):
        raise ValueError("Canonical loop execution receipt verification failed")


@contextmanager
def agent_loop_recovery_lock(backend: Any, *, tenant_id: str, session_id: str):
    require_tenant_authority_if_active(tenant_id)
    identity = hashlib.sha256(f"{tenant_id}\0{session_id}".encode()).hexdigest()
    if type(backend) is FileWALLedgerBackend:
        if not backend._fsync_enabled:
            raise RuntimeError("Canonical loop recovery requires fsync-enabled WAL")
        path = backend.path()
        with FileLockMutex(path.with_name(f"{path.name}.loop-{identity}.lock")).locked():
            require_tenant_authority_if_active(tenant_id)
            yield
    else:
        raise RuntimeError("Canonical loop resume requires File-WAL; non-file recovery is not certified")
