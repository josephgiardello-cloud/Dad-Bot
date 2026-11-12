from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dadbot.core.orchestrator import DadBotOrchestrator

pytestmark = pytest.mark.unit


def _finalizer_stub() -> SimpleNamespace:
    return SimpleNamespace(
        _tool_action_audit_from_context=Mock(return_value=[]),
        _emit_sovereign_event=Mock(),
        _checkpointer=None,
        _update_session_state_after_turn=Mock(),
        _publish_health_evidence=Mock(),
    )


def _finalize(context: SimpleNamespace, job: SimpleNamespace) -> None:
    DadBotOrchestrator._finalize_orchestrator_success(
        _finalizer_stub(),
        context=context,
        job=job,
        session={},
        result=("reply", False),
        manifest={},
        loaded_checkpoint_anchor="",
    )


def test_verified_receipts_are_attached_to_loop_completion_metadata() -> None:
    evidence = {
        "schema_version": 1,
        "trace_id": "trace",
        "key_id": "0123456789abcdef",
        "stages": ["save"],
        "receipts": [{"stage": "save", "signature": "signed"}],
    }
    context = SimpleNamespace(
        trace_id="trace",
        metadata={
            "execution_receipt_status": "verified",
            "execution_receipt_evidence": evidence,
        },
        state={},
    )
    job = SimpleNamespace(metadata={"agent_loop": {"schema_version": 1}})

    _finalize(context, job)

    assert job.metadata["execution_receipt_status"] == "verified"
    assert job.metadata["execution_receipt_evidence"] == evidence
    assert job.metadata["execution_receipt_evidence"] is not evidence


def test_unavailable_receipts_are_not_attached_to_loop_completion() -> None:
    context = SimpleNamespace(
        trace_id="trace",
        metadata={"execution_receipt_status": "unavailable_unstable_signer"},
        state={},
    )
    job = SimpleNamespace(
        metadata={
            "agent_loop": {"schema_version": 1},
            "execution_receipt_evidence": {"stale": True},
        },
    )

    _finalize(context, job)

    assert job.metadata["execution_receipt_status"] == "unavailable_unstable_signer"
    assert "execution_receipt_evidence" not in job.metadata


def test_malformed_verified_receipts_fail_before_completion_metadata_is_persisted() -> None:
    context = SimpleNamespace(
        trace_id="trace",
        metadata={
            "execution_receipt_status": "verified",
            "execution_receipt_evidence": {
                "schema_version": 1,
                "trace_id": "trace",
                "key_id": "0123456789abcdef",
                "stages": ["save"],
                "receipts": [{"stage": "inference"}],
            },
        },
        state={},
    )
    job = SimpleNamespace(metadata={"agent_loop": {"schema_version": 1}})

    with pytest.raises(RuntimeError, match="receipt evidence is malformed"):
        _finalize(context, job)
