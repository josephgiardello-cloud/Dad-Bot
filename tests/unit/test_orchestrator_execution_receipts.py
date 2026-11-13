from __future__ import annotations

from copy import deepcopy
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


@pytest.mark.parametrize("damage", ["signature", "trace", "stage", "sequence", "key", "extra", "nan"])
def test_resume_reverifies_receipts_instead_of_trusting_verified_label(damage):
    from dadbot.core.agent_loop_recovery import verify_completion_receipts
    from dadbot.core.execution_receipt import ReceiptSigner

    signer = ReceiptSigner(b"test-receipt-key-for-recovery-00001")
    receipt = signer.sign(
        turn_id="trace", stage="save", sequence=1, stage_call_id="call",
        checkpoint_hash="a" * 64, prev_receipt_sig="",
    ).to_dict()
    envelope = {
        "schema_version": 1, "trace_id": "trace", "key_id": signer.key_id,
        "stages": ["save"], "receipts": [receipt],
    }
    completion = {"trace_id": "trace", "metadata": {
        "execution_receipt_status": "verified", "execution_receipt_evidence": envelope,
    }}
    verify_completion_receipts(deepcopy(completion), signer)
    if damage == "signature":
        receipt["signature"] = "0" * 64
    elif damage == "trace":
        receipt["turn_id"] = "foreign-trace"
    elif damage == "stage":
        envelope["stages"] = ["execute"]
    elif damage == "sequence":
        receipt["sequence"] = True
    elif damage == "key":
        envelope["key_id"] = "foreign-key"
    elif damage == "extra":
        receipt["raw_output"] = "not receipt evidence"
    else:
        receipt["completed_at"] = float("nan")
    with pytest.raises(ValueError, match="receipt"):
        verify_completion_receipts(completion, signer)
