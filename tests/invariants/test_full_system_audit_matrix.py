from __future__ import annotations

from pathlib import Path

import pytest

from dadbot.core.decision_snapshot import DecisionSnapshot
from tools.audit_runner import run_full_system_audit

pytestmark = pytest.mark.unit


ROOT = Path(__file__).resolve().parents[2]


def test_decision_snapshot_is_immutable_after_capture() -> None:
    ctx = {
        "memory": {"topic": "trust", "score": 0.4},
        "state": {"phase": "inference"},
    }
    snapshot = DecisionSnapshot.capture(
        trace_id="trace-1",
        memory=ctx["memory"],
        context=ctx["state"],
        metadata={"stage": "score"},
    )

    ctx["memory"]["score"] = 0.9
    ctx["state"]["phase"] = "post"

    payload = snapshot.to_payload()
    assert payload["memory"]["score"] == 0.4
    assert payload["context"]["phase"] == "inference"
    assert snapshot.verify_hash_lock() is True


def test_decision_snapshot_hash_lock_detects_tamper() -> None:
    snapshot = DecisionSnapshot.capture(
        trace_id="trace-2",
        memory={"a": 1},
        context={"b": 2},
        metadata={"stage": "freeze"},
    )

    # Simulate post-snapshot mutation by rebuilding with modified payload.
    tampered = DecisionSnapshot.capture(
        trace_id=snapshot.trace_id,
        memory={"a": 999},
        context={"b": 2},
        metadata={"stage": "freeze"},
    )
    assert tampered.snapshot_hash != snapshot.snapshot_hash


def test_full_system_audit_matrix_contract() -> None:
    payload = run_full_system_audit(strict=False, no_shadow_paths=True)

    invariant_ids = {item["id"] for item in payload["invariants"]}
    assert {"A1", "A2", "A3", "B1", "B2", "C1", "D1", "D2", "E1", "E2", "F1", "G1", "H1", "H2"}.issubset(
        invariant_ids
    )
    assert "shadow_path_map" in payload
    assert "mutation_surface_map" in payload
    assert "determinism_score" in payload
    assert "closure_score" in payload


def test_runtime_audit_command_shape() -> None:
    report_path = ROOT / "artifacts" / "system_audit_matrix_report.json"
    payload = run_full_system_audit(strict=True, no_shadow_paths=True)

    assert report_path.exists()
    assert isinstance(payload["summary"]["total"], int)
    assert isinstance(payload["overall_pass"], bool)
