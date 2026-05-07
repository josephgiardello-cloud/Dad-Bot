from __future__ import annotations

import pytest

from tools.dadbot_2026_compliance_stress import run_suite

pytestmark = pytest.mark.unit


def test_2026_compliance_suite_passes_all_scenarios() -> None:
    payload = run_suite()

    assert str(payload.get("overall_status") or "") == "PASS"
    results = list(payload.get("results") or [])
    assert len(results) == 3
    by_name = {str(item.get("compliance_test") or ""): item for item in results}

    split_brain = dict(by_name.get("Memory_Integrity_Divergence") or {})
    split_evidence = dict(split_brain.get("evidence") or {})
    assert str(split_brain.get("status") or "") == "PASS"
    assert split_evidence.get("halt_intercepted") is True
    assert str(split_evidence.get("ledger_hash") or "")
    assert str(split_evidence.get("projection_hash") or "")
    assert str(split_evidence.get("ledger_hash") or "") != str(split_evidence.get("projection_hash") or "")

    ghost = dict(by_name.get("Permission_Ghost_Idempotency") or {})
    ghost_evidence = dict(ghost.get("evidence") or {})
    assert str(ghost.get("status") or "") == "PASS"
    assert ghost_evidence.get("fresh_call_forced") is True
    assert str(ghost_evidence.get("restricted_idempotency_key") or "")
    assert str(ghost_evidence.get("admin_idempotency_key") or "")
    assert str(ghost_evidence.get("restricted_idempotency_key") or "") != str(
        ghost_evidence.get("admin_idempotency_key") or ""
    )

    facet = dict(by_name.get("Conflicting_Facet_Safety_Override") or {})
    facet_evidence = dict(facet.get("evidence") or {})
    assert str(facet.get("status") or "") == "PASS"
    assert str(facet_evidence.get("policy_rule") or "") == "enforce_policies"
    assert facet_evidence.get("output_mutated") is True
    assert str(facet_evidence.get("candidate_hash") or "")
    assert str(facet_evidence.get("output_hash") or "")
    assert str(facet_evidence.get("candidate_hash") or "") != str(facet_evidence.get("output_hash") or "")
