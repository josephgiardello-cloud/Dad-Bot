from __future__ import annotations

from dadbot.core.contract_evaluator import evaluation_contract_payload
from tools import execution_invariance_audit as audit


def test_evaluation_contract_excludes_envelope_fields_from_behavioral_gate() -> None:
    contract = evaluation_contract_payload()["behavioral_invariance"]

    assert contract["signals"] == ["replay_hash", "tool_trace_hash"]
    assert "determinism_manifest_hash" in contract["must_not_depend_on"]
    assert "lock_hash" in contract["must_not_depend_on"]
    assert "PYTEST_CURRENT_TEST" in contract["must_not_depend_on"]


def test_envelope_detail_is_canonicalized_sorted_and_environment_stripped() -> None:
    runtime_mode = {
        "replay_hash": "replay-1",
        "tool_trace_hash": "tool-1",
        "determinism_manifest_hash": "manifest-runtime",
        "lock_hash": "lock-runtime",
        "mode": "runtime",
        "extra_runtime_only": "ignored",
    }
    test_mode = {
        "tool_trace_hash": "tool-1",
        "replay_hash": "replay-1",
        "lock_hash": "lock-test",
        "determinism_manifest_hash": "manifest-test",
        "mode": "test",
        "extra_test_only": "ignored",
    }

    detail = audit._canonical_envelope_detail(runtime_mode, test_mode)
    fields = detail["fields"]

    assert [item["field"] for item in fields] == ["determinism_manifest_hash", "lock_hash"]
    assert fields[0]["runtime"] == "manifest-runtime"
    assert fields[0]["test"] == "manifest-test"
    assert fields[1]["runtime"] == "lock-runtime"
    assert fields[1]["test"] == "lock-test"
    assert (
        detail["diagnostic_hash"]
        == audit._canonical_envelope_detail(
            dict(reversed(list(runtime_mode.items()))),
            dict(reversed(list(test_mode.items()))),
        )["diagnostic_hash"]
    )


def test_envelope_changes_do_not_affect_behavioral_gate() -> None:
    envelope_divergence = audit._evaluate_envelope_divergence(
        runtime_mode={
            "replay_hash": "same-replay",
            "tool_trace_hash": "same-tool-trace",
            "determinism_manifest_hash": "runtime-manifest",
            "lock_hash": "runtime-lock",
        },
        test_mode={
            "replay_hash": "same-replay",
            "tool_trace_hash": "same-tool-trace",
            "determinism_manifest_hash": "test-manifest",
            "lock_hash": "test-lock",
        },
    )

    payload = audit._build_audit_payload(
        trace_stability={"passed": True},
        orchestrator_determinism={"passed": True},
        envelope_divergence=envelope_divergence,
    )

    assert envelope_divergence["envelope_pass"] is False
    assert envelope_divergence["behavioral_cross_reference"]["replay_hash_identical"] is True
    assert envelope_divergence["behavioral_cross_reference"]["tool_trace_hash_identical"] is True
    assert payload["behavioral_invariance"]["passed"] is True
    assert payload["envelope_invariance"]["passed"] is False
    assert payload["overall_pass"] is True
