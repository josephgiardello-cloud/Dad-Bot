from __future__ import annotations

from typing import Any

CAPABILITY_CONTRACTS: dict[str, dict[str, Any]] = {
    "temporal_ordering": {
        "required_stages": ["temporal", "inference", "save"],
        "invariant": "strict ordering",
        "runtime_enforcement": True,
        "test_enforcement": True,
    },
    "mutation_safety": {
        "rule": "all mutations must pass SaveNode",
        "runtime_enforcement": True,
        "test_enforcement": True,
    },
    "deterministic_replay": {
        "rule": "identical inputs => identical ledger hash",
        "runtime_enforcement": False,
        "test_enforcement": True,
    },
    "save_node_single_execution": {
        "rule": "SaveNode executes exactly once per completed turn",
        "runtime_enforcement": True,
        "test_enforcement": True,
    },
    "capability_audit_emission": {
        "rule": "audit_mode emits a structured capability audit report",
        "runtime_enforcement": True,
        "test_enforcement": True,
    },
}


def capability_contracts() -> dict[str, dict[str, Any]]:
    return {name: dict(contract) for name, contract in CAPABILITY_CONTRACTS.items()}
