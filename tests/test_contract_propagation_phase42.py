"""Phase 4.2 — ContractPropagationMap tests."""

from __future__ import annotations

import pytest

from dadbot.core.contract_evaluator import (
    ContractNode,
    ContractPropagationMap,
    build_dadbot_contract_map,
    validate_sovereign_ledger_transition,
)
from dadbot.core.contracts_adapter import ContractViolationError

pytestmark = pytest.mark.phase4

# ---------------------------------------------------------------------------
# ContractNode / ContractPropagationMap unit tests
# ---------------------------------------------------------------------------


def test_register_and_get():
    cmap = ContractPropagationMap()
    node = ContractNode(contract_id="a", version="1.0.0")
    cmap.register(node)
    assert cmap.get("a") is node


def test_mark_changed_triggers_downstream():
    called = []

    def _val_b() -> list[str]:
        called.append("b")
        return []

    def _val_c() -> list[str]:
        called.append("c")
        return []

    cmap = ContractPropagationMap()
    cmap.register(
        ContractNode(
            contract_id="a",
            version="1.0.0",
            downstream_consumers=["b"],
        )
    )
    cmap.register(
        ContractNode(
            contract_id="b",
            version="1.0.0",
            upstream_dependencies=["a"],
            downstream_consumers=["c"],
            validator_fn=_val_b,
        )
    )
    cmap.register(
        ContractNode(
            contract_id="c",
            version="1.0.0",
            upstream_dependencies=["b"],
            validator_fn=_val_c,
        )
    )

    results = cmap.mark_changed("a")
    # Should visit a, b, c in BFS order
    assert [r.contract_id for r in results] == ["a", "b", "c"]
    assert "b" in called
    assert "c" in called


def test_mark_changed_returns_violations():
    cmap = ContractPropagationMap()
    cmap.register(
        ContractNode(
            contract_id="broken",
            version="1.0.0",
            validator_fn=lambda: ["something is wrong"],
        )
    )
    results = cmap.mark_changed("broken")
    assert len(results) == 1
    assert not results[0].valid
    assert "something is wrong" in results[0].violations


def test_mark_changed_valid_result():
    cmap = ContractPropagationMap()
    cmap.register(
        ContractNode(
            contract_id="ok",
            version="1.0.0",
            validator_fn=lambda: [],
        )
    )
    results = cmap.mark_changed("ok")
    assert results[0].valid


def test_mark_changed_unknown_contract_returns_empty():
    cmap = ContractPropagationMap()
    results = cmap.mark_changed("nonexistent")
    assert results == []


def test_mark_changed_no_cycles_visited_once():
    """Diamond dependency — each node visited exactly once."""
    cmap = ContractPropagationMap()
    visit_counts: dict[str, int] = {"b": 0, "c": 0, "d": 0}

    def _counter(name: str):
        def _fn() -> list[str]:
            visit_counts[name] += 1
            return []

        return _fn

    cmap.register(ContractNode(contract_id="a", version="1", downstream_consumers=["b", "c"]))
    cmap.register(ContractNode(contract_id="b", version="1", downstream_consumers=["d"], validator_fn=_counter("b")))
    cmap.register(ContractNode(contract_id="c", version="1", downstream_consumers=["d"], validator_fn=_counter("c")))
    cmap.register(ContractNode(contract_id="d", version="1", validator_fn=_counter("d")))

    cmap.mark_changed("a")
    assert visit_counts == {"b": 1, "c": 1, "d": 1}


def test_revalidate_all_topological_order():
    order_observed: list[str] = []
    cmap = ContractPropagationMap()
    for cid in ["c", "a", "b"]:
        cmap.register(
            ContractNode(
                contract_id=cid,
                version="1",
                upstream_dependencies={"b": ["a"], "c": ["b"], "a": []}.get(cid, []),
                validator_fn=lambda _n=cid: order_observed.append(_n) or [],
            )
        )

    cmap.revalidate_all()
    assert order_observed == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# DadBot pre-wired contract map
# ---------------------------------------------------------------------------


def test_build_dadbot_contract_map_registered_nodes():
    cmap = build_dadbot_contract_map()
    ids = set(cmap.all_ids())
    assert ids == {
        "runtime_contract",
        "persistence_contract",
        "graph_integrity_contract",
        "determinism_boundary_contract",
    }


def test_build_dadbot_contract_map_downstream_chain():
    cmap = build_dadbot_contract_map()
    results = cmap.mark_changed("runtime_contract")
    visited = [r.contract_id for r in results]
    # Should propagate through all 4 in order
    assert visited == [
        "runtime_contract",
        "persistence_contract",
        "graph_integrity_contract",
        "determinism_boundary_contract",
    ]


def test_build_dadbot_contract_map_validators_pass():
    """All built-in validators should report no violations in a clean environment."""
    cmap = build_dadbot_contract_map()
    results = cmap.revalidate_all()
    failed = [r for r in results if not r.valid]
    assert failed == [], f"Unexpected violations: {[(r.contract_id, r.violations) for r in failed]}"


def test_chaos_bad_ledger_payload_raises_contract_violation() -> None:
    before = {
        "session_id": "s1",
        "trace_id": "t1",
        "execution_mode": "live",
        "execution_state": "running",
        "execution_status": "running",
        "causal_step_count": 1,
        "metadata": {},
    }
    after = {
        "session_id": "s1",
        "trace_id": "t1",
        "execution_mode": "live",
        "execution_state": "completed",
        "execution_status": "completed",
        "turn_truth_ok": True,
        "invariance_hash": "abc123",
        "causal_step_count": 2,
        "metadata": {
            "ledger_mutations": [
                {
                    "op": "append_history",
                    "payload": "not-a-dict",
                    "source": "chaos_test",
                }
            ]
        },
    }

    with pytest.raises(ContractViolationError, match="ledger_mutation"):
        validate_sovereign_ledger_transition(before, after)
