"""Phase 4.3 — GraphSemanticValidator tests."""

from __future__ import annotations

from dadbot.core.semantic_graph_validator import (
    GraphSemanticValidator,
    NodeIntentContract,
    build_dadbot_semantic_validator,
)

# ---------------------------------------------------------------------------
# NodeIntentContract / GraphSemanticValidator unit tests
# ---------------------------------------------------------------------------


def _make_validator():
    v = GraphSemanticValidator()
    v.register_node_contract(
        NodeIntentContract(
            node_name="a",
            output_provides=["x", "y"],
        )
    )
    v.register_node_contract(
        NodeIntentContract(
            node_name="b",
            input_intent_schema={"x": "required from a"},
            output_provides=["z"],
        )
    )
    v.register_node_contract(
        NodeIntentContract(
            node_name="c",
            input_intent_schema={"z": "required from b", "y": "required from a"},
            output_provides=["done"],
        )
    )
    return v


def test_validate_graph_no_violations():
    v = _make_validator()
    violations = v.validate_graph(["a", "b", "c"])
    assert violations == []


def test_validate_graph_missing_input_key():
    v = GraphSemanticValidator()
    v.register_node_contract(
        NodeIntentContract(
            node_name="a",
            output_provides=["x"],  # doesn't provide "y"
        )
    )
    v.register_node_contract(
        NodeIntentContract(
            node_name="b",
            input_intent_schema={"x": "ok", "y": "missing"},
            output_provides=["z"],
        )
    )
    violations = v.validate_graph(["a", "b"])
    assert len(violations) == 1
    assert violations[0].missing_key == "y"
    assert violations[0].downstream_node == "b"
    assert violations[0].violation_type == "unsatisfied_input"


def test_validate_graph_cumulative_satisfies_later_node():
    """Node c requires 'x' which a provides, even though b is between them."""
    v = GraphSemanticValidator()
    v.register_node_contract(
        NodeIntentContract(
            node_name="a",
            output_provides=["x"],
        )
    )
    v.register_node_contract(
        NodeIntentContract(
            node_name="b",
            output_provides=["y"],
        )
    )
    v.register_node_contract(
        NodeIntentContract(
            node_name="c",
            input_intent_schema={"x": "from a", "y": "from b"},
            output_provides=["done"],
        )
    )
    violations = v.validate_graph(["a", "b", "c"])
    assert violations == []


def test_validate_graph_missing_contract_reports_violation():
    v = GraphSemanticValidator()
    # Only register 'a'; 'b' has no contract
    v.register_node_contract(NodeIntentContract(node_name="a", output_provides=["x"]))
    violations = v.validate_graph(["a", "b"])
    assert any(viol.violation_type == "missing_contract" for viol in violations)


def test_validate_edge_compatible():
    v = _make_validator()
    violations = v.validate_edge("a", "b")
    assert violations == []


def test_validate_edge_missing_output():
    v = GraphSemanticValidator()
    v.register_node_contract(
        NodeIntentContract(
            node_name="a",
            output_provides=[],  # provides nothing
        )
    )
    v.register_node_contract(
        NodeIntentContract(
            node_name="b",
            input_intent_schema={"x": "required"},
        )
    )
    violations = v.validate_edge("a", "b")
    assert len(violations) == 1
    assert violations[0].missing_key == "x"


def test_validate_edge_missing_upstream_contract():
    v = GraphSemanticValidator()
    v.register_node_contract(NodeIntentContract(node_name="b", input_intent_schema={}))
    violations = v.validate_edge("a", "b")
    assert any(viol.violation_type == "missing_contract" for viol in violations)


def test_register_node_contract_and_retrieve():
    v = GraphSemanticValidator()
    c = NodeIntentContract(node_name="x", output_provides=["k"])
    v.register_node_contract(c)
    assert v.get_contract("x") is c
    assert "x" in v.registered_nodes()


def test_validate_graph_empty_order():
    v = GraphSemanticValidator()
    violations = v.validate_graph([])
    assert violations == []


# ---------------------------------------------------------------------------
# DadBot pre-wired semantic validator
# ---------------------------------------------------------------------------


def test_dadbot_validator_all_nodes_registered():
    v = build_dadbot_semantic_validator()
    expected = {"temporal", "preflight", "planner", "inference", "safety", "reflection", "save"}
    assert expected.issubset(set(v.registered_nodes()))


def test_dadbot_validator_full_pipeline_no_violations():
    """Full DadBot pipeline node order should have zero semantic violations."""
    v = build_dadbot_semantic_validator()
    # Standard pipeline order (no tools)
    pipeline = ["temporal", "preflight", "planner", "inference", "safety", "reflection", "save"]
    violations = v.validate_graph(pipeline)
    assert violations == [], "Unexpected violations:\n" + "\n".join(v.detail for v in violations)


def test_dadbot_validator_skipped_temporal_causes_violation():
    """Removing temporal from the pipeline should cause downstream violations."""
    v = build_dadbot_semantic_validator()
    pipeline = ["preflight", "planner", "inference", "safety", "reflection", "save"]
    violations = v.validate_graph(pipeline)
    missing_temporal = [viol for viol in violations if "temporal_axis" in (viol.missing_key or "")]
    assert len(missing_temporal) >= 1


def test_dadbot_validator_temporal_provides_correct_class():
    v = build_dadbot_semantic_validator()
    contract = v.get_contract("temporal")
    assert contract is not None
    assert contract.output_expectation_class == "TurnTemporalAxis"
    assert "temporal_axis" in contract.output_provides


def test_dadbot_validator_save_mutation_constraints():
    v = build_dadbot_semantic_validator()
    contract = v.get_contract("save")
    assert contract is not None
    assert "MutationQueue.drain_only" in contract.mutation_constraints
    assert "save_node_single_execution" in contract.mutation_constraints
