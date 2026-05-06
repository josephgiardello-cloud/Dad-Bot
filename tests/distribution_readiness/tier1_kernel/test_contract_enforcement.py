"""Tier 1 — Contract Enforcement: node input/output schema + TurnContext boundary.

Hard-fail: missing required fields at node edges indicate contract regression.
"""
from __future__ import annotations

import pytest

from dadbot.core.graph import TurnGraph, _NODE_STAGE_CONTRACTS
from dadbot.core.graph_context import TurnContext, TurnContextContractViolation
from dadbot.core.execution_context import _require_execution_context_contract


pytestmark = pytest.mark.unit


class TestNodeStageContracts:
    def test_contract_map_covers_canonical_stages(self):
        """All canonical production stages must have contracts registered."""
        required_stages = {"inference", "safety", "save", "temporal", "context_builder"}
        assert required_stages.issubset(set(_NODE_STAGE_CONTRACTS.keys()))

    def test_contract_map_entry_shape_is_input_output_tuple(self):
        """Every stage contract entry must expose input/output key tuples."""
        for stage_name, contract in _NODE_STAGE_CONTRACTS.items():
            assert isinstance(stage_name, str)
            assert isinstance(contract, tuple)
            assert len(contract) == 2
            input_keys, output_keys = contract
            assert isinstance(input_keys, tuple)
            assert isinstance(output_keys, tuple)

    def test_execute_stamps_contract_version_in_determinism_manifest(self):
        """Graph execution must stamp node-contract hash into determinism manifest."""

        class _SafeNode:
            async def run(self, ctx):
                ctx.state["safe_result"] = ("ok", False)
                return ctx

        graph = TurnGraph(registry=None, nodes=[_SafeNode()])
        ctx = TurnContext(user_input="contract version stamp")
        result = __import__("asyncio").run(graph.execute(ctx))

        assert result == ("ok", False)
        version_blob = dict(ctx.determinism_manifest.get("contract_version") or {})
        assert version_blob.get("schema_version") == "1"
        assert isinstance(version_blob.get("node_contracts_hash"), str)
        assert len(str(version_blob.get("node_contracts_hash") or "")) == 16


class TestExecutionContextContractEnforcement:
    def test_non_dict_state_raises_type_error(self):
        """_require_execution_context_contract must reject non-dict state."""
        from types import SimpleNamespace
        ctx = SimpleNamespace(state="not a dict", metadata={})
        with pytest.raises(TypeError, match="context.state must be a dict"):
            _require_execution_context_contract(ctx)

    def test_non_dict_metadata_raises_type_error(self):
        """_require_execution_context_contract must reject non-dict metadata."""
        from types import SimpleNamespace
        ctx = SimpleNamespace(state={}, metadata=["not", "a", "dict"])
        with pytest.raises(TypeError, match="context.metadata must be a dict"):
            _require_execution_context_contract(ctx)

    def test_dict_state_and_metadata_passes(self):
        """_require_execution_context_contract must pass when both are dicts."""
        from types import SimpleNamespace
        ctx = SimpleNamespace(state={"k": "v"}, metadata={"m": 1})
        state, metadata = _require_execution_context_contract(ctx)
        assert state == {"k": "v"}
        assert metadata == {"m": 1}


class TestTurnContextConstructionContract:
    def test_non_dict_state_raises_at_construction(self):
        """TurnContext must reject a non-dict state at construction time."""
        with pytest.raises(TurnContextContractViolation, match="state must be a dict"):
            TurnContext(user_input="test", state="not a dict")  # type: ignore[arg-type]

    def test_non_dict_metadata_raises_at_construction(self):
        """TurnContext must reject a non-dict metadata at construction time."""
        with pytest.raises(TurnContextContractViolation, match="metadata must be a dict"):
            TurnContext(user_input="test", metadata=["bad"])  # type: ignore[arg-type]

    def test_non_dict_determinism_manifest_raises_at_construction(self):
        """TurnContext must reject a non-dict determinism_manifest at construction time."""
        with pytest.raises(TurnContextContractViolation, match="determinism_manifest must be a dict"):
            TurnContext(user_input="test", determinism_manifest="bad")  # type: ignore[arg-type]

    def test_valid_construction_does_not_raise(self):
        """TurnContext constructed with valid dicts must not raise."""
        ctx = TurnContext(
            user_input="ok",
            state={"k": "v"},
            metadata={"m": 1},
            determinism_manifest={"seed": "42"},
        )
        assert ctx.state == {"k": "v"}
        assert ctx.metadata == {"m": 1}

    def test_default_construction_always_valid(self):
        """Default field factories always produce dicts — construction must pass."""
        ctx = TurnContext(user_input="default check")
        assert isinstance(ctx.state, dict)
        assert isinstance(ctx.metadata, dict)
        assert isinstance(ctx.determinism_manifest, dict)

    def test_turn_context_contract_violation_is_type_error_subclass(self):
        """TurnContextContractViolation must be a TypeError subclass for isinstance compatibility."""
        assert issubclass(TurnContextContractViolation, TypeError)
