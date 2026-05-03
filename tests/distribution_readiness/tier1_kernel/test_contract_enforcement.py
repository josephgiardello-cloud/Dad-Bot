"""Tier 1 — Contract Enforcement: node input/output schema + TurnContext boundary.

Hard-fail: missing required fields at node edges indicate contract regression.
"""
from __future__ import annotations

import pytest

from dadbot.core.graph import NodeContractViolation, TurnGraph, _NODE_STAGE_CONTRACTS
from dadbot.core.graph_context import TurnContext, TurnContextContractViolation
from dadbot.core.execution_context import _require_execution_context_contract


pytestmark = pytest.mark.unit


class TestNodeStageContracts:
    # Shared instance with the full default pipeline (all canonical nodes).
    # Using an instance because _enforce_node_stage_contract is now an instance
    # method that checks self._pipeline_items() for prerequisite-aware enforcement.
    _graph: TurnGraph = TurnGraph(registry=None)

    def test_contract_map_covers_canonical_stages(self):
        """All canonical production stages must have contracts registered."""
        required_stages = {"inference", "safety", "save", "temporal", "context_builder"}
        assert required_stages.issubset(set(_NODE_STAGE_CONTRACTS.keys()))

    def test_inference_input_contract_requires_rich_context(self):
        """InferenceNode must fail the input contract if rich_context is absent."""
        ctx = TurnContext(user_input="missing rich_context")
        # rich_context intentionally absent
        with pytest.raises(NodeContractViolation, match="rich_context"):
            self._graph._enforce_node_stage_contract("inference", ctx, phase="input")

    def test_inference_output_contract_requires_candidate(self):
        """InferenceNode must fail the output contract if candidate was not written."""
        ctx = TurnContext(user_input="missing candidate after inference")
        ctx.state["rich_context"] = {}
        # candidate intentionally absent
        with pytest.raises(NodeContractViolation, match="candidate"):
            self._graph._enforce_node_stage_contract("inference", ctx, phase="output")

    def test_safety_input_contract_requires_candidate(self):
        """SafetyNode must fail the input contract if candidate is absent."""
        ctx = TurnContext(user_input="missing candidate before safety")
        with pytest.raises(NodeContractViolation, match="candidate"):
            self._graph._enforce_node_stage_contract("safety", ctx, phase="input")

    def test_safety_output_contract_requires_safe_result(self):
        """SafetyNode must fail the output contract if safe_result was not written."""
        ctx = TurnContext(user_input="missing safe_result after safety")
        ctx.state["candidate"] = ("reply", {})
        with pytest.raises(NodeContractViolation, match="safe_result"):
            self._graph._enforce_node_stage_contract("safety", ctx, phase="output")

    def test_save_input_contract_requires_safe_result(self):
        """SaveNode must fail the input contract if safe_result is absent."""
        ctx = TurnContext(user_input="missing safe_result before save")
        with pytest.raises(NodeContractViolation, match="safe_result"):
            self._graph._enforce_node_stage_contract("save", ctx, phase="input")

    def test_temporal_output_contract_requires_temporal_key(self):
        """TemporalNode must fail the output contract if temporal was not written."""
        ctx = TurnContext(user_input="missing temporal after TemporalNode")
        with pytest.raises(NodeContractViolation, match="temporal"):
            self._graph._enforce_node_stage_contract("temporal", ctx, phase="output")

    def test_unknown_stage_name_is_silently_skipped(self):
        """Stages without a registered contract must not raise — open extension."""
        ctx = TurnContext(user_input="unknown stage")
        # Should not raise
        self._graph._enforce_node_stage_contract("reflection", ctx, phase="input")
        self._graph._enforce_node_stage_contract("unknown_custom_node", ctx, phase="output")

    def test_satisfied_input_contract_does_not_raise(self):
        """When all required input keys are present, no exception is raised."""
        ctx = TurnContext(user_input="ok")
        ctx.state["rich_context"] = {"loaded": True}
        self._graph._enforce_node_stage_contract("inference", ctx, phase="input")

    def test_satisfied_output_contract_does_not_raise(self):
        """When all required output keys are present, no exception is raised."""
        ctx = TurnContext(user_input="ok")
        ctx.state["candidate"] = ("reply", {})
        self._graph._enforce_node_stage_contract("inference", ctx, phase="output")


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
