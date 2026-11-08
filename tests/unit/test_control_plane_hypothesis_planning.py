from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dadbot.core.control_plane_planning import _prepare_submit_runtime_planning_impl
from dadbot.core.control_plane_submit_metadata import _prepare_submit_metadata_impl

pytestmark = pytest.mark.unit


def test_hypotheses_are_inferred_once_after_runtime_planning() -> None:
    hypotheses = [{"id": "h1", "label": "direct_intent_execution"}]
    hypothesis_engine = Mock()
    hypothesis_engine.infer_hypotheses.return_value = hypotheses
    session = {"state": {}}
    control_plane = SimpleNamespace(
        _prepare_global_confluence_law=Mock(),
        _discover_tool_specs=Mock(return_value=[]),
        _cognitive_policy_engine=SimpleNamespace(
            build_plan=Mock(return_value={"intent_type": "statement", "strategy": "direct_answer"}),
        ),
        _tool_routing_engine=SimpleNamespace(build_routing_plan=Mock(return_value={})),
        _compositional_tool_planner=SimpleNamespace(build_plan=Mock(return_value={})),
        _hypothesis_engine=hypothesis_engine,
        _semantic_safety_engine=SimpleNamespace(classify=Mock(return_value={})),
        _inject_semantic_memory_context=Mock(),
        registry=SimpleNamespace(get_or_create=Mock(return_value=session)),
        _tool_self_model=SimpleNamespace(apply_routing_feedback=Mock(return_value={})),
        _planning_optimizer=SimpleNamespace(suggest=Mock(return_value="")),
        _belief_state_engine=SimpleNamespace(next_best_strategy=Mock(return_value="")),
        _alignment_trainer=SimpleNamespace(recommend_strategy=Mock(return_value="")),
        _consume_pending_turn_steering=Mock(return_value=None),
        _interactive_cognition_ui=None,
        _tool_ecosystem_hub=None,
        _multi_agent_swarm=None,
        _post_planning_pre_tool_contract_gate=Mock(),
        _merge_session_state=Mock(),
    )

    metadata, _, _, _ = _prepare_submit_metadata_impl(
        control_plane,
        metadata={"session_id": "session-1", "trace_id": "trace-1"},
        user_input="hello",
        attachments=None,
        timeout_seconds=None,
        normalize_tool_runtime_contract=lambda _metadata: {},
        extract_execution_degradations=lambda _metadata: [],
    )

    hypothesis_engine.infer_hypotheses.assert_not_called()
    _prepare_submit_runtime_planning_impl(
        control_plane,
        session_key="session-1",
        user_input="hello",
        metadata=metadata,
    )

    hypothesis_engine.infer_hypotheses.assert_called_once()
    assert metadata["reasoning_hypotheses"] == hypotheses
