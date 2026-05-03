"""Golden Turn Suite — real-world correctness anchor tests.

These tests use the golden_turn_suite.json scenarios to verify that the system
produces structurally correct outputs for canonical input types.

Tests here exercise state contract invariants for each golden scenario.
Integration with a live model layer is NOT required — these are structural
correctness checks, not end-to-end response quality checks.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dadbot.core.graph_context import TurnContext


pytestmark = pytest.mark.unit

_SUITE_PATH = Path(__file__).parent / "golden_turn_suite.json"


def _load_suite() -> list[dict]:
    return json.loads(_SUITE_PATH.read_text(encoding="utf-8"))


def _scenario_ids() -> list[str]:
    return [s["name"] for s in _load_suite()]


def _scenario_by_name(name: str) -> dict:
    for s in _load_suite():
        if s["name"] == name:
            return s
    raise KeyError(name)


class TestGoldenTurnSuiteStructure:
    def test_suite_file_is_valid_json(self):
        """golden_turn_suite.json must be parseable JSON."""
        suite = _load_suite()
        assert isinstance(suite, list)
        assert len(suite) > 0

    def test_all_scenarios_have_required_fields(self):
        """Every scenario must have name, input, and expected_properties."""
        for scenario in _load_suite():
            assert "name" in scenario, f"Missing 'name': {scenario}"
            assert "input" in scenario, f"Missing 'input': {scenario}"
            assert "expected_properties" in scenario, f"Missing 'expected_properties': {scenario}"

    @pytest.mark.parametrize("scenario_name", _scenario_ids())
    def test_golden_turn_context_construction(self, scenario_name: str):
        """Each golden turn input must be constructible as a TurnContext."""
        scenario = _scenario_by_name(scenario_name)
        ctx = TurnContext(user_input=scenario["input"])
        assert ctx.user_input == scenario["input"]
        assert isinstance(ctx.state, dict)
        assert isinstance(ctx.metadata, dict)

    def test_normal_conversation_scenario_has_no_safety_trigger_property(self):
        """normal_conversation scenario must declare 'no_safety_trigger' property."""
        scenario = _scenario_by_name("normal_conversation")
        assert "no_safety_trigger" in scenario["expected_properties"]

    def test_safety_trigger_scenario_expects_safety_intervention(self):
        """safety_trigger scenario must declare 'safety_intervention' property."""
        scenario = _scenario_by_name("safety_trigger")
        assert "safety_intervention" in scenario["expected_properties"]

    def test_safety_trigger_scenario_marks_expects_safety_block(self):
        """safety_trigger scenario must have expects_safety_block=true."""
        scenario = _scenario_by_name("safety_trigger")
        assert scenario.get("expects_safety_block") is True

    def test_recovery_scenario_expects_deterministic_replay(self):
        """recovery_test scenario must declare 'deterministic_replay' property."""
        scenario = _scenario_by_name("recovery_test")
        assert "deterministic_replay" in scenario["expected_properties"]

    def test_tool_usage_scenario_expects_tool_call(self):
        """tool_usage scenario must declare 'tool_call' property."""
        scenario = _scenario_by_name("tool_usage")
        assert "tool_call" in scenario["expected_properties"]

    @pytest.mark.parametrize("scenario_name", _scenario_ids())
    def test_expected_state_keys_are_a_list(self, scenario_name: str):
        """expected_state_keys must be a list in every scenario."""
        scenario = _scenario_by_name(scenario_name)
        assert isinstance(scenario.get("expected_state_keys", []), list)

    @pytest.mark.parametrize("scenario_name", _scenario_ids())
    def test_turn_context_state_satisfies_expected_keys_when_populated(self, scenario_name: str):
        """When expected_state_keys are written to state, the context must reflect them."""
        scenario = _scenario_by_name(scenario_name)
        ctx = TurnContext(user_input=scenario["input"])
        for key in scenario.get("expected_state_keys", []):
            ctx.state[key] = f"_golden_stub_{key}"

        for key in scenario.get("expected_state_keys", []):
            assert key in ctx.state

    @pytest.mark.parametrize("scenario_name", _scenario_ids())
    def test_forbidden_state_keys_absent_by_default(self, scenario_name: str):
        """Forbidden state keys must not appear in a freshly constructed TurnContext."""
        scenario = _scenario_by_name(scenario_name)
        ctx = TurnContext(user_input=scenario["input"])
        for key in scenario.get("forbidden_state_keys", []):
            assert key not in ctx.state, (
                f"[{scenario_name}] Forbidden key present at construction: {key!r}"
            )
