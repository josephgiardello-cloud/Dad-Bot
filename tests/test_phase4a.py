"""Phase 4A: Orchestrator Integration Tests

These tests demonstrate scenario execution through real DadBotOrchestrator.

Tests are structured as:
1. Phase 1 baseline (mock execution) - always passes
2. Phase 4A integration (with orchestrator) - real measurements

When orchestrator is available, Phase 4A tests reveal real capability gaps.
When orchestrator is unavailable, tests skip gracefully.

DESIGN PRINCIPLE:
- Phase 1: Validates scenario framework (deterministic)
- Phase 4A: Measures agent intelligence (stochastic, real)
"""

import asyncio
import pytest
from typing import Optional

from tests.benchmark_runner import BenchmarkRunner, print_benchmark_results
from tests.scenario_suite import (
    Scenario,
    SCENARIOS,
    get_scenarios_by_category,
)


class TestPhase1MockExecution:
    """Phase 1: Mock execution baseline (always works)."""

    def test_all_scenarios_pass_mock(self):
        """Phase 1: All 15 scenarios pass with mock backend."""
        runner = BenchmarkRunner(strict=False, mode="mock")
        results = runner.run_all_scenarios()

        # Verify all scenarios complete
        assert len(results) == 15
        assert all(r["execution"]["completed"] for r in results)
        assert all(r["scoring"]["success"] for r in results)

    def test_categories_complete_mock(self):
        """Phase 1: All categories present and passing."""
        runner = BenchmarkRunner(strict=False, mode="mock")
        results = runner.run_all_scenarios()

        by_category = {}
        for r in results:
            cat = r["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(r)

        # Verify expected categories
        expected_categories = ["planning", "tool", "memory", "ux", "robustness"]
        for cat in expected_categories:
            assert cat in by_category
            scenarios = by_category[cat]
            assert all(s["scoring"]["success"] for s in scenarios)

    def test_specific_scenario_mock(self):
        """Phase 1: Specific scenario execution and trace capture."""
        runner = BenchmarkRunner(strict=False, mode="mock")

        # Get specific scenario
        planning_scenarios = get_scenarios_by_category("planning")
        assert len(planning_scenarios) > 0

        scenario = planning_scenarios[0]
        result = runner.run_scenario(scenario)

        # Verify result structure
        assert result["scenario"] == scenario.name
        assert result["category"] == "planning"
        assert result["execution"]["completed"] is True
        assert result["trace"]["planner_output"] is not None
        assert result["scoring"]["success"] is True

    def test_trace_structure_mock(self):
        """Phase 1: Trace structure is correct."""
        runner = BenchmarkRunner(strict=False, mode="mock")
        results = runner.run_all_scenarios()

        for result in results:
            # Execution struct
            execution = result["execution"]
            assert "completed" in execution
            assert "steps" in execution
            assert "error" in execution

            # Trace struct
            trace = result["trace"]
            assert "planner_output" in trace
            assert "tools_executed" in trace
            assert "memory_accessed" in trace
            assert "final_response" in trace

            # Scoring struct
            scoring = result["scoring"]
            assert "success" in scoring
            assert "steps" in scoring
            assert "tool_used_correctly" in scoring
            assert "error" in scoring


class TestPhase4AOrchestratorIntegration:
    """Phase 4A: Real orchestrator execution (with graceful skip if unavailable)."""

    @pytest.fixture(scope="class")
    def orchestrator(self):
        """Fixture: Get pre-initialized orchestrator if available."""
        try:
            from dadbot.core.dadbot import DadBot

            bot = DadBot()
            return getattr(bot, "turn_orchestrator", None)
        except Exception as e:
            pytest.skip(f"Orchestrator unavailable: {e}")

    def test_orchestrator_available(self, orchestrator):
        """Phase 4A: Verify orchestrator is available."""
        assert orchestrator is not None
        from dadbot.core.orchestrator import DadBotOrchestrator

        assert isinstance(orchestrator, DadBotOrchestrator)

    def test_single_scenario_orchestrator(self, orchestrator):
        """Phase 4A: Execute single scenario through orchestrator."""
        runner = BenchmarkRunner(
            strict=False,
            mode="orchestrator",
            orchestrator=orchestrator,
        )

        # Test with simplest scenario
        scenario = SCENARIOS[0]
        result = runner.run_scenario(scenario)

        # Verify execution completed (may fail gracefully)
        assert "execution" in result
        assert "trace" in result

        # If successful, verify trace structure
        if result["execution"]["completed"]:
            trace = result["trace"]
            assert isinstance(trace, dict)
            # Real trace should have planner data
            if "planner" in trace:
                assert isinstance(trace["planner"], dict)

    def test_all_scenarios_orchestrator(self, orchestrator):
        """Phase 4A: Execute all scenarios through orchestrator."""
        runner = BenchmarkRunner(
            strict=False,
            mode="orchestrator",
            orchestrator=orchestrator,
        )

        results = runner.run_all_scenarios()

        # Verify all scenarios attempted
        assert len(results) == 15

        # At least N scenarios must produce an execution_result envelope.
        attempted = sum(1 for r in results if isinstance(r.get("execution_result"), dict))
        assert attempted >= 10

        # Execution errors should be classified, never fatal to the harness.
        classified = [
            r["execution_result"].get("execution_error_class")
            for r in results
            if isinstance(r.get("execution_result"), dict)
        ]
        assert all(isinstance(c, str) and len(c) > 0 for c in classified)

        # Verify no crashes
        assert all("execution" in r for r in results)
        assert all("trace" in r for r in results)

    def test_orchestrator_trace_capture(self, orchestrator):
        """Phase 4A: Verify real trace capture from orchestrator."""
        runner = BenchmarkRunner(
            strict=False,
            mode="orchestrator",
            orchestrator=orchestrator,
        )

        # Run tool scenario to verify tool tracing
        tool_scenarios = get_scenarios_by_category("tool")
        if tool_scenarios:
            result = runner.run_scenario(tool_scenarios[0])

            if result["execution"]["completed"]:
                # Traces should have real data
                trace = result["trace"]
                
                # Real orchestrator traces include nested dict structure
                if isinstance(trace, dict):
                    # May have tools, planner, memory keys
                    trace_keys = set(trace.keys())
                    assert len(trace_keys) > 0


class TestPhase4ACapabilityMeasurement:
    """Phase 4A: Capability measurement and gap analysis."""

    @pytest.fixture(scope="class")
    def orchestrator(self):
        """Fixture: Get pre-initialized orchestrator if available."""
        try:
            from dadbot.core.dadbot import DadBot

            bot = DadBot()
            return getattr(bot, "turn_orchestrator", None)
        except Exception:
            pytest.skip("Orchestrator unavailable")

    def test_capability_profile_structure(self, orchestrator):
        """Phase 4A: Generate capability profile from results."""
        runner = BenchmarkRunner(
            strict=False,
            mode="orchestrator",
            orchestrator=orchestrator,
        )

        results = runner.run_all_scenarios()

        # Compute capability profile by category from intelligence scores
        # (separate from execution validity).
        profile = {}
        by_category: dict[str, dict[str, float]] = {}

        for r in results:
            cat = r["category"]
            if cat not in by_category:
                by_category[cat] = {"score_sum": 0.0, "total": 0.0}

            by_category[cat]["total"] += 1.0
            cap = r.get("capability_score") or {}
            by_category[cat]["score_sum"] += float((cap.get(cat) or 0.0))

        # Compute scores
        for cat, counts in by_category.items():
            score = counts["score_sum"] / counts["total"] if counts["total"] > 0 else 0.0
            profile[cat] = score

        # Verify all categories present
        assert "planning" in profile
        assert "tool" in profile
        assert "memory" in profile
        assert "ux" in profile
        assert "robustness" in profile

        # Scores should be 0.0-1.0
        for score in profile.values():
            assert 0.0 <= score <= 1.0

    def test_real_vs_mock_difference(self):
        """Phase 4A: Demonstrate difference between mock and real execution."""
        # Mock always returns 100%
        mock_runner = BenchmarkRunner(strict=False, mode="mock")
        mock_results = mock_runner.run_all_scenarios()
        mock_successes = sum(1 for r in mock_results if r["execution"]["completed"])
        assert mock_successes == 15

        # Try real orchestrator (may skip if unavailable)
        try:
            from dadbot.core.dadbot import DadBot

            bot = DadBot()
            orchestrator = getattr(bot, "turn_orchestrator", None)
            if orchestrator:
                real_runner = BenchmarkRunner(
                    strict=False,
                    mode="orchestrator",
                    orchestrator=orchestrator,
                )
                real_results = real_runner.run_all_scenarios()
                real_successes = sum(
                    1 for r in real_results if r["execution"]["completed"]
                )

                # Real execution may have failures (that's the point!)
                # This demonstrates mock vs. real difference
                print(f"\nMock success rate: {mock_successes}/15 (100%)")
                print(f"Real success rate: {real_successes}/15 ({100*real_successes/15:.1f}%)")

        except Exception:
            # Expected if orchestrator unavailable
            pytest.skip("Orchestrator unavailable for real vs mock comparison")


class TestScenarioSuiteValidation:
    """Validate scenario suite structure (independent of execution)."""

    def test_scenarios_completeness(self):
        """Verify 15 scenarios are defined."""
        assert len(SCENARIOS) == 15

    def test_scenario_structure(self):
        """Verify each scenario has required fields."""
        required_fields = [
            "name",
            "category",
            "input_text",
            "expected_capabilities",
            "success_criteria",
            "description",
        ]

        for scenario in SCENARIOS:
            for field in required_fields:
                assert hasattr(scenario, field), f"Missing {field} in {scenario.name}"
                assert getattr(scenario, field) is not None

    def test_categories_distribution(self):
        """Verify scenarios cover all capability categories."""
        categories = {}
        for scenario in SCENARIOS:
            cat = scenario.category
            categories[cat] = categories.get(cat, 0) + 1

        expected = {
            "planning": 3,
            "tool": 4,
            "memory": 3,
            "ux": 3,
            "robustness": 2,
        }

        for cat, count in expected.items():
            assert categories.get(cat, 0) == count


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
