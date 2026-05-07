from tests.benchmark_runner import BenchmarkRunner
from tests.scenario_suite import Scenario
from tests.trace_schema import NormalizedTrace, PlannerTrace


def test_condense_capability_score_includes_phase4b_metrics():
    scenario = Scenario(
        name="phase4b_tool_case",
        category="tool",
        input_text="test",
        behavioral_spec={"max_steps": 4, "quality_threshold": 0.8},
    )
    capability_score = {
        "overall": 0.82,
        "planning": {"score": 0.8, "partial_success": True},
        "tools": {
            "score": 0.9,
            "partial_success": False,
            "signals": [{"name": "tool_success_rate", "value": 0.75}],
        },
        "memory": {"score": 0.7, "partial_success": True},
        "ux": {"score": 0.9, "partial_success": False},
        "robustness": {"score": 0.8, "partial_success": False},
    }

    condensed = BenchmarkRunner._condense_capability_score(
        capability_score,
        scenario=scenario,
        execution_steps=6,
        execution_error_class="timeout",
        planner_diagnostics={
            "plan_length": 3.0,
            "branching_factor": 0.5,
            "revision_count": 1.0,
            "dependency_correctness": 1.0,
        },
        phase45_diagnostics={
            "coherence_score": 0.8,
            "coherence_penalty_count": 1,
            "contradiction_detected": True,
            "contradiction_count": 2,
            "contradiction_resolution": 0.7,
            "tool_selection_optimality": 0.9,
        },
    )

    assert condensed is not None
    assert condensed["overall"] == 0.82
    assert condensed["tool_correctness"] == 0.75
    assert condensed["failure_type"] == "timeout"
    assert condensed["step_efficiency"] == round(4 / 6, 4)
    assert condensed["partial_success"] == 0.4
    assert condensed["quality_threshold"] == 0.8
    assert condensed["quality_threshold_met"] is True
    assert condensed["subsystem_coverage"] == 1.0
    assert condensed["plan_length"] == 3.0
    assert condensed["branching_factor"] == 0.5
    assert condensed["revision_count"] == 1.0
    assert condensed["dependency_correctness"] == 1.0
    assert condensed["coherence_score"] == 0.8
    assert condensed["coherence_penalty_count"] == 1
    assert condensed["contradiction_detected"] is True
    assert condensed["contradiction_count"] == 2
    assert condensed["contradiction_resolution"] == 0.7
    assert condensed["tool_selection_optimality"] == 0.9


def test_condense_capability_score_defaults_when_optional_data_missing():
    scenario = Scenario(
        name="phase4b_missing_signal",
        category="planning",
        input_text="test",
        behavioral_spec={"max_steps": 0},
    )
    capability_score = {
        "overall": 0.3,
        "planning": {"score": 0.3, "partial_success": True},
        "tools": {"score": 0.2, "partial_success": True, "signals": []},
    }

    condensed = BenchmarkRunner._condense_capability_score(
        capability_score,
        scenario=scenario,
        execution_steps=0,
        execution_error_class="none",
    )

    assert condensed is not None
    assert condensed["tool_correctness"] == 0.2
    assert condensed["step_efficiency"] == 1.0
    assert condensed["failure_type"] == "none"
    assert condensed["quality_threshold"] == 0.0
    assert condensed["quality_threshold_met"] is True
    assert 0.0 <= condensed["subsystem_coverage"] <= 1.0
    assert condensed["plan_length"] == 0.0
    assert condensed["branching_factor"] == 0.0
    assert condensed["revision_count"] == 0.0
    assert condensed["dependency_correctness"] == 0.0
    assert condensed["coherence_score"] == 0.0
    assert condensed["coherence_penalty_count"] == 0
    assert condensed["contradiction_detected"] is False
    assert condensed["contradiction_count"] == 0
    assert condensed["contradiction_resolution"] == 0.0
    assert condensed["tool_selection_optimality"] == 0.0


def test_extract_planner_diagnostics_from_normalized_trace():
    scenario = Scenario(
        name="dependency_aware_task",
        category="planning",
        input_text="test",
        expected_capabilities=["dependency_graph_construction"],
    )
    normalized = NormalizedTrace(
        scenario_name="dependency_aware_task",
        category="planning",
        input_text="test",
        final_response="ok",
        completed=True,
        total_duration_ms=1.0,
        planner=PlannerTrace(
            plan_steps=[{"step": "a"}, {"step": "b"}, {"step": "c"}],
            dependencies=[("a", "b"), ("b", "c")],
            replan_count=2,
            plan_completeness=1.0,
        ),
    )

    diagnostics = BenchmarkRunner._extract_planner_diagnostics(normalized, scenario)

    assert diagnostics["plan_length"] == 3.0
    assert diagnostics["branching_factor"] == round(2 / 3, 4)
    assert diagnostics["revision_count"] == 2.0
    assert 0.0 <= diagnostics["dependency_correctness"] <= 1.0


def test_extract_phase45_diagnostics_from_normalized_trace():
    scenario = Scenario(
        name="tool_failure_recovery",
        category="tool",
        input_text="test",
        behavioral_spec={"expected_tool_use": True, "min_tool_calls": 1},
    )
    normalized = NormalizedTrace(
        scenario_name="tool_failure_recovery",
        category="tool",
        input_text="test",
        final_response="ok",
        completed=True,
        total_duration_ms=2.0,
        planner=PlannerTrace(
            plan_steps=[{"step": "a"}],
            dependencies=[],
            replan_count=0,
            plan_completeness=1.0,
        ),
        raw_state={
            "planner_causal_trace": {
                "intent_delta_vector": ["shift"],
                "planner_replan_reason": "",
                "dependency_graph_diff": [],
            },
            "tool_failure_semantics": [{"failure_class": "wrong_tool"}],
            "ux_trace": {"user_confusion_detected": True},
            "memory_contradictions": [{"reason": "conflicting fact"}],
        },
    )

    runner = BenchmarkRunner(strict=False, mode="mock")
    diagnostics = runner._extract_phase45_diagnostics(normalized, scenario)

    assert 0.0 <= diagnostics["coherence_score"] <= 1.0
    assert diagnostics["coherence_penalty_count"] >= 1
    assert diagnostics["contradiction_detected"] is True
    assert diagnostics["contradiction_count"] >= 1
    assert 0.0 <= diagnostics["contradiction_resolution"] <= 1.0
    assert 0.0 <= diagnostics["tool_selection_optimality"] <= 1.0
