from dataclasses import dataclass, field

from tests.scoring_engine import ScoringEngine
from tests.trace_schema import NormalizedTrace, PlannerTrace


@dataclass
class _Scenario:
    name: str = "dependency_aware_task"
    category: str = "planning"
    behavioral_spec: dict = field(default_factory=dict)


def _trace_with_planner(planner_causal_trace: dict) -> NormalizedTrace:
    planner = PlannerTrace(
        goals_detected=["complete house purchase"],
        plan_steps=[{"step": "mortgage"}, {"step": "inspection"}],
        dependencies=[("mortgage", "inspection")],
        replan_count=1,
        plan_completeness=0.9,
    )
    return NormalizedTrace(
        scenario_name="dependency_aware_task",
        category="planning",
        input_text="Order these dependent tasks.",
        final_response="",
        completed=True,
        total_duration_ms=12.0,
        planner=planner,
        raw_state={"planner_causal_trace": planner_causal_trace},
        execution_mode="orchestrator",
    )


def test_planner_causal_fields_affect_score():
    engine = ScoringEngine()
    scenario = _Scenario()

    missing = _trace_with_planner({})
    present = _trace_with_planner(
        {
            "planner_replan_reason": "user correction requested",
            "intent_delta_vector": ["cuisine:italian->vietnamese"],
            "dependency_graph_diff": ["replace step:italian_plan with vietnamese_plan"],
        }
    )

    missing_score = engine.score(missing, scenario)
    present_score = engine.score(present, scenario)

    assert missing_score.planning is not None
    assert present_score.planning is not None
    assert present_score.planning.score > missing_score.planning.score
