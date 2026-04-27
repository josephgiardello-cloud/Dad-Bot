from dataclasses import dataclass, field

from evaluation.coherence_engine import CoherenceEngine
from tests.scoring_engine import ScoringEngine
from tests.trace_schema import NormalizedTrace, PlannerTrace


@dataclass
class _Scenario:
    name: str = "coherence_probe"
    category: str = "planning"
    behavioral_spec: dict = field(default_factory=dict)


def _trace(raw_state: dict) -> NormalizedTrace:
    planner = PlannerTrace(
        goals_detected=["goal"],
        plan_steps=[{"step": "a"}],
        dependencies=[],
        replan_count=1,
        plan_completeness=0.8,
    )
    return NormalizedTrace(
        scenario_name="coherence_probe",
        category="planning",
        input_text="probe",
        final_response="",
        completed=True,
        total_duration_ms=8.0,
        planner=planner,
        raw_state=raw_state,
        execution_mode="orchestrator",
    )


def test_coherence_engine_penalizes_inconsistency():
    engine = CoherenceEngine()
    incoherent = {
        "planner_causal_trace": {"intent_delta_vector": ["delta-only"], "planner_replan_reason": ""},
        "memory_causal_trace": {"influenced_final_response": True, "overridden": True},
        "ux_trace": {"user_confusion_detected": True},
        "tool_failure_semantics": [{"failure_class": "timeout"}],
    }
    coherent = {
        "planner_causal_trace": {
            "intent_delta_vector": ["delta"],
            "planner_replan_reason": "user correction",
            "dependency_graph_diff": ["edge changed"],
        },
        "memory_causal_trace": {"influenced_final_response": True, "overridden": False},
        "ux_trace": {"user_confusion_detected": False},
        "tool_failure_semantics": [{"failure_class": "timeout"}],
    }

    incoherent_score = engine.score(incoherent)
    coherent_score = engine.score(coherent)

    assert coherent_score.score > incoherent_score.score
    assert len(incoherent_score.penalties) > 0


def test_scoring_engine_applies_coherence_multiplier_to_overall():
    scoring = ScoringEngine()
    scenario = _Scenario()

    strong_base = {
        "planner_causal_trace": {
            "planner_replan_reason": "user correction",
            "intent_delta_vector": ["topic shift"],
            "dependency_graph_diff": ["reordered"],
        },
        "ux_trace": {
            "intent_shift_detected": True,
            "clarification_requested": True,
            "repair_event_emitted": True,
            "user_confusion_detected": False,
            "replan_triggered": True,
            "memory_correction_written": True,
        },
        "memory_causal_trace": {
            "trigger": "user correction",
            "read_link_id": "r1",
            "write_link_id": "w1",
            "influenced_final_response": True,
            "overridden": False,
        },
        "tool_failure_semantics": [{"failure_class": "timeout"}],
    }

    incoherent = dict(strong_base)
    incoherent["planner_causal_trace"] = {
        "planner_replan_reason": "",
        "intent_delta_vector": ["topic shift"],
        "dependency_graph_diff": [],
    }
    incoherent["ux_trace"] = {
        "intent_shift_detected": True,
        "clarification_requested": False,
        "repair_event_emitted": False,
        "user_confusion_detected": True,
        "replan_triggered": False,
        "memory_correction_written": False,
    }

    coherent_score = scoring.score(_trace(strong_base), scenario)
    incoherent_score = scoring.score(_trace(incoherent), scenario)

    assert coherent_score.overall > incoherent_score.overall
