from dataclasses import dataclass, field

from tests.scoring_engine import ScoringEngine
from tests.trace_schema import NormalizedTrace


@dataclass
class _Scenario:
    name: str = "user_correction_mid_flow"
    category: str = "ux"
    behavioral_spec: dict = field(default_factory=dict)


def _trace_with_ux(ux_payload: dict) -> NormalizedTrace:
    return NormalizedTrace(
        scenario_name="user_correction_mid_flow",
        category="ux",
        input_text="I meant something else, redo it differently",
        final_response="",
        completed=True,
        total_duration_ms=10.0,
        raw_state={"ux_trace": ux_payload},
        execution_mode="orchestrator",
    )


def test_ux_score_requires_structured_ux_trace_in_strict_mode():
    engine = ScoringEngine()
    scenario = _Scenario()

    trace = _trace_with_ux({
        "intent_shift_detected": False,
        "clarification_requested": False,
        "repair_event_emitted": False,
        "user_confusion_detected": True,
        "replan_triggered": False,
        "memory_correction_written": False,
    })
    score = engine.score(trace, scenario)

    assert score.ux is not None
    assert score.ux.score < 0.3


def test_ux_repair_signals_raise_ux_score():
    engine = ScoringEngine()
    scenario = _Scenario()

    weak = _trace_with_ux({
        "intent_shift_detected": False,
        "clarification_requested": False,
        "repair_event_emitted": False,
        "user_confusion_detected": True,
        "replan_triggered": False,
        "memory_correction_written": False,
    })
    strong = _trace_with_ux({
        "intent_shift_detected": True,
        "clarification_requested": True,
        "repair_event_emitted": True,
        "user_confusion_detected": False,
        "replan_triggered": True,
        "memory_correction_written": True,
    })

    weak_score = engine.score(weak, scenario)
    strong_score = engine.score(strong, scenario)

    assert weak_score.ux is not None
    assert strong_score.ux is not None
    assert strong_score.ux.score > weak_score.ux.score
