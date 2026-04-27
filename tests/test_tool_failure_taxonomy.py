from dataclasses import dataclass, field

from tests.scoring_engine import ScoringEngine
from tests.trace_schema import NormalizedTrace, ToolStatus, ToolTrace


@dataclass
class _Scenario:
    name: str = "tool_failure_recovery"
    category: str = "tool"
    behavioral_spec: dict = field(default_factory=lambda: {"expected_tool_use": True})


def _trace_with_failure_semantics(semantics: list[dict]) -> NormalizedTrace:
    tools = [
        ToolTrace(
            tool_name="weather_api",
            intent="weather lookup",
            inputs={"city": "Austin"},
            output=None,
            status=ToolStatus.FAILED,
            duration_ms=150.0,
            sequence=1,
        )
    ]
    return NormalizedTrace(
        scenario_name="tool_failure_recovery",
        category="tool",
        input_text="check weather",
        final_response="",
        completed=True,
        total_duration_ms=20.0,
        tools=tools,
        raw_state={"tool_failure_semantics": semantics},
        execution_mode="orchestrator",
    )


def test_tool_failure_taxonomy_is_scored():
    engine = ScoringEngine()
    scenario = _Scenario()

    unclassified = _trace_with_failure_semantics([])
    classified = _trace_with_failure_semantics(
        [{"tool_name": "weather_api", "failure_class": "timeout", "reason": "request timed out"}]
    )

    unclassified_score = engine.score(unclassified, scenario)
    classified_score = engine.score(classified, scenario)

    assert unclassified_score.tools is not None
    assert classified_score.tools is not None
    assert classified_score.tools.score > unclassified_score.tools.score
