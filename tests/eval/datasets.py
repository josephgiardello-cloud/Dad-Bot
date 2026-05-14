from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from tests.eval.tracer import Trace


@dataclass
class EvalCase:
    input: str
    expected_tools: Optional[List[str]] = None
    expects_tool: bool = False


def default_eval_cases() -> list[EvalCase]:
    return [
        EvalCase(input="Why did the dad go to the bank?"),
        EvalCase(
            input="Remind me to call mom in 10 minutes",
            expected_tools=["set_reminder"],
            expects_tool=True,
        ),
        EvalCase(
            input="Search for Python datetime documentation",
            expected_tools=["web_search"],
            expects_tool=True,
        ),
        EvalCase(input="I feel overwhelmed today, can you help me think through next steps?"),
        EvalCase(input="What should I do first: get pre-approved for a mortgage or schedule an inspection?"),
    ]


def full_benchmark_cases() -> list[EvalCase]:
    from tests.scenario_suite import SCENARIOS

    cases: list[EvalCase] = []
    for scenario in SCENARIOS:
        expected_tools = None
        behavior = dict(scenario.behavioral_spec or {})
        if bool(behavior.get("expected_tool_use")):
            # Keep this minimal and non-prescriptive for mixed tool pathways.
            expected_tools = None
        cases.append(
            EvalCase(
                input=str(scenario.input_text),
                expected_tools=expected_tools,
                expects_tool=bool(behavior.get("expected_tool_use")),
            ),
        )
    return cases


def tool_match(trace: Trace, expected: Optional[List[str]]):
    if not expected:
        return None
    return [call.name for call in trace.tool_calls] == list(expected)
