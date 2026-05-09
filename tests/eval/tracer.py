from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional


DecisionOutcome = Literal[
    "executed_tool",
    "no_tool_needed",
    "robustness_suppressed",
]


@dataclass
class ToolCall:
    name: str
    input: Any
    output: Any


@dataclass
class Trace:
    input: str
    final_output: str
    tool_calls: List[ToolCall]
    error: Optional[str]
    steps: int
    latency_ms: Optional[int]
    robustness_suppressed: bool = False
    decision_outcome: DecisionOutcome = "no_tool_needed"
    planner_status: str = ""
    planner_tool: str = ""
    robustness_reason: Optional[str] = None
    execution_truth_contract: Optional[dict] = None


@dataclass
class TraceCollector:
    input_text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    steps: int = 0

    def record_tool_call(self, *, name: str, tool_input: Any, tool_output: Any) -> None:
        self.tool_calls.append(ToolCall(name=name, input=tool_input, output=tool_output))

    def bump_step(self, delta: int = 1) -> None:
        self.steps += int(delta)

    def finalize(
        self,
        *,
        final_output: str,
        error: Optional[str],
        latency_ms: Optional[int],
        robustness_suppressed: bool = False,
        decision_outcome: DecisionOutcome = "no_tool_needed",
        planner_status: str = "",
        planner_tool: str = "",
        robustness_reason: Optional[str] = None,
        execution_truth_contract: Optional[dict] = None,
    ) -> Trace:
        return Trace(
            input=self.input_text,
            final_output=final_output,
            tool_calls=list(self.tool_calls),
            error=error,
            steps=int(self.steps),
            latency_ms=latency_ms,
            robustness_suppressed=bool(robustness_suppressed),
            decision_outcome=decision_outcome,
            planner_status=str(planner_status or ""),
            planner_tool=str(planner_tool or ""),
            robustness_reason=(str(robustness_reason) if robustness_reason else None),
            execution_truth_contract=execution_truth_contract,
        )
