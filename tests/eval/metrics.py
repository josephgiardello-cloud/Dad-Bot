from __future__ import annotations

import re
from typing import Any

from tests.eval.tracer import Trace


_WORD = re.compile(r"[a-zA-Z0-9_]+")


def _tokens(value: Any) -> set[str]:
    text = str(value or "").lower()
    return {tok for tok in _WORD.findall(text) if len(tok) > 2}


def _is_irrelevant(trace: Trace, call_input: Any) -> bool:
    query_tokens = _tokens(trace.input)
    input_tokens = _tokens(call_input)
    if not query_tokens or not input_tokens:
        return True
    return len(query_tokens & input_tokens) == 0


def compute_metrics(trace: Trace) -> dict[str, Any]:
    final_output = str(trace.final_output or "").strip()
    generic_failures = {
        "something went wrong. please try again.",
        "something went wrong",
    }
    success = bool(not trace.error and final_output and final_output.lower() not in generic_failures)
    tool_calls = len(trace.tool_calls)
    decision_outcome = str(getattr(trace, "decision_outcome", "no_tool_needed") or "no_tool_needed")
    executed_tool = decision_outcome == "executed_tool"
    robustness_suppressed = decision_outcome == "robustness_suppressed"
    no_tool_needed = decision_outcome == "no_tool_needed"

    if tool_calls == 0:
        irrelevance = 0.0
    else:
        irrelevant = sum(1 for call in trace.tool_calls if _is_irrelevant(trace, call.input))
        irrelevance = irrelevant / tool_calls

    if not success:
        efficiency = 0.0
    else:
        complexity = max(1, int(trace.steps) + tool_calls)
        latency = max(1, int(trace.latency_ms or 1))
        efficiency = min(1.0, (1000.0 / latency) * (1.0 / complexity))

    return {
        "success": success,
        "tool_calls": tool_calls,
        "irrelevance": round(float(irrelevance), 4),
        "efficiency": round(float(efficiency), 4),
        "tool_execution_rate": 1.0 if executed_tool else 0.0,
        "no_tool_rate": 1.0 if no_tool_needed else 0.0,
        "robustness_suppression_rate": 1.0 if robustness_suppressed else 0.0,
    }
