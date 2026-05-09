from __future__ import annotations

from typing import Any

from tests.eval.tracer import Trace


def to_eval_case(trace: Trace) -> dict[str, Any]:
    return {
        "input": trace.input,
        "output": trace.final_output,
    }


def append_eval_case(sink: list[dict[str, Any]], trace: Trace) -> None:
    sink.append(to_eval_case(trace))
