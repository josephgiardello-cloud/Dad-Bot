from __future__ import annotations

from typing import Any

from tests.eval.tracer import Trace


def to_langsmith(trace: Trace) -> dict[str, Any]:
    return {
        "inputs": {"input": trace.input},
        "outputs": {"output": trace.final_output},
        "intermediate_steps": [
            {
                "tool": t.name,
                "input": t.input,
                "output": t.output,
            }
            for t in trace.tool_calls
        ],
        "metadata": {
            "steps": trace.steps,
            "error": trace.error,
        },
    }


def log_run(client: Any, trace: Trace) -> None:
    if client is None:
        return
    payload = to_langsmith(trace)
    create_run = getattr(client, "create_run", None)
    if callable(create_run):
        try:
            create_run(
                name="dadbot-eval",
                run_type="chain",
                inputs=payload["inputs"],
                outputs=payload["outputs"],
                extra={"metadata": payload["metadata"], "intermediate_steps": payload["intermediate_steps"]},
            )
        except Exception:
            # Optional external sink; failures must not fail local benchmark execution.
            return
