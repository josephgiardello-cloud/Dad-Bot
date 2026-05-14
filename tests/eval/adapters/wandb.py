from __future__ import annotations

from typing import Any

from tests.eval.tracer import Trace


def log(trace: Trace, metrics: dict[str, Any]) -> None:
    payload = {
        "steps": trace.steps,
        "success": metrics["success"],
        "irrelevance": metrics["irrelevance"],
        "tool_calls": metrics["tool_calls"],
    }

    try:
        import wandb  # type: ignore

        wandb.log(payload)
    except Exception:
        # Optional dependency: metrics are still returned by the runner.
        return
