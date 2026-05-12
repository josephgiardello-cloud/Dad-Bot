from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import asdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dadbot.core.adversarial_closure import (
    TransitionSystem,
    classify_closure_report,
    explore_closure,
)


class _DeterministicConvergingClosure(TransitionSystem):
    """Deterministic closure probe used as a CI baseline canary."""

    def initial_state(self):
        return {"x": 0, "y": 0}

    def enabled_actions(self, state):
        if self.is_terminal(state):
            return []
        return ["inc-x", "inc-y"]

    def step(self, state, action):
        nxt = dict(state)
        if action == "inc-x":
            nxt["x"] = min(1, int(nxt.get("x", 0)) + 1)
        elif action == "inc-y":
            nxt["y"] = min(1, int(nxt.get("y", 0)) + 1)
        return nxt

    def is_terminal(self, state):
        return int(state.get("x", 0)) == 1 and int(state.get("y", 0)) == 1


def run_closure_gate() -> int:
    report = explore_closure(_DeterministicConvergingClosure(), max_depth=4, max_states=1024)
    classification = classify_closure_report(report)

    payload = {
        "gate": "adversarial_closure",
        "report": asdict(report),
        "classification": asdict(classification),
    }
    print(json.dumps(payload, sort_keys=True))

    if classification.unsafe:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run_closure_gate())
