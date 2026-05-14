"""CognitionEnvelope — volatile observation events for UX streaming.

These events are NOT part of the execution trace and MUST NOT be used for
state reconstruction.  They exist solely for the UI "Live Thought Stream."

Fields
------
step_id          : unique ID for this cognition step (e.g. "{trace_id}:planner:start")
thought_trace    : human-readable description of what the node is doing
confidence_score : 0.0–1.0 signal of how certain the node is at this moment
target_node      : name of the pipeline node emitting this event
volatile         : always True — excluded from replay / state reconstruction
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

COGNITION_EVENT_TYPE = "cognition_envelope"


@dataclass
class CognitionEnvelope:
    step_id: str
    thought_trace: str
    target_node: str
    confidence_score: float = 0.0
    volatile: bool = field(default=True, init=False)
    event_type: str = field(default=COGNITION_EVENT_TYPE, init=False)
    emitted_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "volatile": self.volatile,
            "step_id": str(self.step_id or ""),
            "thought_trace": str(self.thought_trace or ""),
            "target_node": str(self.target_node or ""),
            "confidence_score": float(self.confidence_score),
            "emitted_at": float(self.emitted_at),
        }


def emit_cognition(context: Any, envelope: CognitionEnvelope) -> None:
    """Append a CognitionEnvelope to the volatile cognition stream on context.

    The ``cognition_stream`` key is explicitly excluded from state reconstruction
    — it is a write-only observation buffer for the UI layer.
    """
    stream = list(context.state.get("cognition_stream") or [])
    stream.append(envelope.to_dict())
    context.state["cognition_stream"] = stream
