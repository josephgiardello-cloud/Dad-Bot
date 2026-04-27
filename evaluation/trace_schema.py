"""Structured evaluation traces used by strict contract scoring.

These dataclasses are intentionally causal and auditable:
- They record why a subsystem behaved a certain way.
- They avoid scorer-side text heuristics in strict mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


@dataclass
class UXTrace:
    """Causal UX signals emitted by runtime instrumentation."""

    intent_shift_detected: bool
    clarification_requested: bool
    repair_event_emitted: bool
    user_confusion_detected: bool
    replan_triggered: bool
    memory_correction_written: bool

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "UXTrace":
        payload = dict(state.get("ux_trace") or state.get("ux_feedback") or {})
        return cls(
            intent_shift_detected=bool(payload.get("intent_shift_detected", False)),
            clarification_requested=bool(payload.get("clarification_requested", False)),
            repair_event_emitted=bool(payload.get("repair_event_emitted", False)),
            user_confusion_detected=bool(payload.get("user_confusion_detected", False)),
            replan_triggered=bool(payload.get("replan_triggered", False)),
            memory_correction_written=bool(payload.get("memory_correction_written", False)),
        )


@dataclass
class PlannerCausalTrace:
    """Planner causality fields required for strict evaluation."""

    planner_replan_reason: str = ""
    intent_delta_vector: List[str] = field(default_factory=list)
    dependency_graph_diff: List[str] = field(default_factory=list)

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "PlannerCausalTrace":
        payload = dict(state.get("planner_causal_trace") or {})
        return cls(
            planner_replan_reason=str(payload.get("planner_replan_reason") or "").strip(),
            intent_delta_vector=[str(x) for x in list(payload.get("intent_delta_vector") or [])],
            dependency_graph_diff=[str(x) for x in list(payload.get("dependency_graph_diff") or [])],
        )


@dataclass
class MemoryCausalTrace:
    """Memory read/write causal linkage for strict evaluation."""

    trigger: str = ""
    read_link_id: str = ""
    write_link_id: str = ""
    influenced_final_response: bool = False
    overridden: bool = False

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "MemoryCausalTrace":
        payload = dict(state.get("memory_causal_trace") or {})
        return cls(
            trigger=str(payload.get("trigger") or "").strip(),
            read_link_id=str(payload.get("read_link_id") or "").strip(),
            write_link_id=str(payload.get("write_link_id") or "").strip(),
            influenced_final_response=bool(payload.get("influenced_final_response", False)),
            overridden=bool(payload.get("overridden", False)),
        )


class ToolFailureClass(str, Enum):
    TIMEOUT = "timeout"
    WRONG_TOOL = "wrong_tool"
    BAD_INPUT = "bad_input"
    MISSING_CONTEXT = "missing_context"
    RUNTIME_EXCEPTION = "runtime_exception"
    UNKNOWN = "unknown"


@dataclass
class ToolFailureSemanticTrace:
    """Semantic failure typing for tool intelligence scoring."""

    tool_name: str
    failure_class: ToolFailureClass
    reason: str = ""


@dataclass
class CrossSubsystemCoherenceScore:
    """Global consistency indicator across subsystem traces."""

    score: float
    penalties: List[str] = field(default_factory=list)

    @property
    def coherent(self) -> bool:
        return self.score >= 0.7 and not self.penalties
