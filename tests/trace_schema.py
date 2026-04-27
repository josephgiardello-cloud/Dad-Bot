"""Normalized execution trace schema for capability measurement.

This is the single source of truth for what an execution trace contains.
ALL scoring, diagnostics, and reporting must go through this schema.

Design principles:
- Normalized: no loose dicts anywhere that a downstream scorer touches
- Extensible: raw_state preserved for future signals
- Typed: every field has a clear type and meaning
- Ordered: timestamps on everything that has sequence

Trace hierarchy:
  NormalizedTrace
  ├── NodeTransition[]    (pipeline nodes in execution order)
  ├── PlannerTrace        (planner output, plan quality signals)
  ├── ToolTrace[]         (each tool invocation with inputs/outputs)
  ├── MemoryAccess[]      (each memory read with type and relevance)
  ├── ErrorRecord[]       (errors by type and node)
  └── raw_state: dict    (full TurnContext.state, for extensibility)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums – stable categorical values
# ---------------------------------------------------------------------------

class ToolStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"
    FALLBACK = "fallback"


class MemoryType(str, Enum):
    GRAPH = "graph"         # semantic graph memory
    SEMANTIC = "semantic"   # vector/embedding memory
    ARCHIVE = "archive"     # long-term archive
    WORKING = "working"     # in-context working memory
    GOAL = "goal"           # goal tracker memory
    UNKNOWN = "unknown"


class ErrorClass(str, Enum):
    TIMEOUT = "timeout"
    LOGIC_ERROR = "logic_error"
    SAFETY_BLOCK = "safety_block"
    MEMORY_MISS = "memory_miss"
    TOOL_FAILURE = "tool_failure"
    PLAN_FAILURE = "plan_failure"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Node Transitions
# ---------------------------------------------------------------------------

@dataclass
class NodeTransition:
    """Records a single node's execution in the pipeline."""
    node: str
    started_at_ms: float
    completed_at_ms: float
    success: bool
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        return self.completed_at_ms - self.started_at_ms

    @classmethod
    def from_state(
        cls,
        node: str,
        started_at_ms: Optional[float] = None,
        completed_at_ms: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> "NodeTransition":
        now = time.monotonic() * 1000
        return cls(
            node=node,
            started_at_ms=started_at_ms or now,
            completed_at_ms=completed_at_ms or now,
            success=success,
            error=error,
        )


# ---------------------------------------------------------------------------
# Tool Execution
# ---------------------------------------------------------------------------

@dataclass
class ToolTrace:
    """Records a single tool invocation with inputs, outputs, and quality signals."""
    tool_name: str
    intent: str
    inputs: Dict[str, Any]
    output: Any
    status: ToolStatus
    duration_ms: float
    sequence: int
    is_retry: bool = False
    retry_of: Optional[str] = None   # deterministic_id of original invocation
    is_fallback: bool = False
    deterministic_id: str = ""

    @property
    def succeeded(self) -> bool:
        return self.status == ToolStatus.SUCCESS

    @property
    def failed(self) -> bool:
        return self.status in (ToolStatus.FAILED, ToolStatus.TIMEOUT)

    @classmethod
    def from_execution(cls, execution: dict, results: list) -> "ToolTrace":
        """Build ToolTrace from orchestrator's tool_ir execution dict."""
        tool_name = str(execution.get("tool_name") or "")
        sequence = int(execution.get("sequence") or 0)
        det_id = str(execution.get("deterministic_id") or "")

        # Find matching result by deterministic_id or sequence
        matched_result = None
        for r in results:
            if det_id and r.get("deterministic_id") == det_id:
                matched_result = r
                break
            if r.get("sequence") == sequence and r.get("tool_name") == tool_name:
                matched_result = r
                break

        status_str = str((matched_result or execution).get("status") or "").lower()
        try:
            status = ToolStatus(status_str)
        except ValueError:
            status = ToolStatus.FAILED if status_str else ToolStatus.UNKNOWN

        output = (matched_result or {}).get("output")
        inputs = dict(execution.get("inputs") or execution.get("parameters") or {})

        return cls(
            tool_name=tool_name,
            intent=str(execution.get("intent") or ""),
            inputs=inputs,
            output=output,
            status=status,
            duration_ms=float(execution.get("duration_ms") or 0.0),
            sequence=sequence,
            is_retry=bool(execution.get("is_retry") or execution.get("retry_of")),
            retry_of=execution.get("retry_of"),
            is_fallback=bool(execution.get("is_fallback")),
            deterministic_id=det_id,
        )


# ---------------------------------------------------------------------------
# Memory Access
# ---------------------------------------------------------------------------

@dataclass
class MemoryAccess:
    """Records a single memory read with type and relevance signal."""
    key: str
    memory_type: MemoryType
    retrieved: bool
    value_summary: str = ""
    relevance_score: Optional[float] = None
    source: str = ""       # session|long_term|goal_tracker|...

    @property
    def is_hit(self) -> bool:
        return self.retrieved

    @classmethod
    def from_structured(
        cls,
        key: str,
        value: Any,
        memory_type: str = "unknown",
    ) -> "MemoryAccess":
        """Build MemoryAccess from context.state['memory_structured'] entry."""
        try:
            mtype = MemoryType(memory_type.lower())
        except ValueError:
            mtype = MemoryType.UNKNOWN

        value_summary = ""
        if value is not None:
            raw = str(value)
            value_summary = raw[:80] + "..." if len(raw) > 80 else raw

        return cls(
            key=key,
            memory_type=mtype,
            retrieved=value is not None,
            value_summary=value_summary,
        )


# ---------------------------------------------------------------------------
# Planner Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class PlannerTrace:
    """Structured planner output with quality signals."""
    goals_detected: List[str] = field(default_factory=list)
    plan_steps: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[Tuple[str, str]] = field(default_factory=list)
    replan_count: int = 0
    plan_completeness: float = 0.0  # 0.0-1.0 estimated completeness

    @property
    def step_count(self) -> int:
        return len(self.plan_steps)

    @property
    def goal_count(self) -> int:
        return len(self.goals_detected)

    @property
    def has_dependencies(self) -> bool:
        return len(self.dependencies) > 0

    @property
    def replanned(self) -> bool:
        return self.replan_count > 0

    def branching_factor(self) -> float:
        """Average number of next steps from each step (rough proxy for plan complexity)."""
        if self.step_count == 0:
            return 0.0
        dep_count = len(self.dependencies)
        return dep_count / self.step_count

    @classmethod
    def from_context_state(cls, state: dict) -> "PlannerTrace":
        """Extract planner diagnostics from TurnContext.state."""
        plan = state.get("plan") or []
        goals = state.get("detected_goals") or state.get("new_goals") or []

        # Normalize goals to str list
        goal_strs: List[str] = []
        for g in goals:
            if isinstance(g, dict):
                goal_strs.append(str(g.get("description") or g.get("id") or ""))
            else:
                goal_strs.append(str(g))

        # Normalize plan to step list
        plan_steps: List[Dict] = []
        if isinstance(plan, list):
            plan_steps = [dict(s) if isinstance(s, dict) else {"step": str(s)} for s in plan]
        elif isinstance(plan, dict):
            plan_steps = [plan]

        # Extract dependencies from task decomposition
        decomp = state.get("task_decomposition") or {}
        deps: List[Tuple[str, str]] = []
        for dep in decomp.get("dependencies") or []:
            if isinstance(dep, (list, tuple)) and len(dep) >= 2:
                deps.append((str(dep[0]), str(dep[1])))
            elif isinstance(dep, dict):
                deps.append((str(dep.get("from", "")), str(dep.get("to", ""))))

        replan_count = int(state.get("replan_count") or 0)

        # Estimate completeness: have plan + goals at minimum
        completeness = 0.0
        if plan_steps:
            completeness += 0.5
        if goal_strs:
            completeness += 0.3
        if deps:
            completeness += 0.2

        return cls(
            goals_detected=goal_strs,
            plan_steps=plan_steps,
            dependencies=deps,
            replan_count=replan_count,
            plan_completeness=min(1.0, completeness),
        )


# ---------------------------------------------------------------------------
# Error Records
# ---------------------------------------------------------------------------

@dataclass
class ErrorRecord:
    """Structured error record with classification."""
    error_class: ErrorClass
    message: str
    node: Optional[str] = None
    recoverable: bool = False

    @classmethod
    def classify(cls, message: str, node: Optional[str] = None) -> "ErrorRecord":
        """Classify error from message string."""
        msg_lower = message.lower()
        if "timeout" in msg_lower:
            ec = ErrorClass.TIMEOUT
            recoverable = True
        elif "safety" in msg_lower or "blocked" in msg_lower or "policy" in msg_lower:
            ec = ErrorClass.SAFETY_BLOCK
            recoverable = False
        elif "memory" in msg_lower and ("miss" in msg_lower or "not found" in msg_lower):
            ec = ErrorClass.MEMORY_MISS
            recoverable = True
        elif "tool" in msg_lower and ("fail" in msg_lower or "error" in msg_lower):
            ec = ErrorClass.TOOL_FAILURE
            recoverable = True
        elif "plan" in msg_lower and "fail" in msg_lower:
            ec = ErrorClass.PLAN_FAILURE
            recoverable = True
        else:
            ec = ErrorClass.UNKNOWN
            recoverable = False
        return cls(error_class=ec, message=message[:200], node=node, recoverable=recoverable)


# ---------------------------------------------------------------------------
# Normalized Trace (top-level container)
# ---------------------------------------------------------------------------

@dataclass
class NormalizedTrace:
    """Complete normalized execution trace for one scenario run.

    This is the source of truth for all scoring, diagnostics, and reporting.
    """
    scenario_name: str
    category: str
    input_text: str
    final_response: str
    completed: bool
    total_duration_ms: float

    node_transitions: List[NodeTransition] = field(default_factory=list)
    planner: Optional[PlannerTrace] = None
    tools: List[ToolTrace] = field(default_factory=list)
    memory_accesses: List[MemoryAccess] = field(default_factory=list)
    errors: List[ErrorRecord] = field(default_factory=list)

    # Preserved for extensibility – future scorers can read directly
    raw_state: Dict[str, Any] = field(default_factory=dict)

    # Execution mode – signals how to interpret the trace
    execution_mode: str = "mock"   # "mock" | "orchestrator"

    # -----------------------------------------------------------------------
    # Derived properties
    # -----------------------------------------------------------------------

    @property
    def tool_count(self) -> int:
        return len(self.tools)

    @property
    def tools_succeeded(self) -> List[ToolTrace]:
        return [t for t in self.tools if t.succeeded]

    @property
    def tools_failed(self) -> List[ToolTrace]:
        return [t for t in self.tools if t.failed]

    @property
    def tool_success_rate(self) -> float:
        if not self.tools:
            return 0.0
        return len(self.tools_succeeded) / len(self.tools)

    @property
    def retry_count(self) -> int:
        return sum(1 for t in self.tools if t.is_retry)

    @property
    def fallback_count(self) -> int:
        return sum(1 for t in self.tools if t.is_fallback)

    @property
    def memory_hit_count(self) -> int:
        return sum(1 for m in self.memory_accesses if m.is_hit)

    @property
    def memory_hit_rate(self) -> float:
        if not self.memory_accesses:
            return 0.0
        return self.memory_hit_count / len(self.memory_accesses)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def error_classes(self) -> List[ErrorClass]:
        return [e.error_class for e in self.errors]

    @property
    def is_mock(self) -> bool:
        return self.execution_mode == "mock"

    # -----------------------------------------------------------------------
    # Builders
    # -----------------------------------------------------------------------

    @classmethod
    def from_mock(
        cls,
        scenario_name: str,
        category: str,
        input_text: str,
        final_response: str,
        completed: bool,
        error: Optional[str] = None,
        planner_output: Optional[Dict] = None,
        tools_executed: Optional[List[str]] = None,
        memory_accessed: Optional[List[str]] = None,
    ) -> "NormalizedTrace":
        """Build a NormalizedTrace from Phase 1 mock execution output."""
        errors = [ErrorRecord.classify(error)] if error else []
        tools = [
            ToolTrace(
                tool_name=t,
                intent="mock",
                inputs={},
                output=None,
                status=ToolStatus.SUCCESS,
                duration_ms=0.0,
                sequence=i,
            )
            for i, t in enumerate(tools_executed or [])
        ]
        memory = [
            MemoryAccess.from_structured(k, True)
            for k in (memory_accessed or [])
        ]
        planner = None
        if planner_output:
            plan_steps = [planner_output] if isinstance(planner_output, dict) else []
            planner = PlannerTrace(
                plan_steps=plan_steps,
                plan_completeness=0.5,  # mock completeness
            )
        return cls(
            scenario_name=scenario_name,
            category=category,
            input_text=input_text,
            final_response=final_response,
            completed=completed,
            total_duration_ms=0.0,
            planner=planner,
            tools=tools,
            memory_accesses=memory,
            errors=errors,
            execution_mode="mock",
        )

    @classmethod
    def from_orchestrator_context(
        cls,
        scenario_name: str,
        category: str,
        input_text: str,
        final_response: str,
        completed: bool,
        context: Any,  # TurnContext – avoid hard import
        duration_ms: float = 0.0,
        error: Optional[str] = None,
    ) -> "NormalizedTrace":
        """Build a NormalizedTrace from a real TurnContext after execution."""
        state: Dict[str, Any] = dict(getattr(context, "state", {}) or {})

        # Planner
        planner = PlannerTrace.from_context_state(state)

        # Tools
        tool_ir = dict(state.get("tool_ir") or {})
        executions = list(tool_ir.get("executions") or [])
        tool_results = list(state.get("tool_results") or [])
        tools = [ToolTrace.from_execution(e, tool_results) for e in executions]

        # Memory
        memory_structured = dict(state.get("memory_structured") or {})
        memory_accesses = [
            MemoryAccess.from_structured(k, v)
            for k, v in memory_structured.items()
        ]

        # Node transitions (from checkpoint/execution witness if available)
        node_transitions: List[NodeTransition] = []
        checkpoint_log = list(state.get("_checkpoint_log") or [])
        for entry in checkpoint_log:
            if isinstance(entry, dict) and entry.get("node"):
                node_transitions.append(
                    NodeTransition.from_state(
                        node=entry.get("node", ""),
                        started_at_ms=float(entry.get("started_at_ms") or 0),
                        completed_at_ms=float(entry.get("completed_at_ms") or 0),
                        success=bool(entry.get("success", True)),
                        error=entry.get("error"),
                    )
                )

        # Errors
        errors: List[ErrorRecord] = []
        if error:
            errors.append(ErrorRecord.classify(error))
        state_error = state.get("error")
        if state_error and str(state_error) != str(error or ""):
            errors.append(ErrorRecord.classify(str(state_error)))

        return cls(
            scenario_name=scenario_name,
            category=category,
            input_text=input_text,
            final_response=final_response,
            completed=completed,
            total_duration_ms=duration_ms,
            node_transitions=node_transitions,
            planner=planner,
            tools=tools,
            memory_accesses=memory_accesses,
            errors=errors,
            raw_state=state,
            execution_mode="orchestrator",
        )
