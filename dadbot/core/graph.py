from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

from dadbot.contracts import AttachmentList, ChunkCallback, FinalizedTurnResult
from dadbot.core.determinism import DeterminismBoundary, DeterminismMode
from dadbot.core.execution_boundary import ControlPlaneExecutionBoundary
from dadbot.core.kernel import TurnKernel


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


@dataclass
class StageTrace:
    """Execution record for a single pipeline stage stored inside TurnContext."""

    stage: str
    duration_ms: float
    error: str | None = None


class TurnPhase(str, Enum):
    PLAN = "PLAN"
    ACT = "ACT"
    OBSERVE = "OBSERVE"
    RESPOND = "RESPOND"


_PHASE_ORDER: tuple[TurnPhase, ...] = (
    TurnPhase.PLAN,
    TurnPhase.ACT,
    TurnPhase.OBSERVE,
    TurnPhase.RESPOND,
)


@dataclass
class TurnContext:
    user_input: str
    attachments: AttachmentList | None = None
    chunk_callback: ChunkCallback | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    # Correlation ID propagated through all log/telemetry calls for this turn.
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Ordered per-stage timing records appended by TurnGraph.execute().
    stage_traces: list[StageTrace] = field(default_factory=list)
    # Deterministic state-machine phase for this turn.
    phase: TurnPhase = TurnPhase.PLAN
    # Ordered history of phase transitions for replay/forensics.
    phase_history: list[dict[str, str]] = field(default_factory=list)
    # Monotonic sequence for append-only turn events.
    event_sequence: int = 0
    # Set True by any node to halt the pipeline immediately after that stage.
    short_circuit: bool = False
    # Required when short_circuit=True; becomes the turn's final result.
    short_circuit_result: FinalizedTurnResult | None = None
    # Determinism enforcement boundary: seals/replays all non-deterministic ops.
    determinism_boundary: DeterminismBoundary = field(default_factory=DeterminismBoundary)

    def snapshot(self, result: FinalizedTurnResult) -> dict[str, Any]:
        return {
            "user_input": self.user_input,
            "attachments": list(self.attachments or []),
            "result": result,
            "state": dict(self.state),
            "metadata": dict(self.metadata),
            "trace_id": self.trace_id,
            "phase": self.phase.value,
            "phase_history": list(self.phase_history),
            "event_sequence": self.event_sequence,
            "stage_traces": [
                {"stage": t.stage, "duration_ms": t.duration_ms, "error": t.error}
                for t in self.stage_traces
            ],
            "determinism_boundary": self.determinism_boundary.snapshot(),
        }

    def checkpoint_snapshot(self, *, stage: str, status: str, error: str | None = None) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "stage": str(stage or "unknown"),
            "status": str(status or "unknown"),
            "error": str(error or "").strip(),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "phase": self.phase.value,
            "user_input": self.user_input,
            "attachments": _json_safe(list(self.attachments or [])),
            "metadata": _json_safe(self.metadata),
            "state": _json_safe(self.state),
            "event_sequence": int(self.event_sequence),
            "phase_history": _json_safe(self.phase_history),
            "stage_traces": [
                {"stage": trace.stage, "duration_ms": trace.duration_ms, "error": trace.error}
                for trace in self.stage_traces
            ],
            "short_circuit": bool(self.short_circuit),
            "short_circuit_result": _json_safe(self.short_circuit_result),
            "determinism_boundary": _json_safe(self.determinism_boundary.snapshot()),
        }

    def transition_phase(self, target: TurnPhase, *, reason: str) -> list[dict[str, str]]:
        transitions: list[dict[str, str]] = []
        if self.phase == target:
            return transitions

        try:
            current_index = _PHASE_ORDER.index(self.phase)
            target_index = _PHASE_ORDER.index(target)
        except ValueError as exc:
            raise RuntimeError(f"Unknown phase transition request: {self.phase} -> {target}") from exc

        if target_index < current_index:
            raise RuntimeError(f"Non-deterministic phase regression: {self.phase} -> {target}")

        for idx in range(current_index + 1, target_index + 1):
            previous = self.phase
            self.phase = _PHASE_ORDER[idx]
            transition = {
                "from": previous.value,
                "to": self.phase.value,
                "reason": str(reason or "stage-enter"),
            }
            self.phase_history.append(transition)
            transitions.append(transition)
        return transitions


class GraphNode(Protocol):
    name: str

    async def execute(self, registry: Any, turn_context: TurnContext) -> None: ...


class HealthNode:
    name = "health"

    async def execute(self, registry: Any, turn_context: TurnContext) -> None:
        service = registry.get("maintenance_service")
        turn_context.state["health"] = service.tick(turn_context)


class ContextBuilderNode:
    name = "context_builder"

    async def execute(self, registry: Any, turn_context: TurnContext) -> None:
        service = registry.get("context_service")
        turn_context.state["rich_context"] = service.build_context(turn_context)


class MemoryNode(ContextBuilderNode):
    """Backward-compatible alias for legacy graph stage naming."""

    name = "memory"


class InferenceNode:
    name = "inference"

    async def execute(self, registry: Any, turn_context: TurnContext) -> None:
        service = registry.get("agent_service")
        rich_context = turn_context.state.get("rich_context", {})
        turn_context.state["candidate"] = await service.run_agent(turn_context, rich_context)


class SafetyNode:
    name = "safety"

    async def execute(self, registry: Any, turn_context: TurnContext) -> None:
        service = registry.get("safety_service")
        candidate = turn_context.state.get("candidate")
        turn_context.state["safe_result"] = service.enforce_policies(turn_context, candidate)


class SaveNode:
    name = "save"

    async def execute(self, registry: Any, turn_context: TurnContext) -> None:
        service = registry.get("persistence_service")
        result = turn_context.state.get("safe_result")
        service.save_turn(turn_context, result)


class TurnGraph:
    """Declarative turn execution graph."""

    def __init__(self, registry: Any = None, nodes: list[GraphNode] | None = None):
        self.registry = registry
        self.nodes = nodes or [HealthNode(), ContextBuilderNode(), InferenceNode(), SafetyNode(), SaveNode()]
        self._node_map: dict[str, Any] = {}
        self._edges: dict[str, str] = {}
        self._entry_node: str | None = None
        self._kernel = None  # TurnKernel | None
        self._required_execution_token: str = ""
        self._execution_witness_emitter: Callable[[str, TurnContext], None] | None = None

    def add_node(self, name: str, node: Any) -> None:
        if name not in self._node_map:
            if self._entry_node is None:
                self._entry_node = name
            self._node_map[name] = node

    def set_edge(self, source: str, target: str) -> None:
        self._edges[source] = target

    def set_kernel(self, kernel: Any) -> None:
        """Attach an execution kernel.  All node calls route through it."""
        self._kernel = kernel

    def set_required_execution_token(self, execution_token: str) -> None:
        """Require that execute() runs inside a control-plane boundary token."""
        self._required_execution_token = str(execution_token or "").strip()

    def set_execution_witness_emitter(self, emitter: Callable[[str, TurnContext], None] | None) -> None:
        self._execution_witness_emitter = emitter

    def _emit_execution_witness(self, component: str, turn_context: TurnContext) -> None:
        emitter = self._execution_witness_emitter
        if callable(emitter):
            emitter(str(component or ""), turn_context)

    @staticmethod
    def _fork_context(turn_context: TurnContext) -> TurnContext:
        return TurnContext(
            user_input=turn_context.user_input,
            attachments=turn_context.attachments,
            chunk_callback=turn_context.chunk_callback,
            metadata=dict(turn_context.metadata),
            state=dict(turn_context.state),
            trace_id=turn_context.trace_id,
            # stage_traces starts empty in each fork; merged back after parallel execution
        )

    async def _execute_parallel_nodes(self, nodes: list[Any] | tuple[Any, ...], turn_context: TurnContext) -> TurnContext:
        forked_contexts = [self._fork_context(turn_context) for _ in nodes]
        results = await asyncio.gather(
            *(self._execute_node(node, forked_context) for node, forked_context in zip(nodes, forked_contexts)),
        )
        for result_context in results:
            turn_context.metadata.update(result_context.metadata)
            turn_context.state.update(result_context.state)
            turn_context.stage_traces.extend(result_context.stage_traces)
        return turn_context

    async def _execute_node(self, node: Any, turn_context: TurnContext) -> TurnContext:
        if isinstance(node, (list, tuple)):
            return await self._execute_parallel_nodes(node, turn_context)

        async def _call_node() -> TurnContext:
            if callable(getattr(node, "run", None)):
                result = await node.run(turn_context)
                return result or turn_context
            execute_method = getattr(node, "execute", None)
            if callable(execute_method):
                await execute_method(self.registry, turn_context)
                return turn_context
            raise TypeError(f"Unsupported node type: {type(node).__name__}")

        if self._kernel is not None:
            node_name = getattr(node, "name", type(node).__name__)
            step_result = await self._kernel.execute_step(turn_context, node_name, _call_node)
            if step_result.status == "error":
                raise RuntimeError(step_result.error)
            # "rejected" -> step was skipped by policy; pipeline continues normally.
            return turn_context

        return await _call_node()
    def _emit_checkpoint(self, turn_context: TurnContext, *, stage: str, status: str, error: str | None = None) -> None:
        if self.registry is None:
            return
        service = self.registry.get("persistence_service")
        save_checkpoint = getattr(service, "save_graph_checkpoint", None)
        determinism_lock = dict(turn_context.metadata.get("determinism") or {})
        turn_context.event_sequence += 1
        checkpoint = turn_context.checkpoint_snapshot(stage=stage, status=status, error=error)
        if callable(save_checkpoint):
            save_checkpoint(checkpoint)

        save_event = getattr(service, "save_turn_event", None)
        if callable(save_event):
            save_event(
                {
                    "event_type": "graph_checkpoint",
                    "trace_id": turn_context.trace_id,
                    "sequence": turn_context.event_sequence,
                    "occurred_at": datetime.now().isoformat(timespec="seconds"),
                    "stage": str(stage),
                    "status": str(status),
                    "error": str(error or "").strip(),
                    "phase": turn_context.phase.value,
                    "determinism_lock": _json_safe(determinism_lock),
                    "checkpoint": checkpoint,
                }
            )

    @staticmethod
    def _phase_for_stage(stage: str, current: TurnPhase) -> TurnPhase:
        lowered = str(stage or "").strip().lower()
        if lowered in {"preflight", "health", "memory", "context", "plan"}:
            return TurnPhase.PLAN
        if lowered in {"inference", "agent", "tool", "act"}:
            return TurnPhase.ACT
        if lowered in {"safety", "guard", "observe", "moderate", "moderation"}:
            return TurnPhase.OBSERVE
        if lowered in {"save", "respond", "final", "finalize", "persist"}:
            return TurnPhase.RESPOND
        return current

    def _sync_stage_phase(self, turn_context: TurnContext, *, stage: str) -> None:
        target_phase = self._phase_for_stage(stage, turn_context.phase)
        transitions = turn_context.transition_phase(target_phase, reason=f"enter:{stage}")
        for transition in transitions:
            if self.registry is None:
                continue
            service = self.registry.get("persistence_service")
            save_event = getattr(service, "save_turn_event", None)
            if callable(save_event):
                turn_context.event_sequence += 1
                determinism_lock = dict(turn_context.metadata.get("determinism") or {})
                save_event(
                    {
                        "event_type": "phase_transition",
                        "trace_id": turn_context.trace_id,
                        "sequence": turn_context.event_sequence,
                        "occurred_at": datetime.now().isoformat(timespec="seconds"),
                        "stage": str(stage),
                        "phase": turn_context.phase.value,
                        "transition": dict(transition),
                        "determinism_lock": _json_safe(determinism_lock),
                    }
                )

    async def execute(self, turn_context: TurnContext) -> FinalizedTurnResult:
        if self._required_execution_token:
            active_token = ControlPlaneExecutionBoundary.current()
            if active_token != self._required_execution_token:
                raise RuntimeError(
                    "TurnGraph.execute boundary violation: graph execution must be routed through ExecutionControlPlane"
                )

        self._emit_execution_witness("graph.execute", turn_context)

        telemetry = self.registry.get("telemetry") if self.registry is not None else None

        if self._node_map:
            node_name = self._entry_node
            while node_name is not None:
                node = self._node_map[node_name]
                started_at = time.perf_counter()
                error_msg: str | None = None
                self._sync_stage_phase(turn_context, stage=node_name)
                self._emit_checkpoint(turn_context, stage=node_name, status="before")
                if telemetry is not None:
                    telemetry.trace("turn_graph.node_start", node=node_name, trace_id=turn_context.trace_id)
                try:
                    turn_context = await self._execute_node(node, turn_context)
                except Exception as exc:
                    error_msg = str(exc)
                    self._emit_checkpoint(turn_context, stage=node_name, status="error", error=error_msg)
                    raise
                finally:
                    duration_ms = round((time.perf_counter() - started_at) * 1000, 3)
                    turn_context.stage_traces.append(
                        StageTrace(stage=node_name, duration_ms=duration_ms, error=error_msg)
                    )
                    if telemetry is not None:
                        telemetry.trace(
                            "turn_graph.node_done",
                            node=node_name,
                            duration_ms=duration_ms,
                            trace_id=turn_context.trace_id,
                        )
                if error_msg is None:
                    self._emit_checkpoint(turn_context, stage=node_name, status="after")
                if turn_context.short_circuit:
                    if turn_context.short_circuit_result is not None:
                        return turn_context.short_circuit_result
                    break
                prev_node = node_name
                node_name = self._edges.get(node_name)
                # Emit a durable edge-transition checkpoint so a crash mid-edge can be
                # replayed from the exact transition boundary (LangGraph-style immortality).
                if node_name is not None and error_msg is None:
                    self._emit_checkpoint(
                        turn_context,
                        stage=f"{prev_node}\u2192{node_name}",
                        status="edge",
                    )
        else:
            for idx, node in enumerate(self.nodes):
                node_name = getattr(node, "name", type(node).__name__)
                started_at = time.perf_counter()
                error_msg = None
                self._sync_stage_phase(turn_context, stage=node_name)
                self._emit_checkpoint(turn_context, stage=node_name, status="before")
                if telemetry is not None:
                    telemetry.trace("turn_graph.node_start", node=node_name, trace_id=turn_context.trace_id)
                try:
                    turn_context = await self._execute_node(node, turn_context)
                except Exception as exc:
                    error_msg = str(exc)
                    self._emit_checkpoint(turn_context, stage=node_name, status="error", error=error_msg)
                    raise
                finally:
                    duration_ms = round((time.perf_counter() - started_at) * 1000, 3)
                    turn_context.stage_traces.append(
                        StageTrace(stage=node_name, duration_ms=duration_ms, error=error_msg)
                    )
                    if telemetry is not None:
                        telemetry.trace(
                            "turn_graph.node_done",
                            node=node_name,
                            duration_ms=duration_ms,
                            trace_id=turn_context.trace_id,
                        )
                if error_msg is None:
                    self._emit_checkpoint(turn_context, stage=node_name, status="after")
                if turn_context.short_circuit:
                    if turn_context.short_circuit_result is not None:
                        return turn_context.short_circuit_result
                    break
                # Keep edge checkpoints in the linear fallback path too.
                next_index = idx + 1
                if next_index < len(self.nodes):
                    next_name = getattr(self.nodes[next_index], "name", type(self.nodes[next_index]).__name__)
                    self._emit_checkpoint(turn_context, stage=f"{node_name}\u2192{next_name}", status="edge")

        result = turn_context.state.get("safe_result")
        if isinstance(result, tuple) and len(result) >= 2:
            return result
        return (str(result or ""), False)
