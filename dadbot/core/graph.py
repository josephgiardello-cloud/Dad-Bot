from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any, Callable, Literal, Protocol

from dadbot.contracts import AttachmentList, ChunkCallback, FinalizedTurnResult
from dadbot.core.capability_audit_runner import build_runtime_capability_audit_report
from dadbot.core.determinism import DeterminismBoundary, DeterminismMode
from dadbot.core.execution_boundary import ControlPlaneExecutionBoundary
from dadbot.core.execution_firewall import ExecutionFirewall
from dadbot.core.execution_kernel import ExecutionKernel
from dadbot.core.invariant_registry import InvariantRegistry
from dadbot.core.kernel import TurnKernel


logger = logging.getLogger(__name__)


class FatalTurnError(RuntimeError):
    """Unrecoverable turn invariant violation.

    Raised when core pipeline guarantees are violated (e.g., mutation queue
    cannot be fully drained or required stages did not execute).
    """


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


@dataclass
class MutationIntent:
    """A single deferred persistent mutation to be committed at SaveNode."""

    type: "MutationKind"
    payload: dict[str, Any]
    requires_temporal: bool = True
    source: str = ""  # caller tag for audit/replay
    priority: int = 100
    turn_index: int = 0
    sequence_id: int = 0

    def __post_init__(self) -> None:
        try:
            self.type = MutationKind(self.type)
        except ValueError as exc:
            raise RuntimeError(f"MutationIntent.type must be one of {[item.value for item in MutationKind]}") from exc
        if not isinstance(self.payload, dict):
            raise RuntimeError("MutationIntent.payload must be a dict")
        if self.requires_temporal:
            temporal = self.payload.get("temporal")
            if not isinstance(temporal, dict):
                raise RuntimeError("TemporalNode required — execution invalid")
            wall_time = str(temporal.get("wall_time") or "").strip()
            wall_date = str(temporal.get("wall_date") or "").strip()
            if not wall_time or not wall_date:
                raise RuntimeError("TemporalNode required — execution invalid")
        try:
            self.priority = int(self.priority)
            self.turn_index = int(self.turn_index)
            self.sequence_id = int(self.sequence_id)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("MutationIntent ordering fields must be integers") from exc
        op = str(self.payload.get("op") or "").strip().lower()
        if self.type is MutationKind.MEMORY:
            if op and op not in {item.value for item in MemoryMutationOp}:
                raise RuntimeError(f"Unsupported memory mutation op: {op!r}")
        elif self.type is MutationKind.RELATIONSHIP:
            if op and op not in {item.value for item in RelationshipMutationOp}:
                raise RuntimeError(f"Unsupported relationship mutation op: {op!r}")
        elif self.type is MutationKind.LEDGER:
            if op and op not in {item.value for item in LedgerMutationOp}:
                raise RuntimeError(f"Unsupported ledger mutation op: {op!r}")


class MutationKind(StrEnum):
    MEMORY = "memory"
    RELATIONSHIP = "relationship"
    GRAPH = "graph"
    LEDGER = "ledger"


class MemoryMutationOp(StrEnum):
    SAVE_MOOD_STATE = "save_mood_state"


class RelationshipMutationOp(StrEnum):
    UPDATE = "update"


class LedgerMutationOp(StrEnum):
    APPEND_HISTORY = "append_history"
    SYNC_THREAD_SNAPSHOT = "sync_thread_snapshot"
    CLEAR_TURN_CONTEXT = "clear_turn_context"
    SCHEDULE_MAINTENANCE = "schedule_maintenance"
    HEALTH_SNAPSHOT = "health_snapshot"


class MutationQueue:
    """Turn-scoped queue for all pending persistent mutations.

    Rules:
    - Inside SaveNode (commit_active=True): execute immediately or drain
    - Outside SaveNode: queue for SaveNode to drain
    - If queue cannot be drained at SaveNode: hard fail (nothing dropped silently)
    """

    def __init__(self) -> None:
        self._queue: list[MutationIntent] = []
        self._drained: list[MutationIntent] = []
        self._failed: list[tuple[MutationIntent, str]] = []
        self._owner_trace_id: str = ""
        self._sequence_counter: int = 0

    def bind_owner(self, trace_id: str) -> None:
        owner = str(trace_id or "").strip()
        if not owner:
            raise RuntimeError("MutationQueue owner binding requires non-empty trace_id")
        if self._owner_trace_id and self._owner_trace_id != owner:
            raise RuntimeError(
                "MutationQueue cross-turn reuse detected: "
                f"owner={self._owner_trace_id!r}, new_owner={owner!r}"
            )
        self._owner_trace_id = owner

    def _assert_owner(self) -> None:
        if not self._owner_trace_id:
            raise RuntimeError("MutationQueue is not bound to a turn trace_id")

    def queue(self, intent: MutationIntent) -> None:
        """Add a mutation intent to the pending queue."""
        self._assert_owner()
        if int(getattr(intent, "sequence_id", 0) or 0) <= 0:
            self._sequence_counter += 1
            intent.sequence_id = self._sequence_counter
        self._queue.append(intent)

    def pending(self) -> list[MutationIntent]:
        self._assert_owner()
        return list(self._queue)

    def is_empty(self) -> bool:
        self._assert_owner()
        return len(self._queue) == 0

    def size(self) -> int:
        self._assert_owner()
        return len(self._queue)

    def drain(
        self,
        executor: Callable[[MutationIntent], None],
        *,
        hard_fail_on_error: bool = True,
    ) -> list[tuple[MutationIntent, str]]:
        """Execute all queued intents through ``executor``.  Returns failures.

        If ``hard_fail_on_error=True`` (default for SaveNode), any failure
        raises ``RuntimeError`` immediately — nothing is silently dropped.
        """
        self._assert_owner()
        to_drain = sorted(
            list(self._queue),
            key=lambda m: (
                int(getattr(m, "priority", 0) or 0),
                int(getattr(m, "turn_index", 0) or 0),
                int(getattr(m, "sequence_id", 0) or 0),
            ),
        )
        self._queue.clear()
        for index, intent in enumerate(to_drain):
            try:
                executor(intent)
                self._drained.append(intent)
            except Exception as exc:
                failure = (intent, str(exc))
                self._failed.append(failure)
                # Preserve failed + remaining intents for explicit visibility.
                self._queue = [intent, *to_drain[index + 1 :], *self._queue]
                if hard_fail_on_error:
                    raise FatalTurnError(
                        f"MutationQueue drain failed at SaveNode — "
                        f"type={intent.type!r} source={intent.source!r}: {exc}"
                    ) from exc
        return list(self._failed)

    def snapshot(self) -> dict[str, Any]:
        self._assert_owner()
        pending_ledger = sum(1 for intent in self._queue if intent.type is MutationKind.LEDGER)
        drained_ledger = sum(1 for intent in self._drained if intent.type is MutationKind.LEDGER)
        failed_ledger = sum(1 for intent, _ in self._failed if intent.type is MutationKind.LEDGER)
        return {
            "owner_trace_id": self._owner_trace_id,
            # Contract counters keep non-ledger mutation accounting stable for
            # restart-boundary audits that verify durable mutation invariants.
            "pending": len(self._queue) - pending_ledger,
            "drained": len(self._drained) - drained_ledger,
            "failed": len(self._failed) - failed_ledger,
            # Ledger counters remain visible for observability.
            "ledger_pending": pending_ledger,
            "ledger_drained": drained_ledger,
            "ledger_failed": failed_ledger,
        }


@dataclass
class TurnFidelity:
    """Pipeline completeness record: tracks which canonical stages ran."""

    temporal: bool = False
    inference: bool = False
    reflection: bool = False
    save: bool = False

    @property
    def full_pipeline(self) -> bool:
        """True when all four canonical stages executed successfully."""
        return self.temporal and self.inference and self.reflection and self.save

    def to_dict(self) -> dict[str, Any]:
        return {
            "temporal": bool(self.temporal),
            "inference": bool(self.inference),
            "reflection": bool(self.reflection),
            "save": bool(self.save),
            "full_pipeline": bool(self.full_pipeline),
        }


class HealthStatus(StrEnum):
    OK = "OK"
    DEGRADED_CAPABILITY = "DEGRADED_CAPABILITY"
    DEGRADED_PERFORMANCE = "DEGRADED_PERFORMANCE"
    DEGRADED_STRUCTURE = "DEGRADED_STRUCTURE"


@dataclass
class TurnHealthState:
    """User-facing per-turn health telemetry derived from canonical stage timing."""

    status: str
    latency_ms: float
    memory_ops_time: float
    graph_sync_time: float
    inference_time: float
    fallback_used: bool = False
    fidelity: TurnFidelity = field(default_factory=TurnFidelity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status or HealthStatus.OK),
            "latency_ms": round(float(self.latency_ms or 0.0), 3),
            "memory_ops_time": round(float(self.memory_ops_time or 0.0), 3),
            "graph_sync_time": round(float(self.graph_sync_time or 0.0), 3),
            "inference_time": round(float(self.inference_time or 0.0), 3),
            "fallback_used": bool(self.fallback_used),
            "fidelity": self.fidelity.to_dict(),
        }


@dataclass(frozen=True)
class TurnTemporalAxis:
    """Frozen temporal base shared by every stage in a single turn."""

    turn_started_at: str
    wall_time: str
    wall_date: str
    timezone: str
    utc_offset_minutes: int
    epoch_seconds: float

    @classmethod
    def from_now(cls) -> "TurnTemporalAxis":
        now = datetime.now().astimezone().replace(microsecond=0)
        offset = now.utcoffset()
        offset_minutes = int(offset.total_seconds() // 60) if offset is not None else 0
        wall_time = now.isoformat(timespec="seconds")
        return cls(
            turn_started_at=wall_time,
            wall_time=wall_time,
            wall_date=now.date().isoformat(),
            timezone=str(now.tzname() or "local").strip() or "local",
            utc_offset_minutes=offset_minutes,
            epoch_seconds=now.timestamp(),
        )

    @classmethod
    def from_lock_hash(cls, lock_hash: str) -> "TurnTemporalAxis":
        """Derive a deterministic temporal axis from a lock hash.

        This is used by strict replay mode so identical lock payloads produce
        identical turn timestamps and event ordering metadata.
        """
        seed = str(lock_hash or "").strip().lower()
        if not seed:
            return cls.from_now()
        try:
            seed_int = int(seed[:16], 16)
        except ValueError:
            return cls.from_now()

        # Base at 2024-01-01T00:00:00+00:00 and keep deterministic offsets.
        base_epoch = 1704067200
        # Keep offset bounded to avoid far-future drift while remaining stable.
        offset_seconds = seed_int % (365 * 24 * 60 * 60)
        epoch = float(base_epoch + offset_seconds)
        dt = datetime.fromtimestamp(epoch).astimezone().replace(microsecond=0)
        offset = dt.utcoffset()
        offset_minutes = int(offset.total_seconds() // 60) if offset is not None else 0
        wall_time = dt.isoformat(timespec="seconds")
        return cls(
            turn_started_at=wall_time,
            wall_time=wall_time,
            wall_date=dt.date().isoformat(),
            timezone=str(dt.tzname() or "local").strip() or "local",
            utc_offset_minutes=offset_minutes,
            epoch_seconds=epoch,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_started_at": self.turn_started_at,
            "wall_time": self.wall_time,
            "wall_date": self.wall_date,
            "timezone": self.timezone,
            "utc_offset_minutes": self.utc_offset_minutes,
            "epoch_seconds": self.epoch_seconds,
        }


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
    # Canonical frozen temporal axis for the entire turn.
    temporal: TurnTemporalAxis = field(default_factory=TurnTemporalAxis.from_now)
    # User-facing health snapshot emitted once per turn.
    turn_health: TurnHealthState | None = None
    # Turn-scoped mutation queue: all persistent mutations queue here until SaveNode drains them.
    mutation_queue: MutationQueue = field(default_factory=MutationQueue)
    # Pipeline fidelity: which canonical stages ran this turn.
    fidelity: TurnFidelity = field(default_factory=TurnFidelity)

    def __post_init__(self) -> None:
        # Mutation queues are turn-scoped. Bind queue ownership to this trace.
        self.mutation_queue.bind_owner(self.trace_id)

    def temporal_snapshot(self) -> dict[str, Any]:
        return self.temporal.to_dict()

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
            "temporal": self.temporal_snapshot(),
            "stage_traces": [
                {"stage": t.stage, "duration_ms": t.duration_ms, "error": t.error}
                for t in self.stage_traces
            ],
            "turn_health_state": _json_safe(
                self.state.get("turn_health_state")
                or (self.turn_health.to_dict() if self.turn_health is not None else {})
            ),
            "ux_feedback": _json_safe(self.state.get("ux_feedback") or {}),
            "determinism_boundary": self.determinism_boundary.snapshot(),
        }

    def checkpoint_snapshot(self, *, stage: str, status: str, error: str | None = None) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "stage": str(stage or "unknown"),
            "status": str(status or "unknown"),
            "error": str(error or "").strip(),
            "updated_at": self.temporal.wall_time,
            "phase": self.phase.value,
            "user_input": self.user_input,
            "attachments": _json_safe(list(self.attachments or [])),
            "metadata": _json_safe(self.metadata),
            "state": _json_safe(self.state),
            "temporal": _json_safe(self.temporal_snapshot()),
            "event_sequence": int(self.event_sequence),
            "phase_history": _json_safe(self.phase_history),
            "stage_traces": [
                {"stage": trace.stage, "duration_ms": trace.duration_ms, "error": trace.error}
                for trace in self.stage_traces
            ],
            "turn_health_state": _json_safe(
                self.state.get("turn_health_state")
                or (self.turn_health.to_dict() if self.turn_health is not None else {})
            ),
            "ux_feedback": _json_safe(self.state.get("ux_feedback") or {}),
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
        finalize = getattr(service, "finalize_turn", None)
        if callable(finalize):
            try:
                turn_context.state["safe_result"] = finalize(turn_context, result)
                return
            except Exception:
                pass
        service.save_turn(turn_context, result)


class TemporalNode:
    name = "temporal"

    async def execute(self, _registry: Any, turn_context: TurnContext) -> None:
        if getattr(turn_context, "temporal", None) is None:
            raise RuntimeError("TemporalNode missing — deterministic execution violated")
        temporal_payload = turn_context.temporal_snapshot()
        turn_context.state.setdefault("temporal", temporal_payload)
        turn_context.metadata.setdefault("temporal", temporal_payload)


class ReflectionNode:
    name = "reflection"

    async def execute(self, registry: Any, turn_context: TurnContext) -> None:
        try:
            service = registry.get("reflection")
        except Exception:
            return
        result = turn_context.state.get("safe_result") or turn_context.state.get("candidate")
        turn_text = turn_context.state.get("turn_text") or turn_context.user_input
        current_mood = turn_context.state.get("mood") or "neutral"
        reply_text = result[0] if isinstance(result, tuple) else str(result or "")

        reflect_after_turn = getattr(service, "reflect_after_turn", None)
        if callable(reflect_after_turn):
            try:
                turn_context.state["reflection"] = reflect_after_turn(turn_text, current_mood, reply_text)
            except TypeError:
                turn_context.state["reflection"] = reflect_after_turn(turn_context, result)
            return

        reflect = getattr(service, "reflect", None)
        if callable(reflect):
            try:
                turn_context.state["reflection"] = reflect(turn_context, result)
            except TypeError:
                try:
                    turn_context.state["reflection"] = reflect(force=True)
                except TypeError:
                    turn_context.state["reflection"] = reflect(turn_context)


class TurnGraph:
    """Declarative turn execution graph."""

    def __init__(self, registry: Any = None, nodes: list[GraphNode] | None = None):
        self.registry = registry
        self.nodes = nodes or [
            TemporalNode(),
            HealthNode(),
            ContextBuilderNode(),
            InferenceNode(),
            SafetyNode(),
            ReflectionNode(),
            SaveNode(),
        ]
        self._node_map: dict[str, Any] = {}
        self._edges: dict[str, str] = {}
        self._entry_node: str | None = None
        self._kernel = None  # TurnKernel | None
        self._required_execution_token: str = ""
        self._execution_witness_emitter: Callable[[str, TurnContext], None] | None = None
        self._degraded_latency_threshold_ms = 2500.0
        self._degraded_inference_threshold_ms = 2200.0
        self._degraded_memory_threshold_ms = 1200.0
        self._degraded_graph_sync_threshold_ms = 1200.0
        self._execution_kernel = ExecutionKernel(
            firewall=ExecutionFirewall(),
            invariant_registry=InvariantRegistry(),
            quarantine=None,
            strict=False,
        )

    def set_execution_kernel(self, kernel: ExecutionKernel) -> None:
        self._execution_kernel = kernel

    def _pipeline_items(self) -> list[tuple[str, Any]]:
        if self._node_map:
            items: list[tuple[str, Any]] = []
            node_name = self._entry_node
            while node_name is not None:
                items.append((node_name, self._node_map[node_name]))
                node_name = self._edges.get(node_name)
            return items
        return [
            (str(getattr(node, "name", type(node).__name__) or type(node).__name__), node)
            for node in self.nodes
        ]

    @staticmethod
    def _stage_duration_ms(turn_context: TurnContext, stage_name: str) -> float:
        total = 0.0
        for trace in list(turn_context.stage_traces or []):
            if str(getattr(trace, "stage", "") or "").strip().lower() == str(stage_name or "").strip().lower():
                total += float(getattr(trace, "duration_ms", 0.0) or 0.0)
        return round(total, 3)

    def _emit_turn_health_state(self, turn_context: TurnContext, *, total_latency_ms: float, failed: bool) -> None:
        memory_ops_ms = float(turn_context.state.get("_timing_memory_ops_ms") or 0.0)
        graph_sync_ms = float(turn_context.state.get("_timing_graph_sync_ms") or 0.0)
        inference_ms = self._stage_duration_ms(turn_context, "inference")

        degraded_performance = any(
            [
                total_latency_ms >= self._degraded_latency_threshold_ms,
                inference_ms >= self._degraded_inference_threshold_ms,
                memory_ops_ms >= self._degraded_memory_threshold_ms,
                graph_sync_ms >= self._degraded_graph_sync_threshold_ms,
            ]
        )
        fallback_used = bool(turn_context.state.get("fallback_used", False))
        degraded_capability = bool(failed or fallback_used or turn_context.state.get("_capability_degraded", False))
        degraded_structure = bool(turn_context.state.get("_structural_degradation", False))
        if degraded_structure:
            status = HealthStatus.DEGRADED_STRUCTURE
        elif degraded_capability:
            status = HealthStatus.DEGRADED_CAPABILITY
        elif degraded_performance:
            status = HealthStatus.DEGRADED_PERFORMANCE
        else:
            status = HealthStatus.OK

        # Update turn fidelity from completed stage traces.
        stage_order = [str(trace.stage or "") for trace in list(turn_context.stage_traces or [])]
        fidelity = turn_context.fidelity
        fidelity.temporal = "temporal" in stage_order or any(
            "temporal" in str(s) for s in turn_context.state.get("_graph_executed_stages") or set()
        ) or bool(turn_context.state.get("temporal"))
        fidelity.inference = "inference" in stage_order
        fidelity.reflection = "reflection" in stage_order
        fidelity.save = "save" in stage_order

        health = TurnHealthState(
            status=status,
            latency_ms=total_latency_ms,
            memory_ops_time=memory_ops_ms,
            graph_sync_time=graph_sync_ms,
            inference_time=inference_ms,
            fallback_used=fallback_used,
            fidelity=fidelity,
        )
        health_payload = health.to_dict()
        evidence = {
            "stage_order": stage_order,
            "save_node_executed": stage_order.count("save") == 1,
            "temporal_enforced": bool(fidelity.temporal),
            "pipeline_fidelity": fidelity.to_dict(),
            "mutation_queue": turn_context.mutation_queue.snapshot(),
            "trace_id": str(turn_context.trace_id or ""),
            "health_status_tier": str(status),
        }

        thinking = any(
            [
                inference_ms >= self._degraded_inference_threshold_ms,
                memory_ops_ms >= self._degraded_memory_threshold_ms,
                graph_sync_ms >= self._degraded_graph_sync_threshold_ms,
            ]
        )
        checking_memory = bool(
            memory_ops_ms >= self._degraded_memory_threshold_ms
            or graph_sync_ms >= self._degraded_graph_sync_threshold_ms
        )
        mood_hint = str(turn_context.state.get("mood") or "neutral")
        ux_feedback = {
            "dad_is_thinking": bool(thinking),
            "message": "Dad is thinking..." if thinking else "",
            "checking_memory": checking_memory,
            "memory_message": "Checking memory..." if checking_memory else "",
            "mood_hint": mood_hint,
            "status": status,
        }

        turn_context.turn_health = health
        turn_context.state["turn_health_state"] = health_payload
        turn_context.metadata["turn_health_state"] = dict(health_payload)
        turn_context.state["turn_health_evidence"] = evidence
        turn_context.metadata["turn_health_evidence"] = dict(evidence)
        turn_context.state["ux_feedback"] = ux_feedback
        turn_context.metadata["ux_feedback"] = dict(ux_feedback)

    @staticmethod
    def _mark_structural_degradation(turn_context: TurnContext, reason: str) -> None:
        state = getattr(turn_context, "state", None)
        if not isinstance(state, dict):
            return
        state["_structural_degradation"] = True
        health = dict(state.get("turn_health_state") or {})
        if health:
            health["status"] = str(HealthStatus.DEGRADED_STRUCTURE)
            state["turn_health_state"] = health
        evidence = dict(state.get("turn_health_evidence") or {})
        evidence["fidelity_degraded_reason"] = str(reason or "structural_invariant_violation")
        evidence["health_status_tier"] = str(HealthStatus.DEGRADED_STRUCTURE)
        state["turn_health_evidence"] = evidence

    @staticmethod
    def _mark_stage_enter(turn_context: TurnContext, stage_name: str) -> None:
        executed = turn_context.state.setdefault("_graph_executed_stages", set())
        if not isinstance(executed, set):
            executed = set(executed) if isinstance(executed, (list, tuple)) else set()
            turn_context.state["_graph_executed_stages"] = executed
        if stage_name in executed:
            raise RuntimeError(
                f"TurnGraph execution violation: stage {stage_name!r} executed more than once in trace {turn_context.trace_id!r}"
            )

        last_stage = str(turn_context.state.get("_graph_last_stage") or "").strip()
        expected_next = {
            "": {"preflight", "health", "temporal", "context_builder"},
            "preflight": {"inference"},
            "health": {"context_builder"},
            # temporal is now always sequential-first; it may be followed by the parallel
            # preflight group or by health/context_builder individually.
            "temporal": {"preflight", "health", "context_builder"},
            "context_builder": {"inference"},
            "inference": {"safety"},
            "safety": {"reflection"},
            "reflection": {"save"},
            "save": set(),
        }
        # Only enforce canonical pipeline ordering when the incoming stage is a
        # known canonical stage.  Custom / test nodes with arbitrary names are
        # allowed to execute without triggering the ordering gate.
        _canonical = set(expected_next.keys()) | {s for vals in expected_next.values() for s in vals}
        if stage_name in _canonical:
            allowed = expected_next.get(last_stage)
            if allowed is not None and allowed and stage_name not in allowed:
                raise RuntimeError(
                    "TurnGraph order violation: "
                    f"stage {stage_name!r} cannot execute after {last_stage!r}; expected one of {sorted(allowed)!r}"
                )

        executed.add(stage_name)
        turn_context.state["_graph_last_stage"] = stage_name
        turn_context.state["_active_graph_stage"] = stage_name

        # Hard invariant: TemporalNode must run before any mutation-capable stage.
        # Allow temporal itself, preflight group members, and health/context to proceed.
        # Any stage that can write state (inference, safety, reflection, save) must
        # only execute after temporal has populated turn_context.state["temporal"].
        _requires_temporal = {"inference", "safety", "reflection", "save"}
        if stage_name in _requires_temporal:
            if not turn_context.state.get("temporal"):
                raise RuntimeError(
                    f"TemporalNode not initialized before {stage_name!r} — "
                    "deterministic execution violated: temporal must be first in pipeline"
                )

    def add_node(self, name: str, node: Any) -> None:
        # Kernel in shadow mode during graph construction.
        shadow_context = TurnContext(user_input="graph_build")
        result = self._execution_kernel.validate(
            stage="graph_build",
            operation=f"turn_graph.add_node:{name}",
            context=shadow_context,
        )
        if not bool(getattr(result, "ok", True)):
            logger.warning("[KERNEL SHADOW VIOLATION] %s", getattr(result, "reason", "graph build validation failed"))
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

        node_name = str(getattr(node, "name", type(node).__name__) or type(node).__name__)

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
            save_checkpoint(checkpoint, _skip_turn_event=True)

        save_event = getattr(service, "save_turn_event", None)
        if callable(save_event):
            save_event(
                {
                    "event_type": "graph_checkpoint",
                    "trace_id": turn_context.trace_id,
                    "sequence": turn_context.event_sequence,
                    "occurred_at": turn_context.temporal.wall_time,
                    "stage": str(stage),
                    "status": str(status),
                    "error": str(error or "").strip(),
                    "phase": turn_context.phase.value,
                    "active_stage": str(turn_context.state.get("_active_graph_stage") or ""),
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
                        "occurred_at": turn_context.temporal.wall_time,
                        "stage": str(stage),
                        "phase": turn_context.phase.value,
                        "transition": dict(transition),
                        "determinism_lock": _json_safe(determinism_lock),
                    }
                )

    def _emit_capability_audit_report(
        self,
        turn_context: TurnContext,
        *,
        stage_order: list[str],
        failed: bool,
        error: str = "",
    ) -> dict[str, Any]:
        report = build_runtime_capability_audit_report(
            turn_context,
            stage_order=stage_order,
            failed=failed,
            error=error,
        )
        payload = report.to_dict()
        turn_context.state["capability_audit_report"] = payload
        turn_context.metadata["capability_audit_report"] = dict(payload)
        logger.info("capability_audit_report=%s", json.dumps(payload, sort_keys=True, default=str))
        return payload

    async def execute(self, turn_context: TurnContext, *, audit_mode: bool = False) -> FinalizedTurnResult:
        if self._required_execution_token:
            active_token = ControlPlaneExecutionBoundary.current()
            if active_token != self._required_execution_token:
                raise RuntimeError(
                    "TurnGraph.execute boundary violation: graph execution must be routed through ExecutionControlPlane"
                )

        self._emit_execution_witness("graph.execute", turn_context)
        turn_context.state["_execution_kernel"] = self._execution_kernel
        execute_started_at = time.perf_counter()

        def _enforce_fidelity_invariants() -> None:
            fidelity = getattr(turn_context, "fidelity", None)
            if fidelity is None:
                self._mark_structural_degradation(turn_context, "fidelity_state_missing")
                raise RuntimeError("Structural turn invariant violated: turn fidelity state missing")
            if not bool(getattr(fidelity, "temporal", False)):
                self._mark_structural_degradation(turn_context, "temporal_stage_missing")
                raise RuntimeError("Structural turn invariant violated: TemporalNode missing")
            if not bool(getattr(fidelity, "save", False)):
                self._mark_structural_degradation(turn_context, "save_stage_missing")
                raise RuntimeError("Structural turn invariant violated: SaveNode did not execute")
            # Strict mode: inference must always execute in the canonical pipeline.
            # Only enforce this when the graph was built with an "inference" node
            # so that unit-test graphs with partial pipelines are not rejected.
            if "inference" in self._node_map and not bool(getattr(fidelity, "inference", False)):
                self._mark_structural_degradation(turn_context, "inference_stage_missing")
                raise RuntimeError("Structural turn invariant violated: InferenceNode missing")

        telemetry = self.registry.get("telemetry") if self.registry is not None else None
        executed_stage_names: list[str] = []
        audit_enabled = bool(audit_mode or turn_context.metadata.get("audit_mode"))

        async def _execute_stage(stage_name: str, node: Any, stage_context: TurnContext) -> TurnContext:
            started_at = time.perf_counter()
            error_msg: str | None = None
            self._mark_stage_enter(stage_context, stage_name)
            self._sync_stage_phase(stage_context, stage=stage_name)
            self._emit_checkpoint(stage_context, stage=stage_name, status="before")
            if telemetry is not None:
                telemetry.trace("turn_graph.node_start", node=stage_name, trace_id=stage_context.trace_id)
            try:
                stage_context = await self._execute_node(node, stage_context)
            except Exception as exc:
                error_msg = str(exc)
                self._emit_checkpoint(stage_context, stage=stage_name, status="error", error=error_msg)
                raise
            finally:
                stage_context.state["_active_graph_stage"] = ""
                duration_ms = round((time.perf_counter() - started_at) * 1000, 3)
                stage_context.stage_traces.append(
                    StageTrace(stage=stage_name, duration_ms=duration_ms, error=error_msg)
                )
                if telemetry is not None:
                    telemetry.trace(
                        "turn_graph.node_done",
                        node=stage_name,
                        duration_ms=duration_ms,
                        trace_id=stage_context.trace_id,
                    )
            if error_msg is None:
                self._emit_checkpoint(stage_context, stage=stage_name, status="after")
                executed_stage_names.append(stage_name)
            return stage_context

        try:
            pipeline = self._pipeline_items()
            turn_context = await self._execution_kernel.run(turn_context, pipeline, _execute_stage)
            for idx, stage_name in enumerate(executed_stage_names[:-1]):
                next_name = executed_stage_names[idx + 1]
                self._emit_checkpoint(turn_context, stage=f"{stage_name}\u2192{next_name}", status="edge")
        except Exception as exc:
            total_ms = round((time.perf_counter() - execute_started_at) * 1000, 3)
            self._emit_turn_health_state(turn_context, total_latency_ms=total_ms, failed=True)
            if audit_enabled:
                self._emit_capability_audit_report(
                    turn_context,
                    stage_order=[trace.stage for trace in list(turn_context.stage_traces or [])],
                    failed=True,
                    error=str(exc),
                )
            raise

        if turn_context.short_circuit and turn_context.short_circuit_result is not None:
            total_ms = round((time.perf_counter() - execute_started_at) * 1000, 3)
            self._emit_turn_health_state(turn_context, total_latency_ms=total_ms, failed=False)
            _enforce_fidelity_invariants()
            if audit_enabled:
                self._emit_capability_audit_report(turn_context, stage_order=executed_stage_names, failed=False)
            return turn_context.short_circuit_result

        result = turn_context.state.get("safe_result")
        total_ms = round((time.perf_counter() - execute_started_at) * 1000, 3)
        self._emit_turn_health_state(turn_context, total_latency_ms=total_ms, failed=False)
        _enforce_fidelity_invariants()
        if audit_enabled:
            self._emit_capability_audit_report(turn_context, stage_order=executed_stage_names, failed=False)
        if isinstance(result, tuple) and len(result) >= 2:
            return result
        return (str(result or ""), False)
