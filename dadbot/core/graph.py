from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from dadbot.contracts import AttachmentList, ChunkCallback, FinalizedTurnResult
from dadbot.core.capability_audit_runner import build_runtime_capability_audit_report
from dadbot.core.determinism import DeterminismBoundary, DeterminismMode
from dadbot.core.execution_boundary import ControlPlaneExecutionBoundary
from dadbot.core.execution_firewall import ExecutionFirewall
from dadbot.core.execution_kernel import ExecutionKernel
from dadbot.core.execution_kernel_spec import validate_execution_kernel_spec
from dadbot.core.invariant_registry import InvariantRegistry
from dadbot.core.kernel import TurnKernel
from dadbot.core.execution_identity import ExecutionIdentity, ExecutionIdentityViolation
from dadbot.core.capability_registry import (
    CapabilityRegistry,
    CapabilityViolationError,
)
from dadbot.core.determinism_seal import DEFAULT_SEAL as _DETERMINISM_SEAL
from dadbot.core.execution_receipt import (
    DEFAULT_SIGNER as _DEFAULT_RECEIPT_SIGNER,
    ReceiptSigner,
)
from dadbot.core.execution_policy import (
    ExecutionPolicyEngine,
    FatalTurnError,
    KernelRejectionSemantics,
    PersistenceServiceContract,
    ResumabilityPolicy,
    StagePhaseMappingPolicy,
    TurnFailureSeverity,
)
from dadbot.core.execution_recovery import ExecutionRecovery
from dadbot.core.turn_resume_store import TurnResumeStore
from dadbot.core.execution_trace_schema import stamp_trace_contract_version
from dadbot.core.graph_side_effects import GraphSideEffectsOrchestrator
from dadbot.core.execution_policy_service import StageEntryGate
from dadbot.core.side_effect_adapter import SideEffectAdapter
from dadbot.core.persistence_event_adapter import GraphPersistenceEventAdapter  # re-export
from dadbot.core.topology_provider import TopologyProvider, TopologyProviderFactory, TurnPipelineState
from dadbot.core.ux_projection import TurnHealthState, TurnUxProjector  # re-export


logger = logging.getLogger(__name__)


class _LinearTopologyGraph:
    """Minimal interface-only topology runtime used as a local fallback.

    This keeps TurnGraph decoupled from concrete topology engines and
    preserves direct TurnGraph() test usage when no provider is injected.
    """

    def __init__(
        self,
        pipeline_items: list[tuple[str, Any]],
        node_executor: Callable[[str, Any, TurnPipelineState], Any],
    ) -> None:
        self._pipeline_items = pipeline_items
        self._node_executor = node_executor

    async def ainvoke(self, state: TurnPipelineState) -> TurnPipelineState:
        current_state = dict(state)
        for stage_name, node in self._pipeline_items:
            updates = await self._node_executor(stage_name, node, current_state)
            if isinstance(updates, dict):
                current_state.update(updates)
            if current_state.get("abort") or current_state.get("short_circuit"):
                break
        return current_state  # type: ignore[return-value]


class _LinearTopologyProvider(TopologyProvider):
    def __init__(
        self,
        pipeline_items: list[tuple[str, Any]],
        node_executor: Callable[[str, Any, TurnPipelineState], Any],
    ) -> None:
        self._pipeline_items = pipeline_items
        self._node_executor = node_executor

    def build(self) -> _LinearTopologyGraph:
        return _LinearTopologyGraph(self._pipeline_items, self._node_executor)


def _default_topology_provider_factory(
    pipeline_items: list[tuple[str, Any]],
    node_executor: Callable[[str, Any, TurnPipelineState], Any],
) -> TopologyProvider:
    return _LinearTopologyProvider(pipeline_items, node_executor)

# Canonical durability boundary contract for the execution graph.
SAVE_NODE_COMMIT_CONTRACT = (
    "All durable mutations MUST go through SaveNode. "
    "The graph guarantees speculative execution until that boundary."
)

# ---------------------------------------------------------------------------
# Mutation intent payload schema (pre-commit validation gate)
# ---------------------------------------------------------------------------

# Required top-level keys per mutation kind (runtime-enforced)
_INTENT_REQUIRED_PAYLOAD_KEYS: dict[str, list[str]] = {
    "memory": ["temporal"],
    "relationship": ["temporal"],
    "graph": [],
    "ledger": ["temporal"],
    "goal": [],
}


def _validate_mutation_intent_payload(intent: "MutationIntent") -> None:
    """Enforce per-kind required-field schema on a MutationIntent payload.

    Raises ``ValueError`` when a required field is absent, converting silent
    data corruption into an explicit pre-commit schema failure.
    """
    kind = str(intent.type).lower()
    required = _INTENT_REQUIRED_PAYLOAD_KEYS.get(kind, [])
    # Only validate temporal when requires_temporal=True — matches existing guard.
    if not bool(intent.requires_temporal) and "temporal" in required:
        required = [k for k in required if k != "temporal"]
    for key in required:
        if key not in intent.payload:
            raise ValueError(
                f"MutationIntent({kind!r}) is missing required payload key: {key!r}"
            )


# FatalTurnError, TurnFailureSeverity, KernelRejectionSemantics,
# PersistenceServiceContract, ExecutionPolicyEngine, StagePhaseMappingPolicy
# are defined in dadbot.core.execution_policy and imported above.
# They are re-exported here for backward compatibility with code that imports
# them from dadbot.core.graph.


@dataclass(frozen=True)
class ExecutionTraceEvent:
    """Deterministic execution event used for replay equivalence checks."""

    sequence: int
    event_type: str
    stage: str
    phase: str
    trace_id: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": int(self.sequence),
            "event_type": str(self.event_type or ""),
            "stage": str(self.stage or ""),
            "phase": str(self.phase or ""),
            "trace_id": str(self.trace_id or ""),
            "detail": _json_safe(dict(self.detail or {})),
        }


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
    # Optional compensator used by transactional drain rollback.
    compensator: Callable[[], None] | None = field(default=None, repr=False)
    # Content hash computed in __post_init__; not part of __init__ signature.
    payload_hash: str = field(default="", init=False, repr=False)

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
        elif self.type is MutationKind.GOAL:
            if op and op not in {item.value for item in GoalMutationOp}:
                raise RuntimeError(f"Unsupported goal mutation op: {op!r}")
        # Schema validation: enforce required fields per mutation kind.
        _validate_mutation_intent_payload(self)
        # Payload content hash for audit/replay integrity.
        self.payload_hash = hashlib.sha256(
            json.dumps(self.payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:24]


class MutationKind(StrEnum):
    MEMORY = "memory"
    RELATIONSHIP = "relationship"
    GRAPH = "graph"
    LEDGER = "ledger"
    GOAL = "goal"


class MemoryMutationOp(StrEnum):
    SAVE_MOOD_STATE = "save_mood_state"


class RelationshipMutationOp(StrEnum):
    UPDATE = "update"


class LedgerMutationOp(StrEnum):
    APPEND_HISTORY = "append_history"
    RECORD_TURN_STATE = "record_turn_state"
    SYNC_THREAD_SNAPSHOT = "sync_thread_snapshot"
    CLEAR_TURN_CONTEXT = "clear_turn_context"
    SCHEDULE_MAINTENANCE = "schedule_maintenance"
    HEALTH_SNAPSHOT = "health_snapshot"
    CAPABILITY_AUDIT_EVENT = "capability_audit_event"


class GoalMutationOp(StrEnum):
    UPSERT_GOAL = "upsert_goal"
    COMPLETE_GOAL = "complete_goal"
    ABANDON_GOAL = "abandon_goal"


class NodeType(StrEnum):
    STANDARD = "standard"
    COMMIT = "commit"


class MutationTransactionStatus(StrEnum):
    COMMITTED = "committed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    ROLLBACK_FAILED = "rollback_failed"


@dataclass
class MutationTransactionRecord:
    transaction_id: str
    status: MutationTransactionStatus
    applied_count: int
    failed_count: int
    rollback_count: int
    rollback_failures: int
    trace_id: str
    error: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": str(self.transaction_id or ""),
            "status": str(self.status.value),
            "applied_count": int(self.applied_count),
            "failed_count": int(self.failed_count),
            "rollback_count": int(self.rollback_count),
            "rollback_failures": int(self.rollback_failures),
            "trace_id": str(self.trace_id or ""),
            "error": str(self.error or ""),
            "created_at": float(self.created_at),
        }



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
        self._transactions: list[MutationTransactionRecord] = []
        self._owner_trace_id: str = ""
        self._sequence_counter: int = 0
        self._mutations_locked: bool = False

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
        if self._mutations_locked:
            raise RuntimeError(
                "MutationGuard violation: mutation attempted outside SaveNode "
                f"(type={getattr(intent, 'type', '?')!r})"
            )
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
        executor: Callable[[MutationIntent], Any],
        *,
        hard_fail_on_error: bool = True,
        transactional: bool = True,
    ) -> list[tuple[MutationIntent, str]]:
        """Execute all queued intents through ``executor``. Returns failures.

        If ``hard_fail_on_error=True`` (default for SaveNode), failures raise
        ``FatalTurnError``. With ``transactional=True``, hard failures attempt
        rollback and re-queue the full transaction for replay-safe retries.
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

        tx_id = uuid.uuid4().hex
        applied: list[tuple[MutationIntent, Callable[[], None] | None]] = []
        rollback_count = 0
        rollback_failures = 0

        for index, intent in enumerate(to_drain):
            try:
                result = executor(intent)
                compensator = result if callable(result) else getattr(intent, "compensator", None)
                if compensator is not None and not callable(compensator):
                    compensator = None
                applied.append((intent, compensator))
                self._drained.append(intent)
            except Exception as exc:
                failure = (intent, str(exc))
                self._failed.append(failure)

                if not hard_fail_on_error:
                    self._queue = [intent, *to_drain[index + 1 :], *self._queue]
                    continue

                if transactional:
                    for applied_intent, compensator in reversed(applied):
                        if callable(compensator):
                            try:
                                compensator()
                                rollback_count += 1
                            except Exception:
                                rollback_failures += 1

                    self._queue = [*to_drain, *self._queue]
                    for applied_intent, _ in applied:
                        with contextlib.suppress(ValueError):
                            self._drained.remove(applied_intent)
                    status = (
                        MutationTransactionStatus.ROLLBACK_FAILED
                        if rollback_failures
                        else MutationTransactionStatus.ROLLED_BACK
                    )
                    self._transactions.append(
                        MutationTransactionRecord(
                            transaction_id=tx_id,
                            status=status,
                            applied_count=len(applied),
                            failed_count=1,
                            rollback_count=rollback_count,
                            rollback_failures=rollback_failures,
                            trace_id=self._owner_trace_id,
                            error=str(exc),
                        )
                    )
                else:
                    self._queue = [intent, *to_drain[index + 1 :], *self._queue]

                raise FatalTurnError(
                    f"MutationQueue drain failed at SaveNode - "
                    f"type={intent.type!r} source={intent.source!r}: {exc}"
                ) from exc

        if to_drain:
            self._transactions.append(
                MutationTransactionRecord(
                    transaction_id=tx_id,
                    status=MutationTransactionStatus.COMMITTED,
                    applied_count=len(applied),
                    failed_count=0,
                    rollback_count=0,
                    rollback_failures=0,
                    trace_id=self._owner_trace_id,
                )
            )
        return list(self._failed)

    def snapshot(self) -> dict[str, Any]:
        self._assert_owner()
        pending_ledger = sum(1 for intent in self._queue if intent.type is MutationKind.LEDGER)
        drained_ledger = sum(1 for intent in self._drained if intent.type is MutationKind.LEDGER)
        failed_ledger = sum(1 for intent, _ in self._failed if intent.type is MutationKind.LEDGER)
        # Strip created_at and transaction_id: both are non-deterministic across
        # separate executions (wall-clock time and UUID4 respectively) and must
        # not appear in determinism-audit snapshots compared across runs.
        latest_tx_raw = self._transactions[-1].to_dict() if self._transactions else {}
        latest_tx = {k: v for k, v in latest_tx_raw.items() if k not in ("created_at", "transaction_id")}
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
            "transactions": len(self._transactions),
            "latest_transaction": latest_tx,
        }


class MutationGuard:
    """Context manager that blocks mutation queueing outside SaveNode.

    Wrap every non-SaveNode stage execution with this guard to enforce the
    invariant that only SaveNode may commit mutations at runtime. Any attempt
    to call ``MutationQueue.queue()`` while the guard is active raises
    ``RuntimeError``, converting a convention into an enforced runtime contract.
    """

    def __init__(self, mutation_queue: "MutationQueue") -> None:
        self._queue = mutation_queue

    def __enter__(self) -> "MutationGuard":
        self._queue._mutations_locked = True
        return self

    def __exit__(self, *_: Any) -> None:
        self._queue._mutations_locked = False


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


@dataclass
class VirtualClock:
    """Deterministic seeded clock for eliminating wall-time nondeterminism in replay/tests.

    Assign to ``TurnContext.virtual_clock`` before turn execution.  ``TemporalNode``
    will call ``tick()`` and derive the turn's ``TurnTemporalAxis`` from the virtual
    timestamp instead of the real wall clock, making temporal fields 100% reproducible
    across replay runs.

    Usage::

        vc = VirtualClock(base_epoch=1_700_000_000.0, step_size_seconds=30.0)
        ctx.virtual_clock = vc
        # First tick => epoch=1_700_000_030.0 (base + 1 step)
    """

    base_epoch: float = field(default_factory=time.time)
    step_size_seconds: float = 1.0
    _step: int = field(default=0, init=False, repr=False)

    def now(self) -> float:
        """Current virtual epoch (does NOT advance the clock)."""
        return self.base_epoch + self._step * self.step_size_seconds

    def tick(self) -> float:
        """Advance clock by one step and return the new epoch."""
        self._step += 1
        return self.now()

    def to_datetime(self) -> "datetime":
        """Convert current virtual epoch to a timezone-aware datetime."""
        return datetime.fromtimestamp(self.now()).astimezone().replace(microsecond=0)


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
        # Always derive the integer via sha256 so that any seed string
        # (including non-hex ones like "phase4-lock-0001") produces a
        # deterministic temporal axis without silently falling back to the
        # real wall clock.
        seed_int = int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16)
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
    # Optional deterministic virtual clock; TemporalNode uses it when set instead of wall time.
    virtual_clock: "VirtualClock | None" = field(default=None)
    # Hash-chain pointers for checkpoint integrity across load/save boundaries.
    last_checkpoint_hash: str = field(default="", init=False)
    prev_checkpoint_hash: str = field(default="", init=False)

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
            "last_checkpoint_hash": self.last_checkpoint_hash,
            "prev_checkpoint_hash": self.prev_checkpoint_hash,
        }

    def checkpoint_snapshot(
        self,
        *,
        stage: str,
        status: str,
        error: str | None = None,
        advance_chain: bool = True,
    ) -> dict[str, Any]:
        chain_prev = str(self.last_checkpoint_hash or self.prev_checkpoint_hash or "")
        snapshot = {
            "trace_id": self.trace_id,
            "stage": str(stage or "unknown"),
            "status": str(status or "unknown"),
            "error": str(error or "").strip(),
            "updated_at": self.temporal.wall_time,
            "phase": self.phase.value,
            "user_input": self.user_input,
            "attachments": _json_safe(list(self.attachments or [])),
            "metadata": _json_safe(self.metadata),
            # Apply determinism seal before hashing: normalise floats and tool output
            # so identical logical state always produces the same checkpoint hash.
            "state": _json_safe(_DETERMINISM_SEAL.apply(self.state)),
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
            # Hash chain: each checkpoint includes the previous checkpoint's hash so
            # the full sequence forms a tamper-evident integrity chain.
            "prev_checkpoint_hash": chain_prev,
        }
        # Compute hash over the lightweight header fields (not the full state blob).
        _chain_payload = {
            "trace_id": self.trace_id,
            "stage": snapshot["stage"],
            "status": snapshot["status"],
            "event_sequence": snapshot["event_sequence"],
            "prev_checkpoint_hash": chain_prev,
        }
        checkpoint_hash = hashlib.sha256(
            json.dumps(_chain_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:32]
        snapshot["checkpoint_hash"] = checkpoint_hash
        if advance_chain:
            # Advance the chain for the next durable checkpoint edge.
            self.prev_checkpoint_hash = chain_prev
            self.last_checkpoint_hash = checkpoint_hash
        return snapshot

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
    node_type = NodeType.COMMIT

    async def execute(self, registry: Any, turn_context: TurnContext) -> None:
        service = registry.get("persistence_service")
        result = turn_context.state.get("safe_result")
        finalize = getattr(service, "finalize_turn", None)
        if callable(finalize):
            try:
                turn_context.state["safe_result"] = finalize(turn_context, result)
                # Ensure hash-chain checkpoints are also emitted from SaveNode so
                # commit-time state is captured inside the finalize boundary.
                save_checkpoint = getattr(service, "save_graph_checkpoint", None)
                if callable(save_checkpoint):
                    checkpoint = turn_context.checkpoint_snapshot(
                        stage="save",
                        status="atomic_commit",
                        error=None,
                    )
                    save_checkpoint(checkpoint, _skip_turn_event=True)
                return
            except Exception:
                pass
        service.save_turn(turn_context, result)


class TemporalNode:
    name = "temporal"
    node_type = NodeType.STANDARD

    async def execute(self, _registry: Any, turn_context: TurnContext) -> None:
        if getattr(turn_context, "temporal", None) is None:
            raise RuntimeError("TemporalNode missing — deterministic execution violated")
        temporal_payload = turn_context.temporal_snapshot()
        turn_context.state.setdefault("temporal", temporal_payload)
        turn_context.metadata.setdefault("temporal", temporal_payload)


class ReflectionNode:
    name = "reflection"
    node_type = NodeType.STANDARD

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

    def __init__(
        self,
        registry: Any = None,
        nodes: list[GraphNode] | None = None,
        *,
        topology_provider_factory: TopologyProviderFactory | None = None,
    ):
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
        validate_execution_kernel_spec(self._execution_kernel, raise_on_failure=True)
        self._persistence_contract = PersistenceServiceContract()
        self._policy_engine = ExecutionPolicyEngine(
            persistence_contract=self._persistence_contract,
        )
        self._persistence_event_adapter = GraphPersistenceEventAdapter(json_safe=_json_safe)
        self._ux_projector = TurnUxProjector(
            degraded_latency_threshold_ms=self._degraded_latency_threshold_ms,
            degraded_inference_threshold_ms=self._degraded_inference_threshold_ms,
            degraded_memory_threshold_ms=self._degraded_memory_threshold_ms,
            degraded_graph_sync_threshold_ms=self._degraded_graph_sync_threshold_ms,
        )
        # Single side-effects orchestrator — the graph routes all persistence and
        # UX side effects through this object, not directly to the adapters.
        self._side_effects = GraphSideEffectsOrchestrator(
            persistence_event_adapter=self._persistence_event_adapter,
            ux_projector=self._ux_projector,
            policy_engine=self._policy_engine,
            json_safe=_json_safe,
        )
        # Stage-entry policy gate: idempotency, capability, and ordering checks.
        self._stage_entry_gate = StageEntryGate()
        # Side-effect adapter: all record-and-emit writes go through this layer.
        self._side_effect_adapter = SideEffectAdapter()
        # Durable execution: crash-safe resume.  None unless configure_resume() is called.
        self._recovery: ExecutionRecovery | None = None
        # Capability security: stage-level enforcement.  None unless configure_capabilities() is called.
        self._capability_registry: CapabilityRegistry | None = None
        self._capability_policy: Any | None = None   # SessionAuthorizationPolicy | None
        self._capability_session_id: str = ""
        # Execution receipts: signed per-stage proof of completion.
        # Uses the module-level DEFAULT_SIGNER; override via configure_receipt_signer().
        self._receipt_signer: ReceiptSigner = _DEFAULT_RECEIPT_SIGNER
        self._topology_provider_factory = topology_provider_factory or _default_topology_provider_factory

    def set_kernel_rejection_semantics(self, stage: str, semantics: KernelRejectionSemantics) -> None:
        """Override rejection semantics for a stage name.

        Use stage='*' for global defaults.
        """
        self._side_effects.set_kernel_rejection_semantics(stage, semantics)

    def configure_resume(
        self,
        store_dir: Path,
        *,
        policy: ResumabilityPolicy | None = None,
    ) -> None:
        """Enable crash-safe turn resumption backed by *store_dir*.

        After this call, a resume point is written to disk after every
        successful pipeline stage.  If execute() is called with a trace_id
        that has a valid resume record, already-completed stages are skipped
        automatically (idempotent node guarantee).

        Parameters
        ----------
        store_dir:
            Directory where ``.resume.json`` files are written.  Created
            on first save if it does not yet exist.
        policy:
            Resumability settings.  Defaults to ``ResumabilityPolicy()``
            (enabled, 1-hour TTL, skip completed stages).
        """
        resolved_policy = policy or ResumabilityPolicy()
        self._recovery = ExecutionRecovery(
            resume_store=TurnResumeStore(Path(store_dir)),
            policy=resolved_policy,
        )

    def configure_capabilities(
        self,
        registry: "CapabilityRegistry",
        *,
        policy: Any,
        session_id: str = "",
    ) -> None:
        """Enable per-stage capability enforcement.

        After this call, every pipeline stage is checked against *registry*
        before execution.  The session's capability set is frozen at turn start
        and verified on resume to prevent privilege escalation.

        Parameters
        ----------
        registry:
            ``CapabilityRegistry`` mapping stage names to requirements.
        policy:
            ``SessionAuthorizationPolicy`` providing capability lookups.
        session_id:
            The session identifier whose capabilities are checked.  May be
            overridden per-turn via ``TurnContext.metadata["session_id"]``.
        """
        self._capability_registry = registry
        self._capability_policy = policy
        self._capability_session_id = str(session_id or "")

    def configure_receipt_signer(self, signer: "ReceiptSigner") -> None:
        """Override the HMAC signing key used for execution receipts.

        Use this when you need cross-process receipt verification (e.g.
        verifying receipts from a prior crashed process).  Provide the same
        ``ReceiptSigner`` instance (or one constructed with the same key) on
        all processes.
        """
        self._receipt_signer = signer

    def _rejection_semantics_for_stage(self, stage: str) -> KernelRejectionSemantics:
        return self._side_effects.rejection_semantics_for_stage(stage)

    @staticmethod
    def _classify_failure(error: Exception) -> TurnFailureSeverity:
        return ExecutionPolicyEngine.classify_failure(error)

    def _record_execution_trace(
        self,
        turn_context: TurnContext,
        *,
        event_type: str,
        stage: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Delegate to SideEffectAdapter.emit_execution_event."""
        self._side_effect_adapter.emit_execution_event(
            turn_context,
            event_type=event_type,
            stage=stage,
            detail=detail,
        )

    def _finalize_execution_trace_contract(self, turn_context: TurnContext) -> None:
        trace = list(turn_context.state.get("execution_trace") or [])
        canonical = {
            "trace_id": str(turn_context.trace_id or ""),
            "events": [
                {
                    "sequence": int(item.get("sequence", 0) or 0),
                    "event_type": str(item.get("event_type", "") or ""),
                    "stage": str(item.get("stage", "") or ""),
                    "phase": str(item.get("phase", "") or ""),
                    "detail": _json_safe(item.get("detail") or {}),
                }
                for item in trace
            ],
        }
        digest = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        contract = stamp_trace_contract_version({
            "version": "1.0",
            "event_count": len(trace),
            "trace_hash": digest,
        })
        turn_context.state["execution_trace_contract"] = contract
        turn_context.metadata["execution_trace_contract"] = dict(contract)

        expected = str(turn_context.metadata.get("expected_execution_trace_hash") or "").strip()
        if expected and expected != digest:
            raise RuntimeError(
                "Execution trace determinism mismatch: "
                f"expected={expected!r}, actual={digest!r}"
            )

    def _validate_persistence_service_contract(self, turn_context: TurnContext, service: Any) -> None:
        """Validate minimum persistence interface for graph execution semantics.

        Strict mode is opt-in via metadata['persistence_contract_strict'].
        """
        self._side_effects.validate_persistence_service_contract(turn_context, service)

    def _seal_execution_identity(self, turn_context: TurnContext) -> ExecutionIdentity:
        """Build, validate, persist, and seal the canonical execution identity.

        This is the HARD RUNTIME CONTRACT enforcement point.  It runs at every
        turn exit — success, failure, and short-circuit — so no execution can
        complete without producing a verifiable identity fingerprint.

        Raises ExecutionIdentityViolation if metadata['expected_execution_fingerprint']
        is set and does not match the computed fingerprint.
        """
        identity = ExecutionIdentity.from_turn_context(turn_context)

        # Hard contract: enforce expected fingerprint if the caller provided one.
        expected = str(turn_context.metadata.get("expected_execution_fingerprint") or "").strip()
        identity.raise_if_mismatch(expected)

        # Seal into the context so downstream observers can read it.
        identity_dict = identity.to_dict()
        turn_context.state["execution_identity"] = identity_dict
        turn_context.metadata["execution_identity"] = dict(identity_dict)

        # Emit as a durable persistence event via the side-effects orchestrator
        # so the replay validator can reconstruct the fingerprint from the event
        # stream alone.
        self._side_effects.emit_execution_identity(
            registry=self.registry,
            turn_context=turn_context,
            identity=identity,
        )

        return identity

    def _persist_kernel_rejection(self, turn_context: TurnContext, *, stage: str, reason: str) -> None:
        self._side_effects.emit_kernel_rejection(
            registry=self.registry,
            turn_context=turn_context,
            stage=stage,
            reason=reason,
            semantics=self._rejection_semantics_for_stage(stage),
        )

    def set_execution_kernel(self, kernel: ExecutionKernel) -> None:
        validate_execution_kernel_spec(kernel, raise_on_failure=True)
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
        self._side_effects.project_turn_health(
            turn_context,
            total_latency_ms=float(total_latency_ms or 0.0),
            failed=bool(failed),
            stage_duration_lookup=self._stage_duration_ms,
        )

    def _mark_structural_degradation(self, turn_context: TurnContext, reason: str) -> None:
        self._side_effects.mark_structural_degradation(turn_context, reason)

    @staticmethod
    def _mark_stage_enter(turn_context: TurnContext, stage_name: str) -> None:
        """Backward-compatible shim: delegates to StageEntryGate.enforce_stage_ordering.

        New code should use TurnGraph._stage_entry_gate.enforce_stage_ordering directly.
        This staticmethod is retained for test compatibility.
        """
        StageEntryGate().enforce_stage_ordering(turn_context, stage_name)

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
        self._side_effect_adapter.record_witness(
            component,
            turn_context,
            emitter=self._execution_witness_emitter,
        )

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
        node_names = [str(getattr(node, "name", type(node).__name__) or type(node).__name__) for node in nodes]
        self._record_execution_trace(
            turn_context,
            event_type="parallel_start",
            stage="parallel_group",
            detail={"nodes": list(node_names)},
        )
        forked_contexts = [self._fork_context(turn_context) for _ in nodes]
        results = await asyncio.gather(
            *(self._execute_node(node, forked_context) for node, forked_context in zip(nodes, forked_contexts)),
        )
        for result_context in results:
            turn_context.metadata.update(result_context.metadata)
            turn_context.state.update(result_context.state)
            turn_context.stage_traces.extend(result_context.stage_traces)
        self._record_execution_trace(
            turn_context,
            event_type="parallel_done",
            stage="parallel_group",
            detail={"nodes": list(node_names)},
        )
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
                self._record_execution_trace(
                    turn_context,
                    event_type="kernel_error",
                    stage=node_name,
                    detail={"error": str(step_result.error or "")},
                )
                raise RuntimeError(step_result.error)
            if step_result.status == "rejected":
                semantics = self._rejection_semantics_for_stage(node_name)
                rejection_reason = str(getattr(getattr(step_result, "policy", None), "reason", "") or "")
                turn_context.state.setdefault("kernel_rejection_contract", []).append(
                    {
                        "stage": node_name,
                        "reason": rejection_reason,
                        "retryable": bool(semantics.retryable),
                        "state_mutation_allowed": bool(semantics.state_mutation_allowed),
                        "invalidate_downstream": bool(semantics.invalidate_downstream),
                        "action": str(semantics.action),
                        "persistence_behavior": str(semantics.persistence_behavior),
                    }
                )
                self._record_execution_trace(
                    turn_context,
                    event_type="kernel_rejected",
                    stage=node_name,
                    detail={
                        "reason": rejection_reason,
                        "action": str(semantics.action),
                        "invalidate_downstream": bool(semantics.invalidate_downstream),
                        "retryable": bool(semantics.retryable),
                    },
                )
                if semantics.persistence_behavior == "persist_rejection_event":
                    self._persist_kernel_rejection(turn_context, stage=node_name, reason=rejection_reason)

                if semantics.action == "abort_turn" or semantics.invalidate_downstream:
                    raise RuntimeError(
                        "Kernel step rejected under abort semantics: "
                        f"stage={node_name!r}, reason={rejection_reason!r}"
                    )
                # Backward-compatible default: stage is skipped and pipeline continues.
                return turn_context
            self._record_execution_trace(
                turn_context,
                event_type="kernel_ok",
                stage=node_name,
                detail={},
            )
            return turn_context

        return await _call_node()

    @staticmethod
    def _is_commit_boundary_node(node: Any) -> bool:
        node_type = getattr(node, "node_type", None)
        if isinstance(node_type, NodeType):
            return node_type is NodeType.COMMIT
        return str(node_type or "").strip().lower() == NodeType.COMMIT.value

    def _emit_checkpoint(self, turn_context: TurnContext, *, stage: str, status: str, error: str | None = None) -> None:
        active_stage = str(turn_context.state.get("_active_graph_stage") or "")
        self._side_effects.emit_checkpoint(
            registry=self.registry,
            turn_context=turn_context,
            stage=stage,
            status=status,
            error=error,
            active_stage=active_stage,
            checkpoint_snapshot_fn=turn_context.checkpoint_snapshot,
        )

    @staticmethod
    def _phase_for_stage(stage: str, current: TurnPhase) -> TurnPhase:
        phase_name = StagePhaseMappingPolicy.phase_name_for_stage(stage)
        if not phase_name:
            return current
        try:
            return TurnPhase(phase_name)
        except ValueError:
            return current

    def _sync_stage_phase(self, turn_context: TurnContext, *, stage: str) -> None:
        target_phase = self._phase_for_stage(stage, turn_context.phase)
        transitions = turn_context.transition_phase(target_phase, reason=f"enter:{stage}")
        for transition in transitions:
            self._side_effects.emit_phase_transition(
                registry=self.registry,
                turn_context=turn_context,
                stage=stage,
                transition=dict(transition),
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

    # ------------------------------------------------------------------
    # Dispatcher authority: structural fidelity validator
    # ------------------------------------------------------------------

    def _check_structural_fidelity(self, turn_context: TurnContext) -> None:
        """Enforce post-execution structural invariants.

        Named authority holder for the fidelity check: the dispatcher calls this
        instead of inlining conditional logic.  Raises RuntimeError on violation.
        """
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
        # Strict mode: only enforced when the graph was built with an "inference" node
        # so that unit-test graphs with partial pipelines are not rejected.
        if "inference" in self._node_map and not bool(getattr(fidelity, "inference", False)):
            self._mark_structural_degradation(turn_context, "inference_stage_missing")
            raise RuntimeError("Structural turn invariant violated: InferenceNode missing")

    # ------------------------------------------------------------------
    # Dispatcher authority: per-stage execution worker
    # ------------------------------------------------------------------

    async def _run_stage(
        self,
        stage_name: str,
        node: Any,
        stage_context: TurnContext,
        *,
        telemetry: Any,
        executed_stage_names: list[str],
    ) -> TurnContext:
        """Execute a single pipeline stage.

        Authority contract
        ------------------
        - Reads own configuration via ``self.*`` attributes.
        - Delegates entry policy decisions to ``self._stage_entry_gate``.
        - Delegates all side-effect writes to ``self._side_effect_adapter``.
        - Receives mutable ``executed_stage_names`` list; appends stage name on
          completion so the outer dispatcher can read the completed set.
        - ``telemetry`` is passed explicitly; no implicit capture from execute().
        """
        started_at = time.perf_counter()
        error_msg: str | None = None

        # ── Stage-entry policy gate (delegated to StageEntryGate) ──────────
        # Idempotency guard: skip stages already completed in a prior run.
        if self._stage_entry_gate.check_recovery_skip(stage_name, stage_context, self._recovery):
            executed_stage_names.append(stage_name)
            return stage_context
        # Capability gate: skip if session lacks required capabilities.
        _sid = str(stage_context.metadata.get("session_id") or self._capability_session_id or "")
        if self._stage_entry_gate.check_capability_skip(
            stage_name, stage_context, self._capability_registry, self._capability_policy, _sid
        ):
            executed_stage_names.append(stage_name)
            return stage_context
        # Ordering + temporal invariants: raises on violation.
        self._stage_entry_gate.enforce_stage_ordering(stage_context, stage_name)

        self._record_execution_trace(
            stage_context,
            event_type="stage_enter",
            stage=stage_name,
            detail={"active_stage": stage_name},
        )
        self._sync_stage_phase(stage_context, stage=stage_name)
        self._emit_checkpoint(stage_context, stage=stage_name, status="before")
        if telemetry is not None:
            telemetry.trace("turn_graph.node_start", node=stage_name, trace_id=stage_context.trace_id)

        # Side-effect deduplication: persist in_flight marker + inject call_id
        # BEFORE the node runs so a crash between here and record_stage_completion
        # leaves a recoverable record.
        if self._recovery is not None:
            self._recovery.mark_stage_started(stage_name, stage_context)

        try:
            _guard = (
                MutationGuard(stage_context.mutation_queue)
                if not self._is_commit_boundary_node(node)
                else contextlib.nullcontext()
            )
            with _guard:
                stage_context = await self._execute_node(node, stage_context)
        except Exception as exc:
            error_msg = str(exc)
            self._record_execution_trace(
                stage_context,
                event_type="stage_error",
                stage=stage_name,
                detail={"error": error_msg},
            )
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
            self._record_execution_trace(
                stage_context,
                event_type="stage_done",
                stage=stage_name,
                detail={"duration_ms": duration_ms},
            )
            self._emit_checkpoint(stage_context, stage=stage_name, status="after")
            executed_stage_names.append(stage_name)

            # Persist a durable resume point after each successful stage.
            if self._recovery is not None:
                pipeline_names = list(stage_context.state.get("_pipeline_stage_names") or [])
                _next_for_resume = ""
                if pipeline_names and stage_name in pipeline_names:
                    _cidx = pipeline_names.index(stage_name)
                    if _cidx + 1 < len(pipeline_names):
                        _next_for_resume = pipeline_names[_cidx + 1]
                self._recovery.record_stage_completion(
                    stage_name,
                    _next_for_resume,
                    stage_context,
                    list(executed_stage_names),
                )

            # Execution receipt: signed proof of stage completion.
            self._side_effect_adapter.record_receipt(
                stage_context,
                stage_name,
                signer=self._receipt_signer,
                stage_call_id=str(stage_context.state.get("_stage_call_id") or ""),
                checkpoint_hash=str(stage_context.last_checkpoint_hash or ""),
            )

            # Emit edge checkpoint immediately after this stage completes.
            pipeline_names = list(stage_context.state.get("_pipeline_stage_names") or [])
            if pipeline_names and stage_name in pipeline_names:
                current_idx = pipeline_names.index(stage_name)
                if current_idx + 1 < len(pipeline_names):
                    next_stage_name = pipeline_names[current_idx + 1]
                    self._emit_checkpoint(
                        stage_context,
                        stage=f"{stage_name}\u2192{next_stage_name}",
                        status="edge",
                    )

        return stage_context

    async def execute(self, turn_context: TurnContext, *, audit_mode: bool = False) -> FinalizedTurnResult:
        if self._required_execution_token:
            active_token = ControlPlaneExecutionBoundary.current()
            if active_token != self._required_execution_token:
                raise RuntimeError(
                    "TurnGraph.execute boundary violation: graph execution must be routed through ExecutionControlPlane"
                )

        self._emit_execution_witness("graph.execute", turn_context)
        self._record_execution_trace(
            turn_context,
            event_type="turn_start",
            stage="graph",
            detail={"audit_mode": bool(audit_mode)},
        )
        turn_context.state["_execution_kernel"] = self._execution_kernel
        execute_started_at = time.perf_counter()

        # Initialise resume point.  Assigned inside the recovery block below;
        # declared here so the capability block can reference it unconditionally.
        _resume_pt: Any = None

        # Crash-safe recovery: restore completed stages from a durable resume record
        # so that already-executed stages are skipped on this invocation.
        if self._recovery is not None:
            _resume_pt = self._recovery.check_resume(turn_context.trace_id)
            if _resume_pt is not None:
                self._recovery.restore_executed_stages(_resume_pt, turn_context)

        # Capability security: freeze capability set at turn start (or verify it
        # on resume to block privilege escalation).
        if self._capability_registry is not None and self._capability_policy is not None:
            _sid = str(
                turn_context.metadata.get("session_id") or self._capability_session_id or ""
            )
            if _resume_pt is not None if self._recovery is not None else False:
                # Resumed turn: verify no escalation since turn started.
                try:
                    self._side_effect_adapter.verify_capability_freeze(
                        turn_context,
                        policy=self._capability_policy,
                        session_id=_sid,
                    )
                except CapabilityViolationError:
                    raise
            else:
                # Fresh turn: freeze the current capability set.
                self._side_effect_adapter.freeze_capabilities(
                    turn_context,
                    policy=self._capability_policy,
                    session_id=_sid,
                )

        telemetry = self.registry.get("telemetry") if self.registry is not None else None
        executed_stage_names: list[str] = []
        audit_enabled = bool(audit_mode or turn_context.metadata.get("audit_mode"))

        try:
            pipeline = self._pipeline_items()
            # Store all pipeline stage names in turn_context so _run_stage can find the next stage
            pipeline_stage_names = [name for name, _ in pipeline]
            turn_context.state["_pipeline_stage_names"] = pipeline_stage_names
            # ------------------------------------------------------------------
            # Layer 1 — LangGraph declarative dispatch
            # The execution loop is now owned by LangGraph's StateGraph.
            # Policy, mutation, and recovery logic stay in Layer 2 (_run_stage).
            # ------------------------------------------------------------------

            # Pre-execute kernel validation (previously inside ExecutionKernel.run)
            _preflight = self._execution_kernel.validate(
                stage="pre_execute",
                operation="execution_kernel.run",
                context=turn_context,
            )
            if not _preflight.ok:
                logger.warning("[KERNEL SHADOW VIOLATION] %s", _preflight.reason)

            async def _lg_node_executor(
                stage_name: str, node: Any, state: TurnPipelineState
            ) -> dict:
                """Bridge: per-stage kernel validation (Layer 1→2) + stage dispatch (Layer 2)."""
                _tc = state["context"]
                _stage_check = self._execution_kernel.validate(
                    stage=stage_name,
                    operation=f"kernel.phase:{stage_name}",
                    context=_tc,
                )
                if not _stage_check.ok:
                    logger.warning("[KERNEL SHADOW VIOLATION] %s", _stage_check.reason)
                _tc = await self._run_stage(
                    stage_name, node, _tc,
                    telemetry=telemetry,
                    executed_stage_names=executed_stage_names,
                )
                return {
                    "context": _tc,
                    "short_circuit": bool(getattr(_tc, "short_circuit", False)),
                    "abort": False,
                    "error": None,
                }

            _topology_provider = self._topology_provider_factory(pipeline, _lg_node_executor)
            _lg_pipeline = _topology_provider.build()
            _lg_state: TurnPipelineState = {
                "context": turn_context,
                "short_circuit": False,
                "abort": False,
                "error": None,
            }
            _final_state = await _lg_pipeline.ainvoke(_lg_state)
            turn_context = _final_state["context"]

            # Post-execute kernel validation (previously inside ExecutionKernel.run)
            _post = self._execution_kernel.validate(
                stage="post_execute",
                operation="execution_kernel.run.complete",
                context=turn_context,
            )
            if not _post.ok:
                logger.warning("[KERNEL SHADOW VIOLATION] %s", _post.reason)

        except Exception as exc:
            failure_class = self._classify_failure(exc)
            self._record_execution_trace(
                turn_context,
                event_type="turn_failed",
                stage="graph",
                detail={"severity": str(failure_class), "error": str(exc)},
            )
            self._side_effect_adapter.record_failure_taxonomy(
                turn_context,
                severity_str=str(failure_class),
                error_str=str(exc),
            )
            total_ms = round((time.perf_counter() - execute_started_at) * 1000, 3)
            self._emit_turn_health_state(turn_context, total_latency_ms=total_ms, failed=True)
            self._finalize_execution_trace_contract(turn_context)
            self._seal_execution_identity(turn_context)
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
            self._check_structural_fidelity(turn_context)
            self._record_execution_trace(
                turn_context,
                event_type="turn_short_circuit",
                stage="graph",
                detail={"latency_ms": total_ms},
            )
            self._finalize_execution_trace_contract(turn_context)
            self._seal_execution_identity(turn_context)
            if audit_enabled:
                self._emit_capability_audit_report(turn_context, stage_order=executed_stage_names, failed=False)
            if self._recovery is not None:
                self._recovery.clear(turn_context.trace_id)
            return turn_context.short_circuit_result

        result = turn_context.state.get("safe_result")
        total_ms = round((time.perf_counter() - execute_started_at) * 1000, 3)
        self._emit_turn_health_state(turn_context, total_latency_ms=total_ms, failed=False)
        self._check_structural_fidelity(turn_context)
        self._record_execution_trace(
            turn_context,
            event_type="turn_succeeded",
            stage="graph",
            detail={"latency_ms": total_ms, "stages": list(executed_stage_names)},
        )
        self._finalize_execution_trace_contract(turn_context)
        self._seal_execution_identity(turn_context)
        if audit_enabled:
            self._emit_capability_audit_report(turn_context, stage_order=executed_stage_names, failed=False)
        if self._recovery is not None:
            self._recovery.clear(turn_context.trace_id)
        if isinstance(result, tuple) and len(result) >= 2:
            return result
        return (str(result or ""), False)
