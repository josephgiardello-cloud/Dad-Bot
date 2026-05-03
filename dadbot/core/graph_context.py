"""graph_context — TurnFidelity and TurnContext turn-scoped state containers.

Extracted from graph.py to reduce TurnGraph god-class surface area.
All names are re-exported from dadbot.core.graph for backward compatibility.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from dadbot.contracts import AttachmentList, ChunkCallback, FinalizedTurnResult
from dadbot.core.determinism import DeterminismBoundary
from dadbot.core.determinism_seal import DEFAULT_SEAL as _DETERMINISM_SEAL
from dadbot.core.graph_mutation import MutationQueue
from dadbot.core.graph_temporal import (
    TurnPhase,
    TurnTemporalAxis,
    VirtualClock,
    _PHASE_ORDER,
)
from dadbot.core.graph_types import StageTrace, _json_safe
from dadbot.core.ux_projection import TurnHealthState


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


class TurnContextContractViolation(TypeError):
    """Raised when TurnContext is constructed with an invalid field type.

    This is the hard construction-time boundary contract: state, metadata, and
    determinism_manifest must always be dicts.  Any non-dict value passed at
    construction is a caller contract violation.
    """


@dataclass
class TurnContext:
    user_input: str
    attachments: AttachmentList | None = None
    chunk_callback: ChunkCallback | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    # Correlation ID propagated through all log/telemetry calls for this turn.
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    kernel_step_id: str = ""
    determinism_manifest: dict[str, Any] = field(default_factory=dict)
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
    determinism_boundary: DeterminismBoundary = field(
        default_factory=DeterminismBoundary,
    )
    # Canonical frozen temporal axis for the entire turn.
    temporal: TurnTemporalAxis = field(default_factory=TurnTemporalAxis.from_now)
    # User-facing health snapshot emitted once per turn.
    turn_health: TurnHealthState | None = None
    # Turn-scoped mutation queue: all persistent mutations queue here until SaveNode drains them.
    mutation_queue: MutationQueue = field(default_factory=MutationQueue)
    # Pipeline fidelity: which canonical stages ran this turn.
    fidelity: TurnFidelity = field(default_factory=TurnFidelity)
    # Optional deterministic virtual clock; TemporalNode uses it when set instead of wall time.
    virtual_clock: VirtualClock | None = field(default=None)
    # Hash-chain pointers for checkpoint integrity across load/save boundaries.
    last_checkpoint_hash: str = field(default="", init=False)
    prev_checkpoint_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        # Strict boundary contract: state/metadata/determinism_manifest must be dicts.
        # Enforce at construction so no TurnContext can ever carry an invalid shape.
        if not isinstance(self.state, dict):
            raise TurnContextContractViolation(
                f"TurnContext.state must be a dict, got {type(self.state).__name__!r}",
            )
        if not isinstance(self.metadata, dict):
            raise TurnContextContractViolation(
                f"TurnContext.metadata must be a dict, got {type(self.metadata).__name__!r}",
            )
        if not isinstance(self.determinism_manifest, dict):
            raise TurnContextContractViolation(
                f"TurnContext.determinism_manifest must be a dict, got {type(self.determinism_manifest).__name__!r}",
            )
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
                {"stage": t.stage, "duration_ms": t.duration_ms, "error": t.error} for t in self.stage_traces
            ],
            "turn_health_state": _json_safe(
                self.state.get("turn_health_state")
                or (self.turn_health.to_dict() if self.turn_health is not None else {}),
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
                {
                    "stage": trace.stage,
                    "duration_ms": trace.duration_ms,
                    "error": trace.error,
                }
                for trace in self.stage_traces
            ],
            "turn_health_state": _json_safe(
                self.state.get("turn_health_state")
                or (self.turn_health.to_dict() if self.turn_health is not None else {}),
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
            json.dumps(_chain_payload, sort_keys=True).encode("utf-8"),
        ).hexdigest()[:32]
        snapshot["checkpoint_hash"] = checkpoint_hash
        if advance_chain:
            # Advance the chain for the next durable checkpoint edge.
            self.prev_checkpoint_hash = chain_prev
            self.last_checkpoint_hash = checkpoint_hash
        return snapshot

    def transition_phase(
        self,
        target: TurnPhase,
        *,
        reason: str,
    ) -> list[dict[str, str]]:
        transitions: list[dict[str, str]] = []
        if self.phase == target:
            return transitions

        try:
            current_index = _PHASE_ORDER.index(self.phase)
            target_index = _PHASE_ORDER.index(target)
        except ValueError as exc:
            raise RuntimeError(
                f"Unknown phase transition request: {self.phase} -> {target}",
            ) from exc

        if target_index < current_index:
            raise RuntimeError(
                f"Non-deterministic phase regression: {self.phase} -> {target}",
            )

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
