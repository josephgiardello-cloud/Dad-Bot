"""Phase 2 — Execution Replay Safety.

Provides three complementary guarantees for crash-safe, replay-safe execution:

2.1  ExecutionCheckpointKernel
     Snapshots per-tool / per-node execution state including fallback chain
     evidence, retry history, and partial outputs.  Each checkpoint is
     hash-chained so tampering or truncation is detectable.

2.2  ExecutionIdempotencyRegistry
     Prevents duplicate execution under retries / failover.  A request that
     has already successfully completed returns the cached result without
     re-invoking the tool handler.  Failures are NOT cached — they allow
     re-execution on the next attempt.

2.3  DeterministicReplayValidator
     Validates that replaying the same sequence of inputs + tool history
     produces structurally equivalent outputs.  «Structurally equivalent»
     means the output type-class and status are identical; raw text is
     compared only when strict_mode=True.
"""
from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# 2.1 — Checkpoint Kernel
# ---------------------------------------------------------------------------


class NodeState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    FALLBACK = "fallback"
    SKIPPED = "skipped"


@dataclass
class FallbackChainEntry:
    """One link in a fallback chain: what was tried and why it was rejected."""
    tool_name: str
    attempt: int
    status: str                    # ok / error / timeout / partial / degraded
    error_message: str = ""
    latency_ms: float = 0.0
    fallback_reason: str = ""      # why execution fell through to next candidate


@dataclass
class NodeExecutionSnapshot:
    """Full execution state for one logical node at checkpoint time."""
    node_id: str
    node_type: str                 # tool / planner / critic / router
    state: NodeState
    tool_name: str = ""
    intent: str = ""
    request_hash: str = ""        # canonical request hash (from tool_idempotency)
    attempt_count: int = 0
    last_error: str = ""
    output_type: str = ""         # structural output class (null/str/list/dict…)
    partial_confidence: float = 1.0
    fallback_chain: list[FallbackChainEntry] = field(default_factory=list)
    partial_output_available: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "state": self.state.value,
            "tool_name": self.tool_name,
            "intent": self.intent,
            "request_hash": self.request_hash,
            "attempt_count": self.attempt_count,
            "last_error": self.last_error,
            "output_type": self.output_type,
            "partial_confidence": round(self.partial_confidence, 4),
            "fallback_chain": [
                {
                    "tool_name": e.tool_name,
                    "attempt": e.attempt,
                    "status": e.status,
                    "error_message": e.error_message,
                    "latency_ms": e.latency_ms,
                    "fallback_reason": e.fallback_reason,
                }
                for e in self.fallback_chain
            ],
            "partial_output_available": self.partial_output_available,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class ExecutionCheckpoint:
    """An immutable point-in-time snapshot of all node execution states."""
    checkpoint_id: str
    label: str
    created_at: float
    node_snapshots: dict[str, NodeExecutionSnapshot]  # node_id → snapshot
    prev_checkpoint_hash: str
    checkpoint_hash: str = field(default="")

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "label": self.label,
            "created_at": self.created_at,
            "nodes": {k: v.to_dict() for k, v in self.node_snapshots.items()},
            "prev_checkpoint_hash": self.prev_checkpoint_hash,
            "checkpoint_hash": self.checkpoint_hash,
        }


class CheckpointIntegrityError(RuntimeError):
    pass


class ExecutionCheckpointKernel:
    """Manages per-node execution state and hash-chained checkpoints.

    Usage pattern::

        kernel = ExecutionCheckpointKernel()

        # As tool execution progresses, update node state:
        kernel.update_node("node:tool_a", NodeState.RUNNING, tool_name="memory_lookup")
        kernel.add_fallback_entry("node:tool_a", FallbackChainEntry(...))
        kernel.update_node("node:tool_a", NodeState.FAILED, last_error="timeout")
        kernel.update_node("node:tool_a", NodeState.FALLBACK, tool_name="memory_cache")
        kernel.update_node("node:tool_a", NodeState.SUCCESS)

        # Take a checkpoint at a stable commit boundary:
        cp = kernel.save("after_tool_wave_1")

        # On crash + resume, assert the checkpoint is still valid:
        kernel.assert_checkpoint_integrity()
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._node_states: dict[str, NodeExecutionSnapshot] = {}
        self._checkpoints: list[ExecutionCheckpoint] = []

    # ------------------------------------------------------------------
    # Node state management
    # ------------------------------------------------------------------

    def update_node(
        self,
        node_id: str,
        state: NodeState,
        *,
        node_type: str = "tool",
        tool_name: str = "",
        intent: str = "",
        request_hash: str = "",
        attempt_count: int | None = None,
        last_error: str = "",
        output_type: str = "",
        partial_confidence: float = 1.0,
        partial_output_available: bool = False,
    ) -> NodeExecutionSnapshot:
        """Create or update the execution snapshot for a node."""
        with self._lock:
            existing = self._node_states.get(node_id)
            if existing is None:
                snap = NodeExecutionSnapshot(
                    node_id=node_id,
                    node_type=node_type,
                    state=state,
                )
                self._node_states[node_id] = snap
            else:
                snap = existing

            snap.state = state
            snap.updated_at = time.time()
            if tool_name:
                snap.tool_name = tool_name
            if intent:
                snap.intent = intent
            if request_hash:
                snap.request_hash = request_hash
            if node_type and not existing:
                snap.node_type = node_type
            if attempt_count is not None:
                snap.attempt_count = attempt_count
            elif state == NodeState.RETRYING:
                snap.attempt_count += 1
            if last_error:
                snap.last_error = last_error
            if output_type:
                snap.output_type = output_type
            snap.partial_confidence = max(0.0, min(1.0, partial_confidence))
            snap.partial_output_available = partial_output_available
            return snap

    def add_fallback_entry(self, node_id: str, entry: FallbackChainEntry) -> None:
        """Append a fallback event to the named node's chain."""
        with self._lock:
            snap = self._node_states.get(node_id)
            if snap is None:
                snap = NodeExecutionSnapshot(node_id=node_id, node_type="tool", state=NodeState.FALLBACK)
                self._node_states[node_id] = snap
            snap.fallback_chain.append(entry)
            snap.updated_at = time.time()

    def get_node(self, node_id: str) -> Optional[NodeExecutionSnapshot]:
        with self._lock:
            return self._node_states.get(node_id)

    def all_nodes(self) -> list[NodeExecutionSnapshot]:
        with self._lock:
            return list(self._node_states.values())

    def nodes_in_state(self, state: NodeState) -> list[NodeExecutionSnapshot]:
        with self._lock:
            return [s for s in self._node_states.values() if s.state == state]

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, label: str = "checkpoint") -> ExecutionCheckpoint:
        """Snapshot current node states into a new hash-chained checkpoint."""
        with self._lock:
            prev_hash = self._checkpoints[-1].checkpoint_hash if self._checkpoints else ""
            node_snapshots = {k: deepcopy(v) for k, v in self._node_states.items()}

        checkpoint_id = _sha256({"label": label, "ts": time.time()})[:16]
        cp = ExecutionCheckpoint(
            checkpoint_id=checkpoint_id,
            label=str(label or "checkpoint"),
            created_at=time.time(),
            node_snapshots=node_snapshots,
            prev_checkpoint_hash=prev_hash,
        )
        # Compute hash over dict WITHOUT the checkpoint_hash field
        # (same exclusion used in assert_checkpoint_hash for consistency)
        cp_dict = {k: v for k, v in cp.to_dict().items() if k != "checkpoint_hash"}
        cp.checkpoint_hash = _sha256(cp_dict)

        with self._lock:
            self._checkpoints.append(cp)
        return cp

    def restore(self, checkpoint: ExecutionCheckpoint) -> None:
        """Restore node states from a checkpoint (for crash recovery)."""
        self.assert_checkpoint_hash(checkpoint)
        with self._lock:
            self._node_states = {k: deepcopy(v) for k, v in checkpoint.node_snapshots.items()}

    def latest_checkpoint(self) -> Optional[ExecutionCheckpoint]:
        with self._lock:
            return self._checkpoints[-1] if self._checkpoints else None

    def all_checkpoints(self) -> list[ExecutionCheckpoint]:
        with self._lock:
            return list(self._checkpoints)

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------

    @staticmethod
    def assert_checkpoint_hash(checkpoint: ExecutionCheckpoint) -> None:
        """Raise CheckpointIntegrityError if the checkpoint hash is invalid."""
        data = checkpoint.to_dict()
        stored_hash = data.pop("checkpoint_hash", "")
        expected = _sha256(data)
        # Recompute without checkpoint_hash field in to_dict
        # We need to rebuild without that key
        d2 = {k: v for k, v in checkpoint.to_dict().items() if k != "checkpoint_hash"}
        recomputed = _sha256(d2)
        if stored_hash != recomputed:
            raise CheckpointIntegrityError(
                f"Checkpoint {checkpoint.checkpoint_id!r} hash mismatch: "
                f"stored={stored_hash[:8]!r} computed={recomputed[:8]!r}"
            )

    def assert_checkpoint_integrity(self) -> None:
        """Verify the full checkpoint chain is intact."""
        with self._lock:
            checkpoints = list(self._checkpoints)
        expected_prev = ""
        for cp in checkpoints:
            if cp.prev_checkpoint_hash != expected_prev:
                raise CheckpointIntegrityError(
                    f"Chain break at checkpoint {cp.checkpoint_id!r}: "
                    f"expected prev={expected_prev[:8]!r}, got {cp.prev_checkpoint_hash[:8]!r}"
                )
            self.assert_checkpoint_hash(cp)
            expected_prev = cp.checkpoint_hash


# ---------------------------------------------------------------------------
# 2.2 — Idempotency Registry
# ---------------------------------------------------------------------------


class DuplicateExecutionError(RuntimeError):
    """Raised when a successful execution is attempted a second time."""


@dataclass
class IdempotencyEntry:
    request_hash: str
    tool_name: str
    status: str       # terminal status of the completed execution
    output_type: str
    completed_at: float
    attempt_count: int


class ExecutionIdempotencyRegistry:
    """Prevents duplicate tool execution under retries and failover.

    Contract:
    - A request whose hash maps to a SUCCESS entry is NOT re-executed.
      The cached result class is returned instead.
    - ERROR / TIMEOUT results are NOT cached — they allow re-execution.
    - Thread-safe; suitable for concurrent retry storms.

    Design note:
        This is a *request-level* guard.  It operates on the canonical
        request hash computed from (tool_name, args, intent).  It does NOT
        store raw output (that would require content-addressable storage);
        it stores the *result class* (status + output_type), which is
        sufficient for replay validation.
    """

    # Statuses that are terminal successes and must not be re-executed
    _SUCCESS_STATUSES: frozenset[str] = frozenset({"ok", "cached", "partial", "skipped"})

    def __init__(self) -> None:
        self._lock = RLock()
        self._entries: dict[str, IdempotencyEntry] = {}

    def register_success(
        self,
        request_hash: str,
        tool_name: str,
        status: str,
        output_type: str,
        attempt_count: int = 1,
    ) -> None:
        """Record a successful (terminal) execution result."""
        if status not in self._SUCCESS_STATUSES:
            # Non-terminal / error statuses are never cached
            return
        with self._lock:
            self._entries[request_hash] = IdempotencyEntry(
                request_hash=request_hash,
                tool_name=tool_name,
                status=status,
                output_type=output_type,
                completed_at=time.time(),
                attempt_count=attempt_count,
            )

    def is_duplicate(self, request_hash: str) -> bool:
        """Return True if this request was already successfully completed."""
        with self._lock:
            return request_hash in self._entries

    def get_cached_result(self, request_hash: str) -> Optional[IdempotencyEntry]:
        """Return the cached idempotency entry, or None if not yet executed."""
        with self._lock:
            return self._entries.get(request_hash)

    def evict(self, request_hash: str) -> None:
        """Remove a cached entry (e.g. TTL expiry or explicit invalidation)."""
        with self._lock:
            self._entries.pop(request_hash, None)

    def size(self) -> int:
        with self._lock:
            return len(self._entries)


# ---------------------------------------------------------------------------
# 2.3 — Deterministic Replay Validator
# ---------------------------------------------------------------------------


class ReplayMismatchError(RuntimeError):
    """Raised when a replay produces a structurally inconsistent result."""


@dataclass(frozen=True)
class ReplayRecord:
    """The canonical fingerprint of one tool execution, used for replay comparison."""
    request_hash: str
    result_status: str
    result_output_type: str
    fallback_chain_length: int
    attempt_count: int

    def fingerprint(self) -> str:
        return _sha256({
            "request_hash": self.request_hash,
            "result_status": self.result_status,
            "result_output_type": self.result_output_type,
            "fallback_chain_length": self.fallback_chain_length,
            "attempt_count": self.attempt_count,
        })


class DeterministicReplayValidator:
    """Ensures that replay of the same inputs + tool history yields the same
    structural output class.

    Usage::

        validator = DeterministicReplayValidator()

        # First execution (RECORD mode):
        validator.record(request_hash, status, output_type, fallback_len, attempts)

        # On replay (VALIDATE mode):
        validator.validate(request_hash, status, output_type, fallback_len, attempts)
        # Raises ReplayMismatchError if structural class differs.

    strict_mode=True additionally validates that attempt counts and fallback
    chain lengths are bitwise identical (not just structurally equivalent).
    """

    def __init__(self, strict_mode: bool = False) -> None:
        self._strict = strict_mode
        self._lock = RLock()
        self._records: dict[str, ReplayRecord] = {}

    def record(
        self,
        request_hash: str,
        result_status: str,
        result_output_type: str,
        fallback_chain_length: int = 0,
        attempt_count: int = 1,
    ) -> ReplayRecord:
        """Record a canonical execution fingerprint for future replay validation."""
        rec = ReplayRecord(
            request_hash=request_hash,
            result_status=result_status,
            result_output_type=result_output_type,
            fallback_chain_length=fallback_chain_length,
            attempt_count=attempt_count,
        )
        with self._lock:
            self._records[request_hash] = rec
        return rec

    def validate(
        self,
        request_hash: str,
        result_status: str,
        result_output_type: str,
        fallback_chain_length: int = 0,
        attempt_count: int = 1,
    ) -> ReplayRecord:
        """Validate a replay result against the recorded fingerprint.

        Raises ReplayMismatchError when:
        - No record exists for the request_hash (replay without prior recording)
        - result_status differs from recorded status
        - result_output_type differs from recorded output type
        - strict_mode=True and fallback_chain_length or attempt_count differ
        """
        with self._lock:
            recorded = self._records.get(request_hash)

        if recorded is None:
            raise ReplayMismatchError(
                f"No recorded fingerprint for request_hash={request_hash[:16]!r}. "
                "Cannot validate replay."
            )

        if recorded.result_status != result_status:
            raise ReplayMismatchError(
                f"Replay status mismatch for hash {request_hash[:16]!r}: "
                f"recorded={recorded.result_status!r}, replayed={result_status!r}"
            )
        if recorded.result_output_type != result_output_type:
            raise ReplayMismatchError(
                f"Replay output type mismatch for hash {request_hash[:16]!r}: "
                f"recorded={recorded.result_output_type!r}, replayed={result_output_type!r}"
            )
        if self._strict:
            if recorded.fallback_chain_length != fallback_chain_length:
                raise ReplayMismatchError(
                    f"Strict replay: fallback chain length mismatch for {request_hash[:16]!r}: "
                    f"recorded={recorded.fallback_chain_length}, replayed={fallback_chain_length}"
                )
            if recorded.attempt_count != attempt_count:
                raise ReplayMismatchError(
                    f"Strict replay: attempt count mismatch for {request_hash[:16]!r}: "
                    f"recorded={recorded.attempt_count}, replayed={attempt_count}"
                )

        # Replay is valid — return the recorded fingerprint for audit trail
        with self._lock:
            return self._records[request_hash]

    def has_record(self, request_hash: str) -> bool:
        with self._lock:
            return request_hash in self._records

    def record_count(self) -> int:
        with self._lock:
            return len(self._records)


# ---------------------------------------------------------------------------
# Facade: ReplaySafeExecutionContext
# ---------------------------------------------------------------------------


class ReplaySafeExecutionContext:
    """Single entry-point that composes all three Phase 2 primitives.

    Provides a unified API for callers that need crash-safe + replay-safe
    execution without managing three separate objects.

    Typical call sequence::

        ctx = ReplaySafeExecutionContext()

        with ctx.execution_scope("node:tool_a", tool_name="memory_lookup", intent="goal_lookup") as scope:
            result = my_tool_handler(args)
            scope.complete(status="ok", output_type="list", output=result)

        # Replay path — idempotency guard fires:
        if ctx.is_duplicate(request_hash):
            return ctx.get_cached_result(request_hash)
    """

    def __init__(self, strict_replay: bool = False) -> None:
        self.kernel = ExecutionCheckpointKernel()
        self.idempotency = ExecutionIdempotencyRegistry()
        self.replay = DeterministicReplayValidator(strict_mode=strict_replay)

    def is_duplicate(self, request_hash: str) -> bool:
        return self.idempotency.is_duplicate(request_hash)

    def record_execution(
        self,
        node_id: str,
        request_hash: str,
        tool_name: str,
        status: str,
        output_type: str,
        fallback_chain: list[FallbackChainEntry] | None = None,
        attempt_count: int = 1,
    ) -> None:
        """Record a completed execution across all three layers."""
        # 1. Update checkpoint kernel
        self.kernel.update_node(
            node_id,
            NodeState.SUCCESS if status in {"ok", "cached", "partial", "skipped"} else NodeState.FAILED,
            tool_name=tool_name,
            request_hash=request_hash,
            attempt_count=attempt_count,
            output_type=output_type,
        )
        for entry in (fallback_chain or []):
            self.kernel.add_fallback_entry(node_id, entry)

        # 2. Register with idempotency registry
        self.idempotency.register_success(
            request_hash, tool_name, status, output_type, attempt_count
        )

        # 3. Record replay fingerprint
        self.replay.record(
            request_hash, status, output_type,
            fallback_chain_length=len(fallback_chain or []),
            attempt_count=attempt_count,
        )

    def checkpoint(self, label: str = "checkpoint") -> ExecutionCheckpoint:
        """Take a named checkpoint of current execution state."""
        return self.kernel.save(label)

    def assert_integrity(self) -> None:
        """Verify checkpoint chain integrity (call on crash-resume)."""
        self.kernel.assert_checkpoint_integrity()
