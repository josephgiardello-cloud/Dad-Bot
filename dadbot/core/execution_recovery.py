"""Execution recovery orchestration for crash-safe turn resumption.

This module bridges TurnResumeStore and TurnGraph:
- Determines which stages to skip when resuming a crashed turn.
- Provides the node-level idempotency check that guards against re-execution.
- Exposes startup discovery of pending (incomplete) turns.

Architecture role
-----------------
Thin coordinator: no domain logic, no graph imports.  TurnGraph calls this
module; this module calls TurnResumeStore.  Direction of dependency:

    TurnGraph  →  ExecutionRecovery  →  TurnResumeStore

Usage inside TurnGraph.execute()
---------------------------------
1. On entry:
       recovery = ExecutionRecovery(resume_store, policy)
       resume = recovery.check_resume(turn_context.trace_id)
       if resume:
           recovery.restore_executed_stages(resume, turn_context)

2. Inside _execute_stage (BEFORE node execution — side-effect deduplication):
       recovery.mark_stage_started(stage_name, turn_context)
       # _stage_call_id now in state for tool-level deduplication

3. Inside _execute_stage (AFTER check for already-completed stages):
       if recovery.is_already_completed(stage_name, turn_context):
           return stage_context  # idempotent skip

4. After each successful stage:
       recovery.record_stage_completion(stage_name, next_stage, turn_context, pipeline_names)

5. On success (all stages done):
       recovery.clear(turn_context.trace_id)

Side-effect Duplication Contract
---------------------------------
``mark_stage_started`` writes an ``in_flight_stage`` marker to disk BEFORE any
external I/O.  It also injects ``_stage_call_id = sha256(turn_id + ':' + stage_name)``
into ``turn_context.state``.  Tool implementations that perform external side
effects (writes, HTTP calls, DB mutations) SHOULD check ``_stage_call_id`` against
their own idempotency store before acting.  The call_id is deterministic:
identical for every retry of the same stage in the same turn.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any

from dadbot.core.turn_resume_store import ResumePoint, TurnResumeStore

if TYPE_CHECKING:
    from dadbot.core.execution_policy import ResumabilityPolicy


logger = logging.getLogger(__name__)

_EXECUTED_STAGES_KEY = "_graph_executed_stages"
_STAGE_CALL_ID_KEY = "_stage_call_id"


def _compute_stage_call_id(turn_id: str, stage_name: str) -> str:
    """Return a deterministic idempotency key for (turn_id, stage_name).

    The key is a 16-hex-char prefix of SHA-256(turn_id + ':' + stage_name).
    It is injected into TurnContext.state before every node execution so that
    external tools can perform their own deduplication checks.
    """
    payload = f"{turn_id}:{stage_name}".encode()
    return hashlib.sha256(payload).hexdigest()[:32]


class ExecutionRecovery:
    """Orchestrates crash-safe turn resumption.

    Parameters
    ----------
    resume_store:
        Durable storage for per-turn resume records.
    policy:
        ResumabilityPolicy controlling max age and whether to skip stages.

    """

    def __init__(
        self,
        resume_store: TurnResumeStore,
        policy: ResumabilityPolicy,
    ) -> None:
        self._store = resume_store
        self._policy = policy

    # ------------------------------------------------------------------
    # Startup phase
    # ------------------------------------------------------------------

    def check_resume(self, turn_id: str) -> ResumePoint | None:
        """Return a valid resume point for *turn_id*, or None.

        Returns None if:
        - Recovery is disabled in policy.
        - No record exists.
        - Record is too old (exceeds policy.max_age_seconds).
        - Record has an empty completed_stages list (nothing to skip).
        """
        if not self._policy.enabled:
            return None
        point = self._store.load(turn_id)
        if point is None:
            return None
        if point.is_expired(max_age_seconds=self._policy.max_age_seconds):
            logger.warning(
                "Resume point for turn %r is expired (age exceeds %.0fs); discarding",
                turn_id,
                self._policy.max_age_seconds,
            )
            self._store.clear(turn_id)
            return None
        if not point.completed_stages:
            return None
        return point

    def restore_executed_stages(
        self,
        resume: ResumePoint,
        turn_context: Any,
    ) -> None:
        """Populate turn_context's executed-stages set from *resume*.

        After this call, is_already_completed() returns True for all stages
        in resume.completed_stages, and the idempotency guard in TurnGraph
        will skip them without raising.

        Also restores ``_graph_last_stage`` so the pipeline ordering gate in
        ``_mark_stage_enter`` does not reject the first resumed stage.
        """
        if not self._policy.skip_completed_stages:
            return
        state = getattr(turn_context, "state", None)
        if not isinstance(state, dict):
            return
        executed: set[str] = state.setdefault(_EXECUTED_STAGES_KEY, set())
        if not isinstance(executed, set):
            executed = set(executed) if isinstance(executed, (list, tuple)) else set()
        executed.update(resume.completed_stages)
        state[_EXECUTED_STAGES_KEY] = executed

        # Restore the last-stage pointer so _mark_stage_enter's ordering gate passes
        # for the first non-skipped stage.
        if resume.completed_stages:
            state["_graph_last_stage"] = resume.completed_stages[-1]

        logger.info(
            "Recovery: restored %d completed stages for turn %r (resuming from %r)",
            len(resume.completed_stages),
            resume.turn_id,
            resume.next_stage,
        )

    # ------------------------------------------------------------------
    # Per-stage phase (called from inside _execute_stage)
    # ------------------------------------------------------------------

    def mark_stage_started(self, stage_name: str, turn_context: Any) -> str:
        """Persist an in-flight marker BEFORE the node executes and inject call_id.

        This is the side-effect deduplication hook.  It does two things atomically
        from the caller's perspective:

        1. Writes ``in_flight_stage=stage_name`` to the durable resume record on
           disk so that a crash between here and ``record_stage_completion`` leaves
           a recoverable marker.

        2. Injects ``_stage_call_id`` (deterministic sha256 hash of turn_id+stage)
           into ``turn_context.state``.  External tools that perform writes or HTTP
           calls SHOULD check this key against their own idempotency store before
           acting.

        Returns the call_id string so callers can log it.
        Non-fatal: storage errors are swallowed (execution continues).
        """
        if not self._policy.enabled:
            return ""
        turn_id = str(getattr(turn_context, "trace_id", "") or "")
        call_id = _compute_stage_call_id(turn_id, stage_name)
        state = getattr(turn_context, "state", None)
        if isinstance(state, dict):
            state[_STAGE_CALL_ID_KEY] = call_id
        try:
            self._store.mark_started(turn_id, stage_name)
        except OSError as exc:
            logger.debug(
                "mark_stage_started: non-fatal storage error for turn %r stage %r: %s",
                turn_id,
                stage_name,
                exc,
            )
        return call_id

    def is_already_completed(self, stage_name: str, turn_context: Any) -> bool:
        """Return True if *stage_name* is already in the executed-stages set.

        This is the node-level idempotency check.  When True, TurnGraph must
        skip execution of this stage entirely and return the context unchanged.
        """
        state = getattr(turn_context, "state", None)
        if not isinstance(state, dict):
            return False
        executed = state.get(_EXECUTED_STAGES_KEY)
        if not isinstance(executed, (set, frozenset)):
            return False
        return stage_name in executed

    def record_stage_completion(
        self,
        stage_name: str,
        next_stage: str,
        turn_context: Any,
        completed_stages: list[str],
    ) -> None:
        """Persist a resume point after *stage_name* completes successfully.

        Parameters
        ----------
        stage_name:
            The stage that just finished.
        next_stage:
            The next pipeline stage, or '' if this was the last stage.
        turn_context:
            Current TurnContext (used for trace_id and checkpoint_hash).
        completed_stages:
            Ordered list of all stages completed so far in this turn.

        """
        if not self._policy.enabled:
            return
        turn_id = str(getattr(turn_context, "trace_id", "") or "")
        checkpoint_hash = str(getattr(turn_context, "last_checkpoint_hash", "") or "")
        try:
            self._store.save(
                turn_id=turn_id,
                last_completed_stage=stage_name,
                next_stage=next_stage,
                checkpoint_hash=checkpoint_hash,
                completed_stages=list(completed_stages),
            )
        except OSError as exc:
            # Non-fatal: warn but don't interrupt execution.
            logger.warning(
                "Failed to persist resume point for turn %r after stage %r: %s",
                turn_id,
                stage_name,
                exc,
            )

    # ------------------------------------------------------------------
    # Completion phase
    # ------------------------------------------------------------------

    def clear(self, turn_id: str) -> None:
        """Remove the resume record on successful turn completion.

        Non-fatal: silently ignores missing or unremovable records.
        """
        if not self._policy.enabled:
            return
        try:
            self._store.clear(turn_id)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Startup discovery
    # ------------------------------------------------------------------

    def list_pending_turns(self) -> list[ResumePoint]:
        """Return all stored resume points that have not yet been cleared.

        Used at startup to discover turns interrupted by a crash.  The caller
        is responsible for deciding whether to retry or discard each record.
        """
        points = self._store.list_pending()
        if not self._policy.enabled:
            return []
        valid = []
        for point in points:
            if not point.is_expired(max_age_seconds=self._policy.max_age_seconds):
                valid.append(point)
        return valid

    def purge_expired_records(self) -> int:
        """Remove expired resume records from the store.

        Returns number of records removed.
        """
        return self._store.purge_expired(max_age_seconds=self._policy.max_age_seconds)

    # ------------------------------------------------------------------
    # Failure-domain recovery semantics (Phase 3)
    # ------------------------------------------------------------------

    @staticmethod
    def repair_partial_trace_context(
        trace_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Repair partial/invalid trace payloads into a replay-safe envelope."""
        trace = dict(trace_context or {})
        raw_steps = list(trace.get("steps") or [])
        repaired_steps: list[dict[str, Any]] = []
        for seq, step in enumerate(raw_steps):
            if not isinstance(step, dict):
                continue
            repaired_steps.append(
                {
                    "seq": int(
                        step.get("seq") if isinstance(step.get("seq"), int) else seq,
                    ),
                    "operation": str(step.get("operation") or "unknown").strip().lower() or "unknown",
                    "payload": dict(step.get("payload") or {}),
                },
            )
        trace["steps"] = repaired_steps
        trace.setdefault("schema_version", "2.0")
        trace.setdefault("normalized_response", "")
        trace.setdefault("memory_retrieval_set", [])
        trace.setdefault("execution_dag", {})
        return trace

    @staticmethod
    def reconcile_checkpoint_with_trace(
        *,
        checkpoint: dict[str, Any] | None,
        trace_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Reconcile durable checkpoint metadata with repaired trace context."""
        cp = dict(checkpoint or {})
        trace = ExecutionRecovery.repair_partial_trace_context(trace_context)
        checkpoint_state = dict(cp.get("state") or {})
        checkpoint_meta = dict(cp.get("metadata") or {})
        reconciled = {
            "state": checkpoint_state,
            "metadata": checkpoint_meta,
            "trace_context": trace,
            "reconciliation": {
                "checkpoint_present": bool(cp),
                "trace_step_count": len(list(trace.get("steps") or [])),
                "trace_final_hash": str(trace.get("final_hash") or ""),
            },
        }
        return reconciled

    @staticmethod
    def safe_fallback_reconstruction(
        *,
        checkpoint: dict[str, Any] | None,
        trace_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Fallback reconstruction is intentionally disabled (replay-only recovery)."""
        raise RuntimeError(
            "Recovery fallback path disabled: replay-only recovery policy is enforced",
        )
