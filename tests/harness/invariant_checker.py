"""InvariantChecker — post-execution judge system for TurnContext state.

Call validate() after every graph.execute() call to assert all
system-level invariants hold. Each check is independent so failures
identify exactly which invariant broke.
"""

from __future__ import annotations

from typing import Any

from dadbot.core.graph import TurnContext, TurnPhase

_PHASE_ORDER = [TurnPhase.PLAN, TurnPhase.ACT, TurnPhase.OBSERVE, TurnPhase.RESPOND]
_PHASE_INDEX = {p: i for i, p in enumerate(_PHASE_ORDER)}


class InvariantViolation(AssertionError):
    """Raised when a post-run graph invariant is violated."""


class InvariantChecker:
    """Post-execution validator for TurnContext state.

    Checks (in order):
        1. Temporal — state["temporal"] populated with valid wall_time/wall_date
        2. Phase    — phase_history contains only monotonic (non-regressing) transitions
        3. Fidelity — save ran exactly once; temporal ran if requested
        4. Mutations — no failed drains in snapshot
        5. Checkpoints — hash chain non-empty if checkpoints were recorded

    Usage::

        checker = InvariantChecker()
        checker.validate(ctx, result)  # raises InvariantViolation on breach
    """

    def validate(
        self,
        ctx: TurnContext,
        result: Any = None,
        *,
        expect_save: bool = True,
        expect_temporal: bool = True,
    ) -> None:
        self._check_temporal(ctx, expect_temporal)
        self._check_phase_monotonic(ctx)
        self._check_fidelity(ctx, expect_save=expect_save)
        self._check_mutation_queue(ctx)
        self._check_checkpoint_integrity(ctx)

    # ------------------------------------------------------------------
    # Temporal invariant
    # ------------------------------------------------------------------

    def _check_temporal(self, ctx: TurnContext, required: bool) -> None:
        if not required:
            return
        temporal = ctx.state.get("temporal")
        if not temporal:
            raise InvariantViolation(f"TEMPORAL [trace={ctx.trace_id[:8]}]: state['temporal'] absent after execution")
        if not str(temporal.get("wall_time") or "").strip():
            raise InvariantViolation(f"TEMPORAL [trace={ctx.trace_id[:8]}]: wall_time is empty or missing")
        if not str(temporal.get("wall_date") or "").strip():
            raise InvariantViolation(f"TEMPORAL [trace={ctx.trace_id[:8]}]: wall_date is empty or missing")

    # ------------------------------------------------------------------
    # Phase monotonicity invariant
    # ------------------------------------------------------------------

    def _check_phase_monotonic(self, ctx: TurnContext) -> None:
        prev_idx = -1
        for transition in ctx.phase_history:
            to_val = transition.get("to", "")
            try:
                to_phase = TurnPhase(to_val)
            except ValueError:
                raise InvariantViolation(f"PHASE [trace={ctx.trace_id[:8]}]: unknown phase in history: {to_val!r}")
            idx = _PHASE_INDEX.get(to_phase, -1)
            if idx < prev_idx:
                raise InvariantViolation(
                    f"PHASE [trace={ctx.trace_id[:8]}]: regression — {transition.get('from')!r} → {to_val!r}"
                )
            prev_idx = idx

    # ------------------------------------------------------------------
    # Fidelity invariant
    # ------------------------------------------------------------------

    def _check_fidelity(self, ctx: TurnContext, *, expect_save: bool) -> None:
        fidelity = ctx.fidelity
        if expect_save and not fidelity.save:
            raise InvariantViolation(f"FIDELITY [trace={ctx.trace_id[:8]}]: save stage did not execute")
        if fidelity.save:
            save_traces = [t for t in ctx.stage_traces if t.stage == "save"]
            if len(save_traces) != 1:
                raise InvariantViolation(
                    f"FIDELITY [trace={ctx.trace_id[:8]}]: SaveNode must run exactly once, "
                    f"found {len(save_traces)} trace(s)"
                )

    # ------------------------------------------------------------------
    # Mutation queue invariant
    # ------------------------------------------------------------------

    def _check_mutation_queue(self, ctx: TurnContext) -> None:
        snap = ctx.mutation_queue.snapshot()
        failed = snap.get("failed", 0)
        if failed > 0:
            raise InvariantViolation(f"MUTATION_QUEUE [trace={ctx.trace_id[:8]}]: {failed} mutation(s) failed drain")

    # ------------------------------------------------------------------
    # Checkpoint hash-chain integrity
    # ------------------------------------------------------------------

    def _check_checkpoint_integrity(self, ctx: TurnContext) -> None:
        if ctx.last_checkpoint_hash and len(ctx.last_checkpoint_hash) < 8:
            raise InvariantViolation(
                f"CHECKPOINT [trace={ctx.trace_id[:8]}]: checkpoint_hash suspiciously short: "
                f"{ctx.last_checkpoint_hash!r}"
            )
