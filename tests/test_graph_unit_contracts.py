"""Unit-contract tests for dadbot/core/graph.py.

Industry-standard coverage gaps filled:
  - VirtualClock value semantics and monotonicity
  - TurnTemporalAxis.from_lock_hash determinism + empty-string fallback
  - TurnFidelity.full_pipeline property and to_dict roundtrip
  - _json_safe edge cases (bytes, nested structures, non-serialisable values)
  - MutationIntent constructor validation (all rejection paths)
  - MutationIntent payload_hash stability
  - MutationQueue: unbound raises, locked raises, drain sort ordering,
    hard_fail=False accumulates without raising, snapshot ledger split
  - TurnContext.checkpoint_snapshot hash-chain linkage and advance_chain=False
  - TurnContext.transition_phase regression guard, same-phase noop,
    gap-spanning transitions
  - MutationGuard lock lifecycle
  - TurnGraph._mark_stage_enter: duplicate stage, out-of-order stage,
    mutation-capable stage without temporal, custom stage allowed
  - TurnGraph._phase_for_stage keyword mapping
  - TurnGraph.execute execution-token boundary violation
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.unit
from dadbot.core.graph import (
    MutationGuard,
    MutationIntent,
    MutationKind,
    MutationQueue,
    TurnContext,
    TurnFidelity,
    TurnGraph,
    TurnPhase,
    TurnTemporalAxis,
    VirtualClock,
    _json_safe,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEMPORAL = {"wall_time": "2026-01-01T00:00:00", "wall_date": "2026-01-01"}


def _memory_intent(**kw) -> MutationIntent:
    payload = {"op": "save_mood_state", "mood": "neutral", "temporal": _TEMPORAL, **kw}
    return MutationIntent(type="memory", payload=payload)


# ---------------------------------------------------------------------------
# VirtualClock
# ---------------------------------------------------------------------------


class TestVirtualClock:
    def test_now_returns_base_before_any_tick(self):
        vc = VirtualClock(base_epoch=1_000_000.0, step_size_seconds=10.0)
        assert vc.now() == pytest.approx(1_000_000.0)

    def test_tick_advances_by_step(self):
        vc = VirtualClock(base_epoch=1_000_000.0, step_size_seconds=30.0)
        first = vc.tick()
        assert first == pytest.approx(1_000_030.0)

    def test_tick_is_strictly_monotonic(self):
        vc = VirtualClock(base_epoch=0.0, step_size_seconds=1.0)
        values = [vc.tick() for _ in range(10)]
        assert values == sorted(values)
        assert len(set(values)) == 10

    def test_to_datetime_is_timezone_aware(self):
        import datetime as dt

        vc = VirtualClock(base_epoch=1_700_000_000.0, step_size_seconds=1.0)
        d = vc.to_datetime()
        assert isinstance(d, dt.datetime)
        assert d.tzinfo is not None


# ---------------------------------------------------------------------------
# TurnTemporalAxis
# ---------------------------------------------------------------------------


class TestTurnTemporalAxis:
    def test_from_lock_hash_is_deterministic(self):
        a = TurnTemporalAxis.from_lock_hash("abc123def456789a")
        b = TurnTemporalAxis.from_lock_hash("abc123def456789a")
        assert a == b

    def test_from_lock_hash_different_inputs_differ(self):
        a = TurnTemporalAxis.from_lock_hash("aaaaaaaaaaaaaaaa")
        b = TurnTemporalAxis.from_lock_hash("bbbbbbbbbbbbbbbb")
        assert a.epoch_seconds != b.epoch_seconds

    def test_from_lock_hash_empty_falls_back_to_real_time(self):
        import datetime as dt

        today = dt.date.today().isoformat()
        axis = TurnTemporalAxis.from_lock_hash("")
        # Empty hash falls back to from_now() — wall_date must be today
        assert axis.wall_date == today

    def test_to_dict_is_json_serialisable(self):
        import json

        axis = TurnTemporalAxis.from_lock_hash("deadbeefdeadbeef")
        payload = axis.to_dict()
        serialised = json.dumps(payload)
        assert axis.wall_time in serialised
        assert axis.wall_date in serialised


# ---------------------------------------------------------------------------
# TurnFidelity
# ---------------------------------------------------------------------------


class TestTurnFidelity:
    def test_full_pipeline_false_when_any_stage_missing(self):
        f = TurnFidelity(temporal=True, inference=True, reflection=True, save=False)
        assert f.full_pipeline is False

    def test_full_pipeline_true_when_all_stages_present(self):
        f = TurnFidelity(temporal=True, inference=True, reflection=True, save=True)
        assert f.full_pipeline is True

    def test_to_dict_contains_full_pipeline_key(self):
        f = TurnFidelity(temporal=True, inference=False, reflection=False, save=False)
        d = f.to_dict()
        assert "full_pipeline" in d
        assert d["full_pipeline"] is False
        assert d["temporal"] is True


# ---------------------------------------------------------------------------
# _json_safe
# ---------------------------------------------------------------------------


class TestJsonSafe:
    def test_bytes_converted_to_descriptor(self):
        result = _json_safe(b"hello world")
        assert result == {"type": "bytes", "size": 11}

    def test_nested_dict_recursed(self):
        result = _json_safe({"a": {"b": b"x"}})
        assert result == {"a": {"b": {"type": "bytes", "size": 1}}}

    def test_set_converted_to_list(self):
        result = _json_safe({1, 2, 3})
        assert isinstance(result, list)
        assert set(result) == {1, 2, 3}

    def test_non_serialisable_object_uses_repr(self):
        class _Obj:
            def __repr__(self):
                return "<Obj>"

        result = _json_safe(_Obj())
        assert result == "<Obj>"

    def test_primitives_pass_through(self):
        for v in (None, True, False, 0, 3.14, "hello"):
            assert _json_safe(v) is v or _json_safe(v) == v


# ---------------------------------------------------------------------------
# MutationIntent construction validation
# ---------------------------------------------------------------------------


class TestMutationIntentValidation:
    def test_invalid_type_string_raises(self):
        with pytest.raises(RuntimeError, match="must be one of"):
            MutationIntent(type="bogus_kind", payload={})

    def test_non_dict_payload_raises(self):
        with pytest.raises(RuntimeError, match="payload must be a dict"):
            MutationIntent(type="goal", payload=["not", "a", "dict"])  # type: ignore[arg-type]

    def test_missing_temporal_when_required_raises(self):
        with pytest.raises(RuntimeError, match="TemporalNode required"):
            MutationIntent(type="memory", payload={"op": "save_mood_state"})

    def test_empty_wall_time_raises(self):
        bad_temporal = {"wall_time": "", "wall_date": "2026-01-01"}
        with pytest.raises(RuntimeError, match="TemporalNode required"):
            MutationIntent(type="memory", payload={"temporal": bad_temporal})

    def test_requires_temporal_false_skips_temporal_check(self):
        intent = MutationIntent(
            type="memory",
            payload={"op": "save_mood_state"},
            requires_temporal=False,
        )
        assert intent.type is MutationKind.MEMORY

    def test_invalid_memory_op_raises(self):
        with pytest.raises(RuntimeError, match="Unsupported memory mutation op"):
            MutationIntent(
                type="memory",
                payload={"op": "not_a_real_op", "temporal": _TEMPORAL},
            )

    def test_invalid_ledger_op_raises(self):
        with pytest.raises(RuntimeError, match="Unsupported ledger mutation op"):
            MutationIntent(
                type="ledger",
                payload={"op": "write_arbitrary_data", "temporal": _TEMPORAL},
            )

    def test_payload_hash_is_stable_for_identical_payloads(self):
        payload = {"op": "save_mood_state", "mood": "happy", "temporal": _TEMPORAL}
        a = MutationIntent(type="memory", payload=dict(payload))
        b = MutationIntent(type="memory", payload=dict(payload))
        assert a.payload_hash == b.payload_hash
        assert len(a.payload_hash) == 24

    def test_payload_hash_differs_for_different_payloads(self):
        t = _TEMPORAL
        a = MutationIntent(type="memory", payload={"op": "save_mood_state", "mood": "happy", "temporal": t})
        b = MutationIntent(type="memory", payload={"op": "save_mood_state", "mood": "sad", "temporal": t})
        assert a.payload_hash != b.payload_hash

    def test_non_integer_ordering_field_raises(self):
        with pytest.raises(RuntimeError, match="ordering fields must be integers"):
            MutationIntent(
                type="memory",
                payload={"op": "save_mood_state", "temporal": _TEMPORAL},
                priority="not-an-int",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# MutationQueue
# ---------------------------------------------------------------------------


class TestMutationQueue:
    def test_operations_on_unbound_queue_raise(self):
        queue = MutationQueue()
        with pytest.raises(RuntimeError, match="not bound"):
            queue.pending()

    def test_queue_while_locked_raises(self):
        queue = MutationQueue()
        queue.bind_owner("trace-x")
        intent = _memory_intent()
        queue._mutations_locked = True
        with pytest.raises(RuntimeError, match="MutationGuard violation"):
            queue.queue(intent)

    def test_drain_soft_fail_accumulates_without_raising(self):
        queue = MutationQueue()
        queue.bind_owner("trace-y")
        queue.queue(_memory_intent())
        queue.queue(_memory_intent())

        failures = queue.drain(lambda _: (_ for _ in ()).throw(RuntimeError("nope")), hard_fail_on_error=False)

        assert len(failures) == 2
        # All failures captured; no FatalTurnError raised
        for intent, msg in failures:
            assert "nope" in msg

    def test_drain_sorts_by_priority_then_sequence(self):
        queue = MutationQueue()
        queue.bind_owner("trace-z")
        t = _TEMPORAL

        low = MutationIntent(
            type="memory",
            payload={"op": "save_mood_state", "temporal": t},
            priority=200,
            requires_temporal=True,
        )
        high = MutationIntent(
            type="memory",
            payload={"op": "save_mood_state", "temporal": t},
            priority=50,
            requires_temporal=True,
        )
        queue.queue(low)
        queue.queue(high)

        drained_order: list[MutationIntent] = []
        queue.drain(drained_order.append, hard_fail_on_error=False)

        assert drained_order[0].priority == 50
        assert drained_order[1].priority == 200

    def test_snapshot_separates_ledger_and_non_ledger_counts(self):
        queue = MutationQueue()
        queue.bind_owner("trace-snap")

        ledger_intent = MutationIntent(
            type="ledger",
            payload={"op": "append_history", "temporal": _TEMPORAL},
        )
        memory_intent = _memory_intent()
        queue.queue(ledger_intent)
        queue.queue(memory_intent)

        snap = queue.snapshot()
        # Non-ledger pending count excludes the ledger entry
        assert snap["pending"] == 1
        assert snap["ledger_pending"] == 1

    def test_sequence_counter_autoincrements(self):
        queue = MutationQueue()
        queue.bind_owner("trace-seq")
        queue.queue(_memory_intent())
        queue.queue(_memory_intent())
        sequences = [intent.sequence_id for intent in queue.pending()]
        assert sequences == [1, 2]


# ---------------------------------------------------------------------------
# MutationGuard
# ---------------------------------------------------------------------------


class TestMutationGuard:
    def test_enter_sets_locked_exit_restores(self):
        queue = MutationQueue()
        queue.bind_owner("trace-guard")
        assert queue._mutations_locked is False

        guard = MutationGuard(queue)
        guard.__enter__()
        assert queue._mutations_locked is True

        guard.__exit__(None, None, None)
        assert queue._mutations_locked is False

    def test_context_manager_blocks_queueing(self):
        queue = MutationQueue()
        queue.bind_owner("trace-guard2")

        with MutationGuard(queue):
            with pytest.raises(RuntimeError, match="MutationGuard violation"):
                queue.queue(_memory_intent())

    def test_context_manager_unlocks_on_exception(self):
        queue = MutationQueue()
        queue.bind_owner("trace-guard3")

        with pytest.raises(ValueError):
            with MutationGuard(queue):
                raise ValueError("intentional")

        assert queue._mutations_locked is False


# ---------------------------------------------------------------------------
# TurnContext — checkpoint hash chain
# ---------------------------------------------------------------------------


class TestTurnContextCheckpointChain:
    def test_successive_checkpoints_form_linked_chain(self):
        ctx = TurnContext(user_input="hello")
        snap1 = ctx.checkpoint_snapshot(stage="temporal", status="after", advance_chain=True)
        snap2 = ctx.checkpoint_snapshot(stage="inference", status="after", advance_chain=True)

        # Second checkpoint's prev_checkpoint_hash must equal first checkpoint's hash
        assert snap2["prev_checkpoint_hash"] == snap1["checkpoint_hash"]

    def test_advance_chain_false_does_not_mutate_pointers(self):
        ctx = TurnContext(user_input="hello")
        ctx.checkpoint_snapshot(stage="temporal", status="after", advance_chain=True)
        before_last = ctx.last_checkpoint_hash

        ctx.checkpoint_snapshot(stage="peek", status="diagnostic", advance_chain=False)

        assert ctx.last_checkpoint_hash == before_last

    def test_checkpoint_hash_differs_per_stage(self):
        ctx = TurnContext(user_input="hello")
        snap1 = ctx.checkpoint_snapshot(stage="temporal", status="before", advance_chain=False)
        snap2 = ctx.checkpoint_snapshot(stage="inference", status="before", advance_chain=False)
        assert snap1["checkpoint_hash"] != snap2["checkpoint_hash"]


# ---------------------------------------------------------------------------
# TurnContext — transition_phase
# ---------------------------------------------------------------------------


class TestTurnContextTransitionPhase:
    def test_same_phase_returns_empty_list(self):
        ctx = TurnContext(user_input="hello")
        result = ctx.transition_phase(TurnPhase.PLAN, reason="noop")
        assert result == []
        assert ctx.phase == TurnPhase.PLAN

    def test_regression_raises(self):
        ctx = TurnContext(user_input="hello")
        ctx.transition_phase(TurnPhase.ACT, reason="forward")
        with pytest.raises(RuntimeError, match="Non-deterministic phase regression"):
            ctx.transition_phase(TurnPhase.PLAN, reason="backward")

    def test_gap_spanning_fills_intermediate_transitions(self):
        ctx = TurnContext(user_input="hello")
        # Jump from PLAN directly to RESPOND, skipping ACT and OBSERVE
        transitions = ctx.transition_phase(TurnPhase.RESPOND, reason="skip")
        phases = [t["to"] for t in transitions]
        assert phases == ["ACT", "OBSERVE", "RESPOND"]

    def test_phase_history_appended(self):
        ctx = TurnContext(user_input="hello")
        ctx.transition_phase(TurnPhase.ACT, reason="step")
        assert len(ctx.phase_history) == 1
        assert ctx.phase_history[0]["from"] == "PLAN"
        assert ctx.phase_history[0]["to"] == "ACT"


# ---------------------------------------------------------------------------
# TurnGraph._mark_stage_enter
# ---------------------------------------------------------------------------


class TestMarkStageEnter:
    def test_duplicate_stage_raises(self):
        ctx = TurnContext(user_input="hello")
        ctx.state["temporal"] = {"wall_time": "t", "wall_date": "d"}
        # Simulate context_builder already ran so inference is the valid next stage.
        ctx.state["_graph_last_stage"] = "context_builder"
        ctx.state["_graph_executed_stages"] = {"temporal", "context_builder"}
        TurnGraph._mark_stage_enter(ctx, "inference")
        with pytest.raises(RuntimeError, match="executed more than once"):
            TurnGraph._mark_stage_enter(ctx, "inference")

    def test_out_of_order_canonical_stage_raises(self):
        ctx = TurnContext(user_input="hello")
        ctx.state["temporal"] = {"wall_time": "t", "wall_date": "d"}
        # Simulate inference already ran; next valid stage is safety, not reflection.
        ctx.state["_graph_last_stage"] = "inference"
        ctx.state["_graph_executed_stages"] = {"temporal", "context_builder", "inference"}
        with pytest.raises(RuntimeError, match="order violation"):
            TurnGraph._mark_stage_enter(ctx, "reflection")

    def test_mutation_capable_stage_without_temporal_raises(self):
        ctx = TurnContext(user_input="hello")
        # Set ordering preconditions so context_builder is the last stage,
        # but deliberately omit state["temporal"] to trigger the temporal guard.
        ctx.state["_graph_last_stage"] = "context_builder"
        ctx.state["_graph_executed_stages"] = {"temporal", "context_builder"}
        with pytest.raises(RuntimeError, match="TemporalNode not initialized"):
            TurnGraph._mark_stage_enter(ctx, "inference")

    def test_custom_stage_name_bypasses_ordering_gate(self):
        ctx = TurnContext(user_input="hello")
        # No temporal in state — but custom stage is not canonical, so no error
        TurnGraph._mark_stage_enter(ctx, "my_custom_stage")
        assert "my_custom_stage" in ctx.state["_graph_executed_stages"]


# ---------------------------------------------------------------------------
# TurnGraph._phase_for_stage keyword mapping
# ---------------------------------------------------------------------------


class TestPhaseForStage:
    @pytest.mark.parametrize(
        "stage,expected",
        [
            ("health", TurnPhase.PLAN),
            ("preflight", TurnPhase.PLAN),
            ("memory", TurnPhase.PLAN),
            ("context", TurnPhase.PLAN),
            ("inference", TurnPhase.ACT),
            ("agent", TurnPhase.ACT),
            ("tool", TurnPhase.ACT),
            ("safety", TurnPhase.OBSERVE),
            ("moderation", TurnPhase.OBSERVE),
            ("save", TurnPhase.RESPOND),
            ("finalize", TurnPhase.RESPOND),
            ("persist", TurnPhase.RESPOND),
        ],
    )
    def test_stage_maps_to_expected_phase(self, stage, expected):
        result = TurnGraph._phase_for_stage(stage, TurnPhase.PLAN)
        assert result == expected

    def test_unknown_stage_returns_current_phase(self):
        result = TurnGraph._phase_for_stage("zz_unknown", TurnPhase.ACT)
        assert result == TurnPhase.ACT


# ---------------------------------------------------------------------------
# TurnGraph.execute — execution token boundary
# ---------------------------------------------------------------------------


class TestExecutionTokenBoundary:
    def test_execute_raises_when_token_mismatch(self):
        from dadbot.core.nodes import TemporalNode

        class _SaveNode:
            async def run(self, ctx):
                ctx.state["safe_result"] = ("ok", False)
                return ctx

        graph = TurnGraph(registry=None)
        graph.add_node("temporal", TemporalNode())
        graph.add_node("save", _SaveNode())
        graph.set_edge("temporal", "save")
        graph.set_required_execution_token("expected-token-abc")

        ctx = TurnContext(user_input="hello")
        with pytest.raises(RuntimeError, match="boundary violation"):
            asyncio.run(graph.execute(ctx))
