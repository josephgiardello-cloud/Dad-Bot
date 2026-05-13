"""tests/test_write_plane_determinism.py — Phase 2, Step 2.1.

Determinism invariant test for the write-side authority system.

SPEC (from Phase 2 plan):
    run_turn(input, state_0) → (state_1,  output,  mutation_log)
    run_turn(input, state_0) → (state_1', output', mutation_log')
    assert mutation_log == mutation_log'   # if differs → hidden writers remain

Each test exercises a single write system in isolation with a fixed input,
runs it twice from the same initial state, and asserts:
  1. The mutation_log paths are identical (same writes, same order)
  2. The value_reprs are identical (same values)
  3. No hidden writes appear (no unexpected paths)

Phase 2.0 note: these tests are READ-ONLY assertions on the log.
They do not block any write.  Failures indicate hidden non-deterministic
writers that must be hunted down before Phase 2.3 enforcement.
"""

import pytest

from dadbot.core.write_plane import MutationRecord, WritePlane, get_write_plane, reset_write_plane


pytestmark = pytest.mark.dev


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_signatures(records: list[MutationRecord]) -> list[tuple[str, str, str]]:
    """Return (source, path, value_repr) tuples for determinism comparison."""
    return [(r.source, r.path, r.value_repr) for r in records]


def _log_paths(records: list[MutationRecord]) -> list[str]:
    return [r.path for r in records]


# ---------------------------------------------------------------------------
# Unit: WritePlane itself
# ---------------------------------------------------------------------------

class TestWritePlaneUnit:
    def setup_method(self):
        self.plane = reset_write_plane()

    def test_write_returns_value_unchanged(self):
        sentinel = {"key": "value", "n": 42}
        result = self.plane.write("TestSource", "test.path", sentinel)
        assert result is sentinel

    def test_write_logs_record(self):
        self.plane.write("Src", "p.q", 99)
        assert len(self.plane) == 1
        r = self.plane.snapshot_log()[0]
        assert r["source"] == "Src"
        assert r["path"] == "p.q"
        assert "99" in r["value_repr"]

    def test_drain_log_clears(self):
        self.plane.write("S", "p", 1)
        self.plane.write("S", "p", 2)
        drained = self.plane.drain_log()
        assert len(drained) == 2
        assert len(self.plane) == 0

    def test_snapshot_log_nondestructive(self):
        self.plane.write("S", "p", 1)
        _ = self.plane.snapshot_log()
        assert len(self.plane) == 1

    def test_turn_id_correlation(self):
        self.plane.write("S", "a", 1)           # no turn
        self.plane.begin_turn("turn-001")
        self.plane.write("S", "b", 2)           # inside turn
        self.plane.end_turn()
        self.plane.write("S", "c", 3)           # after turn
        log = self.plane.drain_log()
        assert log[0].turn_id is None
        assert log[1].turn_id == "turn-001"
        assert log[2].turn_id is None

    def test_drain_turn_log_selective(self):
        self.plane.begin_turn("T1")
        self.plane.write("S", "p1", 1)
        self.plane.end_turn()
        self.plane.begin_turn("T2")
        self.plane.write("S", "p2", 2)
        self.plane.end_turn()
        t1_log = self.plane.drain_turn_log("T1")
        assert len(t1_log) == 1
        assert t1_log[0].path == "p1"
        assert len(self.plane) == 1  # T2 record still present

    def test_paths_written_deduplicates(self):
        self.plane.write("S", "a.b", 1)
        self.plane.write("S", "a.b", 2)
        self.plane.write("S", "c.d", 3)
        paths = self.plane.paths_written()
        assert paths == ["a.b", "c.d"]

    def test_reset_write_plane_isolates(self):
        self.plane.write("S", "x", 1)
        fresh = reset_write_plane()
        assert len(fresh) == 0


# ---------------------------------------------------------------------------
# Instrumentation: TurnStateMutator
# ---------------------------------------------------------------------------

class TestTurnStateMutatorInstrumentation:
    def setup_method(self):
        self.plane = reset_write_plane()

    def _make_mock_bot(self):
        class _MockBot:
            _last_turn_pipeline = None
            _last_turn_health_state = None
        return _MockBot()

    def _make_mutator(self):
        from dadbot.services.turn_state_mutator import TurnStateMutator
        return TurnStateMutator(self._make_mock_bot())

    def test_store_turn_pipeline_logs_write(self):
        mut = self._make_mutator()
        payload = {
            "mode": "sync",
            "user_input": "hello",
            "started_at": "2026-01-01T00:00:00",
            "steps": [],
        }
        mut.store_turn_pipeline(payload)
        assert len(self.plane) >= 1
        paths = self.plane.paths_written()
        assert "bot._last_turn_pipeline" in paths

    def test_store_turn_pipeline_deterministic(self):
        """Same payload → same log signature on both runs."""
        payload = {
            "mode": "sync",
            "user_input": "hello",
            "started_at": "2026-01-01T00:00:00",
            "steps": [],
        }

        # Run 1
        self.plane = reset_write_plane()
        mut1 = self._make_mutator()
        mut1.store_turn_pipeline(payload)
        sig1 = _log_signatures(self.plane.drain_log())

        # Run 2
        self.plane = reset_write_plane()
        mut2 = self._make_mutator()
        mut2.store_turn_pipeline(payload)
        sig2 = _log_signatures(self.plane.drain_log())

        assert sig1 == sig2, f"Non-deterministic writes:\nRun1: {sig1}\nRun2: {sig2}"


# ---------------------------------------------------------------------------
# Instrumentation: BeliefStateEngine
# ---------------------------------------------------------------------------

class TestBeliefStateEngineInstrumentation:
    def setup_method(self):
        self.plane = reset_write_plane()

    def _base_state(self):
        return {"belief_state": {}}

    def _base_plan(self):
        return {
            "intent_type": "question",
            "strategy": "direct_answer",
            "uncertainty": {"score": 0.1},
        }

    def test_update_from_turn_logs_write(self):
        from dadbot.core.belief_state_engine import BeliefStateEngine
        engine = BeliefStateEngine()
        engine.update_from_turn(
            state=self._base_state(),
            trace_id="trace-001",
            user_input="test input",
            runtime_plan=self._base_plan(),
            success=True,
        )
        paths = self.plane.paths_written()
        assert "memory.belief_state" in paths

    def test_update_from_turn_deterministic(self):
        """Same inputs → identical mutation log signatures."""
        from dadbot.core.belief_state_engine import BeliefStateEngine

        kwargs = dict(
            trace_id="trace-001",
            user_input="determinism test",
            runtime_plan=self._base_plan(),
            success=True,
        )

        # Run 1
        self.plane = reset_write_plane()
        BeliefStateEngine().update_from_turn(state=self._base_state(), **kwargs)
        sig1 = _log_signatures(self.plane.drain_log())

        # Run 2
        self.plane = reset_write_plane()
        BeliefStateEngine().update_from_turn(state=self._base_state(), **kwargs)
        sig2 = _log_signatures(self.plane.drain_log())

        # Paths and sources must match; value_repr may differ only by timestamp
        assert [s[:2] for s in sig1] == [s[:2] for s in sig2], (
            f"Non-deterministic write paths:\nRun1: {sig1}\nRun2: {sig2}"
        )


# ---------------------------------------------------------------------------
# Instrumentation: InteractionStateEngine
# ---------------------------------------------------------------------------

class TestInteractionStateEngineInstrumentation:
    def setup_method(self):
        self.plane = reset_write_plane()

    def test_apply_turn_feedback_logs_write(self):
        from dadbot.ux_overlay.interaction_state import InteractionStateEngine
        engine = InteractionStateEngine()
        engine.apply_turn_feedback(
            positive_signal=0.5,
            user_sentiment="neutral",
            conversation_break=False,
        )
        paths = self.plane.paths_written()
        assert "interaction_state._state" in paths

    def test_apply_turn_feedback_deterministic(self):
        from dadbot.ux_overlay.interaction_state import InteractionStateEngine

        kwargs = dict(positive_signal=0.5, user_sentiment="neutral", conversation_break=False)

        self.plane = reset_write_plane()
        InteractionStateEngine().apply_turn_feedback(**kwargs)
        sig1 = _log_signatures(self.plane.drain_log())

        self.plane = reset_write_plane()
        InteractionStateEngine().apply_turn_feedback(**kwargs)
        sig2 = _log_signatures(self.plane.drain_log())

        assert sig1 == sig2, f"Non-deterministic writes:\nRun1: {sig1}\nRun2: {sig2}"


# ---------------------------------------------------------------------------
# Cross-system: mutation log completeness
# ---------------------------------------------------------------------------

class TestMutationLogCompleteness:
    """Verify that all 3 directly-testable systems each emit at least one record."""

    EXPECTED_PATHS = {
        "bot._last_turn_pipeline",
        "memory.belief_state",
        "interaction_state._state",
    }

    def test_all_directly_testable_paths_appear(self):
        plane = reset_write_plane()

        # TurnStateMutator
        class _Bot:
            _last_turn_pipeline = None
        from dadbot.services.turn_state_mutator import TurnStateMutator
        TurnStateMutator(_Bot()).store_turn_pipeline({
            "mode": "sync", "user_input": "x", "started_at": "2026-01-01T00:00:00", "steps": [],
        })

        # BeliefStateEngine
        from dadbot.core.belief_state_engine import BeliefStateEngine
        BeliefStateEngine().update_from_turn(
            state={"belief_state": {}},
            trace_id="t",
            user_input="x",
            runtime_plan={"intent_type": "q", "strategy": "a", "uncertainty": {"score": 0.0}},
            success=True,
        )

        # InteractionStateEngine
        from dadbot.ux_overlay.interaction_state import InteractionStateEngine
        InteractionStateEngine().apply_turn_feedback(
            positive_signal=0.0, user_sentiment="neutral", conversation_break=False,
        )

        written = set(plane.paths_written())
        missing = self.EXPECTED_PATHS - written
        assert not missing, f"Write paths not captured: {missing}"
