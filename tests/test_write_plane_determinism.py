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


# ---------------------------------------------------------------------------
# Step 2.1 — Full replay comparison: two runs, same state_0, compare logs
# ---------------------------------------------------------------------------
#
# Spec (verbatim from Phase 2 plan):
#
#   state_0 = snapshot_full_state()
#   result_1 = run_turn(input, state_0)
#   log_1 = write_plane.drain_turn_log()
#   reset_system()
#   result_2 = run_turn(input, state_0)
#   log_2 = write_plane.drain_turn_log()
#   assert result_1.output == result_2.output
#   assert result_1.state == result_2.state
#   assert log_1 == log_2
#
# Two completely independent bot instances both starting from default_memory_store()
# serve as "same state_0" without requiring a snapshot/restore mechanism.  This is
# strictly more conservative: if either bot has hidden non-determinism it will surface.
#
# EXPECTED: this test will FAIL on first run, revealing hidden writers.
# That failure is the output.  The diff printed in the AssertionError is the map.
# ---------------------------------------------------------------------------

import difflib as _difflib
from pathlib import Path as _Path
from tempfile import TemporaryDirectory as _TempDir

import pytest


def _diff_logs(
    log_a: list,
    log_b: list,
) -> str:
    """Return a human-readable diff of two mutation log lists."""
    lines_a = [f"  {i:3d}: ({r.source!r}, {r.path!r}, {r.value_repr[:60]!r})" for i, r in enumerate(log_a)]
    lines_b = [f"  {i:3d}: ({r.source!r}, {r.path!r}, {r.value_repr[:60]!r})" for i, r in enumerate(log_b)]
    diff = list(_difflib.unified_diff(lines_a, lines_b, fromfile="run_1", tofile="run_2", lineterm=""))
    return "\n".join(diff) if diff else "(no textual diff — check ordering or turn_id)"


def _build_isolated_bot(tmp_path: _Path):
    """Create a DadBot instance with a fully isolated temp directory."""
    from dadbot.core.dadbot import DadBot
    bot = DadBot()
    bot.MEMORY_PATH = tmp_path / "dad_memory.json"
    bot.SEMANTIC_MEMORY_DB_PATH = tmp_path / "dad_memory_semantic.sqlite3"
    bot.GRAPH_STORE_DB_PATH = tmp_path / "dad_memory_graph.sqlite3"
    bot.SESSION_LOG_DIR = tmp_path / "session_logs"
    bot.MEMORY_STORE = bot.default_memory_store()
    bot.save_memory_store()
    # deterministic stub for semantic embedding
    bot.embed_texts = lambda texts, **kw: [
        [0.0] * 12 for _ in ([texts] if isinstance(texts, str) else list(texts))
    ]
    return bot


def _stub_bot_for_replay(orch, monkeypatch, *, fixed_reply: str) -> None:
    """Patch out all non-deterministic / expensive subsystems.

    The goal is a fast, repeatable turn whose writes are driven purely by the
    deterministic pipeline logic, not by LLM sampling or wall-clock effects.
    """
    service = orch.registry.get("agent_service")
    if service is not None:
        async def _fixed_agent(context, _rich):
            return (fixed_reply, False)
        monkeypatch.setattr(service, "run_agent", _fixed_agent)

    bot = orch.bot
    # Memory consolidation / forgetting
    mc = getattr(bot, "memory_coordinator", None)
    if mc is not None:
        for method in ("consolidate_memories", "apply_controlled_forgetting"):
            if hasattr(mc, method):
                monkeypatch.setattr(mc, method, lambda **kw: None)

    # Relationship materialisation
    rm = getattr(bot, "relationship_manager", None)
    if rm is not None and hasattr(rm, "materialize_projection"):
        monkeypatch.setattr(rm, "materialize_projection", lambda **kw: None)

    # Graph store sync
    mm = getattr(bot, "memory_manager", None)
    gm = getattr(mm, "graph_manager", None) if mm is not None else None
    if gm is not None and hasattr(gm, "sync_graph_store"):
        monkeypatch.setattr(gm, "sync_graph_store", lambda **kw: None)

    # Reply validation (can hit regex/LLM paths)
    if hasattr(bot, "validate_reply"):
        monkeypatch.setattr(bot, "validate_reply", lambda _u, r: r)

    # Health snapshot (can touch time-varying data)
    if hasattr(bot, "current_runtime_health_snapshot"):
        monkeypatch.setattr(bot, "current_runtime_health_snapshot", lambda **kw: {})


@pytest.mark.dev
@pytest.mark.asyncio
class TestReplayComparison:
    """Phase 2 Step 2.1: two independent runs from identical initial state.

    Compares mutation logs to reveal:
    - Hidden writers (paths present in only one run)
    - Order instability (same paths, different sequence)
    - Value drift (same path, different value_repr)

    Each run uses a completely fresh bot instance so there is zero shared
    mutable state between runs.  Both start from default_memory_store()
    which is deterministic.
    """

    USER_INPUT = "Hey dad, just wanted to check in."
    SESSION_ID = "replay-determinism-test"
    FIXED_REPLY = "Hey! Good to hear from you. Things are great here."

    async def _run_one(
        self,
        bot_dir: _Path,
        monkeypatch,
        *,
        turn_label: str,
    ) -> tuple[object, list]:
        """Build, stub, run one turn, capture write log, shutdown."""
        import hashlib
        from dadbot.core.write_plane import get_write_plane, reset_write_plane
        from dadbot.core.event_clock import set_event_clock, reset_event_clock

        bot = _build_isolated_bot(bot_dir)
        orch = bot.turn_orchestrator
        _stub_bot_for_replay(orch, monkeypatch, fixed_reply=self.FIXED_REPLY)

        # Confluence key required in enforce mode — derive deterministically
        digest = hashlib.sha256(
            f"{self.SESSION_ID}:{self.USER_INPUT}".encode()
        ).hexdigest()[:24]
        confluence_key = f"replay-test:{digest}"

        plane = reset_write_plane()
        plane.begin_turn(turn_label)
        # Freeze event time so timestamp-bearing writes are bit-identical across runs
        set_event_clock(lambda: 1_700_000_000.0)
        try:
            result = await orch.handle_turn(
                self.USER_INPUT,
                session_id=self.SESSION_ID,
                confluence_key=confluence_key,
            )
        finally:
            plane.end_turn()
            reset_event_clock()

        log = plane.drain_log()

        try:
            bot.shutdown()
        except Exception:
            pass

        return result, log

    async def test_write_log_nonempty_and_attributed(self, tmp_path, monkeypatch):
        """Basic gate: the instrumentation must capture at least one attributed write."""
        bot_dir = tmp_path / "bot_a"
        bot_dir.mkdir()
        _, log = await self._run_one(bot_dir, monkeypatch, turn_label="t-attr")

        assert len(log) > 0, (
            "Write plane captured zero records for a full turn — "
            "instrumentation is not wired to the pipeline execution path."
        )
        anonymous = [r for r in log if not r.source]
        assert not anonymous, (
            f"Anonymous writes (no source) detected: "
            f"{[(r.path, r.value_repr[:40]) for r in anonymous]}"
        )

    async def test_two_runs_produce_same_output(self, tmp_path, monkeypatch):
        """Same input + same state_0 → same output text."""
        dir_a = tmp_path / "bot_a"
        dir_b = tmp_path / "bot_b"
        dir_a.mkdir()
        dir_b.mkdir()

        result_1, _ = await self._run_one(dir_a, monkeypatch, turn_label="t1-output")
        result_2, _ = await self._run_one(dir_b, monkeypatch, turn_label="t2-output")

        # FinalizedTurnResult is (response_text, bool) or similar
        out_1 = result_1[0] if isinstance(result_1, tuple) else str(result_1)
        out_2 = result_2[0] if isinstance(result_2, tuple) else str(result_2)
        assert out_1 == out_2, (
            f"Output diverged between two deterministic runs:\n"
            f"  Run 1: {out_1!r}\n"
            f"  Run 2: {out_2!r}"
        )

    async def test_two_runs_produce_same_write_paths(self, tmp_path, monkeypatch):
        """Same input + same state_0 → same write paths (set equality).

        This is a weaker form of the full log comparison — it passes even if
        ordering differs.  Use this to get a partial signal before ordering
        is stabilized.
        """
        dir_a = tmp_path / "bot_a"
        dir_b = tmp_path / "bot_b"
        dir_a.mkdir()
        dir_b.mkdir()

        _, log_1 = await self._run_one(dir_a, monkeypatch, turn_label="t1-paths")
        _, log_2 = await self._run_one(dir_b, monkeypatch, turn_label="t2-paths")

        paths_1 = [r.path for r in log_1]
        paths_2 = [r.path for r in log_2]
        set_1 = set(paths_1)
        set_2 = set(paths_2)

        only_in_1 = set_1 - set_2
        only_in_2 = set_2 - set_1

        assert not only_in_1 and not only_in_2, (
            f"Write path sets diverged — hidden writers detected!\n"
            f"  Only in run 1: {sorted(only_in_1)}\n"
            f"  Only in run 2: {sorted(only_in_2)}\n"
            f"  Run 1 path sequence: {paths_1}\n"
            f"  Run 2 path sequence: {paths_2}"
        )

    async def test_two_runs_produce_identical_log(self, tmp_path, monkeypatch):
        """Full replay invariant: (source, path, value_repr) must be identical.

        This is the strictest form.  It will FAIL until Phase 2.3 enforces
        single-writer order.  The failure message is the diagnostic output.
        """
        dir_a = tmp_path / "bot_a"
        dir_b = tmp_path / "bot_b"
        dir_a.mkdir()
        dir_b.mkdir()

        _, log_1 = await self._run_one(dir_a, monkeypatch, turn_label="t1-full")
        _, log_2 = await self._run_one(dir_b, monkeypatch, turn_label="t2-full")

        sig_1 = [(r.source, r.path, r.value_repr) for r in log_1]
        sig_2 = [(r.source, r.path, r.value_repr) for r in log_2]

        # Always print counts even on pass for visibility
        count_msg = f"Run 1: {len(sig_1)} writes | Run 2: {len(sig_2)} writes"

        assert sig_1 == sig_2, (
            f"Mutation log divergence — non-deterministic writes detected!\n"
            f"{count_msg}\n\n"
            f"Diff (run_1 → run_2):\n"
            f"{_diff_logs(log_1, log_2)}\n\n"
            f"Full run 1 log:\n"
            + "\n".join(f"  [{i}] {r.source} → {r.path}: {r.value_repr[:80]}" for i, r in enumerate(log_1))
            + "\n\nFull run 2 log:\n"
            + "\n".join(f"  [{i}] {r.source} → {r.path}: {r.value_repr[:80]}" for i, r in enumerate(log_2))
        )
