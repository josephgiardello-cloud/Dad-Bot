"""Wave 3 production hardening tests — Steps 1-2 and 4-8."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from dadbot.core.durable_checkpoint import DurableCheckpoint
from dadbot.core.event_reducer import CanonicalEventReducer
from dadbot.core.execution_ledger import (
    ExecutionLedger,
    WriteBoundaryGuard,
    WriteBoundaryViolationError,
)
from dadbot.core.execution_lease import ExecutionLease
from dadbot.core.invariant_gate import InvariantGate, InvariantViolationError
from dadbot.core.ledger_backend import (
    FileWALLedgerBackend,
    InMemoryLedgerBackend,
)
from dadbot.core.ledger_writer import LedgerWriter
from dadbot.core.observability import (
    MetricsSink,
    TracingContext,
    EventStreamExporter,
    get_metrics,
)
from dadbot.core.recovery_manager import RecoveryManager
from dadbot.core.session_store import SessionStore
from dadbot.core.snapshot_engine import SnapshotEngine
from dadbot.core.system_health_checker import SystemHealthChecker


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_ledger_with_jobs(
    session_id: str = "s1",
    n: int = 1,
    *,
    backend=None,
) -> ExecutionLedger:
    from dadbot.core.control_plane import ExecutionJob

    kwargs = {"backend": backend} if backend is not None else {}
    ledger = ExecutionLedger(**kwargs)
    writer = LedgerWriter(ledger)
    for _ in range(n):
        job = ExecutionJob(session_id=session_id, user_input="hello")
        writer.append_job_submitted(job)
        writer.append_session_bound(session_id, job.job_id)
        writer.append_job_queued(job)
        writer.append_job_started(job)
        writer.append_job_completed(job, {"reply": "ok"})
    return ledger


# ===========================================================================
# Step 1 — LedgerBackend + FileWALLedgerBackend
# ===========================================================================

class TestInMemoryLedgerBackend:
    def test_append_and_load_roundtrip(self):
        backend = InMemoryLedgerBackend()
        backend.append({"type": "TEST", "seq": 1})
        backend.append({"type": "TEST", "seq": 2})
        loaded = backend.load()
        assert len(loaded) == 2
        assert loaded[0]["seq"] == 1
        assert loaded[1]["seq"] == 2

    def test_load_returns_deep_copy(self):
        backend = InMemoryLedgerBackend()
        event = {"type": "TEST", "mutable": []}
        backend.append(event)
        loaded = backend.load()
        loaded[0]["mutable"].append("x")
        # Underlying stored copy should be unaffected.
        assert backend.load()[0]["mutable"] == []


class TestFileWALLedgerBackend:
    def test_persist_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wal_path = Path(tmpdir) / "ledger.wal"
            backend = FileWALLedgerBackend(wal_path, fsync=False)
            backend.append({"type": "JOB_QUEUED", "seq": 1})
            backend.append({"type": "JOB_COMPLETED", "seq": 2})

            # Reload from disk — simulates restart.
            backend2 = FileWALLedgerBackend(wal_path, fsync=False)
            events = backend2.load()
            assert len(events) == 2
            assert events[0]["type"] == "JOB_QUEUED"
            assert events[1]["type"] == "JOB_COMPLETED"

    def test_fsync_committed_types_only(self):
        """FileWAL appends without error for both committed and non-committed events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal_path = Path(tmpdir) / "ledger.wal"
            backend = FileWALLedgerBackend(wal_path, fsync=True)
            # JOB_QUEUED is in COMMITTED_TYPES — will fsync.
            backend.append({"type": "JOB_QUEUED", "seq": 1}, committed=False)
            # Non-critical type with committed=False — no fsync.
            backend.append({"type": "SESSION_BOUND", "seq": 2}, committed=False)
            events = backend.load()
            assert len(events) == 2

    def test_corrupt_line_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wal_path = Path(tmpdir) / "ledger.wal"
            # Write a valid line then a corrupt one.
            wal_path.write_text('{"type": "JOB_QUEUED"}\nNOT_JSON\n{"type": "JOB_COMPLETED"}\n')
            import warnings
            with warnings.catch_warnings(record=True):
                backend = FileWALLedgerBackend(wal_path, fsync=False)
                events = backend.load()
            assert len(events) == 2

    def test_ledger_load_from_backend(self):
        """ExecutionLedger.load_from_backend() repopulates in-memory state from WAL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal_path = Path(tmpdir) / "ledger.wal"
            backend1 = FileWALLedgerBackend(wal_path, fsync=False)
            ledger1 = _make_ledger_with_jobs(backend=backend1)
            event_count = len(ledger1.read())

            # Simulate restart.
            backend2 = FileWALLedgerBackend(wal_path, fsync=False)
            ledger2 = ExecutionLedger(backend=backend2)
            loaded = ledger2.load_from_backend()
            assert loaded == event_count


# ===========================================================================
# Step 2 — Write confirmation / committed semantics
# ===========================================================================

class TestCommittedWrites:
    def test_job_queued_is_written_with_committed_true(self):
        """LedgerWriter automatically sets committed=True for critical event types."""
        committed_calls: list[bool] = []

        class TrackingBackend(InMemoryLedgerBackend):
            def append(self, event, *, committed=False):
                if str(event.get("type") or "") in {"JOB_QUEUED", "JOB_COMPLETED"}:
                    committed_calls.append(committed)
                super().append(event, committed=committed)

        ledger = ExecutionLedger(backend=TrackingBackend())
        writer = LedgerWriter(ledger)
        from dadbot.core.control_plane import ExecutionJob
        job = ExecutionJob(session_id="s1", user_input="hi")
        writer.append_job_submitted(job)
        writer.append_session_bound("s1", job.job_id)
        writer.append_job_queued(job)
        writer.append_job_started(job)
        writer.append_job_completed(job, {"reply": "ok"})

        # JOB_QUEUED and JOB_COMPLETED must have been committed=True.
        assert all(committed_calls), f"Some critical events were not committed: {committed_calls}"

    def test_non_critical_events_not_committed_by_default(self):
        committed_calls: list[bool] = []

        class TrackingBackend(InMemoryLedgerBackend):
            def append(self, event, *, committed=False):
                if str(event.get("type") or "") == "SESSION_BOUND":
                    committed_calls.append(committed)
                super().append(event, committed=committed)

        ledger = ExecutionLedger(backend=TrackingBackend())
        writer = LedgerWriter(ledger)
        from dadbot.core.control_plane import ExecutionJob
        job = ExecutionJob(session_id="s1", user_input="hi")
        writer.append_job_submitted(job)
        writer.append_session_bound("s1", job.job_id)

        assert committed_calls and not all(committed_calls)


# ===========================================================================
# Step 4 — InvariantGate runtime enforcement
# ===========================================================================

class TestInvariantGate:
    def test_valid_event_passes(self):
        gate = InvariantGate()
        gate.validate_event({
            "type": "JOB_QUEUED",
            "session_id": "s1",
            "kernel_step_id": "control_plane.enqueue",
            "timestamp": time.time(),
            "payload": {},
        })  # Must not raise.

    def test_missing_type_raises(self):
        gate = InvariantGate()
        with pytest.raises(InvariantViolationError, match="type"):
            gate.validate_event({
                "type": "",
                "session_id": "s1",
                "kernel_step_id": "step",
                "timestamp": time.time(),
            })

    def test_empty_session_id_raises(self):
        gate = InvariantGate()
        with pytest.raises(InvariantViolationError, match="session_id"):
            gate.validate_event({
                "type": "JOB_QUEUED",
                "session_id": "",
                "kernel_step_id": "step",
                "timestamp": time.time(),
            })

    def test_future_timestamp_raises(self):
        gate = InvariantGate()
        with pytest.raises(InvariantViolationError, match="future"):
            gate.validate_event({
                "type": "JOB_QUEUED",
                "session_id": "s1",
                "kernel_step_id": "step",
                "timestamp": time.time() + 999,
            })

    def test_payload_non_dict_raises(self):
        gate = InvariantGate()
        with pytest.raises(InvariantViolationError, match="payload"):
            gate.validate_event({
                "type": "JOB_QUEUED",
                "session_id": "s1",
                "kernel_step_id": "step",
                "timestamp": time.time(),
                "payload": "not a dict",
            })

    def test_missing_kernel_step_id_raises(self):
        gate = InvariantGate()
        with pytest.raises(InvariantViolationError, match="kernel_step_id"):
            gate.validate_event({
                "type": "JOB_QUEUED",
                "session_id": "s1",
                "kernel_step_id": "",
                "timestamp": time.time(),
            })

    def test_ledger_writer_rejects_invalid_event_type(self):
        """InvariantGate is integrated into LedgerWriter — invalid events are blocked."""
        ledger = ExecutionLedger()
        writer = LedgerWriter(ledger)
        with pytest.raises(InvariantViolationError):
            writer.write_event(
                event_type="",  # empty type — invariant violation
                session_id="s1",
                kernel_step_id="test",
            )

    def test_job_invariant_rejects_terminated_session(self):
        from dadbot.core.control_plane import ExecutionJob
        gate = InvariantGate()
        job = ExecutionJob(session_id="s1", user_input="hi")
        session = {"status": "terminated", "session_id": "s1"}
        with pytest.raises(InvariantViolationError, match="terminated"):
            gate.validate_job(session, job)

    def test_violations_observed_counter_increments(self):
        gate = InvariantGate()
        assert gate.violations_observed == 0
        try:
            gate.validate_event({"type": "", "session_id": "s", "kernel_step_id": "k", "timestamp": time.time()})
        except InvariantViolationError:
            pass
        assert gate.violations_observed == 1


# ===========================================================================
# Step 5 — ReducerEngine / semantic check
# ===========================================================================

class TestReducerSemanticCheck:
    def test_reducer_produces_correct_state(self):
        reducer = CanonicalEventReducer()
        ledger = _make_ledger_with_jobs("s1")
        events = ledger.read()
        state = reducer.reduce(events)
        assert "sessions" in state
        assert "s1" in state["sessions"]
        assert state["sessions"]["s1"].get("last_result") == {"reply": "ok"}

    def test_rebuild_from_ledger_matches_reducer_for_state_patches(self):
        reducer = CanonicalEventReducer()
        ledger = ExecutionLedger()
        ledger.write(
            {
                "type": "SESSION_STATE_UPDATED",
                "session_id": "s1",
                "trace_id": "t1",
                "timestamp": 1.0,
                "kernel_step_id": "test.patch.one",
                "payload": {"state": {"name": "Dad"}},
            }
        )
        ledger.write(
            {
                "type": "SESSION_STATE_UPDATED",
                "session_id": "s1",
                "trace_id": "t1",
                "timestamp": 2.0,
                "kernel_step_id": "test.patch.two",
                "payload": {"state": {"mood": "calm"}},
            }
        )

        events = ledger.read()
        reduced = reducer.reduce(events)
        store = SessionStore()
        store.rebuild_from_ledger(events)

        assert store.get("s1") == reduced["sessions"]["s1"]
        assert store.get("s1") == {"name": "Dad", "mood": "calm"}

    def test_system_health_check_reducer_semantic_passes(self):
        ledger = _make_ledger_with_jobs("s1")
        checker = SystemHealthChecker()
        result = checker.check_reducer_semantic_correctness(ledger)
        assert result["ok"] is True

    def test_reducer_semantic_check_empty_ledger(self):
        ledger = ExecutionLedger()
        checker = SystemHealthChecker()
        result = checker.check_reducer_semantic_correctness(ledger)
        assert result["ok"] is True
        assert "empty" in result["reason"]


# ===========================================================================
# Step 6 — Structured observability
# ===========================================================================

class TestObservability:
    def test_metrics_increment_and_counter(self):
        metrics = MetricsSink()
        metrics.increment("job.completed", 3)
        assert metrics.counter("job.completed") == 3

    def test_metrics_observe_and_histogram_summary(self):
        metrics = MetricsSink()
        for i in range(100):
            metrics.observe("job.latency_ms", float(i))
        summary = metrics.histogram_summary("job.latency_ms")
        assert summary["count"] == 100
        assert summary["min"] == 0.0
        assert summary["max"] == 99.0
        assert summary["p99"] >= 97.0

    def test_metrics_snapshot(self):
        metrics = MetricsSink()
        metrics.increment("x")
        metrics.observe("y", 1.5)
        snap = metrics.snapshot()
        assert snap["counters"]["x"] == 1
        assert snap["histograms"]["y"]["count"] == 1

    def test_tracer_span_propagates_trace_id(self):
        tracer = TracingContext()
        with tracer.span("test.op") as span:
            assert TracingContext.current_trace_id() == span.trace_id
        # After span exits, context var should be restored.
        assert TracingContext.current_trace_id() == ""

    def test_tracer_nested_spans(self):
        tracer = TracingContext()
        with tracer.span("outer") as outer:
            outer_trace = TracingContext.current_trace_id()
            with tracer.span("inner", trace_id=outer.trace_id) as inner:
                assert TracingContext.current_trace_id() == inner.trace_id

    def test_event_stream_exporter_captures_records(self):
        import queue
        q = queue.Queue()
        exporter = EventStreamExporter(sink=q, enabled=True)
        exporter.export({"event": "job.completed", "job_id": "123"})
        record = q.get_nowait()
        assert record["event"] == "job.completed"
        assert "exported_at" in record

    def test_exporter_disabled_emits_nothing(self):
        import queue
        q = queue.Queue()
        exporter = EventStreamExporter(sink=q, enabled=False)
        exporter.export({"event": "should-not-appear"})
        assert q.empty()

    def test_ledger_writer_emits_metrics(self):
        # Use the global metrics sink to verify LedgerWriter populates it.
        from dadbot.core.observability import get_metrics
        metrics = get_metrics()
        metrics.reset()  # Clear state from other tests.

        ledger = _make_ledger_with_jobs("obs-session")
        # After writing 5 events (submit, bind, queue, start, complete):
        assert metrics.counter("ledger.write.job_queued") >= 1
        assert metrics.counter("ledger.write.job_completed") >= 1
        assert metrics.counter("ledger.committed_writes") >= 2  # JOB_QUEUED + JOB_COMPLETED

    def test_scheduler_emits_metrics(self):
        from dadbot.core.observability import get_metrics
        metrics = get_metrics()
        metrics.reset()

        async def _run():
            from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry

            async def executor(session, job):
                return {"reply": "ok"}

            plane = ExecutionControlPlane(
                registry=SessionRegistry(),
                kernel_executor=executor,
            )
            return await plane.submit_turn(session_id="metrics-sess", user_input="hi")

        asyncio.run(_run())
        assert metrics.counter("scheduler.job.completed") >= 1
        summary = metrics.histogram_summary("scheduler.job.latency_ms")
        assert summary["count"] >= 1


# ===========================================================================
# Step 7 — Strict write boundary
# ===========================================================================

class TestWriteBoundary:
    def test_strict_mode_blocks_direct_write(self):
        ledger = ExecutionLedger(strict_writes=True)
        with pytest.raises(WriteBoundaryViolationError):
            ledger.write({
                "type": "JOB_QUEUED",
                "session_id": "s1",
                "trace_id": "",
                "timestamp": time.time(),
                "kernel_step_id": "test",
            })

    def test_write_boundary_guard_allows_write(self):
        ledger = ExecutionLedger(strict_writes=True)
        with WriteBoundaryGuard(ledger):
            ledger.write({
                "type": "JOB_QUEUED",
                "session_id": "s1",
                "trace_id": "",
                "timestamp": time.time(),
                "kernel_step_id": "test",
            })
        assert len(ledger.read()) == 1

    def test_write_boundary_guard_revokes_after_exit(self):
        ledger = ExecutionLedger(strict_writes=True)
        with WriteBoundaryGuard(ledger):
            pass
        with pytest.raises(WriteBoundaryViolationError):
            ledger.write({
                "type": "JOB_QUEUED",
                "session_id": "s1",
                "trace_id": "",
                "timestamp": time.time(),
                "kernel_step_id": "test",
            })

    def test_ledger_writer_passes_strict_mode(self):
        """LedgerWriter uses WriteBoundaryGuard internally — should work with strict mode."""
        ledger = ExecutionLedger(strict_writes=True)
        writer = LedgerWriter(ledger)
        from dadbot.core.control_plane import ExecutionJob
        job = ExecutionJob(session_id="s1", user_input="hi")
        writer.append_job_submitted(job)  # Must not raise.
        assert len(ledger.read()) == 1

    def test_non_strict_ledger_allows_direct_write(self):
        ledger = ExecutionLedger(strict_writes=False)
        ledger.write({
            "type": "JOB_QUEUED",
            "session_id": "s1",
            "trace_id": "",
            "timestamp": time.time(),
            "kernel_step_id": "test",
        })  # Must not raise.
        assert len(ledger.read()) == 1


# ===========================================================================
# Step 8 — Snapshot + Restore engine
# ===========================================================================

class TestSnapshotEngine:
    def test_take_snapshot_captures_ledger_head(self):
        ledger = _make_ledger_with_jobs("s1")
        store = SessionStore(ledger=ledger)
        store.rebuild_from_ledger(ledger.read())
        engine = SnapshotEngine()
        snap = engine.take_snapshot(ledger=ledger, session_store=store, label="test")
        assert snap["head_sequence"] > 0
        assert snap["event_count"] > 0
        assert snap["replay_hash"]
        assert snap["snapshot_hash"]

    def test_restore_from_snapshot_and_replay_tail(self):
        ledger = _make_ledger_with_jobs("s1", n=1)
        store = SessionStore(ledger=ledger)
        store.rebuild_from_ledger(ledger.read())
        engine = SnapshotEngine()
        snap = engine.take_snapshot(ledger=ledger, session_store=store, label="mid")

        # Add more events after snapshot.
        writer = LedgerWriter(ledger)
        from dadbot.core.control_plane import ExecutionJob
        job2 = ExecutionJob(session_id="s1", user_input="second")
        writer.append_job_submitted(job2)
        writer.append_session_bound("s1", job2.job_id)
        writer.append_job_queued(job2)
        writer.append_job_started(job2)
        writer.append_job_completed(job2, {"reply": "second"})

        # Restore fresh store from snapshot + replay tail.
        fresh_store = SessionStore()
        engine.restore_from_snapshot(snap, session_store=fresh_store)
        report = engine.replay_tail(snap, ledger=ledger, session_store=fresh_store)

        assert report["tail_events_applied"] > 0
        final_state = fresh_store.get("s1")
        assert final_state is not None
        assert final_state.get("last_result") == {"reply": "second"}

    def test_verify_snapshot_passes_for_consistent_snapshot(self):
        ledger = _make_ledger_with_jobs("s1")
        store = SessionStore(ledger=ledger)
        store.rebuild_from_ledger(ledger.read())
        engine = SnapshotEngine()
        snap = engine.take_snapshot(ledger=ledger, session_store=store)
        report = engine.verify_snapshot(snap, ledger=ledger)
        assert report["ok"] is True

    def test_engine_latest_returns_most_recent(self):
        ledger = _make_ledger_with_jobs("s1")
        store = SessionStore(ledger=ledger)
        store.rebuild_from_ledger(ledger.read())
        engine = SnapshotEngine()
        engine.take_snapshot(ledger=ledger, session_store=store, label="first")
        engine.take_snapshot(ledger=ledger, session_store=store, label="second")
        assert engine.latest()["label"] == "second"

    def test_system_health_check_snapshot_consistency(self):
        ledger = _make_ledger_with_jobs("s1")
        store = SessionStore(ledger=ledger)
        store.rebuild_from_ledger(ledger.read())
        engine = SnapshotEngine()
        engine.take_snapshot(ledger=ledger, session_store=store)
        checker = SystemHealthChecker()
        result = checker.check_snapshot_consistency(ledger=ledger, snapshot_engine=engine)
        assert result["ok"] is True

    def test_no_snapshot_returns_ok(self):
        ledger = ExecutionLedger()
        checker = SystemHealthChecker()
        result = checker.check_snapshot_consistency(ledger=ledger)
        assert result["ok"] is True

    def test_run_all_includes_new_checks(self):
        ledger = _make_ledger_with_jobs("s1")
        store = SessionStore(ledger=ledger)
        store.rebuild_from_ledger(ledger.read())
        checker = SystemHealthChecker()
        result = checker.run_all(ledger=ledger, session_store=store)
        assert "reducer_semantic_correctness" in result
        assert "startup_reconciliation" in result
        assert result["reducer_semantic_correctness"]["ok"] is True
