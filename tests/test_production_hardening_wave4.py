"""Wave 4 production hardening tests — Tier 0/1/2/3 gaps."""
from __future__ import annotations

import asyncio
import gzip
import json
import os
import tempfile
import time
import warnings
import uuid
from pathlib import Path

import pytest
pytestmark = pytest.mark.unit
from dadbot.core.authorization import (
    AuthorizationError,
    Capability,
    CapabilitySet,
    CapabilityToken,
    SessionAuthorizationPolicy,
    TenantBoundary,
    authorize_write,
)
from dadbot.core.compaction import ArchiveTier, CompactionPolicy, EventCompactor
from dadbot.core.durability import (
    AtomicWriteUnit,
    CRC32LineCodec,
    FileLockMutex,
    UNIT_BEGIN_TYPE,
    UNIT_COMMIT_TYPE,
)
from dadbot.core.event_schema import (
    CURRENT_SCHEMA_VERSION,
    EventSchemaMigrator,
    get_migrator,
    stamp_schema_version,
)
from dadbot.core.execution_lease import ExecutionLease, WorkerIdentity
from dadbot.core.execution_ledger import ExecutionLedger, WriteBoundaryGuard
from dadbot.core.fault_injection import (
    ErrorClassification,
    FaultBoundary,
    FaultInjector,
    RetryPolicy,
    RetryableError,
    TerminalError,
    classify_error,
)
from dadbot.core.ledger_backend import (
    BatchWriteBackend,
    CRCFileWALLedgerBackend,
    EventualConsistencyBackend,
    FileWALLedgerBackend,
    InMemoryLedgerBackend,
    SequenceValidator,
    StrongConsistencyBackend,
)
from dadbot.core.ledger_writer import LedgerWriter
from dadbot.core.observability import (
    CorrelationContext,
    ReplayDebugger,
    StructuredLogger,
)
from dadbot.core.snapshot_engine import SnapshotEngine
from dadbot.core.session_store import SessionStore


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _write_jobs(ledger: ExecutionLedger, session_id: str = "s1", n: int = 1) -> None:
    from dadbot.core.control_plane import ExecutionJob
    writer = LedgerWriter(ledger)
    for _ in range(n):
        job = ExecutionJob(session_id=session_id, user_input="hi")
        writer.append_job_submitted(job)
        writer.append_session_bound(session_id, job.job_id)
        writer.append_job_queued(job)
        writer.append_job_started(job)
        writer.append_job_completed(job, {"reply": "ok"})


# ===========================================================================
# Tier 0 — Real durability semantics
# ===========================================================================

class TestCRC32LineCodec:
    def test_encode_decode_roundtrip(self):
        event = {"type": "JOB_QUEUED", "seq": 1, "nested": {"x": 2}}
        line = CRC32LineCodec.encode(event)
        assert line.endswith("\n")
        decoded = CRC32LineCodec.decode(line)
        assert decoded == event

    def test_decode_corrupt_crc_returns_none(self):
        line = "deadbeef {\"type\": \"JOB_QUEUED\"}\n"
        result = CRC32LineCodec.decode(line)
        assert result is None

    def test_decode_truncated_line_returns_none(self):
        result = CRC32LineCodec.decode('0000000a {"type":')  # truncated JSON
        assert result is None

    def test_decode_legacy_plain_json(self):
        """Plain JSON lines (no CRC prefix) are decoded as legacy fallback."""
        result = CRC32LineCodec.decode('{"type": "JOB_QUEUED"}\n')
        assert result is not None
        assert result["type"] == "JOB_QUEUED"

    def test_empty_line_returns_none(self):
        assert CRC32LineCodec.decode("") is None
        assert CRC32LineCodec.decode("   \n") is None


class TestCRCFileWALBackend:
    def test_persist_and_reload_with_crc(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.wal"
            b1 = CRCFileWALLedgerBackend(path, fsync=False)
            b1.append({"type": "JOB_QUEUED", "seq": 1})
            b1.append({"type": "JOB_COMPLETED", "seq": 2})

            b2 = CRCFileWALLedgerBackend(path, fsync=False)
            events = b2.load()
            assert len(events) == 2
            assert events[0]["type"] == "JOB_QUEUED"

    def test_corrupt_line_skipped_with_warning(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.wal"
            # Write one valid CRC line, one corrupt line, one valid.
            valid1 = CRC32LineCodec.encode({"type": "JOB_QUEUED"})
            valid2 = CRC32LineCodec.encode({"type": "JOB_COMPLETED"})
            path.write_text(valid1 + "CORRUPT_LINE\n" + valid2, encoding="utf-8")

            backend = CRCFileWALLedgerBackend(path, fsync=False)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                events = backend.load()
            assert len(events) == 2
            assert any("corrupt" in str(warning.message).lower() for warning in w)

    def test_partial_write_detection(self):
        """A partial last line (no CRC match) is skipped on load."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.wal"
            valid = CRC32LineCodec.encode({"type": "JOB_QUEUED"})
            # Simulate a partial write: truncate the last CRC line mid-way.
            path.write_text(valid + "0000000", encoding="utf-8")
            backend = CRCFileWALLedgerBackend(path, fsync=False)
            events = backend.load()
            assert len(events) == 1

    def test_load_from_backend_with_crc_backend(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.wal"
            b1 = CRCFileWALLedgerBackend(path, fsync=False)
            ledger = ExecutionLedger(backend=b1)
            _write_jobs(ledger, "s1")
            original_count = len(ledger.read())

            b2 = CRCFileWALLedgerBackend(path, fsync=False)
            ledger2 = ExecutionLedger(backend=b2)
            loaded = ledger2.load_from_backend()
            assert loaded == original_count


class TestSequenceValidator:
    def test_monotonic_sequence_passes(self):
        events = [{"sequence": i} for i in range(1, 6)]
        report = SequenceValidator.validate(events)
        assert report["ok"] is True
        assert report["violations"] == []

    def test_out_of_order_sequence_violation(self):
        events = [{"sequence": 1}, {"sequence": 3}, {"sequence": 2}]
        report = SequenceValidator.validate(events)
        assert report["ok"] is False
        assert len(report["violations"]) == 1

    def test_missing_sequence_fields_are_skipped(self):
        events = [{"type": "no_seq"}, {"type": "also_no_seq"}]
        report = SequenceValidator.validate(events)
        assert report["ok"] is True

    def test_load_from_backend_warns_on_sequence_anomaly(self):
        """load_from_backend emits a RuntimeWarning when sequence ordering is broken."""
        class OutOfOrderBackend(InMemoryLedgerBackend):
            def load(self):
                # Return events with out-of-order sequences.
                return [
                    {"type": "JOB_QUEUED", "sequence": 3, "session_id": "s1",
                     "trace_id": "", "timestamp": time.time(), "kernel_step_id": "x",
                     "event_id": "e1", "parent_event_id": "", "_schema_version": "1.0"},
                    {"type": "JOB_QUEUED", "sequence": 1, "session_id": "s1",
                     "trace_id": "", "timestamp": time.time(), "kernel_step_id": "x",
                     "event_id": "e2", "parent_event_id": "e1", "_schema_version": "1.0"},
                ]
        ledger = ExecutionLedger(backend=OutOfOrderBackend())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ledger.load_from_backend()
        assert any("sequence anomaly" in str(warning.message).lower() for warning in w)


class TestFileLockMutex:
    def test_acquire_and_release(self):
        with tempfile.TemporaryDirectory() as d:
            lock = FileLockMutex(Path(d) / f"lock_{uuid.uuid4().hex}.lock")
            token = lock.acquire(timeout_seconds=2)
            assert token
            assert lock.is_held
            released = lock.release(token)
            assert released
            assert not lock.is_held

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as d:
            lock = FileLockMutex(Path(d) / f"lock_{uuid.uuid4().hex}.lock")
            with lock.locked(timeout_seconds=2) as token:
                assert token
                assert lock.is_held
            assert not lock.is_held

    def test_wrong_token_cannot_release(self):
        with tempfile.TemporaryDirectory() as d:
            lock = FileLockMutex(Path(d) / f"lock_{uuid.uuid4().hex}.lock")
            token = lock.acquire(timeout_seconds=2)
            released = lock.release("wrong-token")
            assert not released
            assert lock.is_held
            lock.release(token)  # cleanup

    def test_stale_lock_evicted(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / f"stale_{uuid.uuid4().hex}.lock"
            # Write a lock file with a dead PID and old timestamp.
            record = json.dumps({
                "pid": 99999999,
                "token": "staletoken",
                "acquired_at": time.time() - 9999,
            })
            path.write_text(record)
            lock = FileLockMutex(path, stale_after_seconds=60.0)
            # Should acquire over the stale lock.
            token = lock.acquire(timeout_seconds=1.0)
            assert token
            lock.release(token)


class TestAtomicWriteUnit:
    def test_committed_transaction_events_survive_filter(self):
        ledger = ExecutionLedger()
        writer = LedgerWriter(ledger)
        unit = AtomicWriteUnit(writer)
        with unit.transaction() as txn:
            txn.write_event(
                event_type="JOB_QUEUED",
                session_id="s1",
                kernel_step_id="test.enqueue",
            )
        events = ledger.read()
        filtered = AtomicWriteUnit.filter_committed(events)
        # JOB_QUEUED from the committed unit must survive.
        types = [e["type"] for e in filtered]
        assert "JOB_QUEUED" in types

    def test_uncommitted_transaction_events_filtered_out(self):
        ledger = ExecutionLedger()
        writer = LedgerWriter(ledger)
        unit = AtomicWriteUnit(writer)

        # Simulate crash: write BEGIN but not COMMIT.
        writer.write_event(
            event_type=UNIT_BEGIN_TYPE,
            session_id="__system__",
            kernel_step_id="atomic_write_unit.begin",
            payload={"unit_id": "orphan-unit-id", "ts": time.time()},
        )
        writer.write_event(
            event_type="JOB_QUEUED",
            session_id="s1",
            kernel_step_id="test.enqueue",
            payload={"_unit_id": "orphan-unit-id"},
        )
        # No UNIT_COMMIT written.

        events = ledger.read()
        filtered = AtomicWriteUnit.filter_committed(events)
        types = [e["type"] for e in filtered]
        assert UNIT_BEGIN_TYPE not in types
        assert "JOB_QUEUED" not in types

    def test_transaction_with_exception_is_not_committed(self):
        ledger = ExecutionLedger()
        writer = LedgerWriter(ledger)
        unit = AtomicWriteUnit(writer)
        txn = unit.transaction()
        txn.__enter__()
        try:
            raise RuntimeError("simulated crash")
        except RuntimeError:
            txn.__exit__(RuntimeError, RuntimeError("x"), None)

        assert not txn.committed
        events = ledger.read()
        filtered = AtomicWriteUnit.filter_committed(events)
        assert all(e.get("type") not in (UNIT_BEGIN_TYPE, UNIT_COMMIT_TYPE) or
                   e.get("payload", {}).get("unit_id") != txn.unit_id
                   for e in filtered)

    def test_ledger_replay_filter_integration(self):
        """add_replay_filter(AtomicWriteUnit.filter_committed) is applied on load."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.wal"
            b = FileWALLedgerBackend(path, fsync=False)
            ledger = ExecutionLedger(backend=b)
            ledger.add_replay_filter(AtomicWriteUnit.filter_committed)
            writer = LedgerWriter(ledger)

            # Write an orphaned unit (BEGIN but no COMMIT).
            writer.write_event(
                event_type=UNIT_BEGIN_TYPE,
                session_id="__system__",
                kernel_step_id="atomic_write_unit.begin",
                payload={"unit_id": "phantom", "ts": time.time()},
            )
            writer.write_event(
                event_type="JOB_QUEUED",
                session_id="s1",
                kernel_step_id="test",
                payload={"_unit_id": "phantom"},
            )

            # Reload.
            b2 = FileWALLedgerBackend(path, fsync=False)
            ledger2 = ExecutionLedger(backend=b2)
            ledger2.add_replay_filter(AtomicWriteUnit.filter_committed)
            ledger2.load_from_backend()
            types = [e["type"] for e in ledger2.read()]
            assert "JOB_QUEUED" not in types


# ===========================================================================
# Tier 0 — Transactional atomicity
# ===========================================================================

class TestTransactionalAtomicity:
    def test_transaction_commit_writes_all_events(self):
        ledger = ExecutionLedger()
        writer = LedgerWriter(ledger)
        unit = AtomicWriteUnit(writer)
        with unit.transaction() as txn:
            txn.write_event(event_type="JOB_QUEUED", session_id="s1", kernel_step_id="step")
            txn.write_event(event_type="JOB_STARTED", session_id="s1", kernel_step_id="step")
        assert txn.committed
        types = {e["type"] for e in ledger.read()}
        assert "JOB_QUEUED" in types
        assert "JOB_STARTED" in types
        assert UNIT_COMMIT_TYPE in types

    def test_transaction_rollback_discards_events_on_filter(self):
        ledger = ExecutionLedger()
        writer = LedgerWriter(ledger)
        unit = AtomicWriteUnit(writer)
        txn = unit.transaction()
        txn.__enter__()
        txn.write_event(event_type="JOB_QUEUED", session_id="s1", kernel_step_id="step")
        txn.__exit__(RuntimeError, RuntimeError("boom"), None)

        filtered = AtomicWriteUnit.filter_committed(ledger.read())
        assert all(e["type"] not in ("JOB_QUEUED",) for e in filtered
                   if e.get("payload", {}).get("_unit_id") == txn.unit_id)


# ===========================================================================
# Tier 1 — Fault injection and retry
# ===========================================================================

class TestFaultInjector:
    def test_arm_and_trigger_once(self):
        inj = FaultInjector()
        inj.arm("ledger.append", count=1)
        with pytest.raises(RetryableError, match="ledger.append"):
            inj.check("ledger.append")
        # Second call — no fault.
        inj.check("ledger.append")  # must not raise

    def test_triggered_count_increments(self):
        inj = FaultInjector()
        inj.arm("x", count=2)
        for _ in range(2):
            try:
                inj.check("x")
            except RetryableError:
                pass
        assert inj.triggered_count("x") == 2

    def test_terminal_error_type(self):
        inj = FaultInjector()
        inj.arm("y", count=1, exc_type=TerminalError)
        with pytest.raises(TerminalError):
            inj.check("y")

    def test_disarm_stops_faults(self):
        inj = FaultInjector()
        inj.arm("z", count=5)
        inj.disarm("z")
        inj.check("z")  # must not raise

    def test_reset_clears_all(self):
        inj = FaultInjector()
        inj.arm("a", count=3)
        inj.reset()
        inj.check("a")  # must not raise


class TestErrorClassifier:
    def test_classify_retryable_error(self):
        assert classify_error(RetryableError("x")) == ErrorClassification.RETRYABLE

    def test_classify_terminal_error(self):
        assert classify_error(TerminalError("x")) == ErrorClassification.TERMINAL

    def test_classify_connection_error(self):
        assert classify_error(ConnectionError("x")) == ErrorClassification.RETRYABLE

    def test_classify_value_error(self):
        assert classify_error(ValueError("x")) == ErrorClassification.TERMINAL


class TestRetryPolicy:
    def test_succeeds_on_second_attempt(self):
        call_count = [0]

        def flaky():
            call_count[0] += 1
            if call_count[0] < 2:
                raise RetryableError("not yet")
            return "ok"

        policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.0)
        result = policy.execute(flaky)
        assert result == "ok"
        assert call_count[0] == 2

    def test_terminal_error_not_retried(self):
        call_count = [0]

        def always_terminal():
            call_count[0] += 1
            raise TerminalError("done")

        policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.0)
        with pytest.raises(TerminalError):
            policy.execute(always_terminal)
        assert call_count[0] == 1  # not retried

    def test_exhausted_retries_raises(self):
        policy = RetryPolicy(max_attempts=2, base_delay_seconds=0.0)
        with pytest.raises(RetryableError):
            policy.execute(lambda: (_ for _ in ()).throw(RetryableError("x")))

    def test_should_retry_respects_max_attempts(self):
        policy = RetryPolicy(max_attempts=3)
        assert policy.should_retry(RetryableError("x"), 1) is True
        assert policy.should_retry(RetryableError("x"), 2) is True
        assert policy.should_retry(RetryableError("x"), 3) is False


class TestFaultBoundary:
    def test_retryable_handler_called_from_body(self):
        """on_retryable handler is called when a RetryableError is raised in the body."""
        captured = []
        with FaultBoundary(on_retryable=captured.append):
            raise RetryableError("from body")
        assert len(captured) == 1
        assert isinstance(captured[0], RetryableError)

    def test_terminal_handler_called_from_body(self):
        captured = []
        with FaultBoundary(on_terminal=captured.append):
            raise TerminalError("permanent")
        assert len(captured) == 1

    def test_no_exception_noop(self):
        called = []
        with FaultBoundary("unused", on_terminal=called.append):
            pass
        assert called == []

    def test_unhandled_exception_reraises(self):
        with pytest.raises(ValueError):
            with FaultBoundary():
                raise ValueError("unhandled")


# ===========================================================================
# Tier 1 — Event schema versioning
# ===========================================================================

class TestEventSchemaVersioning:
    def test_stamp_adds_version(self):
        event = {"type": "JOB_QUEUED"}
        stamped = stamp_schema_version(event)
        assert stamped["_schema_version"] == CURRENT_SCHEMA_VERSION

    def test_stamp_does_not_overwrite_existing(self):
        event = {"type": "JOB_QUEUED", "_schema_version": "0.9"}
        stamped = stamp_schema_version(event)
        assert stamped["_schema_version"] == "0.9"

    def test_new_events_are_stamped_by_ledger(self):
        ledger = ExecutionLedger()
        writer = LedgerWriter(ledger)
        from dadbot.core.control_plane import ExecutionJob
        job = ExecutionJob(session_id="s1", user_input="hi")
        writer.append_job_submitted(job)
        event = ledger.read()[0]
        assert event.get("_schema_version") == CURRENT_SCHEMA_VERSION

    def test_migrator_upgrades_legacy_event(self):
        migrator = EventSchemaMigrator()
        event = {"type": "JOB_QUEUED"}  # no _schema_version
        migrated = migrator.migrate(event)
        assert migrated["_schema_version"] == CURRENT_SCHEMA_VERSION

    def test_migrator_applies_registered_step(self):
        migrator = EventSchemaMigrator()
        migrator.register(
            "0.9",
            "1.0",
            lambda e: {**e, "kernel_step_id": e.get("old_step_id", "")},
        )
        event = {"type": "JOB_QUEUED", "_schema_version": "0.9", "old_step_id": "sched.run"}
        migrated = migrator.migrate(event)
        assert migrated["_schema_version"] == CURRENT_SCHEMA_VERSION
        assert migrated.get("kernel_step_id") == "sched.run"

    def test_migrator_migrate_all(self):
        migrator = EventSchemaMigrator()
        events = [{"type": "A"}, {"type": "B", "_schema_version": "1.0"}]
        migrated = migrator.migrate_all(events)
        assert all(e["_schema_version"] == CURRENT_SCHEMA_VERSION for e in migrated)

    def test_needs_migration_false_for_current(self):
        migrator = EventSchemaMigrator()
        event = {"_schema_version": CURRENT_SCHEMA_VERSION}
        assert not migrator.needs_migration(event)

    def test_needs_migration_true_for_legacy(self):
        migrator = EventSchemaMigrator()
        event = {"type": "X"}  # no version
        assert migrator.needs_migration(event)


# ===========================================================================
# Tier 2 — Fencing tokens + WorkerIdentity
# ===========================================================================

class TestFencingTokens:
    def test_fencing_token_increments_on_new_acquisition(self):
        lease = ExecutionLease()
        l1 = lease.acquire(session_id="s1", owner_id="worker-1")
        lease.release(session_id="s1", owner_id="worker-1")
        l2 = lease.acquire(session_id="s1", owner_id="worker-2")
        assert l2["fencing_token"] > l1["fencing_token"]

    def test_fencing_token_stable_on_renewal(self):
        lease = ExecutionLease()
        l1 = lease.acquire(session_id="s1", owner_id="worker-1")
        l2 = lease.renew(session_id="s1", owner_id="worker-1")
        assert l2["fencing_token"] == l1["fencing_token"]

    def test_fencing_token_for_returns_current(self):
        lease = ExecutionLease()
        l1 = lease.acquire(session_id="s1", owner_id="w1")
        token = lease.fencing_token_for(session_id="s1")
        assert token == l1["fencing_token"]

    def test_fencing_token_for_returns_none_when_not_held(self):
        lease = ExecutionLease()
        assert lease.fencing_token_for(session_id="no-session") is None


class TestWorkerIdentity:
    def test_generates_stable_id(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / ".worker_identity.json"
            w1 = WorkerIdentity(path)
            w2 = WorkerIdentity(path)
            assert w1.worker_id == w2.worker_id

    def test_creates_identity_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / ".worker_identity.json"
            w = WorkerIdentity(path)
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["worker_id"] == w.worker_id
            assert data["pid"] == os.getpid()

    def test_worker_id_is_nonempty_string(self):
        with tempfile.TemporaryDirectory() as d:
            w = WorkerIdentity(Path(d) / ".worker_identity.json")
            assert isinstance(w.worker_id, str)
            assert len(w.worker_id) > 0


# ===========================================================================
# Tier 2 — Observability extensions
# ===========================================================================

class TestCorrelationContext:
    def setup_method(self):
        from dadbot.core.observability import _current_correlation_id
        _current_correlation_id.set("")

    def test_bind_sets_correlation_id(self):
        with CorrelationContext.bind("req-001") as cid:
            assert cid == "req-001"
            assert CorrelationContext.current() == "req-001"
        assert CorrelationContext.current() == ""

    def test_auto_generated_id(self):
        with CorrelationContext.bind() as cid:
            assert len(cid) > 0
            assert CorrelationContext.current() == cid

    def test_ensure_creates_if_absent(self):
        cid = CorrelationContext.ensure()
        assert cid  # non-empty
        # Reset for next test.
        from dadbot.core.observability import _current_correlation_id
        _current_correlation_id.set("")


class TestStructuredLogger:
    def test_info_record_emitted(self):
        records: list = []
        logger = StructuredLogger("test.logger", sink=records)
        logger.info("job started", session_id="s1", event_id="evt-1")
        assert len(records) == 1
        r = records[0]
        assert r["level"] == "INFO"
        assert r["message"] == "job started"
        assert r["session_id"] == "s1"
        assert r["event_id"] == "evt-1"

    def test_records_include_trace_context(self):
        records: list = []
        logger = StructuredLogger("test", sink=records)
        from dadbot.core.observability import get_tracer
        tracer = get_tracer()
        with tracer.span("test.span") as span:
            logger.info("inside span")
        r = records[0]
        assert r["trace_id"] == span.trace_id

    def test_records_include_correlation_id(self):
        records: list = []
        logger = StructuredLogger("test", sink=records)
        with CorrelationContext.bind("corr-xyz"):
            logger.info("correlated")
        assert records[0]["correlation_id"] == "corr-xyz"

    def test_min_level_filters_lower_levels(self):
        records: list = []
        logger = StructuredLogger("test", sink=records, min_level="WARNING")
        logger.debug("should be filtered")
        logger.info("also filtered")
        logger.warning("should appear")
        assert len(records) == 1
        assert records[0]["level"] == "WARNING"

    def test_clear_empties_records(self):
        records: list = []
        logger = StructuredLogger("test", sink=records)
        logger.info("one")
        logger.clear()
        assert logger.records() == []


class TestReplayDebugger:
    def test_step_through_yields_incremental_state(self):
        ledger = ExecutionLedger()
        _write_jobs(ledger, "s1")
        debugger = ReplayDebugger()
        steps = list(debugger.step_through(ledger.read()))
        assert len(steps) == len(ledger.read())
        for i, step in enumerate(steps, start=1):
            assert step["seq"] == i
            assert "event" in step
            assert "state" in step

    def test_debug_session_filters_by_session(self):
        ledger = ExecutionLedger()
        _write_jobs(ledger, "s1")
        _write_jobs(ledger, "s2")
        debugger = ReplayDebugger()
        steps = debugger.debug_session("s1", ledger.read())
        assert all(s["event"]["session_id"] == "s1" for s in steps)
        assert len(steps) > 0

    def test_diff_states_detects_change(self):
        debugger = ReplayDebugger()
        before = {"status": "idle", "count": 0}
        after  = {"status": "running", "count": 1}
        diff = debugger.diff_states(before, after)
        assert "status" in diff
        assert "count" in diff
        assert diff["status"]["before"] == "idle"
        assert diff["status"]["after"] == "running"

    def test_diff_states_empty_when_equal(self):
        debugger = ReplayDebugger()
        state = {"x": 1}
        assert debugger.diff_states(state, state) == {}


# ===========================================================================
# Tier 2 — Snapshot versioning / schema evolution
# ===========================================================================

class TestSchemaEvolutionWithReplay:
    def test_migrate_all_events_on_load(self):
        """load_from_backend automatically migrates legacy events via EventSchemaMigrator."""
        class LegacyBackend(InMemoryLedgerBackend):
            def load(self):
                # Return events without _schema_version (legacy).
                return [
                    {
                        "type": "JOB_QUEUED",
                        "session_id": "s1",
                        "trace_id": "",
                        "timestamp": time.time(),
                        "kernel_step_id": "test",
                        "event_id": "e1",
                        "parent_event_id": "",
                    },
                ]

        ledger = ExecutionLedger(backend=LegacyBackend())
        ledger.load_from_backend()
        events = ledger.read()
        # All events should now have a schema version.
        assert all(e.get("_schema_version") == CURRENT_SCHEMA_VERSION for e in events)


# ===========================================================================
# Tier 3 — Consistency mode wrappers
# ===========================================================================

class TestConsistencyModeBackends:
    def test_strong_consistency_backend_passes_through(self):
        with tempfile.TemporaryDirectory() as d:
            inner = FileWALLedgerBackend(Path(d) / "ledger.wal", fsync=False)
            strong = StrongConsistencyBackend(inner)
            strong.append({"type": "JOB_QUEUED"})
            events = strong.load()
            assert len(events) == 1

    def test_strong_consistency_requires_file_backend(self):
        with pytest.raises(TypeError):
            StrongConsistencyBackend(InMemoryLedgerBackend())

    def test_eventual_consistency_buffers_writes(self):
        inner = InMemoryLedgerBackend()
        eventual = EventualConsistencyBackend(inner, buffer_size=5)
        for i in range(3):
            eventual.append({"type": "X", "seq": i})
        # Not yet flushed to inner.
        assert eventual.buffered_count == 3
        assert len(inner.load()) == 0

    def test_eventual_consistency_flushes_on_committed(self):
        inner = InMemoryLedgerBackend()
        eventual = EventualConsistencyBackend(inner, buffer_size=100)
        eventual.append({"type": "JOB_QUEUED"}, committed=True)  # triggers flush
        assert eventual.buffered_count == 0
        assert len(inner.load()) == 1

    def test_eventual_consistency_flushes_on_load(self):
        inner = InMemoryLedgerBackend()
        eventual = EventualConsistencyBackend(inner, buffer_size=100)
        for i in range(3):
            eventual.append({"type": "X"})
        events = eventual.load()  # triggers flush
        assert len(events) == 3

    def test_batch_write_backend_batches_writes(self):
        inner = InMemoryLedgerBackend()
        batch = BatchWriteBackend(inner, batch_size=3)
        batch.append({"type": "A"})
        batch.append({"type": "B"})
        assert batch.pending_count == 2
        assert len(inner.load()) == 0

    def test_batch_write_flushes_at_threshold(self):
        inner = InMemoryLedgerBackend()
        batch = BatchWriteBackend(inner, batch_size=2)
        batch.append({"type": "A"})
        batch.append({"type": "B"})  # threshold hit
        assert batch.pending_count == 0
        assert len(inner.load()) == 2

    def test_batch_write_flushes_on_committed(self):
        inner = InMemoryLedgerBackend()
        batch = BatchWriteBackend(inner, batch_size=100)
        batch.append({"type": "JOB_QUEUED"}, committed=True)
        assert batch.pending_count == 0
        assert len(inner.load()) == 1


# ===========================================================================
# Tier 3 — Event compaction and archive tier
# ===========================================================================

class TestCompaction:
    def test_compact_removes_pre_snapshot_events(self):
        ledger = ExecutionLedger()
        _write_jobs(ledger, "s1", n=2)
        store = SessionStore(ledger=ledger)
        store.rebuild_from_ledger(ledger.read())
        engine = SnapshotEngine()
        snapshot = engine.take_snapshot(ledger=ledger, session_store=store)

        before = len(ledger.read())
        policy = CompactionPolicy(max_events=1, min_snapshot_distance=0)  # force cutoff to head
        compactor = EventCompactor(policy=policy)
        report = compactor.compact(ledger=ledger, snapshot=snapshot, force=True)
        assert report["compacted"] is True
        assert report["events_removed"] > 0
        assert len(ledger.read()) < before

    def test_compact_without_snapshot_skipped(self):
        ledger = ExecutionLedger()
        _write_jobs(ledger, "s1")
        compactor = EventCompactor()
        report = compactor.compact(ledger=ledger, snapshot=None)
        assert report["compacted"] is False
        assert report["reason"] == "no_snapshot"

    def test_compact_below_threshold_skipped(self):
        ledger = ExecutionLedger()
        _write_jobs(ledger, "s1")
        store = SessionStore(ledger=ledger)
        store.rebuild_from_ledger(ledger.read())
        engine = SnapshotEngine()
        snapshot = engine.take_snapshot(ledger=ledger, session_store=store)

        policy = CompactionPolicy(max_events=100_000)  # won't trigger
        compactor = EventCompactor(policy=policy)
        report = compactor.compact(ledger=ledger, snapshot=snapshot)
        assert report["compacted"] is False

    def test_archive_tier_writes_and_reads_back(self):
        with tempfile.TemporaryDirectory() as d:
            archive = ArchiveTier(d)
            events = [{"type": "JOB_QUEUED", "seq": i} for i in range(5)]
            path = archive.archive(events, label="test")
            assert path.exists()
            loaded = archive.load_archive(path)
            assert len(loaded) == 5
            assert loaded[0]["type"] == "JOB_QUEUED"

    def test_archive_tier_list_archives(self):
        with tempfile.TemporaryDirectory() as d:
            archive = ArchiveTier(d)
            archive.archive([{"type": "A"}], label="first")
            archive.archive([{"type": "B"}], label="second")
            archives = archive.list_archives()
            assert len(archives) == 2

    def test_compaction_with_archive_tier(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = ExecutionLedger()
            _write_jobs(ledger, "s1", n=3)
            store = SessionStore(ledger=ledger)
            store.rebuild_from_ledger(ledger.read())
            engine = SnapshotEngine()
            snapshot = engine.take_snapshot(ledger=ledger, session_store=store)

            archive = ArchiveTier(d)
            policy = CompactionPolicy(max_events=1, min_snapshot_distance=0)
            compactor = EventCompactor(policy=policy, archive=archive)
            report = compactor.compact(ledger=ledger, snapshot=snapshot, force=True)

            if report["compacted"]:
                assert report["archive_path"] is not None
                assert Path(report["archive_path"]).exists()


# ===========================================================================
# Tier 3 — Authorization and tenant isolation
# ===========================================================================

class TestCapabilitySet:
    def test_admin_grants_all(self):
        caps = CapabilitySet.full()
        assert caps.has(Capability.READ)
        assert caps.has(Capability.WRITE)
        assert caps.has(Capability.EXECUTE)

    def test_read_only_denies_write(self):
        caps = CapabilitySet.read_only()
        assert caps.has(Capability.READ)
        assert not caps.has(Capability.WRITE)

    def test_empty_denies_all(self):
        caps = CapabilitySet.empty()
        for c in Capability:
            assert not caps.has(c)


class TestTenantBoundary:
    def test_valid_prefix_allowed(self):
        boundary = TenantBoundary({"acme-", "beta-"})
        boundary.validate_session("acme-user-001")  # must not raise

    def test_unknown_prefix_raises(self):
        boundary = TenantBoundary({"acme-"})
        with pytest.raises(AuthorizationError):
            boundary.validate_session("evil-user-001")

    def test_disabled_boundary_allows_all(self):
        boundary = TenantBoundary({"acme-"}, enabled=False)
        boundary.validate_session("any-session-id")  # must not raise

    def test_no_prefixes_allows_all(self):
        boundary = TenantBoundary()
        boundary.validate_session("any-session-id")  # must not raise


class TestSessionAuthorizationPolicy:
    def test_grant_and_require(self):
        policy = SessionAuthorizationPolicy()
        policy.grant("s1", CapabilitySet.read_write())
        policy.require("s1", Capability.WRITE)  # must not raise

    def test_require_raises_on_missing_cap(self):
        policy = SessionAuthorizationPolicy(strict=True)
        policy.grant("s1", CapabilitySet.read_only())
        with pytest.raises(AuthorizationError, match="write"):
            policy.require("s1", Capability.WRITE)

    def test_strict_mode_denies_unregistered(self):
        policy = SessionAuthorizationPolicy(strict=True)
        with pytest.raises(AuthorizationError):
            policy.require("unknown-session", Capability.READ)

    def test_revoke_removes_grant(self):
        policy = SessionAuthorizationPolicy(strict=True)
        policy.grant("s1", CapabilitySet.full())
        policy.revoke("s1")
        with pytest.raises(AuthorizationError):
            policy.require("s1", Capability.READ)

    def test_authorize_write_convenience(self):
        policy = SessionAuthorizationPolicy(strict=True)
        policy.grant("s1", CapabilitySet.read_write())
        authorize_write(policy, "s1")  # must not raise

    def test_authorize_write_none_policy_noop(self):
        authorize_write(None, "any-session")  # must not raise


class TestCapabilityToken:
    def test_issue_and_verify(self):
        issuer = CapabilityToken()
        token = issuer.issue(session_id="s1", capability=Capability.WRITE)
        assert issuer.verify(token, session_id="s1", capability=Capability.WRITE)

    def test_wrong_session_rejected(self):
        issuer = CapabilityToken()
        token = issuer.issue(session_id="s1", capability=Capability.WRITE)
        assert not issuer.verify(token, session_id="s2", capability=Capability.WRITE)

    def test_wrong_capability_rejected(self):
        issuer = CapabilityToken()
        token = issuer.issue(session_id="s1", capability=Capability.READ)
        assert not issuer.verify(token, session_id="s1", capability=Capability.WRITE)

    def test_expired_token_rejected(self):
        issuer = CapabilityToken()
        token = issuer.issue(session_id="s1", capability=Capability.READ, ttl_seconds=-1)
        assert not issuer.verify(token, session_id="s1", capability=Capability.READ)

    def test_tampered_token_rejected(self):
        issuer = CapabilityToken()
        token = issuer.issue(session_id="s1", capability=Capability.WRITE)
        tampered = token[:-5] + "XXXXX"
        assert not issuer.verify(tampered, session_id="s1", capability=Capability.WRITE)

    def test_different_key_rejected(self):
        issuer1 = CapabilityToken(secret_key=b"a" * 32)
        issuer2 = CapabilityToken(secret_key=b"b" * 32)
        token = issuer1.issue(session_id="s1", capability=Capability.WRITE)
        assert not issuer2.verify(token, session_id="s1", capability=Capability.WRITE)


# ===========================================================================
# sealed_events and add_replay_filter
# ===========================================================================

class TestSealedEventsAndReplayFilters:
    def test_sealed_events_returns_tuple(self):
        ledger = ExecutionLedger()
        _write_jobs(ledger, "s1")
        sealed = ledger.sealed_events
        assert isinstance(sealed, tuple)
        assert len(sealed) > 0

    def test_sealed_events_is_immutable_copy(self):
        ledger = ExecutionLedger()
        _write_jobs(ledger, "s1")
        sealed = ledger.sealed_events
        # Mutating the returned tuple's dict should not affect internal state.
        sealed[0]["type"] = "HACKED"
        assert ledger.read()[0]["type"] != "HACKED"

    def test_replay_filter_applied_on_load(self):
        inner = InMemoryLedgerBackend()
        inner.append({
            "type": "FILTERED_OUT",
            "session_id": "s1",
            "trace_id": "",
            "timestamp": time.time(),
            "kernel_step_id": "x",
            "event_id": "e1",
            "parent_event_id": "",
        })
        ledger = ExecutionLedger(backend=inner)
        ledger.add_replay_filter(
            lambda events: [e for e in events if e.get("type") != "FILTERED_OUT"]
        )
        ledger.load_from_backend()
        assert len(ledger.read()) == 0
