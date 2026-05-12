import asyncio
from datetime import datetime

import pytest

from dadbot.core.control_plane import (
    ExecutionControlPlane,
    ExecutionJob,
    InMemoryExecutionLedger,
    LedgerReader,
    Scheduler,
    SessionRegistry,
    _classify_execution_failure,
)
from dadbot.core.contracts.lifecycle_events import Completed
from dadbot.core.ledger_writer import LedgerWriter
from dadbot.core.replay_verifier import ReplayVerifier
from dadbot.core.runtime_errors import ReplayMismatch
from dadbot.core.session_store import SessionStore

pytestmark = pytest.mark.integration


def test_control_plane_idempotency_dedupes_concurrent_requests():
    registry = SessionRegistry()
    calls = {"count": 0}

    async def _kernel_execute(_session, _job):
        calls["count"] += 1
        await asyncio.sleep(0.01)
        return ("ok", False)

    control_plane = ExecutionControlPlane(registry=registry, kernel_executor=_kernel_execute)

    async def _run():
        return await asyncio.gather(
            control_plane.submit_turn(
                session_id="s1",
                user_input="hi",
                metadata={"request_id": "req-1", "trace_id": "t1"},
            ),
            control_plane.submit_turn(
                session_id="s1",
                user_input="hi again",
                metadata={"request_id": "req-1", "trace_id": "t1"},
            ),
        )

    results = asyncio.run(_run())
    assert results == [("ok", False), ("ok", False)]
    assert calls["count"] == 1

    queued = [event for event in control_plane.ledger_events() if str(event.get("type") or "") == "JOB_QUEUED"]
    assert len(queued) == 1
    assert queued[0]["payload"]["request_id"] == "req-1"


def test_control_plane_idempotency_dedupes_across_restart_with_shared_ledger():
    shared_ledger = InMemoryExecutionLedger()
    registry = SessionRegistry()
    calls = {"count": 0}

    async def _kernel_execute(_session, _job):
        calls["count"] += 1
        return ("ok", False)

    first_plane = ExecutionControlPlane(
        registry=registry,
        kernel_executor=_kernel_execute,
        ledger=shared_ledger,
    )

    first = asyncio.run(
        first_plane.submit_turn(
            session_id="s1",
            user_input="hello",
            metadata={"request_id": "req-restart-1", "trace_id": "tr-restart-1"},
        )
    )
    assert first == ("ok", False)
    assert calls["count"] == 1

    second_plane = ExecutionControlPlane(
        registry=registry,
        kernel_executor=_kernel_execute,
        ledger=shared_ledger,
    )
    second = asyncio.run(
        second_plane.submit_turn(
            session_id="s1",
            user_input="hello-again",
            metadata={"request_id": "req-restart-1", "trace_id": "tr-restart-1b"},
        )
    )

    assert second == ("ok", False)
    assert calls["count"] == 1
    queued = [event for event in shared_ledger.read() if str(event.get("type") or "") == "JOB_QUEUED"]
    assert len(queued) == 1


def test_control_plane_idempotency_blocks_ambiguous_effect_replay():
    shared_ledger = InMemoryExecutionLedger()
    registry = SessionRegistry()
    calls = {"count": 0}

    async def _kernel_execute(_session, _job):
        calls["count"] += 1
        return ("ok", False)

    seed_plane = ExecutionControlPlane(
        registry=registry,
        kernel_executor=_kernel_execute,
        ledger=shared_ledger,
    )
    seeded_job = ExecutionJob(
        session_id="s1",
        user_input="seed",
        metadata={"request_id": "req-ambiguous-1", "trace_id": "tr-ambiguous-1"},
        trace_id="tr-ambiguous-1",
    )
    seed_writer = LedgerWriter(shared_ledger)
    seed_writer.append_job_submitted(seeded_job)
    seed_writer.append_job_started(seeded_job)

    replay_plane = ExecutionControlPlane(
        registry=registry,
        kernel_executor=_kernel_execute,
        ledger=shared_ledger,
    )

    with pytest.raises(ReplayMismatch):
        asyncio.run(
            replay_plane.submit_turn(
                session_id="s1",
                user_input="retry",
                metadata={"request_id": "req-ambiguous-1", "trace_id": "tr-ambiguous-retry"},
            )
        )
    assert calls["count"] == 0


def test_effect_journal_begin_commit_dedupes_replay_with_effect_id():
    shared_ledger = InMemoryExecutionLedger()
    registry = SessionRegistry()
    calls = {"count": 0}

    async def _kernel_execute(_session, _job):
        calls["count"] += 1
        return ("ok", False)

    first_plane = ExecutionControlPlane(
        registry=registry,
        kernel_executor=_kernel_execute,
        ledger=shared_ledger,
    )
    first = asyncio.run(
        first_plane.submit_turn(
            session_id="s-effect",
            user_input="hello",
            metadata={"request_id": "req-effect-1", "effect_id": "eff-1", "trace_id": "tr-eff-1"},
        )
    )
    assert first == ("ok", False)
    assert calls["count"] == 1

    second_plane = ExecutionControlPlane(
        registry=registry,
        kernel_executor=_kernel_execute,
        ledger=shared_ledger,
    )
    second = asyncio.run(
        second_plane.submit_turn(
            session_id="s-effect",
            user_input="hello-again",
            metadata={"request_id": "req-effect-1", "effect_id": "eff-1", "trace_id": "tr-eff-1b"},
        )
    )
    assert second == ("ok", False)
    assert calls["count"] == 1

    events = shared_ledger.read()
    assert len([event for event in events if str(event.get("type") or "") == "EFFECT_BEGIN"]) == 1
    assert len([event for event in events if str(event.get("type") or "") == "EFFECT_COMMIT"]) == 1


def test_effect_journal_blocks_crash_restart_mid_lease_replay():
    shared_ledger = InMemoryExecutionLedger()
    registry = SessionRegistry()

    seed_writer = LedgerWriter(shared_ledger)
    seed_writer.append_effect_begin(
        session_id="s-effect-crash",
        trace_id="tr-effect-crash",
        effect_id="eff-crash-1",
        request_id="req-crash-1",
    )

    async def _kernel_execute(_session, _job):
        return ("ok", False)

    replay_plane = ExecutionControlPlane(
        registry=registry,
        kernel_executor=_kernel_execute,
        ledger=shared_ledger,
    )

    with pytest.raises(ReplayMismatch):
        asyncio.run(
            replay_plane.submit_turn(
                session_id="s-effect-crash",
                user_input="retry",
                metadata={"request_id": "req-crash-1", "effect_id": "eff-crash-1", "trace_id": "tr-effect-retry"},
            )
        )

    events = shared_ledger.read()
    reconcile_required = [
        event for event in events if str(event.get("type") or "") == "JOB_RECONCILE_REQUIRED"
    ]
    assert reconcile_required, "expected explicit reconcile-required policy event"
    payload = dict(reconcile_required[-1].get("payload") or {})
    assert payload.get("effect_id") == "eff-crash-1"
    assert payload.get("request_id") == "req-crash-1"
    assert payload.get("reason") == "ambiguous_effect_begin_without_commit"


def test_apply_reconciliation_closes_ambiguous_request_and_effect_state():
    shared_ledger = InMemoryExecutionLedger()
    registry = SessionRegistry()

    seed_writer = LedgerWriter(shared_ledger)
    seed_job = ExecutionJob(
        session_id="s-reconcile",
        user_input="hello",
        metadata={"request_id": "req-reconcile-1", "trace_id": "tr-seed-reconcile"},
        trace_id="tr-seed-reconcile",
    )
    seed_writer.append_job_started(seed_job)
    seed_writer.append_effect_begin(
        session_id="s-reconcile",
        trace_id="tr-seed-reconcile",
        effect_id="eff-reconcile-1",
        request_id="req-reconcile-1",
    )

    async def _kernel_execute(_session, _job):
        return ("ok", False)

    plane = ExecutionControlPlane(
        registry=registry,
        kernel_executor=_kernel_execute,
        ledger=shared_ledger,
    )

    before = plane.boot_reconcile()
    before_reconcile = dict(before.get("effect_reconciliation") or {})
    assert bool(before_reconcile.get("reconcile_required")) is True

    report = plane.apply_reconciliation(
        session_id="s-reconcile",
        request_id="req-reconcile-1",
        effect_id="eff-reconcile-1",
        reason="test_manual_reconcile",
    )
    assert bool(report.get("applied")) is True
    assert "JOB_RECONCILED" in list(report.get("events") or [])
    assert "EFFECT_RECONCILED" in list(report.get("events") or [])

    after = plane.boot_reconcile()
    after_reconcile = dict(after.get("effect_reconciliation") or {})
    assert bool(after_reconcile.get("reconcile_required")) is False

    event_types = [str(event.get("type") or "") for event in shared_ledger.read()]
    assert "JOB_RECONCILED" in event_types
    assert "EFFECT_RECONCILED" in event_types


def test_apply_reconciliation_resume_mode_records_resume_eligible_resolution():
    shared_ledger = InMemoryExecutionLedger()
    registry = SessionRegistry()

    seed_writer = LedgerWriter(shared_ledger)
    seed_job = ExecutionJob(
        session_id="s-reconcile-resume",
        user_input="hello",
        metadata={"request_id": "req-reconcile-resume-1", "trace_id": "tr-seed-resume"},
        trace_id="tr-seed-resume",
    )
    seed_writer.append_job_started(seed_job)
    seed_writer.append_effect_begin(
        session_id="s-reconcile-resume",
        trace_id="tr-seed-resume",
        effect_id="eff-reconcile-resume-1",
        request_id="req-reconcile-resume-1",
    )

    async def _kernel_execute(_session, _job):
        return ("ok", False)

    plane = ExecutionControlPlane(
        registry=registry,
        kernel_executor=_kernel_execute,
        ledger=shared_ledger,
    )

    report = plane.apply_reconciliation(
        session_id="s-reconcile-resume",
        request_id="req-reconcile-resume-1",
        effect_id="eff-reconcile-resume-1",
        reason="test_resume_mode",
        mode="resume_eligible",
    )
    assert report.get("mode") == "resume_eligible"
    assert bool(report.get("applied")) is True

    events = shared_ledger.read()
    job_reconciled = [event for event in events if str(event.get("type") or "") == "JOB_RECONCILED"]
    effect_reconciled = [event for event in events if str(event.get("type") or "") == "EFFECT_RECONCILED"]
    assert job_reconciled
    assert effect_reconciled
    job_payload = dict(job_reconciled[-1].get("payload") or {})
    effect_payload = dict(effect_reconciled[-1].get("payload") or {})
    assert job_payload.get("resolution") == "resume_eligible_without_terminal"
    assert effect_payload.get("resolution") == "resume_eligible_without_commit"
    assert job_payload.get("mode") == "resume_eligible"
    assert effect_payload.get("mode") == "resume_eligible"


def test_apply_reconciliation_rejects_conflicting_mode_for_same_effect_and_request():
    shared_ledger = InMemoryExecutionLedger()
    registry = SessionRegistry()

    seed_writer = LedgerWriter(shared_ledger)
    seed_job = ExecutionJob(
        session_id="s-reconcile-conflict",
        user_input="hello",
        metadata={"request_id": "req-reconcile-conflict-1", "trace_id": "tr-seed-conflict"},
        trace_id="tr-seed-conflict",
    )
    seed_writer.append_job_started(seed_job)
    seed_writer.append_effect_begin(
        session_id="s-reconcile-conflict",
        trace_id="tr-seed-conflict",
        effect_id="eff-reconcile-conflict-1",
        request_id="req-reconcile-conflict-1",
    )

    async def _kernel_execute(_session, _job):
        return ("ok", False)

    plane = ExecutionControlPlane(
        registry=registry,
        kernel_executor=_kernel_execute,
        ledger=shared_ledger,
    )

    first = plane.apply_reconciliation(
        session_id="s-reconcile-conflict",
        request_id="req-reconcile-conflict-1",
        effect_id="eff-reconcile-conflict-1",
        mode="close_only",
    )
    assert bool(first.get("applied")) is True

    with pytest.raises(RuntimeError, match="Conflicting reconciliation resolution"):
        plane.apply_reconciliation(
            session_id="s-reconcile-conflict",
            request_id="req-reconcile-conflict-1",
            effect_id="eff-reconcile-conflict-1",
            mode="resume_eligible",
        )


def test_scheduler_enforces_backpressure_limit():
    registry = SessionRegistry()
    shared_ledger = InMemoryExecutionLedger()
    scheduler = Scheduler(
        registry,
        reader=LedgerReader(shared_ledger),
        writer=LedgerWriter(shared_ledger),
        max_inflight_jobs=1,
    )

    async def _run():
        first = ExecutionJob(session_id="s-backpressure", user_input="first")
        second = ExecutionJob(session_id="s-backpressure", user_input="second")
        await scheduler.register(first)
        with pytest.raises(RuntimeError):
            await scheduler.register(second)

    asyncio.run(_run())


def test_scheduler_rejects_invalid_emission_transition_before_ledger_write():
    registry = SessionRegistry()
    shared_ledger = InMemoryExecutionLedger()
    scheduler = Scheduler(
        registry,
        reader=LedgerReader(shared_ledger),
        writer=LedgerWriter(shared_ledger),
    )
    job = ExecutionJob(session_id="s-transition", user_input="hello", trace_id="tr-transition")

    with pytest.raises(RuntimeError, match="cannot be emitted before Submitted"):
        scheduler._append_lifecycle_event(
            job,
            Completed(
                execution_id=job.job_id,
                occurred_at=datetime.now(),
                result_ref="job:result",
            ),
            step_key="test.invalid.transition",
        )


def test_scheduler_pending_order_is_deterministic_under_concurrency():
    registry = SessionRegistry()
    shared_ledger = InMemoryExecutionLedger()
    scheduler = Scheduler(
        registry,
        reader=LedgerReader(shared_ledger),
        writer=LedgerWriter(shared_ledger),
    )

    async def _run() -> None:
        job_b = ExecutionJob(session_id="s-order", user_input="second", trace_id="tr-b")
        job_a = ExecutionJob(session_id="s-order", user_input="first", trace_id="tr-a")
        await scheduler.register(job_b)
        await scheduler.register(job_a)
        scheduler._pending_job_ids = [job_b.job_id, job_a.job_id]
        scheduler._job_ready = lambda *_args, **_kwargs: True
        first = scheduler._pop_next_ready_job_id()
        assert first == sorted([job_b.job_id, job_a.job_id])[0]

    asyncio.run(_run())


def test_scheduler_claim_ordering_handles_clock_skew_deterministically():
    registry = SessionRegistry()
    shared_ledger = InMemoryExecutionLedger()
    scheduler = Scheduler(
        registry,
        reader=LedgerReader(shared_ledger),
        writer=LedgerWriter(shared_ledger),
    )

    async def _run() -> None:
        job_late = ExecutionJob(session_id="s-skew", user_input="late", trace_id="tr-late")
        job_early = ExecutionJob(session_id="s-skew", user_input="early", trace_id="tr-early")
        job_late.metadata["claim_order"] = {"timestamp": 200.0, "worker_id": "w-b", "lease_epoch": 0}
        job_early.metadata["claim_order"] = {"timestamp": 100.0, "worker_id": "w-a", "lease_epoch": 0}
        await scheduler.register(job_late)
        await scheduler.register(job_early)
        scheduler._pending_job_ids = [job_late.job_id, job_early.job_id]
        scheduler._job_ready = lambda *_args, **_kwargs: True
        first = scheduler._pop_next_ready_job_id()
        assert first == job_early.job_id

    asyncio.run(_run())


def test_scheduler_claim_tiebreak_double_claim_and_redelivery_collision():
    registry = SessionRegistry()
    shared_ledger = InMemoryExecutionLedger()
    scheduler = Scheduler(
        registry,
        reader=LedgerReader(shared_ledger),
        writer=LedgerWriter(shared_ledger),
    )

    async def _run() -> None:
        # Same claim timestamp; tie breaks by worker_id then lease_epoch.
        job_worker_b = ExecutionJob(session_id="s-race", user_input="b", trace_id="tr-b")
        job_worker_a = ExecutionJob(session_id="s-race", user_input="a", trace_id="tr-a")
        job_worker_b.metadata["claim_order"] = {"timestamp": 500.0, "worker_id": "worker-b", "lease_epoch": 1}
        job_worker_a.metadata["claim_order"] = {"timestamp": 500.0, "worker_id": "worker-a", "lease_epoch": 2}

        await scheduler.register(job_worker_b)
        await scheduler.register(job_worker_a)
        scheduler._pending_job_ids = [job_worker_b.job_id, job_worker_a.job_id]
        scheduler._job_ready = lambda *_args, **_kwargs: True
        first = scheduler._pop_next_ready_job_id()
        assert first == job_worker_a.job_id

        # Same timestamp + worker_id; lower lease_epoch should win first.
        job_epoch_2 = ExecutionJob(session_id="s-race", user_input="epoch2", trace_id="tr-e2")
        job_epoch_1 = ExecutionJob(session_id="s-race", user_input="epoch1", trace_id="tr-e1")
        job_epoch_2.metadata["claim_order"] = {"timestamp": 800.0, "worker_id": "worker-z", "lease_epoch": 2}
        job_epoch_1.metadata["claim_order"] = {"timestamp": 800.0, "worker_id": "worker-z", "lease_epoch": 1}

        await scheduler.register(job_epoch_2)
        await scheduler.register(job_epoch_1)
        scheduler._pending_job_ids = [job_epoch_2.job_id, job_epoch_1.job_id]
        second = scheduler._pop_next_ready_job_id()
        assert second == job_epoch_1.job_id

    asyncio.run(_run())


def test_failure_taxonomy_unknown_state_is_explicit_and_non_retryable():
    failure = _classify_execution_failure(RuntimeError("opaque failure that does not map to known classes"))
    assert failure.get("failure_type") == "unknown_state"
    assert failure.get("failure_class") == "runtime_exception"
    assert failure.get("failure_action") == "reconcile"
    assert bool(failure.get("auto_retry")) is False
    assert bool(failure.get("retryable")) is False


def test_failure_taxonomy_partial_commit_maps_to_reconcile_action():
    failure = _classify_execution_failure(RuntimeError("partial commit detected in external sink"))
    assert failure.get("failure_type") == "partial_commit"
    assert failure.get("failure_action") == "reconcile"
    assert bool(failure.get("auto_retry")) is False


def test_failure_taxonomy_poison_maps_to_quarantine_action():
    failure = _classify_execution_failure(RuntimeError("poison payload from upstream"))
    assert failure.get("failure_type") == "poison"
    assert failure.get("failure_action") == "quarantine"
    assert bool(failure.get("auto_retry")) is False


def test_ledger_writer_rejects_semantically_incomplete_execution_lifecycle_event():
    shared_ledger = InMemoryExecutionLedger()
    writer = LedgerWriter(shared_ledger)

    with pytest.raises(ValueError, match="semantic payload incomplete"):
        writer.write_event(
            event_type="EXECUTION_LIFECYCLE",
            session_id="s-semantic",
            trace_id="tr-semantic",
            kernel_step_id="test.semantic",
            payload={"type": "Claimed"},
        )


def test_scheduler_aging_fairness_promotes_starved_job_deterministically():
    registry = SessionRegistry()
    shared_ledger = InMemoryExecutionLedger()
    scheduler = Scheduler(
        registry,
        reader=LedgerReader(shared_ledger),
        writer=LedgerWriter(shared_ledger),
        fairness_aging_rate=200.0,
        tenant_balance_weight=0.0,
    )

    async def _run() -> None:
        old_low_priority = ExecutionJob(
            session_id="s-fairness",
            user_input="old",
            trace_id="tr-old",
            metadata={"priority": 300, "tenant_id": "tenant-a"},
        )
        new_high_priority = ExecutionJob(
            session_id="s-fairness",
            user_input="new",
            trace_id="tr-new",
            metadata={"priority": 1, "tenant_id": "tenant-a"},
        )
        await scheduler.register(old_low_priority)
        await scheduler.register(new_high_priority)

        # Simulate long wait for old job so age compensates for worse base priority.
        old_low_priority.metadata.setdefault("scheduling", {})["submitted_monotonic"] = 0.0
        new_high_priority.metadata.setdefault("scheduling", {})["submitted_monotonic"] = 999999999.0

        scheduler._pending_job_ids = [new_high_priority.job_id, old_low_priority.job_id]
        scheduler._job_ready = lambda *_args, **_kwargs: True
        first = scheduler._pop_next_ready_job_id()
        assert first == old_low_priority.job_id

    asyncio.run(_run())


def test_ledger_replay_rebuilds_projection_and_returns_pending_jobs():
    """Phase 3: recovery is ledger-only via direct replay."""
    ledger = InMemoryExecutionLedger()
    session_store = SessionStore(ledger=ledger, projection_only=True)

    ledger.write(
        {
            "type": "SESSION_STATE_UPDATED",
            "session_id": "s-recover",
            "trace_id": "tr",
            "timestamp": 1.0,
            "kernel_step_id": "session_store.apply",
            "payload": {"state": {"x": 1}, "version": 1},
        }
    )
    ledger.write(
        {
            "type": "JOB_QUEUED",
            "session_id": "s-recover",
            "trace_id": "tr",
            "timestamp": 2.0,
            "kernel_step_id": "control_plane.enqueue",
            "payload": {
                "job_id": "job-recover-1",
                "request_id": "req-recover-1",
                "user_input": "hello",
                "attachments": [],
                "metadata": {"trace_id": "tr"},
                "priority": 0,
                "submitted_at": 2.0,
            },
        }
    )

    # Phase 3: direct ledger replay
    events = ledger.read()
    session_store.rebuild_from_ledger(events)

    snap = session_store.snapshot()
    pending = list(session_store.pending_jobs())

    assert len(events) == 2
    assert len(dict(snap.get("sessions") or {})) == 1
    assert int(snap.get("version") or 0) >= 1
    assert len(pending) == 1
    assert pending[0]["job_id"] == "job-recover-1"

    state = session_store.get("s-recover")
    assert state == {"x": 1}


def test_replay_hash_ignores_job_submission_wall_clock_metadata():
    left = InMemoryExecutionLedger()
    right = InMemoryExecutionLedger()
    verifier = ReplayVerifier()

    left.write(
        {
            "type": "JOB_QUEUED",
            "session_id": "s-deterministic",
            "trace_id": "tr-deterministic",
            "timestamp": 2.0,
            "kernel_step_id": "control_plane.enqueue",
            "payload": {
                "job_id": "job-deterministic-1",
                "request_id": "req-deterministic-1",
                "user_input": "hello",
                "attachments": [],
                "metadata": {"trace_id": "tr-deterministic"},
                "priority": 0,
                "submitted_at": 2.0,
            },
        }
    )
    right.write(
        {
            "type": "JOB_QUEUED",
            "session_id": "s-deterministic",
            "trace_id": "tr-deterministic",
            "timestamp": 2.0,
            "kernel_step_id": "control_plane.enqueue",
            "payload": {
                "job_id": "job-deterministic-1",
                "request_id": "req-deterministic-1",
                "user_input": "hello",
                "attachments": [],
                "metadata": {"trace_id": "tr-deterministic"},
                "priority": 0,
                "submitted_at": 999.0,
            },
        }
    )

    assert left.replay_hash() == right.replay_hash()
    assert verifier.trace_hash(left.read()) == verifier.trace_hash(right.read())


# ---------------------------------------------------------------------------
# Canonicalization boundary — system-wide policy tests (Step 1–5)
# ---------------------------------------------------------------------------


def test_lease_time_not_in_replay_hash():
    """Lease temporal fields (acquired_at, expires_at) must not affect the
    canonical replay hash even if they accidentally appear in an event payload.

    ExecutionLease never writes to the ledger, but if any downstream code
    ever copies lease data into an event payload, the hash must remain stable.
    """
    from dadbot.core.canonical_event import canonicalize_event_payload, validate_trace

    # Simulate two otherwise-identical events whose payloads carry different
    # lease timestamps (the kind of leakage we are guarding against).
    event_base = {
        "type": "JOB_STARTED",
        "session_id": "s-lease",
        "trace_id": "tr-lease",
        "timestamp": 1.0,
        "kernel_step_id": "scheduler.execute.start",
        "sequence": 1,
        "session_index": 1,
        "event_id": "evt-1",
        "parent_event_id": "",
        "payload": {"job_id": "job-lease-1"},
    }
    event_with_lease_t1 = dict(
        event_base, payload={**event_base["payload"], "acquired_at": 1000.0, "expires_at": 1030.0}
    )
    event_with_lease_t2 = dict(
        event_base, payload={**event_base["payload"], "acquired_at": 2000.0, "expires_at": 2030.0}
    )

    canon1 = canonicalize_event_payload(event_with_lease_t1["payload"])
    canon2 = canonicalize_event_payload(event_with_lease_t2["payload"])
    assert canon1 == canon2, "acquired_at/expires_at must be stripped before hashing"

    # validate_trace must NOT raise for a clean event (no forbidden fields).
    clean_trace = [dict(event_base)]
    validate_trace(clean_trace)  # no AssertionError expected

    # validate_trace MUST raise for an event whose payload leaks a forbidden field.
    dirty_trace = [event_with_lease_t1]
    with pytest.raises(AssertionError, match="acquired_at|expires_at"):
        validate_trace(dirty_trace)


def test_health_checker_witness_payload_has_no_forbidden_fields():
    """SystemHealthChecker._append_health_witness() must produce a payload that
    passes validate_trace — i.e. it must not embed any wall-clock timestamps
    in the event payload dict.
    """
    from dadbot.core.canonical_event import validate_trace
    from dadbot.core.execution_ledger import ExecutionLedger
    from dadbot.core.system_health_checker import SystemHealthChecker

    ledger = ExecutionLedger()
    checker = SystemHealthChecker()
    checker._append_health_witness(ledger, component="test.component")

    events = ledger.read()
    assert events, "health witness write should have produced at least one event"
    # validate_trace inspects the payload dicts — raises AssertionError on any
    # forbidden field found.
    validate_trace(events)


def test_validate_trace_catches_all_forbidden_fields():
    """validate_trace must catch every field in NON_CANONICAL_PAYLOAD_FIELDS,
    not just submitted_at or acquired_at.
    """
    import pytest

    from dadbot.core.canonical_event import FORBIDDEN_TRACE_FIELDS, validate_trace


    base_event = {
        "type": "JOB_QUEUED",
        "session_id": "s-validate",
        "trace_id": "tr-validate",
        "timestamp": 1.0,
        "kernel_step_id": "test",
        "sequence": 1,
        "session_index": 1,
        "event_id": "evt-v1",
        "parent_event_id": "",
        "payload": {"job_id": "job-v1"},
    }
    for field in FORBIDDEN_TRACE_FIELDS:
        dirty = dict(base_event, payload={**base_event["payload"], field: 99999.0})
        with pytest.raises(AssertionError, match=field):
            validate_trace([dirty])


def test_validate_trace_rejects_request_and_correlation_fields() -> None:
    import pytest

    from dadbot.core.canonical_event import validate_trace

    event = {
        "type": "JOB_QUEUED",
        "session_id": "s-validate-full",
        "trace_id": "tr-validate-full",
        "timestamp": 1.0,
        "kernel_step_id": "test.full_policy",
        "sequence": 1,
        "session_index": 1,
        "event_id": "evt-v2",
        "parent_event_id": "",
        "payload": {
            "request_id": "req-1",
            "correlation_id": "corr-1",
        },
    }

    with pytest.raises(AssertionError, match="request_id|correlation_id"):
        validate_trace([event], enforce_full_policy=True)


def test_capability_audit_event_is_excluded_from_replay_hash():
    left = InMemoryExecutionLedger()
    right = InMemoryExecutionLedger()
    verifier = ReplayVerifier()

    baseline = {
        "type": "JOB_COMPLETED",
        "session_id": "s-audit-replay",
        "trace_id": "tr-audit-replay",
        "timestamp": 1.0,
        "kernel_step_id": "scheduler.execute.complete",
        "payload": {
            "result": "ok",
            "metadata": {"quality": "stable"},
        },
    }
    left.write(dict(baseline))
    right.write(dict(baseline))

    right.write(
        {
            "type": "CAPABILITY_AUDIT_EVENT",
            "session_id": "s-audit-replay",
            "trace_id": "tr-audit-replay",
            "timestamp": 2.0,
            "kernel_step_id": "save_node.capability_audit",
            "payload": {
                "audit_version": "v1",
                "scenario": "runtime_turn",
                "result": "ok",
                "metrics": {
                    "temporal_violation": False,
                    "mutation_leak": False,
                    "save_node_compliance": True,
                },
                "timestamp": None,
            },
        }
    )

    assert left.replay_hash() == right.replay_hash()
    assert verifier.trace_hash(left.read()) == verifier.trace_hash(right.read())


def test_policy_trace_event_is_excluded_from_replay_hash():
    left = InMemoryExecutionLedger()
    right = InMemoryExecutionLedger()
    verifier = ReplayVerifier()

    baseline = {
        "type": "JOB_COMPLETED",
        "session_id": "s-policy-replay",
        "trace_id": "tr-policy-replay",
        "timestamp": 1.0,
        "kernel_step_id": "scheduler.execute.complete",
        "payload": {
            "result": "ok",
            "metadata": {"quality": "stable"},
        },
    }
    left.write(dict(baseline))
    right.write(dict(baseline))

    right.write(
        {
            "type": "PolicyTraceEvent",
            "session_id": "s-policy-replay",
            "trace_id": "tr-policy-replay",
            "timestamp": 2.0,
            "kernel_step_id": "save_node.policy_trace",
            "payload": {
                "summary": {
                    "policy": "safety",
                    "decision_action": "handled",
                    "decision_step": "validate",
                },
                "policy_trace": {
                    "event_type": "policy_decision",
                    "trace": {
                        "final_action": {"action": "handled", "step_name": "validate"},
                    },
                },
            },
        }
    )

    assert left.replay_hash() == right.replay_hash()
    assert verifier.trace_hash(left.read()) == verifier.trace_hash(right.read())
