import asyncio

import pytest

from dadbot.core.control_plane import (
    ExecutionControlPlane,
    ExecutionJob,
    InMemoryExecutionLedger,
    LedgerReader,
    Scheduler,
    SessionRegistry,
)
from dadbot.core.ledger_writer import LedgerWriter
from dadbot.core.replay_verifier import ReplayVerifier
from dadbot.core.session_store import SessionStore


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
