from __future__ import annotations

import asyncio
import random
import time

import pytest

from dadbot.core.determinism import DeterminismBoundary, DeterminismMode, DeterminismViolation
from dadbot.core.replay_verifier import ReplayVerifier


def _event(
    *,
    idx: int,
    event_type: str,
    kernel_step_id: str,
    payload: dict,
    parent_event_id: str = "",
) -> dict:
    return {
        "type": event_type,
        "session_id": "stress-session",
        "trace_id": "stress-trace",
        "timestamp": time.time() + (idx * 0.01),
        "kernel_step_id": kernel_step_id,
        "session_index": idx,
        "event_id": f"evt-{idx}",
        "parent_event_id": parent_event_id,
        "payload": dict(payload or {}),
    }


def _baseline_events() -> list[dict]:
    return [
        _event(
            idx=1,
            event_type="JOB_QUEUED",
            kernel_step_id="control_plane.enqueue",
            payload={"job_id": "job-1", "user_input": "determinism check"},
        ),
        _event(
            idx=2,
            event_type="JOB_STARTED",
            kernel_step_id="scheduler.execute.start",
            parent_event_id="evt-1",
            payload={"job_id": "job-1"},
        ),
        _event(
            idx=3,
            event_type="TOOL_CALLED",
            kernel_step_id="graph.inference.tool_call",
            parent_event_id="evt-2",
            payload={"tool": "memory_lookup", "status": "ok"},
        ),
        _event(
            idx=4,
            event_type="JOB_COMPLETED",
            kernel_step_id="scheduler.execute.complete",
            parent_event_id="evt-3",
            payload={"job_id": "job-1", "result": {"reply": "ok"}},
        ),
    ]


def _with_noncanonical_variance(events: list[dict]) -> list[dict]:
    varied = []
    for item in events:
        event = dict(item)
        event["timestamp"] = float(event.get("timestamp") or 0.0) + random.uniform(0.0, 0.002)
        event["trace_id"] = f"stress-{random.randint(100, 999)}"
        payload = dict(event.get("payload") or {})
        payload["updated_at"] = time.time()
        payload["correlation_id"] = f"corr-{random.randint(1000, 9999)}"
        event["payload"] = payload
        varied.append(event)
    return varied


@pytest.mark.stress
def test_randomized_execution_timing_keeps_trace_equivalent() -> None:
    verifier = ReplayVerifier()
    baseline = _baseline_events()

    for _ in range(100):
        replay = _with_noncanonical_variance(baseline)
        result = verifier.verify_equivalence(baseline, replay)
        assert result["ok"] is True


@pytest.mark.stress
def test_forced_tool_failures_preserve_replay_equivalence() -> None:
    verifier = ReplayVerifier()
    baseline = _baseline_events()
    failure_events = list(baseline)
    failure_events[2] = _event(
        idx=3,
        event_type="TOOL_CALLED",
        kernel_step_id="graph.inference.tool_call",
        parent_event_id="evt-2",
        payload={"tool": "memory_lookup", "status": "failed", "error": "forced_failure"},
    )
    failure_events[3] = _event(
        idx=4,
        event_type="JOB_COMPLETED",
        kernel_step_id="scheduler.execute.complete",
        parent_event_id="evt-3",
        payload={"job_id": "job-1", "result": {"reply": "fallback"}},
    )

    replay = _with_noncanonical_variance(failure_events)
    result = verifier.verify_equivalence(failure_events, replay)
    assert result["ok"] is True


@pytest.mark.stress
def test_mid_execution_interruption_resume_keeps_equivalent_result() -> None:
    verifier = ReplayVerifier()
    baseline = _baseline_events()

    interrupted_prefix = baseline[:2]
    resumed_suffix = baseline[2:]
    resumed_run = interrupted_prefix + resumed_suffix

    result = verifier.verify_equivalence(baseline, resumed_run)
    assert result["ok"] is True


@pytest.mark.stress
def test_replay_equivalence_under_mixed_adversarial_conditions() -> None:
    verifier = ReplayVerifier()
    baseline = _baseline_events()

    mixed = _with_noncanonical_variance(_with_noncanonical_variance(baseline))
    result = verifier.verify_equivalence(baseline, mixed)
    assert result["ok"] is True


# ---------------------------------------------------------------------------
# DeterminismBoundary adversarial tests
# ---------------------------------------------------------------------------


@pytest.mark.stress
def test_boundary_record_then_replay_returns_sealed_value() -> None:
    """RECORD mode captures a value; REPLAY mode returns the same without calling fn."""
    boundary = DeterminismBoundary()
    call_count = [0]

    def _fn() -> str:
        call_count[0] += 1
        return f"result-{call_count[0]}"

    first = boundary.capture("slot.a", _fn)
    assert first == "result-1"
    assert call_count[0] == 1

    boundary.seal()  # switch to REPLAY mode

    replayed = boundary.capture("slot.a", _fn)
    assert replayed == "result-1"
    assert call_count[0] == 1  # fn was never called again


@pytest.mark.stress
def test_boundary_replay_raises_on_missing_slot() -> None:
    """REPLAY mode raises DeterminismViolation for any un-sealed slot."""
    boundary = DeterminismBoundary()
    boundary.seal()

    with pytest.raises(DeterminismViolation) as exc_info:
        boundary.capture("slot.unseen", lambda: "should-not-run")

    assert "no sealed value" in str(exc_info.value).lower()
    assert len(boundary.violations) == 1
    assert boundary.violations[0]["slot"] == "slot.unseen"


@pytest.mark.stress
def test_boundary_record_is_idempotent_within_turn() -> None:
    """Calling capture() for the same slot twice in RECORD mode returns the first value."""
    boundary = DeterminismBoundary()
    call_count = [0]

    def _fn() -> int:
        call_count[0] += 1
        return call_count[0]

    first = boundary.capture("slot.x", _fn)
    second = boundary.capture("slot.x", _fn)
    assert first == 1
    assert second == 1
    assert call_count[0] == 1


@pytest.mark.stress
def test_boundary_inject_and_replay_returns_injected_value() -> None:
    """Manually injected slots can be replayed without ever recording."""
    boundary = DeterminismBoundary(mode=DeterminismMode.REPLAY)
    boundary.inject("slot.injected", {"reply": "injected-response"})

    result = boundary.capture("slot.injected", lambda: {"reply": "should-not-run"})
    assert result == {"reply": "injected-response"}


@pytest.mark.stress
def test_boundary_open_mode_never_records() -> None:
    """OPEN mode passes through every call without recording."""
    boundary = DeterminismBoundary(mode=DeterminismMode.OPEN)
    call_count = [0]

    def _fn() -> int:
        call_count[0] += 1
        return call_count[0]

    r1 = boundary.capture("slot.open", _fn)
    r2 = boundary.capture("slot.open", _fn)
    assert r1 == 1
    assert r2 == 2  # called twice because OPEN never caches
    assert boundary.sealed_values == {}


@pytest.mark.stress
def test_boundary_snapshot_reflects_sealed_slots() -> None:
    """snapshot() returns all recorded slots and mode."""
    boundary = DeterminismBoundary()
    boundary.capture("slot.a", lambda: "alpha")
    boundary.capture("slot.b", lambda: "beta")
    snap = boundary.snapshot()
    assert snap["mode"] == "RECORD"
    assert set(snap["slots"]) == {"slot.a", "slot.b"}


@pytest.mark.stress
def test_boundary_record_mode_switch_cycle() -> None:
    """Boundary can cycle: RECORD → REPLAY → RECORD without corruption."""
    boundary = DeterminismBoundary()
    boundary.capture("slot.y", lambda: "v1")
    boundary.seal()

    replayed = boundary.capture("slot.y", lambda: "v2")
    assert replayed == "v1"

    boundary.record()
    # In RECORD mode the already-recorded slot is returned (idempotent)
    still_v1 = boundary.capture("slot.y", lambda: "v3")
    assert still_v1 == "v1"


@pytest.mark.stress
def test_boundary_async_capture_round_trip() -> None:
    """capture_async seals on first call and returns sealed value on second."""
    boundary = DeterminismBoundary()
    call_count = [0]

    async def _async_fn() -> str:
        call_count[0] += 1
        return f"async-{call_count[0]}"

    async def _run() -> None:
        first = await boundary.capture_async("slot.async", _async_fn)
        assert first == "async-1"

        boundary.seal()
        second = await boundary.capture_async("slot.async", _async_fn)
        assert second == "async-1"
        assert call_count[0] == 1

    asyncio.run(_run())


@pytest.mark.stress
def test_randomized_slot_order_does_not_affect_replay_correctness() -> None:
    """Slots captured in random order still replay correctly."""
    slots = [f"slot.{i}" for i in range(50)]
    random.shuffle(slots)

    boundary = DeterminismBoundary()
    values = {}
    for slot in slots:
        val = random.randint(0, 10_000)
        values[slot] = val
        boundary.capture(slot, lambda v=val: v)

    boundary.seal()

    for slot in slots:
        replayed = boundary.capture(slot, lambda: -1)
        assert replayed == values[slot]


@pytest.mark.stress
def test_high_frequency_replay_under_load() -> None:
    """1 000 rapid RECORD→REPLAY cycles produce zero violations."""
    violations = 0
    for _ in range(1_000):
        boundary = DeterminismBoundary()
        boundary.capture("slot.load", lambda: {"ts": time.monotonic()})
        boundary.seal()
        try:
            boundary.capture("slot.load", lambda: None)
        except DeterminismViolation:
            violations += 1
    assert violations == 0
