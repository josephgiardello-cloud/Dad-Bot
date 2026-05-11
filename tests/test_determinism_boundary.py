"""Tests for DeterminismBoundary enforcement guarantees."""

import pytest

from dadbot.core.determinism import DeterminismBoundary, DeterminismMode, DeterminismViolation

pytestmark = pytest.mark.integration


def test_record_mode_seals_value_and_returns_it():
    boundary = DeterminismBoundary()
    calls = []

    def llm():
        calls.append(1)
        return "sealed reply"

    result = boundary.capture("inference.llm_reply", llm)
    assert result == "sealed reply"
    assert len(calls) == 1


def test_record_mode_is_idempotent_within_turn():
    boundary = DeterminismBoundary()
    calls = []

    def llm():
        calls.append(1)
        return "sealed reply"

    boundary.capture("inference.llm_reply", llm)
    result2 = boundary.capture("inference.llm_reply", llm)
    assert result2 == "sealed reply"
    # Callable must only be invoked once even when capture() is called twice.
    assert len(calls) == 1


def test_replay_mode_returns_sealed_value_without_calling_fn():
    boundary = DeterminismBoundary()
    calls = []

    def llm():
        calls.append(1)
        return "original reply"

    boundary.capture("inference.llm_reply", llm)
    boundary.seal()

    result = boundary.capture("inference.llm_reply", lambda: "NEVER CALLED")
    assert result == "original reply"
    # The second lambda must never be called.
    assert len(calls) == 1


def test_replay_mode_raises_violation_for_unknown_slot():
    boundary = DeterminismBoundary()
    boundary.seal()

    with pytest.raises(DeterminismViolation) as exc_info:
        boundary.capture("inference.llm_reply", lambda: "nope")

    assert exc_info.value.slot == "inference.llm_reply"
    assert exc_info.value.mode == DeterminismMode.REPLAY
    assert not boundary.is_consistent()


def test_open_mode_bypasses_enforcement():
    boundary = DeterminismBoundary(mode=DeterminismMode.OPEN)
    results = []

    def llm():
        results.append("called")
        return "open reply"

    result = boundary.capture("inference.llm_reply", llm)
    assert result == "open reply"
    assert len(results) == 1
    # Nothing sealed in OPEN mode.
    assert "inference.llm_reply" not in boundary.sealed_values


def test_inject_preloads_slot_for_replay():
    boundary = DeterminismBoundary()
    boundary.inject("inference.llm_reply", "injected reply")
    boundary.seal()

    result = boundary.capture("inference.llm_reply", lambda: "NEVER CALLED")
    assert result == "injected reply"


def test_snapshot_contains_expected_fields():
    boundary = DeterminismBoundary()
    boundary.capture("slot1", lambda: "v1")
    snap = boundary.snapshot()

    assert snap["mode"] == "RECORD"
    assert "slot1" in snap["slots"]
    assert "slot1" in snap["hashes"]
    assert snap["consistent"] is True


def test_capture_async_record_and_replay():
    import asyncio


    boundary = DeterminismBoundary()
    calls = []

    async def async_llm():
        calls.append(1)
        return "async reply"

    result = asyncio.run(boundary.capture_async("inference.llm_reply", async_llm))
    assert result == "async reply"
    assert len(calls) == 1

    boundary.seal()
    result2 = asyncio.run(boundary.capture_async("inference.llm_reply", async_llm))
    assert result2 == "async reply"
    assert len(calls) == 1  # never called again


def test_determinism_boundary_record_mode_detects_structural_drift_when_enabled():
    boundary = DeterminismBoundary(strict_structural_drift=True)

    boundary.inject("slot.strict", {"result": "stable"})

    with pytest.raises(DeterminismViolation, match="structural|drift"):
        boundary.capture("slot.strict", lambda: {"result": "changed"})


def test_replay_completeness_rehydrates_sealed_slots_after_restart():
    calls = []

    def _producer():
        calls.append("called")
        return {"reply": "stable"}

    record_boundary = DeterminismBoundary()
    record_boundary.capture("slot.reply", _producer)
    snapshot = record_boundary.snapshot()

    replay_boundary = DeterminismBoundary.from_snapshot(snapshot)
    replay_boundary.seal()
    result = replay_boundary.capture("slot.reply", lambda: {"reply": "should-not-run"})

    assert result == {"reply": "stable"}
    assert calls == ["called"]
