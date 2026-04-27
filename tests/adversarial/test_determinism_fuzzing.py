from __future__ import annotations

import asyncio
import random
import time

from dadbot.core.replay_verifier import ReplayVerifier


def _base_events() -> list[dict]:
    now = time.time()
    return [
        {
            "type": "JOB_QUEUED",
            "session_id": "fuzz-session",
            "trace_id": "trace-a",
            "timestamp": now,
            "kernel_step_id": "control_plane.enqueue",
            "session_index": 1,
            "event_id": "evt-1",
            "parent_event_id": "",
            "payload": {
                "submitted_at": now,
                "job_id": "job-1",
                "trace_id": "trace-a",
                "correlation_id": "corr-a",
                "user_input": "hello",
            },
        },
        {
            "type": "JOB_STARTED",
            "session_id": "fuzz-session",
            "trace_id": "trace-a",
            "timestamp": now + 0.01,
            "kernel_step_id": "scheduler.execute.start",
            "session_index": 2,
            "event_id": "evt-2",
            "parent_event_id": "evt-1",
            "payload": {
                "occurred_at": now + 0.01,
                "trace_id": "trace-a",
                "correlation_id": "corr-a",
            },
        },
        {
            "type": "JOB_COMPLETED",
            "session_id": "fuzz-session",
            "trace_id": "trace-a",
            "timestamp": now + 0.02,
            "kernel_step_id": "scheduler.execute.complete",
            "session_index": 3,
            "event_id": "evt-3",
            "parent_event_id": "evt-2",
            "payload": {
                "updated_at": now + 0.02,
                "trace_id": "trace-a",
                "correlation_id": "corr-a",
                "result": {"reply": "ok"},
            },
        },
    ]


def _perturb_noncanonical(base: list[dict], *, temperature: float, latency_ms: float) -> list[dict]:
    out: list[dict] = []
    for event in base:
        item = dict(event)
        item["timestamp"] = float(item.get("timestamp") or 0.0) + random.uniform(0.0, 0.005)
        item["trace_id"] = f"trace-temp-{temperature:.3f}"
        payload = dict(item.get("payload") or {})
        # Inject stochastic metadata into fields excluded by canonicalization.
        payload["trace_id"] = f"trace-temp-{temperature:.3f}"
        payload["correlation_id"] = f"corr-lat-{latency_ms:.1f}"
        payload["submitted_at"] = time.time()
        payload["occurred_at"] = time.time()
        payload["updated_at"] = time.time()
        payload["created_at"] = time.time()
        payload["last_checked_at"] = time.time()
        item["payload"] = payload
        out.append(item)
    return out


async def _jitter() -> None:
    await asyncio.sleep(random.uniform(0.0, 0.002))


def test_temperature_perturbation_fuzz_hash_stability_100_runs() -> None:
    verifier = ReplayVerifier()
    baseline = _base_events()
    baseline_hash = verifier.trace_hash(baseline)

    for _ in range(100):
        temp = random.uniform(0.1, 1.1)
        latency = random.uniform(0.0, 250.0)
        perturbed = _perturb_noncanonical(baseline, temperature=temp, latency_ms=latency)
        assert verifier.trace_hash(perturbed) == baseline_hash


def test_async_jitter_injection_keeps_replay_equivalence() -> None:
    verifier = ReplayVerifier()
    baseline = _base_events()

    async def _run() -> list[dict]:
        events = baseline
        for _ in range(12):
            await _jitter()
        return _perturb_noncanonical(events, temperature=0.42, latency_ms=12.0)

    replay = asyncio.run(_run())
    result = verifier.verify_equivalence(baseline, replay)
    assert result["ok"] is True


def test_tool_latency_randomization_preserves_replay_invariants() -> None:
    verifier = ReplayVerifier()
    baseline = _base_events()
    baseline_state_hash = verifier.state_hash(baseline)

    for _ in range(100):
        perturbed = _perturb_noncanonical(
            baseline,
            temperature=random.uniform(0.2, 0.9),
            latency_ms=random.uniform(1.0, 500.0),
        )
        assert verifier.state_hash(perturbed) == baseline_state_hash
