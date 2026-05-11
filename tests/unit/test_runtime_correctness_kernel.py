from __future__ import annotations

import pytest

from dadbot.core.runtime_correctness_kernel import RuntimeCorrectnessKernel

pytestmark = pytest.mark.unit


def _minimal_snapshot() -> dict:
    return {
        "trace_id": "tr-1",
        "session_id": "s-1",
        "execution_dag_hash": "dag-1",
        "determinism_closure_hash": "det-1",
        "post_commit_mutation_effects_hash": "mut-1",
        "memory_retrieval_set": [],
        "commit_boundary_count": 1,
        "trace_events": [],
    }


def test_runtime_correctness_kernel_returns_stable_fingerprint() -> None:
    kernel = RuntimeCorrectnessKernel()
    snapshot = _minimal_snapshot()

    report1 = kernel.run({}, dict(snapshot))
    report2 = kernel.run({}, dict(snapshot))

    assert report1["fingerprint"] == report2["fingerprint"]
    assert bool(report1["determinism"]["replay_equivalent"]) is True
    assert bool(report1["canonical"]["schema_closed"]) is True


def test_runtime_correctness_kernel_raises_on_violation() -> None:
    kernel = RuntimeCorrectnessKernel()
    snapshot = _minimal_snapshot()
    snapshot["uuid_entropy"] = True

    with pytest.raises(RuntimeError, match="RuntimeCorrectnessViolation"):
        kernel.run({}, snapshot)
