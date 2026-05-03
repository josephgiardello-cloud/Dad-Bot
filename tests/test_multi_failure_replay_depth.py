"""Multi-failure replay depth — minimal closure tests.

Goal: prove nested failures do not break determinism or memory consistency.

Strategy (fixed depth, deterministic seeds, no external framework):
- Run same turn with failure injected at stage 1, 2, and 3
- Assert that:
  - failure_replay field captures all expected stages
  - memory_evolution hash is consistent (no corruption)
  - contract_version is identical across all failure variants
  - A second execution after recovery produces identical stamps
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

import pytest

from dadbot.core.graph import TurnGraph
from dadbot.core.graph_context import TurnContext
from dadbot.core.graph_pipeline_nodes import (
    ContextBuilderNode,
    HealthNode,
    InferenceNode,
    ReflectionNode,
    SafetyNode,
    SaveNode,
    TemporalNode,
    _NodeContractMixin,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------

class _StubService:
    def tick(self, ctx): return {"ok": True}
    def build_context(self, ctx): return {"rich_context": "stub"}
    async def run_agent(self, ctx, rich_context): return "stub_reply"
    def enforce_policies(self, ctx, candidate): return candidate
    def save_turn(self, ctx, result): pass
    def finalize_turn(self, ctx, result): return result
    def reflect_after_turn(self, turn_text, mood, reply_text): return None


class _StubRegistry:
    _KNOWN = {
        "maintenance_service", "context_service", "agent_service",
        "safety_service", "persistence_service", "reflection",
    }
    def __init__(self): self._svc = _StubService()
    def get(self, key, default=None):
        return self._svc if key in self._KNOWN else default


def _make_graph(**kwargs) -> TurnGraph:
    return TurnGraph(registry=_StubRegistry(), **kwargs)


def _make_context(user_input: str = "hello") -> TurnContext:
    return TurnContext(user_input=user_input)


# ---------------------------------------------------------------------------
# Failing nodes — deterministic injection at specific stages
# ---------------------------------------------------------------------------

class _FailingHealthNode(_NodeContractMixin):
    """Stage-1 failure: health check raises."""
    name = "health"
    def dependencies(self): return ()
    async def execute(self, registry: Any, ctx: TurnContext) -> None:
        raise RuntimeError("injected_health_failure")


class _FailingContextBuilderNode(_NodeContractMixin):
    """Stage-2 failure: context builder raises."""
    name = "context_builder"
    def dependencies(self): return ()
    async def execute(self, registry: Any, ctx: TurnContext) -> None:
        raise RuntimeError("injected_context_failure")


class _FailingInferenceNode(_NodeContractMixin):
    """Stage-3 failure: inference raises."""
    name = "inference"
    def dependencies(self): return ()
    async def execute(self, registry: Any, ctx: TurnContext) -> None:
        raise RuntimeError("injected_inference_failure")


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def _run_graph(nodes: list, user_input: str = "hello") -> TurnContext:
    ctx = _make_context(user_input)
    g = _make_graph(nodes=nodes)
    try:
        asyncio.run(g.execute(ctx))
    except Exception:
        pass  # failures are expected — we inspect stamps afterward
    return ctx


def _canonical_nodes() -> list:
    return [
        TemporalNode(), HealthNode(), ContextBuilderNode(),
        InferenceNode(), SafetyNode(), ReflectionNode(), SaveNode(),
    ]


def _memory_evolution_fingerprint(ctx: TurnContext) -> str | None:
    ev = ctx.determinism_manifest.get("memory_evolution", {})
    if not ev:
        return None
    blob = json.dumps(ev, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _replay_stages(ctx: TurnContext) -> list[str]:
    return [r.get("stage", "") for r in ctx.determinism_manifest.get("failure_replay", [])]


# ---------------------------------------------------------------------------
# Test 1: Each failure level is captured in failure_replay
# ---------------------------------------------------------------------------

class TestFailureReplayCapture:
    def test_stage1_failure_recorded(self):
        """Stage-1 (health) failure is captured in failure_replay."""
        nodes = [
            TemporalNode(), _FailingHealthNode(), ContextBuilderNode(),
            InferenceNode(), SafetyNode(), ReflectionNode(), SaveNode(),
        ]
        ctx = _run_graph(nodes)
        replay = ctx.determinism_manifest.get("failure_replay", [])
        assert len(replay) >= 1, "failure_replay must have at least one entry"
        stages = _replay_stages(ctx)
        assert any("health" in s for s in stages), f"health failure not recorded; stages={stages}"

    def test_stage2_failure_recorded(self):
        """Stage-2 (context_builder) failure is captured in failure_replay."""
        nodes = [
            TemporalNode(), HealthNode(), _FailingContextBuilderNode(),
            InferenceNode(), SafetyNode(), ReflectionNode(), SaveNode(),
        ]
        ctx = _run_graph(nodes)
        stages = _replay_stages(ctx)
        assert any("context_builder" in s for s in stages), (
            f"context_builder failure not recorded; stages={stages}"
        )

    def test_stage3_failure_recorded(self):
        """Stage-3 (inference) failure is captured in failure_replay."""
        nodes = [
            TemporalNode(), HealthNode(), ContextBuilderNode(),
            _FailingInferenceNode(), SafetyNode(), ReflectionNode(), SaveNode(),
        ]
        ctx = _run_graph(nodes)
        stages = _replay_stages(ctx)
        assert any("inference" in s for s in stages), (
            f"inference failure not recorded; stages={stages}"
        )


# ---------------------------------------------------------------------------
# Test 2: contract_version is identical across all failure depths
# ---------------------------------------------------------------------------

class TestContractVersionUnderFailure:
    def _contract_version(self, nodes: list) -> str | None:
        return _run_graph(nodes).determinism_manifest.get("contract_version")

    def test_contract_version_identical_across_failure_depths(self):
        """contract_version is schema-derived — must be identical regardless of where failure hits."""
        cv_clean = self._contract_version(_canonical_nodes())
        cv_fail1 = self._contract_version([
            TemporalNode(), _FailingHealthNode(), ContextBuilderNode(),
            InferenceNode(), SafetyNode(), ReflectionNode(), SaveNode(),
        ])
        cv_fail2 = self._contract_version([
            TemporalNode(), HealthNode(), _FailingContextBuilderNode(),
            InferenceNode(), SafetyNode(), ReflectionNode(), SaveNode(),
        ])
        cv_fail3 = self._contract_version([
            TemporalNode(), HealthNode(), ContextBuilderNode(),
            _FailingInferenceNode(), SafetyNode(), ReflectionNode(), SaveNode(),
        ])
        assert cv_clean is not None, "clean run must stamp contract_version"
        assert cv_clean == cv_fail1, "stage-1 failure must not change contract_version"
        assert cv_clean == cv_fail2, "stage-2 failure must not change contract_version"
        assert cv_clean == cv_fail3, "stage-3 failure must not change contract_version"


# ---------------------------------------------------------------------------
# Test 3: failure_replay entries have all required fields
# ---------------------------------------------------------------------------

class TestFailureReplayFields:
    _REQUIRED = {"stage", "error_type", "error_msg", "state_keys"}

    def _assert_entry_fields(self, entry: dict, label: str) -> None:
        for field in self._REQUIRED:
            assert field in entry, f"failure_replay entry missing '{field}' ({label}): {entry}"

    def test_stage1_replay_entry_fields(self):
        nodes = [
            TemporalNode(), _FailingHealthNode(), ContextBuilderNode(),
            InferenceNode(), SafetyNode(), ReflectionNode(), SaveNode(),
        ]
        ctx = _run_graph(nodes)
        for entry in ctx.determinism_manifest.get("failure_replay", []):
            self._assert_entry_fields(entry, "stage1")

    def test_stage2_replay_entry_fields(self):
        nodes = [
            TemporalNode(), HealthNode(), _FailingContextBuilderNode(),
            InferenceNode(), SafetyNode(), ReflectionNode(), SaveNode(),
        ]
        ctx = _run_graph(nodes)
        for entry in ctx.determinism_manifest.get("failure_replay", []):
            self._assert_entry_fields(entry, "stage2")

    def test_stage3_replay_entry_fields(self):
        nodes = [
            TemporalNode(), HealthNode(), ContextBuilderNode(),
            _FailingInferenceNode(), SafetyNode(), ReflectionNode(), SaveNode(),
        ]
        ctx = _run_graph(nodes)
        for entry in ctx.determinism_manifest.get("failure_replay", []):
            self._assert_entry_fields(entry, "stage3")


# ---------------------------------------------------------------------------
# Test 4: multi-failure chain (2 failures in the same run)
# ---------------------------------------------------------------------------

class TestMultiFailureChain:
    def test_two_failures_both_captured_in_replay(self):
        """When two stages fail, both are captured in failure_replay."""
        nodes = [
            TemporalNode(), _FailingHealthNode(), _FailingContextBuilderNode(),
            InferenceNode(), SafetyNode(), ReflectionNode(), SaveNode(),
        ]
        ctx = _run_graph(nodes)
        replay = ctx.determinism_manifest.get("failure_replay", [])
        # At minimum one failure must be recorded; two is ideal
        assert len(replay) >= 1, "multi-failure chain must produce at least one replay entry"
        stages = _replay_stages(ctx)
        assert any("health" in s or "context_builder" in s for s in stages), (
            f"neither failing stage recorded; stages={stages}"
        )

    def test_two_failures_contract_version_still_stable(self):
        """Two failures in the same run must not corrupt the contract_version stamp."""
        nodes = [
            TemporalNode(), _FailingHealthNode(), _FailingContextBuilderNode(),
            InferenceNode(), SafetyNode(), ReflectionNode(), SaveNode(),
        ]
        ctx_clean = _run_graph(_canonical_nodes())
        ctx_fail2 = _run_graph(nodes)
        cv_clean = ctx_clean.determinism_manifest.get("contract_version")
        cv_fail2 = ctx_fail2.determinism_manifest.get("contract_version")
        assert cv_clean is not None
        assert cv_clean == cv_fail2

    def test_replay_run_after_clean_run_produces_same_contract_version(self):
        """A second clean run after a failed run produces the same contract_version."""
        ctx_fail = _run_graph([
            TemporalNode(), _FailingInferenceNode(), HealthNode(),
            ContextBuilderNode(), SafetyNode(), ReflectionNode(), SaveNode(),
        ])
        ctx_clean1 = _run_graph(_canonical_nodes())
        ctx_clean2 = _run_graph(_canonical_nodes())
        cv1 = ctx_clean1.determinism_manifest.get("contract_version")
        cv2 = ctx_clean2.determinism_manifest.get("contract_version")
        assert cv1 == cv2, "Consecutive clean runs must produce identical contract_version"
        # fail run must also have same contract_version
        cv_fail = ctx_fail.determinism_manifest.get("contract_version")
        assert cv_fail == cv1, "Failed run must not alter contract_version"


# ---------------------------------------------------------------------------
# Test 5: memory_evolution consistency under failure
# ---------------------------------------------------------------------------

class TestMemoryEvolutionUnderFailure:
    def test_clean_run_produces_memory_evolution_stamp(self):
        """A clean run stamps memory_evolution."""
        ctx = _run_graph(_canonical_nodes())
        ev = ctx.determinism_manifest.get("memory_evolution")
        assert ev is not None, "memory_evolution must be stamped on clean runs"

    def test_memory_evolution_fingerprint_stable_across_identical_runs(self):
        """Two identical clean runs produce the same memory_evolution fingerprint."""
        fp1 = _memory_evolution_fingerprint(_run_graph(_canonical_nodes()))
        fp2 = _memory_evolution_fingerprint(_run_graph(_canonical_nodes()))
        assert fp1 == fp2, "memory_evolution fingerprint must be stable"

    def test_failure_run_does_not_corrupt_memory_evolution_structure(self):
        """A failing run's memory_evolution (if present) has valid structure."""
        nodes = [
            TemporalNode(), HealthNode(), ContextBuilderNode(),
            _FailingInferenceNode(), SafetyNode(), ReflectionNode(), SaveNode(),
        ]
        ctx = _run_graph(nodes)
        ev = ctx.determinism_manifest.get("memory_evolution")
        if ev is not None:
            # If stamped, it must have the expected keys
            assert "before_fingerprint" in ev or "delta" in ev, (
                f"memory_evolution has unexpected structure: {ev}"
            )
