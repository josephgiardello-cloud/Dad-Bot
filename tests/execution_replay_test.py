"""Tests for Phase 2 — Execution Replay Safety (execution_checkpoint.py)."""
from __future__ import annotations

import pytest
import time

from dadbot.core.execution_checkpoint import (
    ExecutionCheckpointKernel,
    ExecutionIdempotencyRegistry,
    DeterministicReplayValidator,
    ReplaySafeExecutionContext,
    FallbackChainEntry,
    NodeState,
    NodeExecutionSnapshot,
    CheckpointIntegrityError,
    DuplicateExecutionError,
    ReplayMismatchError,
)


# ---------------------------------------------------------------------------
# 2.1 Checkpoint Kernel
# ---------------------------------------------------------------------------


class TestCheckpointKernel:
    def setup_method(self):
        self.kernel = ExecutionCheckpointKernel()

    def test_update_node_creates_new_snapshot(self):
        snap = self.kernel.update_node("n1", NodeState.RUNNING, tool_name="memory_lookup")
        assert snap.node_id == "n1"
        assert snap.state == NodeState.RUNNING
        assert snap.tool_name == "memory_lookup"

    def test_update_node_transitions_state(self):
        self.kernel.update_node("n1", NodeState.RUNNING)
        snap = self.kernel.update_node("n1", NodeState.SUCCESS, output_type="list")
        assert snap.state == NodeState.SUCCESS
        assert snap.output_type == "list"

    def test_update_node_increments_attempt_on_retry(self):
        self.kernel.update_node("n1", NodeState.RUNNING, attempt_count=1)
        snap = self.kernel.update_node("n1", NodeState.RETRYING)
        assert snap.attempt_count == 2

    def test_update_node_explicit_attempt_count(self):
        snap = self.kernel.update_node("n1", NodeState.RUNNING, attempt_count=5)
        assert snap.attempt_count == 5

    def test_add_fallback_entry(self):
        entry = FallbackChainEntry(
            tool_name="fallback_tool",
            attempt=2,
            status="timeout",
            fallback_reason="primary timed out",
        )
        self.kernel.update_node("n1", NodeState.RUNNING)
        self.kernel.add_fallback_entry("n1", entry)
        snap = self.kernel.get_node("n1")
        assert len(snap.fallback_chain) == 1
        assert snap.fallback_chain[0].tool_name == "fallback_tool"

    def test_add_fallback_entry_creates_node_if_missing(self):
        entry = FallbackChainEntry(tool_name="t", attempt=1, status="error")
        self.kernel.add_fallback_entry("n_new", entry)
        snap = self.kernel.get_node("n_new")
        assert snap is not None
        assert snap.state == NodeState.FALLBACK

    def test_nodes_in_state_filter(self):
        self.kernel.update_node("n1", NodeState.SUCCESS)
        self.kernel.update_node("n2", NodeState.FAILED)
        self.kernel.update_node("n3", NodeState.SUCCESS)
        successes = self.kernel.nodes_in_state(NodeState.SUCCESS)
        assert len(successes) == 2

    def test_save_creates_hash_chained_checkpoint(self):
        self.kernel.update_node("n1", NodeState.SUCCESS)
        cp1 = self.kernel.save("cp1")
        cp2 = self.kernel.save("cp2")
        assert cp2.prev_checkpoint_hash == cp1.checkpoint_hash
        assert cp1.prev_checkpoint_hash == ""

    def test_checkpoint_hash_is_deterministic(self):
        self.kernel.update_node("n1", NodeState.SUCCESS, tool_name="t1")
        cp = self.kernel.save("test")
        # Re-verify using the assert method
        ExecutionCheckpointKernel.assert_checkpoint_hash(cp)  # must not raise

    def test_assert_checkpoint_integrity_passes_clean_chain(self):
        self.kernel.update_node("n1", NodeState.SUCCESS)
        self.kernel.save("cp1")
        self.kernel.save("cp2")
        self.kernel.assert_checkpoint_integrity()  # no exception

    def test_tampered_checkpoint_raises_integrity_error(self):
        self.kernel.update_node("n1", NodeState.SUCCESS)
        cp = self.kernel.save("cp1")
        # Tamper with the stored hash
        cp.checkpoint_hash = "deadbeef" * 8
        with pytest.raises(CheckpointIntegrityError):
            ExecutionCheckpointKernel.assert_checkpoint_hash(cp)

    def test_restore_from_checkpoint(self):
        self.kernel.update_node("n1", NodeState.SUCCESS)
        cp = self.kernel.save("before")
        # Mutate state after checkpoint
        self.kernel.update_node("n1", NodeState.FAILED, last_error="crash")
        # Restore
        self.kernel.restore(cp)
        snap = self.kernel.get_node("n1")
        assert snap.state == NodeState.SUCCESS

    def test_restore_rejects_tampered_checkpoint(self):
        self.kernel.update_node("n1", NodeState.SUCCESS)
        cp = self.kernel.save("cp")
        cp.checkpoint_hash = "bad_hash"
        with pytest.raises(CheckpointIntegrityError):
            self.kernel.restore(cp)

    def test_all_nodes_returns_all(self):
        for i in range(5):
            self.kernel.update_node(f"n{i}", NodeState.RUNNING)
        assert len(self.kernel.all_nodes()) == 5

    def test_snapshot_includes_fallback_chain_in_dict(self):
        self.kernel.update_node("n1", NodeState.FALLBACK)
        self.kernel.add_fallback_entry("n1", FallbackChainEntry("fb", 1, "error", fallback_reason="gone"))
        cp = self.kernel.save("cp")
        node_dict = cp.node_snapshots["n1"].to_dict()
        assert len(node_dict["fallback_chain"]) == 1
        assert node_dict["fallback_chain"][0]["fallback_reason"] == "gone"


# ---------------------------------------------------------------------------
# 2.2 Idempotency Registry
# ---------------------------------------------------------------------------


class TestIdempotencyRegistry:
    def setup_method(self):
        self.registry = ExecutionIdempotencyRegistry()

    def test_new_hash_is_not_duplicate(self):
        assert not self.registry.is_duplicate("abc123")

    def test_success_registration_marks_duplicate(self):
        self.registry.register_success("abc123", "memory_lookup", "ok", "list")
        assert self.registry.is_duplicate("abc123")

    def test_error_status_not_cached(self):
        self.registry.register_success("err_hash", "tool", "error", "null")
        assert not self.registry.is_duplicate("err_hash")

    def test_timeout_status_not_cached(self):
        self.registry.register_success("to_hash", "tool", "timeout", "null")
        assert not self.registry.is_duplicate("to_hash")

    def test_partial_status_is_cached(self):
        self.registry.register_success("partial_hash", "tool", "partial", "list")
        assert self.registry.is_duplicate("partial_hash")

    def test_cached_result_returns_entry(self):
        self.registry.register_success("h1", "tool_a", "ok", "dict", attempt_count=2)
        entry = self.registry.get_cached_result("h1")
        assert entry is not None
        assert entry.tool_name == "tool_a"
        assert entry.attempt_count == 2

    def test_get_cached_result_none_for_unknown(self):
        assert self.registry.get_cached_result("unknown") is None

    def test_evict_removes_entry(self):
        self.registry.register_success("h2", "t", "ok", "str")
        self.registry.evict("h2")
        assert not self.registry.is_duplicate("h2")

    def test_size_tracks_registrations(self):
        assert self.registry.size() == 0
        self.registry.register_success("h1", "t", "ok", "list")
        self.registry.register_success("h2", "t", "ok", "dict")
        assert self.registry.size() == 2

    def test_duplicate_registration_overwrites(self):
        self.registry.register_success("h1", "t", "ok", "list", attempt_count=1)
        self.registry.register_success("h1", "t", "ok", "dict", attempt_count=3)
        entry = self.registry.get_cached_result("h1")
        assert entry.attempt_count == 3


# ---------------------------------------------------------------------------
# 2.3 Deterministic Replay Validator
# ---------------------------------------------------------------------------


class TestReplayValidator:
    def setup_method(self):
        self.validator = DeterministicReplayValidator()

    def test_record_stores_fingerprint(self):
        rec = self.validator.record("h1", "ok", "list")
        assert rec.request_hash == "h1"
        assert self.validator.has_record("h1")

    def test_validate_same_as_recorded_passes(self):
        self.validator.record("h1", "ok", "list", fallback_chain_length=0, attempt_count=1)
        result = self.validator.validate("h1", "ok", "list")
        assert result.request_hash == "h1"

    def test_validate_status_mismatch_raises(self):
        self.validator.record("h1", "ok", "list")
        with pytest.raises(ReplayMismatchError, match="status mismatch"):
            self.validator.validate("h1", "error", "list")

    def test_validate_output_type_mismatch_raises(self):
        self.validator.record("h1", "ok", "list")
        with pytest.raises(ReplayMismatchError, match="output type mismatch"):
            self.validator.validate("h1", "ok", "dict")

    def test_validate_missing_record_raises(self):
        with pytest.raises(ReplayMismatchError, match="No recorded fingerprint"):
            self.validator.validate("unknown_hash", "ok", "list")

    def test_strict_mode_chain_length_mismatch_raises(self):
        strict = DeterministicReplayValidator(strict_mode=True)
        strict.record("h1", "ok", "list", fallback_chain_length=0, attempt_count=1)
        with pytest.raises(ReplayMismatchError, match="fallback chain length"):
            strict.validate("h1", "ok", "list", fallback_chain_length=1, attempt_count=1)

    def test_strict_mode_attempt_count_mismatch_raises(self):
        strict = DeterministicReplayValidator(strict_mode=True)
        strict.record("h1", "ok", "list", fallback_chain_length=0, attempt_count=1)
        with pytest.raises(ReplayMismatchError, match="attempt count"):
            strict.validate("h1", "ok", "list", fallback_chain_length=0, attempt_count=2)

    def test_non_strict_ignores_attempt_count_difference(self):
        self.validator.record("h1", "ok", "list", attempt_count=1)
        # Different attempt count should be fine in non-strict mode
        result = self.validator.validate("h1", "ok", "list", attempt_count=3)
        assert result.request_hash == "h1"

    def test_record_count(self):
        assert self.validator.record_count() == 0
        self.validator.record("h1", "ok", "list")
        self.validator.record("h2", "error", "null")
        assert self.validator.record_count() == 2


# ---------------------------------------------------------------------------
# ReplaySafeExecutionContext (facade)
# ---------------------------------------------------------------------------


class TestReplaySafeContext:
    def setup_method(self):
        self.ctx = ReplaySafeExecutionContext()

    def test_initial_state_clean(self):
        assert not self.ctx.is_duplicate("any_hash")

    def test_record_execution_marks_all_layers(self):
        self.ctx.record_execution(
            node_id="n1",
            request_hash="h1",
            tool_name="memory_lookup",
            status="ok",
            output_type="list",
            attempt_count=1,
        )
        assert self.ctx.is_duplicate("h1")
        assert self.ctx.replay.has_record("h1")
        snap = self.ctx.kernel.get_node("n1")
        assert snap is not None
        assert snap.state == NodeState.SUCCESS

    def test_failed_execution_not_idempotency_cached(self):
        self.ctx.record_execution(
            node_id="n1",
            request_hash="h_err",
            tool_name="tool_a",
            status="error",
            output_type="null",
        )
        assert not self.ctx.is_duplicate("h_err")

    def test_checkpoint_taken_after_execution(self):
        self.ctx.record_execution("n1", "h1", "t", "ok", "list")
        cp = self.ctx.checkpoint("after_wave_1")
        assert cp is not None
        assert cp.label == "after_wave_1"

    def test_integrity_passes_after_checkpoint(self):
        self.ctx.record_execution("n1", "h1", "t", "ok", "list")
        self.ctx.checkpoint("cp1")
        self.ctx.assert_integrity()  # no exception

    def test_fallback_chain_recorded_in_kernel(self):
        chain = [
            FallbackChainEntry("primary", 1, "timeout", fallback_reason="primary timed out"),
            FallbackChainEntry("secondary", 1, "ok"),
        ]
        self.ctx.record_execution(
            "n1", "h1", "secondary", "ok", "list", fallback_chain=chain
        )
        snap = self.ctx.kernel.get_node("n1")
        assert len(snap.fallback_chain) == 2
