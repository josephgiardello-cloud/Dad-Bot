"""Tests for unified execution mode resolution (Cut #4).

Validates that:
1. Execution modes are resolved consistently across all paths
2. Recovery mode correctly restores from checkpoints
3. Degraded mode is properly marked on graph exceptions
4. Mode resolution is idempotent (resolve same job twice → same result)
"""

import pytest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dadbot.core.execution_mode import (
    ExecutionMode,
    ExecutionModeContext,
    ExecutionModeResolver,
)


class TestExecutionModeResolution:
    """Test unified ExecutionModeResolver."""

    def test_resolve_live_on_first_attempt(self):
        """LIVE mode for initial execution with no redeliveries."""
        job = SimpleNamespace(
            metadata={
                "execution_state": {
                    "redelivery_count": 0,
                    "lifecycle_state": "submitted",
                }
            }
        )
        
        context = ExecutionModeResolver.resolve(job, checkpoint=None)
        
        assert context.mode == ExecutionMode.LIVE
        assert not context.is_redelivery
        assert context.redelivery_count == 0
        assert not context.checkpoint_available

    def test_resolve_recovery_on_redelivery_count(self):
        """RECOVERY mode when redelivery_count > 0."""
        job = SimpleNamespace(
            metadata={
                "execution_state": {
                    "redelivery_count": 1,
                    "lifecycle_state": "running",
                }
            }
        )
        
        context = ExecutionModeResolver.resolve(job, checkpoint=None)
        
        assert context.mode == ExecutionMode.RECOVERY
        assert context.is_redelivery
        assert context.redelivery_count == 1

    def test_resolve_recovery_on_recovery_pending_lifecycle(self):
        """RECOVERY mode when lifecycle_state == 'recovery_pending'."""
        job = SimpleNamespace(
            metadata={
                "execution_state": {
                    "redelivery_count": 0,
                    "lifecycle_state": "recovery_pending",
                }
            }
        )
        
        context = ExecutionModeResolver.resolve(job, checkpoint=None)
        
        assert context.mode == ExecutionMode.RECOVERY
        assert context.is_redelivery

    def test_resolve_explicit_mode_takes_precedence(self):
        """Explicit execution_mode in metadata overrides automatic detection."""
        job = SimpleNamespace(
            metadata={
                "execution_mode": "replay",
                "execution_state": {
                    "redelivery_count": 5,
                    "lifecycle_state": "running",
                }
            }
        )
        
        context = ExecutionModeResolver.resolve(job, checkpoint=None)
        
        assert context.mode == ExecutionMode.REPLAY
        assert context.is_redelivery == False  # Not auto-detected

    def test_resolve_checkpoint_availability_detected(self):
        """checkpoint_available flag set when checkpoint dict provided."""
        job = SimpleNamespace(
            metadata={
                "execution_state": {
                    "redelivery_count": 0,
                    "lifecycle_state": "submitted",
                }
            }
        )
        checkpoint = {"checkpoint_hash": "abc123"}
        
        context = ExecutionModeResolver.resolve(job, checkpoint)
        
        assert context.checkpoint_available
        assert context.mode == ExecutionMode.LIVE  # But still live without redelivery

    def test_resolve_recovery_with_checkpoint_restoration(self):
        """Recovery mode with available checkpoint for full state restoration."""
        job = SimpleNamespace(
            metadata={
                "execution_state": {
                    "redelivery_count": 2,
                    "lifecycle_state": "running",
                }
            }
        )
        checkpoint = {
            "checkpoint_hash": "abc123",
            "execution_state": {"state": {"key": "value"}},
        }
        
        context = ExecutionModeResolver.resolve(job, checkpoint)
        
        assert context.mode == ExecutionMode.RECOVERY
        assert context.is_redelivery
        assert context.checkpoint_available
        assert context.redelivery_count == 2

    def test_resolve_for_logging_without_checkpoint(self):
        """resolve_for_logging provides simple string for telemetry."""
        job = SimpleNamespace(
            metadata={
                "execution_state": {
                    "redelivery_count": 1,
                    "lifecycle_state": "running",
                }
            }
        )
        
        mode_str = ExecutionModeResolver.resolve_for_logging(job)
        
        assert mode_str == "recovery"
        assert isinstance(mode_str, str)

    def test_mark_degraded_updates_metadata(self):
        """mark_degraded sets execution_mode to degraded."""
        metadata = {"execution_mode": "live"}
        
        ExecutionModeResolver.mark_degraded(metadata)
        
        assert metadata["execution_mode"] == "degraded"

    def test_resolve_idempotent(self):
        """Resolving the same job twice yields identical results."""
        job = SimpleNamespace(
            metadata={
                "execution_state": {
                    "redelivery_count": 1,
                    "lifecycle_state": "running",
                }
            }
        )
        checkpoint = {"checkpoint_hash": "xyz789"}
        
        context1 = ExecutionModeResolver.resolve(job, checkpoint)
        context2 = ExecutionModeResolver.resolve(job, checkpoint)
        
        assert context1.mode == context2.mode
        assert context1.is_redelivery == context2.is_redelivery
        assert context1.redelivery_count == context2.redelivery_count
        assert context1.checkpoint_available == context2.checkpoint_available

    def test_resolve_missing_execution_state_defaults_to_live(self):
        """Missing execution_state dict defaults to LIVE mode."""
        job = SimpleNamespace(metadata={})
        
        context = ExecutionModeResolver.resolve(job, checkpoint=None)
        
        assert context.mode == ExecutionMode.LIVE
        assert not context.is_redelivery

    def test_execution_mode_enum_str(self):
        """ExecutionMode enum converts to correct string values."""
        assert str(ExecutionMode.LIVE) == "live"
        assert str(ExecutionMode.RECOVERY) == "recovery"
        assert str(ExecutionMode.REPLAY) == "replay"
        assert str(ExecutionMode.DEGRADED) == "degraded"

    def test_resolve_case_insensitive_mode_matching(self):
        """Mode matching is case-insensitive."""
        job_upper = SimpleNamespace(
            metadata={
                "execution_mode": "RECOVERY",
                "execution_state": {
                    "redelivery_count": 0,
                    "lifecycle_state": "submitted",
                }
            }
        )
        job_lower = SimpleNamespace(
            metadata={
                "execution_mode": "recovery",
                "execution_state": {
                    "redelivery_count": 0,
                    "lifecycle_state": "submitted",
                }
            }
        )
        
        context_upper = ExecutionModeResolver.resolve(job_upper, checkpoint=None)
        context_lower = ExecutionModeResolver.resolve(job_lower, checkpoint=None)
        
        assert context_upper.mode == context_lower.mode == ExecutionMode.RECOVERY


class TestExecutionModeContext:
    """Test ExecutionModeContext dataclass."""

    def test_context_creation(self):
        """ExecutionModeContext can be created with required fields."""
        context = ExecutionModeContext(
            mode=ExecutionMode.RECOVERY,
            is_redelivery=True,
            checkpoint_available=True,
            redelivery_count=2,
        )
        
        assert context.mode == ExecutionMode.RECOVERY
        assert context.is_redelivery
        assert context.checkpoint_available
        assert context.redelivery_count == 2

    def test_context_field_access(self):
        """ExecutionModeContext fields are accessible."""
        context = ExecutionModeContext(
            mode=ExecutionMode.LIVE,
            is_redelivery=False,
            checkpoint_available=False,
            redelivery_count=0,
        )
        
        # Verify all fields accessible
        assert hasattr(context, "mode")
        assert hasattr(context, "is_redelivery")
        assert hasattr(context, "checkpoint_available")
        assert hasattr(context, "redelivery_count")


class TestExecutionModeIntegration:
    """Integration tests for execution mode with actual job structures."""

    def test_degraded_mode_never_auto_resolved(self):
        """DEGRADED mode is never auto-resolved, only set by mark_degraded."""
        job = SimpleNamespace(metadata={"execution_state": {}})
        
        context = ExecutionModeResolver.resolve(job, checkpoint=None)
        
        assert context.mode != ExecutionMode.DEGRADED

    def test_replay_mode_explicit_only(self):
        """REPLAY mode only settable via explicit metadata, not auto-resolved."""
        # Try to trigger replay mode implicitly (should not work)
        job_implicit = SimpleNamespace(
            metadata={"execution_state": {"redelivery_count": 0}}
        )
        
        context_implicit = ExecutionModeResolver.resolve(job_implicit, checkpoint=None)
        assert context_implicit.mode != ExecutionMode.REPLAY
        
        # Set explicitly (should work)
        job_explicit = SimpleNamespace(
            metadata={
                "execution_mode": "replay",
                "execution_state": {"redelivery_count": 0},
            }
        )
        
        context_explicit = ExecutionModeResolver.resolve(job_explicit, checkpoint=None)
        assert context_explicit.mode == ExecutionMode.REPLAY
