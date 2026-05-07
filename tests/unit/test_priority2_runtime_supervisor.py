"""Priority 2: Runtime Lifecycle Supervisor tests.

Tests for:
1. Lock acquisition and release
2. Stale lock detection
3. Lifecycle state transitions
4. Preflight checks
5. Port conflict detection
"""

import json
import os
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from dadbot.runtime.supervisor import (
    RuntimeLock,
    RuntimeSupervisor,
    acquire_runtime_lock,
    release_runtime_lock,
    get_runtime_status,
)


@pytest.mark.unit
class TestRuntimeLock:
    """Test RuntimeLock data structure and serialization."""

    def test_runtime_lock_creation(self):
        """Verify RuntimeLock can be created with valid data."""
        lock = RuntimeLock(
            pid=12345,
            port=8501,
            timestamp=time.time(),
            state="RUNNING",
            command_hash="cmd-abc123",
            owner_id="streamlit-12345",
        )
        
        assert lock.pid == 12345
        assert lock.port == 8501
        assert lock.state == "RUNNING"
        assert lock.command_hash == "cmd-abc123"

    def test_runtime_lock_is_stale(self):
        """Verify stale detection works correctly."""
        # Recent lock
        recent = RuntimeLock(
            pid=123,
            port=8501,
            timestamp=time.time(),
            state="RUNNING",
            command_hash="cmd-abc",
            owner_id="test",
        )
        assert not recent.is_stale(timeout_seconds=60)
        
        # Old lock
        old = RuntimeLock(
            pid=123,
            port=8501,
            timestamp=time.time() - 120,  # 2 minutes old
            state="RUNNING",
            command_hash="cmd-abc",
            owner_id="test",
        )
        assert old.is_stale(timeout_seconds=60)

    def test_runtime_lock_serialization(self):
        """Verify lock can be serialized to dict and back."""
        original = RuntimeLock(
            pid=12345,
            port=8501,
            timestamp=time.time(),
            state="RUNNING",
            command_hash="cmd-abc123",
            owner_id="streamlit-12345",
        )
        
        data = original.to_dict()
        restored = RuntimeLock.from_dict(data)
        
        assert restored.pid == original.pid
        assert restored.port == original.port
        assert restored.state == original.state
        assert restored.command_hash == original.command_hash


@pytest.mark.unit
class TestRuntimeSupervisor:
    """Test RuntimeSupervisor lock management."""

    @pytest.fixture
    def temp_lock_file(self):
        """Create temporary lock file for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / "runtime.lock"
            yield lock_file

    @pytest.fixture
    def supervisor(self, temp_lock_file):
        """Create supervisor with temporary lock file."""
        return RuntimeSupervisor(lock_file=temp_lock_file)

    def test_acquire_lock_success(self, supervisor):
        """Verify successful lock acquisition."""
        success, msg = supervisor.acquire_lock(
            pid=os.getpid(),
            port=8501,
            owner_id="test-owner",
        )
        
        assert success is True
        assert supervisor.lock_file.exists()

    def test_acquire_lock_fails_on_existing_lock(self, supervisor):
        """Verify lock acquisition fails when lock exists."""
        # Acquire first lock
        supervisor.acquire_lock(pid=1111, port=8501, owner_id="first")
        
        # Try to acquire second lock
        success, msg = supervisor.acquire_lock(pid=2222, port=8502, owner_id="second")
        
        assert success is False
        assert "already locked" in msg.lower()

    def test_acquire_lock_recovers_from_stale(self, supervisor):
        """Verify stale locks are automatically recovered."""
        # Create stale lock
        stale_lock = RuntimeLock(
            pid=1111,
            port=8501,
            timestamp=time.time() - 120,  # 2 minutes old
            state="RUNNING",
            command_hash="cmd-old",
            owner_id="stale",
        )
        supervisor._write_lock(stale_lock)
        
        # Try to acquire new lock
        success, msg = supervisor.acquire_lock(pid=2222, port=8502, owner_id="new")
        
        assert success is True
        assert not supervisor.lock_file.exists() or supervisor._read_lock().pid == 2222

    def test_release_lock(self, supervisor):
        """Verify lock can be released."""
        supervisor.acquire_lock(pid=os.getpid(), port=8501, owner_id="test")
        assert supervisor.lock_file.exists()
        
        success = supervisor.release_lock()
        
        assert success is True
        assert not supervisor.lock_file.exists()

    def test_set_state_transitions(self, supervisor):
        """Verify state transitions work."""
        supervisor.acquire_lock(pid=os.getpid(), port=8501, owner_id="test")
        
        assert supervisor.set_state("RUNNING")
        lock = supervisor._read_lock()
        assert lock.state == "RUNNING"
        
        assert supervisor.set_state("DEGRADED")
        lock = supervisor._read_lock()
        assert lock.state == "DEGRADED"
        
        assert supervisor.set_state("SHUTDOWN")
        lock = supervisor._read_lock()
        assert lock.state == "SHUTDOWN"

    def test_get_status_no_lock(self, supervisor):
        """Verify status reports when no lock exists."""
        status = supervisor.get_status()
        
        assert status["status"] == "no_lock"
        assert status["pid"] is None
        assert status["port"] is None

    def test_get_status_with_lock(self, supervisor):
        """Verify status reports when lock exists."""
        supervisor.acquire_lock(pid=9999, port=8501, owner_id="test")
        
        status = supervisor.get_status()
        
        assert status["status"] == "locked"
        assert status["pid"] == 9999
        assert status["port"] == 8501
        assert status["state"] == "RUNNING"
        assert not status["stale"]

    def test_preflight_check_no_issues(self, supervisor):
        """Verify preflight passes with clean state."""
        with patch("dadbot.runtime.supervisor.RuntimeSupervisor._is_port_in_use") as mock_port:
            mock_port.return_value = False
            ok, issues = supervisor.preflight_check()
        
        assert ok is True
        assert len(issues) == 0

    def test_preflight_check_detects_active_lock(self, supervisor):
        """Verify preflight detects active locks."""
        supervisor.acquire_lock(pid=1111, port=8501, owner_id="test")
        
        ok, issues = supervisor.preflight_check()
        
        assert ok is False
        assert len(issues) > 0
        assert any("active lock" in issue.lower() or "conflict" in issue.lower() for issue in issues)

    def test_preflight_check_ignores_stale_locks(self, supervisor):
        """Verify preflight ignores stale locks."""
        stale_lock = RuntimeLock(
            pid=1111,
            port=8501,
            timestamp=time.time() - 120,
            state="RUNNING",
            command_hash="cmd-old",
            owner_id="stale",
        )
        supervisor._write_lock(stale_lock)
        
        ok, issues = supervisor.preflight_check()
        
        # Should report stale lock but allow recovery
        assert len(issues) > 0
        assert any("stale" in issue.lower() for issue in issues)

    def test_lock_file_creation(self, supervisor):
        """Verify lock file is created in correct location."""
        supervisor.acquire_lock(pid=os.getpid(), port=8501, owner_id="test")
        
        assert supervisor.lock_file.exists()
        assert supervisor.lock_file.parent.exists()

    def test_lock_file_contains_valid_json(self, supervisor):
        """Verify lock file contains valid JSON."""
        supervisor.acquire_lock(pid=os.getpid(), port=8501, owner_id="test")
        
        content = supervisor.lock_file.read_text()
        data = json.loads(content)
        
        assert isinstance(data, dict)
        assert "pid" in data
        assert "port" in data
        assert "state" in data
        assert data["state"] == "RUNNING"

    def test_read_lock_handles_corrupt_file(self, supervisor, caplog):
        """Verify read_lock handles corrupt JSON gracefully."""
        # Write corrupt data
        supervisor.lock_file.parent.mkdir(parents=True, exist_ok=True)
        supervisor.lock_file.write_text("{ invalid json }")
        
        lock = supervisor._read_lock()
        
        assert lock is None

    @patch("dadbot.runtime.supervisor.RuntimeSupervisor._is_port_in_use")
    def test_preflight_check_port_binding(self, mock_is_port_in_use, supervisor):
        """Verify preflight detects port binding conflicts."""
        supervisor.acquire_lock(pid=1111, port=8501, owner_id="test")
        mock_is_port_in_use.return_value = True
        
        ok, issues = supervisor.preflight_check()
        
        assert ok is False
        assert any("port" in issue.lower() for issue in issues)


@pytest.mark.unit
class TestRuntimeSupervisorConvenienceFunctions:
    """Test module-level convenience functions."""

    @pytest.fixture
    def temp_supervisor(self):
        """Patch supervisor to use temporary file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / "runtime.lock"
            supervisor = RuntimeSupervisor(lock_file=lock_file)
            
            # Patch the module-level instance
            import dadbot.runtime.supervisor as sup_module
            original = sup_module._supervisor_instance
            sup_module._supervisor_instance = supervisor
            
            yield supervisor
            
            sup_module._supervisor_instance = original

    def test_acquire_runtime_lock_convenience(self, temp_supervisor):
        """Verify convenience function works."""
        success, msg = acquire_runtime_lock(
            pid=os.getpid(),
            port=8501,
            owner_id="test",
        )
        
        assert success is True

    def test_release_runtime_lock_convenience(self, temp_supervisor):
        """Verify release convenience function works."""
        acquire_runtime_lock(pid=os.getpid(), port=8501, owner_id="test")
        
        success = release_runtime_lock()
        
        assert success is True

    def test_get_runtime_status_convenience(self, temp_supervisor):
        """Verify status convenience function works."""
        acquire_runtime_lock(pid=os.getpid(), port=8501, owner_id="test")
        
        status = get_runtime_status()
        
        assert status["status"] == "locked"
        assert status["port"] == 8501


@pytest.mark.integration
class TestRuntimeSupervisorIntegration:
    """Integration tests for complete lock lifecycle."""

    def test_full_lock_lifecycle(self):
        """Test complete acquire -> state change -> release cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / "runtime.lock"
            supervisor = RuntimeSupervisor(lock_file=lock_file)
            
            # Acquire lock
            success, msg = supervisor.acquire_lock(pid=9999, port=8501, owner_id="test")
            assert success
            assert supervisor.lock_file.exists()
            
            # Verify initial state is RUNNING
            status = supervisor.get_status()
            assert status["state"] == "RUNNING"
            
            # Transition to DEGRADED
            assert supervisor.set_state("DEGRADED")
            status = supervisor.get_status()
            assert status["state"] == "DEGRADED"
            
            # Release lock
            assert supervisor.release_lock()
            assert not supervisor.lock_file.exists()
            
            # Verify status is cleared
            status = supervisor.get_status()
            assert status["status"] == "no_lock"

    def test_multiple_lock_attempts_same_session(self):
        """Test that same owner cannot acquire lock twice."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / "runtime.lock"
            supervisor = RuntimeSupervisor(lock_file=lock_file)
            
            # First acquisition
            success1, _ = supervisor.acquire_lock(pid=1111, port=8501, owner_id="owner1")
            assert success1
            
            # Second acquisition by different owner
            success2, msg = supervisor.acquire_lock(pid=2222, port=8502, owner_id="owner2")
            assert not success2
            assert "already locked" in msg.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
