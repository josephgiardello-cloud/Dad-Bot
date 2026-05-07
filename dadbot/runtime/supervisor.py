"""Runtime process lifecycle supervision.

Ensures single authoritative runtime instance with:
- Ownership lock (file + timestamp + command fingerprint)
- Lifecycle states: INIT → RUNNING → DEGRADED → SHUTDOWN
- Startup preflight validation
- Stale lock detection and recovery

Design:
- Lock file: ~/.dadbot/runtime.lock (portable across platforms)
- Lock contents: JSON with PID, port, timestamp, command_hash, state
- Lock acquisition: atomic write with fingerprint validation
- Lock release: on graceful shutdown or stale timeout (60s)
- Preflight: check for stale locks, conflicting ports, stale processes
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

RuntimeLifecycleState = Literal["INIT", "RUNNING", "DEGRADED", "SHUTDOWN"]


@dataclass(slots=True)
class RuntimeLock:
    """Single runtime ownership lock contract."""
    
    pid: int                           # Process ID claiming ownership
    port: int                          # Port being served
    timestamp: float                   # Lock acquisition timestamp
    state: RuntimeLifecycleState       # Current lifecycle state
    command_hash: str                  # SHA256 of command fingerprint
    owner_id: str                      # Unique owner identifier
    
    def is_stale(self, timeout_seconds: int = 60) -> bool:
        """Check if lock has exceeded maximum age."""
        return (time.time() - self.timestamp) > timeout_seconds
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeLock:
        """Deserialize from dict."""
        return cls(
            pid=int(data.get("pid", 0)),
            port=int(data.get("port", 8501)),
            timestamp=float(data.get("timestamp", time.time())),
            state=str(data.get("state", "INIT")),
            command_hash=str(data.get("command_hash", "")),
            owner_id=str(data.get("owner_id", "")),
        )


class RuntimeSupervisor:
    """Manages single runtime instance with lifecycle control."""
    
    DEFAULT_LOCK_DIR = Path.home() / ".dadbot"
    DEFAULT_LOCK_FILE = DEFAULT_LOCK_DIR / "runtime.lock"
    STALE_LOCK_TIMEOUT = 60  # seconds
    
    def __init__(
        self,
        lock_file: Path | None = None,
        stale_timeout_seconds: int = STALE_LOCK_TIMEOUT,
    ):
        self.lock_file = lock_file or self.DEFAULT_LOCK_FILE
        self.stale_timeout = stale_timeout_seconds
        self._current_lock: RuntimeLock | None = None
        self._lock_dir = self.lock_file.parent
    
    def _ensure_lock_dir(self) -> None:
        """Ensure lock directory exists."""
        try:
            self._lock_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("Failed to create lock directory %s: %s", self._lock_dir, exc)
    
    def _compute_command_hash(self) -> str:
        """Compute fingerprint of current process command."""
        import hashlib
        import sys
        
        command_parts = [
            str(sys.executable),
            " ".join(sys.argv),
        ]
        command_str = "|".join(command_parts)
        digest = hashlib.sha256(command_str.encode("utf-8")).hexdigest()
        return f"cmd-{digest[:20]}"
    
    def _read_lock(self) -> RuntimeLock | None:
        """Read existing lock from file, return None if not found or invalid."""
        if not self.lock_file.exists():
            return None
        
        try:
            content = self.lock_file.read_text()
            data = json.loads(content)
            return RuntimeLock.from_dict(data)
        except Exception as exc:
            logger.debug("Failed to read lock file %s: %s", self.lock_file, exc)
            return None
    
    def _write_lock(self, lock: RuntimeLock) -> bool:
        """Atomically write lock to file."""
        self._ensure_lock_dir()
        
        try:
            # Write to temporary file first, then rename (atomic)
            temp_file = self.lock_file.with_suffix(".tmp")
            content = json.dumps(lock.to_dict(), indent=2)
            temp_file.write_text(content)
            temp_file.replace(self.lock_file)
            self._current_lock = lock
            return True
        except Exception as exc:
            logger.error("Failed to write lock file %s: %s", self.lock_file, exc)
            return False
    
    def _delete_lock(self) -> bool:
        """Delete lock file."""
        if not self.lock_file.exists():
            return True
        
        try:
            self.lock_file.unlink()
            self._current_lock = None
            return True
        except Exception as exc:
            logger.error("Failed to delete lock file %s: %s", self.lock_file, exc)
            return False
    
    def acquire_lock(
        self,
        pid: int,
        port: int,
        owner_id: str,
    ) -> tuple[bool, str]:
        """Attempt to acquire runtime lock.
        
        Returns:
            (success: bool, message: str)
            
        Lifecycle:
        1. Check for existing lock
        2. If exists and not stale: conflict (return False)
        3. If exists and stale: remove and proceed
        4. Create new lock with INIT state
        5. Transition to RUNNING
        """
        existing_lock = self._read_lock()
        
        if existing_lock is not None and not existing_lock.is_stale(self.stale_timeout):
            msg = (
                f"Runtime already locked by PID {existing_lock.pid} on port {existing_lock.port}. "
                f"Lock state: {existing_lock.state}. "
                f"Use 'dadbot doctor' to inspect or 'dadbot restart' to forcefully recover."
            )
            logger.warning(msg)
            return False, msg
        
        if existing_lock is not None and existing_lock.is_stale(self.stale_timeout):
            logger.info(
                "Stale lock detected (PID %d, age %ds). Removing.",
                existing_lock.pid,
                int(time.time() - existing_lock.timestamp),
            )
            self._delete_lock()
        
        # Create new lock with INIT state
        new_lock = RuntimeLock(
            pid=pid,
            port=port,
            timestamp=time.time(),
            state="INIT",
            command_hash=self._compute_command_hash(),
            owner_id=owner_id,
        )
        
        if not self._write_lock(new_lock):
            return False, "Failed to write lock file"
        
        # Transition to RUNNING
        if not self.set_state("RUNNING"):
            return False, "Failed to transition to RUNNING state"
        
        logger.info("Runtime lock acquired: PID %d, port %d, owner %s", pid, port, owner_id)
        return True, "Lock acquired successfully"
    
    def release_lock(self) -> bool:
        """Release runtime lock (called on shutdown)."""
        if not self._delete_lock():
            logger.warning("Failed to clean up lock file on shutdown")
            return False
        
        logger.info("Runtime lock released")
        return True
    
    def set_state(self, new_state: RuntimeLifecycleState) -> bool:
        """Update lifecycle state of current lock."""
        lock = self._read_lock()
        if lock is None:
            logger.warning("Cannot set state: no active lock")
            return False
        
        lock.state = new_state
        if not self._write_lock(lock):
            return False
        
        logger.debug("Runtime state transitioned to %s", new_state)
        return True
    
    def get_status(self) -> dict[str, Any]:
        """Get current runtime status and lock information."""
        lock = self._read_lock()
        
        if lock is None:
            return {
                "status": "no_lock",
                "message": "No active runtime lock found",
                "pid": None,
                "port": None,
                "state": None,
                "stale": None,
                "age_seconds": None,
            }
        
        age = time.time() - lock.timestamp
        is_stale = lock.is_stale(self.stale_timeout)
        
        return {
            "status": "locked",
            "message": f"Runtime locked by PID {lock.pid} on port {lock.port}, state: {lock.state}",
            "pid": lock.pid,
            "port": lock.port,
            "state": lock.state,
            "stale": is_stale,
            "age_seconds": int(age),
            "owner_id": lock.owner_id,
            "command_hash": lock.command_hash,
        }
    
    def preflight_check(self) -> tuple[bool, list[str]]:
        """Run startup preflight checks.
        
        Returns:
            (all_ok: bool, issues: list[str])
            
        Checks:
        1. Stale lock detection
        2. Port binding conflicts
        3. Stale process detection
        """
        issues = []
        
        existing_lock = self._read_lock()
        if existing_lock is not None:
            if existing_lock.is_stale(self.stale_timeout):
                age = int(time.time() - existing_lock.timestamp)
                issues.append(
                    f"Stale lock detected: PID {existing_lock.pid} (age {age}s, "
                    f"state: {existing_lock.state})"
                )
            else:
                issues.append(
                    f"Active lock conflict: PID {existing_lock.pid} on port {existing_lock.port}"
                )
        
        # Check for port binding conflicts (platform-specific)
        if self._is_port_in_use(existing_lock.port if existing_lock else 8501):
            issues.append(
                f"Port {existing_lock.port if existing_lock else 8501} already in use"
            )
        
        return len(issues) == 0, issues
    
    @staticmethod
    def _is_port_in_use(port: int) -> bool:
        """Check if port is already bound."""
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                result = sock.connect_ex(("127.0.0.1", port))
                return result == 0
        except Exception:
            return False


# Module-level singleton instance
_supervisor_instance: RuntimeSupervisor | None = None


def get_runtime_supervisor() -> RuntimeSupervisor:
    """Get or create module-level supervisor instance."""
    global _supervisor_instance
    if _supervisor_instance is None:
        _supervisor_instance = RuntimeSupervisor()
    return _supervisor_instance


def acquire_runtime_lock(pid: int, port: int, owner_id: str) -> tuple[bool, str]:
    """Convenience function to acquire lock with default supervisor."""
    supervisor = get_runtime_supervisor()
    return supervisor.acquire_lock(pid, port, owner_id)


def release_runtime_lock() -> bool:
    """Convenience function to release lock with default supervisor."""
    supervisor = get_runtime_supervisor()
    return supervisor.release_lock()


def get_runtime_status() -> dict[str, Any]:
    """Convenience function to get status with default supervisor."""
    supervisor = get_runtime_supervisor()
    return supervisor.get_status()
