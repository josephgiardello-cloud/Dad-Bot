"""Local MCP server process management.

This module isolates MCP server control (start/stop, PID management, logging)
from DadBot core. No model or graph semantics—pure system I/O boundary.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


class LocalMcpServerController:
    """Manages local MCP server process lifecycle.
    
    Responsibilities:
    - Process start/stop with proper file handle management
    - PID file persistence
    - Log tail reading  
    - Status introspection
    
    No dependency on graph, execution, or model logic.
    """

    def __init__(self, runtime_root_path: Path):
        """Initialize controller.
        
        Args:
            runtime_root_path: Root directory for session logs and PID file
        """
        self.runtime_root_path = runtime_root_path

    def get_runtime_paths(self) -> dict[str, Path]:
        """Get runtime file paths for MCP server.
        
        Returns dict with keys: 'pid', 'stdout', 'stderr'
        """
        base_dir = self.runtime_root_path / "session_logs"
        base_dir.mkdir(parents=True, exist_ok=True)
        return {
            "pid": base_dir / "local_mcp_server.pid",
            "stdout": base_dir / "local_mcp_server.stdout.log",
            "stderr": base_dir / "local_mcp_server.stderr.log",
        }

    def read_pid(self) -> int | None:
        """Read stored PID from file.
        
        Returns:
            Process ID if readable, None otherwise
        """
        pid_path = self.get_runtime_paths()["pid"]
        if not pid_path.exists():
            return None
        try:
            return int(pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    @staticmethod
    def is_process_running(pid: int | None) -> bool:
        """Check if a process is running.
        
        Args:
            pid: Process ID to check
        
        Returns:
            True if process is running, False otherwise
        """
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def get_log_tail(self, *, lines: int = 20) -> dict[str, str]:
        """Get last N lines from stdout and stderr logs.
        
        Args:
            lines: Number of lines to return (default 20)
        
        Returns:
            Dict with 'stdout' and 'stderr' keys containing log text
        """
        def _tail(path: Path) -> str:
            try:
                content = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                return ""
            return "\n".join(content[-max(1, int(lines or 1)) :])

        paths = self.get_runtime_paths()
        return {
            "stdout": _tail(paths["stdout"]),
            "stderr": _tail(paths["stderr"]),
        }

    def start_process(self, *, restart: bool = False) -> dict[str, Any]:
        """Start MCP server process.
        
        Args:
            restart: If True, stop existing server before starting
        
        Returns:
            Status dict with process information
        """
        if restart:
            self.stop_process()

        # Check if already running
        pid = self.read_pid()
        if pid is not None and self.is_process_running(pid):
            return {"running": True, "pid": pid, "reused": True}

        paths = self.get_runtime_paths()
        
        # Platform-specific process creation flags
        creationflags = 0
        if os.name == "nt":
            creationflags = (
                getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
            )

        # Open log files and start process
        stdout_handle = paths["stdout"].open("ab")
        stderr_handle = paths["stderr"].open("ab")
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "dadbot_system.local_mcp_server"],
                cwd=str(self.runtime_root_path),
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                creationflags=creationflags,
                close_fds=True,
            )
        finally:
            stdout_handle.close()
            stderr_handle.close()

        # Persist PID
        paths["pid"].write_text(str(process.pid), encoding="utf-8")
        
        return {
            "running": True,
            "pid": process.pid,
            "stdout_log_path": str(paths["stdout"]),
            "stderr_log_path": str(paths["stderr"]),
        }

    def stop_process(self) -> None:
        """Stop MCP server process.
        
        Uses taskill on Windows, SIGTERM on Unix.
        Cleans up PID file.
        """
        paths = self.get_runtime_paths()
        pid = self.read_pid()
        
        if pid is not None and self.is_process_running(pid):
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            else:
                try:
                    os.kill(pid, 15)
                except OSError:
                    pass
        
        # Clean up PID file
        try:
            paths["pid"].unlink(missing_ok=True)
        except OSError:
            pass

    def get_status(self) -> dict[str, Any]:
        """Get current server status.
        
        Returns:
            Dict with keys: running, pid, stdout_log_path, stderr_log_path
        """
        paths = self.get_runtime_paths()
        pid = self.read_pid()
        running = self.is_process_running(pid)
        
        # Clean stale PID file if process died
        if pid is not None and not running:
            try:
                paths["pid"].unlink(missing_ok=True)
            except OSError:
                pass
        
        return {
            "running": running,
            "pid": pid if running else None,
            "stdout_log_path": str(paths["stdout"]),
            "stderr_log_path": str(paths["stderr"]),
        }
