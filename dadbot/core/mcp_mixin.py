"""DadBotMcpMixin — MCP server management for the DadBot facade.

Extracted from the DadBot god-class. Owns all local-MCP-server lifecycle
methods: status, log tails, start/stop, PID tracking, and server construction.
All methods route through self._mcp_controller (LocalMcpServerController)
which is wired in DadBotBootMixin._initialize_services().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dadbot.runtime.mcp import LocalMcpServerController


class DadBotMcpMixin:
    """MCP server management for the DadBot facade."""

    def local_mcp_status(self) -> dict[str, Any]:
        """Get MCP server status.

        Routes through: LocalMcpServerController.get_status()
        """
        try:
            from dadbot_system.local_mcp_server import (
                local_mcp_status as describe_local_mcp,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "available": False,
                "configured": False,
                "server_name": "dadbot-local-services",
                "tool_count": 0,
                "local_state_entries": len(
                    dict(self.MEMORY_STORE.get("mcp_local_store") or {}),
                ),
                "error": str(exc).strip() or exc.__class__.__name__,
                "running": False,
            }

        payload = dict(describe_local_mcp(self) or {})
        controller_status = self._mcp_controller.get_status()
        payload.setdefault(
            "local_state_entries",
            len(dict(self.MEMORY_STORE.get("mcp_local_store") or {})),
        )
        payload.update(controller_status)
        payload.setdefault("task_label", "Run Dad Bot MCP Server")
        return payload

    def local_mcp_runtime_paths(self) -> dict[str, Path]:
        """Get MCP runtime file paths (pid, logs).

        Routes through: LocalMcpServerController.get_runtime_paths()
        """
        return self._mcp_controller.get_runtime_paths()

    @staticmethod
    def _local_mcp_process_running(pid: int | None) -> bool:
        """Check if MCP process is running.

        Routes through: LocalMcpServerController.is_process_running()
        """
        return LocalMcpServerController.is_process_running(pid)

    def _read_local_mcp_pid(self) -> int | None:
        """Read stored MCP process ID.

        Routes through: LocalMcpServerController.read_pid()
        """
        return self._mcp_controller.read_pid()

    def local_mcp_log_tail(self, *, lines: int = 20) -> dict[str, str]:
        """Get last N lines from MCP server logs.

        Routes through: LocalMcpServerController.get_log_tail()
        """
        return self._mcp_controller.get_log_tail(lines=lines)

    def start_local_mcp_server_process(
        self,
        *,
        restart: bool = False,
    ) -> dict[str, Any]:
        """Start MCP server process.

        Routes through: LocalMcpServerController.start_process()
        """
        self._mcp_controller.start_process(restart=restart)
        return self.local_mcp_status()

    def stop_local_mcp_server_process(self) -> dict[str, Any]:
        """Stop MCP server process.

        Routes through: LocalMcpServerController.stop_process()
        """
        self._mcp_controller.stop_process()
        return self.local_mcp_status()

    def build_local_mcp_server(self):
        from dadbot_system.local_mcp_server import build_server

        return build_server(bot=self)

    def run_local_mcp_server(self) -> None:
        from dadbot_system.local_mcp_server import main as run_local_mcp_main

        run_local_mcp_main(bot=self)
