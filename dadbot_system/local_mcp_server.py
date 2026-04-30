from __future__ import annotations

import atexit
import json
import os
import pathlib
from typing import Any

from Dad import DadBot

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - optional dependency
    FastMCP = None


_BOT: DadBot | None = None


def mcp_available() -> bool:
    return FastMCP is not None


def _get_bot(bot: DadBot | None = None) -> DadBot:
    global _BOT
    if bot is not None:
        return bot
    if _BOT is None:
        _BOT = DadBot()
    return _BOT


def _shutdown_bot() -> None:
    global _BOT
    if _BOT is None:
        return
    try:
        _BOT.shutdown()
    finally:
        _BOT = None


atexit.register(_shutdown_bot)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


def local_mcp_status(bot: DadBot | None = None) -> dict[str, Any]:
    runtime = _get_bot(bot) if bot is not None else None
    local_store = dict(getattr(runtime, "MEMORY_STORE", {}).get("mcp_local_store") or {}) if runtime is not None else {}
    return {
        "available": mcp_available(),
        "configured": mcp_available(),
        "server_name": "dadbot-local-services",
        "tool_count": 13,
        "local_state_entries": len(local_store),
    }


def build_server(bot: DadBot | None = None):
    if FastMCP is None:  # pragma: no cover - import guard only
        raise RuntimeError("Local MCP support requires the optional 'mcp' package. Install dad-bot[mcp].")

    server = FastMCP("dadbot-local-services")

    @server.tool()
    def runtime_health_snapshot() -> dict[str, Any]:
        """Return DadBot's current runtime health snapshot."""
        return _json_safe(_get_bot(bot).current_runtime_health_snapshot(force=True, log_warnings=False, persist=True))

    @server.tool()
    def maintenance_snapshot() -> dict[str, Any]:
        """Return maintenance cadence state, including daily memory compaction."""
        return _json_safe(_get_bot(bot).maintenance_snapshot())

    @server.tool()
    def workshop_snapshot() -> dict[str, Any]:
        """Return DadBot's workshop/dashboard snapshot."""
        return _json_safe(_get_bot(bot).dashboard_status_snapshot())

    @server.tool()
    def memory_snapshot() -> dict[str, Any]:
        """Return compressed memory surfaces including narrative memories."""
        runtime = _get_bot(bot)
        return _json_safe(
            {
                "narrative_memories": runtime.narrative_memories(),
                "consolidated_memories": runtime.consolidated_memories()[:12],
                "memory_graph_summary": runtime.build_graph_summary_context(limit=4),
                "last_memory_compaction_summary": runtime.MEMORY_STORE.get("last_memory_compaction_summary", ""),
            }
        )

    @server.tool()
    def memory_search(query: str, limit: int = 5) -> dict[str, Any]:
        """Search semantic and graph-backed memory context."""
        runtime = _get_bot(bot)
        return _json_safe(runtime.retrieve_context(str(query or ""), strategy="hybrid", limit=max(1, int(limit or 1))))

    @server.tool()
    def add_reminder(title: str, due_text: str = "") -> dict[str, Any] | None:
        """Create or update a local reminder."""
        result = _get_bot(bot).agentic_handler.add_reminder(title, due_text)
        return _json_safe(result) if result is not None else None

    @server.tool()
    def list_reminders(limit: int = 8) -> list[dict[str, Any]]:
        """List open reminders from local DadBot storage."""
        return _json_safe(_get_bot(bot).reminder_catalog(include_done=False)[: max(1, int(limit or 1))])

    @server.tool()
    def add_calendar_event(title: str, due_text: str = "") -> dict[str, Any] | None:
        """Create a local calendar event stored by DadBot."""
        result = _get_bot(bot).agentic_handler.add_calendar_event(title, due_text)
        return _json_safe(result) if result is not None else None

    @server.tool()
    def list_calendar_events(limit: int = 8) -> list[dict[str, Any]]:
        """List local calendar events stored by DadBot."""
        return _json_safe(_get_bot(bot).agentic_handler.list_calendar_events(limit=limit))

    @server.tool()
    def draft_email(recipient: str, subject: str = "", body: str = "") -> dict[str, Any] | None:
        """Create a local email draft on disk and return its metadata."""
        result = _json_safe(_get_bot(bot).agentic_handler.draft_email(recipient, subject=subject, body=body))
        return result if result is not None else None

    @server.tool()
    def read_local_state(key: str) -> dict[str, Any]:
        """Read a small JSON-compatible record from DadBot's local MCP store."""
        store = dict(_get_bot(bot).MEMORY_STORE.get("mcp_local_store") or {})
        return {"key": key, "value": _json_safe(store.get(str(key).strip()))}

    @server.tool()
    def write_local_state(key: str, value: str) -> dict[str, Any]:
        """Write a JSON-compatible record into DadBot's local MCP store."""
        runtime_bot = _get_bot(bot)
        normalized_key = str(key or "").strip()
        if not normalized_key:
            raise ValueError("key must not be empty")
        try:
            parsed_value: Any = json.loads(value)
        except Exception:
            parsed_value = value
        store = dict(runtime_bot.MEMORY_STORE.get("mcp_local_store") or {})
        store[normalized_key] = parsed_value
        runtime_bot.mutate_memory_store(mcp_local_store=store)
        return {"key": normalized_key, "stored": True, "value": _json_safe(parsed_value)}

    @server.tool()
    def heritage_cross_link_query(context: str, max_links: int = 3) -> list[dict[str, Any]]:
        """Find semantically cross-linked narrative memories from past life arcs."""
        from dadbot.managers.heritage_graph import HeritageGraphManager

        return _json_safe(
            HeritageGraphManager(_get_bot(bot)).cross_link_query(
                str(context or ""), max_links=max(1, int(max_links or 3))
            )
        )

    if hasattr(server, "resource"):

        @server.resource("dadbot://memory/narratives")
        def narrative_memory_resource() -> str:
            runtime = _get_bot(bot)
            return json.dumps(_json_safe(runtime.narrative_memories()), indent=2)

        @server.resource("dadbot://workshop/status")
        def workshop_resource() -> str:
            runtime = _get_bot(bot)
            return json.dumps(_json_safe(runtime.dashboard_status_snapshot()), indent=2)

        @server.resource("dadbot://memory/heritage")
        def heritage_graph_resource() -> str:
            from dadbot.managers.heritage_graph import HeritageGraphManager

            runtime = _get_bot(bot)
            # Return cross-links based on last known session topic
            last_archive = list(runtime.session_archive() or [])
            seed = str((last_archive[-1].get("summary") or "") if last_archive else "")
            return json.dumps(_json_safe(HeritageGraphManager(runtime).cross_link_query(seed, max_links=5)), indent=2)

    return server


def _resolve_pid_path() -> pathlib.Path:
    """Return the canonical PID path: <project_root>/session_logs/local_mcp_server.pid.

    Derived from __file__ so it works even when DadBot has not been initialised yet.
    """
    project_root = pathlib.Path(__file__).resolve().parent.parent
    base = project_root / "session_logs"
    base.mkdir(parents=True, exist_ok=True)
    return base / "local_mcp_server.pid"


def main(bot: DadBot | None = None) -> None:
    pid_path = _resolve_pid_path()
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    def _cleanup_pid() -> None:
        try:
            pid_path.unlink(missing_ok=True)
        except OSError:
            pass

    atexit.register(_cleanup_pid)

    server = build_server(bot=bot)
    server.run()


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    main()
