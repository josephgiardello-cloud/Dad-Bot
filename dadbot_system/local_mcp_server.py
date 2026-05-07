from __future__ import annotations

import atexit
import json
import os
import pathlib
import webbrowser
from typing import Any
from urllib.parse import urlparse

from Dad import DadBot

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - optional dependency
    FastMCP = None


_BOT: DadBot | None = None
_LOCAL_MCP_TOOL_NAMES = [
    "runtime_health_snapshot",
    "maintenance_snapshot",
    "workshop_snapshot",
    "memory_snapshot",
    "memory_search",
    "add_reminder",
    "list_reminders",
    "add_calendar_event",
    "list_calendar_events",
    "draft_email",
    "read_local_state",
    "write_local_state",
    "heritage_cross_link_query",
    "browser_capabilities",
    "browser_open_url",
    "computer_list_directory",
    "computer_read_text_file",
    "computer_write_text_file",
]


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
        "tool_count": len(_LOCAL_MCP_TOOL_NAMES),
        "tool_names": list(_LOCAL_MCP_TOOL_NAMES),
        "capabilities": ["memory", "calendar", "email", "browser", "filesystem"],
        "local_state_entries": len(local_store),
    }


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def _resolve_workspace_path(path: str, *, allow_create: bool = False) -> pathlib.Path:
    candidate = pathlib.Path(str(path or "").strip() or ".")
    root = _project_root()
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("path must stay within the Dad-Bot workspace") from exc
    if allow_create:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


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

    @server.tool()
    def browser_capabilities() -> dict[str, Any]:
        """Describe DadBot's local browser/computer-use capabilities."""
        payload = dict(local_mcp_status(_get_bot(bot)) or {})
        payload["workspace_root"] = str(_project_root())
        payload["browser_actions"] = ["open_url"]
        payload["computer_actions"] = ["list_directory", "read_text_file", "write_text_file"]
        return _json_safe(payload)

    @server.tool()
    def browser_open_url(url: str) -> dict[str, Any]:
        """Open a browser URL using the local machine default browser."""
        normalized_url = str(url or "").strip()
        parsed = urlparse(normalized_url)
        if parsed.scheme not in {"http", "https", "file"}:
            raise ValueError("url must use http, https, or file scheme")
        opened = bool(webbrowser.open(normalized_url, new=2))
        runtime = _get_bot(bot)
        runtime.mutate_memory_store(
            mcp_local_store={
                **dict(runtime.MEMORY_STORE.get("mcp_local_store") or {}),
                "last_browser_action": {"url": normalized_url, "opened": opened},
            }
        )
        return {"url": normalized_url, "opened": opened}

    @server.tool()
    def computer_list_directory(path: str = ".", limit: int = 40) -> dict[str, Any]:
        """List files inside the Dad-Bot workspace."""
        target = _resolve_workspace_path(path)
        if not target.exists() or not target.is_dir():
            raise ValueError("path must resolve to an existing directory inside the workspace")
        entries = []
        for item in sorted(target.iterdir(), key=lambda value: value.name.lower())[: max(1, int(limit or 1))]:
            entry = {
                "name": item.name,
                "path": str(item.relative_to(_project_root())),
                "kind": "directory" if item.is_dir() else "file",
            }
            if item.is_file():
                entry["size"] = item.stat().st_size
            entries.append(entry)
        return {"path": str(target.relative_to(_project_root())), "entries": entries}

    @server.tool()
    def computer_read_text_file(path: str, max_chars: int = 4000) -> dict[str, Any]:
        """Read a UTF-8 text file from the Dad-Bot workspace."""
        target = _resolve_workspace_path(path)
        if not target.exists() or not target.is_file():
            raise ValueError("path must resolve to an existing file inside the workspace")
        content = target.read_text(encoding="utf-8")
        clipped = content[: max(1, int(max_chars or 1))]
        return {
            "path": str(target.relative_to(_project_root())),
            "content": clipped,
            "truncated": len(clipped) < len(content),
        }

    @server.tool()
    def computer_write_text_file(path: str, content: str, append: bool = False) -> dict[str, Any]:
        """Write a UTF-8 text file inside the Dad-Bot workspace."""
        target = _resolve_workspace_path(path, allow_create=True)
        mode = "a" if append else "w"
        with target.open(mode, encoding="utf-8") as handle:
            handle.write(str(content or ""))
        return {
            "path": str(target.relative_to(_project_root())),
            "bytes_written": len(str(content or "").encode("utf-8")),
            "appended": bool(append),
        }

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
