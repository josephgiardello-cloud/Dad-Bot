from __future__ import annotations

import atexit
import ast
import fnmatch
import html
import json
import os
import pathlib
import re
import webbrowser
from datetime import datetime
from typing import Any
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen

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
    "computer_search_directory",
    "computer_summarize_directory",
    "computer_refactor_python_function",
    "research_fetch_url",
    "research_search_web",
    "executive_add_task",
    "executive_list_tasks",
    "executive_complete_task",
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
        "capabilities": ["memory", "calendar", "email", "browser", "filesystem", "research", "executive"],
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


def _iter_workspace_files(base: pathlib.Path) -> list[pathlib.Path]:
    return [item for item in base.rglob("*") if item.is_file()]


def _safe_read_text(path: pathlib.Path, *, max_chars: int = 120_000) -> str:
    return path.read_text(encoding="utf-8", errors="replace")[: max(1, int(max_chars or 1))]


def _task_file_path() -> pathlib.Path:
    target = _project_root() / "session_logs" / "executive_tasks.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text("# Dad-Bot Executive Tasks\n\n", encoding="utf-8")
    return target


def _parse_due_iso(due_text: str) -> str:
    raw = str(due_text or "").strip()
    if not raw:
        return ""
    try:
        from dateutil import parser as _dateutil_parser  # pyright: ignore[reportMissingImports]

        parsed = _dateutil_parser.parse(raw, fuzzy=True)
        return parsed.isoformat(timespec="seconds")
    except Exception:
        return ""


def _load_markdown_tasks(path: pathlib.Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    tasks: list[dict[str, Any]] = []
    pattern = re.compile(
        r"^- \[(?P<done>[ xX])\] (?P<title>.*?) \{id:(?P<id>[A-Za-z0-9_-]+), due:(?P<due>[^,]*), pri:(?P<pri>[^}]*)\}$"
    )
    for line in lines:
        match = pattern.match(line.strip())
        if not match:
            continue
        tasks.append(
            {
                "id": match.group("id"),
                "title": match.group("title"),
                "due": match.group("due"),
                "priority": match.group("pri"),
                "done": match.group("done").lower() == "x",
            }
        )
    return tasks


def _write_markdown_tasks(path: pathlib.Path, tasks: list[dict[str, Any]]) -> None:
    header = ["# Dad-Bot Executive Tasks", ""]
    body = []
    for task in tasks:
        done_marker = "x" if bool(task.get("done")) else " "
        body.append(
            f"- [{done_marker}] {str(task.get('title') or '').strip()} "
            f"{{id:{str(task.get('id') or '')}, due:{str(task.get('due') or '')}, pri:{str(task.get('priority') or 'normal')}}}"
        )
    path.write_text("\n".join(header + body) + "\n", encoding="utf-8")


def get_pending_executive_tasks(*, limit: int = 8) -> list[dict[str, Any]]:
    path = _task_file_path()
    tasks = _load_markdown_tasks(path)
    pending = [task for task in tasks if not bool(task.get("done"))]
    pending.sort(
        key=lambda task: (
            str(task.get("due") or "9999"),
            str(task.get("priority") or "normal"),
            str(task.get("id") or ""),
        )
    )
    return pending[: max(1, int(limit or 1))]


def _persist_research_memory(runtime_bot: DadBot, *, url: str, preview: str) -> dict[str, Any]:
    summary = str(preview or "").strip()
    if len(summary) > 320:
        summary = summary[:320].rstrip() + "..."

    memory_row = {
        "summary": f"Research finding from {str(url or '').strip()}: {summary}".strip(),
        "category": "research",
        "mood": "informative",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "source_url": str(url or "").strip(),
    }

    indexed = False
    sync_fn = getattr(runtime_bot, "sync_semantic_memory_index", None)
    if callable(sync_fn):
        try:
            sync_fn([memory_row])
            indexed = True
        except Exception:
            indexed = False

    event_written = False
    writer = getattr(runtime_bot, "ledger_writer", None)
    if callable(getattr(writer, "write_event", None)):
        try:
            writer.write_event(
                "MEMORY_NOTE",
                session_id="default",
                step_key="mcp.research.fetch",
                trace_token=f"mcp-research:{datetime.now().strftime('%Y%m%d%H%M%S')}",
                payload={
                    "event_type": "ResearchFindingEvent",
                    "summary": memory_row["summary"],
                    "category": "research",
                    "source_url": memory_row["source_url"],
                    "occurred_at": memory_row["updated_at"],
                },
                committed=True,
            )
            event_written = True
        except Exception:
            event_written = False

    return {
        "memory_indexed": indexed,
        "memory_event_written": event_written,
        "memory_summary": memory_row["summary"],
    }


def _python_function_refactor(
    target: pathlib.Path,
    *,
    function_name: str,
    new_name: str = "",
    prepend_docstring: str = "",
) -> dict[str, Any]:
    source = _safe_read_text(target)
    tree = ast.parse(source)
    lines = source.splitlines()

    node: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for candidate in ast.walk(tree):
        if isinstance(candidate, ast.FunctionDef | ast.AsyncFunctionDef) and candidate.name == function_name:
            node = candidate
            break
    if node is None:
        raise ValueError(f"function '{function_name}' not found")

    changed = False
    if new_name:
        decl_line = node.lineno - 1
        lines[decl_line] = re.sub(
            rf"\bdef\s+{re.escape(function_name)}\b",
            f"def {new_name}",
            lines[decl_line],
        )
        lines[decl_line] = re.sub(
            rf"\basync\s+def\s+{re.escape(function_name)}\b",
            f"async def {new_name}",
            lines[decl_line],
        )
        changed = True

    if prepend_docstring:
        fn_start_line = node.lineno
        insert_at = fn_start_line
        indent = " " * (int(getattr(node, "col_offset", 0)) + 4)
        doc = f'{indent}"""{str(prepend_docstring).strip()}"""'
        # Insert only if no existing docstring
        has_doc = bool(ast.get_docstring(node))
        if not has_doc:
            lines.insert(insert_at, doc)
            changed = True

    if not changed:
        return {"changed": False, "path": str(target.relative_to(_project_root()))}

    updated = "\n".join(lines) + ("\n" if source.endswith("\n") else "")
    target.write_text(updated, encoding="utf-8")
    return {
        "changed": True,
        "path": str(target.relative_to(_project_root())),
        "function": function_name,
        "renamed_to": str(new_name or ""),
        "docstring_added": bool(prepend_docstring),
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

    @server.tool()
    def computer_search_directory(
        path: str = ".",
        pattern: str = "*.py",
        query: str = "",
        max_results: int = 20,
    ) -> dict[str, Any]:
        """Search files under a workspace directory and summarize matched snippets."""
        base = _resolve_workspace_path(path)
        if not base.exists() or not base.is_dir():
            raise ValueError("path must resolve to an existing directory inside the workspace")
        q = str(query or "").strip().lower()
        if not q:
            raise ValueError("query must not be empty")

        matches: list[dict[str, Any]] = []
        for file_path in _iter_workspace_files(base):
            rel = file_path.relative_to(_project_root())
            if not fnmatch.fnmatch(file_path.name, str(pattern or "*")):
                continue
            text = _safe_read_text(file_path, max_chars=200_000)
            low = text.lower()
            index = low.find(q)
            if index < 0:
                continue
            before = max(0, index - 140)
            after = min(len(text), index + len(q) + 140)
            snippet = text[before:after].replace("\n", " ").strip()
            matches.append({"path": str(rel), "snippet": snippet})
            if len(matches) >= max(1, int(max_results or 1)):
                break

        summary = (
            f"Found {len(matches)} matches for '{query}' under {str(base.relative_to(_project_root()))}."
            if matches
            else f"No matches found for '{query}' under {str(base.relative_to(_project_root()))}."
        )
        return {"path": str(base.relative_to(_project_root())), "query": query, "count": len(matches), "summary": summary, "matches": matches}

    @server.tool()
    def computer_summarize_directory(path: str = ".", max_files: int = 60) -> dict[str, Any]:
        """Summarize directory contents by type, size, and recent activity."""
        base = _resolve_workspace_path(path)
        if not base.exists() or not base.is_dir():
            raise ValueError("path must resolve to an existing directory inside the workspace")

        files = _iter_workspace_files(base)
        limited = files[: max(1, int(max_files or 1))]
        by_ext: dict[str, int] = {}
        total_size = 0
        newest: list[tuple[float, pathlib.Path]] = []
        for file_path in limited:
            ext = (file_path.suffix or "<none>").lower()
            by_ext[ext] = by_ext.get(ext, 0) + 1
            stat = file_path.stat()
            total_size += int(stat.st_size)
            newest.append((float(stat.st_mtime), file_path))
        newest.sort(key=lambda item: item[0], reverse=True)
        recent = [
            {
                "path": str(path_item.relative_to(_project_root())),
                "modified_at": datetime.fromtimestamp(ts).isoformat(timespec="seconds"),
            }
            for ts, path_item in newest[:10]
        ]
        return {
            "path": str(base.relative_to(_project_root())),
            "file_count": len(files),
            "scanned": len(limited),
            "total_size_bytes": total_size,
            "extensions": dict(sorted(by_ext.items(), key=lambda item: (-item[1], item[0]))),
            "recent_files": recent,
        }

    @server.tool()
    def computer_refactor_python_function(
        path: str,
        function_name: str,
        new_name: str = "",
        prepend_docstring: str = "",
    ) -> dict[str, Any]:
        """Refactor a specific Python function by renaming it and/or adding a docstring."""
        target = _resolve_workspace_path(path)
        if not target.exists() or not target.is_file():
            raise ValueError("path must resolve to an existing file inside the workspace")
        if target.suffix.lower() != ".py":
            raise ValueError("computer_refactor_python_function only supports Python files")
        if not str(function_name or "").strip():
            raise ValueError("function_name must not be empty")
        return _python_function_refactor(
            target,
            function_name=str(function_name).strip(),
            new_name=str(new_name or "").strip(),
            prepend_docstring=str(prepend_docstring or "").strip(),
        )

    @server.tool()
    def research_fetch_url(url: str, max_chars: int = 6000) -> dict[str, Any]:
        """Fetch live web content from a URL and return stripped text preview."""
        normalized_url = str(url or "").strip()
        parsed = urlparse(normalized_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("url must use http or https scheme")
        req = Request(
            normalized_url,
            headers={"User-Agent": "DadBot/1.0 (+local-mcp research tool)"},
        )
        with urlopen(req, timeout=20) as response:  # nosec B310 - controlled schemes validated above
            raw = response.read().decode("utf-8", errors="replace")
        # naive HTML-to-text strip is sufficient for lightweight research previews
        stripped = re.sub(r"<script[\\s\\S]*?</script>", " ", raw, flags=re.IGNORECASE)
        stripped = re.sub(r"<style[\\s\\S]*?</style>", " ", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        stripped = html.unescape(re.sub(r"\\s+", " ", stripped)).strip()
        preview = stripped[: max(1, int(max_chars or 1))]
        memory_status = _persist_research_memory(_get_bot(bot), url=normalized_url, preview=preview)
        return {
            "url": normalized_url,
            "preview": preview,
            "truncated": len(preview) < len(stripped),
            **memory_status,
        }

    @server.tool()
    def research_search_web(query: str, max_results: int = 5) -> dict[str, Any]:
        """Run a live web search via DuckDuckGo HTML and return top organic links."""
        q = str(query or "").strip()
        if not q:
            raise ValueError("query must not be empty")
        search_url = f"https://duckduckgo.com/html/?q={quote_plus(q)}"
        req = Request(
            search_url,
            headers={"User-Agent": "DadBot/1.0 (+local-mcp research tool)"},
        )
        with urlopen(req, timeout=20) as response:  # nosec B310 - fixed trusted host
            html_doc = response.read().decode("utf-8", errors="replace")

        link_pattern = re.compile(r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>', re.IGNORECASE)
        results = []
        for match in link_pattern.finditer(html_doc):
            href = html.unescape(match.group("href") or "").strip()
            title = html.unescape(re.sub(r"<[^>]+>", "", match.group("title") or "")).strip()
            if not href or not title:
                continue
            results.append({"title": title, "url": href})
            if len(results) >= max(1, int(max_results or 1)):
                break

        return {"query": q, "count": len(results), "results": results}

    @server.tool()
    def executive_add_task(title: str, due_text: str = "", priority: str = "normal") -> dict[str, Any]:
        """Add an executive task to markdown-backed task board."""
        normalized_title = str(title or "").strip()
        if not normalized_title:
            raise ValueError("title must not be empty")
        normalized_priority = str(priority or "normal").strip().lower() or "normal"
        due_iso = _parse_due_iso(due_text)
        task_id = datetime.now().strftime("tsk%Y%m%d%H%M%S")
        path = _task_file_path()
        tasks = _load_markdown_tasks(path)
        task = {
            "id": task_id,
            "title": normalized_title,
            "due": due_iso or str(due_text or "").strip(),
            "priority": normalized_priority,
            "done": False,
        }
        tasks.append(task)
        _write_markdown_tasks(path, tasks[-400:])
        return {"stored": True, "task": task, "path": str(path.relative_to(_project_root()))}

    @server.tool()
    def executive_list_tasks(status: str = "open", limit: int = 20) -> dict[str, Any]:
        """List executive tasks from markdown board."""
        normalized_status = str(status or "open").strip().lower()
        path = _task_file_path()
        tasks = _load_markdown_tasks(path)
        if normalized_status == "open":
            open_tasks = [task for task in tasks if not task.get("done")]
            filtered = get_pending_executive_tasks(limit=max(1, int(limit or 1)))
            total_count = len(open_tasks)
        elif normalized_status == "done":
            filtered = [task for task in tasks if task.get("done")]
            total_count = len(filtered)
        else:
            filtered = list(tasks)
            total_count = len(filtered)
        return {
            "status": normalized_status,
            "count": total_count,
            "tasks": filtered[: max(1, int(limit or 1))],
            "path": str(path.relative_to(_project_root())),
        }

    @server.tool()
    def executive_complete_task(task_id: str) -> dict[str, Any]:
        """Mark a markdown-backed executive task complete by ID."""
        normalized_id = str(task_id or "").strip()
        if not normalized_id:
            raise ValueError("task_id must not be empty")
        path = _task_file_path()
        tasks = _load_markdown_tasks(path)
        updated = False
        for task in tasks:
            if str(task.get("id") or "") == normalized_id:
                task["done"] = True
                updated = True
                break
        if updated:
            _write_markdown_tasks(path, tasks)
        return {"task_id": normalized_id, "completed": updated, "path": str(path.relative_to(_project_root()))}

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
