from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

MCP_SERVER_NAME = "dadbot-local-services"
MCP_CONNECT_SCOPE = "mcp:connect"
MCP_TOKEN_MAX_TTL_SECONDS = 300
MCP_RESOURCE_URL_DEFAULT = "http://127.0.0.1:8765/mcp"
MCP_RESOURCE_TO_TOOL = {
    "dadbot://memory/narratives": "memory_snapshot",
    "dadbot://workshop/status": "workshop_snapshot",
    "dadbot://memory/heritage": "heritage_cross_link_query",
}

MCP_TOOL_BASE_SCOPES = {
    "runtime_health_snapshot": "read",
    "maintenance_snapshot": "read",
    "workshop_snapshot": "read",
    "memory_snapshot": "read",
    "memory_search": "read",
    "add_reminder": "write",
    "list_reminders": "read",
    "add_calendar_event": "write",
    "list_calendar_events": "read",
    "draft_email": "write",
    "read_local_state": "read",
    "write_local_state": "write",
    "heritage_cross_link_query": "read",
    "browser_capabilities": "read",
    "browser_open_url": "write",
    "computer_list_directory": "read",
    "computer_read_text_file": "read",
    "computer_write_text_file": "write",
    "computer_search_directory": "read",
    "computer_summarize_directory": "read",
    "computer_refactor_python_function": "write",
    "research_fetch_url": "write",
    "research_search_web": "write",
    "executive_add_task": "write",
    "executive_list_tasks": "read",
    "executive_complete_task": "write",
}


class MCPToolManifestError(RuntimeError):
    pass


def mcp_tool_scope(tool_name: str) -> str:
    if tool_name not in MCP_TOOL_BASE_SCOPES:
        raise MCPToolManifestError(f"Unknown MCP tool: {tool_name}")
    return f"mcp:tool:{tool_name}"


def mcp_resource_url() -> str:
    value = str(os.environ.get("DADBOT_MCP_RESOURCE_URL") or MCP_RESOURCE_URL_DEFAULT).strip()
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or not parsed.path
    ):
        raise ValueError("DADBOT_MCP_RESOURCE_URL must be an absolute HTTP(S) URL without query or fragment")
    return value.rstrip("/")


def mcp_manifest_path() -> Path:
    return Path(__file__).with_name("tool_manifest.json")


def load_mcp_tool_manifest(path: Path | None = None) -> dict[str, dict[str, Any]]:
    manifest_path = path or mcp_manifest_path()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MCPToolManifestError("MCP tool manifest is unavailable or invalid JSON") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1 or payload.get("server_name") != MCP_SERVER_NAME:
        raise MCPToolManifestError("MCP tool manifest header is invalid")
    raw_tools = payload.get("tools")
    if not isinstance(raw_tools, dict) or set(raw_tools) != set(MCP_TOOL_BASE_SCOPES):
        raise MCPToolManifestError("MCP tool manifest names do not match the registered tool set")

    tools: dict[str, dict[str, Any]] = {}
    for name, base_scope in MCP_TOOL_BASE_SCOPES.items():
        entry = raw_tools.get(name)
        if not isinstance(entry, dict):
            raise MCPToolManifestError(f"MCP tool manifest entry is invalid: {name}")
        digest = entry.get("schema_sha256")
        if (
            entry.get("approved") is not True
            or entry.get("base_scope") != base_scope
            or entry.get("required_scope") != mcp_tool_scope(name)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise MCPToolManifestError(f"MCP tool manifest approval or policy is invalid: {name}")
        tools[name] = dict(entry)
    return tools


def _tool_schema_digest(tool: Any, *, base_scope: str) -> str:
    if hasattr(tool, "model_dump"):
        payload = tool.model_dump(mode="json", by_alias=True, exclude_none=True)
    elif isinstance(tool, dict):
        payload = tool
    else:
        raise MCPToolManifestError("MCP server returned an unsupported tool schema")
    canonical = {
        "name": payload.get("name"),
        "title": payload.get("title"),
        "description": payload.get("description"),
        "inputSchema": payload.get("inputSchema"),
        "outputSchema": payload.get("outputSchema"),
        "_meta": payload.get("_meta"),
        "base_scope": base_scope,
        "required_scope": mcp_tool_scope(str(payload.get("name") or "")),
    }
    encoded = json.dumps(canonical, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_mcp_tool_manifest(tools: list[Any], path: Path | None = None) -> None:
    expected = load_mcp_tool_manifest(path)
    actual: dict[str, Any] = {}
    for tool in tools:
        payload = (
            tool.model_dump(mode="json", by_alias=True, exclude_none=True)
            if hasattr(tool, "model_dump")
            else tool
        )
        if not isinstance(payload, dict):
            raise MCPToolManifestError("MCP server returned an unsupported tool schema")
        name = payload.get("name")
        if not isinstance(name, str) or name in actual:
            raise MCPToolManifestError("MCP server returned an invalid or duplicate tool name")
        actual[name] = tool
    if set(actual) != set(expected):
        raise MCPToolManifestError("Registered MCP tools differ from the approved tool manifest")
    for name, entry in expected.items():
        if _tool_schema_digest(actual[name], base_scope=MCP_TOOL_BASE_SCOPES[name]) != entry["schema_sha256"]:
            raise MCPToolManifestError(f"Registered MCP tool schema drift detected: {name}")


def approved_mcp_tools() -> dict[str, dict[str, Any]]:
    return load_mcp_tool_manifest()
