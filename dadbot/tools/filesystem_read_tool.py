"""Safe read-only filesystem tool — plugs into ExternalToolRuntime / DynamicToolRegistry.

Supports two actions:
    read   — return the UTF-8 content of a file
    list   — return the names of entries in a directory

Security contract
-----------------
- All paths are resolved to absolute form and verified to remain inside
  `root_dir`.  Attempting to escape via `..` or symlinks pointing outside
  the root raises an error and returns ToolExecutionStatus.ERROR.
- Symlinks are NOT followed for the final path component of `read`; the
  check is performed on the resolved path so symlinks to outside the root
  are also blocked.
- Binary files are decoded with errors="replace"; the caller receives the
  replacement character for undecodable bytes but never a crash.
- File size is capped at max_file_bytes (default 256 KB).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dadbot.core.external_tool_runtime import (
    IsolationProfile,
    ResourceLimits,
    ToolCapability,
    ToolExecutionResult,
    ToolExecutionStatus,
)

_DEFAULT_MAX_FILE_BYTES = 256 * 1024  # 256 KB
_DEFAULT_MAX_LIST_ENTRIES = 1000


def _safe_resolve(root: Path, user_path: str) -> Path | None:
    """Resolve user_path relative to root, returning None if it escapes."""
    if not str(user_path or "").strip():
        return None
    raw = Path(user_path)
    if raw.is_absolute():
        candidate = raw.resolve()
    else:
        candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def build_filesystem_read_tool(
    root_dir: str,
    *,
    max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
    max_list_entries: int = _DEFAULT_MAX_LIST_ENTRIES,
) -> tuple[ToolCapability, Any]:
    """Build a (ToolCapability, handler) pair for safe read-only filesystem access.

    All access is sandboxed inside *root_dir*.  Attempts to escape via path
    traversal or absolute paths pointing outside the root are rejected.

    Register the returned pair with DynamicToolRegistry:
        cap, handler = build_filesystem_read_tool("/data/sandbox")
        registry.register(cap, handler)

    Payload schema:
        {
            "action": "read" | "list"   (required)
            "path":   "relative/path"   (required; relative to root_dir)
        }
    """
    root = Path(str(root_dir)).resolve()
    max_bytes = max(1, int(max_file_bytes))
    max_entries = max(1, int(max_list_entries))

    def handler(payload: dict[str, Any]) -> ToolExecutionResult:
        action = str(payload.get("action") or "read").strip().lower()
        user_path = str(payload.get("path") or "").strip()

        if action not in {"read", "list"}:
            return ToolExecutionResult(
                tool_name="filesystem_read",
                status=ToolExecutionStatus.ERROR,
                output=None,
                error=f"unknown_action:{action}",
            )

        resolved = _safe_resolve(root, user_path)
        if resolved is None:
            return ToolExecutionResult(
                tool_name="filesystem_read",
                status=ToolExecutionStatus.ERROR,
                output=None,
                error="path_traversal_rejected",
                metadata={"path": user_path},
            )

        if not resolved.exists():
            return ToolExecutionResult(
                tool_name="filesystem_read",
                status=ToolExecutionStatus.ERROR,
                output=None,
                error="path_not_found",
                metadata={"path": user_path},
            )

        if action == "read":
            if not resolved.is_file():
                return ToolExecutionResult(
                    tool_name="filesystem_read",
                    status=ToolExecutionStatus.ERROR,
                    output=None,
                    error="not_a_file",
                    metadata={"path": user_path},
                )
            file_size = resolved.stat().st_size
            if file_size > max_bytes:
                return ToolExecutionResult(
                    tool_name="filesystem_read",
                    status=ToolExecutionStatus.ERROR,
                    output=None,
                    error=f"file_too_large:{file_size}",
                    metadata={"path": user_path, "size_bytes": file_size},
                )
            content = resolved.read_bytes()[:max_bytes].decode("utf-8", errors="replace")
            return ToolExecutionResult(
                tool_name="filesystem_read",
                status=ToolExecutionStatus.OK,
                output={
                    "path": str(resolved.relative_to(root)),
                    "content": content,
                    "size_bytes": file_size,
                },
                confidence=1.0,
                metadata={"root": str(root)},
            )

        # action == "list"
        if not resolved.is_dir():
            return ToolExecutionResult(
                tool_name="filesystem_read",
                status=ToolExecutionStatus.ERROR,
                output=None,
                error="not_a_directory",
                metadata={"path": user_path},
            )
        entries = []
        for entry in sorted(resolved.iterdir())[:max_entries]:
            entries.append({
                "name": entry.name,
                "type": "dir" if entry.is_dir() else "file",
                "size_bytes": entry.stat().st_size if entry.is_file() else None,
            })
        return ToolExecutionResult(
            tool_name="filesystem_read",
            status=ToolExecutionStatus.OK,
            output={
                "path": str(resolved.relative_to(root)),
                "entries": entries,
                "count": len(entries),
                "truncated": len(list(resolved.iterdir())) > max_entries,
            },
            confidence=1.0,
            metadata={"root": str(root)},
        )

    capability = ToolCapability(
        name="filesystem_read",
        version="1.0.0",
        intents=("file_read", "read_file", "list_directory", "filesystem_access"),
        cost_units=0.1,
        avg_latency_ms=5.0,
        reliability=0.99,
        supports_partial=False,
        tags=("filesystem", "local", "readonly"),
    )
    return capability, handler


def build_filesystem_isolation_profile(root_dir: str) -> IsolationProfile:
    return IsolationProfile(
        tool_name="filesystem_read",
        compartment="local_filesystem",
        limits=ResourceLimits(
            max_cpu_ms=500.0,
            max_memory_mb=16.0,
            max_io_ops=10,
            max_network_calls=0,
        ),
        allow_network=False,
        allow_filesystem=True,
    )
