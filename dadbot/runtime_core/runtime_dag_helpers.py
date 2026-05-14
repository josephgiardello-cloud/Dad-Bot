"""DAG-level helper utilities for runtime node processing."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def node_signature(node: dict[str, Any]) -> str:
    """Return a stable content-hash signature for a DAG node dict."""
    key = {
        "name": str(node.get("name") or node.get("step") or ""),
        "kind": str(node.get("kind") or "reasoning"),
        "tool_name": str(node.get("tool_name") or ""),
    }
    raw = json.dumps(key, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


__all__ = ["node_signature"]
