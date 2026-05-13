from __future__ import annotations

import time
from typing import Any


class ToolEcosystemHub:
    """Broad connector inventory and external integration scoring layer."""

    def register_connector(
        self,
        *,
        state: dict[str, Any],
        name: str,
        capabilities: list[str],
        endpoint: str = "",
        health: float = 1.0,
    ) -> dict[str, Any]:
        store = dict(state.get("tool_ecosystem") or {})
        connectors = dict(store.get("connectors") or {})
        key = str(name or "").strip().lower()
        if not key:
            return {}
        row = {
            "name": key,
            "capabilities": sorted({str(item).strip().lower() for item in list(capabilities or []) if str(item).strip()}),
            "endpoint": str(endpoint or ""),
            "health": max(0.0, min(1.0, float(health))),
            "updated_at": float(time.time()),
            "calls": int(dict(connectors.get(key) or {}).get("calls") or 0),
            "failures": int(dict(connectors.get(key) or {}).get("failures") or 0),
            "latency_ms_avg": float(dict(connectors.get(key) or {}).get("latency_ms_avg") or 0.0),
        }
        connectors[key] = row
        store["connectors"] = connectors
        store["updated_at"] = float(time.time())
        state["tool_ecosystem"] = store
        return row

    def rank_connectors(
        self,
        *,
        state: dict[str, Any],
        needed_capabilities: list[str],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        store = dict(state.get("tool_ecosystem") or {})
        connectors = [dict(item) for item in dict(store.get("connectors") or {}).values() if isinstance(item, dict)]
        needs = {str(item).strip().lower() for item in list(needed_capabilities or []) if str(item).strip()}

        ranked: list[tuple[float, dict[str, Any]]] = []
        for row in connectors:
            caps = {str(item).strip().lower() for item in list(row.get("capabilities") or []) if str(item).strip()}
            match = 0.0
            if needs:
                match = float(len(needs.intersection(caps))) / float(max(1, len(needs)))
            failure_rate = 0.0
            calls = int(row.get("calls") or 0)
            fails = int(row.get("failures") or 0)
            if calls > 0:
                failure_rate = float(fails) / float(calls)
            latency_penalty = min(1.0, float(row.get("latency_ms_avg") or 0.0) / 5000.0)
            health = max(0.0, min(1.0, float(row.get("health") or 0.0)))
            score = (0.45 * match) + (0.35 * health) + (0.20 * (1.0 - failure_rate - latency_penalty))
            out = dict(row)
            out["score"] = round(score, 6)
            ranked.append((score, out))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [row for _score, row in ranked[: max(1, int(limit))]]

    def observe_tool_execution(
        self,
        *,
        state: dict[str, Any],
        name: str,
        success: bool,
        latency_ms: float,
    ) -> dict[str, Any]:
        store = dict(state.get("tool_ecosystem") or {})
        connectors = dict(store.get("connectors") or {})
        key = str(name or "").strip().lower()
        if not key:
            return {}
        row = dict(connectors.get(key) or {})
        row.setdefault("name", key)
        row.setdefault("capabilities", [])
        row.setdefault("endpoint", "")
        row.setdefault("health", 0.7)
        calls = int(row.get("calls") or 0) + 1
        failures = int(row.get("failures") or 0) + (0 if success else 1)
        prev_avg = float(row.get("latency_ms_avg") or 0.0)
        latency = max(0.0, float(latency_ms))
        row["calls"] = calls
        row["failures"] = failures
        row["latency_ms_avg"] = ((prev_avg * float(calls - 1)) + latency) / float(max(1, calls))
        row["health"] = max(0.0, min(1.0, 1.0 - (float(failures) / float(max(1, calls)))))
        row["updated_at"] = float(time.time())
        connectors[key] = row
        store["connectors"] = connectors
        store["updated_at"] = float(time.time())
        state["tool_ecosystem"] = store
        return row

    def summary(self, *, state: dict[str, Any]) -> dict[str, Any]:
        store = dict(state.get("tool_ecosystem") or {})
        connectors = [dict(item) for item in dict(store.get("connectors") or {}).values() if isinstance(item, dict)]
        total = int(len(connectors))
        healthy = int(sum(1 for item in connectors if float(item.get("health") or 0.0) >= 0.8))
        return {
            "total_connectors": total,
            "healthy_connectors": healthy,
            "updated_at": float(store.get("updated_at") or 0.0),
            "top_connectors": sorted(
                connectors,
                key=lambda item: float(item.get("health") or 0.0),
                reverse=True,
            )[:5],
        }
