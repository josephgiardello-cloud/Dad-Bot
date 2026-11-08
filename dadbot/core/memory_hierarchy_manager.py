from __future__ import annotations

import time
from typing import Any


class MemoryHierarchyManager:
    """Maintains working, episodic, semantic, and long-term memory tiers."""

    def promote_turn(
        self,
        *,
        state: dict[str, Any],
        trace_id: str,
        user_input: str,
        response_text: str,
        semantic_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        hierarchy = dict(state.get("memory_hierarchy") or {})
        working = list(hierarchy.get("working") or [])
        episodic = list(hierarchy.get("episodic") or [])
        long_term = dict(hierarchy.get("long_term") or {})

        now = float(time.time())
        turn_record = {
            "trace_id": str(trace_id or ""),
            "user_input": str(user_input or "")[:260],
            "response": str(response_text or "")[:260],
            "timestamp": now,
        }
        working.append(turn_record)
        working = working[-24:]

        if len(str(user_input or "").strip()) >= 12:
            episodic.append(
                {
                    "trace_id": str(trace_id or ""),
                    "summary": str(user_input or "")[:220],
                    "assistant_summary": str(response_text or "")[:220],
                    "weight": 0.6,
                    "timestamp": now,
                },
            )
        profile = dict(long_term.get("profile") or {})
        if "my " in str(user_input or "").lower() or "i " in str(user_input or "").lower():
            profile["last_user_statement"] = str(user_input or "")[:220]
            profile["updated_at"] = now

        long_term["profile"] = profile
        long_term["semantic_digest"] = {
            "items": [dict(item) for item in list(semantic_items or [])[:12]],
            "updated_at": now,
        }

        hierarchy["working"] = working
        hierarchy["episodic"] = episodic
        hierarchy["semantic_ref"] = {
            "count": int(len(semantic_items or [])),
            "updated_at": now,
        }
        hierarchy["long_term"] = long_term
        hierarchy["updated_at"] = now
        state["memory_hierarchy"] = hierarchy
        return hierarchy

    def lifecycle_maintenance(self, *, state: dict[str, Any]) -> dict[str, Any]:
        hierarchy = dict(state.get("memory_hierarchy") or {})
        episodic = list(hierarchy.get("episodic") or [])
        now = float(time.time())
        distilled: list[dict[str, Any]] = []
        for item in episodic:
            row = dict(item)
            last_decay = row.get("last_decay_at")
            if last_decay is None:
                last_decay = row.get("timestamp")
            if last_decay is None:
                last_decay = now
            last_decay = float(last_decay)
            age_hours = max(0.0, (now - last_decay) / 3600.0)
            weight = row.get("weight")
            weight = 0.5 if weight is None else float(weight)
            row["weight"] = max(0.0, min(1.0, weight * (0.995**age_hours)))
            row["last_decay_at"] = max(now, last_decay)
            distilled.append(row)
        hierarchy["episodic"] = distilled
        hierarchy["updated_at"] = now
        state["memory_hierarchy"] = hierarchy
        return hierarchy
