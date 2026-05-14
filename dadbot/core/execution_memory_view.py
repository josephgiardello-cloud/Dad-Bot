from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)


def _stable_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


def _memory_entry_id(normalized: dict[str, Any]) -> str:
    return str(
        normalized.get("memory_id")
        or normalized.get("id")
        or normalized.get("summary_key")
        or normalized.get("key")
        or "",
    ).strip()


def _memory_entry_fallback_candidate(normalized: dict[str, Any]) -> dict[str, Any]:
    candidate = {
        "summary": str(normalized.get("summary") or normalized.get("title") or ""),
        "content": str(normalized.get("content") or normalized.get("value") or ""),
        "category": str(normalized.get("category") or normalized.get("type") or ""),
        "source": str(normalized.get("source") or normalized.get("origin") or ""),
        "tool": str(normalized.get("tool") or normalized.get("tool_name") or ""),
    }
    if not any(str(value or "").strip() for value in candidate.values()):
        candidate["payload"] = dict(normalized)
    return candidate


def _memory_entry_signature(item: dict[str, Any]) -> str:
    normalized = dict(item or {})
    memory_id = _memory_entry_id(normalized)
    if memory_id:
        candidate = {"memory_id": memory_id}
    else:
        candidate = _memory_entry_fallback_candidate(normalized)
    return _stable_sha256(candidate)


def _merge_memory_candidate(
    *,
    item: dict[str, Any],
    label: str,
    source_index: int,
    merged: list[dict[str, Any]],
    index_by_signature: dict[str, int],
    conflicts: list[dict[str, Any]],
) -> None:
    signature = _memory_entry_signature(item)
    candidate = dict(item)
    candidate["memory_merge_source"] = label
    candidate["memory_merge_order"] = int(source_index)
    if signature not in index_by_signature:
        index_by_signature[signature] = len(merged)
        merged.append(candidate)
        return

    existing_index = index_by_signature[signature]
    existing = merged[existing_index]
    if existing != candidate:
        conflicts.append(
            {
                "signature": signature,
                "kept_source": str(existing.get("memory_merge_source") or ""),
                "replaced_by": label,
                "kept_summary": str(existing.get("summary") or existing.get("content") or ""),
                "new_summary": str(candidate.get("summary") or candidate.get("content") or ""),
            },
        )
    merged[existing_index] = candidate


def merge_memory_retrieval_sets(
    *sources: list[dict[str, Any]] | None,
    source_labels: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically merge memory retrieval lists.

    Source order is authoritative: later sources replace earlier items that
    describe the same memory key. This gives the turn a single reconciliation
    rule instead of three competing refresh paths.
    """
    labels = list(source_labels or [])
    merged: list[dict[str, Any]] = []
    index_by_signature: dict[str, int] = {}
    conflicts: list[dict[str, Any]] = []
    source_summary: list[dict[str, Any]] = []

    for source_index, source in enumerate(sources):
        label = labels[source_index] if source_index < len(labels) else f"source_{source_index + 1}"
        normalized_source = [dict(item) for item in list(source or []) if isinstance(item, dict)]
        source_summary.append({"label": label, "count": len(normalized_source)})

        for item in normalized_source:
            _merge_memory_candidate(
                item=item,
                label=label,
                source_index=source_index,
                merged=merged,
                index_by_signature=index_by_signature,
                conflicts=conflicts,
            )

    reconciliation = {
        "reconciliation_id": _stable_sha256(
            {
                "sources": source_summary,
                "signatures": [
                    _memory_entry_signature(item)
                    for item in merged
                ],
            },
        )[:16],
        "merged_count": len(merged),
        "conflict_count": len(conflicts),
        "sources": source_summary,
        "conflicts": conflicts,
    }
    return merged, reconciliation


def _coerce_memory_snapshot(snapshot: Any) -> tuple[dict[str, Any], str]:
    if not isinstance(snapshot, dict):
        return {
            "memory_structured": {},
            "memory_full_history_id": "",
        }, "missing_or_non_dict"

    structured = snapshot.get("memory_structured")
    history_id = snapshot.get("memory_full_history_id")
    issues: list[str] = []

    if not isinstance(structured, dict):
        issues.append("memory_structured_not_dict")
    if history_id is not None and not isinstance(history_id, str):
        issues.append("memory_full_history_id_not_str")

    normalized = {
        "memory_structured": dict(structured) if isinstance(structured, dict) else {},
        "memory_full_history_id": str(history_id or ""),
    }
    return normalized, ",".join(issues)


@dataclass(frozen=True)
class ExecutionMemoryView:
    """Execution-scoped memory snapshot used for deterministic replay."""

    state_id: str
    memory_structured: dict[str, Any]
    memory_full_history_id: str
    memory_retrieval_set: list[dict[str, Any]]

    @classmethod
    def from_context(cls, context: Any) -> ExecutionMemoryView:
        state = dict(getattr(context, "state", {}) or {})
        snapshot, warning = _coerce_memory_snapshot(state.get("memory_snapshot"))
        if warning:
            logger.warning(
                "ExecutionMemoryView.from_context coerced malformed memory_snapshot (warning=%s)",
                warning,
            )
        structured = dict(snapshot.get("memory_structured") or {})
        history_id = str(snapshot.get("memory_full_history_id") or "")
        retrieval = [
            item if isinstance(item, dict) else {"value": item}
            for item in list(state.get("memory_retrieval_set") or [])
        ]
        state_id = _stable_sha256(
            {
                "memory_structured": structured,
                "memory_full_history_id": history_id,
                "memory_retrieval_set": retrieval,
            },
        )
        return cls(
            state_id=f"memv-{state_id[:16]}",
            memory_structured=structured,
            memory_full_history_id=history_id,
            memory_retrieval_set=retrieval,
        )

    @classmethod
    def from_trace(
        cls,
        execution_trace_context: dict[str, Any],
        *,
        fallback_memory_view: dict[str, Any] | None = None,
    ) -> ExecutionMemoryView:
        fallback = dict(fallback_memory_view or {})
        memory_snapshot = dict(
            execution_trace_context.get("memory_snapshot_used") or {},
        )

        structured = dict(
            memory_snapshot.get("memory_structured") or fallback.get("memory_structured") or {},
        )
        history_id = str(
            memory_snapshot.get("memory_full_history_id") or fallback.get("memory_full_history_id") or "",
        )
        retrieval = [
            item if isinstance(item, dict) else {"value": item}
            for item in list(
                execution_trace_context.get("memory_retrieval_set") or fallback.get("memory_retrieval_set") or [],
            )
        ]

        state_id = _stable_sha256(
            {
                "memory_structured": structured,
                "memory_full_history_id": history_id,
                "memory_retrieval_set": retrieval,
            },
        )
        return cls(
            state_id=f"memv-{state_id[:16]}",
            memory_structured=structured,
            memory_full_history_id=history_id,
            memory_retrieval_set=retrieval,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "memory_structured": dict(self.memory_structured),
            "memory_full_history_id": self.memory_full_history_id,
            "memory_retrieval_set": list(self.memory_retrieval_set),
        }


__all__ = [
    "ExecutionMemoryView",
    "merge_memory_retrieval_sets",
]
