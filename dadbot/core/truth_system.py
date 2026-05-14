from __future__ import annotations

from typing import Any

AUTHORITATIVE_TRUTH_SYSTEM = {
    "execution_kernel": "dadbot.core.graph.TurnGraph",
    "replay_engine": "dadbot.core.execution_replay_engine",
    "memory_truth": "dadbot.memory.semantic_manager.SemanticIndexManager",
}


class TruthSystemViolation(RuntimeError):
    """Raised when a non-authoritative execution/replay/memory truth path is active."""


def enforce_authoritative_truth_system(
    *,
    metadata: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current_metadata = dict(metadata or {})
    current_state = dict(state or {})

    experimental_kernel = bool(
        current_metadata.get("experimental_execution_kernel_enabled")
        or current_state.get("experimental_execution_kernel_enabled"),
    )
    legacy_replay = bool(
        current_metadata.get("use_legacy_replay_engine") or current_state.get("use_legacy_replay_engine"),
    )
    alternate_memory_truth = str(
        current_metadata.get("memory_truth_system") or current_state.get("memory_truth_system") or "",
    ).strip()
    violations: list[str] = []
    if experimental_kernel:
        violations.append("experimental_execution_kernel_enabled")
    if legacy_replay:
        violations.append("use_legacy_replay_engine")
    if alternate_memory_truth and alternate_memory_truth.lower() not in {
        "",
        "semanticindexmanager",
        "dadbot.memory.semantic_manager.semanticindexmanager",
    }:
        violations.append(f"memory_truth_system:{alternate_memory_truth}")

    if violations:
        raise TruthSystemViolation(
            "Non-authoritative truth system path detected: " + ", ".join(violations),
        )

    return {
        "authoritative": True,
        "execution_kernel": AUTHORITATIVE_TRUTH_SYSTEM["execution_kernel"],
        "replay_engine": AUTHORITATIVE_TRUTH_SYSTEM["replay_engine"],
        "memory_truth": AUTHORITATIVE_TRUTH_SYSTEM["memory_truth"],
        "violations": [],
    }
