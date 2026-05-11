from __future__ import annotations

import hashlib
import json
from typing import Any

from dadbot.core.canonical_event import NON_CANONICAL_PAYLOAD_FIELDS, validate_trace


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


def _coerce_bool(value: Any, *, default: bool = True) -> bool:
    if value is None:
        return default
    return bool(value)


class RuntimeCorrectnessKernel:
    """Single post-commit enforcement point for cross-system runtime invariants."""

    def __init__(
        self,
        *,
        determinism: Any = None,
        memory: Any = None,
        graph: Any = None,
        canonicalizer: Any = None,
        facade: Any = None,
    ) -> None:
        self.determinism = determinism
        self.memory = memory
        self.graph = graph
        self.canonicalizer = canonicalizer
        self.facade = facade

    def run(self, turn_context: Any, snapshot: dict[str, Any]) -> dict[str, Any]:
        report = {
            "determinism": self._check_determinism(turn_context, snapshot),
            "memory": self._check_memory(snapshot),
            "graph": self._check_graph(snapshot),
            "canonical": self._check_canonical(snapshot),
            "facade": self._check_facade(snapshot),
        }
        self._assert_global_consistency(report)
        report["fingerprint"] = self._compute_fingerprint(report)
        return report

    def _check_determinism(self, _ctx: Any, snapshot: dict[str, Any]) -> dict[str, Any]:
        expected = str(snapshot.get("expected_execution_confluence_hash") or "").strip()
        observed = str(snapshot.get("observed_execution_confluence_hash") or "").strip()
        replay_equivalent = True if not expected else expected == observed
        sealed_violations = list(snapshot.get("sealed_slot_violations") or [])
        return {
            "replay_equivalent": bool(replay_equivalent),
            "no_runtime_entropy": not bool(snapshot.get("uuid_entropy", False)),
            "sealed_slots_valid": len(sealed_violations) == 0,
        }

    def _check_memory(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        retrieval_set = list(snapshot.get("memory_retrieval_set") or [])
        commit_boundary_count = int(snapshot.get("commit_boundary_count") or 0)
        has_terminal_mutation_hash = bool(str(snapshot.get("post_commit_mutation_effects_hash") or "").strip())
        return {
            "graph_memory_consistent": _coerce_bool(snapshot.get("graph_memory_consistent"), default=True),
            "mutation_chain_valid": bool(has_terminal_mutation_hash or commit_boundary_count in {0, 1}),
            "canonical_order_stable": _coerce_bool(snapshot.get("canonical_order_stable"), default=True)
            and isinstance(retrieval_set, list),
        }

    def _check_graph(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        generation_before = snapshot.get("graph_generation_before")
        generation_after = snapshot.get("graph_generation_after")
        monotonic = True
        if generation_before is not None and generation_after is not None:
            monotonic = int(generation_after) >= int(generation_before)
        return {
            "generation_monotonic": bool(monotonic),
            "cache_invalidation_correct": _coerce_bool(snapshot.get("graph_cache_invalidation_ok"), default=True),
            "projection_fresh": not bool(snapshot.get("graph_projection_stale", False)),
        }

    def _check_canonical(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        trace_events = list(snapshot.get("trace_events") or [])
        no_non_canonical_fields = True
        if isinstance(snapshot.get("payload_sample"), dict):
            payload_sample = dict(snapshot.get("payload_sample") or {})
            no_non_canonical_fields = not any(
                field in payload_sample for field in NON_CANONICAL_PAYLOAD_FIELDS
            )

        trace_rejection_correct = True
        if trace_events:
            try:
                validate_trace(trace_events)
            except AssertionError:
                trace_rejection_correct = False

        required = {
            "trace_id",
            "session_id",
            "execution_dag_hash",
            "determinism_closure_hash",
        }
        schema_closed = all(key in snapshot for key in required)

        return {
            "no_non_canonical_fields": bool(no_non_canonical_fields),
            "trace_rejection_correct": bool(trace_rejection_correct),
            "schema_closed": bool(schema_closed),
        }

    def _check_facade(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "config_mapping_valid": _coerce_bool(snapshot.get("facade_config_mapping_valid"), default=True),
            "runtime_map_consistent": _coerce_bool(snapshot.get("facade_runtime_map_consistent"), default=True),
            "no_orphan_attributes": _coerce_bool(snapshot.get("facade_no_orphan_attributes"), default=True),
        }

    def _assert_global_consistency(self, report: dict[str, Any]) -> None:
        failures: list[tuple[str, str]] = []
        for section, checks in report.items():
            if section == "fingerprint":
                continue
            for key, value in dict(checks or {}).items():
                if value is False or value is None:
                    failures.append((section, key))
        if failures:
            raise RuntimeError(f"RuntimeCorrectnessViolation: {failures}")

    def _compute_fingerprint(self, report: dict[str, Any]) -> str:
        payload = {key: value for key, value in dict(report).items() if key != "fingerprint"}
        return _sha256_json(payload)


__all__ = ["RuntimeCorrectnessKernel"]