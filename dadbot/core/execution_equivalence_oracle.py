from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from dadbot.core.execution_replay_engine import verify_terminal_state_replay_equivalence
from dadbot.core.execution_semantics import (
    AllowedTransformations,
    build_execution_state,
    execution_equivalence_relation,
)


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


@dataclass(frozen=True)
class ExecutionEquivalenceOracleResult:
    equivalent: bool
    invariance_hash: str
    violations: list[str]
    replay_report: dict[str, Any]
    semantic_equivalent: bool = False
    semantic_mode: str = "exact"
    semantic_report: dict[str, Any] | None = None
    structural_equivalent: bool = False
    invariants_preserved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "equivalent": bool(self.equivalent),
            "invariance_hash": str(self.invariance_hash),
            "violations": list(self.violations),
            "replay_report": dict(self.replay_report),
            "semantic_equivalent": bool(self.semantic_equivalent),
            "semantic_mode": str(self.semantic_mode),
            "semantic_report": dict(self.semantic_report or {}),
            "structural_equivalent": bool(self.structural_equivalent),
            "invariants_preserved": bool(self.invariants_preserved),
        }


class ExecutionEquivalenceOracle:
    """Single authoritative oracle for full turn replay equivalence."""

    _SemanticMode = Literal["exact", "approximate"]

    @staticmethod
    def _invariance_hash(
        *,
        input_seed: str,
        trace_seed: str,
        memory_state_id: str,
        execution_trace_context: dict[str, Any],
    ) -> str:
        payload = {
            "input_seed": str(input_seed or ""),
            "trace_seed": str(trace_seed or ""),
            "memory_state_id": str(memory_state_id or ""),
            "trace_final_hash": str(execution_trace_context.get("final_hash") or ""),
            "execution_dag_hash": str(
                (execution_trace_context.get("execution_dag") or {}).get("dag_hash") or "",
            ),
        }
        return _stable_hash(payload)

    @classmethod
    def evaluate(
        cls,
        *,
        input_seed: str,
        trace_seed: str,
        memory_state_id: str,
        terminal_state_seed: dict[str, Any],
        execution_trace_context: dict[str, Any],
        semantic_mode: _SemanticMode = "exact",
        allow_causal_reorder: bool = False,
        allow_representation_variance: bool = False,
        tool_output_tolerance: float = 0.0,
    ) -> ExecutionEquivalenceOracleResult:
        replay_report = verify_terminal_state_replay_equivalence(
            terminal_state_seed=terminal_state_seed,
            execution_trace_context=execution_trace_context,
            enforce_dag_equivalence=True,
        )
        expected_terminal_state = dict(
            replay_report.get("expected_terminal_state") or {},
        )
        replayed_terminal_state = dict(
            replay_report.get("replayed_terminal_state") or {},
        )

        left_state = build_execution_state(
            terminal_state=expected_terminal_state,
            execution_trace_context=execution_trace_context,
            semantic_mode=semantic_mode,
        )
        right_state = build_execution_state(
            terminal_state=replayed_terminal_state,
            execution_trace_context=execution_trace_context,
            semantic_mode=semantic_mode,
        )
        relation = execution_equivalence_relation(
            left_state,
            right_state,
            rules=AllowedTransformations(
                allow_causal_reorder=bool(allow_causal_reorder),
                allow_representation_variance=bool(allow_representation_variance),
                tool_output_tolerance=float(tool_output_tolerance or 0.0),
            ),
        )
        semantic_report = {
            "semantic_equivalent": bool(relation.semantic_equivalent),
            "structural_equivalent": bool(relation.structural_equivalent),
            "invariants_preserved": bool(relation.invariants_preserved),
            "semantic_action_equivalent": "semantic_actions" not in relation.violations,
            "memory_state_equivalent": "memory_state" not in relation.violations,
            "tool_output_equivalent": "model_outputs" not in relation.violations,
            "mode": str(semantic_mode),
            "violations": list(relation.violations),
        }
        invariance_hash = cls._invariance_hash(
            input_seed=input_seed,
            trace_seed=trace_seed,
            memory_state_id=memory_state_id,
            execution_trace_context=execution_trace_context,
        )
        violations = list(replay_report.get("violations") or [])
        violations.extend(list(relation.violations))

        equivalent = bool(replay_report.get("equivalent", False)) and bool(
            relation.equivalent,
        )
        return ExecutionEquivalenceOracleResult(
            equivalent=equivalent,
            invariance_hash=invariance_hash,
            violations=violations,
            replay_report=dict(replay_report),
            semantic_equivalent=bool(relation.semantic_equivalent),
            semantic_mode=str(semantic_mode),
            semantic_report=semantic_report,
            structural_equivalent=bool(relation.structural_equivalent),
            invariants_preserved=bool(relation.invariants_preserved),
        )
