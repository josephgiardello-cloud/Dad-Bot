"""dadbot/core/behavior_contract_lock.py — Strict convergence enforcement.

This module implements hard equivalence checks that ensure thin-spine and legacy
execution paths produce IDENTICAL outputs given the same input.

NOT similarity or "close enough" — exact structural and semantic equivalence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class ContractInvariant:
    """Single locked invariant within the behavior contract."""

    invariant_name: str
    description: str
    check_enabled: bool = True
    failure_is_fatal: bool = True


class BehaviorContractLock:
    """Enforce deterministic equivalence between thin-spine and legacy execution paths.
    
    This lock validates three hard invariants:
    
    1. SEMANTIC_OUTPUT: hash(thin_spine_output) == hash(legacy_output)
       Same input must produce byte-for-byte identical response text
    
    2. TOOL_CALL_SEQUENCE: sequence of tool calls must be identical
       Same order, same parameters, same cardinality
    
    3. STATE_MUTATION_GRAPH: final state mutations must be identical
       Same memory writes, same persistence events, same order
    """

    INVARIANTS = {
        "SEMANTIC_OUTPUT": ContractInvariant(
            invariant_name="SEMANTIC_OUTPUT",
            description="Same input → identical response hash",
            check_enabled=True,
            failure_is_fatal=True,
        ),
        "TOOL_CALL_SEQUENCE": ContractInvariant(
            invariant_name="TOOL_CALL_SEQUENCE",
            description="Same input → identical tool invocation sequence",
            check_enabled=True,
            failure_is_fatal=True,
        ),
        "STATE_MUTATION_GRAPH": ContractInvariant(
            invariant_name="STATE_MUTATION_GRAPH",
            description="Same input → identical state mutation graph",
            check_enabled=True,
            failure_is_fatal=True,
        ),
    }

    def __init__(self, execution_mode: str = "strict"):
        """
        Args:
            execution_mode: "strict" (all invariants fatal), "warn" (log only), "audit" (record but continue)
        """
        self.execution_mode = execution_mode
        self._violations: list[dict[str, Any]] = []
        self._verified_calls: int = 0

    def lock_semantic_output(self, *, input_hash: str, legacy_output: str, thin_spine_output: str) -> bool:
        """Verify semantic output equivalence.
        
        Args:
            input_hash: Hash of input to identify mismatch source
            legacy_output: Response from legacy execution path
            thin_spine_output: Response from thin-spine execution path
            
        Returns:
            True if equivalent, False if violation
        """
        legacy_hash = self._hash_output(legacy_output)
        thin_hash = self._hash_output(thin_spine_output)

        if legacy_hash == thin_hash:
            self._verified_calls += 1
            return True

        violation = {
            "invariant": "SEMANTIC_OUTPUT",
            "input_hash": input_hash,
            "legacy_hash": legacy_hash,
            "thin_spine_hash": thin_hash,
            "legacy_output_preview": legacy_output[:200] if legacy_output else "",
            "thin_spine_output_preview": thin_spine_output[:200] if thin_spine_output else "",
        }
        return self._handle_violation(violation)

    def lock_tool_call_sequence(
        self,
        *,
        input_hash: str,
        legacy_calls: list[dict[str, Any]],
        thin_spine_calls: list[dict[str, Any]],
    ) -> bool:
        """Verify tool call sequence equivalence.
        
        Args:
            input_hash: Hash of input
            legacy_calls: Tool calls from legacy path [{"tool": "name", "params": {...}, ...}]
            thin_spine_calls: Tool calls from thin-spine path
            
        Returns:
            True if sequence identical, False if violation
        """
        legacy_seq = self._normalize_call_sequence(legacy_calls)
        thin_seq = self._normalize_call_sequence(thin_spine_calls)

        if legacy_seq == thin_seq:
            self._verified_calls += 1
            return True

        violation = {
            "invariant": "TOOL_CALL_SEQUENCE",
            "input_hash": input_hash,
            "legacy_sequence": legacy_seq,
            "thin_spine_sequence": thin_seq,
            "legacy_count": len(legacy_seq),
            "thin_spine_count": len(thin_seq),
        }
        return self._handle_violation(violation)

    def lock_state_mutation_graph(
        self,
        *,
        input_hash: str,
        legacy_mutations: list[dict[str, Any]],
        thin_spine_mutations: list[dict[str, Any]],
    ) -> bool:
        """Verify state mutation graph equivalence.
        
        Args:
            input_hash: Hash of input
            legacy_mutations: Memory/state writes from legacy path
            thin_spine_mutations: Memory/state writes from thin-spine path
            
        Returns:
            True if mutation graph identical, False if violation
        """
        legacy_graph = self._normalize_mutation_graph(legacy_mutations)
        thin_graph = self._normalize_mutation_graph(thin_spine_mutations)

        if legacy_graph == thin_graph:
            self._verified_calls += 1
            return True

        violation = {
            "invariant": "STATE_MUTATION_GRAPH",
            "input_hash": input_hash,
            "legacy_graph": legacy_graph,
            "thin_spine_graph": thin_graph,
            "legacy_mutation_count": len(legacy_mutations),
            "thin_spine_mutation_count": len(thin_spine_mutations),
        }
        return self._handle_violation(violation)

    # -------- HELPERS --------

    @staticmethod
    def _hash_output(output: str) -> str:
        """SHA256 hash of output for deterministic comparison."""
        normalized = str(output or "").strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_call_sequence(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize tool call sequence for comparison (remove timestamps, trace IDs, etc)."""
        normalized = []
        for call in calls:
            nc = {
                "tool": call.get("tool", ""),
                "params_hash": hashlib.sha256(
                    json.dumps(call.get("params", {}), sort_keys=True).encode()
                ).hexdigest(),
                "cardinality": 1,
            }
            normalized.append(nc)
        return normalized

    @staticmethod
    def _normalize_mutation_graph(mutations: list[dict[str, Any]]) -> str:
        """Normalize mutation graph into deterministic hash.
        
        Includes:
        - Memory keys written
        - Persistence events
        - Order of mutations
        """
        sorted_mutations = sorted(
            mutations,
            key=lambda m: (m.get("timestamp", 0), m.get("key", ""), m.get("value", "")),
        )
        graph_str = json.dumps(sorted_mutations, sort_keys=True)
        return hashlib.sha256(graph_str.encode()).hexdigest()

    def _handle_violation(self, violation: dict[str, Any]) -> bool:
        """Handle contract violation per execution mode."""
        self._violations.append(violation)

        if self.execution_mode == "strict":
            import logging

            logger = logging.getLogger(__name__)
            logger.critical(
                "BEHAVIOR CONTRACT VIOLATION: %s",
                violation.get("invariant", "UNKNOWN"),
                extra=violation,
            )
            raise AssertionError(
                f"Behavior contract locked: {violation.get('invariant')} failed. "
                f"Thin-spine and legacy paths are NOT equivalent."
            )
        elif self.execution_mode == "warn":
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                "Behavior contract violation (audit mode): %s",
                violation.get("invariant", "UNKNOWN"),
                extra=violation,
            )
        elif self.execution_mode == "audit":
            pass  # Just record, don't raise or warn

        return False

    def get_contract_status(self) -> dict[str, Any]:
        """Return contract validation status."""
        return {
            "execution_mode": self.execution_mode,
            "verified_calls": self._verified_calls,
            "violations": len(self._violations),
            "violations_list": self._violations[:10],  # Show first 10
        }

    def reset(self) -> None:
        """Reset violation tracking (for new test session)."""
        self._violations.clear()
        self._verified_calls = 0


# Singleton contract lock
_global_lock = None


def get_behavior_contract_lock(execution_mode: str = "strict") -> BehaviorContractLock:
    """Get or create global contract lock."""
    global _global_lock
    if _global_lock is None:
        _global_lock = BehaviorContractLock(execution_mode=execution_mode)
    return _global_lock


__all__ = [
    "BehaviorContractLock",
    "ContractInvariant",
    "get_behavior_contract_lock",
]
