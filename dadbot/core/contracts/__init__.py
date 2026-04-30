"""Canonical core contracts for execution context and mutation taxonomy.

Exports are resolved lazily to avoid import cycles between graph modules and
contract helpers during module initialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dadbot.core.contracts.execution_context import ExecutionContextCarrier, require_turn_context
    from dadbot.core.contracts.mutation import CANONICAL_MUTATION_VOCAB, ensure_valid_mutation_op

__all__ = [
    "CANONICAL_MUTATION_VOCAB",
    "ExecutionContextCarrier",
    "ensure_valid_mutation_op",
    "require_turn_context",
]


def __getattr__(name: str):
    if name in {"ExecutionContextCarrier", "require_turn_context"}:
        from dadbot.core.contracts.execution_context import (
            ExecutionContextCarrier,
            require_turn_context,
        )

        return {
            "ExecutionContextCarrier": ExecutionContextCarrier,
            "require_turn_context": require_turn_context,
        }[name]
    if name in {"CANONICAL_MUTATION_VOCAB", "ensure_valid_mutation_op"}:
        from dadbot.core.contracts.mutation import (
            CANONICAL_MUTATION_VOCAB,
            ensure_valid_mutation_op,
        )

        return {
            "CANONICAL_MUTATION_VOCAB": CANONICAL_MUTATION_VOCAB,
            "ensure_valid_mutation_op": ensure_valid_mutation_op,
        }[name]
    raise AttributeError(name)
