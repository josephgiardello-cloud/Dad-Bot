from __future__ import annotations

from dadbot.core.graph_types import (
    GoalMutationOp,
    LedgerMutationOp,
    MemoryMutationOp,
    MutationKind,
    RelationshipMutationOp,
)

CANONICAL_MUTATION_VOCAB: dict[str, set[str]] = {
    MutationKind.MEMORY.value: {item.value for item in MemoryMutationOp},
    MutationKind.RELATIONSHIP.value: {item.value for item in RelationshipMutationOp},
    MutationKind.LEDGER.value: {item.value for item in LedgerMutationOp},
    MutationKind.GOAL.value: {item.value for item in GoalMutationOp},
    MutationKind.GRAPH.value: set(),
}


def ensure_valid_mutation_op(kind: MutationKind, op: str) -> None:
    """Raise when a mutation op is not part of the canonical taxonomy."""

    op_name = str(op or "").strip().lower()
    allowed = CANONICAL_MUTATION_VOCAB.get(kind.value, set())
    if op_name and op_name not in allowed:
        raise RuntimeError(f"Unsupported {kind.value} mutation op: {op_name!r}")


def coerce_mutation_kind(value: MutationKind | str) -> MutationKind:
    """Canonical constructor for mutation kind normalization."""

    try:
        return MutationKind(value)
    except ValueError as exc:
        raise RuntimeError(
            f"MutationIntent.type must be one of {[item.value for item in MutationKind]}",
        ) from exc
