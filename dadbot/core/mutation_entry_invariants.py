from __future__ import annotations

from typing import Any

from dadbot.core.runtime_errors import InvariantViolation


_ALLOWED_MUTATION_KINDS = frozenset({"core_state_event", "memory_store", "persistence"})


def enforce_mutation_entry_invariants(
    *,
    mutation_kind: str,
    source: str,
    changed_keys: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    kind = str(mutation_kind or "").strip().lower()
    if kind not in _ALLOWED_MUTATION_KINDS:
        raise InvariantViolation(
            "Mutation entry invariant violation: unknown mutation kind",
            context={"mutation_kind": mutation_kind},
        )

    normalized_source = str(source or "").strip()
    if not normalized_source:
        raise InvariantViolation(
            "Mutation entry invariant violation: source is required",
            context={"mutation_kind": kind},
        )

    normalized_keys = sorted({str(key).strip() for key in list(changed_keys or []) if str(key).strip()})
    if any(not key for key in normalized_keys):
        raise InvariantViolation(
            "Mutation entry invariant violation: changed_keys must be non-empty strings",
            context={"mutation_kind": kind, "source": normalized_source},
        )

    meta = dict(metadata or {})
    if "trace_id" in meta and str(meta.get("trace_id") or "").strip() == "":
        raise InvariantViolation(
            "Mutation entry invariant violation: trace_id cannot be blank when provided",
            context={"mutation_kind": kind, "source": normalized_source},
        )
