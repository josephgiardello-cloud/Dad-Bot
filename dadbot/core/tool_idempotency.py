"""Phase 2.1 — Tool Idempotency Primitive.

Canonical tool request hash → canonical result class mapping.

Invariant:
    identical_request_hash → identical_semantic_output_class

"Output class" is not the raw text output (which is LLM-dependent), but
the semantic structure class: (tool_name, status_class, output_type).

Formal contract:
    For any two tool requests R1 and R2:
    if canonical_request_hash(R1) == canonical_request_hash(R2)
    then idempotency_class(R1) == idempotency_class(R2)

This enables:
    - Deterministic replay without LLM re-execution
    - Cache hit detection (same request → skip re-execution)
    - Audit: "was this the same computation as before?"
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from threading import RLock
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


# ---------------------------------------------------------------------------
# Canonical request hash
# ---------------------------------------------------------------------------


def canonical_request_hash(
    tool_name: str,
    args: dict[str, Any],
    intent: str = "",
) -> str:
    """Compute the canonical hash of a tool request.

    The hash is computed from:
    - Normalized tool_name (lowercase, stripped)
    - Canonicalized args (sorted keys, normalized values)
    - intent (optional, included when present for disambiguation)

    Two requests with identical (tool_name, args, intent) produce the
    same hash, regardless of creation time, session, or machine.
    """
    normalized = {
        "tool_name": str(tool_name or "").strip().lower(),
        "args": {k: v for k, v in sorted((dict(args or {})).items())},
        "intent": str(intent or "").strip().lower(),
    }
    return _sha256(normalized)


def canonical_result_class(tool_name: str, status: str, output_type: str = "") -> str:
    """Compute the canonical result class identifier.

    The result class is NOT the raw output — it is the structural class
    that describes what kind of output was produced:
        (tool_name, status_class, output_type)

    Examples:
        ("memory_lookup", "ok", "list")   → class A
        ("memory_lookup", "ok", "dict")   → class B
        ("memory_lookup", "error", "")    → class C

    """
    normalized = {
        "tool_name": str(tool_name or "").strip().lower(),
        "status": str(status or "ok").strip().lower(),
        "output_type": str(output_type or "").strip().lower(),
    }
    return _sha256(normalized)[:16]


def infer_output_type(output: Any) -> str:
    """Infer the structural output type from an actual output value."""
    if output is None:
        return "null"
    if isinstance(output, dict):
        return "dict"
    if isinstance(output, (list, tuple)):
        return "list"
    if isinstance(output, str):
        return "str"
    if isinstance(output, bool):
        return "bool"
    if isinstance(output, (int, float)):
        return "number"
    return type(output).__name__


# ---------------------------------------------------------------------------
# Idempotency record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdempotencyRecord:
    """A stored idempotency record: request hash → result class.

    Used by ToolIdempotencyRegistry to enforce the invariant:
        same request_hash → same result_class (on replay)
    """

    request_hash: str
    result_class: str
    tool_name: str
    intent: str
    status: str
    output_type: str
    hit_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_hash": self.request_hash,
            "result_class": self.result_class,
            "tool_name": self.tool_name,
            "intent": self.intent,
            "status": self.status,
            "output_type": self.output_type,
            "hit_count": self.hit_count,
        }


class IdempotencyViolationError(ValueError):
    """Raised when a result class mismatch violates the idempotency invariant."""

    def __init__(
        self,
        message: str,
        *,
        request_hash: str,
        stored_class: str,
        new_class: str,
    ) -> None:
        super().__init__(message)
        self.request_hash = request_hash
        self.stored_class = stored_class
        self.new_class = new_class


# ---------------------------------------------------------------------------
# ToolIdempotencyRegistry
# ---------------------------------------------------------------------------


class ToolIdempotencyRegistry:
    """Maps canonical request hashes to result classes.

    Thread-safe: all operations use an RLock.

    Contract:
        - register(request_hash, result_class): store the mapping on first call.
        - is_cache_hit(request_hash): True iff the hash was seen before.
        - assert_idempotent(request_hash, result_class): raises on class mismatch.
        - get_class(request_hash): return stored class or None.

    The registry does NOT store raw outputs — only structural classes.
    This makes it safe for audit and cross-run comparison.
    """

    def __init__(self) -> None:
        self._registry: dict[str, IdempotencyRecord] = {}
        self._lock = RLock()

    def register(
        self,
        tool_name: str,
        args: dict[str, Any],
        status: str,
        output: Any,
        *,
        intent: str = "",
    ) -> IdempotencyRecord:
        """Record an observed tool result.

        On first call: stores the mapping.
        On repeated call: increments hit_count and verifies class consistency.

        Raises IdempotencyViolationError if result class changes for same request.
        """
        req_hash = canonical_request_hash(tool_name, args, intent)
        out_type = infer_output_type(output)
        res_class = canonical_result_class(tool_name, status, out_type)

        with self._lock:
            existing = self._registry.get(req_hash)
            if existing is None:
                record = IdempotencyRecord(
                    request_hash=req_hash,
                    result_class=res_class,
                    tool_name=tool_name,
                    intent=intent,
                    status=status,
                    output_type=out_type,
                    hit_count=1,
                )
                self._registry[req_hash] = record
                return record

            # Already registered — verify class consistency.
            if existing.result_class != res_class:
                raise IdempotencyViolationError(
                    f"Idempotency violation: request {req_hash[:16]}... "
                    f"stored class={existing.result_class}, new class={res_class}",
                    request_hash=req_hash,
                    stored_class=existing.result_class,
                    new_class=res_class,
                )

            # Same class — bump hit count.
            updated = IdempotencyRecord(
                request_hash=existing.request_hash,
                result_class=existing.result_class,
                tool_name=existing.tool_name,
                intent=existing.intent,
                status=existing.status,
                output_type=existing.output_type,
                hit_count=existing.hit_count + 1,
            )
            self._registry[req_hash] = updated
            return updated

    def is_cache_hit(
        self,
        tool_name: str,
        args: dict[str, Any],
        intent: str = "",
    ) -> bool:
        """True iff this exact request was seen before."""
        req_hash = canonical_request_hash(tool_name, args, intent)
        with self._lock:
            return req_hash in self._registry

    def get_record(
        self,
        tool_name: str,
        args: dict[str, Any],
        intent: str = "",
    ) -> IdempotencyRecord | None:
        req_hash = canonical_request_hash(tool_name, args, intent)
        with self._lock:
            return self._registry.get(req_hash)

    def get_class(
        self,
        tool_name: str,
        args: dict[str, Any],
        intent: str = "",
    ) -> str | None:
        record = self.get_record(tool_name, args, intent)
        return record.result_class if record else None

    def assert_idempotent(
        self,
        tool_name: str,
        args: dict[str, Any],
        new_status: str,
        new_output: Any,
        *,
        intent: str = "",
    ) -> None:
        """Assert that a new result is class-compatible with the stored record.

        Raises IdempotencyViolationError if class changed.
        If no record exists, silently passes (nothing stored yet).
        """
        req_hash = canonical_request_hash(tool_name, args, intent)
        out_type = infer_output_type(new_output)
        new_class = canonical_result_class(tool_name, new_status, out_type)

        with self._lock:
            existing = self._registry.get(req_hash)
            if existing is None:
                return
            if existing.result_class != new_class:
                raise IdempotencyViolationError(
                    f"Idempotency assertion failed: request {req_hash[:16]}... "
                    f"stored={existing.result_class}, new={new_class}",
                    request_hash=req_hash,
                    stored_class=existing.result_class,
                    new_class=new_class,
                )

    def registry_size(self) -> int:
        with self._lock:
            return len(self._registry)

    def all_records(self) -> list[IdempotencyRecord]:
        with self._lock:
            return list(self._registry.values())

    def idempotency_proof(
        self,
        tool_name: str,
        args: dict[str, Any],
        intent: str = "",
    ) -> dict[str, Any]:
        """Non-raising: return an idempotency proof for the given request."""
        req_hash = canonical_request_hash(tool_name, args, intent)
        with self._lock:
            record = self._registry.get(req_hash)
        if record is None:
            return {
                "ok": False,
                "request_hash": req_hash,
                "result_class": None,
                "hit_count": 0,
                "registered": False,
            }
        return {
            "ok": True,
            "request_hash": req_hash,
            "result_class": record.result_class,
            "hit_count": record.hit_count,
            "registered": True,
        }


__all__ = [
    "IdempotencyRecord",
    "IdempotencyViolationError",
    "ToolIdempotencyRegistry",
    "canonical_request_hash",
    "canonical_result_class",
    "infer_output_type",
]
