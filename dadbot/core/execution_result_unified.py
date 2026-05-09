from __future__ import annotations

import copy
from typing import Any


_UNIFIED_STATUS_VALUES = {"pending", "ok", "failed"}
_TERMINAL_STATUS_VALUES = {"ok", "failed"}


class ExecutionResultInvariantError(ValueError):
    """Raised when an execution_result envelope violates a semantic invariant."""


class _FrozenExecutionResult(dict):  # type: ignore[type-arg]
    """A dict subclass that refuses writes once a terminal status has been set.

    * ``isinstance(x, dict)`` → True — existing callers that pass dicts through
      without modification continue to work.
    * ``dict(frozen)`` → plain mutable copy — serialisation / storage is unaffected.
    * Any ``__setitem__`` / ``__delitem__`` raises ``ExecutionResultInvariantError``
      instead of silently corrupting the post-terminal envelope.
    """

    def __setitem__(self, key: Any, value: Any) -> None:
        raise ExecutionResultInvariantError(
            f"execution_result is frozen (status='{self.get('status')}'); "
            f"attempted to write [{key!r}]. "
            "Use mark_unified_execution_success / mark_unified_execution_failure "
            "or build_unified_execution_result for a new envelope."
        )

    def __delitem__(self, key: Any) -> None:
        raise ExecutionResultInvariantError(
            f"execution_result is frozen (status='{self.get('status')}'); "
            f"attempted to delete [{key!r}]."
        )

    def __copy__(self) -> _FrozenExecutionResult:
        new: _FrozenExecutionResult = _FrozenExecutionResult.__new__(_FrozenExecutionResult)
        dict.update(new, self)
        return new

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenExecutionResult:
        new: _FrozenExecutionResult = _FrozenExecutionResult.__new__(_FrozenExecutionResult)
        memo[id(self)] = new
        for k, v in self.items():
            dict.__setitem__(new, copy.deepcopy(k, memo), copy.deepcopy(v, memo))
        return new


def assert_execution_result_invariants(
    execution_result: dict[str, Any],
    *,
    context: str = "",
) -> None:
    """Assert all four semantic invariants of the unified execution_result envelope.

    Invariants
    ----------
    1. success ⇒ all failure fields are empty.
    2. failure ⇒ outputs dict is present (may be partial).
    3. timeout ⇒ failure.class == "timeout"  OR  timeout.timed_out == True.
    4. degradation is always present and well-typed (even if empty).
    """
    tag = f" [{context}]" if context else ""
    status = str(execution_result.get("status") or "").lower()

    # ── Invariant 4 ── degradation must always be present ─────────────────────
    degradation = execution_result.get("degradation")
    if not isinstance(degradation, dict):
        raise ExecutionResultInvariantError(
            f"Invariant 4 violated{tag}: degradation must be a dict, got {type(degradation).__name__}"
        )
    if "items" not in degradation:
        raise ExecutionResultInvariantError(
            f"Invariant 4 violated{tag}: degradation.items is missing"
        )
    if not isinstance(degradation["items"], list):
        raise ExecutionResultInvariantError(
            f"Invariant 4 violated{tag}: degradation.items must be a list"
        )

    # ── Invariant 1 ── success ⇒ failure fields empty ─────────────────────────
    if status == "ok":
        failure = dict(execution_result.get("failure") or {})
        dirty_fields = [
            k for k in ("class", "type", "message", "source")
            if str(failure.get(k) or "").strip()
        ]
        if dirty_fields:
            raise ExecutionResultInvariantError(
                f"Invariant 1 violated{tag}: status='ok' but failure fields are non-empty: {dirty_fields}"
            )

    # ── Invariant 2 ── failure ⇒ outputs dict must exist ──────────────────────
    if status == "failed":
        outputs = execution_result.get("outputs")
        if not isinstance(outputs, dict):
            raise ExecutionResultInvariantError(
                f"Invariant 2 violated{tag}: status='failed' but outputs is absent or not a dict"
            )
        # failure.class must be non-empty so callers know what failed
        failure = dict(execution_result.get("failure") or {})
        if not str(failure.get("class") or "").strip():
            raise ExecutionResultInvariantError(
                f"Invariant 2 violated{tag}: status='failed' but failure.class is empty"
            )

    # ── Invariant 3 ── timeout ⇒ failure.class == "timeout" OR timed_out ──────
    timeout = dict(execution_result.get("timeout") or {})
    timed_out = bool(timeout.get("timed_out", False))
    failure = dict(execution_result.get("failure") or {})
    failure_class = str(failure.get("class") or "").strip().lower()
    if timed_out and failure_class not in {"timeout", ""}:
        raise ExecutionResultInvariantError(
            f"Invariant 3 violated{tag}: timeout.timed_out=True but failure.class='{failure_class}' (must be 'timeout')"
        )
    if status == "failed" and failure_class == "timeout" and not timed_out:
        raise ExecutionResultInvariantError(
            f"Invariant 3 violated{tag}: failure.class='timeout' but timeout.timed_out is False"
        )


def build_unified_execution_result(
    *,
    timeout_seconds: float = 0.0,
    degradation_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    items = [dict(item) for item in list(degradation_items or []) if isinstance(item, dict)]
    return {
        "status": "pending",
        "degradation": {
            "count": int(len(items)),
            "items": items,
        },
        "failure": {
            "class": "",
            "type": "",
            "message": "",
            "source": "",
            "retryable": False,
        },
        "timeout": {
            "seconds": max(0.0, float(timeout_seconds or 0.0)),
            "timed_out": False,
        },
        "outputs": {
            "response": "",
            "should_end": False,
            "semantic_eval_input_hash": "",
        },
    }


def ensure_unified_execution_result(payload: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(payload or {})
    baseline = build_unified_execution_result()

    status = str(raw.get("status") or "pending").strip().lower()
    baseline["status"] = status if status in _UNIFIED_STATUS_VALUES else "pending"

    degradation = raw.get("degradation")
    if isinstance(degradation, dict):
        items = degradation.get("items")
        if isinstance(items, list):
            safe_items = [dict(item) for item in items if isinstance(item, dict)]
            baseline["degradation"]["items"] = safe_items
            baseline["degradation"]["count"] = int(len(safe_items))
        else:
            baseline["degradation"]["count"] = int(degradation.get("count") or 0)
    else:
        fallback_items = raw.get("turn_ir_degradations") or raw.get("ir_degradations")
        if isinstance(fallback_items, list):
            safe_items = [dict(item) for item in fallback_items if isinstance(item, dict)]
            baseline["degradation"]["items"] = safe_items
            baseline["degradation"]["count"] = int(len(safe_items))

    failure = raw.get("failure")
    if isinstance(failure, dict):
        baseline["failure"] = {
            "class": str(failure.get("class") or ""),
            "type": str(failure.get("type") or ""),
            "message": str(failure.get("message") or ""),
            "source": str(failure.get("source") or ""),
            "retryable": bool(failure.get("retryable", False)),
        }

    timeout = raw.get("timeout")
    if isinstance(timeout, dict):
        baseline["timeout"] = {
            "seconds": max(0.0, float(timeout.get("seconds") or 0.0)),
            "timed_out": bool(timeout.get("timed_out", False)),
        }

    outputs = raw.get("outputs")
    if isinstance(outputs, dict):
        baseline["outputs"] = {
            "response": str(outputs.get("response") or ""),
            "should_end": bool(outputs.get("should_end", False)),
            "semantic_eval_input_hash": str(outputs.get("semantic_eval_input_hash") or ""),
        }

    # Preserve legacy fields used by older planner/reply surfaces.
    for key in ("initial_result", "result", "candidates"):
        if key in raw:
            baseline[key] = raw[key]

    assert_execution_result_invariants(baseline, context="ensure_unified_execution_result")
    return baseline


def get_unified_execution_result(turn_context: Any) -> dict[str, Any]:
    if turn_context is None:
        return ensure_unified_execution_result(None)

    attr_value = getattr(turn_context, "execution_result", None)
    if isinstance(attr_value, dict):
        return ensure_unified_execution_result(attr_value)

    metadata = getattr(turn_context, "metadata", None)
    if isinstance(metadata, dict) and isinstance(metadata.get("execution_result"), dict):
        return ensure_unified_execution_result(dict(metadata.get("execution_result") or {}))

    state = getattr(turn_context, "state", None)
    if isinstance(state, dict) and isinstance(state.get("execution_result"), dict):
        return ensure_unified_execution_result(dict(state.get("execution_result") or {}))

    if isinstance(turn_context, dict):
        value = turn_context.get("execution_result")
        if isinstance(value, dict):
            return ensure_unified_execution_result(value)

    return ensure_unified_execution_result(None)


def set_unified_execution_result(turn_context: Any, execution_result: dict[str, Any]) -> None:
    normalized = ensure_unified_execution_result(execution_result)

    if turn_context is None:
        return

    if isinstance(turn_context, dict):
        turn_context["execution_result"] = dict(normalized)
        return

    setattr(turn_context, "execution_result", dict(normalized))

    metadata = getattr(turn_context, "metadata", None)
    if isinstance(metadata, dict):
        metadata["execution_result"] = dict(normalized)

    state = getattr(turn_context, "state", None)
    if isinstance(state, dict):
        state["execution_result"] = dict(normalized)


def mark_unified_execution_success(
    execution_result: dict[str, Any],
    *,
    response: str,
    should_end: bool,
) -> dict[str, Any]:
    _current_status = str((dict(execution_result) if execution_result else {}).get("status") or "").lower()
    if _current_status in _TERMINAL_STATUS_VALUES:
        raise ExecutionResultInvariantError(
            f"Cannot re-mark a terminal execution_result (status='{_current_status}') as success. "
            "Build a new envelope with build_unified_execution_result() instead."
        )
    normalized = ensure_unified_execution_result(execution_result)
    normalized["status"] = "ok"
    normalized["outputs"]["response"] = str(response or "")
    normalized["outputs"]["should_end"] = bool(should_end)
    normalized["failure"] = {
        "class": "",
        "type": "",
        "message": "",
        "source": "",
        "retryable": False,
    }
    assert_execution_result_invariants(normalized, context="mark_unified_execution_success")
    return _FrozenExecutionResult(normalized)


def mark_unified_execution_failure(
    execution_result: dict[str, Any],
    *,
    failure_class: str,
    failure_source: str,
    retryable: bool,
    exception_type: str,
    message: str,
) -> dict[str, Any]:
    _current_status = str((dict(execution_result) if execution_result else {}).get("status") or "").lower()
    if _current_status in _TERMINAL_STATUS_VALUES:
        raise ExecutionResultInvariantError(
            f"Cannot re-mark a terminal execution_result (status='{_current_status}') as failure. "
            "Build a new envelope with build_unified_execution_result() instead."
        )
    normalized = ensure_unified_execution_result(execution_result)
    normalized["status"] = "failed"
    normalized["failure"] = {
        "class": str(failure_class or ""),
        "type": str(exception_type or ""),
        "message": str(message or ""),
        "source": str(failure_source or ""),
        "retryable": bool(retryable),
    }
    if str(failure_class or "").strip().lower() == "timeout":
        normalized["timeout"]["timed_out"] = True
    assert_execution_result_invariants(normalized, context="mark_unified_execution_failure")
    return _FrozenExecutionResult(normalized)


def set_unified_execution_eval_hash(
    execution_result: dict[str, Any],
    *,
    eval_input_hash: str,
) -> dict[str, Any]:
    normalized = ensure_unified_execution_result(execution_result)
    normalized["outputs"]["semantic_eval_input_hash"] = str(eval_input_hash or "")
    return normalized
