"""Offline replay validator — verifies execution trace contracts deterministically.

Design contract
---------------
This module is **completely independent of the runtime execution system**.
It consumes only serialised execution trace artifacts (plain dicts / JSON) and
performs offline verification without importing from:
  - ``dadbot.core.graph``
  - ``dadbot.core.execution_kernel``
  - ``dadbot.core.execution_identity``  (identity fingerprint is re-computed
    locally from first principles, no shared runtime state)

Architectural role
------------------
::

    Offline validator (this module)
        ├── Input: execution_trace_contract  (dict from turn_context.metadata)
        ├── Input: execution_events          (list[dict] from execution_trace)
        ├── Input: execution_identity        (dict from turn_context.state)
        └── Output: ReplayValidationReport   (verdict + details, no side effects)

Usage
-----
::

    from dadbot.core.offline_replay_validator import OfflineReplayValidator

    validator = OfflineReplayValidator()

    # Verify execution trace contract (re-hash events → compare stored hash)
    report = validator.validate_trace_contract(
        contract=turn_context.metadata["execution_trace_contract"],
        events=turn_context.state.get("execution_trace", []),
    )

    # Verify identity fingerprint (re-build fingerprint → compare)
    id_report = validator.validate_execution_identity(
        identity=turn_context.state.get("execution_identity", {}),
    )

    assert report.passed, report.violations
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Internal helpers — no imports from runtime modules
# ---------------------------------------------------------------------------

def _json_safe_offline(value: Any) -> Any:
    """Serialisation-safe reducer — mirrors graph._json_safe without import."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value)}
    if isinstance(value, dict):
        return {str(k): _json_safe_offline(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_offline(item) for item in value]
    return repr(value)


def _recompute_trace_hash(events: list[dict[str, Any]], trace_id: str) -> str:
    """Re-compute the execution trace hash from raw events.

    Mirrors the canonicalisation in ``TurnGraph._finalize_execution_trace_contract``
    without importing that class.
    """
    canonical = {
        "trace_id": str(trace_id or ""),
        "events": [
            {
                "sequence": int(item.get("sequence", 0) or 0),
                "event_type": str(item.get("event_type", "") or ""),
                "stage": str(item.get("stage", "") or ""),
                "phase": str(item.get("phase", "") or ""),
                "detail": _json_safe_offline(item.get("detail") or {}),
            }
            for item in events
        ],
    }
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _recompute_identity_fingerprint(identity: dict[str, Any]) -> str:
    """Re-compute the execution identity fingerprint from a stored identity dict.

    Mirrors ``ExecutionIdentity.fingerprint`` without importing that class.
    """
    canonical = {
        "trace_id": str(identity.get("trace_id") or ""),
        "trace_hash": str(identity.get("trace_hash") or ""),
        "lock_hash": str(identity.get("lock_hash") or ""),
        "checkpoint_chain_hash": str(identity.get("checkpoint_chain_hash") or ""),
        "mutation_tx_count": int(identity.get("mutation_tx_count") or 0),
        "event_count": int(identity.get("event_count") or 0),
    }
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ReplayValidationReport:
    """Result of an offline replay verification pass."""

    passed: bool
    """True iff all verification checks succeeded."""

    verdict: str
    """Human-readable verdict: 'PASS' | 'FAIL'."""

    violations: list[str] = field(default_factory=list)
    """Ordered list of violation descriptions.  Empty on a clean pass."""

    checks_run: list[str] = field(default_factory=list)
    """Names of checks that were executed."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Diagnostic data (recomputed hashes, event counts, etc.)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "verdict": str(self.verdict),
            "violations": list(self.violations),
            "checks_run": list(self.checks_run),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class OfflineReplayValidator:
    """Deterministic, runtime-independent verifier for execution trace artifacts.

    All verification is offline — no I/O, no shared mutable state, no imports
    from the execution runtime.  Input and output are plain dicts.

    Invariants enforced
    -------------------
    1. **Trace hash integrity** — the stored ``trace_hash`` in the
       ``execution_trace_contract`` must match a fresh re-computation over the
       raw event list.
    2. **Event count consistency** — ``event_count`` in the contract must equal
       ``len(events)``.
    3. **Schema version presence** — the contract must carry a ``version`` field.
    4. **Identity fingerprint** — the stored ``fingerprint`` in the
       ``execution_identity`` dict must match a fresh re-computation over the
       identity's canonical fields.
    5. **Identity event count** — ``event_count`` in the identity must equal the
       number of events supplied (when both are provided together).
    6. **Phase ordering** — every event's ``phase`` field must follow the canonical
       PLAN → ACT → OBSERVE → RESPOND progression (monotone, no back-transitions
       within a single trace).
    """

    _CANONICAL_PHASE_ORDER: dict[str, int] = {
        "PLAN": 0,
        "ACT": 1,
        "OBSERVE": 2,
        "RESPOND": 3,
    }

    def validate_trace_contract(
        self,
        contract: dict[str, Any],
        events: list[dict[str, Any]],
        *,
        trace_id: str = "",
    ) -> ReplayValidationReport:
        """Verify an execution trace contract against its event list.

        Parameters
        ----------
        contract:
            The ``execution_trace_contract`` dict stored in turn metadata::

                {"version": "1.0", "event_count": N, "trace_hash": "<hex>"}

        events:
            The raw ``execution_trace`` list from turn state.  Each element is
            the dict produced by ``ExecutionTraceEvent.to_dict()``.

        trace_id:
            Optional trace identifier for better diagnostics.  If omitted it is
            read from the first event's ``trace_id`` field.

        Returns
        -------
        ReplayValidationReport
        """
        violations: list[str] = []
        checks: list[str] = []
        meta: dict[str, Any] = {}

        contract = dict(contract or {})
        events = list(events or [])

        # Derive trace_id from events if not supplied.
        if not trace_id and events:
            trace_id = str(events[0].get("trace_id") or "")

        # --- Check 1: schema version presence ---------------------------
        checks.append("schema_version_present")
        version = str(contract.get("version") or "").strip()
        meta["contract_version"] = version
        if not version:
            violations.append("execution_trace_contract missing 'version' field")

        # --- Check 2: event count consistency ---------------------------
        checks.append("event_count_consistent")
        stored_count = int(contract.get("event_count") or 0)
        actual_count = len(events)
        meta["stored_event_count"] = stored_count
        meta["actual_event_count"] = actual_count
        if stored_count != actual_count:
            violations.append(
                f"event_count mismatch: contract says {stored_count}, "
                f"actual event list has {actual_count}"
            )

        # --- Check 3: trace hash integrity ------------------------------
        checks.append("trace_hash_integrity")
        stored_hash = str(contract.get("trace_hash") or "").strip()
        recomputed_hash = _recompute_trace_hash(events, trace_id)
        meta["stored_trace_hash"] = stored_hash
        meta["recomputed_trace_hash"] = recomputed_hash
        if stored_hash != recomputed_hash:
            violations.append(
                f"trace_hash mismatch: stored={stored_hash!r}, "
                f"recomputed={recomputed_hash!r}"
            )

        # --- Check 4: phase ordering ------------------------------------
        checks.append("phase_ordering_monotone")
        phase_violations = self._check_phase_ordering(events)
        meta["phase_ordering_violations"] = phase_violations
        violations.extend(phase_violations)

        passed = len(violations) == 0
        return ReplayValidationReport(
            passed=passed,
            verdict="PASS" if passed else "FAIL",
            violations=violations,
            checks_run=checks,
            metadata=meta,
        )

    def validate_execution_identity(
        self,
        identity: dict[str, Any],
        *,
        events: list[dict[str, Any]] | None = None,
    ) -> ReplayValidationReport:
        """Verify an execution identity fingerprint offline.

        Parameters
        ----------
        identity:
            The ``execution_identity`` dict stored in turn state — produced by
            ``ExecutionIdentity.to_dict()``.

        events:
            Optional raw event list.  When supplied the validator also checks
            that ``identity["event_count"]`` matches ``len(events)``.

        Returns
        -------
        ReplayValidationReport
        """
        violations: list[str] = []
        checks: list[str] = []
        meta: dict[str, Any] = {}

        identity = dict(identity or {})

        # --- Check 1: fingerprint re-computation ------------------------
        checks.append("identity_fingerprint_integrity")
        stored_fp = str(identity.get("fingerprint") or "").strip()
        recomputed_fp = _recompute_identity_fingerprint(identity)
        meta["stored_fingerprint"] = stored_fp
        meta["recomputed_fingerprint"] = recomputed_fp
        if stored_fp != recomputed_fp:
            violations.append(
                f"identity fingerprint mismatch: stored={stored_fp!r}, "
                f"recomputed={recomputed_fp!r}"
            )

        # --- Check 2: required fields -----------------------------------
        checks.append("identity_required_fields")
        for field_name in ("trace_id", "trace_hash", "lock_hash"):
            if not str(identity.get(field_name) or "").strip():
                violations.append(f"execution_identity missing required field: {field_name!r}")

        # --- Check 3: event_count vs supplied events --------------------
        if events is not None:
            checks.append("identity_event_count_consistent")
            id_event_count = int(identity.get("event_count") or 0)
            actual = len(events)
            meta["identity_event_count"] = id_event_count
            meta["actual_event_count"] = actual
            if id_event_count != actual:
                violations.append(
                    f"identity event_count={id_event_count} does not match "
                    f"supplied event list length={actual}"
                )

        passed = len(violations) == 0
        return ReplayValidationReport(
            passed=passed,
            verdict="PASS" if passed else "FAIL",
            violations=violations,
            checks_run=checks,
            metadata=meta,
        )

    def validate_full(
        self,
        *,
        contract: dict[str, Any],
        events: list[dict[str, Any]],
        identity: dict[str, Any],
        trace_id: str = "",
    ) -> ReplayValidationReport:
        """Run all checks: trace contract + identity fingerprint together.

        Combines ``validate_trace_contract`` and ``validate_execution_identity``
        into a single report.  This is the recommended entry point for full
        offline verification of a persisted turn.
        """
        trace_report = self.validate_trace_contract(
            contract, events, trace_id=trace_id
        )
        identity_report = self.validate_execution_identity(identity, events=events)

        all_violations = trace_report.violations + identity_report.violations
        all_checks = trace_report.checks_run + identity_report.checks_run
        combined_meta = {**trace_report.metadata, **identity_report.metadata}

        passed = len(all_violations) == 0
        return ReplayValidationReport(
            passed=passed,
            verdict="PASS" if passed else "FAIL",
            violations=all_violations,
            checks_run=all_checks,
            metadata=combined_meta,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_phase_ordering(self, events: list[dict[str, Any]]) -> list[str]:
        """Return violation strings for any non-monotone phase transitions."""
        violations: list[str] = []
        max_phase_rank = -1
        max_phase_name = ""
        for event in events:
            phase = str(event.get("phase") or "").strip().upper()
            if not phase or phase not in self._CANONICAL_PHASE_ORDER:
                continue
            rank = self._CANONICAL_PHASE_ORDER[phase]
            if rank < max_phase_rank:
                violations.append(
                    f"phase back-transition detected: {phase!r} (rank {rank}) "
                    f"after {max_phase_name!r} (rank {max_phase_rank}) "
                    f"at event sequence={event.get('sequence', '?')}"
                )
            else:
                max_phase_rank = rank
                max_phase_name = phase
        return violations
