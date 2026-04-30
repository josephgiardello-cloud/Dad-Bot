"""system_audit.py — Pure post-hoc verification layer over immutable execution traces.

This module is a thin orchestrator:
  - Calls read-only check runners (audit_runners)
  - Records pass/fail results into a local RESULTS dict
  - Delegates formatted output to audit_reporter

Invariants:
  - NO handle_turn() calls
  - NO graph.execute() monkey-patching
  - NO scheduler instantiation
  - NO lease acquisition
  - NO worker execution
  - NO recovery triggers
  - ALL data sourced from completed execution traces (ledger, metrics, attributes)
"""
from __future__ import annotations

import traceback
from typing import Any

from audit_registry import AuditRegistry
from audit_reporter import print_results

# ---------------------------------------------------------------------------
# Audit result accumulator — local to this module, not shared state
# ---------------------------------------------------------------------------

RESULTS: dict[str, dict[str, Any]] = {}


def record(name: str, success: bool, detail: str = "", *, elapsed_s: float | None = None) -> None:
    RESULTS[name] = {
        "status": "PASS" if success else "FAIL",
        "detail": str(detail or "").strip(),
    }
    if elapsed_s is not None:
        RESULTS[name]["elapsed_s"] = round(float(elapsed_s), 4)


# ---------------------------------------------------------------------------
# Audit orchestration — reads from completed execution state only
# ---------------------------------------------------------------------------

def run_full_audit(orchestrator: Any) -> dict[str, dict[str, Any]]:
    """Run all post-hoc verification checks over the immutable execution trace.

    Requires an orchestrator that has already executed at least one turn so
    that ledger events, metrics counters, and trace state are populated.
    Returns the RESULTS accumulator.
    """
    try:
        registry = AuditRegistry(orchestrator)
        for entry in registry:
            result = entry.runner()
            elapsed = result.get("elapsed_s") if getattr(entry, "_has_elapsed", False) else None
            record(entry.name, result["ok"], entry.detail_fn(result), elapsed_s=elapsed)
    except Exception:
        record("audit_runtime", False, traceback.format_exc(limit=5))

    print_results(RESULTS)
    return RESULTS


# ---------------------------------------------------------------------------
# CLI entry — requires an already-exercised orchestrator for meaningful results
# ---------------------------------------------------------------------------

def _main() -> None:
    from dadbot.core.orchestrator import DadBotOrchestrator
    from dadbot.registry import ServiceRegistry

    registry = ServiceRegistry()
    orchestrator = DadBotOrchestrator(registry=registry, strict=False, enable_observability=True)
    run_full_audit(orchestrator)


if __name__ == "__main__":
    _main()
