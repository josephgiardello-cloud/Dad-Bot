"""audit_registry.py — Check registration and dispatch.

Decouples which checks run from how they run.
system_audit.py iterates this registry; no check selection logic lives there.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from audit_runners import (
    build_wiring_map,
    check_causality,
    check_exporter,
    check_ledger,
    check_mid_execution,
    check_observability,
    check_retry,
    check_runtime_activation,
    check_wiring_integrity,
    full_module_activation_scan,
)

# ---------------------------------------------------------------------------
# CheckResult: name + runner callable + how to build the record entry
# ---------------------------------------------------------------------------


class CheckEntry:
    """Binds a check name to its runner and result-to-record translation."""

    def __init__(
        self,
        name: str,
        runner: Callable[..., dict[str, Any]],
        detail_fn: Callable[[dict[str, Any]], str],
    ) -> None:
        self.name = name
        self.runner = runner
        self.detail_fn = detail_fn


def _ledger_entry(wiring: dict[str, Any]) -> CheckEntry:
    def run() -> dict[str, Any]:
        return check_ledger(wiring["ledger"])

    def detail(r: dict[str, Any]) -> str:
        return f"events={r.get('events')}"

    entry = CheckEntry("ledger_sequence", run, detail)
    entry._elapsed = True
    return entry


def _build_entries(orchestrator: Any) -> list[CheckEntry]:
    """Build the ordered list of check entries for a given orchestrator."""
    wiring = build_wiring_map(orchestrator)

    entries: list[tuple[str, Callable[[], dict[str, Any]], Callable[[dict[str, Any]], str], bool]] = [
        (
            "ledger_sequence",
            lambda: check_ledger(wiring["ledger"]),
            lambda r: f"events={r.get('events')}",
            True,
        ),
        (
            "wiring_integrity",
            lambda: check_wiring_integrity(orchestrator),
            lambda r: f"missing={r.get('missing_components')}",
            False,
        ),
        (
            "runtime_activation",
            lambda: check_runtime_activation(wiring["ledger"], wiring["metrics"]),
            lambda r: f"flows={r.get('details')}, metrics_active={r.get('metrics_active')}",
            False,
        ),
        (
            "causality",
            lambda: check_causality(orchestrator),
            lambda r: (
                f"orphans={r.get('orphan_jobs')}, "
                f"incomplete={r.get('incomplete_jobs')}, "
                f"missing_trace={r.get('missing_trace_jobs')}"
            ),
            False,
        ),
        (
            "observability_metrics",
            lambda: check_observability(wiring["metrics"]),
            lambda r: f"completed={r.get('completed')}, failed={r.get('failed')}",
            True,
        ),
        (
            "mid_execution_durability",
            lambda: check_mid_execution(orchestrator),
            lambda r: (
                f"save_graph_checkpoint={r.get('save_graph_checkpoint')}, save_turn_event={r.get('save_turn_event')}"
            ),
            True,
        ),
        (
            "retry_backoff",
            lambda: check_retry(orchestrator),
            lambda r: (
                f"orchestrator_retry_policy={r.get('orchestrator_retry_policy')}, "
                f"fault_injection_retry={r.get('fault_injection_retry')}"
            ),
            True,
        ),
        (
            "external_export",
            lambda: check_exporter(wiring["exporter"]),
            lambda r: f"enabled={r.get('enabled')}",
            True,
        ),
        (
            "module_activation_scan",
            lambda: full_module_activation_scan(orchestrator),
            lambda r: (
                f"missing_runtime={r.get('missing_runtime_components')}, "
                f"missing_aliases={r.get('missing_declared_aliases')}, "
                f"unused_imports={r.get('unused_imports')}"
            ),
            False,
        ),
    ]

    result = []
    for name, runner, detail_fn, has_elapsed in entries:
        entry = CheckEntry(name, runner, detail_fn)
        entry._has_elapsed = has_elapsed  # type: ignore[attr-defined]
        result.append(entry)
    return result


class AuditRegistry:
    """Holds the ordered set of checks to run for a given orchestrator."""

    def __init__(self, orchestrator: Any) -> None:
        self._entries = _build_entries(orchestrator)

    def __iter__(self):  # type: ignore[override]
        return iter(self._entries)

    def names(self) -> list[str]:
        return [e.name for e in self._entries]
