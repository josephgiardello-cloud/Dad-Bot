"""System-level health scorer that composes multiple signal sources.

This aggregates:
  - Invariant violation counts (from InvariantGate.violations_observed)
  - Reconciliation queue convergence stats
  - Persistence telemetry SLO pass/fail
  - Tool execution error rate

All signal sources are optional — the scorer degrades gracefully when
a source is unavailable (e.g., control plane not yet started).

Typical usage
-------------
    from dadbot.core.system_health_scorer import SystemHealthScorer, score_from_gate
    gate = InvariantGate()
    report = score_from_gate(gate)
    print(report.overall_score, report.failure_modes)

Or supply all sources:
    scorer = SystemHealthScorer()
    report = scorer.score(
        invariant_gate=gate,
        reconcile_metrics={"converged": 100, "attempted": 100, "remaining": 0},
        persistence_telemetry={"slo_ok": {"write_p95": True}},
        tool_execution_stats={"total": 50, "errors": 1},
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class SystemHealthReport:
    """Snapshot of system-wide health at a point in time."""

    overall_score: int               # 0–100; 100 = fully healthy
    invariant_violations: int        # lifetime count from InvariantGate
    failure_modes: list[str]         # classified labels for active degradations
    is_healthy: bool                 # True when overall_score >= healthy_threshold
    signals: dict[str, Any]          # raw signal values for diagnostic output
    captured_at: str                 # ISO 8601 timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "invariant_violations": self.invariant_violations,
            "failure_modes": list(self.failure_modes),
            "is_healthy": self.is_healthy,
            "signals": dict(self.signals),
            "captured_at": self.captured_at,
        }


# Penalty weights applied to overall_score (deduct from 100).
_WEIGHT_INVARIANT_VIOLATION = 15   # per violation, capped at 60
_WEIGHT_RECONCILE_MISS = 20        # applied when convergence < 100%
_WEIGHT_PERSISTENCE_SLO_MISS = 15  # applied per missed SLO dimension
_WEIGHT_TOOL_ERROR = 10            # applied when tool error rate > 5%
_WEIGHT_TOOL_FAILURE_CLASS = 8     # applied when timeout/corruption classes emerge
_WEIGHT_TOOL_LATENCY_SPIKE = 8     # applied when p95 latency exceeds threshold

_TOOL_FAILURE_CLASS_CRITICAL = frozenset({"timeout", "dns", "connection", "server", "corrupted_payload"})
_TOOL_P95_LATENCY_SPIKE_MS = 2_500.0

_HEALTHY_THRESHOLD = 70


class SystemHealthScorer:
    """Compose heterogeneous signal sources into a single SystemHealthReport."""

    def __init__(self, *, healthy_threshold: int = _HEALTHY_THRESHOLD) -> None:
        self._healthy_threshold = max(0, min(100, int(healthy_threshold)))

    def score(
        self,
        *,
        invariant_gate: Any = None,
        reconcile_metrics: dict[str, Any] | None = None,
        persistence_telemetry: dict[str, Any] | None = None,
        tool_execution_stats: dict[str, Any] | None = None,
    ) -> SystemHealthReport:
        """Compute a SystemHealthReport from any combination of available signals.

        Parameters
        ----------
        invariant_gate:
            An InvariantGate instance (or any object with .violations_observed int).
        reconcile_metrics:
            Dict with keys: converged (int), attempted (int), remaining (int).
            All optional — missing keys are treated as 0.
        persistence_telemetry:
            Dict from ConversationPersistence.persistence_telemetry_snapshot().
            Checked for slo_ok.write_p95 / slo_ok.compaction_p95.
        tool_execution_stats:
            Dict with keys: total (int), errors (int).
        """
        deductions = 0
        failure_modes: list[str] = []
        signals: dict[str, Any] = {}

        # ── Invariant violations ─────────────────────────────────────────────
        violations = 0
        if invariant_gate is not None:
            violations = max(0, int(getattr(invariant_gate, "violations_observed", 0) or 0))
        signals["invariant_violations"] = violations
        if violations > 0:
            penalty = min(60, violations * _WEIGHT_INVARIANT_VIOLATION)
            deductions += penalty
            failure_modes.append(f"invariant_violation(count={violations})")

        # ── Reconciliation convergence ────────────────────────────────────────
        recon = dict(reconcile_metrics or {})
        recon_attempted = max(0, int(recon.get("attempted") or 0))
        recon_converged = max(0, int(recon.get("converged") or 0))
        recon_remaining = max(0, int(recon.get("remaining") or 0))
        signals["reconcile_attempted"] = recon_attempted
        signals["reconcile_converged"] = recon_converged
        signals["reconcile_remaining"] = recon_remaining
        if recon_attempted > 0 and recon_converged < recon_attempted:
            convergence_rate = recon_converged / recon_attempted
            signals["reconcile_convergence_rate"] = round(convergence_rate, 3)
            deductions += _WEIGHT_RECONCILE_MISS
            failure_modes.append(
                f"reconciliation_lag(rate={round(convergence_rate, 2)},remaining={recon_remaining})"
            )
        elif recon_remaining > 0:
            signals["reconcile_convergence_rate"] = None
            deductions += _WEIGHT_RECONCILE_MISS // 2
            failure_modes.append(f"reconciliation_backlog(remaining={recon_remaining})")
        else:
            signals["reconcile_convergence_rate"] = 1.0 if recon_attempted > 0 else None

        # ── Persistence SLO ───────────────────────────────────────────────────
        p_telem = dict(persistence_telemetry or {})
        slo_ok = dict(p_telem.get("slo_ok") or {})
        signals["persistence_write_p95_ok"] = bool(slo_ok.get("write_p95", True))
        signals["persistence_compaction_p95_ok"] = bool(slo_ok.get("compaction_p95", True))
        if not signals["persistence_write_p95_ok"]:
            deductions += _WEIGHT_PERSISTENCE_SLO_MISS
            failure_modes.append("persistence_write_p95_slo_miss")
        if not signals["persistence_compaction_p95_ok"]:
            deductions += _WEIGHT_PERSISTENCE_SLO_MISS
            failure_modes.append("persistence_compaction_p95_slo_miss")

        # ── Tool execution error rate ─────────────────────────────────────────
        t_stats = dict(tool_execution_stats or {})
        tool_total = max(0, int(t_stats.get("total") or 0))
        tool_errors = max(0, int(t_stats.get("errors") or 0))
        tool_p95_latency_ms = float(t_stats.get("p95_latency_ms") or 0.0)
        failure_classes = dict(t_stats.get("failure_classes") or {})
        signals["tool_total"] = tool_total
        signals["tool_errors"] = tool_errors
        signals["tool_p95_latency_ms"] = round(tool_p95_latency_ms, 3) if tool_total > 0 else None
        signals["tool_failure_classes"] = {
            str(k): max(0, int(v or 0)) for k, v in failure_classes.items()
        }
        if tool_total > 0:
            error_rate = tool_errors / tool_total
            signals["tool_error_rate"] = round(error_rate, 3)
            if error_rate > 0.05:
                deductions += _WEIGHT_TOOL_ERROR
                failure_modes.append(f"tool_error_rate_elevated(rate={round(error_rate, 2)})")

            if tool_p95_latency_ms > _TOOL_P95_LATENCY_SPIKE_MS:
                deductions += _WEIGHT_TOOL_LATENCY_SPIKE
                failure_modes.append(
                    f"tool_latency_spike(p95_ms={round(tool_p95_latency_ms, 1)})"
                )

            critical_failure_count = 0
            for key, value in signals["tool_failure_classes"].items():
                if str(key).strip().lower() in _TOOL_FAILURE_CLASS_CRITICAL:
                    critical_failure_count += max(0, int(value))
            if critical_failure_count > 0:
                deductions += _WEIGHT_TOOL_FAILURE_CLASS
                failure_modes.append(
                    f"tool_failure_classes_elevated(count={critical_failure_count})"
                )
        else:
            signals["tool_error_rate"] = None

        overall = max(0, min(100, 100 - deductions))
        return SystemHealthReport(
            overall_score=overall,
            invariant_violations=violations,
            failure_modes=failure_modes,
            is_healthy=overall >= self._healthy_threshold,
            signals=signals,
            captured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )


# Module-level convenience
def score_from_gate(gate: Any, **kwargs: Any) -> SystemHealthReport:
    """Shortcut: score using only an InvariantGate instance."""
    return SystemHealthScorer().score(invariant_gate=gate, **kwargs)
