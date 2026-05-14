from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from dadbot.core.contract_evaluator import CAPABILITY_CONTRACTS

CapabilityStatus = Literal["pass", "fail", "not_evaluated"]
CAPABILITY_AUDIT_EVENT_TYPE = "CAPABILITY_AUDIT_EVENT"
CAPABILITY_AUDIT_VERSION = "v1"


@dataclass
class CapabilityCheck:
    name: str
    status: CapabilityStatus
    contract: dict[str, Any]
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "contract": dict(self.contract),
            "details": dict(self.details),
        }


@dataclass
class CapabilityAuditReport:
    trace_id: str
    audit_mode: bool
    failed: bool
    stage_order: list[str]
    mutation_queue: dict[str, Any]
    checks: list[CapabilityCheck]
    error: str = ""

    @property
    def ok(self) -> bool:
        return (not self.failed) and all(check.status != "fail" for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "audit_mode": self.audit_mode,
            "failed": self.failed,
            "ok": self.ok,
            "error": self.error,
            "stage_order": list(self.stage_order),
            "mutation_queue": dict(self.mutation_queue),
            "checks": [check.to_dict() for check in self.checks],
        }


def build_runtime_capability_audit_report(
    turn_context: Any,
    *,
    stage_order: list[str],
    failed: bool,
    error: str = "",
) -> CapabilityAuditReport:
    normalized_stage_order = [str(stage or "").strip().lower() for stage in list(stage_order or [])]
    mutation_queue = dict(turn_context.mutation_queue.snapshot())
    save_count = normalized_stage_order.count("save")
    checks = [
        CapabilityCheck(
            name="temporal_ordering",
            status=(
                "pass"
                if normalized_stage_order[:1] == ["temporal"]
                and all(
                    stage in normalized_stage_order
                    for stage in CAPABILITY_CONTRACTS["temporal_ordering"]["required_stages"]
                )
                else "fail"
            ),
            contract=CAPABILITY_CONTRACTS["temporal_ordering"],
            details={
                "required_stages": list(
                    CAPABILITY_CONTRACTS["temporal_ordering"]["required_stages"],
                ),
                "observed_stage_order": list(normalized_stage_order),
            },
        ),
        CapabilityCheck(
            name="mutation_safety",
            status=(
                "pass"
                if int(mutation_queue.get("pending") or 0) == 0 and int(mutation_queue.get("failed") or 0) == 0
                else "fail"
            ),
            contract=CAPABILITY_CONTRACTS["mutation_safety"],
            details={
                "mutation_queue": dict(mutation_queue),
                "empty_after_save": int(mutation_queue.get("pending") or 0) == 0,
            },
        ),
        CapabilityCheck(
            name="deterministic_replay",
            status="not_evaluated",
            contract=CAPABILITY_CONTRACTS["deterministic_replay"],
            details={
                "reason": "single-turn runtime audit cannot prove cross-run hash equivalence",
            },
        ),
        CapabilityCheck(
            name="save_node_single_execution",
            status="pass" if save_count == 1 else "fail",
            contract=CAPABILITY_CONTRACTS["save_node_single_execution"],
            details={
                "save_count": save_count,
            },
        ),
        CapabilityCheck(
            name="capability_audit_emission",
            status="pass",
            contract=CAPABILITY_CONTRACTS["capability_audit_emission"],
            details={
                "report_emitted": True,
            },
        ),
    ]
    return CapabilityAuditReport(
        trace_id=str(getattr(turn_context, "trace_id", "") or ""),
        audit_mode=True,
        failed=bool(failed),
        error=str(error or ""),
        stage_order=normalized_stage_order,
        mutation_queue=mutation_queue,
        checks=checks,
    )


def build_capability_coverage_matrix(
    scenario_reports: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    matrix: dict[str, dict[str, Any]] = {}
    for capability_name, contract in CAPABILITY_CONTRACTS.items():
        evaluated = 0
        passed = 0
        violations = 0
        for scenario in list(scenario_reports or []):
            checks = list(scenario.get("audit_report", {}).get("checks") or [])
            match = next(
                (check for check in checks if str(check.get("name") or "") == capability_name),
                None,
            )
            if match is None:
                continue
            status = str(match.get("status") or "")
            if status != "not_evaluated":
                evaluated += 1
            if status == "pass":
                passed += 1
            elif status == "fail":
                violations += 1
        coverage = 100.0 if evaluated == 0 else round((passed / evaluated) * 100.0, 2)
        matrix[capability_name] = {
            "contract": dict(contract),
            "test_coverage": coverage,
            "runtime_enforcement": bool(contract.get("runtime_enforcement", False)),
            "violations_detected": violations,
            "last_pass": str(date.today()) if violations == 0 else "",
        }
    return matrix


def build_capability_audit_event_payload(
    report: CapabilityAuditReport,
    *,
    scenario: str,
) -> dict[str, Any]:
    checks = {check.name: check for check in list(report.checks or [])}
    temporal_ordering_ok = (
        checks.get("temporal_ordering").status == "pass" if checks.get("temporal_ordering") else False
    )
    mutation_safety_ok = checks.get("mutation_safety").status == "pass" if checks.get("mutation_safety") else False
    save_ok = (
        checks.get("save_node_single_execution").status == "pass" if checks.get("save_node_single_execution") else False
    )

    return {
        "audit_version": CAPABILITY_AUDIT_VERSION,
        "scenario": str(scenario or "runtime_turn"),
        "result": "ok" if report.ok else "fail",
        "metrics": {
            "temporal_violation": not temporal_ordering_ok,
            "mutation_leak": not mutation_safety_ok,
            "save_node_compliance": bool(save_ok),
        },
        # Explicitly non-canonical operational field.
        "timestamp": None,
    }
