from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dadbot.core.execution_firewall import ExecutionFirewall
from dadbot.core.execution_kernel import ExecutionKernel
from dadbot.core.graph import TurnContext
from dadbot.core.invariant_registry import InvariantRegistry
from tools.phase4_legacy_integrity_scan import build_quarantine_registry, run_scan


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 4 execution-kernel firewall gate")
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--strict", action="store_true", help="Enable hard enforcement in kernel")
    parser.add_argument(
        "--scan-output",
        default="session_logs/phase4_legacy_integrity_scan.json",
        help="Path to write static scan report",
    )
    parser.add_argument(
        "--quarantine-output",
        default="runtime/phase4_quarantine_registry.json",
        help="Path to write quarantine registry",
    )
    parser.add_argument(
        "--summary-output",
        default="session_logs/phase4_firewall_gate_summary.json",
        help="Path to write gate summary",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()

    scan_report = run_scan(root)

    scan_output = Path(args.scan_output)
    if not scan_output.is_absolute():
        scan_output = root / scan_output
    scan_output.parent.mkdir(parents=True, exist_ok=True)
    scan_output.write_text(json.dumps(scan_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    quarantine_payload = build_quarantine_registry(scan_report)
    quarantine_output = Path(args.quarantine_output)
    if not quarantine_output.is_absolute():
        quarantine_output = root / quarantine_output
    quarantine_output.parent.mkdir(parents=True, exist_ok=True)
    quarantine_output.write_text(json.dumps(quarantine_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    firewall = ExecutionFirewall(quarantine_path=quarantine_output)
    registry = InvariantRegistry()
    kernel = ExecutionKernel(
        firewall=firewall, invariant_registry=registry, quarantine=quarantine_payload, strict=bool(args.strict)
    )

    context = TurnContext(user_input="phase4 kernel gate self-check")

    try:
        self_check = kernel.validate(
            stage="cli_gate",
            operation="phase4_firewall_gate.self_check",
            context=context,
        )
    except Exception as exc:
        self_check = type("_KernelResult", (), {"ok": False, "reason": str(exc)})()

    blocked_legacy_reason = ""
    if kernel.strict:
        try:
            kernel.validate(
                stage="cli_gate",
                operation="handle_graph_failure",
                context=context,
            )
            legacy_blocked_ok = False
        except Exception as exc:
            legacy_blocked_ok = True
            blocked_legacy_reason = str(exc)
    else:
        blocked_legacy = kernel.validate(
            stage="cli_gate",
            operation="handle_graph_failure",
            context=context,
        )
        legacy_blocked_ok = not blocked_legacy.ok
        blocked_legacy_reason = str(getattr(blocked_legacy, "reason", "") or "")

    status = "PASS"
    if scan_report.get("overall_integrity") != "PASS":
        status = "FAIL"
    if not self_check.ok:
        status = "FAIL"
    if not legacy_blocked_ok:
        status = "FAIL"

    summary = {
        "phase4_firewall_gate": status,
        "strict_mode": bool(kernel.strict),
        "scan_overall_integrity": scan_report.get("overall_integrity", "FAIL"),
        "self_check": {
            "ok": bool(self_check.ok),
            "reason": str(getattr(self_check, "reason", "") or ""),
        },
        "legacy_block_check": {
            "ok": bool(legacy_blocked_ok),
            "reason": blocked_legacy_reason,
        },
        "scan_output": str(scan_output),
        "quarantine_output": str(quarantine_output),
    }

    summary_output = Path(args.summary_output)
    if not summary_output.is_absolute():
        summary_output = root / summary_output
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"PHASE4_FIREWALL_GATE: {status}")
    print(f"SUMMARY: {summary_output}")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
