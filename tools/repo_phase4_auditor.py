from __future__ import annotations

import ast
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _required_files() -> list[Path]:
    return [
        ROOT / "dadbot" / "core" / "execution_kernel.py",
        ROOT / "dadbot" / "core" / "capability_registry.py",
        ROOT / "dadbot" / "core" / "execution_receipt.py",
        ROOT / "dadbot" / "core" / "observability.py",
        ROOT / "dadbot" / "core" / "otel_bridge.py",
        ROOT / "dadbot" / "core" / "prometheus_bridge.py",
        ROOT / "tools" / "phase4_certify.py",
    ]


def _structural_checks() -> dict[str, Any]:
    missing = [str(path.relative_to(ROOT)) for path in _required_files() if not path.exists()]

    core_files = sorted((ROOT / "dadbot" / "core").glob("*.py"))
    duplicate_stems = sorted({p.stem for p in core_files if sum(1 for q in core_files if q.stem == p.stem) > 1})

    # Orphan heuristic: python module that is not imported anywhere else.
    all_py = [p for p in ROOT.rglob("*.py") if ".venv" not in p.parts and "__pycache__" not in p.parts]
    corpus = "\n".join(p.read_text(encoding="utf-8-sig", errors="replace") for p in all_py)
    orphans: list[str] = []
    for module in core_files:
        stem = module.stem
        marker = f".{stem}"
        if marker not in corpus:
            orphans.append(str(module.relative_to(ROOT)))

    return {
        "missing_files": missing,
        "duplicate_responsibility_clusters": duplicate_stems,
        "orphan_modules": orphans,
        "ok": not missing and not duplicate_stems,
    }


def _architectural_checks() -> dict[str, Any]:
    violations: list[str] = []
    core_files = [p for p in (ROOT / "dadbot" / "core").glob("*.py")]

    forbidden_prefixes = (
        "dadbot.ui",
        "dadbot.static",
    )

    for path in core_files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
        except SyntaxError as exc:
            violations.append(f"{path.relative_to(ROOT)}: syntax error {exc}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = str(alias.name or "")
                    if name.startswith(forbidden_prefixes):
                        violations.append(f"{path.relative_to(ROOT)} imports forbidden {name}")
            elif isinstance(node, ast.ImportFrom):
                mod = str(node.module or "")
                if mod.startswith(forbidden_prefixes):
                    violations.append(f"{path.relative_to(ROOT)} imports forbidden {mod}")

    return {
        "ok": not violations,
        "violations": violations,
    }


async def _runtime_checks() -> dict[str, Any]:
    from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
    from dadbot.core.replay_verifier import ReplayVerifier

    async def _noop_executor(_session: dict[str, Any], _job):
        return {"reply": "ok", "should_end": False, "checkpoint_hash": "chk"}

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_noop_executor)
    await plane.submit_turn(session_id="audit-session", user_input="hello")
    events = plane.ledger.read()

    verifier = ReplayVerifier()
    replay = verifier.verify_equivalence(events, list(events))

    return {
        "ok": bool(events and replay.get("ok")),
        "event_count": len(events),
        "replay_ok": bool(replay.get("ok")),
    }


def _observability_checks() -> dict[str, Any]:
    from dadbot.core.observability import EventStreamExporter, TraceLevel

    sink: list[dict] = []
    exporter = EventStreamExporter(sink=sink.append, enabled=True, min_level=TraceLevel.MINIMAL, sample_rate=0.0)
    exporter.export({"event": "audit.probe"}, level=TraceLevel.AUDIT)

    otel_present = (ROOT / "dadbot" / "core" / "otel_bridge.py").exists()
    prom_present = (ROOT / "dadbot" / "core" / "prometheus_bridge.py").exists()

    return {
        "ok": bool(sink and otel_present and prom_present),
        "trace_levels": [level.name for level in TraceLevel],
        "otel_bridge": otel_present,
        "prometheus_bridge": prom_present,
    }


def _security_checks() -> dict[str, Any]:
    from dadbot.core.authorization import Capability
    from dadbot.core.capability_registry import CapabilityRegistry, NodeCapabilityRequirement
    from dadbot.core.execution_receipt import ReceiptChain, ReceiptSigner

    registry = CapabilityRegistry()
    registry.register(
        "inference",
        NodeCapabilityRequirement(stage="inference", required_capabilities=frozenset({Capability.EXECUTE})),
    )

    signer = ReceiptSigner(secret_key=b"repo-phase4-audit-key-000000000000")
    chain = ReceiptChain()
    first = signer.sign(
        turn_id="turn-a",
        stage="temporal",
        sequence=1,
        stage_call_id="call-1",
        checkpoint_hash="chk-1",
        prev_receipt_sig="",
    )
    second = signer.sign(
        turn_id="turn-a",
        stage="inference",
        sequence=2,
        stage_call_id="call-2",
        checkpoint_hash="chk-2",
        prev_receipt_sig=first.signature,
    )
    chain.append(first)
    chain.append(second)

    violations = chain.verify_continuity(signer, expected_stages=["temporal", "inference"])

    return {
        "ok": bool(registry.requirement_for("inference").required_capabilities and not violations),
        "receipt_chain_valid": len(violations) == 0,
    }


def _coverage(
    structural: dict[str, Any], observability: dict[str, Any], runtime: dict[str, Any], security: dict[str, Any]
) -> dict[str, float]:
    core = 1.0 if structural.get("ok") and runtime.get("ok") and security.get("ok") else 0.0
    observ = 1.0 if observability.get("ok") else 0.0
    export = 1.0 if observability.get("otel_bridge") and observability.get("prometheus_bridge") else 0.0
    fuzzing = 1.0 if (ROOT / "tests" / "adversarial" / "test_determinism_fuzzing.py").exists() else 0.0
    return {
        "core": core,
        "observability": observ,
        "export": export,
        "fuzzing": fuzzing,
    }


def main() -> int:
    structural = _structural_checks()
    architecture = _architectural_checks()
    runtime = asyncio.run(_runtime_checks())
    observability = _observability_checks()
    security = _security_checks()

    critical_gaps: list[str] = []
    if not structural.get("ok"):
        critical_gaps.append("structural_integrity")
    if not architecture.get("ok"):
        critical_gaps.append("architectural_violations")
    if not runtime.get("ok"):
        critical_gaps.append("runtime_execution_or_replay")
    if not observability.get("ok"):
        critical_gaps.append("observability_control_plane")
    if not security.get("ok"):
        critical_gaps.append("security_boundary")

    coverage = _coverage(structural, observability, runtime, security)
    risk_score = round(min(1.0, len(critical_gaps) * 0.2 + (1.0 - (sum(coverage.values()) / 4.0))), 3)

    report = {
        "phase4_status": "PASS" if not critical_gaps else "FAIL",
        "coverage": coverage,
        "critical_gaps": critical_gaps,
        "risk_score": risk_score,
        "details": {
            "structural": structural,
            "architecture": architecture,
            "runtime": runtime,
            "observability": observability,
            "security": security,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["phase4_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
