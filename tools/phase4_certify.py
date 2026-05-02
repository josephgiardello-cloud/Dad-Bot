from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import hmac
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PHASE4_TRACE_FILE = ROOT / "phase4_trace.json"
PHASE4_CERT_FILE = ROOT / "phase4_certification.json"
PROTOCOL_NAME = "phase4-certification-protocol"
PROTOCOL_VERSION = "1.0"
SIGNATURE_ALGORITHM = "HMAC-SHA256"
SUPPORTED_TRACE_VERSIONS = ("1.0", "legacy-0")
DEFAULT_DEV_SIGNING_KEY = "phase4-local-dev-signing-key"

REQUIRED_STAGES = ["temporal", "inference", "reflection", "save"]
REQUIRED_SIGNALS = (
    "TurnGraph",
    "MutationQueue",
    "VirtualClock",
    "TurnTemporalAxis",
    "Persistence",
    "checkpoint_hash",
    "KernelRejectionSemantics",
)


@dataclass
class StaticScanResult:
    score: float
    signals: dict[str, bool]
    files_scanned: int


@dataclass
class StructuralCheckResult:
    ok: bool
    violations: list[str]
    graph_file: str


@dataclass
class RuntimeCheckResult:
    ok: bool
    missing: list[str]
    stages: list[str]
    order_ok: bool
    mutation_queue_empty: bool
    persistence_contract_ok: bool
    execution_trace_contract_ok: bool
    execution_identity_ok: bool
    identity_event_emitted: bool
    protocol_trace_hash: str
    protocol_identity_fingerprint: str
    source: str
    trace_version: str
    replay_backward_compatible: bool
    replay_anchor: str
    reason: str = ""


def _score_bool(flag: bool) -> float:
    return 1.0 if bool(flag) else 0.0


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _stable_execution_trace_hash(execution_trace: list[dict[str, Any]]) -> str:
    """Compute a deterministic trace hash from execution events.

    Strips known volatile metrics (`duration_ms`, `latency_ms`) from detail to
    avoid false nondeterminism while preserving semantic event ordering.
    """
    normalized_events: list[dict[str, Any]] = []
    for item in list(execution_trace or []):
        if not isinstance(item, dict):
            continue
        detail = dict(item.get("detail") or {})
        detail.pop("duration_ms", None)
        detail.pop("latency_ms", None)
        normalized_events.append(
            {
                "sequence": int(item.get("sequence", 0) or 0),
                "event_type": str(item.get("event_type") or ""),
                "stage": str(item.get("stage") or ""),
                "phase": str(item.get("phase") or ""),
                "detail": detail,
            }
        )
    return hashlib.sha256(_canonical_json_bytes({"events": normalized_events})).hexdigest()


def _stable_identity_fingerprint(
    *,
    trace_id: str,
    trace_hash: str,
    lock_hash: str,
    checkpoint_chain_hash: str,
    mutation_tx_count: int,
    event_count: int,
) -> str:
    payload = {
        "trace_id": str(trace_id or ""),
        "trace_hash": str(trace_hash or ""),
        "lock_hash": str(lock_hash or ""),
        "checkpoint_chain_hash": str(checkpoint_chain_hash or ""),
        "mutation_tx_count": int(mutation_tx_count),
        "event_count": int(event_count),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _build_protocol_metadata() -> dict[str, Any]:
    return {
        "name": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "signature_algorithm": SIGNATURE_ALGORITHM,
        "trace_versions_supported": list(SUPPORTED_TRACE_VERSIONS),
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _resolve_signing_key(*, require_explicit_key: bool) -> tuple[bytes, str, bool]:
    key_raw = os.environ.get("PHASE4_CERT_SIGNING_KEY", "")
    key_id = os.environ.get("PHASE4_CERT_KEY_ID", "local-dev")
    using_default = False
    if not key_raw:
        if require_explicit_key:
            raise RuntimeError("Explicit signing key required but PHASE4_CERT_SIGNING_KEY is not set")
        key_raw = DEFAULT_DEV_SIGNING_KEY
        using_default = True
    return key_raw.encode("utf-8"), str(key_id or "local-dev"), using_default


def _sign_payload(payload: dict[str, Any], *, require_explicit_key: bool = False) -> dict[str, str]:
    key, key_id, using_default = _resolve_signing_key(require_explicit_key=require_explicit_key)
    digest = hmac.new(key, _canonical_json_bytes(payload), hashlib.sha256).hexdigest()
    return {
        "algorithm": SIGNATURE_ALGORITHM,
        "key_id": key_id,
        "value": digest,
        "using_default_key": str(bool(using_default)).lower(),
    }


def _verify_signed_report(report: dict[str, Any], *, require_explicit_key: bool = False) -> tuple[bool, str]:
    signature = dict(report.get("signature") or {})
    provided = str(signature.get("value") or "").strip()
    if not provided:
        return False, "Missing signature.value"

    if require_explicit_key and str(signature.get("using_default_key") or "").lower() == "true":
        return False, "Report is signed with the default development key"

    unsigned = dict(report)
    unsigned.pop("signature", None)
    expected = _sign_payload(unsigned, require_explicit_key=require_explicit_key).get("value", "")
    if not hmac.compare_digest(provided, expected):
        return False, "Signature mismatch"
    return True, "ok"


def _static_violations(static: StaticScanResult, structural: StructuralCheckResult) -> list[str]:
    violations: list[str] = []
    missing_signals = [name for name, ok in static.signals.items() if not bool(ok)]
    if missing_signals:
        violations.append(f"Missing required static signals: {sorted(missing_signals)!r}")

    # Policy module integrity.
    try:
        from dadbot.core.execution_policy import (
            ExecutionPolicyEngine,
            KernelRejectionSemantics,
            PersistenceServiceContract,
            StagePhaseMappingPolicy,
        )

        _ = ExecutionPolicyEngine
        _ = KernelRejectionSemantics
        _ = PersistenceServiceContract
        _ = StagePhaseMappingPolicy
    except Exception as exc:
        violations.append(f"Policy module integrity check failed: {exc}")

    # Persistence contract schema.
    try:
        from dadbot.core.execution_policy import PersistenceServiceContract

        c = PersistenceServiceContract()
        if not (str(c.save_turn) and str(c.save_graph_checkpoint) and str(c.save_turn_event)):
            violations.append("Persistence contract schema invalid: required method names are empty")
    except Exception as exc:
        violations.append(f"Persistence contract schema check failed: {exc}")

    # Forbidden pattern scan in core/services: import random
    random_import_allow = {
        "external_tool_runtime.py",
        "fault_injection.py",
    }
    for py in (ROOT / "dadbot" / "core").rglob("*.py"):
        text = _safe_read(py)
        if "import random" in text and py.name not in random_import_allow:
            violations.append(f"Forbidden pattern in core: import random -> {py}")
    for py in (ROOT / "dadbot" / "services").rglob("*.py"):
        text = _safe_read(py)
        if "import random" in text:
            violations.append(f"Forbidden pattern in services: import random -> {py}")

    violations.extend(list(structural.violations or []))
    # De-duplicate while preserving order.
    deduped: list[str] = []
    seen: set[str] = set()
    for item in violations:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _run_static_mode() -> int:
    static = static_scan()
    structural = structural_check()
    violations = _static_violations(static, structural)
    static_pass = bool(static.score == 1.0 and structural.ok and len(violations) == 0)
    payload = {
        "mode": "static",
        "static_pass": "PASS" if static_pass else "FAIL",
        "static_score": static.score,
        "files_scanned": static.files_scanned,
        "signals": static.signals,
        "static_violations": violations,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if static_pass else 1


def _iter_python_files() -> list[Path]:
    return [
        p
        for p in ROOT.rglob("*.py")
        if ".venv" not in p.parts and "__pycache__" not in p.parts and ".git" not in p.parts
    ]


# ------------------------------------------------------------
# 1. STATIC STRUCTURAL SCAN
# ------------------------------------------------------------
def static_scan() -> StaticScanResult:
    signals = dict.fromkeys(REQUIRED_SIGNALS, False)
    files = _iter_python_files()

    for py_file in files:
        text = _safe_read(py_file)
        if not text:
            continue
        for key in REQUIRED_SIGNALS:
            if key in text:
                signals[key] = True

    score = sum(1 for v in signals.values() if v) / float(len(signals) or 1)
    return StaticScanResult(score=round(score, 6), signals=signals, files_scanned=len(files))


class _GraphExecuteDeterminismVisitor(ast.NodeVisitor):
    """Find forbidden clock calls inside execute() methods in graph layer."""

    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name == "execute":
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                if isinstance(child.func, ast.Attribute):
                    owner = child.func.value
                    if isinstance(owner, ast.Name):
                        if owner.id == "datetime" and child.func.attr == "now":
                            self.violations.append(
                                f"datetime.now() inside execute() at line {getattr(child, 'lineno', 0)}"
                            )
                        if owner.id == "time" and child.func.attr in {"time"}:
                            self.violations.append(
                                f"time.{child.func.attr}() inside execute() at line {getattr(child, 'lineno', 0)}"
                            )
        self.generic_visit(node)


# ------------------------------------------------------------
# 2. STRUCTURAL GRAPH CHECK
# ------------------------------------------------------------
def structural_check() -> StructuralCheckResult:
    violations: list[str] = []

    graph_candidates = [ROOT / "dadbot" / "core" / "graph.py", ROOT / "dadbot" / "graph.py"]
    graph_path = next((p for p in graph_candidates if p.exists()), graph_candidates[0])

    text = _safe_read(graph_path)
    if not text:
        violations.append(f"Graph file unreadable or missing: {graph_path}")
        return StructuralCheckResult(ok=False, violations=violations, graph_file=str(graph_path))

    # Some files may include a UTF-8 BOM; strip it before AST parsing.
    normalized_text = text.lstrip("\ufeff")

    # Detect forbidden clock access in execute() methods.
    try:
        tree = ast.parse(normalized_text)
        visitor = _GraphExecuteDeterminismVisitor()
        visitor.visit(tree)
        violations.extend(visitor.violations)
    except SyntaxError as exc:
        violations.append(f"Graph file syntax parse failure: {exc}")

    # Mutation guard contract in graph layer.
    if "mutation_queue" + ".queue(" in normalized_text and "MutationGuard" not in normalized_text:
        violations.append("Unprotected mutation access path detected in graph layer")

    # Structural order contract in default graph.
    try:
        from dadbot.core.graph import TurnGraph

        graph = TurnGraph(registry=None)
        names = [str(getattr(n, "name", type(n).__name__) or type(n).__name__).strip().lower() for n in graph.nodes]
        if not names:
            violations.append("TurnGraph default pipeline has no nodes")
        else:
            if names[0] != "temporal":
                violations.append(f"Temporal node is not first: first={names[0]!r}")
            if "save" not in names:
                violations.append("Save node missing from default graph pipeline")
            if "temporal" in names:
                t_idx = names.index("temporal")
                for stage in ("inference", "reflection", "save"):
                    if stage in names and names.index(stage) <= t_idx:
                        violations.append(f"Stage order violation: {stage!r} occurs before/at temporal")
    except Exception as exc:
        violations.append(f"Unable to construct TurnGraph for structural verification: {exc}")

    return StructuralCheckResult(ok=len(violations) == 0, violations=violations, graph_file=str(graph_path))


# ------------------------------------------------------------
# 3. RUNTIME TRACE VALIDATION
# ------------------------------------------------------------
class _MockMaintenanceService:
    def tick(self, _turn_context: Any) -> dict[str, Any]:
        return {"ok": True}


class _MockContextService:
    def build_context(self, _turn_context: Any) -> dict[str, Any]:
        return {"context": "phase4"}


class _MockAgentService:
    async def run_agent(self, _turn_context: Any, _rich_context: dict[str, Any]) -> tuple[str, bool]:
        return ("phase4 runtime probe", False)


class _MockSafetyService:
    def enforce_policies(self, _turn_context: Any, candidate: Any) -> Any:
        return candidate if candidate is not None else ("safe fallback", False)


class _MockReflectionService:
    def reflect(self, _turn_context: Any, _result: Any) -> dict[str, Any]:
        return {"reflected": True}


class _MockPersistenceService:
    contract_version = "1.0"

    def __init__(self) -> None:
        self.checkpoints: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    def save_turn(self, _ctx: Any, _result: Any) -> None:
        return None

    def save_graph_checkpoint(self, payload: dict[str, Any], **_kwargs: Any) -> None:
        self.checkpoints.append(dict(payload or {}))

    def save_turn_event(self, payload: dict[str, Any]) -> None:
        self.events.append(dict(payload or {}))


class _MockRegistry:
    def __init__(self) -> None:
        self._services: dict[str, Any] = {
            "maintenance_service": _MockMaintenanceService(),
            "context_service": _MockContextService(),
            "agent_service": _MockAgentService(),
            "safety_service": _MockSafetyService(),
            "reflection": _MockReflectionService(),
            "persistence_service": _MockPersistenceService(),
            "telemetry": None,
        }

    def get(self, key: str) -> Any:
        return self._services.get(str(key or ""), None)


def _build_runtime_trace_artifact() -> tuple[dict[str, Any], str]:
    from dadbot.core.graph import TurnContext, TurnGraph, TurnTemporalAxis

    registry = _MockRegistry()
    graph = TurnGraph(registry=registry)
    deterministic_lock_hash = "phase4-certification-lock-hash"
    turn_context = TurnContext(
        user_input="phase4 certification gate runtime probe",
        trace_id="phase4-certification-runtime-probe",
        temporal=TurnTemporalAxis.from_lock_hash(deterministic_lock_hash),
        metadata={
            "determinism": {"lock_hash": deterministic_lock_hash},
            "checkpoint_every_node": True,
        },
    )

    result = asyncio.run(graph.execute(turn_context))
    persistence = registry.get("persistence_service")
    event_types = [str(e.get("event_type") or "") for e in list(getattr(persistence, "events", []))]
    execution_trace = list(turn_context.state.get("execution_trace") or [])
    protocol_trace_hash = _stable_execution_trace_hash(execution_trace)
    mutation_snapshot = dict(turn_context.mutation_queue.snapshot() or {})
    execution_trace_contract = dict(turn_context.state.get("execution_trace_contract") or {})
    lock_hash = str((turn_context.metadata.get("determinism") or {}).get("lock_hash") or "")
    protocol_identity_fingerprint = _stable_identity_fingerprint(
        trace_id=str(turn_context.trace_id or ""),
        trace_hash=protocol_trace_hash,
        lock_hash=lock_hash,
        checkpoint_chain_hash=str(turn_context.last_checkpoint_hash or ""),
        mutation_tx_count=int(mutation_snapshot.get("transactions", 0) or 0),
        event_count=int(execution_trace_contract.get("event_count", 0) or 0),
    )

    artifact = {
        "trace_id": turn_context.trace_id,
        "trace_version": "1.0",
        "result": result,
        "stages": [str(trace.stage or "") for trace in list(turn_context.stage_traces or [])],
        "phase_history": list(turn_context.phase_history or []),
        "execution_trace": execution_trace,
        "execution_trace_contract": execution_trace_contract,
        "protocol_trace_hash": protocol_trace_hash,
        "execution_identity": dict(turn_context.state.get("execution_identity") or {}),
        "protocol_identity_fingerprint": protocol_identity_fingerprint,
        "mutation_snapshot": mutation_snapshot,
        "persistence_contract": dict(turn_context.state.get("persistence_contract") or {}),
        "persisted_event_types": event_types,
    }
    PHASE4_TRACE_FILE.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact, "generated"


def _normalize_runtime_trace_artifact(raw: dict[str, Any]) -> dict[str, Any]:
    stages_raw = raw.get("stages")
    if isinstance(stages_raw, list):
        stages = [str(s or "").strip().lower() for s in stages_raw if str(s or "").strip()]
    else:
        # Backward-compatible legacy shape: stage_traces list[dict(stage=...)]
        stage_traces = raw.get("stage_traces")
        stages = []
        if isinstance(stage_traces, list):
            for item in stage_traces:
                if isinstance(item, dict):
                    stage = str(item.get("stage") or "").strip().lower()
                    if stage:
                        stages.append(stage)

    execution_trace_contract = raw.get("execution_trace_contract")
    if not isinstance(execution_trace_contract, dict):
        execution_trace_contract = raw.get("trace_contract")
    if not isinstance(execution_trace_contract, dict):
        execution_trace_contract = {}

    execution_identity = raw.get("execution_identity")
    if not isinstance(execution_identity, dict):
        execution_identity = raw.get("identity")
    if not isinstance(execution_identity, dict):
        execution_identity = {}

    mutation_snapshot = raw.get("mutation_snapshot")
    if not isinstance(mutation_snapshot, dict):
        mutation_snapshot = raw.get("mutation_queue")
    if not isinstance(mutation_snapshot, dict):
        mutation_snapshot = {}

    persistence_contract = raw.get("persistence_contract")
    if not isinstance(persistence_contract, dict):
        persistence_contract = {}

    event_types_raw = raw.get("persisted_event_types")
    if isinstance(event_types_raw, list):
        persisted_event_types = [str(e or "").strip() for e in event_types_raw if str(e or "").strip()]
    else:
        events = raw.get("events")
        persisted_event_types = []
        if isinstance(events, list):
            for event in events:
                if isinstance(event, dict):
                    event_type = str(event.get("event_type") or "").strip()
                    if event_type:
                        persisted_event_types.append(event_type)

    trace_version = str(raw.get("trace_version") or raw.get("schema_version") or "legacy-0").strip()
    if not trace_version:
        trace_version = "legacy-0"

    return {
        "stages": stages,
        "execution_trace": list(raw.get("execution_trace") or []),
        "execution_trace_contract": execution_trace_contract,
        "protocol_trace_hash": str(raw.get("protocol_trace_hash") or ""),
        "execution_identity": execution_identity,
        "protocol_identity_fingerprint": str(raw.get("protocol_identity_fingerprint") or ""),
        "mutation_snapshot": mutation_snapshot,
        "persistence_contract": persistence_contract,
        "persisted_event_types": persisted_event_types,
        "trace_version": trace_version,
    }


def _load_or_generate_runtime_trace() -> tuple[dict[str, Any], str, str]:
    if PHASE4_TRACE_FILE.exists():
        try:
            raw = json.loads(PHASE4_TRACE_FILE.read_text(encoding="utf-8"))
            return (
                _normalize_runtime_trace_artifact(raw if isinstance(raw, dict) else {}),
                "artifact",
                "",
            )
        except Exception as exc:
            # Fall through to runtime probe generation if artifact is unreadable.
            reason = f"Existing artifact unreadable: {exc}. Re-generating from runtime probe."
            artifact, source = _build_runtime_trace_artifact()
            return _normalize_runtime_trace_artifact(artifact), source, reason

    artifact, source = _build_runtime_trace_artifact()
    return _normalize_runtime_trace_artifact(artifact), source, ""


def runtime_check() -> RuntimeCheckResult:
    artifact, source, prior_reason = _load_or_generate_runtime_trace()
    stages = [str(s).strip().lower() for s in list(artifact.get("stages") or [])]

    missing = [stage for stage in REQUIRED_STAGES if stage not in stages]

    order_ok = True
    if len(missing) == 0:
        try:
            order_ok = (
                stages.index("temporal") < stages.index("inference")
                and stages.index("inference") < stages.index("reflection")
                and stages.index("reflection") < stages.index("save")
            )
        except ValueError:
            order_ok = False

    mutation_snapshot = dict(artifact.get("mutation_snapshot") or {})
    mutation_queue_empty = (
        int(mutation_snapshot.get("pending", 0) or 0) == 0 and int(mutation_snapshot.get("ledger_pending", 0) or 0) == 0
    )

    execution_trace_contract = dict(artifact.get("execution_trace_contract") or {})
    trace_hash = str(execution_trace_contract.get("trace_hash") or "")
    protocol_trace_hash = str(artifact.get("protocol_trace_hash") or "").strip()
    if not protocol_trace_hash:
        protocol_trace_hash = _stable_execution_trace_hash(list(artifact.get("execution_trace") or []))
    execution_trace_contract_ok = (
        execution_trace_contract.get("version") == "1.0"
        and isinstance(execution_trace_contract.get("event_count"), int)
        and execution_trace_contract.get("event_count", 0) > 0
        and len(trace_hash) == 64
        and len(protocol_trace_hash) == 64
    )

    execution_identity = dict(artifact.get("execution_identity") or {})
    protocol_identity_fingerprint = str(artifact.get("protocol_identity_fingerprint") or "").strip()
    if not protocol_identity_fingerprint:
        mutation_snapshot = dict(artifact.get("mutation_snapshot") or {})
        lock_hash = str(execution_identity.get("lock_hash") or "").strip()
        protocol_identity_fingerprint = _stable_identity_fingerprint(
            trace_id=str(execution_identity.get("trace_id") or artifact.get("trace_id") or ""),
            trace_hash=protocol_trace_hash,
            lock_hash=lock_hash,
            checkpoint_chain_hash=str(execution_identity.get("checkpoint_chain_hash") or ""),
            mutation_tx_count=int(mutation_snapshot.get("transactions", 0) or 0),
            event_count=int(execution_trace_contract.get("event_count", 0) or 0),
        )
    execution_identity_ok = (
        bool(execution_identity.get("fingerprint"))
        and bool(execution_identity.get("trace_hash"))
        and len(protocol_identity_fingerprint) == 64
    )

    persistence_contract = dict(artifact.get("persistence_contract") or {})
    persistence_contract_ok = bool(persistence_contract.get("ok", False)) and bool(
        persistence_contract.get("compatible", False)
    )

    event_types = [str(e).strip() for e in list(artifact.get("persisted_event_types") or [])]
    identity_event_emitted = "execution_identity" in event_types

    trace_version = str(artifact.get("trace_version") or "legacy-0")
    replay_anchor = str(protocol_identity_fingerprint or protocol_trace_hash or trace_hash or "")
    replay_backward_compatible = bool(
        trace_version in SUPPORTED_TRACE_VERSIONS and replay_anchor and len(missing) == 0 and order_ok
    )

    ok = (
        len(missing) == 0
        and order_ok
        and mutation_queue_empty
        and execution_trace_contract_ok
        and execution_identity_ok
        and persistence_contract_ok
        and identity_event_emitted
        and replay_backward_compatible
    )

    reason_parts: list[str] = []
    if prior_reason:
        reason_parts.append(prior_reason)
    if missing:
        reason_parts.append(f"Missing required stages: {missing}")
    if not order_ok:
        reason_parts.append("Canonical stage ordering violation")
    if not mutation_queue_empty:
        reason_parts.append("Mutation queue not empty after runtime execution")
    if not execution_trace_contract_ok:
        reason_parts.append("Execution trace contract invalid or missing")
    if not execution_identity_ok:
        reason_parts.append("Execution identity missing/invalid")
    if not persistence_contract_ok:
        reason_parts.append("Persistence contract invalid/incompatible")
    if not identity_event_emitted:
        reason_parts.append("execution_identity persistence event missing")
    if not replay_backward_compatible:
        reason_parts.append("Replay compatibility violation: unsupported trace version or missing replay anchor")

    return RuntimeCheckResult(
        ok=ok,
        missing=missing,
        stages=stages,
        order_ok=order_ok,
        mutation_queue_empty=mutation_queue_empty,
        persistence_contract_ok=persistence_contract_ok,
        execution_trace_contract_ok=execution_trace_contract_ok,
        execution_identity_ok=execution_identity_ok,
        identity_event_emitted=identity_event_emitted,
        protocol_trace_hash=protocol_trace_hash,
        protocol_identity_fingerprint=protocol_identity_fingerprint,
        source=source,
        trace_version=trace_version,
        replay_backward_compatible=replay_backward_compatible,
        replay_anchor=replay_anchor,
        reason="; ".join(reason_parts),
    )


# ------------------------------------------------------------
# 4. CONSOLIDATED CERTIFICATION MODEL
# ------------------------------------------------------------
def _build_subsystem_scores(
    static: StaticScanResult,
    structural: StructuralCheckResult,
    runtime: RuntimeCheckResult,
) -> dict[str, float]:
    determinism_verdict = bool(
        static.signals.get("VirtualClock", False)
        and static.signals.get("TurnTemporalAxis", False)
        and structural.ok
        and runtime.execution_trace_contract_ok
    )

    mutation_integrity_verdict = bool(
        static.signals.get("MutationQueue", False) and runtime.mutation_queue_empty and structural.ok
    )

    persistence_compliance_verdict = bool(
        static.signals.get("Persistence", False)
        and static.signals.get("checkpoint_hash", False)
        and runtime.persistence_contract_ok
        and runtime.identity_event_emitted
    )

    replay_correctness_verdict = bool(
        runtime.ok
        and runtime.execution_identity_ok
        and runtime.order_ok
        and len(runtime.missing) == 0
        and runtime.replay_backward_compatible
    )

    scores = {
        "static_completeness": round(static.score, 6),
        "structural_integrity": _score_bool(structural.ok),
        "runtime_trace": _score_bool(runtime.ok),
        "determinism": _score_bool(determinism_verdict),
        "mutation_integrity": _score_bool(mutation_integrity_verdict),
        "persistence_compliance": _score_bool(persistence_compliance_verdict),
        "replay_correctness": _score_bool(replay_correctness_verdict),
    }
    scores["overall"] = round(sum(scores.values()) / float(len(scores) or 1), 6)
    return scores


def build_certification_hash(report: dict[str, Any]) -> str:
    hash_payload = {
        "protocol": dict(report.get("protocol") or {}),
        "GLOBAL_PASS": bool(report.get("GLOBAL_PASS", False)),
        "scores": dict(report.get("subsystem_scores") or {}),
        "static_signals": dict(report.get("static", {}).get("signals") or {}),
        "structural_violations": list(report.get("structural", {}).get("violations") or []),
        "runtime_missing": list(report.get("runtime", {}).get("missing") or []),
        "runtime_trace_version": str(report.get("runtime", {}).get("trace_version") or ""),
        "runtime_replay_anchor": str(report.get("runtime", {}).get("replay_anchor") or ""),
        "determinism_verdict": bool(report.get("determinism_verdict", False)),
        "mutation_integrity_verdict": bool(report.get("mutation_integrity_verdict", False)),
        "persistence_compliance_verdict": bool(report.get("persistence_compliance_verdict", False)),
        "replay_correctness_verdict": bool(report.get("replay_correctness_verdict", False)),
    }
    encoded = json.dumps(hash_payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ------------------------------------------------------------
# 5. MAIN GATE
# ------------------------------------------------------------
def _run_gate(*, require_explicit_key: bool) -> dict[str, Any]:
    print("\n=== PHASE 4 CERTIFICATION GATE ===\n")

    static = static_scan()
    structural = structural_check()
    runtime = runtime_check()

    subsystem_scores = _build_subsystem_scores(static, structural, runtime)

    determinism_verdict = subsystem_scores["determinism"] == 1.0
    mutation_integrity_verdict = subsystem_scores["mutation_integrity"] == 1.0
    persistence_compliance_verdict = subsystem_scores["persistence_compliance"] == 1.0
    replay_correctness_verdict = subsystem_scores["replay_correctness"] == 1.0

    global_pass = bool(
        static.score == 1.0
        and structural.ok
        and runtime.ok
        and determinism_verdict
        and mutation_integrity_verdict
        and persistence_compliance_verdict
        and replay_correctness_verdict
    )

    report: dict[str, Any] = {
        "protocol": _build_protocol_metadata(),
        "GLOBAL_PASS": global_pass,
        "subsystem_scores": subsystem_scores,
        "determinism_verdict": determinism_verdict,
        "mutation_integrity_verdict": mutation_integrity_verdict,
        "persistence_compliance_verdict": persistence_compliance_verdict,
        "replay_correctness_verdict": replay_correctness_verdict,
        "replay_guarantees": {
            "verdict": replay_correctness_verdict,
            "backward_compatible": runtime.replay_backward_compatible,
            "supported_trace_versions": list(SUPPORTED_TRACE_VERSIONS),
            "observed_trace_version": runtime.trace_version,
            "replay_anchor": runtime.replay_anchor,
            "requires": {
                "canonical_stage_order": True,
                "execution_identity": True,
                "trace_contract": True,
            },
        },
        "static": {
            "score": static.score,
            "signals": static.signals,
            "files_scanned": static.files_scanned,
        },
        "structural": {
            "ok": structural.ok,
            "violations": structural.violations,
            "graph_file": structural.graph_file,
        },
        "runtime": {
            "ok": runtime.ok,
            "missing": runtime.missing,
            "stages": runtime.stages,
            "order_ok": runtime.order_ok,
            "mutation_queue_empty": runtime.mutation_queue_empty,
            "persistence_contract_ok": runtime.persistence_contract_ok,
            "execution_trace_contract_ok": runtime.execution_trace_contract_ok,
            "execution_identity_ok": runtime.execution_identity_ok,
            "identity_event_emitted": runtime.identity_event_emitted,
            "source": runtime.source,
            "trace_version": runtime.trace_version,
            "replay_backward_compatible": runtime.replay_backward_compatible,
            "replay_anchor": runtime.replay_anchor,
            "reason": runtime.reason,
        },
    }
    report["final_certification_hash"] = build_certification_hash(report)

    # Signed protocol envelope.
    unsigned_report = dict(report)
    report["signature"] = _sign_payload(unsigned_report, require_explicit_key=require_explicit_key)

    print(json.dumps(report, indent=2, sort_keys=True))
    PHASE4_CERT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def _verify_mode(report_path: Path, *, require_explicit_key: bool) -> int:
    if not report_path.exists():
        print(json.dumps({"ok": False, "reason": f"Missing report file: {report_path}"}, indent=2))
        return 1
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(json.dumps({"ok": False, "reason": f"Unreadable report: {exc}"}, indent=2))
        return 1

    if not isinstance(report, dict):
        print(json.dumps({"ok": False, "reason": "Report is not a JSON object"}, indent=2))
        return 1

    protocol = dict(report.get("protocol") or {})
    version = str(protocol.get("version") or "")
    sig_ok, sig_reason = _verify_signed_report(report, require_explicit_key=require_explicit_key)
    replay_block = dict(report.get("replay_guarantees") or {})
    replay_ok = bool(replay_block.get("backward_compatible", False))

    verdict = {
        "ok": bool(sig_ok and version == PROTOCOL_VERSION and replay_ok),
        "protocol_version": version,
        "expected_protocol_version": PROTOCOL_VERSION,
        "signature_ok": sig_ok,
        "signature_reason": sig_reason,
        "replay_backward_compatible": replay_ok,
    }
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["ok"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 certification protocol gate")
    parser.add_argument(
        "--mode",
        default="full",
        choices=("static", "full"),
        help="Certification mode. static=static structural checks only, full=full protocol run",
    )
    parser.add_argument(
        "--verify",
        default="",
        help="Verify an existing signed certification artifact (default: phase4_certification.json)",
    )
    parser.add_argument(
        "--require-explicit-signing-key",
        action="store_true",
        help="Fail if PHASE4_CERT_SIGNING_KEY is not set, and reject reports signed with default key",
    )
    args = parser.parse_args()

    require_explicit_key = bool(
        args.require_explicit_signing_key
        or str(os.environ.get("PHASE4_CERT_REQUIRE_SIGNING_KEY", "")).strip().lower() in {"1", "true", "yes"}
        or str(os.environ.get("CI", "")).strip().lower() in {"1", "true", "yes"}
    )

    if args.mode == "static":
        exit_code = _run_static_mode()
        raise SystemExit(exit_code)

    if args.verify:
        exit_code = _verify_mode(Path(args.verify), require_explicit_key=require_explicit_key)
        raise SystemExit(exit_code)

    report = _run_gate(require_explicit_key=require_explicit_key)

    if not bool(report.get("GLOBAL_PASS", False)):
        raise SystemExit("\nFAIL: Phase 4 certification gate failed\n")

    print("\nPASS: Phase 4 certification gate passed\n")


if __name__ == "__main__":
    # Ensure project root importability when executed from tools/.
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    main()
