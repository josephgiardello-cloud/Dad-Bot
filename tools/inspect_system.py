"""Minimal system inspection CLI.

Usage examples:

    # Print current system health (with live API check if server is running)
    python tools/inspect_system.py --health

    # Show the last scale validation report
    python tools/inspect_system.py --last-report

    # Show the execution timeline for a specific trace / run_id
    python tools/inspect_system.py --trace <run_id>

    # Show invariant violation summary from saved health snapshots
    python tools/inspect_system.py --violations

    # List all available tool capabilities (registered in DynamicToolRegistry)
    python tools/inspect_system.py --tools

The tool works offline (reads artifacts/) when the API server is not running.
When the API server IS running it queries /health for live signal data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ── Repo root on sys.path so imports work when run directly ────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dadbot.core.system_health_scorer import SystemHealthScorer  # noqa: E402


# ── Health Schema Contract (Frozen) ──────────────────────────────────────────
# Version: v1_canonical
# Enforced: No new fields without explicit version bump
# This prevents silent schema drift and regressions.

HEALTH_SCHEMA_VERSION = "v1_canonical"

# Expected top-level keys in health output (order-independent)
EXPECTED_HEALTH_KEYS = frozenset({
    "anomalies",
    "confidence_avg",
    "confidence_trend",
    "dominant_signal",
    "last_turn",
    "stability",
})

FORBIDDEN_HEALTH_KEYS = frozenset({
    "fallback_rate",  # Deprecated field; regression detection
})


def _validate_health_schema(health_dict: dict[str, any]) -> None:
    """Enforce health schema contract invariants.
    
    Raises ValueError if schema contract is violated.
    """
    if not isinstance(health_dict, dict):
        raise ValueError(f"health output must be dict, got {type(health_dict)}")

    # Hard contract lock: this field must never surface in the canonical schema.
    if "fallback_rate" in health_dict:
        raise AssertionError("Forbidden key surfaced post-schema-lock: fallback_rate")
    
    actual_keys = frozenset(health_dict.keys())
    
    # Check for forbidden keys (regression detection)
    forbidden_found = actual_keys & FORBIDDEN_HEALTH_KEYS
    if forbidden_found:
        raise ValueError(
            f"Forbidden keys detected in health schema (regression): {forbidden_found}. "
            f"Schema version {HEALTH_SCHEMA_VERSION} does not include these fields."
        )
    
    # Check for missing expected keys
    missing = EXPECTED_HEALTH_KEYS - actual_keys
    if missing:
        raise ValueError(
            f"Missing expected keys in health schema: {missing}. "
            f"Expected {EXPECTED_HEALTH_KEYS}, got {actual_keys}."
        )
    
    # Check for unexpected keys (new fields without version bump)
    unexpected = actual_keys - EXPECTED_HEALTH_KEYS
    if unexpected:
        raise ValueError(
            f"Unexpected keys detected in health schema (version bump required): {unexpected}. "
            f"Current version {HEALTH_SCHEMA_VERSION} only supports {EXPECTED_HEALTH_KEYS}."
        )


# ── Helpers ─────────────────────────────────────────────────────────────────

def _print_header(title: str) -> None:
    width = 60
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_kv(key: str, value: object, indent: int = 2) -> None:
    prefix = " " * indent
    if isinstance(value, list):
        if value:
            print(f"{prefix}{key}:")
            for item in value:
                print(f"{prefix}  - {item}")
        else:
            print(f"{prefix}{key}: (none)")
    elif isinstance(value, dict):
        print(f"{prefix}{key}:")
        for k, v in value.items():
            print(f"{prefix}  {k}: {v}")
    else:
        print(f"{prefix}{key}: {value}")


def _mean(values: list[float]) -> float:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return 0.0
    return float(sum(clean) / float(len(clean)))


def _dominant_signal_from_share(share: dict[str, object]) -> str:
    cleaned = {
        "safety": float(share.get("safety", 0.0) or 0.0),
        "tools": float(share.get("tools", 0.0) or 0.0),
        "memory": float(share.get("memory", 0.0) or 0.0),
        "coherence": float(share.get("coherence", 0.0) or 0.0),
    }
    return max(cleaned, key=cleaned.get)


def _validate_and_return_health(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    """Validate health snapshot schema before returning."""
    if snapshot is None:
        return None
    _validate_health_schema(snapshot)
    return snapshot


def _compact_diagnostic_snapshot(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    snapshot: dict[str, Any] | None = None
    if not isinstance(payload, dict):
        return None

    diagnostics = dict(payload.get("response_engine_diagnostics") or {})
    if not diagnostics:
        diagnostics = dict(payload.get("response_engine_drift_monitor") or {})
    if not diagnostics and isinstance(payload.get("signals"), dict):
        signals = dict(payload.get("signals") or {})
        diagnostics = dict(signals.get("response_engine_diagnostics") or signals.get("response_engine_drift_monitor") or {})

    if not diagnostics:
        selected = dict((payload.get("response_engine_decision_report") or {}).get("selected") or {})
        if selected:
            influence_share = dict(selected.get("influence_share") or {})
            if not influence_share:
                components = dict(selected.get("components") or {})
                influence_share = {
                    "safety": abs(float(components.get("safety_weight", 0.0) or 0.0)),
                    "tools": abs(float(components.get("tool_weight", 0.0) or 0.0)),
                    "memory": abs(float(components.get("memory_weight", 0.0) or 0.0)),
                    "coherence": abs(float(components.get("coherence_weight", 0.0) or 0.0)),
                }
                total = float(sum(float(value) for value in influence_share.values()))
                if total > 1e-9:
                    influence_share = {key: float(value) / total for key, value in influence_share.items()}
                else:
                    influence_share = {"safety": 0.0, "tools": 0.0, "memory": 0.0, "coherence": 0.0}
            confidence = float(selected.get("decision_confidence", 0.0) or 0.0)
            dominant_signal = _dominant_signal_from_share(influence_share)
            snapshot = {
                "confidence_avg": round(confidence, 3),
                "confidence_trend": 0.0,
                "dominant_signal": dominant_signal,
                "anomalies": [],
                "last_turn": {
                    "confidence": round(confidence, 3),
                    "dominant_signal": dominant_signal,
                },
                "stability": "stable" if confidence >= 0.12 else "unstable",
            }
            return _validate_and_return_health(snapshot)
        return _validate_and_return_health(snapshot)

    if diagnostics:
        selected = dict((payload.get("response_engine_decision_report") or {}).get("selected") or {})
        influence_share = dict(diagnostics.get("influence_share") or selected.get("influence_share") or {})
        dominant_signal = str(diagnostics.get("dominant_signal") or "").strip().lower()
        if dominant_signal not in {"safety", "tools", "memory", "coherence"}:
            if not influence_share:
                components = dict(selected.get("components") or {})
                influence_share = {
                    "safety": abs(float(components.get("safety_weight", 0.0) or 0.0)),
                    "tools": abs(float(components.get("tool_weight", 0.0) or 0.0)),
                    "memory": abs(float(components.get("memory_weight", 0.0) or 0.0)),
                    "coherence": abs(float(components.get("coherence_weight", 0.0) or 0.0)),
                }
                total = float(sum(float(value) for value in influence_share.values()))
                if total > 1e-9:
                    influence_share = {key: float(value) / total for key, value in influence_share.items()}
                else:
                    influence_share = {"safety": 0.0, "tools": 0.0, "memory": 0.0, "coherence": 0.0}
            dominant_signal = _dominant_signal_from_share(influence_share)

        last_turn = dict(diagnostics.get("last_turn") or {})
        if "confidence" not in last_turn:
            last_turn["confidence"] = float(selected.get("decision_confidence", diagnostics.get("confidence_avg", 0.0)) or 0.0)
        if not str(last_turn.get("dominant_signal") or "").strip():
            last_turn["dominant_signal"] = dominant_signal

        anomalies = list(diagnostics.get("anomalies") or [])
        stability = str(diagnostics.get("stability") or "").strip().lower()
        if stability not in {"stable", "drifting", "unstable"}:
            confidence_avg = float(diagnostics.get("confidence_avg", 0.0) or 0.0)
            confidence_trend = float(diagnostics.get("confidence_trend", 0.0) or 0.0)
            stability = (
                "unstable"
                if confidence_avg < 0.12
                else ("drifting" if len(anomalies) > 1 or abs(confidence_trend) >= 0.05 else "stable")
            )

        snapshot = {
            "confidence_avg": round(float(diagnostics.get("confidence_avg", 0.0) or 0.0), 3),
            "confidence_trend": round(float(diagnostics.get("confidence_trend", 0.0) or 0.0), 3),
            "dominant_signal": dominant_signal,
            "anomalies": anomalies[:8],
            "last_turn": {
                "confidence": round(float(last_turn.get("confidence", 0.0) or 0.0), 3),
                "dominant_signal": str(last_turn.get("dominant_signal") or dominant_signal),
            },
            "stability": stability,
        }
        return _validate_and_return_health(snapshot)

    history = list(diagnostics.get("history") or [])
    window = history[-50:]
    selected = dict((payload.get("response_engine_decision_report") or {}).get("selected") or {})
    influence_share = dict(selected.get("influence_share") or {})
    if not influence_share:
        components = dict(selected.get("components") or {})
        influence_share = {
            "safety": abs(float(components.get("safety_weight", 0.0) or 0.0)),
            "tools": abs(float(components.get("tool_weight", 0.0) or 0.0)),
            "memory": abs(float(components.get("memory_weight", 0.0) or 0.0)),
            "coherence": abs(float(components.get("coherence_weight", 0.0) or 0.0)),
        }
        total = float(sum(float(value) for value in influence_share.values()))
        if total > 1e-9:
            influence_share = {key: float(value) / total for key, value in influence_share.items()}
        else:
            influence_share = {"safety": 0.0, "tools": 0.0, "memory": 0.0, "coherence": 0.0}

    confidence_values = [float(item.get("decision_confidence", 0.0) or 0.0) for item in window]
    confidence_avg = _mean(confidence_values)
    if confidence_avg <= 0.0:
        confidence_avg = float(selected.get("decision_confidence", diagnostics.get("confidence_avg", 0.0)) or 0.0)

    if len(window) >= 4:
        half = max(1, len(window) // 2)
        recent = window[-half:]
        prior = window[:-half]
        confidence_trend = _mean([float(item.get("decision_confidence", 0.0) or 0.0) for item in recent]) - _mean(
            [float(item.get("decision_confidence", 0.0) or 0.0) for item in prior]
        )
    else:
        confidence_trend = float(diagnostics.get("confidence_trend", 0.0) or 0.0)

    anomalies = list(diagnostics.get("anomalies") or [])
    dominant_signal = _dominant_signal_from_share(influence_share)
    last_turn_confidence = float(selected.get("decision_confidence", confidence_avg) or confidence_avg)
    stability = diagnostics.get("stability") or (
        "unstable" if confidence_avg < 0.12 else ("drifting" if len(anomalies) > 1 or abs(confidence_trend) >= 0.05 else "stable")
    )

    snapshot = {
        "confidence_avg": round(float(confidence_avg), 3),
        "confidence_trend": round(float(confidence_trend), 3),
        "dominant_signal": dominant_signal,
        "anomalies": anomalies[:8],
        "last_turn": {
            "confidence": round(float(last_turn_confidence), 3),
            "dominant_signal": dominant_signal,
        },
        "stability": str(stability),
    }
    return _validate_and_return_health(snapshot)


def _live_health() -> dict | None:
    """Try to fetch /health from the running API server.  Returns None if unreachable."""
    try:
        from dadbot_system.client import DadBotClient, ServiceClientConfig
        cfg = ServiceClientConfig(auto_start_local=False)
        client = DadBotClient(config=cfg)
        if client.is_healthy():
            return client.health()
    except Exception:
        pass
    return None


def _load_last_report(artifacts_dir: Path) -> dict | None:
    report_path = artifacts_dir / "validation_scale_report.json"
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_event_store_events(run_id: str) -> list[dict] | None:
    """Load persisted events for *run_id* from the SQLite event store."""
    try:
        from dadbot.core.event_store import EventStore  # type: ignore
        store = EventStore()
        return list(store.load_events(run_id=run_id))
    except Exception:
        pass
    # Fallback: look for a JSON file in artifacts/
    candidate = _REPO_ROOT / "artifacts" / f"trace_{run_id}.json"
    if candidate.exists():
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


# ── Commands ─────────────────────────────────────────────────────────────────

def cmd_health(args: argparse.Namespace) -> int:
    live = _live_health()
    snapshot = _compact_diagnostic_snapshot(live)

    # Offline path — use the last persisted report + static scorer
    if snapshot is None:
        last_report = _load_last_report(_REPO_ROOT / "artifacts")
        snapshot = _compact_diagnostic_snapshot(last_report)

    if snapshot is None:
        scorer = SystemHealthScorer()
        report = scorer.score()   # No live gates available offline
        snapshot = {
            "confidence_avg": 0.0,
            "confidence_trend": 0.0,
            "dominant_signal": "coherence",
            "anomalies": [
                "diagnostic_data_unavailable",
            ],
            "last_turn": {
                "confidence": 0.0,
                "dominant_signal": "coherence",
            },
            "stability": "unstable" if not report.is_healthy else "stable",
        }

    _validate_health_schema(snapshot)
    print(json.dumps(snapshot, indent=2, sort_keys=True))

    return 0


def cmd_last_report(args: argparse.Namespace) -> int:
    report = _load_last_report(_REPO_ROOT / "artifacts")
    if not report:
        print("No scale validation report found at artifacts/validation_scale_report.json")
        return 1

    _print_header("Last Scale Validation Report")
    _print_kv("schema_version", report.get("schema_version"))
    _print_kv("captured_at", report.get("timestamp_utc") or report.get("captured_at"))
    _print_kv("turns_requested", report.get("turns_requested"))
    _print_kv("overall_passed", report.get("overall_passed"))
    print()

    groups = report.get("groups") or {}
    for group_name, group_data in groups.items():
        passed = group_data.get("passed", "?")
        skipped = group_data.get("skipped", 0)
        tests = group_data.get("tests") or []
        status = "PASS" if passed is True else ("FAIL" if passed is False else "UNKNOWN")
        print(f"  [{status}] {group_name}  ({len(tests)} tests, {skipped} skipped)")
        for r in tests:
            t_status = str(r.get("status") or "?").upper()
            t_name = str(r.get("test") or r.get("name") or "?")
            duration = r.get("duration_ms")
            dur_str = f"  {duration:.0f}ms" if duration is not None else ""
            print(f"         [{t_status}] {t_name}{dur_str}")
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    run_id = str(args.trace or "").strip()
    if not run_id:
        print("--trace requires a run_id argument")
        return 1

    _print_header(f"Execution Timeline: {run_id}")
    events = _load_event_store_events(run_id)
    if events is None:
        print(f"  No events found for run_id={run_id!r}")
        print("  (Checked: dadbot.core.event_store, artifacts/trace_{run_id}.json)")
        return 1

    from dadbot_system.execution_timeline import ExecutionTimelineBuilder
    timeline = ExecutionTimelineBuilder.build(events)

    print(f"  total_events: {timeline['event_count']}")
    print()

    lc = timeline.get("turn_lifecycle") or []
    if lc:
        print(f"  Turn lifecycle ({len(lc)} events):")
        for e in lc:
            print(f"    [{e['sequence_id']:>4}] {e['type']:30s}  {e['event_time']}")
        print()

    nt = timeline.get("node_trace") or []
    if nt:
        print(f"  Node trace ({len(nt)} events):")
        for e in nt:
            payload_summary = ""
            node = e.get("payload", {}).get("node_name") or e.get("payload", {}).get("node") or ""
            if node:
                payload_summary = f"  node={node}"
            print(f"    [{e['sequence_id']:>4}] {e['type']:30s}{payload_summary}")
        print()

    tt = timeline.get("tool_timeline") or []
    if tt:
        print(f"  Tool calls ({len(tt)} events):")
        for e in tt:
            tool_name = e.get("payload", {}).get("tool_name") or ""
            print(f"    [{e['sequence_id']:>4}] {e['type']:30s}  tool={tool_name}")
        print()

    mh = timeline.get("memory_access_history") or []
    if mh:
        print(f"  Memory access ({len(mh)} events):")
        for e in mh[:20]:  # cap for readability
            print(f"    [{e['sequence_id']:>4}] {e['type']}")
    return 0


def cmd_violations(args: argparse.Namespace) -> int:
    _print_header("Invariant Violation Summary")
    live = _live_health()
    if live:
        score = live.get("health_score", "n/a")
        warnings = live.get("warnings") or []
        v_warnings = [w for w in warnings if "invariant" in str(w).lower()]
        print(f"  health_score:    {score}")
        print(f"  invariant warnings: {len(v_warnings)}")
        for w in v_warnings:
            print(f"    - {w}")
        print("  (source: live API server)")
        return 0

    print("  API server not reachable. No live violation counters available.")
    print("  Run the DEV test lane to surface any active violations:")
    print("    .venv\\Scripts\\python.exe -m pytest -m unit -q")
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    _print_header("Registered Tool Capabilities (ExternalToolRuntime)")
    try:
        from dadbot.core.external_tool_runtime import DynamicToolRegistry
        from dadbot.tools.http_fetch_tool import build_http_fetch_tool
        from dadbot.tools.filesystem_read_tool import build_filesystem_read_tool
    except ImportError as exc:
        print(f"  Import error: {exc}")
        return 1

    registry = DynamicToolRegistry()

    # Register all known tools for display
    http_cap, http_handler = build_http_fetch_tool()
    registry.register(http_cap, http_handler)

    fs_cap, fs_handler = build_filesystem_read_tool(str(_REPO_ROOT))
    registry.register(fs_cap, fs_handler)

    all_caps = registry.discover()
    if not all_caps:
        print("  (no capabilities registered)")
        return 0

    for cap in all_caps:
        print(f"  {cap.name} v{cap.version}")
        print(f"    intents:     {', '.join(cap.intents)}")
        print(f"    tags:        {', '.join(cap.tags)}")
        print(f"    cost_units:  {cap.cost_units}")
        print(f"    reliability: {cap.reliability}")
        print()
    return 0


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="inspect_system",
        description="Dad-Bot system inspection CLI",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--health",
        action="store_true",
        help="Print current system health score and status",
    )
    group.add_argument(
        "--last-report",
        action="store_true",
        dest="last_report",
        help="Print the last scale validation report",
    )
    group.add_argument(
        "--trace",
        metavar="RUN_ID",
        help="Print the execution timeline for a trace/run_id",
    )
    group.add_argument(
        "--violations",
        action="store_true",
        help="Print invariant violation summary",
    )
    group.add_argument(
        "--tools",
        action="store_true",
        help="List all registered tool capabilities",
    )
    args = parser.parse_args()

    if args.health:
        return cmd_health(args)
    if args.last_report:
        return cmd_last_report(args)
    if args.trace:
        return cmd_trace(args)
    if args.violations:
        return cmd_violations(args)
    if args.tools:
        return cmd_tools(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
