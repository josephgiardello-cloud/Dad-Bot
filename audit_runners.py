"""audit_runners.py — Pure post-hoc verification functions.

All functions here are read-only over immutable execution traces.
No live execution, no monkey-patching, no state mutation, no side effects.
Each function receives already-completed trace data and returns a result dict.
"""
from __future__ import annotations

import inspect
import re
import time
from typing import Any

from dadbot.core.observability import get_exporter, get_metrics


# ---------------------------------------------------------------------------
# Private trace-reading helpers
# ---------------------------------------------------------------------------

def _event_type(event: dict[str, Any]) -> str:
    return str(event.get("type") or event.get("event_type") or "")


def _format_exc(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _registry_get_optional(registry: Any, name: str) -> Any | None:
    try:
        return registry.get(name)
    except Exception:
        return None


def _event_job_id(event: dict[str, Any]) -> str:
    payload = dict(event.get("payload") or {})
    job_id = str(payload.get("job_id") or event.get("job_id") or "").strip()
    return job_id


def _extract_registry_aliases_from_orchestrator(orchestrator: Any) -> list[str]:
    aliases: set[str] = set()
    for fn_name in ("_build_turn_graph", "_build_turn_context"):
        fn = getattr(orchestrator, fn_name, None)
        if fn is None:
            continue
        source = inspect.getsource(fn)
        matches = re.findall(r"registry\.get\(\s*\"([^\"]+)\"\s*\)", source)
        aliases.update(matches)
    return sorted(aliases)


def _unused_import_symbols_in_orchestrator() -> list[str]:
    from dadbot.core import orchestrator as orch_module

    source = inspect.getsource(orch_module)
    import_lines = [
        line.strip() for line in source.splitlines()
        if line.strip().startswith("from ") or line.strip().startswith("import ")
    ]

    imported: set[str] = set()
    for line in import_lines:
        if line.startswith("import "):
            names = [part.strip().split(" as ")[0] for part in line.replace("import ", "").split(",")]
            imported.update(names)
        elif " import " in line:
            rhs = line.split(" import ", 1)[1]
            names = [part.strip().split(" as ")[0] for part in rhs.split(",")]
            imported.update(names)

    ignored = {"__future__", "annotations"}
    imported = {name for name in imported if name and name not in ignored}

    body = "\n".join(
        line for line in source.splitlines()
        if not line.strip().startswith("from ") and not line.strip().startswith("import ")
    )
    unused = sorted(name for name in imported if re.search(rf"\b{re.escape(name)}\b", body) is None)
    return unused


# ---------------------------------------------------------------------------
# LAYER 1 — Wiring Integrity
# ---------------------------------------------------------------------------

def build_wiring_map(orchestrator: Any) -> dict[str, Any]:
    """Read component references from a completed orchestrator instance."""
    registry = getattr(orchestrator, "registry", None)
    metrics = _registry_get_optional(registry, "metrics") if registry is not None else None
    if metrics is None:
        metrics = get_metrics()

    exporter = getattr(orchestrator, "exporter", None)
    if exporter is None:
        exporter = get_exporter()

    return {
        "control_plane": getattr(orchestrator, "control_plane", None),
        "scheduler": getattr(orchestrator, "scheduler", None),
        "graph": getattr(orchestrator, "graph", None),
        "ledger": getattr(getattr(orchestrator, "control_plane", None), "ledger", None),
        "lease": getattr(getattr(orchestrator, "control_plane", None), "execution_lease", None),
        "metrics": metrics,
        "exporter": exporter,
    }


def check_wiring_integrity(orchestrator: Any) -> dict[str, Any]:
    """Verify all declared subsystems are present (read-only attribute inspection)."""
    wiring = build_wiring_map(orchestrator)
    missing = [k for k, v in wiring.items() if v is None]
    return {
        "ok": len(missing) == 0,
        "missing_components": missing,
    }


# ---------------------------------------------------------------------------
# LAYER 2 — Runtime Activation (reads from immutable ledger + metrics)
# ---------------------------------------------------------------------------

def check_runtime_activation(ledger: Any, metrics: Any) -> dict[str, Any]:
    """Verify that required event types appear in the execution trace."""
    events = ledger.read() if hasattr(ledger, "read") else list(ledger.snapshot())
    event_types = {_event_type(e) for e in events}

    required_flows = {
        "scheduler_active": "JOB_QUEUED" in event_types,
        "execution_active": "JOB_STARTED" in event_types,
        "completion_active": "JOB_COMPLETED" in event_types,
        "failure_path_active": "JOB_FAILED" in event_types,
    }

    metric_active = int(metrics.counter("scheduler.job.completed")) > 0
    return {
        "ok": all(required_flows.values()) and metric_active,
        "details": required_flows,
        "metrics_active": metric_active,
    }


def check_ledger(ledger: Any) -> dict[str, Any]:
    """Verify required event sequence exists in the immutable ledger."""
    started = time.perf_counter()
    try:
        if hasattr(ledger, "read"):
            events = ledger.read()
        elif hasattr(ledger, "snapshot"):
            events = ledger.snapshot()
        elif hasattr(ledger, "get_events"):
            events = ledger.get_events()
        else:
            raise RuntimeError("Ledger has no readable event API")

        names = [_event_type(e) for e in events]
        required = ["JOB_SUBMITTED", "JOB_QUEUED", "JOB_STARTED"]
        ok = all(r in names for r in required)
        return {"ok": ok, "events": names, "elapsed_s": round(time.perf_counter() - started, 4)}
    except Exception as exc:
        return {"ok": False, "detail": _format_exc(exc), "elapsed_s": round(time.perf_counter() - started, 4)}


# ---------------------------------------------------------------------------
# LAYER 3 — Cross-Component Causality (reads from immutable ledger)
# ---------------------------------------------------------------------------

def check_causality(orchestrator: Any) -> dict[str, Any]:
    """Verify job lifecycle causality and identity propagation from the ledger."""
    ledger = orchestrator.control_plane.ledger.read()

    job_ids = {_event_job_id(e) for e in ledger if _event_job_id(e)}
    started = {_event_job_id(e) for e in ledger if _event_type(e) == "JOB_STARTED" and _event_job_id(e)}
    completed = {_event_job_id(e) for e in ledger if _event_type(e) == "JOB_COMPLETED" and _event_job_id(e)}
    failed = {_event_job_id(e) for e in ledger if _event_type(e) == "JOB_FAILED" and _event_job_id(e)}

    terminal = completed | failed
    orphan_jobs = sorted(job_ids - started)
    incomplete_jobs = sorted(started - terminal)

    lifecycle = {"JOB_SUBMITTED", "JOB_QUEUED", "JOB_STARTED", "JOB_COMPLETED", "JOB_FAILED"}
    missing_trace = []
    missing_session = []
    for e in ledger:
        et = _event_type(e)
        if et not in lifecycle:
            continue
        jid = _event_job_id(e)
        if not str(e.get("trace_id") or "").strip():
            missing_trace.append(jid or et)
        if not str(e.get("session_id") or "").strip():
            missing_session.append(jid or et)

    metrics = get_metrics()
    completed_counter = int(metrics.counter("scheduler.job.completed"))
    observability_linked = completed_counter >= len(completed)

    ok = (
        len(orphan_jobs) == 0
        and len(incomplete_jobs) == 0
        and len(missing_trace) == 0
        and len(missing_session) == 0
        and observability_linked
    )

    return {
        "ok": ok,
        "orphan_jobs": orphan_jobs,
        "incomplete_jobs": incomplete_jobs,
        "missing_trace_jobs": missing_trace,
        "missing_session_jobs": missing_session,
        "observability_linked": observability_linked,
        "completed_events": len(completed),
        "completed_counter": completed_counter,
    }


# ---------------------------------------------------------------------------
# LAYER 4 — Observability & Capability Inspection (read-only)
# ---------------------------------------------------------------------------

def check_observability(metrics: Any | None = None) -> dict[str, Any]:
    """Read observability counters from the metrics store."""
    started = time.perf_counter()
    try:
        metrics = metrics or get_metrics()
        completed = int(metrics.counter("scheduler.job.completed"))
        failed = int(metrics.counter("scheduler.job.failed"))
        total = completed + failed
        return {
            "ok": total > 0,
            "completed": completed,
            "failed": failed,
            "elapsed_s": round(time.perf_counter() - started, 4),
        }
    except Exception as exc:
        return {"ok": False, "detail": _format_exc(exc), "elapsed_s": round(time.perf_counter() - started, 4)}


def check_mid_execution(orchestrator: Any) -> dict[str, Any]:
    """Verify persistence capability attributes are present (read-only attribute inspection)."""
    started = time.perf_counter()
    try:
        persistence = None
        if hasattr(orchestrator, "registry") and orchestrator.registry is not None:
            persistence = orchestrator.registry.get("storage")

        has_checkpoint = callable(getattr(persistence, "save_graph_checkpoint", None))
        has_turn_event = callable(getattr(persistence, "save_turn_event", None))
        ok = has_checkpoint and has_turn_event
        return {
            "ok": ok,
            "save_graph_checkpoint": has_checkpoint,
            "save_turn_event": has_turn_event,
            "elapsed_s": round(time.perf_counter() - started, 4),
        }
    except Exception as exc:
        return {"ok": False, "detail": _format_exc(exc), "elapsed_s": round(time.perf_counter() - started, 4)}


def check_retry(orchestrator: Any) -> dict[str, Any]:
    """Verify retry policy availability (read-only attribute/import inspection)."""
    started = time.perf_counter()
    try:
        has_inline = hasattr(orchestrator, "retry_policy")
        has_fault_injection_retry = False
        try:
            from dadbot.core.fault_injection import RetryPolicy  # noqa: F401
            has_fault_injection_retry = True
        except Exception:
            has_fault_injection_retry = False

        ok = has_inline or has_fault_injection_retry
        return {
            "ok": ok,
            "orchestrator_retry_policy": has_inline,
            "fault_injection_retry": has_fault_injection_retry,
            "elapsed_s": round(time.perf_counter() - started, 4),
        }
    except Exception as exc:
        return {"ok": False, "detail": _format_exc(exc), "elapsed_s": round(time.perf_counter() - started, 4)}


def check_exporter(exporter: Any | None = None) -> dict[str, Any]:
    """Read exporter enabled state (read-only attribute inspection)."""
    started = time.perf_counter()
    try:
        exporter = exporter or get_exporter()
        enabled = bool(getattr(exporter, "enabled", getattr(exporter, "_enabled", False)))
        return {"ok": enabled, "enabled": enabled, "elapsed_s": round(time.perf_counter() - started, 4)}
    except Exception as exc:
        return {"ok": False, "detail": _format_exc(exc), "elapsed_s": round(time.perf_counter() - started, 4)}


# ---------------------------------------------------------------------------
# LAYER 5 — Module Activation Scan (source inspection — read-only)
# ---------------------------------------------------------------------------

def full_module_activation_scan(orchestrator: Any) -> dict[str, Any]:
    """Verify declared subsystems are instantiated and wired (read-only)."""
    wiring = build_wiring_map(orchestrator)
    missing_runtime = sorted([k for k, v in wiring.items() if v is None])

    registry = getattr(orchestrator, "registry", None)
    services = sorted(list(getattr(registry, "_services", {}).keys())) if registry is not None else []
    declared_aliases = _extract_registry_aliases_from_orchestrator(orchestrator)
    missing_declared_aliases = sorted([alias for alias in declared_aliases if alias not in services])

    unused_imports = _unused_import_symbols_in_orchestrator()

    ok = len(missing_runtime) == 0 and len(missing_declared_aliases) == 0 and len(unused_imports) == 0
    return {
        "ok": ok,
        "missing_runtime_components": missing_runtime,
        "declared_aliases": declared_aliases,
        "missing_declared_aliases": missing_declared_aliases,
        "unused_imports": unused_imports,
        "registered_services": services,
    }
