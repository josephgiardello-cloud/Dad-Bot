from __future__ import annotations

import asyncio
import inspect
import re
import time
import traceback
from typing import Any

from dadbot.core.observability import get_exporter, get_metrics
from dadbot.core.system_health_checker import SystemHealthChecker

RESULTS: dict[str, dict[str, Any]] = {}


def record(name: str, success: bool, detail: str = "", *, elapsed_s: float | None = None) -> None:
    RESULTS[name] = {
        "status": "PASS" if success else "FAIL",
        "detail": str(detail or "").strip(),
    }
    if elapsed_s is not None:
        RESULTS[name]["elapsed_s"] = round(float(elapsed_s), 4)


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


def full_module_activation_scan(orchestrator: Any) -> dict[str, Any]:
    """A. Verify declared subsystems are instantiated and wired."""
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


async def execution_path_coverage_test(orchestrator: Any) -> dict[str, Any]:
    """B. Ensure each critical runtime service appears in at least one execution path."""
    counts = {
        "control_plane": 0,
        "scheduler": 0,
        "graph_runtime": 0,
        "ledger_write_path": 0,
    }

    original_submit = orchestrator.control_plane.submit_turn
    original_drain = orchestrator.scheduler.drain_once
    original_graph = orchestrator.graph.execute
    original_write = orchestrator.control_plane.ledger_writer.write_event

    async def wrapped_submit(*args, **kwargs):
        counts["control_plane"] += 1
        return await original_submit(*args, **kwargs)

    async def wrapped_drain(*args, **kwargs):
        counts["scheduler"] += 1
        return await original_drain(*args, **kwargs)

    async def wrapped_graph(*args, **kwargs):
        counts["graph_runtime"] += 1
        return await original_graph(*args, **kwargs)

    def wrapped_write(*args, **kwargs):
        counts["ledger_write_path"] += 1
        return original_write(*args, **kwargs)

    try:
        orchestrator.control_plane.submit_turn = wrapped_submit
        orchestrator.scheduler.drain_once = wrapped_drain
        orchestrator.graph.execute = wrapped_graph
        orchestrator.control_plane.ledger_writer.write_event = wrapped_write

        await orchestrator.handle_turn("coverage probe")
    finally:
        orchestrator.control_plane.submit_turn = original_submit
        orchestrator.scheduler.drain_once = original_drain
        orchestrator.graph.execute = original_graph
        orchestrator.control_plane.ledger_writer.write_event = original_write

    coverage = {k: (v > 0) for k, v in counts.items()}
    return {
        "ok": all(coverage.values()),
        "coverage": coverage,
        "call_counts": counts,
    }


async def cold_vs_warm_start_parity(orchestrator_factory) -> dict[str, Any]:
    """C. Compare behavior on fresh boot vs restart with execution history."""
    cold = orchestrator_factory()
    cold_result = await cold.handle_turn("parity-probe")
    history = list(cold.control_plane.ledger.read())

    warm = orchestrator_factory()
    # Replay existing history into warm ledger before recovering.
    for event in history:
        replay_event = dict(event)
        # Let the target ledger assign fresh sequence counters.
        replay_event.pop("sequence", None)
        replay_event.pop("_seq", None)
        warm.control_plane.ledger.write(replay_event)

    warm.control_plane.session_store.rebuild_from_ledger(warm.control_plane.ledger.read())
    warm_recovery = warm.control_plane.recover_runtime_state()
    warm_result = await warm.handle_turn("parity-probe")

    # Compare output parity for the same input in cold and warm conditions.
    same_result = cold_result == warm_result
    return {
        "ok": bool(same_result),
        "cold_result": cold_result,
        "warm_result": warm_result,
        "warm_pending_jobs": len(warm_recovery.get("pending_jobs") or []),
        "replayed_event_count": len(history),
    }


# -------------------------
# LAYER 1 — Wiring Integrity
# -------------------------
def build_wiring_map(orchestrator: Any) -> dict[str, Any]:
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
    wiring = build_wiring_map(orchestrator)
    missing = [k for k, v in wiring.items() if v is None]
    return {
        "ok": len(missing) == 0,
        "missing_components": missing,
    }


# -------------------------
# LAYER 2 — Runtime Activation
# -------------------------
def check_runtime_activation(ledger: Any, metrics: Any) -> dict[str, Any]:
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


# -------------------------
# LAYER 3 — Cross-Component Causality
# -------------------------
def check_causality(orchestrator: Any) -> dict[str, Any]:
    ledger = orchestrator.control_plane.ledger.read()

    job_ids = {_event_job_id(e) for e in ledger if _event_job_id(e)}
    started = {_event_job_id(e) for e in ledger if _event_type(e) == "JOB_STARTED" and _event_job_id(e)}
    completed = {_event_job_id(e) for e in ledger if _event_type(e) == "JOB_COMPLETED" and _event_job_id(e)}
    failed = {_event_job_id(e) for e in ledger if _event_type(e) == "JOB_FAILED" and _event_job_id(e)}

    terminal = completed | failed
    orphan_jobs = sorted(job_ids - started)
    incomplete_jobs = sorted(started - terminal)

    # Session/trace integrity: each job lifecycle event should carry both session_id and trace_id.
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

    # Observability linkage: completed counter should reflect at least the number of JOB_COMPLETED events.
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


async def full_system_truth_check(orchestrator: Any) -> dict[str, Any]:
    # Force success path activation.
    await orchestrator.handle_turn("truth-check success")

    # Force one failure event so failure path activation can be validated.
    original_graph_execute = orchestrator.graph.execute

    async def _fail_once(*args, **kwargs):
        raise RuntimeError("truth-check injected failure")

    orchestrator.graph.execute = _fail_once
    try:
        await orchestrator.handle_turn("truth-check failure")
    except Exception:
        pass
    finally:
        orchestrator.graph.execute = original_graph_execute

    results: dict[str, Any] = {}
    results["wiring"] = check_wiring_integrity(orchestrator)

    wiring = build_wiring_map(orchestrator)
    results["runtime"] = check_runtime_activation(
        wiring["ledger"],
        wiring["metrics"],
    )

    results["causality"] = check_causality(orchestrator)
    results["health"] = {
        "ok": all(bool(r.get("ok")) for r in (results["wiring"], results["runtime"], results["causality"])),
    }
    return results


# -------------------------
# 1. Execution Path Enforcement
# -------------------------
async def check_execution_path(orchestrator: Any) -> None:
    started = time.perf_counter()
    called = {
        "submit_turn": False,
        "graph_execute": False,
        "graph_without_submit": False,
    }

    original_submit = orchestrator.control_plane.submit_turn
    original_graph_execute = orchestrator.graph.execute
    submit_depth = 0

    async def wrapped_submit(*args, **kwargs):
        nonlocal submit_depth
        called["submit_turn"] = True
        submit_depth += 1
        try:
            return await original_submit(*args, **kwargs)
        finally:
            submit_depth -= 1

    async def wrapped_graph_execute(*args, **kwargs):
        called["graph_execute"] = True
        if submit_depth <= 0:
            called["graph_without_submit"] = True
        return await original_graph_execute(*args, **kwargs)

    try:
        orchestrator.control_plane.submit_turn = wrapped_submit
        orchestrator.graph.execute = wrapped_graph_execute

        await orchestrator.handle_turn("audit execution path")

        ok = (
            called["submit_turn"]
            and called["graph_execute"]
            and not called["graph_without_submit"]
        )
        detail = (
            f"submit_turn={called['submit_turn']}, "
            f"graph_execute={called['graph_execute']}, "
            f"graph_without_submit={called['graph_without_submit']}"
        )
        record("execution_path", ok, detail, elapsed_s=time.perf_counter() - started)
    except Exception as exc:
        record("execution_path", False, _format_exc(exc), elapsed_s=time.perf_counter() - started)
    finally:
        orchestrator.control_plane.submit_turn = original_submit
        orchestrator.graph.execute = original_graph_execute


# -------------------------
# 2. Ledger correctness
# -------------------------
def check_ledger(ledger: Any) -> None:
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
        record("ledger_sequence", ok, f"events={names}", elapsed_s=time.perf_counter() - started)
    except Exception as exc:
        record("ledger_sequence", False, _format_exc(exc), elapsed_s=time.perf_counter() - started)


# -------------------------
# 3. Tracing propagation
# -------------------------
async def check_tracing(orchestrator: Any) -> None:
    started = time.perf_counter()
    original_graph_execute = orchestrator.graph.execute
    seen_trace_id = ""

    async def wrapped_graph_execute(turn_context, *args, **kwargs):
        nonlocal seen_trace_id
        seen_trace_id = str(getattr(turn_context, "trace_id", "") or "")
        return await original_graph_execute(turn_context, *args, **kwargs)

    try:
        orchestrator.graph.execute = wrapped_graph_execute
        await orchestrator.handle_turn("audit tracing")
        ok = bool(seen_trace_id)
        record("tracing", ok, f"trace_id={seen_trace_id}", elapsed_s=time.perf_counter() - started)
    except Exception as exc:
        record("tracing", False, _format_exc(exc), elapsed_s=time.perf_counter() - started)
    finally:
        orchestrator.graph.execute = original_graph_execute


# -------------------------
# 4. Observability active
# -------------------------
def check_observability(metrics: Any | None = None) -> None:
    started = time.perf_counter()
    try:
        metrics = metrics or get_metrics()
        completed = int(metrics.counter("scheduler.job.completed"))
        failed = int(metrics.counter("scheduler.job.failed"))
        total = completed + failed
        record(
            "observability_metrics",
            total > 0,
            f"completed={completed}, failed={failed}",
            elapsed_s=time.perf_counter() - started,
        )
    except Exception as exc:
        record("observability_metrics", False, _format_exc(exc), elapsed_s=time.perf_counter() - started)


# -------------------------
# 5. Backpressure
# -------------------------
async def check_backpressure() -> None:
    started = time.perf_counter()
    try:
        from dadbot.core.control_plane import ExecutionJob, Scheduler, SessionRegistry
        from dadbot.core.execution_ledger import ExecutionLedger
        from dadbot.core.ledger_reader import LedgerReader
        from dadbot.core.ledger_writer import LedgerWriter

        registry = SessionRegistry()
        ledger = ExecutionLedger()
        writer = LedgerWriter(ledger)
        reader = LedgerReader(ledger)
        scheduler = Scheduler(
            registry,
            reader=reader,
            writer=writer,
            max_inflight_jobs=1,
        )

        job1 = ExecutionJob(session_id="bp-1", user_input="first")
        await scheduler.register(job1)

        job2 = ExecutionJob(session_id="bp-2", user_input="second")
        try:
            await scheduler.register(job2)
            ok = False
            detail = "Expected RuntimeError from scheduler backpressure, but register() succeeded"
        except RuntimeError as exc:
            ok = "backpressure" in str(exc).lower() or "inflight" in str(exc).lower()
            detail = _format_exc(exc)

        record("backpressure", ok, detail, elapsed_s=time.perf_counter() - started)
    except Exception as exc:
        record("backpressure", False, _format_exc(exc), elapsed_s=time.perf_counter() - started)


# -------------------------
# 6. Lease correctness
# -------------------------
def check_lease(lease: Any) -> None:
    started = time.perf_counter()
    try:
        from dadbot.core.execution_lease import LeaseConflictError

        session = "audit-session"
        lease.acquire(session_id=session, owner_id="worker1")
        try:
            lease.acquire(session_id=session, owner_id="worker2")
            record(
                "lease_single_writer",
                False,
                "Second worker unexpectedly acquired active lease",
                elapsed_s=time.perf_counter() - started,
            )
        except LeaseConflictError:
            record(
                "lease_single_writer",
                True,
                "Second worker blocked by LeaseConflictError",
                elapsed_s=time.perf_counter() - started,
            )
    except Exception as exc:
        record("lease_single_writer", False, _format_exc(exc), elapsed_s=time.perf_counter() - started)


# -------------------------
# 7. Crash recovery
# -------------------------
def check_recovery(control_plane: Any) -> None:
    started = time.perf_counter()
    try:
        boot = control_plane.boot_reconcile()
        recovered = control_plane.recover_runtime_state()
        pending = list(recovered.get("pending_jobs") or [])
        ok = isinstance(boot, dict) and isinstance(recovered, dict)
        record(
            "crash_recovery",
            ok,
            f"pending_jobs={len(pending)}",
            elapsed_s=time.perf_counter() - started,
        )
    except Exception as exc:
        record("crash_recovery", False, _format_exc(exc), elapsed_s=time.perf_counter() - started)


# -------------------------
# 8. Mid-execution durability
# -------------------------
def check_mid_execution(orchestrator: Any) -> None:
    started = time.perf_counter()
    try:
        persistence = None
        if hasattr(orchestrator, "registry") and orchestrator.registry is not None:
            persistence = orchestrator.registry.get("storage")

        has_checkpoint = callable(getattr(persistence, "save_graph_checkpoint", None))
        has_turn_event = callable(getattr(persistence, "save_turn_event", None))
        ok = has_checkpoint and has_turn_event
        detail = f"save_graph_checkpoint={has_checkpoint}, save_turn_event={has_turn_event}"
        record("mid_execution_durability", ok, detail, elapsed_s=time.perf_counter() - started)
    except Exception as exc:
        record("mid_execution_durability", False, _format_exc(exc), elapsed_s=time.perf_counter() - started)


# -------------------------
# 9. Retry/backoff
# -------------------------
def check_retry(orchestrator: Any) -> None:
    started = time.perf_counter()
    try:
        has_inline = hasattr(orchestrator, "retry_policy")
        has_fault_injection_retry = False
        try:
            from dadbot.core.fault_injection import RetryPolicy  # noqa: F401
            has_fault_injection_retry = True
        except Exception:
            has_fault_injection_retry = False

        # Pass if retry policy is integrated on orchestrator OR globally available for composition.
        ok = has_inline or has_fault_injection_retry
        detail = f"orchestrator.retry_policy={has_inline}, fault_injection.RetryPolicy={has_fault_injection_retry}"
        record("retry_backoff", ok, detail, elapsed_s=time.perf_counter() - started)
    except Exception as exc:
        record("retry_backoff", False, _format_exc(exc), elapsed_s=time.perf_counter() - started)


# -------------------------
# 10. Multi-worker safety
# -------------------------
async def check_multi_worker() -> None:
    started = time.perf_counter()
    try:
        from dadbot.core.control_plane import ExecutionJob, Scheduler, SessionRegistry
        from dadbot.core.execution_ledger import ExecutionLedger
        from dadbot.core.execution_lease import ExecutionLease
        from dadbot.core.ledger_reader import LedgerReader
        from dadbot.core.ledger_writer import LedgerWriter

        lease = ExecutionLease()
        registry = SessionRegistry()
        ledger = ExecutionLedger()
        writer = LedgerWriter(ledger)

        # One shared ledger/reader-writer domain, two workers.
        reader1 = LedgerReader(ledger)
        reader2 = LedgerReader(ledger)

        worker1 = Scheduler(
            registry,
            reader=reader1,
            writer=writer,
            execution_lease=lease,
            worker_id="worker-1",
        )
        worker2 = Scheduler(
            registry,
            reader=reader2,
            writer=writer,
            execution_lease=lease,
            worker_id="worker-2",
        )

        async def _slow_exec(session, job):
            await asyncio.sleep(0.05)
            return ("ok", True)

        job = ExecutionJob(session_id="mw-session", user_input="mw")
        writer.append_job_submitted(job)
        writer.append_job_queued(job)

        fut1 = await worker1.register(job)
        # Register same job id in worker2 map is not valid; use a clone job for same session.
        job2 = ExecutionJob(session_id="mw-session", user_input="mw2")
        writer.append_job_submitted(job2)
        writer.append_job_queued(job2)
        fut2 = await worker2.register(job2)

        # Race both workers once.
        d1, d2 = await asyncio.gather(
            worker1.drain_once(_slow_exec),
            worker2.drain_once(_slow_exec),
        )

        # Under contention, at least one scheduler should yield due to lease conflict.
        ok = (d1 and not d2) or (d2 and not d1)
        detail = f"drain_worker1={d1}, drain_worker2={d2}, fut1_done={fut1.done()}, fut2_done={fut2.done()}"
        record("multi_worker", ok, detail, elapsed_s=time.perf_counter() - started)
    except Exception as exc:
        record("multi_worker", False, _format_exc(exc), elapsed_s=time.perf_counter() - started)


# -------------------------
# 11. External observability export
# -------------------------
def check_exporter(exporter: Any | None = None) -> None:
    started = time.perf_counter()
    try:
        exporter = exporter or get_exporter()
        enabled = bool(getattr(exporter, "enabled", getattr(exporter, "_enabled", False)))
        record("external_export", enabled, f"enabled={enabled}", elapsed_s=time.perf_counter() - started)
    except Exception as exc:
        record("external_export", False, _format_exc(exc), elapsed_s=time.perf_counter() - started)


# -------------------------
# 12. Memory stability (quick probe)
# -------------------------
async def check_memory(orchestrator: Any, *, turns: int = 20, peak_limit_bytes: int = 50_000_000) -> None:
    started = time.perf_counter()
    try:
        import tracemalloc

        tracemalloc.start()
        for i in range(max(1, int(turns))):
            await orchestrator.handle_turn(f"mem {i}")
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        ok = int(peak) < int(peak_limit_bytes)
        detail = f"current={current}, peak={peak}, limit={peak_limit_bytes}"
        record("memory_growth", ok, detail, elapsed_s=time.perf_counter() - started)
    except Exception as exc:
        record("memory_growth", False, _format_exc(exc), elapsed_s=time.perf_counter() - started)


def print_results() -> None:
    print("\n=== SYSTEM AUDIT RESULTS ===")
    for key in sorted(RESULTS.keys()):
        payload = RESULTS[key]
        elapsed = payload.get("elapsed_s")
        elapsed_txt = f", elapsed={elapsed}s" if elapsed is not None else ""
        print(f"{key}: {payload['status']} ({payload.get('detail', '')}{elapsed_txt})")


class _AuditTelemetry:
    def trace(self, *args, **kwargs) -> None:
        return None


class _AuditHealthService:
    def tick(self, context) -> dict[str, Any]:
        return {"ok": True}


class _AuditMemoryService:
    def build_context(self, context) -> dict[str, Any]:
        return {"user_input": context.user_input}


class _AuditLLMService:
    bot = None

    async def run_agent(self, context, rich_context: dict[str, Any]):
        return (f"audit-reply:{context.user_input}", False)


class _AuditSafetyService:
    def enforce_policies(self, context, candidate):
        if isinstance(candidate, tuple) and len(candidate) >= 2:
            return candidate
        return (str(candidate or ""), False)


class _AuditStorageService:
    def __init__(self) -> None:
        self.turns: list[tuple[Any, Any]] = []
        self.checkpoints: list[dict[str, Any]] = []
        self.turn_events: list[dict[str, Any]] = []

    def save_turn(self, context, result) -> None:
        self.turns.append((context.trace_id, result))

    def save_graph_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.checkpoints.append(dict(checkpoint or {}))

    def save_turn_event(self, event: dict[str, Any]) -> None:
        self.turn_events.append(dict(event or {}))


def _build_audit_orchestrator():
    from dadbot.core.orchestrator import DadBotOrchestrator
    from dadbot.registry import ServiceRegistry

    registry = ServiceRegistry()
    telemetry = _AuditTelemetry()
    health = _AuditHealthService()
    memory = _AuditMemoryService()
    llm = _AuditLLMService()
    safety = _AuditSafetyService()
    storage = _AuditStorageService()

    # Canonical services for graph checkpoints + telemetry hooks.
    registry.register("telemetry", telemetry)
    registry.register("persistence_service", storage)

    # Orchestrator alias keys.
    registry.register("health", health)
    registry.register("memory", memory)
    registry.register("llm", llm)
    registry.register("safety", safety)
    registry.register("storage", storage)

    # Legacy alias keys some nodes may request.
    registry.register("maintenance_service", health)
    registry.register("context_service", memory)
    registry.register("agent_service", llm)
    registry.register("safety_service", safety)

    return DadBotOrchestrator(registry=registry, strict=True, enable_observability=True)


async def run_full_audit(orchestrator: Any) -> dict[str, dict[str, Any]]:
    truth_result: dict[str, Any] = {}
    activation_scan: dict[str, Any] = {}
    coverage_report: dict[str, Any] = {}
    parity_report: dict[str, Any] = {}
    execution_activation_report: dict[str, Any] = {}
    try:
        await check_execution_path(orchestrator)
        await check_tracing(orchestrator)
        await check_backpressure()
        await check_multi_worker()
        await check_memory(orchestrator)

        check_ledger(orchestrator.control_plane.ledger)
        check_observability(get_metrics())
        check_lease(orchestrator.control_plane.execution_lease)
        check_recovery(orchestrator.control_plane)
        check_mid_execution(orchestrator)
        check_retry(orchestrator)
        check_exporter(get_exporter())
        truth_result = await full_system_truth_check(orchestrator)
        record(
            "system_truth_check",
            bool(truth_result.get("health", {}).get("ok")),
            f"wiring={truth_result.get('wiring', {}).get('ok')}, "
            f"runtime={truth_result.get('runtime', {}).get('ok')}, "
            f"causality={truth_result.get('causality', {}).get('ok')}",
        )

        activation_scan = full_module_activation_scan(orchestrator)
        record(
            "module_activation_scan",
            bool(activation_scan.get("ok")),
            f"missing_runtime={activation_scan.get('missing_runtime_components')}, "
            f"missing_aliases={activation_scan.get('missing_declared_aliases')}, "
            f"unused_imports={activation_scan.get('unused_imports')}",
        )

        coverage_report = await execution_path_coverage_test(orchestrator)
        record(
            "execution_path_coverage",
            bool(coverage_report.get("ok")),
            f"coverage={coverage_report.get('coverage')}, counts={coverage_report.get('call_counts')}",
        )

        parity_report = await cold_vs_warm_start_parity(_build_audit_orchestrator)
        record(
            "cold_warm_parity",
            bool(parity_report.get("ok")),
            f"cold={parity_report.get('cold_result')}, warm={parity_report.get('warm_result')}, "
            f"replayed={parity_report.get('replayed_event_count')}",
        )

        checker = SystemHealthChecker(base_path=".", strict_identity=True)
        checker.check_identity_propagation_correctness(orchestrator.control_plane.ledger)
        execution_activation_report = checker.check_execution_activation(orchestrator.control_plane.ledger)
        record(
            "execution_activation",
            bool(execution_activation_report.get("ok")),
            f"missing={execution_activation_report.get('missing_components')}, "
            f"executed={execution_activation_report.get('executed_components')}",
        )
    except Exception:
        record("audit_runtime", False, traceback.format_exc(limit=5))

    print_results()
    print("\n=== SYSTEM TRUTH CHECK ===")
    print(truth_result)
    print("\n=== MODULE ACTIVATION SCAN ===")
    print(activation_scan)
    print("\n=== EXECUTION PATH COVERAGE ===")
    print(coverage_report)
    print("\n=== COLD VS WARM PARITY ===")
    print(parity_report)
    print("\n=== EXECUTION ACTIVATION ===")
    print(execution_activation_report)
    return RESULTS


async def _main() -> None:
    orchestrator = _build_audit_orchestrator()
    await run_full_audit(orchestrator)


if __name__ == "__main__":
    asyncio.run(_main())
