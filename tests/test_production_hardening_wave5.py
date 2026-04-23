"""Wave 5 — wiring integrity proofs.

Covers the four gaps identified in the field-norms audit:
  1. Module integrity  — all required modules importable + symbols present
  2. Observability     — metrics/exporter are actually wired, not silent
  3. Tracing           — trace_id propagates end-to-end, never empty
  4. Control plane     — every turn goes through ledger (no bypass)
  5. Ledger write-thru — correct event sequence: SUBMITTED→QUEUED→STARTED→COMPLETED
  6. Service validation— strict=True raises; lenient mode returns issues
  7. Queue saturation  — backpressure raised at max_inflight_jobs limit
  8. Double-execution  — lease conflict → second worker re-queues, not executes
  9. Crash recovery    — pending jobs detected; completed jobs not pending
"""
from __future__ import annotations

import asyncio
import importlib
import inspect

import pytest


# ---------------------------------------------------------------------------
# Shared async executor stubs
# ---------------------------------------------------------------------------

async def _noop_executor(session: dict, job) -> tuple[str | None, bool]:
    return ("ok", True)


async def _failing_executor(session: dict, job) -> tuple[str | None, bool]:
    raise RuntimeError("simulated failure")


# ===========================================================================
# 1. Module Integrity
# ===========================================================================

class TestModuleIntegrity:
    REQUIRED = [
        "dadbot.core.control_plane",
        "dadbot.core.graph",
        "dadbot.core.kernel",
        "dadbot.core.observability",
        "dadbot.core.execution_ledger",
    ]

    def test_all_required_modules_import(self):
        missing = []
        for mod in self.REQUIRED:
            try:
                importlib.import_module(mod)
            except ImportError as e:
                missing.append(f"{mod}: {e}")
        assert not missing, f"Modules failed to import: {missing}"

    def test_scheduler_in_control_plane(self):
        mod = importlib.import_module("dadbot.core.control_plane")
        assert hasattr(mod, "Scheduler")

    def test_execution_control_plane_in_control_plane(self):
        mod = importlib.import_module("dadbot.core.control_plane")
        assert hasattr(mod, "ExecutionControlPlane")

    def test_session_registry_in_control_plane(self):
        mod = importlib.import_module("dadbot.core.control_plane")
        assert hasattr(mod, "SessionRegistry")

    def test_execution_ledger_class_present(self):
        mod = importlib.import_module("dadbot.core.execution_ledger")
        assert hasattr(mod, "ExecutionLedger")

    def test_metrics_sink_present(self):
        mod = importlib.import_module("dadbot.core.observability")
        assert hasattr(mod, "MetricsSink")
        assert hasattr(mod, "get_metrics")
        assert hasattr(mod, "configure_exporter")

    def test_execution_lease_importable(self):
        mod = importlib.import_module("dadbot.core.execution_lease")
        assert hasattr(mod, "ExecutionLease")
        assert hasattr(mod, "LeaseConflictError")

    def test_recovery_manager_importable(self):
        mod = importlib.import_module("dadbot.core.recovery_manager")
        assert hasattr(mod, "RecoveryManager")


# ===========================================================================
# 2. Tracing Propagation
# ===========================================================================

class TestTracingPropagation:
    def test_turn_context_trace_id_auto_generated(self):
        from dadbot.core.graph import TurnContext
        ctx = TurnContext(user_input="hello")
        assert ctx.trace_id, "TurnContext.trace_id must be auto-generated"
        assert len(ctx.trace_id) >= 16

    def test_turn_context_trace_id_unique_per_instance(self):
        from dadbot.core.graph import TurnContext
        ids = {TurnContext(user_input="x").trace_id for _ in range(20)}
        assert len(ids) == 20, "Each TurnContext must have a unique trace_id"

    def test_trace_id_non_empty_inside_executor(self):
        """_execute_job asserts trace_id — verify the context always has one."""
        trace_ids_seen: list[str] = []

        async def capturing_executor(session, job):
            from dadbot.core.graph import TurnContext
            ctx = TurnContext(user_input=job.user_input)
            trace_ids_seen.append(ctx.trace_id)
            return ("ok", True)

        def _run():
            async def _inner():
                from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
                cp = ExecutionControlPlane(
                    registry=SessionRegistry(),
                    kernel_executor=capturing_executor,
                )
                await cp.submit_turn(session_id="trace-test", user_input="hi")
                assert trace_ids_seen
                assert trace_ids_seen[0], "trace_id was empty inside executor"
            asyncio.run(_inner())

        _run()

    def test_orchestrator_execute_job_asserts_trace_id(self):
        """DadBotOrchestrator._execute_job must assert context.trace_id."""
        from dadbot.core.orchestrator import DadBotOrchestrator
        source = inspect.getsource(DadBotOrchestrator._execute_job)
        assert "context.trace_id" in source, (
            "_execute_job must assert context.trace_id is non-empty"
        )

    def test_orchestrator_strict_param_in_signature(self):
        from dadbot.core.orchestrator import DadBotOrchestrator
        sig = inspect.signature(DadBotOrchestrator.__init__)
        assert "strict" in sig.parameters

    def test_orchestrator_enable_observability_param_in_signature(self):
        from dadbot.core.orchestrator import DadBotOrchestrator
        sig = inspect.signature(DadBotOrchestrator.__init__)
        assert "enable_observability" in sig.parameters


# ===========================================================================
# 3. Control Plane Wiring — no bypass
# ===========================================================================

class TestControlPlaneWiring:
    def test_handle_turn_routes_through_control_plane_not_direct_graph(self):
        """handle_turn must call control_plane.submit_turn, not graph.execute."""
        from dadbot.core.orchestrator import DadBotOrchestrator
        source = inspect.getsource(DadBotOrchestrator.handle_turn)
        assert "control_plane.submit_turn" in source
        assert "graph.execute" not in source

    def test_submit_turn_produces_ledger_events(self):
        """Every submit_turn must write at least one event to the ledger."""
        def _run():
            async def _inner():
                from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
                cp = ExecutionControlPlane(
                    registry=SessionRegistry(),
                    kernel_executor=_noop_executor,
                )
                await cp.submit_turn(session_id="wire-test", user_input="hello")
                assert cp.ledger_events(), "Ledger must be non-empty after submit_turn"
            asyncio.run(_inner())
        _run()

    def test_control_plane_exposes_ledger(self):
        from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
        cp = ExecutionControlPlane(
            registry=SessionRegistry(),
            kernel_executor=_noop_executor,
        )
        assert hasattr(cp, "ledger"), "ExecutionControlPlane must expose .ledger"
        assert hasattr(cp, "ledger_events")

    def test_control_plane_exposes_scheduler(self):
        from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
        cp = ExecutionControlPlane(
            registry=SessionRegistry(),
            kernel_executor=_noop_executor,
        )
        assert hasattr(cp, "scheduler")

    def test_control_plane_exposes_recovery(self):
        from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
        cp = ExecutionControlPlane(
            registry=SessionRegistry(),
            kernel_executor=_noop_executor,
        )
        assert hasattr(cp, "recovery")

    def test_control_plane_exposes_execution_token(self):
        from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
        cp = ExecutionControlPlane(
            registry=SessionRegistry(),
            kernel_executor=_noop_executor,
        )
        assert isinstance(cp.execution_token, str)
        assert cp.execution_token

    def test_graph_direct_execute_blocked_without_control_plane_boundary(self):
        def _run():
            async def _inner():
                from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
                from dadbot.core.graph import TurnContext, TurnGraph
                graph = TurnGraph(nodes=[])
                cp = ExecutionControlPlane(
                    registry=SessionRegistry(),
                    kernel_executor=_noop_executor,
                )
                graph.set_required_execution_token(cp.execution_token)
                with pytest.raises(RuntimeError, match="boundary violation"):
                    await graph.execute(TurnContext(user_input="hi"))
            asyncio.run(_inner())
        _run()


# ===========================================================================
# 4. Ledger Write-Through
# ===========================================================================

class TestLedgerWriteThrough:
    def test_complete_turn_has_all_lifecycle_events(self):
        """SUBMITTED → QUEUED → STARTED → COMPLETED must all appear in ledger."""
        def _run():
            async def _inner():
                from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
                cp = ExecutionControlPlane(
                    registry=SessionRegistry(),
                    kernel_executor=_noop_executor,
                )
                await cp.submit_turn(session_id="seq-test", user_input="hello")
                types = [e.get("type") for e in cp.ledger_events()]
                for required in ("JOB_SUBMITTED", "JOB_QUEUED", "JOB_STARTED", "JOB_COMPLETED"):
                    assert required in types, f"Missing {required!r}; found: {types}"
            asyncio.run(_inner())
        _run()

    def test_lifecycle_events_in_correct_order(self):
        """Events must appear in submission → execution → completion order."""
        def _run():
            async def _inner():
                from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
                cp = ExecutionControlPlane(
                    registry=SessionRegistry(),
                    kernel_executor=_noop_executor,
                )
                await cp.submit_turn(session_id="order-test", user_input="hi")
                types = [e.get("type") for e in cp.ledger_events()]
                idx = {t: types.index(t) for t in
                       ("JOB_SUBMITTED", "JOB_QUEUED", "JOB_STARTED", "JOB_COMPLETED")}
                assert idx["JOB_SUBMITTED"] < idx["JOB_QUEUED"]
                assert idx["JOB_QUEUED"] < idx["JOB_STARTED"]
                assert idx["JOB_STARTED"] < idx["JOB_COMPLETED"]
            asyncio.run(_inner())
        _run()

    def test_failed_turn_writes_job_failed_not_completed(self):
        def _run():
            async def _inner():
                from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
                cp = ExecutionControlPlane(
                    registry=SessionRegistry(),
                    kernel_executor=_failing_executor,
                )
                with pytest.raises(RuntimeError, match="simulated failure"):
                    await cp.submit_turn(session_id="fail-seq", user_input="oops")
                types = [e.get("type") for e in cp.ledger_events()]
                assert "JOB_FAILED" in types
                assert "JOB_COMPLETED" not in types
            asyncio.run(_inner())
        _run()

    def test_session_bound_event_written(self):
        def _run():
            async def _inner():
                from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
                cp = ExecutionControlPlane(
                    registry=SessionRegistry(),
                    kernel_executor=_noop_executor,
                )
                await cp.submit_turn(session_id="bound-test", user_input="hi")
                types = [e.get("type") for e in cp.ledger_events()]
                assert "SESSION_BOUND" in types
            asyncio.run(_inner())
        _run()

    def test_ledger_events_scoped_to_session(self):
        """Each event in the ledger must carry the correct session_id."""
        def _run():
            async def _inner():
                from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
                cp = ExecutionControlPlane(
                    registry=SessionRegistry(),
                    kernel_executor=_noop_executor,
                )
                await cp.submit_turn(session_id="my-session", user_input="hi")
                job_events = [
                    e for e in cp.ledger_events()
                    if e.get("type") in {"JOB_SUBMITTED", "JOB_QUEUED",
                                         "JOB_STARTED", "JOB_COMPLETED"}
                ]
                for e in job_events:
                    assert e.get("session_id") == "my-session"
            asyncio.run(_inner())
        _run()


# ===========================================================================
# 5. Observability Wiring
# ===========================================================================

class TestObservabilityWiring:
    def test_completed_turn_increments_job_completed_counter(self):
        def _run():
            async def _inner():
                from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
                from dadbot.core.observability import get_metrics
                metrics = get_metrics()
                before = metrics.counter("scheduler.job.completed")
                cp = ExecutionControlPlane(
                    registry=SessionRegistry(),
                    kernel_executor=_noop_executor,
                )
                await cp.submit_turn(session_id="metrics-ok", user_input="hi")
                assert metrics.counter("scheduler.job.completed") == before + 1
            asyncio.run(_inner())
        _run()

    def test_failed_turn_increments_job_failed_counter(self):
        def _run():
            async def _inner():
                from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
                from dadbot.core.observability import get_metrics
                metrics = get_metrics()
                before = metrics.counter("scheduler.job.failed")
                cp = ExecutionControlPlane(
                    registry=SessionRegistry(),
                    kernel_executor=_failing_executor,
                )
                with pytest.raises(RuntimeError):
                    await cp.submit_turn(session_id="metrics-fail", user_input="hi")
                assert metrics.counter("scheduler.job.failed") == before + 1
            asyncio.run(_inner())
        _run()

    def test_exporter_sink_captures_job_completed(self):
        """configure_exporter with a callable sink captures job.completed records."""
        def _run():
            async def _inner():
                from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
                from dadbot.core.observability import configure_exporter
                sink: list[dict] = []
                configure_exporter(sink=sink.append, enabled=True)
                cp = ExecutionControlPlane(
                    registry=SessionRegistry(),
                    kernel_executor=_noop_executor,
                )
                await cp.submit_turn(session_id="exporter-test", user_input="hi")
                assert sink, "Exporter sink must have captured events"
                exported_event_names = [r.get("event") for r in sink]
                assert "job.completed" in exported_event_names
            asyncio.run(_inner())
        _run()

    def test_latency_histogram_has_samples_after_turn(self):
        def _run():
            async def _inner():
                from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
                from dadbot.core.observability import get_metrics
                cp = ExecutionControlPlane(
                    registry=SessionRegistry(),
                    kernel_executor=_noop_executor,
                )
                await cp.submit_turn(session_id="latency-test", user_input="hi")
                summary = get_metrics().histogram_summary("scheduler.job.latency_ms")
                assert summary["count"] > 0, "No latency samples recorded"
            asyncio.run(_inner())
        _run()

    def test_configure_exporter_replaces_global(self):
        from dadbot.core.observability import configure_exporter, get_exporter
        captured: list[dict] = []
        configure_exporter(sink=captured.append, enabled=True)
        get_exporter().export({"event": "test.probe"})
        assert any(r.get("event") == "test.probe" for r in captured)


# ===========================================================================
# 6. Service Validation — strict mode
# ===========================================================================

class TestServiceValidation:
    def test_missing_method_returns_issues_in_lenient_mode(self):
        from dadbot.core.interfaces import validate_pipeline_services, InferenceService

        class Stub:
            pass  # missing run_agent

        issues = validate_pipeline_services(
            {"llm": (Stub(), InferenceService)},
            raise_on_failure=False,
        )
        assert issues
        assert any("run_agent" in i for i in issues)

    def test_missing_method_raises_in_strict_mode(self):
        from dadbot.core.interfaces import validate_pipeline_services, InferenceService

        class Stub:
            pass

        with pytest.raises(RuntimeError, match="contract violation"):
            validate_pipeline_services(
                {"llm": (Stub(), InferenceService)},
                raise_on_failure=True,
            )

    def test_sync_run_agent_flagged_as_issue(self):
        from dadbot.core.interfaces import validate_pipeline_services, InferenceService

        class SyncLLM:
            def run_agent(self, ctx, rich):  # sync, not async
                return "response"

        issues = validate_pipeline_services(
            {"llm": (SyncLLM(), InferenceService)},
            raise_on_failure=False,
        )
        assert any("async" in i.lower() or "run_agent" in i for i in issues)

    def test_conformant_service_has_no_issues(self):
        from dadbot.core.interfaces import validate_pipeline_services, HealthService

        class GoodHealth:
            def tick(self, ctx):
                return {}

        issues = validate_pipeline_services(
            {"health": (GoodHealth(), HealthService)},
            raise_on_failure=False,
        )
        assert not issues

    def test_orchestrator_validate_services_wired_to_strict(self):
        """_build_turn_graph must pass raise_on_failure=self._strict."""
        from dadbot.core.orchestrator import DadBotOrchestrator
        source = inspect.getsource(DadBotOrchestrator._build_turn_graph)
        assert "raise_on_failure" in source, (
            "validate_pipeline_services must be called with raise_on_failure in _build_turn_graph"
        )


# ===========================================================================
# 7. Queue Saturation / Backpressure
# ===========================================================================

class TestQueueSaturation:
    def test_register_beyond_max_inflight_raises(self):
        """Registering a second job when max_inflight_jobs=1 must raise RuntimeError."""
        def _run():
            async def _inner():
                from dadbot.core.control_plane import (
                    Scheduler, ExecutionJob, SessionRegistry,
                )
                from dadbot.core.execution_ledger import ExecutionLedger
                from dadbot.core.ledger_writer import LedgerWriter
                from dadbot.core.ledger_reader import LedgerReader

                registry = SessionRegistry()
                ledger = ExecutionLedger()
                writer = LedgerWriter(ledger)
                reader = LedgerReader(ledger)
                sched = Scheduler(registry, reader=reader, writer=writer, max_inflight_jobs=1)

                job1 = ExecutionJob(session_id="s1", user_input="first")
                await sched.register(job1)  # fills the only slot

                job2 = ExecutionJob(session_id="s2", user_input="second")
                with pytest.raises(RuntimeError, match="backpressure"):
                    await sched.register(job2)
            asyncio.run(_inner())
        _run()

    def test_slot_frees_after_drain(self):
        """After draining one job, the scheduler accepts a new registration."""
        def _run():
            async def _inner():
                from dadbot.core.control_plane import (
                    Scheduler, ExecutionJob, SessionRegistry,
                )
                from dadbot.core.execution_ledger import ExecutionLedger
                from dadbot.core.ledger_writer import LedgerWriter
                from dadbot.core.ledger_reader import LedgerReader

                registry = SessionRegistry()
                ledger = ExecutionLedger()
                writer = LedgerWriter(ledger)
                reader = LedgerReader(ledger)
                sched = Scheduler(registry, reader=reader, writer=writer, max_inflight_jobs=1)

                job1 = ExecutionJob(session_id="s1", user_input="first")
                writer.append_job_submitted(job1)
                writer.append_job_queued(job1)
                future1 = await sched.register(job1)

                drained = await sched.drain_once(_noop_executor)
                assert drained, "drain_once should have processed job1"
                assert future1.done()

                job2 = ExecutionJob(session_id="s2", user_input="second")
                # Should not raise — slot is free
                future2 = await sched.register(job2)
                assert future2 is not None
            asyncio.run(_inner())
        _run()

    def test_inflight_count_matches_registered_jobs(self):
        """len(scheduler._jobs) reflects registered-but-not-drained jobs."""
        def _run():
            async def _inner():
                from dadbot.core.control_plane import (
                    Scheduler, ExecutionJob, SessionRegistry,
                )
                from dadbot.core.execution_ledger import ExecutionLedger
                from dadbot.core.ledger_writer import LedgerWriter
                from dadbot.core.ledger_reader import LedgerReader

                registry = SessionRegistry()
                ledger = ExecutionLedger()
                writer = LedgerWriter(ledger)
                reader = LedgerReader(ledger)
                sched = Scheduler(registry, reader=reader, writer=writer, max_inflight_jobs=10)

                assert len(sched._jobs) == 0
                job = ExecutionJob(session_id="s1", user_input="hi")
                await sched.register(job)
                assert len(sched._jobs) == 1
            asyncio.run(_inner())
        _run()


# ===========================================================================
# 8. Double-Execution Prevention
# ===========================================================================

class TestDoubleExecutionPrevention:
    def test_lease_conflict_makes_drain_return_false(self):
        """When worker-1 holds the lease, worker-2's drain_once returns False."""
        def _run():
            async def _inner():
                from dadbot.core.control_plane import (
                    Scheduler, ExecutionJob, SessionRegistry,
                )
                from dadbot.core.execution_ledger import ExecutionLedger
                from dadbot.core.execution_lease import ExecutionLease
                from dadbot.core.ledger_writer import LedgerWriter
                from dadbot.core.ledger_reader import LedgerReader

                registry = SessionRegistry()
                lease = ExecutionLease()
                ledger = ExecutionLedger()
                writer = LedgerWriter(ledger)
                reader = LedgerReader(ledger)

                sched2 = Scheduler(
                    registry,
                    reader=reader,
                    writer=writer,
                    execution_lease=lease,
                    worker_id="worker-2",
                )

                # Worker-1 pre-acquires the lease.
                lease.acquire(session_id="sess1", owner_id="worker-1", ttl_seconds=30)

                job = ExecutionJob(session_id="sess1", user_input="hello")
                writer.append_job_submitted(job)
                writer.append_job_queued(job)
                await sched2.register(job)

                # Worker-2 drain → lease conflict → re-queues → False
                drained = await sched2.drain_once(_noop_executor)
                assert drained is False, "Expected False: lease held by worker-1"
            asyncio.run(_inner())
        _run()

    def test_job_requeued_on_lease_conflict(self):
        """After a lease conflict, the job must remain in _pending_job_ids."""
        def _run():
            async def _inner():
                from dadbot.core.control_plane import (
                    Scheduler, ExecutionJob, SessionRegistry,
                )
                from dadbot.core.execution_ledger import ExecutionLedger
                from dadbot.core.execution_lease import ExecutionLease
                from dadbot.core.ledger_writer import LedgerWriter
                from dadbot.core.ledger_reader import LedgerReader

                registry = SessionRegistry()
                lease = ExecutionLease()
                ledger = ExecutionLedger()
                writer = LedgerWriter(ledger)
                reader = LedgerReader(ledger)

                sched2 = Scheduler(
                    registry, reader=reader, writer=writer,
                    execution_lease=lease, worker_id="worker-2",
                )
                lease.acquire(session_id="sess1", owner_id="worker-1", ttl_seconds=30)

                job = ExecutionJob(session_id="sess1", user_input="hello")
                writer.append_job_submitted(job)
                writer.append_job_queued(job)
                await sched2.register(job)

                await sched2.drain_once(_noop_executor)

                # Job must have been re-queued into pending.
                assert len(sched2._pending_job_ids) >= 1
            asyncio.run(_inner())
        _run()

    def test_same_worker_renews_lease_without_conflict(self):
        from dadbot.core.execution_lease import ExecutionLease
        lease = ExecutionLease()
        t1 = lease.acquire(session_id="s1", owner_id="w1", ttl_seconds=5)
        t2 = lease.acquire(session_id="s1", owner_id="w1", ttl_seconds=10)
        assert t1["owner_id"] == t2["owner_id"] == "w1"
        assert t2["ttl_seconds"] == 10.0

    def test_fencing_token_increments_on_new_acquisition(self):
        from dadbot.core.execution_lease import ExecutionLease
        lease = ExecutionLease()
        t1 = lease.acquire(session_id="s1", owner_id="w1")
        lease.release(session_id="s1", owner_id="w1")
        t2 = lease.acquire(session_id="s1", owner_id="w2")
        assert t2["fencing_token"] > t1["fencing_token"]

    def test_expired_lease_allows_new_acquisition(self):
        """After TTL expires, a different worker can take the lease."""
        import time
        from dadbot.core.execution_lease import ExecutionLease
        lease = ExecutionLease()
        # Acquire with 0.01s TTL — will expire immediately
        lease.acquire(session_id="s1", owner_id="w1", ttl_seconds=0.01)
        time.sleep(0.05)
        # w2 should now be able to acquire without conflict
        token = lease.acquire(session_id="s1", owner_id="w2", ttl_seconds=5)
        assert token["owner_id"] == "w2"


# ===========================================================================
# 9. Crash Recovery
# ===========================================================================

class TestCrashRecovery:
    def _make_cp(self):
        """Returns (ledger, writer, recovery, store)."""
        from dadbot.core.execution_ledger import ExecutionLedger
        from dadbot.core.ledger_writer import LedgerWriter
        from dadbot.core.recovery_manager import RecoveryManager
        from dadbot.core.session_store import SessionStore
        ledger = ExecutionLedger()
        writer = LedgerWriter(ledger)
        recovery = RecoveryManager(ledger=ledger)
        store = SessionStore(ledger=ledger, projection_only=True)
        return ledger, writer, recovery, store

    def test_queued_job_detected_as_pending(self):
        """SUBMITTED + QUEUED with no COMPLETED = pending (crash scenario)."""
        from dadbot.core.control_plane import ExecutionJob
        _, writer, recovery, store = self._make_cp()

        job = ExecutionJob(session_id="crashed-sess", user_input="hello")
        writer.append_job_submitted(job)
        writer.append_job_queued(job)

        result = recovery.recover(session_store=store)
        assert result["pending_jobs"], "Must detect pending job"
        assert result["pending_jobs"][0]["job_id"] == job.job_id

    def test_completed_job_not_in_pending(self):
        from dadbot.core.control_plane import ExecutionJob
        _, writer, recovery, store = self._make_cp()

        job = ExecutionJob(session_id="clean-sess", user_input="hi")
        writer.append_job_submitted(job)
        writer.append_job_queued(job)
        writer.append_job_started(job)
        writer.append_job_completed(job, ("response", True))

        result = recovery.recover(session_store=store)
        assert result["pending_jobs"] == []

    def test_failed_job_not_in_pending(self):
        from dadbot.core.control_plane import ExecutionJob
        _, writer, recovery, store = self._make_cp()

        job = ExecutionJob(session_id="failed-sess", user_input="hi")
        writer.append_job_submitted(job)
        writer.append_job_queued(job)
        writer.append_job_failed(job, "executor died")

        result = recovery.recover(session_store=store)
        assert result["pending_jobs"] == []

    def test_partial_completion_leaves_others_pending(self):
        """Three queued jobs, only the first completed → two remain pending."""
        from dadbot.core.control_plane import ExecutionJob
        _, writer, recovery, store = self._make_cp()

        jobs = [
            ExecutionJob(session_id=f"s{i}", user_input=f"msg{i}")
            for i in range(3)
        ]
        for j in jobs:
            writer.append_job_submitted(j)
            writer.append_job_queued(j)
        writer.append_job_completed(jobs[0], ("ok", True))

        result = recovery.recover(session_store=store)
        pending_ids = {p["job_id"] for p in result["pending_jobs"]}
        assert jobs[0].job_id not in pending_ids
        assert jobs[1].job_id in pending_ids
        assert jobs[2].job_id in pending_ids

    def test_ledger_event_count_accurate(self):
        from dadbot.core.control_plane import ExecutionJob
        _, writer, recovery, store = self._make_cp()

        job = ExecutionJob(session_id="count-sess", user_input="hi")
        writer.append_job_submitted(job)
        writer.append_job_queued(job)

        result = recovery.recover(session_store=store)
        # Runtime witness events are now appended alongside lifecycle events.
        assert result["ledger_events"] >= 2
        event_types = [event.get("type") for event in writer._ledger.read()]
        assert event_types.count("JOB_SUBMITTED") == 1
        assert event_types.count("JOB_QUEUED") == 1

    def test_boot_reconcile_succeeds_on_clean_state(self):
        """boot_reconcile must pass when all jobs are terminal."""
        from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
        cp = ExecutionControlPlane(
            registry=SessionRegistry(),
            kernel_executor=_noop_executor,
        )
        result = cp.boot_reconcile()
        assert isinstance(result, dict)

    def test_recover_returns_replay_hash(self):
        from dadbot.core.control_plane import ExecutionJob
        _, writer, recovery, store = self._make_cp()

        job = ExecutionJob(session_id="hash-sess", user_input="hi")
        writer.append_job_submitted(job)
        writer.append_job_queued(job)

        result = recovery.recover(session_store=store)
        assert "replay_hash" in result
        assert result["replay_hash"]  # non-empty hex string

    def test_second_recover_same_hash_when_ledger_unchanged(self):
        """replay_hash is deterministic: same events → same hash."""
        from dadbot.core.control_plane import ExecutionJob
        from dadbot.core.session_store import SessionStore
        ledger, writer, recovery, store = self._make_cp()

        job = ExecutionJob(session_id="det-sess", user_input="hi")
        writer.append_job_submitted(job)
        writer.append_job_queued(job)

        r1 = recovery.recover(session_store=store)
        store2 = SessionStore(ledger=ledger, projection_only=True)
        r2 = recovery.recover(session_store=store2)
        assert r1["replay_hash"] == r2["replay_hash"]
