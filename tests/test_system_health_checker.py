from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.session_store import SessionStore
from dadbot.core.system_health_checker import SystemHealthChecker


def test_health_checker_passes_basic_ledger_invariants():
    ledger = ExecutionLedger()
    session_store = SessionStore()

    ledger.write(
        {
            "type": "JOB_QUEUED",
            "session_id": "s1",
            "trace_id": "t1",
            "timestamp": 1.0,
            "kernel_step_id": "control_plane.enqueue",
            "payload": {"job_id": "j1"},
        }
    )
    ledger.write(
        {
            "type": "JOB_STARTED",
            "session_id": "s1",
            "trace_id": "t1",
            "timestamp": 2.0,
            "kernel_step_id": "inference",
            "payload": {"job_id": "j1"},
        }
    )
    completed = ledger.write(
        {
            "type": "JOB_COMPLETED",
            "session_id": "s1",
            "trace_id": "t1",
            "timestamp": 3.0,
            "kernel_step_id": "save",
            "payload": {"job_id": "j1", "result": ("ok", False)},
        }
    )
    session_store.rebuild_from_ledger([completed])

    checker = SystemHealthChecker(base_path=".")
    completeness = checker.check_ledger_completeness(ledger)
    ordering = checker.check_event_ordering_integrity(ledger)
    projection = checker.check_session_store_consistency(ledger=ledger, session_store=session_store)

    assert completeness["ok"] is True
    assert ordering["ok"] is True
    assert projection["ok"] is True


def test_health_checker_detects_missing_start_before_finish():
    ledger = ExecutionLedger()
    ledger.write(
        {
            "type": "JOB_COMPLETED",
            "session_id": "s2",
            "trace_id": "t2",
            "timestamp": 10.0,
            "kernel_step_id": "save",
            "payload": {"job_id": "j2", "result": ("ok", False)},
        }
    )

    checker = SystemHealthChecker(base_path=".")
    report = checker.check_ledger_completeness(ledger)

    assert report["ok"] is False
    assert report["missing_start_before_finish"] == ["j2"]


def test_health_checker_identity_check_passes_with_trace_ids():
    ledger = ExecutionLedger()
    ledger.write(
        {
            "type": "JOB_QUEUED",
            "session_id": "s3",
            "trace_id": "trace-1",
            "timestamp": 1.0,
            "kernel_step_id": "control_plane.enqueue",
            "payload": {"job_id": "j3"},
        }
    )
    ledger.write(
        {
            "type": "JOB_STARTED",
            "session_id": "s3",
            "trace_id": "trace-1",
            "timestamp": 2.0,
            "kernel_step_id": "scheduler.execute.start",
            "payload": {"job_id": "j3"},
        }
    )

    checker = SystemHealthChecker(base_path=".")
    identity = checker.check_identity_propagation_correctness(ledger)
    assert identity["ok"] is True


def test_health_checker_run_all_includes_global_invariant_contract():
    ledger = ExecutionLedger()
    session_store = SessionStore()
    ledger.write(
        {
            "type": "JOB_QUEUED",
            "session_id": "sx",
            "trace_id": "tx",
            "timestamp": 1.0,
            "kernel_step_id": "control_plane.enqueue",
            "payload": {"job_id": "jx"},
        }
    )
    checker = SystemHealthChecker(base_path=".")
    report = checker.run_all(ledger=ledger, session_store=session_store)
    assert "global_invariant_contract" in report
    assert "contract_hash" in report["global_invariant_contract"]


def test_execution_activation_detects_missing_components():
    ledger = ExecutionLedger()
    checker = SystemHealthChecker(base_path=".")

    ledger.write(
        {
            "type": "EXECUTION_WITNESS",
            "session_id": "s-activation",
            "trace_id": "trace-activation",
            "timestamp": 1.0,
            "kernel_step_id": "runtime.execution_witness",
            "payload": {"component": "control_plane.submit_turn"},
        }
    )

    report = checker.check_execution_activation(ledger)
    assert report["ok"] is False
    assert "control_plane.submit_turn" in report["executed_components"]
    assert "graph.execute" in report["missing_components"]


def test_execution_activation_passes_when_all_expected_components_witnessed():
    ledger = ExecutionLedger()
    checker = SystemHealthChecker(base_path=".")

    for idx, component in enumerate(sorted(checker._expected_components()), start=1):
        ledger.write(
            {
                "type": "EXECUTION_WITNESS",
                "session_id": f"s-{idx}",
                "trace_id": f"trace-{idx}",
                "timestamp": float(idx),
                "kernel_step_id": "runtime.execution_witness",
                "payload": {"component": component},
            }
        )

    report = checker.check_execution_activation(ledger)
    assert report["ok"] is True
    assert report["missing_components"] == []
