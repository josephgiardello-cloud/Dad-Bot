from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
from dadbot.core.distributed_correctness import NodeRole
from dadbot.core.ledger_writer import LedgerWriter
from dadbot.core.runtime_errors import AuthorityViolation, InvariantViolation


@pytest.mark.asyncio
async def test_submit_turn_triggers_ledger_compaction_report(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DADBOT_LEDGER_MAX_EVENTS", "1")
    monkeypatch.setenv("DADBOT_LEDGER_MIN_SNAPSHOT_DISTANCE", "0")
    monkeypatch.setenv("DADBOT_LEDGER_ARCHIVE_DIR", str(tmp_path / "archives"))

    async def _executor(_session: dict, _job) -> tuple[str, bool]:
        return ("ok", False)

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)
    await plane.submit_turn(session_id="s-1", user_input="hello")

    report = dict(plane._last_compaction_report)
    assert "compacted" in report
    assert "event_count" in report
    assert "lossless_proof" in report


def test_distributed_runtime_authority_rejects_split_brain_claim_path() -> None:
    async def _executor(_session: dict, _job) -> tuple[str, bool]:
        return ("ok", False)

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)
    now_ms = plane._distributed_now_ms()
    plane._distributed_correctness.register_node(
        node_id="shadow-leader",
        epoch=plane._distributed_epoch,
        lease_until_ms=now_ms + 60_000,
        role=NodeRole.LEADER,
        state_hash="split",
    )

    with pytest.raises(AuthorityViolation):
        plane._enforce_distributed_runtime_authority(operation="scheduler_claim")


def test_distributed_runtime_authority_rejects_non_authoritative_replay_path() -> None:
    async def _executor(_session: dict, _job) -> tuple[str, bool]:
        return ("ok", False)

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)
    now_ms = plane._distributed_now_ms()
    # Expire the local leader lease first so the failure mode is non-authoritative,
    # not split-brain.
    plane._distributed_correctness.register_node(
        node_id=plane._scheduler.worker_id,
        epoch=plane._distributed_epoch,
        lease_until_ms=now_ms - 1,
        role=NodeRole.LEADER,
        state_hash="stale",
    )
    plane._distributed_correctness.register_node(
        node_id="higher-epoch-leader",
        epoch=plane._distributed_epoch + 1,
        lease_until_ms=now_ms + 60_000,
        role=NodeRole.LEADER,
        state_hash="authoritative",
    )

    with pytest.raises(AuthorityViolation):
        plane._enforce_distributed_runtime_authority(operation="replay_acceptance_gate")


def test_control_plane_binds_graph_execution_token() -> None:
    class _GraphProbe:
        def __init__(self) -> None:
            self.bound_token = ""

        def set_required_execution_token(self, execution_token: str) -> None:
            self.bound_token = str(execution_token or "")

    async def _executor(_session: dict, _job) -> tuple[str, bool]:
        return ("ok", False)

    graph_probe = _GraphProbe()
    plane = ExecutionControlPlane(
        registry=SessionRegistry(),
        kernel_executor=_executor,
        graph=graph_probe,
    )
    assert graph_probe.bound_token == plane.execution_token


@pytest.mark.asyncio
async def test_submit_turn_raises_when_commit_boundary_invariant_fails(monkeypatch) -> None:
    async def _executor(_session: dict, _job) -> tuple[str, bool]:
        return ("ok", False)

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)

    def _fake_counts(_trace_events: list[dict[str, object]]) -> tuple[int, bool]:
        return (2, True)

    monkeypatch.setattr(type(plane), "_trace_event_invariant_counts", staticmethod(_fake_counts))

    with pytest.raises(InvariantViolation):
        await plane.submit_turn(session_id="s-commit-boundary", user_input="hello")


@pytest.mark.asyncio
async def test_boot_reconcile_exposes_partition_summary(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DADBOT_LEDGER_MAX_EVENTS", "2")
    monkeypatch.setenv("DADBOT_LEDGER_MIN_SNAPSHOT_DISTANCE", "0")
    monkeypatch.setenv("DADBOT_LEDGER_ARCHIVE_DIR", str(tmp_path / "archives"))

    async def _executor(_session: dict, _job) -> tuple[str, bool]:
        return ("ok", False)

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)
    await plane.submit_turn(session_id="s-1", user_input="one")
    await plane.submit_turn(session_id="s-2", user_input="two")

    status = plane.boot_reconcile()
    partitioning = dict(status.get("ledger_partitioning") or {})
    assert int(partitioning.get("partition_count", 0)) >= 1
    assert isinstance(partitioning.get("top_partitions"), list)


@pytest.mark.asyncio
async def test_submit_turn_records_execution_composition_contract() -> None:
    async def _executor(session: dict, _job) -> tuple[str, bool]:
        state = session.setdefault("state", {})
        state["last_terminal_state"] = {
            "execution_dag_hash": "dag-h",
            "policy_hash": "pol-h",
            "post_commit_mutation_effects_hash": "mut-h",
            "determinism_closure_hash": "det-h",
        }
        return ("ok", False)

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)
    await plane.submit_turn(session_id="s-1", user_input="hello")

    session = plane.registry.get_or_create("s-1")
    state = dict(session.get("state") or {})
    contract = dict(state.get("last_execution_composition_contract") or {})
    assert contract.get("contract_version") == "turn-composition-v1"
    assert bool(contract.get("composition_hash"))
    assert bool(contract.get("event_log_hash"))
    assert bool(contract.get("confluence_class_hash"))
    events = list(plane.ledger_events())
    completed = [e for e in events if str(e.get("type") or "") == "JOB_COMPLETED"]
    assert completed, "expected JOB_COMPLETED ledger event"
    execution_result = dict((completed[-1].get("payload") or {}).get("metadata", {}).get("execution_result") or {})
    assert execution_result.get("status") == "ok"
    outputs = dict(execution_result.get("outputs") or {})
    assert outputs.get("response") == "ok"
    assert bool(outputs.get("should_end")) is False


@pytest.mark.asyncio
async def test_submit_turn_records_core_state() -> None:
    async def _executor(session: dict, _job) -> tuple[str, bool]:
        state = session.setdefault("state", {})
        state["last_terminal_state"] = {
            "execution_dag_hash": "dag-r",
            "policy_hash": "pol-r",
            "post_commit_mutation_effects_hash": "mut-r",
            "determinism_closure_hash": "det-r",
        }
        return ("ok", False)

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)
    await plane.submit_turn(session_id="s-runtime-correctness", user_input="hello")

    session = plane.registry.get_or_create("s-runtime-correctness")
    state = dict(session.get("state") or {})

    core_state = dict(state.get("core_state") or {})
    views = dict(state.get("core_state_views") or {})
    assert int(core_state.get("version", 0)) >= 1
    assert bool(dict(views.get("facade") or {}).get("state_hash"))
    assert int(dict(views.get("canonical") or {}).get("event_count", 0)) >= 1


@pytest.mark.asyncio
async def test_global_confluence_law_binds_first_observation_and_reuses_expected_hash(monkeypatch) -> None:
    monkeypatch.setenv("DADBOT_GLOBAL_CONFLUENCE_MODE", "enforce")

    async def _executor(session: dict, _job) -> tuple[str, bool]:
        state = session.setdefault("state", {})
        state["last_terminal_state"] = {
            "execution_dag_hash": "dag-c",
            "policy_hash": "pol-c",
            "post_commit_mutation_effects_hash": "mut-c",
            "determinism_closure_hash": "det-c",
        }
        return ("ok", False)

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)
    await plane.submit_turn(session_id="s-1", user_input="same", metadata={"confluence_key": "k1"})
    await plane.submit_turn(session_id="s-2", user_input="same", metadata={"confluence_key": "k1"})

    assert "k1" in plane._global_confluence_contracts
    report = dict(plane._last_confluence_report or {})
    assert report.get("enforced") is True
    assert report.get("action") in {"bound_first_observation", "matched"}


@pytest.mark.asyncio
async def test_global_confluence_law_raises_on_semantic_divergence(monkeypatch) -> None:
    monkeypatch.setenv("DADBOT_GLOBAL_CONFLUENCE_MODE", "enforce")

    async def _executor(session: dict, job) -> tuple[str, bool]:
        state = session.setdefault("state", {})
        variant = str(dict(job.metadata or {}).get("variant") or "a")
        state["last_terminal_state"] = {
            "execution_dag_hash": "dag-c",
            "policy_hash": "pol-c",
            "post_commit_mutation_effects_hash": f"mut-{variant}",
            "determinism_closure_hash": "det-c",
        }
        return (f"ok-{variant}", False)

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)
    await plane.submit_turn(session_id="s-1", user_input="same", metadata={"confluence_key": "k-div", "variant": "a"})

    with pytest.raises(RuntimeError, match="confluence"):
        await plane.submit_turn(
            session_id="s-2",
            user_input="same",
            metadata={"confluence_key": "k-div", "variant": "b"},
        )


@pytest.mark.asyncio
async def test_enforce_mode_kernel_gateway_stamps_confluence_key(monkeypatch) -> None:
    monkeypatch.setenv("DADBOT_GLOBAL_CONFLUENCE_MODE", "enforce")
    monkeypatch.delenv("DADBOT_ALLOW_LEGACY_CONFLUENCE_KEY", raising=False)

    async def _executor(_session: dict, _job) -> tuple[str, bool]:
        return ("ok", False)

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)
    await plane.submit_turn(session_id="s-1", user_input="same", metadata={})
    events = plane.ledger_events()
    submitted = [e for e in events if str(e.get("type") or "") == "JOB_SUBMITTED"]
    assert submitted, "expected at least one JOB_SUBMITTED event"
    payload = dict(submitted[-1].get("payload") or {})
    confluence_key = str(payload.get("metadata", {}).get("confluence_key") or "")
    assert confluence_key.startswith("kgw:")


@pytest.mark.asyncio
async def test_submit_turn_failure_writes_typed_failure_payload() -> None:
    async def _executor(_session: dict, _job) -> tuple[str, bool]:
        raise TimeoutError("executor timeout")

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)

    with pytest.raises(TimeoutError):
        await plane.submit_turn(session_id="s-timeout", user_input="hello")

    events = list(plane.ledger_events())
    failed = [e for e in events if str(e.get("type") or "") == "JOB_FAILED"]
    assert failed, "expected JOB_FAILED ledger event"
    payload = dict(failed[-1].get("payload") or {})
    assert payload.get("error_type") == "TimeoutError"
    failure = dict(payload.get("failure") or {})
    assert failure.get("failure_class") == "timeout"
    assert failure.get("failure_source") == "infrastructure"
    assert bool(failure.get("retryable")) is True
    execution_state = dict(payload.get("metadata", {}).get("execution_state") or {})
    assert execution_state.get("failure_type") == "retryable"
    assert execution_state.get("failure_action") == "manual_retry"
    assert bool(execution_state.get("auto_retry")) is False
    execution_result = dict(payload.get("metadata", {}).get("execution_result") or {})
    assert execution_result.get("status") == "failed"
    failure_view = dict(execution_result.get("failure") or {})
    assert failure_view.get("class") == "timeout"
    timeout_view = dict(execution_result.get("timeout") or {})
    assert bool(timeout_view.get("timed_out")) is True

    policy_events = [e for e in events if str(e.get("type") or "") == "JOB_MANUAL_RETRY_REQUIRED"]
    assert policy_events, "expected explicit failure policy event for manual retry"


@pytest.mark.asyncio
async def test_submit_turn_failure_emits_quarantine_policy_event() -> None:
    async def _executor(_session: dict, _job) -> tuple[str, bool]:
        raise RuntimeError("poison payload from downstream tool")

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)

    with pytest.raises(RuntimeError):
        await plane.submit_turn(session_id="s-poison", user_input="hello")

    events = list(plane.ledger_events())
    quarantined = [e for e in events if str(e.get("type") or "") == "JOB_QUARANTINED"]
    assert quarantined, "expected JOB_QUARANTINED policy event"

    failed = [e for e in events if str(e.get("type") or "") == "JOB_FAILED"]
    assert failed, "expected JOB_FAILED ledger event"
    payload = dict(failed[-1].get("payload") or {})
    execution_state = dict(payload.get("metadata", {}).get("execution_state") or {})
    assert execution_state.get("failure_type") == "poison"
    assert execution_state.get("failure_action") == "quarantine"
    assert bool(execution_state.get("auto_retry")) is False


@pytest.mark.asyncio
async def test_submit_turn_blocks_on_global_invariant_gate() -> None:
    async def _executor(session: dict, _job) -> tuple[str, bool]:
        state = session.setdefault("state", {})
        state["_retrieval_baseline"] = [
            {
                "memory_id": "m-1",
                "summary": "Do not lose this memory",
                "importance_score": 0.9,
            },
        ]
        # Silent shrink without decay marker should hard-fail at commit gate.
        state["memory_retrieval_set"] = []
        return ("ok", False)

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)

    with pytest.raises(Exception, match="Global invariant gate violation"):
        await plane.submit_turn(session_id="s-gate", user_input="hello")

    events = list(plane.ledger_events())
    completed = [e for e in events if str(e.get("type") or "") == "JOB_COMPLETED"]
    failed = [e for e in events if str(e.get("type") or "") == "JOB_FAILED"]
    assert not completed, "commit must be blocked on invariant violation"
    assert failed, "expected JOB_FAILED event when global invariant gate blocks commit"

    payload = dict(failed[-1].get("payload") or {})
    execution_result = dict(payload.get("metadata", {}).get("execution_result") or {})
    assert execution_result.get("status") == "failed"
    failure_view = dict(execution_result.get("failure") or {})
    assert failure_view.get("class") == "contract_violation"

    session = plane.registry.get_or_create("s-gate")
    state = dict(session.get("state") or {})
    terminal_truth = dict(state.get("turn_truth_terminal") or {})
    assert terminal_truth.get("overall_consistent") is False
    assert int(terminal_truth.get("violation_count", 0)) >= 1


@pytest.mark.asyncio
async def test_submit_turn_blocks_on_timeout_status_inconsistency() -> None:
    async def _executor(_session: dict, _job) -> tuple[str, bool]:
        return ("ok", False)

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)

    with pytest.raises(Exception, match="Global invariant gate violation"):
        await plane.submit_turn(
            session_id="s-timeout-consistency",
            user_input="hello",
            metadata={
                "execution_result": {
                    "timeout": {
                        "timed_out": True,
                    },
                },
            },
        )

    events = list(plane.ledger_events())
    completed = [e for e in events if str(e.get("type") or "") == "JOB_COMPLETED"]
    failed = [e for e in events if str(e.get("type") or "") == "JOB_FAILED"]
    assert not completed
    assert failed


@pytest.mark.asyncio
async def test_boot_reconcile_exposes_confluence_metrics(monkeypatch) -> None:
    monkeypatch.setenv("DADBOT_GLOBAL_CONFLUENCE_MODE", "enforce")

    async def _executor(session: dict, _job) -> tuple[str, bool]:
        state = session.setdefault("state", {})
        state["last_terminal_state"] = {
            "execution_dag_hash": "dag-c",
            "policy_hash": "pol-c",
            "post_commit_mutation_effects_hash": "mut-c",
            "determinism_closure_hash": "det-c",
        }
        return ("ok", False)

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)
    await plane.submit_turn(session_id="s-1", user_input="same", metadata={"confluence_key": "k-metrics"})
    await plane.submit_turn(session_id="s-2", user_input="same", metadata={"confluence_key": "k-metrics"})

    status = plane.boot_reconcile()
    metrics = dict(status.get("execution_confluence_metrics") or {})
    assert int(metrics.get("attempted", 0)) >= 2
    assert int(metrics.get("matched", 0)) >= 1


@pytest.mark.asyncio
async def test_boot_reconcile_reports_ambiguous_effects_for_recovery() -> None:
    async def _executor(_session: dict, _job) -> tuple[str, bool]:
        return ("ok", False)

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)
    LedgerWriter(plane.ledger).append_effect_begin(
        session_id="s-amb",
        trace_id="tr-amb",
        effect_id="eff-amb-1",
        request_id="req-amb-1",
    )

    status = plane.boot_reconcile()
    effect_reconciliation = dict(status.get("effect_reconciliation") or {})
    assert bool(effect_reconciliation.get("reconcile_required")) is True
    ambiguous_effects = list(effect_reconciliation.get("ambiguous_effects") or [])
    assert ambiguous_effects, "expected ambiguous effects to be surfaced"
    assert ambiguous_effects[0].get("effect_id") == "eff-amb-1"


@pytest.mark.asyncio
async def test_boot_reconcile_auto_consumes_reconcile_required_effects(monkeypatch) -> None:
    monkeypatch.setenv("DADBOT_AUTO_RECONCILE_ON_BOOT", "1")
    monkeypatch.setenv("DADBOT_RECONCILE_PASS_MAX", "8")
    monkeypatch.setenv("DADBOT_RECONCILE_MODE", "close_only")

    async def _executor(_session: dict, _job) -> tuple[str, bool]:
        return ("ok", False)

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)
    writer = LedgerWriter(plane.ledger)
    writer.append_effect_begin(
        session_id="s-auto",
        trace_id="tr-auto-begin",
        effect_id="eff-auto-1",
        request_id="req-auto-1",
    )
    writer.write_event(
        event_type="JOB_RECONCILE_REQUIRED",
        session_id="s-auto",
        trace_id="tr-auto-required",
        kernel_step_id="control_plane.reconcile.required",
        payload={
            "request_id": "req-auto-1",
            "effect_id": "eff-auto-1",
            "reason": "ambiguous_effect_state",
        },
    )

    status = plane.boot_reconcile()
    effect_reconciliation = dict(status.get("effect_reconciliation") or {})
    consumer = dict(effect_reconciliation.get("consumer") or {})

    assert bool(consumer.get("enabled")) is True
    assert int(consumer.get("attempted", 0)) >= 1
    assert int(consumer.get("applied", 0)) >= 1
    assert bool(effect_reconciliation.get("reconcile_required")) is False
    assert list(effect_reconciliation.get("ambiguous_effects") or []) == []

    events = list(plane.ledger_events())
    reconciled = [event for event in events if str(event.get("type") or "") == "EFFECT_RECONCILED"]
    assert reconciled, "expected EFFECT_RECONCILED event from auto-reconcile consumer"


@pytest.mark.asyncio
async def test_boot_reconcile_auto_consumer_honors_max_items(monkeypatch) -> None:
    monkeypatch.setenv("DADBOT_AUTO_RECONCILE_ON_BOOT", "1")
    monkeypatch.setenv("DADBOT_RECONCILE_PASS_MAX", "1")
    monkeypatch.setenv("DADBOT_RECONCILE_MODE", "close_only")

    async def _executor(_session: dict, _job) -> tuple[str, bool]:
        return ("ok", False)

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)
    writer = LedgerWriter(plane.ledger)

    writer.append_effect_begin(
        session_id="s-bound",
        trace_id="tr-bound-begin-1",
        effect_id="eff-bound-1",
        request_id="req-bound-1",
    )
    writer.write_event(
        event_type="JOB_RECONCILE_REQUIRED",
        session_id="s-bound",
        trace_id="tr-bound-required-1",
        kernel_step_id="control_plane.reconcile.required",
        payload={
            "request_id": "req-bound-1",
            "effect_id": "eff-bound-1",
            "reason": "ambiguous_effect_state",
        },
    )

    writer.append_effect_begin(
        session_id="s-bound",
        trace_id="tr-bound-begin-2",
        effect_id="eff-bound-2",
        request_id="req-bound-2",
    )
    writer.write_event(
        event_type="JOB_RECONCILE_REQUIRED",
        session_id="s-bound",
        trace_id="tr-bound-required-2",
        kernel_step_id="control_plane.reconcile.required",
        payload={
            "request_id": "req-bound-2",
            "effect_id": "eff-bound-2",
            "reason": "ambiguous_effect_state",
        },
    )

    status = plane.boot_reconcile()
    effect_reconciliation = dict(status.get("effect_reconciliation") or {})
    consumer = dict(effect_reconciliation.get("consumer") or {})

    assert bool(consumer.get("enabled")) is True
    assert int(consumer.get("max_items", 0)) == 1
    assert int(consumer.get("attempted", 0)) == 1
    assert int(consumer.get("applied", 0)) == 1
    assert int(consumer.get("remaining", 0)) >= 1
    assert bool(effect_reconciliation.get("reconcile_required")) is True

    ambiguous_effects = list(effect_reconciliation.get("ambiguous_effects") or [])
    assert len(ambiguous_effects) == 1


@pytest.mark.asyncio
async def test_submit_turn_starts_live_without_projection_redelivery() -> None:
    seen_modes: list[str] = []
    seen_states: list[dict[str, object]] = []

    async def _executor(_session: dict, job) -> tuple[str, bool]:
        seen_modes.append(str(dict(job.metadata or {}).get("execution_mode") or ""))
        seen_states.append(dict(dict(job.metadata or {}).get("execution_state") or {}))
        return ("ok", False)

    plane = ExecutionControlPlane(
        registry=SessionRegistry(),
        kernel_executor=_executor,
        worker_id="worker-test",
        redelivery_retry_interval_seconds=0.005,
    )

    result = await plane.submit_turn(session_id="s-redelivery", user_input="hello", timeout_seconds=1.0)

    assert result == ("ok", False)
    assert seen_modes == ["live"]
    assert seen_states
    assert int(seen_states[0].get("redelivery_count", 0)) == 0
    assert int(seen_states[0].get("lease_conflict_count", 0)) == 0
    assert seen_states[0].get("lifecycle_state") == "running"

    events = list(plane.ledger_events())
    redeliveries = [e for e in events if str(e.get("type") or "") == "JOB_REDELIVERY_SCHEDULED"]
    assert not redeliveries, "unexpected redelivery event for initial live execution"
