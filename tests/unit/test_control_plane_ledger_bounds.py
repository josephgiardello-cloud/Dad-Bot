from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry


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
