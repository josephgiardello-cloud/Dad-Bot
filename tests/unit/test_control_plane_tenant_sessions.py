from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap

import pytest

from dadbot.core.control_plane import ControlPlaneOptions, ExecutionControlPlane, SessionRegistry
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.kernel_gateway import KernelGateway
from dadbot.core.ledger_backend import FileWALLedgerBackend
from dadbot.core.ledger_reader import LedgerReader
from dadbot.core.ledger_writer import LedgerWriter
from dadbot.core.session_store import SessionStore
from dadbot.core.tenant_session import tenant_session_scope


def test_registry_isolates_identical_session_ids_by_tenant() -> None:
    registry = SessionRegistry()
    tenant_a = registry.get_or_create("shared", tenant_id="tenant-a")
    tenant_b = registry.get_or_create("shared", tenant_id="tenant-b")
    tenant_a["state"]["value"] = "a"
    tenant_b["state"]["value"] = "b"

    assert registry.get("shared", tenant_id="tenant-a")["state"]["value"] == "a"
    assert registry.get("shared", tenant_id="tenant-b")["state"]["value"] == "b"
    registry.terminate_session("shared", tenant_id="tenant-a")
    assert registry.is_terminated("shared", tenant_id="tenant-a")
    assert not registry.is_terminated("shared", tenant_id="tenant-b")


def test_session_replay_preserves_tenant_and_quarantines_unowned_events() -> None:
    ledger = ExecutionLedger()
    writer = LedgerWriter(ledger)
    with KernelGateway.open_scope():
        for tenant_id, value in (("tenant-a", "a"), ("tenant-b", "b")):
            writer.write_event(
                "SESSION_STATE_UPDATED",
                tenant_id=tenant_id,
                session_id="shared",
                trace_id=f"trace-{tenant_id}",
                kernel_step_id="test.session_state",
                payload={"tenant_id": tenant_id, "state": {"value": value}},
                committed=True,
            )
        writer.write_event(
            "SESSION_STATE_UPDATED",
            session_id="shared",
            trace_id="trace-unowned",
            kernel_step_id="test.session_state",
            payload={"state": {"value": "legacy"}},
            committed=True,
        )

    store = SessionStore(ledger=ledger, projection_only=True)
    store.rebuild_from_ledger(ledger.read())
    assert store.get("shared", tenant_id="tenant-a") == {"value": "a"}
    assert store.get("shared", tenant_id="tenant-b") == {"value": "b"}
    assert store.get("shared", tenant_id="") is None

    async def executor(_session, _job):
        return "ok", False

    plane = ExecutionControlPlane(
        registry=SessionRegistry(),
        kernel_executor=executor,
        options=ControlPlaneOptions(ledger=ledger),
    )
    assert plane.registry.get("shared", tenant_id="tenant-a")["state"]["value"] == "a"
    assert plane.registry.get("shared", tenant_id="tenant-b")["state"]["value"] == "b"
    assert plane.registry.get("shared") is None


def test_ledger_replay_rejects_mismatched_event_payload_owner() -> None:
    store = SessionStore(projection_only=True)
    with pytest.raises(ValueError, match="event tenant does not match its payload"):
        store.rebuild_from_ledger(
            [
                {
                    "type": "SESSION_STATE_UPDATED",
                    "tenant_id": "tenant-a",
                    "session_id": "shared",
                    "payload": {"tenant_id": "tenant-b", "state": {"value": "bad"}},
                },
            ],
        )


def test_control_plane_session_mutations_reject_mismatched_active_owner() -> None:
    async def executor(_session, _job):
        return "ok", False

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=executor)
    with tenant_session_scope("tenant-a", "shared"):
        with pytest.raises(ValueError, match="tenant identity"):
            plane._merge_session_state(
                session_id="shared",
                state_patch={"value": "cross-tenant"},
                trace_token="trace-cross-tenant",
                tenant_id="tenant-b",
            )
    assert plane.registry.get("shared", tenant_id="tenant-a") is None
    assert plane.registry.get("shared", tenant_id="tenant-b") is None


@pytest.mark.asyncio
async def test_concurrent_same_session_submissions_remain_tenant_owned() -> None:
    async def executor(session, job):
        session.setdefault("state", {})["owner"] = job.tenant_id
        return job.tenant_id, False

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=executor)
    result_a, result_b = await asyncio.gather(
        plane.submit_turn(
            session_id="shared",
            user_input="same request",
            metadata={"tenant_id": "tenant-a", "request_id": "same-request"},
        ),
        plane.submit_turn(
            session_id="shared",
            user_input="same request",
            metadata={"tenant_id": "tenant-b", "request_id": "same-request"},
        ),
    )

    assert result_a[0] == "tenant-a"
    assert result_b[0] == "tenant-b"
    assert plane.registry.get("shared", tenant_id="tenant-a")["state"]["owner"] == "tenant-a"
    assert plane.registry.get("shared", tenant_id="tenant-b")["state"]["owner"] == "tenant-b"


def test_control_plane_event_and_timeline_reads_are_tenant_scoped() -> None:
    ledger = ExecutionLedger()
    writer = LedgerWriter(ledger)
    with KernelGateway.open_scope():
        for tenant_id in ("tenant-a", "tenant-b"):
            writer.write_event(
                "SESSION_STATE_UPDATED",
                tenant_id=tenant_id,
                session_id="shared",
                trace_id=f"trace-{tenant_id}",
                kernel_step_id="test.session_state",
                payload={"tenant_id": tenant_id, "state": {"execution_timeline": [{"event_type": tenant_id}]}},
                committed=True,
            )

    plane = ExecutionControlPlane(
        registry=SessionRegistry(default_tenant_id="tenant-a"),
        kernel_executor=lambda *_args: None,
        options=ControlPlaneOptions(ledger=ledger),
    )
    with tenant_session_scope("tenant-a", "shared"):
        assert {event["tenant_id"] for event in plane.ledger_events()} == {"tenant-a"}
        assert {event["tenant_id"] for event in plane.ledger_reader.events()} == {"tenant-a"}
        assert plane.execution_timeline(session_id="shared")[0]["event_type"] == "tenant-a"
        with pytest.raises(ValueError, match="tenant identity"):
            plane.ledger_events(tenant_id="tenant-b")
        with pytest.raises(ValueError, match="tenant identity"):
            plane.ledger_reader.events_for_job("tenant-b-job", tenant_id="tenant-b")
    assert {event["tenant_id"] for event in plane.ledger_events(tenant_id="tenant-b")} == {"tenant-b"}
    assert {event["tenant_id"] for event in LedgerReader(ledger).events(tenant_id="tenant-b")} == {"tenant-b"}
    assert plane.execution_timeline(session_id="shared", tenant_id="tenant-b")[0]["event_type"] == "tenant-b"


def test_control_plane_bootstrap_recovers_same_session_per_tenant_in_fresh_process(tmp_path) -> None:
    ledger_path = tmp_path / "tenant-sessions.wal"
    ledger = ExecutionLedger(FileWALLedgerBackend(ledger_path))
    writer = LedgerWriter(ledger)
    with KernelGateway.open_scope():
        for tenant_id, value in (("tenant-a", "a"), ("tenant-b", "b")):
            writer.write_event(
                "SESSION_STATE_UPDATED",
                tenant_id=tenant_id,
                session_id="shared",
                trace_id=f"seed-{tenant_id}",
                kernel_step_id="test.session_state",
                payload={"tenant_id": tenant_id, "state": {"value": value}},
                committed=True,
            )

    script = textwrap.dedent(
        """
        import sys
        from dadbot.core.control_plane import ControlPlaneOptions, ExecutionControlPlane, SessionRegistry
        from dadbot.core.execution_ledger import ExecutionLedger
        from dadbot.core.ledger_backend import FileWALLedgerBackend

        async def executor(_session, _job):
            return ("ok", False)

        ledger = ExecutionLedger(FileWALLedgerBackend(sys.argv[1]))
        ledger.load_from_backend()
        plane = ExecutionControlPlane(
            registry=SessionRegistry(), kernel_executor=executor,
            options=ControlPlaneOptions(ledger=ledger),
        )
        assert plane.registry.get("shared", tenant_id="tenant-a")["state"]["value"] == "a"
        assert plane.registry.get("shared", tenant_id="tenant-b")["state"]["value"] == "b"
        assert plane.registry.get("shared", tenant_id="default") is None
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(ledger_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
