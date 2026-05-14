from __future__ import annotations

import pytest

from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry


def _build_plane() -> ExecutionControlPlane:
    plane = ExecutionControlPlane.__new__(ExecutionControlPlane)
    plane.registry = SessionRegistry()
    return plane


def test_canonical_gate_validation_flag_enables_state_metrics() -> None:
    plane = _build_plane()
    metadata = {"canonical_gate_validation": True}

    plane._record_canonical_gate_breach(
        session_key="session-threshold",
        metadata=metadata,
        gate="response_engine_selected_source_missing",
        reason="missing source",
    )

    session = plane.registry.get_or_create("session-threshold")
    metrics = dict(session.get("state", {}).get("canonical_gate_metrics") or {})
    assert int(metrics.get("total_breaches", 0)) == 1
    assert int(dict(metrics.get("per_gate") or {}).get("response_engine_selected_source_missing", 0)) == 1


def test_canonical_gate_validation_flag_off_disables_state_metrics() -> None:
    plane = _build_plane()
    metadata = {"canonical_gate_validation": False}

    plane._record_canonical_gate_breach(
        session_key="session-threshold",
        metadata=metadata,
        gate="response_engine_selected_source_missing",
        reason="missing source",
    )

    session = plane.registry.get_or_create("session-threshold")
    metrics = dict(session.get("state", {}).get("canonical_gate_metrics") or {})
    assert metrics == {}
