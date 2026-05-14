from __future__ import annotations

import pytest

from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry

pytestmark = [pytest.mark.dev]


def _build_plane() -> ExecutionControlPlane:
    plane = ExecutionControlPlane.__new__(ExecutionControlPlane)
    plane.registry = SessionRegistry()
    return plane


def test_canonical_gate_metrics_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DADBOT_CANONICAL_GATE_VALIDATION", raising=False)
    plane = _build_plane()
    metadata: dict[str, object] = {}

    plane._record_canonical_gate_breach(
        session_key="s-metrics",
        metadata=metadata,
        gate="response_engine_empty_finalize",
        reason="empty response",
    )

    session = plane.registry.get_or_create("s-metrics")
    metrics = dict(session.get("state", {}).get("canonical_gate_metrics") or {})
    assert metrics == {}
    assert int(metadata.get("canonical_gate_breach_count", 0) or 0) == 1


def test_canonical_gate_metrics_enabled_for_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DADBOT_CANONICAL_GATE_VALIDATION", "1")
    plane = _build_plane()
    metadata: dict[str, object] = {}

    plane._record_canonical_gate_breach(
        session_key="s-metrics",
        metadata=metadata,
        gate="response_engine_empty_finalize",
        reason="empty response",
    )

    session = plane.registry.get_or_create("s-metrics")
    metrics = dict(session.get("state", {}).get("canonical_gate_metrics") or {})
    assert int(metrics.get("total_breaches", 0)) == 1
    assert int(dict(metrics.get("per_gate") or {}).get("response_engine_empty_finalize", 0)) == 1
