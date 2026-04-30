from __future__ import annotations

import json
from pathlib import Path

import pytest

from dadbot.core.system_identity import compute_system_snapshot_v0_hash

SNAPSHOT_DIR = Path(__file__).resolve().parents[1] / "snapshots" / "system_snapshot_v0"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_system_snapshot_v0_hash_matches_identity_lock():
    metadata = _read_json(SNAPSHOT_DIR / "metadata.json")
    expected = compute_system_snapshot_v0_hash(tool_system_v2_enabled=False)
    assert metadata["snapshot_hash"] == expected


@pytest.mark.asyncio
async def test_system_snapshot_v0_gold_traces_replay(bot, monkeypatch):
    runtime_behavior = _read_json(SNAPSHOT_DIR / "runtime_behavior_snapshot.json")
    traces = list(runtime_behavior.get("gold_traces") or [])
    assert len(traces) >= 20

    orchestrator = bot.turn_orchestrator
    service = orchestrator.registry.get("agent_service")

    async def _deterministic_agent(context, _rich):
        return (f"SNAPSHOT_V0::{str(context.user_input or '').strip()}", False)

    monkeypatch.setattr(service, "run_agent", _deterministic_agent)

    for item in traces:
        prompt = str(item.get("prompt") or "")
        expected_output = str(item.get("expected_output") or "")
        result = await orchestrator.handle_turn(prompt, session_id="system-snapshot-v0-regression")
        assert str(result[0] or "") == expected_output

        context = getattr(orchestrator, "_last_turn_context", None)
        assert context is not None
        assert list(context.state.get("tool_ir", {}).get("execution_plan") or []) == []
        assert list(context.state.get("tool_results") or []) == []


def test_determinism_hash_includes_tool_trace(bot):
    orchestrator = bot.turn_orchestrator
    result = orchestrator.run("quick determinism probe")
    assert isinstance(result, tuple)

    context = getattr(orchestrator, "_last_turn_context", None)
    assert context is not None
    determinism = dict(context.metadata.get("determinism") or {})
    assert str(determinism.get("tool_trace_hash") or "")
    assert str(determinism.get("lock_hash_with_tools") or "")
    assert str(context.metadata.get("determinism_hash_with_tools") or "")
