"""Phase 4.1 guarantee tests.

Covers missing edge guarantees around determinism, delegation safety,
mutation boundaries, execution-loop integrity, persistence idempotence,
blackboard consistency, and tool containment.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from dadbot.core.graph import MutationGuard, MutationIntent, MutationKind, TurnContext
from dadbot.core.nodes import (
    InferenceNode,
    SaveNode,
    _MAX_DELEGATION_SUBTASKS,
    _dispatch_builtin_tool,
)
from dadbot.core.orchestrator import DadBotOrchestrator
from dadbot.core.graph import TurnTemporalAxis


@pytest.fixture
def orchestrator(bot) -> DadBotOrchestrator:
    return bot.turn_orchestrator


@pytest.fixture(autouse=True)
def _fast_stubs(orchestrator: DadBotOrchestrator, monkeypatch):
    service = orchestrator.registry.get("agent_service")

    async def _agent(context: TurnContext, _rich: dict[str, Any]) -> tuple[str, bool]:
        return (f"ok::{str(context.user_input or '')}", False)

    monkeypatch.setattr(service, "run_agent", _agent)

    bot = orchestrator.bot
    mc = getattr(bot, "memory_coordinator", None)
    if mc is not None:
        monkeypatch.setattr(mc, "consolidate_memories", lambda **kw: None)
        monkeypatch.setattr(mc, "apply_controlled_forgetting", lambda **kw: None)
    rm = getattr(bot, "relationship_manager", None)
    if rm is not None:
        monkeypatch.setattr(rm, "materialize_projection", lambda **kw: None)
    mm = getattr(bot, "memory_manager", None)
    gm = getattr(mm, "graph_manager", None) if mm is not None else None
    if gm is not None:
        monkeypatch.setattr(gm, "sync_graph_store", lambda **kw: None)
    if hasattr(bot, "validate_reply"):
        monkeypatch.setattr(bot, "validate_reply", lambda _u, r: r)
    if hasattr(bot, "current_runtime_health_snapshot"):
        monkeypatch.setattr(bot, "current_runtime_health_snapshot", lambda **kw: {})


async def _run(orchestrator: DadBotOrchestrator, text: str, sid: str) -> tuple[tuple[str | None, bool], TurnContext]:
    result = await orchestrator.handle_turn(text, session_id=sid)
    ctx = getattr(orchestrator, "_last_turn_context", None)
    assert isinstance(ctx, TurnContext)
    return result, ctx


def test_determinism_noop_metadata_variance(orchestrator: DadBotOrchestrator, monkeypatch):
    base = orchestrator._build_turn_context
    stamp = {"value": 123}

    def _wrapped(user_input: str, attachments=None):
        c = base(user_input, attachments)
        c.metadata["noop_timestamp"] = stamp["value"]
        return c

    monkeypatch.setattr(orchestrator, "_build_turn_context", _wrapped)

    r1, _ = asyncio.run(_run(orchestrator, "hello", "g-noop"))
    stamp["value"] = 999
    r2, _ = asyncio.run(_run(orchestrator, "hello", "g-noop"))

    assert str(r1[0]) == str(r2[0])


def test_determinism_delegation_mode_equivalent_output(orchestrator: DadBotOrchestrator, monkeypatch):
    service = orchestrator.registry.get("agent_service")

    async def _agent(context: TurnContext, _rich: dict[str, Any]) -> tuple[str, bool]:
        if not context.metadata.get("parent_trace_id"):
            mode = "parallel" if "parallel" in str(context.user_input).lower() else "sequential"
            return (
                json.dumps(
                    {
                        "type": "delegate",
                        "mode": mode,
                        "subtasks": [
                            {"agent": "a", "input": "one"},
                            {"agent": "b", "input": "two"},
                            {"agent": "c", "input": "three"},
                        ],
                    }
                ),
                False,
            )
        return (f"res::{context.metadata.get('agent_name','')}::{context.user_input}", False)

    monkeypatch.setattr(service, "run_agent", _agent)

    _, c_seq = asyncio.run(_run(orchestrator, "run in sequential mode", "g-mode-seq"))
    _, c_par = asyncio.run(_run(orchestrator, "run in parallel mode", "g-mode-par"))

    out_seq = sorted(list(c_seq.state.get("delegation_results") or []))
    out_par = sorted(list(c_par.state.get("delegation_results") or []))
    assert out_seq == out_par


def test_deterministic_arbitration_resolution_stable(orchestrator: DadBotOrchestrator, monkeypatch):
    service = orchestrator.registry.get("agent_service")
    base = orchestrator._build_turn_context

    def _wrapped(user_input: str, attachments=None):
        c = base(user_input, attachments)
        c.trace_id = "fixed-trace"
        c.mutation_queue._owner_trace_id = "fixed-trace"
        return c

    monkeypatch.setattr(orchestrator, "_build_turn_context", _wrapped)

    async def _agent(context: TurnContext, _rich: dict[str, Any]) -> tuple[str, bool]:
        if not context.metadata.get("parent_trace_id"):
            return (
                json.dumps(
                    {
                        "type": "delegate",
                        "mode": "sequential",
                        "subtasks": [{"agent": "a", "input": "x"}, {"agent": "b", "input": "y"}],
                    }
                ),
                False,
            )
        return (f"done::{context.metadata.get('agent_name','')}", False)

    monkeypatch.setattr(service, "run_agent", _agent)

    _, c1 = asyncio.run(_run(orchestrator, "stable arbitration", "g-arb"))
    _, c2 = asyncio.run(_run(orchestrator, "stable arbitration", "g-arb"))

    a1 = dict(c1.state.get("arbitration_metadata") or {})
    a2 = dict(c2.state.get("arbitration_metadata") or {})
    assert a1.get("mode") == a2.get("mode") == "sequential"
    assert int(a1.get("agents_dispatched") or 0) == int(a2.get("agents_dispatched") or 0) == 2
    assert list(c1.state.get("delegation_results") or []) == list(c2.state.get("delegation_results") or [])


def test_depth_guard_propagates_in_nested_delegation(orchestrator: DadBotOrchestrator, monkeypatch):
    service = orchestrator.registry.get("agent_service")

    async def _recursive(_context: TurnContext, _rich: dict[str, Any]) -> tuple[str, bool]:
        return (json.dumps({"type": "delegate", "subtasks": [{"input": "deeper"}]}), False)

    monkeypatch.setattr(service, "run_agent", _recursive)

    _, ctx = asyncio.run(_run(orchestrator, "deep recursion", "g-depth"))
    assert bool(ctx.metadata.get("delegation_depth_exceeded")) is True
    assert bool(ctx.state.get("delegation_depth_exceeded")) is True
    arb_log = list(ctx.state.get("delegation_arbitration_log") or [])
    assert any(str(e.get("event") or "") == "depth_guard_block" for e in arb_log)


def test_trimmed_subtasks_are_not_executed(orchestrator: DadBotOrchestrator, monkeypatch):
    service = orchestrator.registry.get("agent_service")
    seen_agents: list[str] = []

    async def _agent(context: TurnContext, _rich: dict[str, Any]) -> tuple[str, bool]:
        if not context.metadata.get("parent_trace_id"):
            return (
                json.dumps(
                    {
                        "type": "delegate",
                        "mode": "sequential",
                        "subtasks": [{"agent": f"agent_{i}", "input": f"task-{i}"} for i in range(20)],
                    }
                ),
                False,
            )
        seen_agents.append(str(context.metadata.get("agent_name") or ""))
        return ("ok", False)

    monkeypatch.setattr(service, "run_agent", _agent)
    _, _ctx = asyncio.run(_run(orchestrator, "trim check", "g-trim"))

    disallowed = {f"agent_{i}" for i in range(_MAX_DELEGATION_SUBTASKS, 20)}
    assert disallowed.isdisjoint(set(seen_agents))


def test_delegation_state_isolation_to_allowed_keys(orchestrator: DadBotOrchestrator, monkeypatch):
    service = orchestrator.registry.get("agent_service")

    async def _agent(context: TurnContext, _rich: dict[str, Any]) -> tuple[str, bool]:
        if not context.metadata.get("parent_trace_id"):
            return (
                json.dumps(
                    {
                        "type": "delegate",
                        "mode": "sequential",
                        "subtasks": [{"agent": "a", "input": "x"}],
                    }
                ),
                False,
            )
        return ("result", False)

    monkeypatch.setattr(service, "run_agent", _agent)

    _, ctx = asyncio.run(_run(orchestrator, "state isolation", "g-iso"))
    delegation_keys = {k for k in ctx.state.keys() if "delegation" in k or "arbitration" in k or "blackboard" in k}
    allowed = {
        "delegation_results",
        "delegation_arbitration_log",
        "arbitration_metadata",
        "agent_blackboard",
    }
    assert delegation_keys.issubset(allowed)


def test_all_operations_emit_trace(orchestrator: DadBotOrchestrator, monkeypatch):
    service = orchestrator.registry.get("agent_service")

    def _fake_call_llm(messages, **kwargs):
        _ = messages
        _ = kwargs
        return "trace-model-output"

    monkeypatch.setattr(orchestrator.bot.runtime_client, "call_llm", _fake_call_llm)

    async def _agent(context: TurnContext, _rich: dict[str, Any]) -> tuple[str, bool]:
        output = orchestrator.bot.model_port.generate(
            [{"role": "user", "content": str(context.user_input or "")}],
            purpose="trace_test",
        )
        orchestrator.bot.mutate_memory_store(health_quiet_mode=False, save=False)
        return (str(output), False)

    monkeypatch.setattr(service, "run_agent", _agent)

    _, ctx = asyncio.run(_run(orchestrator, "trace all operations", "g-trace-ops"))
    trace = dict(ctx.metadata.get("execution_trace_context") or {})
    operations = set(str(item) for item in list(trace.get("operations") or []))

    assert "model_call" in operations
    assert "memory_read" in operations
    assert "memory_write" in operations


def _minimal_temporal() -> dict[str, Any]:
    return {
        "wall_time": "2026-01-01T00:00:00+00:00",
        "wall_date": "2026-01-01",
        "timezone": "UTC",
        "utc_offset_minutes": 0,
        "epoch_seconds": 1.0,
    }


def test_mutation_guard_blocks_valid_shape_wrong_phase():
    queue = TurnContext(user_input="x").mutation_queue
    intent = MutationIntent(
        type=MutationKind.LEDGER,
        payload={"op": "append_history", "temporal": _minimal_temporal()},
    )
    with MutationGuard(queue):
        with pytest.raises(RuntimeError, match="MutationGuard violation"):
            queue.queue(intent)


def test_partial_mutation_rollback_restores_state():
    @dataclass
    class _Mgr:
        snap: dict[str, Any] | None = None

        def begin_transaction(self, context: TurnContext):
            self.snap = dict(context.state)

        def apply_mutations(self, context: TurnContext):
            context.state["mutated"] = True

        def finalize_turn(self, _context: TurnContext, _result: Any):
            raise RuntimeError("forced failure")

        def commit_transaction(self, _context: TurnContext):
            return None

        def rollback_transaction(self, context: TurnContext):
            context.state.clear()
            context.state.update(dict(self.snap or {}))

    ctx = TurnContext(user_input="rollback")
    ctx.state["before"] = 1
    ctx.temporal = TurnTemporalAxis.from_now()
    node = SaveNode(_Mgr())

    with pytest.raises(RuntimeError):
        asyncio.run(node.run(ctx))

    assert ctx.state.get("before") == 1
    assert "mutated" not in ctx.state


class _FakeCritique:
    def __init__(self):
        self.calls = 0

    def critique(self, reply_text: str, _user: str, _plan: dict[str, Any], iteration: int):
        self.calls += 1
        if iteration == 0:
            return SimpleNamespace(score=0.2, passed=False, issues=["low_quality"], revision_hint="revise")
        return SimpleNamespace(score=0.9, passed=True, issues=[], revision_hint="")


def test_execution_loop_emits_final_revision_only():
    class _Mgr:
        async def run_agent(self, context: TurnContext, _rich: dict[str, Any]):
            if context.state.get("_critique_revision_context"):
                return ("final revised answer", False)
            return ("initial candidate", False)

    node = InferenceNode(_Mgr(), critique_engine=_FakeCritique(), max_loop_iterations=2)
    ctx = TurnContext(user_input="q")
    ctx.temporal = SimpleNamespace(**_minimal_temporal())
    ctx.state["rich_context"] = {}
    asyncio.run(node.run(ctx))

    candidate = ctx.state.get("candidate")
    text = candidate[0] if isinstance(candidate, tuple) else str(candidate)
    assert text == "final revised answer"
    assert text != "initial candidate"


def test_critique_metadata_matches_iterations_run():
    class _Mgr:
        async def run_agent(self, context: TurnContext, _rich: dict[str, Any]):
            if context.state.get("_critique_revision_context"):
                return ("pass", False)
            return ("fail", False)

    critique = _FakeCritique()
    node = InferenceNode(_Mgr(), critique_engine=critique, max_loop_iterations=2)
    ctx = TurnContext(user_input="q")
    ctx.temporal = SimpleNamespace(**_minimal_temporal())
    ctx.state["rich_context"] = {}
    asyncio.run(node.run(ctx))

    record = dict(ctx.state.get("critique_record") or {})
    assert critique.calls == int(record.get("iteration") or 0) + 1


def test_turn_replay_is_idempotent_for_session_state(orchestrator: DadBotOrchestrator):
    sid = "g-idempotent"
    session = {"session_id": sid, "state": {}}
    job = SimpleNamespace(
        user_input="idempotent reply",
        attachments=None,
        session_id=sid,
        metadata={"trace_id": "fixed-trace-idempotent"},
        job_id="jid-1",
    )

    r1 = asyncio.run(orchestrator._execute_job(session, job))
    state_after_first = dict(session.get("state") or {})
    r2 = asyncio.run(orchestrator._execute_job(session, job))
    state_after_second = dict(session.get("state") or {})

    assert r1 == r2
    assert state_after_first.get("last_result") == state_after_second.get("last_result")
    assert state_after_first.get("goals", []) == state_after_second.get("goals", [])


def test_tool_side_effect_containment():
    ctx = TurnContext(user_input="tool")
    before_keys = set(ctx.state.keys())
    out = _dispatch_builtin_tool("echo", {"message": "hello"}, ctx)
    after_keys = set(ctx.state.keys())
    assert out == "hello"
    assert before_keys == after_keys


def test_tool_determinism_same_input_same_output():
    ctx = TurnContext(user_input="tool")
    args = {"message": "stable"}
    assert _dispatch_builtin_tool("echo", args, ctx) == _dispatch_builtin_tool("echo", args, ctx)
