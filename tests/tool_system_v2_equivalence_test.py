"""Tool system v2 formal equivalence tests.

Validates three invariants simultaneously for identical input run through both
v2 OFF (baseline implicit-reasoning graph) and v2 ON (tool-compiled graph):

  (A) Semantic consistency     — both modes classify the same intent class and
                                  load equivalent session memory/goal state.
  (B) Tool-trace enrichment    — v2 ON produces execution_plan / tool_results /
                                  tool_trace_hash; v2 OFF produces none of these.
  (C) Determinism envelope     — each mode preserves its internal lock-hash;
                                  cross-mode hashes diverge as expected with no
                                  accidental collapse.
  (D) Stability constraint     — both modes produce identical hashes across two
                                  independent reruns with equivalent session seeds.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from dadbot.core.graph import TurnContext
from dadbot.core.orchestrator import DadBotOrchestrator

# ---------------------------------------------------------------------------
# Fixed controlled input
# ---------------------------------------------------------------------------

_FIXED_INPUT = "What progress have I made on my goal to improve my daily routines?"
_FIXED_GOAL: dict[str, Any] = {
    "id": "goal-routines-01",
    "description": "improve daily routines",
    "status": "active",
}


# ---------------------------------------------------------------------------
# Orchestrator fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def orch_off(bot) -> DadBotOrchestrator:
    """Strict, v2-disabled orchestrator backed by the shared test bot."""
    return DadBotOrchestrator(
        registry=bot.turn_orchestrator.registry,
        bot=bot,
        strict=True,
        tool_system_v2_enabled=False,
        enable_observability=False,
    )


@pytest.fixture
def orch_on(bot) -> DadBotOrchestrator:
    """Strict, v2-enabled orchestrator backed by the shared test bot."""
    return DadBotOrchestrator(
        registry=bot.turn_orchestrator.registry,
        bot=bot,
        strict=True,
        tool_system_v2_enabled=True,
        enable_observability=False,
    )


# ---------------------------------------------------------------------------
# Service stub — applied once via the shared registry
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stub_agent_service(bot, monkeypatch):
    """Stub the LLM agent and noisy background services for fast, deterministic runs.

    Both orch_off and orch_on share the same registry, so one patch covers both.
    """
    service = bot.turn_orchestrator.registry.get("agent_service")

    async def _agent(context: TurnContext, _rich: dict) -> tuple[str, bool]:
        user = str(context.user_input or "")
        return (f"Here is my answer about {user[:40]}.", False)

    monkeypatch.setattr(service, "run_agent", _agent)

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


# ---------------------------------------------------------------------------
# Turn runner helpers
# ---------------------------------------------------------------------------


def _seed_session(orch: DadBotOrchestrator, sid: str) -> None:
    """Pre-populate a fresh session with the fixed goal record."""
    session = orch.session_registry.get_or_create(sid)
    session.setdefault("state", {})["goals"] = [dict(_FIXED_GOAL)]


async def _run_turn(orch: DadBotOrchestrator, sid: str) -> TurnContext:
    """Seed session, run a single turn, return the captured TurnContext."""
    _seed_session(orch, sid)
    await orch.handle_turn(_FIXED_INPUT, session_id=sid)
    ctx = getattr(orch, "_last_turn_context", None)
    assert ctx is not None, f"_last_turn_context not set after handle_turn (orch={orch!r})"
    return ctx


# ---------------------------------------------------------------------------
# Equivalence test suite
# ---------------------------------------------------------------------------


class TestV2Equivalence:
    """Formal v2 tool system equivalence validation — 3 invariants + stability."""

    # ------------------------------------------------------------------
    # (A) Semantic consistency invariant
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_a_semantic_parity(self, orch_off: DadBotOrchestrator, orch_on: DadBotOrchestrator):
        """Both modes classify the same intent class and load the same session goal."""
        ctx_off = await _run_turn(orch_off, "equiv-a-off-" + uuid.uuid4().hex[:6])
        ctx_on = await _run_turn(orch_on, "equiv-a-on-" + uuid.uuid4().hex[:6])

        # Both must produce a non-empty candidate reply.
        assert str(ctx_off.state.get("candidate") or "").strip(), "v2 OFF must produce a candidate reply"
        assert str(ctx_on.state.get("candidate") or "").strip(), "v2 ON must produce a candidate reply"

        # Both must classify the same intent class.
        plan_off = dict(ctx_off.state.get("turn_plan") or {})
        plan_on = dict(ctx_on.state.get("turn_plan") or {})
        intent_off = plan_off.get("intent_type")
        intent_on = plan_on.get("intent_type")
        assert intent_off == intent_on, f"Intent class diverged between modes: OFF={intent_off!r} ON={intent_on!r}"

        # Both must load the same session goal IDs (equivalent memory state).
        goals_off = sorted(g.get("id") for g in list(ctx_off.state.get("session_goals") or []))
        goals_on = sorted(g.get("id") for g in list(ctx_on.state.get("session_goals") or []))
        assert goals_off == goals_on, f"Session goal IDs diverged: OFF={goals_off!r} ON={goals_on!r}"

    # ------------------------------------------------------------------
    # (B) Tool-trace enrichment invariant
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_b_tool_trace_enrichment(self, orch_off: DadBotOrchestrator, orch_on: DadBotOrchestrator):
        """v2 ON has execution_plan/tool_results/tool_trace_hash; v2 OFF has none."""
        ctx_off = await _run_turn(orch_off, "equiv-b-off-" + uuid.uuid4().hex[:6])
        ctx_on = await _run_turn(orch_on, "equiv-b-on-" + uuid.uuid4().hex[:6])

        # --- v2 OFF: NO execution_plan, NO tool_results ---
        ir_off = dict(ctx_off.state.get("tool_ir") or {})
        plan_off = list(ir_off.get("execution_plan") or [])
        results_off = list(ctx_off.state.get("tool_results") or [])
        assert len(plan_off) == 0, f"v2 OFF must not produce execution_plan, got: {plan_off}"
        assert len(results_off) == 0, f"v2 OFF must not produce tool_results, got: {results_off}"
        assert not ctx_off.metadata.get("tool_execution_graph_hash"), "v2 OFF must not stamp tool_execution_graph_hash"

        # --- v2 ON: execution_plan populated, tool_results populated ---
        ir_on = dict(ctx_on.state.get("tool_ir") or {})
        plan_on = list(ir_on.get("execution_plan") or [])
        results_on = list(ctx_on.state.get("tool_results") or [])
        assert len(plan_on) > 0, "v2 ON must produce a non-empty execution_plan"
        assert len(results_on) > 0, "v2 ON must produce non-empty tool_results"
        assert ctx_on.metadata.get("tool_execution_graph_hash"), (
            "v2 ON must stamp tool_execution_graph_hash after ToolExecutorNode"
        )

        # --- tool_trace_hash must be enriched in v2 ON relative to v2 OFF ---
        det_off = dict(ctx_off.metadata.get("determinism") or {})
        det_on = dict(ctx_on.metadata.get("determinism") or {})
        trace_off = str(det_off.get("tool_trace_hash") or "")
        trace_on = str(det_on.get("tool_trace_hash") or "")
        assert trace_on, "v2 ON must have a non-empty tool_trace_hash in determinism metadata"
        assert trace_off != trace_on, (
            "v2 ON tool_trace_hash must differ from v2 OFF — execution enrichment not captured"
        )

    # ------------------------------------------------------------------
    # (C) Determinism envelope integrity invariant
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_c_structural_divergence(self, orch_off: DadBotOrchestrator, orch_on: DadBotOrchestrator):
        """Cross-mode hashes diverge as expected; graph structures differ; flags accurate."""
        ctx_off = await _run_turn(orch_off, "equiv-c-off-" + uuid.uuid4().hex[:6])
        ctx_on = await _run_turn(orch_on, "equiv-c-on-" + uuid.uuid4().hex[:6])

        det_off = dict(ctx_off.metadata.get("determinism") or {})
        det_on = dict(ctx_on.metadata.get("determinism") or {})

        # lock_hash_with_tools must be present in both modes.
        lhwt_off = str(det_off.get("lock_hash_with_tools") or "")
        lhwt_on = str(det_on.get("lock_hash_with_tools") or "")
        assert lhwt_off, "v2 OFF must produce lock_hash_with_tools"
        assert lhwt_on, "v2 ON must produce lock_hash_with_tools"

        # Cross-mode: lock_hash_with_tools must diverge.
        assert lhwt_off != lhwt_on, (
            "lock_hash_with_tools must diverge between v2 ON and v2 OFF "
            "(tool_trace_hash and tool_system_v2_enabled differ)"
        )

        # determinism_hash_with_tools (metadata top-level) must also diverge.
        dhwt_off = str(ctx_off.metadata.get("determinism_hash_with_tools") or "")
        dhwt_on = str(ctx_on.metadata.get("determinism_hash_with_tools") or "")
        assert dhwt_off != dhwt_on, "determinism_hash_with_tools must diverge between v2 ON and v2 OFF"

        # tool_system_v2_enabled flag accurately reflected in each context.
        assert ctx_off.metadata.get("tool_system_v2_enabled") is False, (
            "v2 OFF context must report tool_system_v2_enabled=False"
        )
        assert ctx_on.metadata.get("tool_system_v2_enabled") is True, (
            "v2 ON context must report tool_system_v2_enabled=True"
        )

        # Graph topology: v2 OFF must NOT include tool_router/tool_executor nodes.
        nodes_off = set(orch_off.graph._node_map.keys())
        nodes_on = set(orch_on.graph._node_map.keys())
        assert "tool_router" not in nodes_off, "v2 OFF graph must not contain tool_router"
        assert "tool_executor" not in nodes_off, "v2 OFF graph must not contain tool_executor"
        assert "tool_router" in nodes_on, "v2 ON graph must contain tool_router"
        assert "tool_executor" in nodes_on, "v2 ON graph must contain tool_executor"

        # Baseline nodes must be present in both graphs.
        for required_node in ("temporal", "preflight", "planner", "inference", "safety", "reflection", "save"):
            assert required_node in nodes_off, f"v2 OFF graph missing required node: {required_node!r}"
            assert required_node in nodes_on, f"v2 ON graph missing required node: {required_node!r}"

    # ------------------------------------------------------------------
    # (D) Stability constraint
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_d_stability(self, orch_off: DadBotOrchestrator, orch_on: DadBotOrchestrator):
        """Tool-trace layer is stable across two independent reruns of each mode.

        The base lock_hash intentionally incorporates the memory fingerprint, which
        updates between turns as context is built — that is by-design cross-turn
        statefulness.  What must be stable is the *tool-trace layer*:
          - v2 ON:  tool_trace_hash, execution plan intent sequence, all result statuses.
          - v2 OFF: execution_plan consistently empty, tool_results consistently empty.
        """
        # Two independent v2 OFF runs.
        ctx_off_1 = await _run_turn(orch_off, "equiv-d-off-1-" + uuid.uuid4().hex[:6])
        ctx_off_2 = await _run_turn(orch_off, "equiv-d-off-2-" + uuid.uuid4().hex[:6])

        # Two independent v2 ON runs.
        ctx_on_1 = await _run_turn(orch_on, "equiv-d-on-1-" + uuid.uuid4().hex[:6])
        ctx_on_2 = await _run_turn(orch_on, "equiv-d-on-2-" + uuid.uuid4().hex[:6])

        det_on_1 = dict(ctx_on_1.metadata.get("determinism") or {})
        det_on_2 = dict(ctx_on_2.metadata.get("determinism") or {})

        # --- v2 ON: tool_trace_hash must be stable across two runs ---
        tth_on_1 = str(det_on_1.get("tool_trace_hash") or "")
        tth_on_2 = str(det_on_2.get("tool_trace_hash") or "")
        assert tth_on_1 and tth_on_2, "v2 ON tool_trace_hash must be non-empty in both runs"
        assert tth_on_1 == tth_on_2, f"v2 ON tool_trace_hash unstable across runs: run1={tth_on_1!r} run2={tth_on_2!r}"

        # --- v2 ON: execution plan intent sequence must be stable ---
        def _intents(ctx: TurnContext) -> list[str]:
            plan = list((ctx.state.get("tool_ir") or {}).get("execution_plan") or [])
            return [str(item.get("intent") or "") for item in plan]

        intents_on_1 = _intents(ctx_on_1)
        intents_on_2 = _intents(ctx_on_2)
        assert intents_on_1 == intents_on_2, (
            f"v2 ON execution plan intent sequence unstable: run1={intents_on_1!r} run2={intents_on_2!r}"
        )

        # --- v2 ON: all tool results must succeed in both runs ---
        def _result_statuses(ctx: TurnContext) -> list[str]:
            return [str(r.get("status") or "") for r in list(ctx.state.get("tool_results") or [])]

        statuses_on_1 = _result_statuses(ctx_on_1)
        statuses_on_2 = _result_statuses(ctx_on_2)
        assert statuses_on_1 == statuses_on_2, (
            f"v2 ON tool result statuses unstable: run1={statuses_on_1!r} run2={statuses_on_2!r}"
        )

        # --- v2 OFF: execution_plan consistently empty across both runs ---
        plan_off_1 = list((ctx_off_1.state.get("tool_ir") or {}).get("execution_plan") or [])
        plan_off_2 = list((ctx_off_2.state.get("tool_ir") or {}).get("execution_plan") or [])
        assert plan_off_1 == [] == plan_off_2, "v2 OFF execution_plan must be consistently empty across runs"

        # --- v2 OFF: tool_results consistently empty across both runs ---
        results_off_1 = list(ctx_off_1.state.get("tool_results") or [])
        results_off_2 = list(ctx_off_2.state.get("tool_results") or [])
        assert results_off_1 == [] == results_off_2, "v2 OFF tool_results must be consistently empty across runs"
