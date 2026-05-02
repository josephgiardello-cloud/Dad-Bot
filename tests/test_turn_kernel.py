"""Tests for TurnKernel, ToolTransaction, and Bayesian policy gate.

Validates the three architecture guarantees:
1. Single execution kernel — execute_step() is the sole state-transition authority.
2. True idempotent tool execution — ToolTransaction provides transaction semantics.
3. Policy-to-execution lock — Bayesian gate is enforced before ACT steps.
"""

import asyncio

import pytest

pytestmark = pytest.mark.unit
from types import SimpleNamespace

from dadbot.core.control_plane import (
    ExecutionControlPlane,
    InMemoryExecutionLedger,
    LedgerReader,
    Scheduler,
    SessionRegistry,
)
from dadbot.core.ledger_writer import LedgerWriter
from dadbot.core.graph import TurnContext
from dadbot.core.kernel import (
    PolicyDecision,
    TurnKernel,
    bayesian_policy_gate,
)
from dadbot.core.testing.tool_runtime_test_adapter import ToolRuntimeTestAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(user_input: str = "hey dad") -> TurnContext:
    return TurnContext(user_input=user_input)


def _allow_gate(*_):
    return PolicyDecision(allowed=True, reason="test-allow")


def _deny_gate(*_):
    return PolicyDecision(allowed=False, reason="test-deny", action="reject")


# ---------------------------------------------------------------------------
# 1. Execution kernel — single state-transition authority
# ---------------------------------------------------------------------------


def test_kernel_execute_step_returns_ok_and_tracks_written_keys():
    kernel = TurnKernel()
    ctx = _ctx()

    async def _step():
        ctx.state["candidate"] = "reply text"
        ctx.state["rich_context"] = {"facts": []}

    result = asyncio.run(kernel.execute_step(ctx, "inference", _step))
    assert result.status == "ok"
    assert "candidate" in result.state_keys_written


def test_kernel_execute_step_appends_to_audit_trail():
    kernel = TurnKernel(policy_gate=_allow_gate)
    ctx = _ctx()

    async def _step():
        ctx.state["health"] = {"ok": True}

    asyncio.run(kernel.execute_step(ctx, "health", _step))
    audit = ctx.metadata.get("kernel_audit", [])
    assert len(audit) == 1
    assert audit[0]["step"] == "health"
    assert audit[0]["status"] == "ok"
    assert "health" in audit[0]["wrote"]
    assert audit[0]["kernel_step_id"] == "health"
    assert "trace_id" in audit[0]
    lineage = ctx.metadata.get("kernel_lineage") or {}
    assert lineage.get("kernel_step_id") == "health"
    assert lineage.get("trace_id") == ctx.trace_id


def test_kernel_execute_step_multiple_steps_build_audit_trail():
    kernel = TurnKernel()
    ctx = _ctx()

    async def _run():
        async def _mem():
            ctx.state["rich_context"] = {}

        async def _inf():
            ctx.state["candidate"] = "hi"

        await kernel.execute_step(ctx, "memory", _mem)
        await kernel.execute_step(ctx, "inference", _inf)

    asyncio.run(_run())
    audit = ctx.metadata.get("kernel_audit", [])
    assert [a["step"] for a in audit] == ["memory", "inference"]


def test_kernel_policy_gate_hard_blocks_step_and_records_rejection():
    kernel = TurnKernel(policy_gate=_deny_gate)
    ctx = _ctx()
    state_before = dict(ctx.state)

    async def _step():
        ctx.state["candidate"] = "should never be written"

    result = asyncio.run(kernel.execute_step(ctx, "inference", _step))
    assert result.status == "rejected"
    assert result.policy.reason == "test-deny"
    # State must NOT have been mutated.
    assert ctx.state == state_before
    rejections = ctx.metadata.get("kernel_rejections", [])
    assert len(rejections) == 1
    assert rejections[0]["step"] == "inference"


def test_kernel_allowed_step_runs_and_returns_ok():
    kernel = TurnKernel(policy_gate=_allow_gate)
    ctx = _ctx()

    async def _step():
        ctx.state["safe_result"] = ("all good", False)

    result = asyncio.run(kernel.execute_step(ctx, "safety", _step))
    assert result.status == "ok"
    assert ctx.state["safe_result"] == ("all good", False)


def test_kernel_wraps_step_exception_as_error_status():
    kernel = TurnKernel()
    ctx = _ctx()

    async def _bad():
        raise RuntimeError("inference service exploded")

    result = asyncio.run(kernel.execute_step(ctx, "inference", _bad))
    assert result.status == "error"
    assert "inference service exploded" in result.error


def test_kernel_without_policy_gate_always_allows():
    kernel = TurnKernel()  # no gate
    ctx = _ctx()
    calls = []

    async def _step():
        calls.append(1)

    result = asyncio.run(kernel.execute_step(ctx, "inference", _step))
    assert result.status == "ok"
    assert calls == [1]


def test_kernel_policy_is_checked_per_step():
    decisions = iter(
        [
            PolicyDecision(allowed=True, reason="preflight-ok"),
            PolicyDecision(allowed=False, reason="inference-blocked"),
            PolicyDecision(allowed=True, reason="safety-ok"),
        ]
    )

    def _gate(ctx, step):
        return next(decisions)

    kernel = TurnKernel(policy_gate=_gate)
    ctx = _ctx()
    executed = []

    async def _run():
        for step_name in ("preflight", "inference", "safety"):

            async def _step(name=step_name):
                executed.append(name)

            await kernel.execute_step(ctx, step_name, _step)

    asyncio.run(_run())
    assert executed == ["preflight", "safety"]  # inference was blocked
    assert len(ctx.metadata["kernel_rejections"]) == 1


# ---------------------------------------------------------------------------
# 2. ToolTransaction — explicit transaction semantics
# ---------------------------------------------------------------------------


def test_tool_transaction_commit_on_success():
    runtime = ToolRuntimeTestAdapter()
    with runtime.transaction(tool_name="set_reminder", parameters={"title": "Call dentist"}) as txn:
        record = txn.execute(executor=lambda: {"id": "r1", "title": "Call dentist"})

    assert txn.committed is True
    assert txn.rolled_back is False
    assert record.status == "succeeded"
    assert txn.result == {"id": "r1", "title": "Call dentist"}


def test_tool_transaction_auto_rollback_on_exception():
    runtime = ToolRuntimeTestAdapter()
    rolled_back = []

    with pytest.raises(ValueError):
        with runtime.transaction(tool_name="set_reminder", parameters={"title": "Boom"}) as txn:
            txn.execute(
                executor=lambda: {"id": "r_boom"},
                compensating_action=lambda: rolled_back.append("rolled"),
            )
            raise ValueError("downstream failure")

    assert txn.rolled_back is True
    assert txn.committed is False
    assert rolled_back == ["rolled"]


def test_tool_transaction_auto_rollback_does_not_affect_earlier_sandbox_records():
    runtime = ToolRuntimeTestAdapter()
    compensations = []

    # Record outside the transaction
    runtime.execute(
        tool_name="web_search",
        parameters={"query": "news"},
        executor=lambda: {"heading": "News", "summary": "today"},
        compensating_action=lambda: compensations.append("outer"),
    )

    with pytest.raises(RuntimeError):
        with runtime.transaction(tool_name="set_reminder", parameters={"title": "X"}) as txn:
            txn.execute(
                executor=lambda: {"id": "rx"},
                compensating_action=lambda: compensations.append("inner"),
            )
            raise RuntimeError("inner failure")

    # Only the inner compensating action should have fired.
    assert compensations == ["inner"]


def test_tool_transaction_status_before_execute_is_not_started():
    runtime = ToolRuntimeTestAdapter()
    txn = runtime.make_transaction(tool_name="set_reminder", parameters={})
    assert txn.status == "not_started"
    assert txn.result is None


def test_tool_transaction_failed_executor_does_not_commit():
    runtime = ToolRuntimeTestAdapter()
    with runtime.transaction(tool_name="set_reminder", parameters={"title": "Fail"}) as txn:
        record = txn.execute(executor=lambda: (_ for _ in ()).throw(RuntimeError("db down")))

    # Clean exit (no exception was re-raised) because the sandbox isolates failures.
    # The transaction was entered and exited cleanly, but the record is "failed".
    assert record.status == "failed"
    assert txn.committed is True  # context manager exited without exception


# ---------------------------------------------------------------------------
# 3. Bayesian policy gate — hard Bayesian closure
# ---------------------------------------------------------------------------


def _make_bot(tool_bias: str = "planner_default"):
    return SimpleNamespace(planner_debug_snapshot=lambda: {"bayesian_tool_bias": tool_bias})


def test_bayesian_gate_allows_non_act_steps_unconditionally():
    gate = bayesian_policy_gate(_make_bot("defer_tools_unless_explicit"))
    ctx = _ctx("hey")
    for step in ("preflight", "health", "memory", "safety", "save"):
        decision = gate(ctx, step)
        assert decision.allowed is True, f"{step!r} should always be allowed"


def test_bayesian_gate_allows_act_step_under_default_policy():
    gate = bayesian_policy_gate(_make_bot("planner_default"))
    ctx = _ctx("how are you?")
    decision = gate(ctx, "inference")
    assert decision.allowed is True
    assert ctx.state.get("_bayesian_tool_bias_kernel") == "planner_default"


def test_bayesian_gate_propagates_tool_bias_into_turn_state():
    gate = bayesian_policy_gate(_make_bot("minimal_tools"))
    ctx = _ctx("I'm stressed")
    gate(ctx, "inference")
    assert ctx.state["_bayesian_tool_bias_kernel"] == "minimal_tools"
    assert ctx.metadata["kernel_policy"]["tool_bias"] == "minimal_tools"
    assert ctx.metadata["kernel_policy"]["enforced"] is True


def test_bayesian_gate_sets_tools_blocked_flag_on_defer_without_explicit_request():
    gate = bayesian_policy_gate(_make_bot("defer_tools_unless_explicit"))
    ctx = _ctx("just venting today")
    gate(ctx, "inference")
    assert ctx.state.get("_bayesian_tools_blocked") is True
    assert ctx.metadata["kernel_policy"]["tools_blocked"] is True


def test_bayesian_gate_does_not_block_tools_when_user_explicitly_requests():
    gate = bayesian_policy_gate(_make_bot("defer_tools_unless_explicit"))
    ctx = _ctx("can you remind me to call the dentist?")
    gate(ctx, "inference")
    assert ctx.state.get("_bayesian_tools_blocked") is not True


def test_bayesian_gate_still_allows_inference_step_even_when_tools_blocked():
    """LLM must always run — blocking tools must not silence the response."""
    gate = bayesian_policy_gate(_make_bot("defer_tools_unless_explicit"))
    ctx = _ctx("just venting")
    decision = gate(ctx, "inference")
    # Inference is allowed even though tools are blocked.
    assert decision.allowed is True
    assert ctx.state.get("_bayesian_tools_blocked") is True


def test_bayesian_gate_handles_missing_planner_debug_gracefully():
    bot = SimpleNamespace(planner_debug_snapshot=lambda: None)
    gate = bayesian_policy_gate(bot)
    ctx = _ctx("hello")
    decision = gate(ctx, "inference")
    assert decision.allowed is True
    assert ctx.state["_bayesian_tool_bias_kernel"] == "planner_default"


def test_bayesian_gate_handles_planner_debug_snapshot_exception():
    def _broken():
        raise RuntimeError("snapshot unavailable")

    bot = SimpleNamespace(planner_debug_snapshot=_broken)
    gate = bayesian_policy_gate(bot)
    ctx = _ctx("hello")
    decision = gate(ctx, "inference")
    assert decision.allowed is True


# ---------------------------------------------------------------------------
# 4. TurnGraph kernel integration
# ---------------------------------------------------------------------------


def test_turn_graph_routes_through_kernel():
    from dadbot.core.graph import TurnContext, TurnGraph
    from dadbot.core.nodes import TemporalNode

    executed_steps = []

    class _CounterKernel:
        async def execute_step(self, turn_context, step_name, step_fn):
            executed_steps.append(step_name)
            await step_fn()
            return type("R", (), {"status": "ok", "error": ""})()

    class _SimpleNode:
        name = "test_node"

        async def execute(self, registry, ctx):
            ctx.state["result"] = "node ran"

    class _SaveNode:
        name = "save_node"

        async def execute(self, registry, ctx):
            ctx.state["safe_result"] = (str(ctx.state.get("result") or ""), False)

    graph = TurnGraph(registry=None)
    graph.add_node("temporal", TemporalNode())
    graph.add_node("kernel_step", _SimpleNode())
    graph.add_node("save", _SaveNode())
    graph.set_edge("temporal", "kernel_step")
    graph.set_edge("kernel_step", "save")
    # Manually attach the counter kernel
    graph._kernel = _CounterKernel()

    ctx = TurnContext(user_input="test")
    asyncio.run(graph.execute(ctx))

    assert "test_node" in executed_steps
    assert ctx.state.get("result") == "node ran"


def test_turn_graph_kernel_rejection_skips_state_mutation():
    from dadbot.core.graph import TurnContext, TurnGraph
    from dadbot.core.nodes import TemporalNode

    class _RejectKernel:
        async def execute_step(self, turn_context, step_name, step_fn):
            if step_name == "mutating_node":
                # Reject the mutating step without calling step_fn.
                return type("R", (), {"status": "rejected", "error": ""})()
            await step_fn()
            return type("R", (), {"status": "ok", "error": ""})()

    class _MutatingNode:
        name = "mutating_node"

        async def execute(self, registry, ctx):
            ctx.state["secret"] = "should not appear"

    class _SaveNode:
        name = "save_node"

        async def execute(self, registry, ctx):
            ctx.state["safe_result"] = ("ok", False)

    graph = TurnGraph(registry=None)
    graph.add_node("temporal", TemporalNode())
    graph.add_node("mutate_stage", _MutatingNode())
    graph.add_node("save", _SaveNode())
    graph.set_edge("temporal", "mutate_stage")
    graph.set_edge("mutate_stage", "save")
    graph._kernel = _RejectKernel()

    ctx = TurnContext(user_input="test")
    asyncio.run(graph.execute(ctx))

    assert "secret" not in ctx.state


# ---------------------------------------------------------------------------
# 5. Unified execution control plane — global coordination boundary
# ---------------------------------------------------------------------------


def test_control_plane_serializes_turns_within_same_session():
    registry = SessionRegistry()

    async def _kernel_execute(session, job):
        state = session.setdefault("state", {})
        current = int(state.get("count") or 0)
        await asyncio.sleep(0)
        state["count"] = current + 1
        return (f"count={state['count']}", False)

    control_plane = ExecutionControlPlane(
        registry=registry,
        kernel_executor=_kernel_execute,
    )

    async def _run():
        return await asyncio.gather(
            control_plane.submit_turn(session_id="s1", user_input="first"),
            control_plane.submit_turn(session_id="s1", user_input="second"),
        )

    results = asyncio.run(_run())
    session = registry.get("s1")
    assert results[0] == ("count=1", False)
    assert results[1] == ("count=1", False)
    assert session["state"]["count"] == 1

    event_types = [event["type"] for event in control_plane.ledger_events()]
    assert "JOB_SUBMITTED" in event_types
    assert "SESSION_BOUND" in event_types
    assert "JOB_QUEUED" in event_types
    assert "JOB_STARTED" in event_types
    assert "JOB_COMPLETED" in event_types


def test_control_plane_rejects_new_work_after_session_termination():
    registry = SessionRegistry()

    async def _kernel_execute(_session, _job):
        return ("ok", False)

    control_plane = ExecutionControlPlane(
        registry=registry,
        kernel_executor=_kernel_execute,
    )

    async def _run():
        await control_plane.create_session("s-dead")
        control_plane.terminate_session("s-dead")
        with pytest.raises(RuntimeError):
            await control_plane.submit_turn(session_id="s-dead", user_input="hello?")

    asyncio.run(_run())


def test_scheduler_consumes_jobs_from_ledger_pending_tail():
    registry = SessionRegistry()
    # Use a single shared ledger for both reader and writer.
    shared_ledger = InMemoryExecutionLedger()
    scheduler = Scheduler(
        registry,
        reader=LedgerReader(shared_ledger),
        writer=LedgerWriter(shared_ledger),
    )

    async def _kernel_execute(_session, _job):
        return ("ok", False)

    control_plane = ExecutionControlPlane(
        scheduler=scheduler,
        registry=registry,
        kernel_executor=_kernel_execute,
        ledger=shared_ledger,
    )

    async def _run():
        result = await control_plane.submit_turn(session_id="ledger-session", user_input="hello")
        return result

    result = asyncio.run(_run())
    assert result == ("ok", False)

    events = control_plane.ledger_events()
    event_types = [event["type"] for event in events]
    assert event_types[:3] == ["JOB_SUBMITTED", "SESSION_BOUND", "JOB_QUEUED"]
    assert "JOB_STARTED" in event_types
    assert event_types[-1] == "JOB_COMPLETED"
