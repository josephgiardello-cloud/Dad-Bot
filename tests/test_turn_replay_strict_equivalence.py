from __future__ import annotations

import hashlib

import pytest

from dadbot.core.tool_executor import execute_tool


pytestmark = [
    pytest.mark.phase4,
    pytest.mark.phase4_cert,
    pytest.mark.durability,
    pytest.mark.integration,
]


@pytest.mark.asyncio
async def test_turn_replay_strict_equivalence(tmp_path):
    """Replay turn must be hash-equivalent to live turn and execute zero live side effects."""
    from dadbot.core.dadbot import DadBot
    from dadbot.core.orchestrator import DadBotOrchestrator
    from dadbot.core.persistence import SQLiteCheckpointer

    db_path = str(tmp_path / "strict_replay_equivalence.sqlite3")
    checkpointer = SQLiteCheckpointer(db_path, auto_migrate=True, prune_every=0)

    live_executor_calls = 0

    async def _run_graph_with_tool(*, context, job):
        del job
        nonlocal live_executor_calls

        def _executor() -> dict[str, str]:
            nonlocal live_executor_calls
            live_executor_calls += 1
            return {"status": "ok", "source": "live"}

        record = execute_tool(
            tool_name="strict_equivalence_tool",
            parameters={"query": "strict-replay-equivalence"},
            executor=_executor,
            turn_context=context,
        )
        trace_context = dict(context.metadata.get("execution_trace_context") or {})
        trace_context["final_hash"] = "strict-equivalence-final-hash"
        context.metadata["execution_trace_context"] = trace_context
        return ("tool-path-ok", True)

    live_bot = DadBot()
    live_orchestrator = DadBotOrchestrator(bot=live_bot, strict=True, checkpointer=checkpointer)
    live_orchestrator._run_graph_with_trace_binding = _run_graph_with_tool

    live_session_id = "strict-eq-live"
    live_user_input = "strict replay equivalence"
    await live_orchestrator.handle_turn(
        live_user_input,
        session_id=live_session_id,
        confluence_key="strict-eq-live-turn-1",
    )

    assert live_executor_calls == 1, "Live turn must execute the side-effecting tool exactly once"

    live_context = getattr(live_orchestrator, "_last_turn_context", None)
    assert live_context is not None, "Live turn must materialize _last_turn_context"

    live_session_state = dict(live_orchestrator.session_registry.get_or_create(live_session_id).get("state") or {})
    live_terminal_state = dict(live_session_state.get("last_terminal_state") or {})
    assert str(live_terminal_state.get("final_trace_hash") or ""), "Live turn missing final_trace_hash"
    assert str(live_terminal_state.get("determinism_closure_hash") or ""), "Live turn missing determinism_closure_hash"

    live_ledger = getattr(live_context, "_tool_io_ledger", None)
    assert live_ledger is not None, "Live turn must produce a tool IO ledger"

    replay_seed_checkpoint_hash = hashlib.sha256(b"strict-replay-seed").hexdigest()
    replay_seed = {
        "checkpoint_hash": replay_seed_checkpoint_hash,
        "prev_checkpoint_hash": "",
        "state": {},
        "metadata": dict(getattr(live_context, "metadata", {}) or {}),
        "tool_io_ledger": dict(live_ledger.to_dict()),
    }
    replay_manifest = dict(live_session_state.get("last_determinism_manifest") or {})
    if not replay_manifest:
        replay_manifest = {
            "env_hash": hashlib.sha256(b"strict-replay-env").hexdigest(),
            "manifest_hash": hashlib.sha256(b"strict-replay-manifest").hexdigest(),
        }
    checkpointer.save_checkpoint(
        "strict-eq-replay",
        "trace-strict-eq-replay-seed",
        replay_seed,
        replay_manifest,
    )

    replay_bot = DadBot()
    replay_orchestrator = DadBotOrchestrator(bot=replay_bot, strict=True, checkpointer=checkpointer)
    replay_orchestrator._run_graph_with_trace_binding = _run_graph_with_tool
    replay_session_id = "strict-eq-replay"
    await replay_orchestrator.handle_turn(
        live_user_input,
        session_id=replay_session_id,
        confluence_key="strict-eq-replay-turn-1",
        metadata={"execution_mode": "replay"},
    )

    assert live_executor_calls == 1, "Replay turn must not execute live tool side effects"

    replay_session_state = dict(replay_orchestrator.session_registry.get_or_create(replay_session_id).get("state") or {})
    replay_terminal_state = dict(replay_session_state.get("last_terminal_state") or {})

    assert replay_terminal_state.get("final_trace_hash") == live_terminal_state.get("final_trace_hash")
    assert replay_terminal_state.get("determinism_closure_hash") == live_terminal_state.get("determinism_closure_hash")

    replay_context = getattr(replay_orchestrator, "_last_turn_context", None)
    assert replay_context is not None, "Replay turn must materialize _last_turn_context"
    replay_ledger = getattr(replay_context, "_tool_io_ledger", None)
    replay_records = list(getattr(replay_ledger, "records", []) or [])
    assert replay_records, "Replay turn must record tool evidence"
    assert replay_records[-1].status == "replayed", "Replay tool record must be tagged as replayed"
