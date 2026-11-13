from unittest.mock import AsyncMock

from dadbot.assistant_runtime import AssistantRuntime
from dadbot.runtime.agent_driver_loop import DriverLoopPolicy


def test_agent_loop_dispatches_plans_and_goal_updates_through_canonical_turns(make_test_dadbot, monkeypatch):
    bot = make_test_dadbot(tenant_id="t1_" + "a" * 26)
    monkeypatch.setattr(bot.model_port, "generate", lambda *_args, **_kwargs: "Step completed.")
    monkeypatch.setattr(bot.model_port, "generate_async", AsyncMock(return_value="Step completed."))
    requests = []
    canonical_execute_turn = bot.execute_turn

    def execute_turn(request):
        requests.append(request)
        return canonical_execute_turn(request)

    monkeypatch.setattr(bot, "execute_turn", execute_turn)
    assistant = AssistantRuntime(bot)
    monkeypatch.setattr(assistant, "_executive_startup_observation", lambda **_kwargs: "")
    result = assistant.run_agent_loop(
        "Finish project",
        policy=DriverLoopPolicy(max_turns=2),
        session_id="project",
        enable_semantic_memory=False,
        reflection_hook=lambda context: {
            "action_input": f"Project step {context['turn_index']}",
            "goal_updates": {"status": f"step {context['turn_index']}"},
        },
    )

    assert [request.input.text for request in requests] == ["Project step 1", "Project step 2"]
    assert [request.metadata["agent_loop"]["goal_blackboard"] for request in requests] == [
        {}, {"status": "step 1"},
    ]
    assert all(request.context.session_id == "project" for request in requests)
    assert all(request.context.tenant_id == bot.config.tenant_id for request in requests)
    assert result.goal_blackboard == {"status": "step 2"}
    assert result.completed_turns == 2
    assert result.stop_reason == "max_turns"
    control_plane = bot.services.turn_orchestrator.control_plane
    assert len(control_plane._ledger_index.completed_agent_loop_entries(
        session_id="project", tenant_id=bot.config.tenant_id,
    )) == 2


def test_background_job_runs_through_runtime_orchestration(bot):
    called = []

    def job():
        called.append(True)
        return "completed"

    future = bot.runtime_orchestration.submit_background_task(job, task_kind="test.background")

    assert future.result(timeout=5) == "completed"
    assert called == [True]
    saved = bot.runtime_state_container.store.load_task(future.dadbot_task_id)
    assert saved["status"] == "completed"
