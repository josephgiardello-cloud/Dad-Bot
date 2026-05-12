from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from dadbot.assistant_runtime import AssistantRuntime

pytestmark = pytest.mark.unit


class _FakeTaskStore:
    def __init__(self) -> None:
        self._tasks = {}

    def load_task(self, task_id: str):
        return self._tasks.get(task_id)


class _FakeOrchestration:
    def __init__(self, store: _FakeTaskStore) -> None:
        self._store = store

    def submit_background_task(self, _func, *_args, **_kwargs):
        task_id = "task-123"
        self._store._tasks[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "error": "",
            "updated_at": "2026-01-01T00:00:00",
        }
        return SimpleNamespace(dadbot_task_id=task_id)


class _FakeKernel:
    def __init__(self) -> None:
        self._last_turn_pipeline = {"phase": "done"}
        self._last_planner_debug = {"tool_calls": ["memory_lookup"]}
        self._last_memory_context_stats = {"tokens": 123}
        self._last_reply_supervisor = {"approved": True}
        self.last_terminal_state = {"determinism_closure_hash": "abc"}
        self.MEMORY_STORE = {"last_continuous_learning_at": None, "learning_cycle_count": 0, "last_learning_turn": 0}

        self._task_store = _FakeTaskStore()
        self.runtime_state_container = SimpleNamespace(store=self._task_store)
        self.runtime_orchestration = _FakeOrchestration(self._task_store)
        self.maintenance_scheduler = SimpleNamespace(run_proactive_heartbeat=lambda force=True: {"forced": force, "queued_total": 1})
        self.long_term_signals = SimpleNamespace(
            schedule_continuous_learning=lambda: SimpleNamespace(dadbot_task_id="learn-123"),
            perform_continuous_learning_cycle=lambda: {"cycle": 1, "hypotheses_updated": True},
        )
        self.local_mcp_status = lambda: {"running": False, "tool_count": 18}
        self.start_local_mcp_server_process = lambda restart=False: {"running": True, "restart": restart}
        self.stop_local_mcp_server_process = lambda: {"running": False}
        self.submit_background_task = lambda _func, **_kwargs: SimpleNamespace(dadbot_task_id="learn-bg-1")

        bg_task = SimpleNamespace(
            status="completed",
            error="",
            started_at=datetime(2026, 1, 1, 0, 0, 0),
            completed_at=datetime(2026, 1, 1, 0, 0, 1),
        )
        self.background_tasks = SimpleNamespace(get_task=lambda _tid: bg_task)
        self.memory_query = SimpleNamespace(
            relevant_memories_for_input=lambda q, limit=5: [{"summary": q, "limit": limit}],
        )
        self.memory_catalog = lambda: [{"summary": "Allergic to peanuts", "category": "health", "mood": "calm"}]
        self.semantic_memory_matches = lambda q, memories, limit=3: list(memories)[:limit]
        self.synced_semantic_index = []
        self.sync_semantic_memory_index = lambda memories: self.synced_semantic_index.extend(list(memories or []))

        self.ledger_writer = SimpleNamespace(
            write_event=lambda event_type, **kwargs: {"event_id": "evt-runtime", "type": event_type, **kwargs}
        )
        self.execution_ledger = SimpleNamespace(
            read=lambda: [
                {
                    "event_id": "e1",
                    "session_id": "default",
                    "type": "JOB_COMPLETED",
                    "payload": {"summary": "Discussed tomato garden plan"},
                    "timestamp": "2026-05-12T00:00:00Z",
                }
            ]
            * 12
        )

    def process_user_message(self, message: str):
        return (f"echo::{message}", True)

    def execute_turn(self, request):
        return SimpleNamespace(as_result=lambda: (f"reply::{request.input.text}", True))

    def reset_session_state(self):
        self._reset_called = True


def test_chat_returns_minimal_materialized_result_shape():
    kernel = _FakeKernel()
    assistant = AssistantRuntime(kernel)

    result = assistant.chat("hello")

    assert result["response"] == "reply::hello"
    assert result["memory_updates"] is None
    assert result["tool_calls"] is None
    assert "debug" not in result


def test_chat_debug_gates_kernel_metadata():
    kernel = _FakeKernel()
    assistant = AssistantRuntime(kernel)

    result = assistant.chat("hello", debug=True)

    assert result["response"] == "reply::hello"
    assert result["debug"]["trace"] == {"phase": "done"}
    assert result["debug"]["planner"] == {"tool_calls": ["memory_lookup"]}


def test_run_task_and_get_state_use_background_task_surface():
    kernel = _FakeKernel()
    assistant = AssistantRuntime(kernel)

    task_id = assistant.run_task("do a thing")
    state = assistant.get_state(task_id)

    assert task_id == "task-123"
    assert state["task_id"] == "task-123"
    assert state["status"] == "queued"


def test_reset_session_delegates_to_kernel():
    kernel = _FakeKernel()
    assistant = AssistantRuntime(kernel)

    assistant.reset_session()

    assert getattr(kernel, "_reset_called", False) is True


def test_memory_delegates_to_memory_query_when_matcher_missing():
    kernel = _FakeKernel()
    assistant = AssistantRuntime(kernel)

    rows = assistant.memory("budget", limit=3)

    assert rows == [{"summary": "budget", "limit": 3}]


def test_run_heartbeat_delegates_to_maintenance_scheduler():
    kernel = _FakeKernel()
    assistant = AssistantRuntime(kernel)

    payload = assistant.run_heartbeat(force=False)

    assert payload == {"forced": False, "queued_total": 1}


def test_run_self_improvement_supports_background_and_sync_modes():
    kernel = _FakeKernel()
    assistant = AssistantRuntime(kernel)

    queued = assistant.run_self_improvement(force=False, background=True)
    completed = assistant.run_self_improvement(force=True, background=False)

    assert queued == {"status": "queued", "task_id": "learn-123"}
    assert completed == {"status": "completed", "result": {"cycle": 1, "hypotheses_updated": True}}


def test_browser_tool_controls_delegate_to_kernel_mcp_surface():
    kernel = _FakeKernel()
    assistant = AssistantRuntime(kernel)

    status = assistant.browser_status()
    started = assistant.start_browser_tools(restart=True)
    stopped = assistant.stop_browser_tools()

    assert status == {"running": False, "tool_count": 18}
    assert started == {"running": True, "restart": True}
    assert stopped == {"running": False}


def test_run_agent_loop_wires_semantic_memory_observation_hook():
    kernel = _FakeKernel()
    assistant = AssistantRuntime(kernel)

    result = assistant.run_agent_loop("hello", enable_semantic_memory=True)

    assert result.completed_turns >= 1


def test_run_memory_consolidation_executes_job():
    kernel = _FakeKernel()
    assistant = AssistantRuntime(kernel)

    payload = assistant.run_memory_consolidation(background=False, force=True)

    assert payload["status"] == "written"


def test_executive_startup_observation_formats_pending_tasks(monkeypatch):
    monkeypatch.setattr(
        "dadbot_system.local_mcp_server.get_pending_executive_tasks",
        lambda limit=8: [
            {"id": "tsk1", "title": "Send invoice", "due": "2026-05-13T09:00:00", "priority": "high", "done": False},
            {"id": "tsk2", "title": "Review PR", "due": "", "priority": "normal", "done": False},
        ][:limit],
    )

    text = AssistantRuntime._executive_startup_observation(max_tasks=8)

    assert "Startup executive tasks (pending):" in text
    assert "[tsk1] Send invoice" in text
    assert "[tsk2] Review PR" in text


def test_executive_startup_observation_returns_empty_when_no_tasks(monkeypatch):
    monkeypatch.setattr(
        "dadbot_system.local_mcp_server.get_pending_executive_tasks",
        lambda limit=8: [],
    )

    assert AssistantRuntime._executive_startup_observation(max_tasks=8) == ""
