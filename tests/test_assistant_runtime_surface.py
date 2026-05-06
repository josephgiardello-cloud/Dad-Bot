from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from dadbot.assistant_runtime import AssistantRuntime


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

        self._task_store = _FakeTaskStore()
        self.runtime_state_container = SimpleNamespace(store=self._task_store)
        self.runtime_orchestration = _FakeOrchestration(self._task_store)

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

    def process_user_message(self, message: str):
        return (f"echo::{message}", True)

    def reset_session_state(self):
        self._reset_called = True


def test_chat_returns_minimal_materialized_result_shape():
    kernel = _FakeKernel()
    assistant = AssistantRuntime(kernel)

    result = assistant.chat("hello")

    assert result["response"] == "echo::hello"
    assert result["memory_updates"] is None
    assert result["tool_calls"] is None
    assert "debug" not in result


def test_chat_debug_gates_kernel_metadata():
    kernel = _FakeKernel()
    assistant = AssistantRuntime(kernel)

    result = assistant.chat("hello", debug=True)

    assert result["response"] == "echo::hello"
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
