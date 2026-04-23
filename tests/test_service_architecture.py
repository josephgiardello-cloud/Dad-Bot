from dadbot_system.contracts import ChatRequest, ChatResponse, ServiceConfig, WorkerResult, WorkerTask
from dadbot_system.orchestration import DadBotOrchestrator
from dadbot_system.state import InMemoryStateStore
from dadbot_system.worker import DadBotTaskProcessor
from dadbot.core.dadbot import DadBot


class FakeBroker:
    def __init__(self):
        self.enqueued = []
        self.results = []

    def enqueue(self, task):
        self.enqueued.append(task)

    def get_result_nowait(self):
        if not self.results:
            return None
        return self.results.pop(0)


class FakeBot:
    def __init__(self, model_name="fake-model"):
        self.MODEL_NAME = model_name
        self.ACTIVE_MODEL = model_name
        self.async_calls = 0
        self.sync_calls = 0
        self.loaded_snapshot = None

    def load_session_state_snapshot(self, snapshot):
        self.loaded_snapshot = dict(snapshot or {})

    async def process_user_message_async(self, user_input, attachments=None):
        self.async_calls += 1
        if self.async_calls == 1:
            raise RuntimeError("transient worker failure")
        return f"Dad heard: {user_input}", False

    def process_user_message(self, user_input, attachments=None):
        self.sync_calls += 1
        return f"Dad heard: {user_input}", False

    def snapshot_session_state(self):
        return {
            "history": [
                {"role": "system", "content": "Dad system prompt"},
                {"role": "assistant", "content": "Dad heard: hello there"},
            ],
            "session_moods": [],
            "session_summary": "",
            "session_summary_updated_at": None,
            "session_summary_covered_messages": 0,
            "last_relationship_reflection_turn": 0,
            "pending_daily_checkin_context": False,
            "active_tool_observation_context": None,
            "planner_debug": {},
            "memory_store": {
                "pending_proactive_messages": [{"message": "Check in tonight.", "source": "life-pattern", "created_at": "2026-04-19T18:00:00"}],
                "recent_moods": ["stressed", "positive"],
                "last_mood": "positive",
            },
        }


def planner_debug_factory():
    return {
        "updated_at": None,
        "user_input": "",
        "current_mood": "neutral",
        "planner_status": "idle",
        "planner_reason": "",
        "planner_tool": "",
        "planner_parameters": {},
        "planner_observation": "",
        "fallback_status": "idle",
        "fallback_reason": "",
        "fallback_tool": "",
        "fallback_observation": "",
        "final_path": "idle",
    }


def test_dadbot_session_snapshot_roundtrip_restores_runtime_state(bot):
    bot.history.append({"role": "user", "content": "I need help today."})
    bot.session_moods = ["stressed"]
    bot.session_summary = "Tony needs help today."
    bot.session_summary_updated_at = "2026-04-19T15:00:00"
    bot.session_summary_covered_messages = 2
    bot.last_relationship_reflection_turn = 3
    bot._pending_daily_checkin_context = True
    bot._active_tool_observation_context = "Fresh web lookup result"
    bot._last_planner_debug = {"planner_status": "used-tool", "planner_parameters": {"query": "weather"}}
    bot.MEMORY_STORE["pending_proactive_messages"] = [{"message": "Check in tonight.", "source": "life-pattern", "created_at": "2026-04-19T18:00:00"}]
    bot.MEMORY_STORE["recent_moods"] = ["stressed", "positive"]
    bot.MEMORY_STORE["last_mood"] = "positive"

    snapshot = bot.snapshot_session_state()

    bot.reset_session_state()
    bot.MEMORY_STORE = bot.default_memory_store()
    restored = bot.load_session_state_snapshot(snapshot)

    assert restored["session_summary"] == "Tony needs help today."
    assert restored["session_moods"] == ["stressed"]
    assert restored["pending_daily_checkin_context"] is True
    assert restored["active_tool_observation_context"] == "Fresh web lookup result"
    assert restored["planner_debug"]["planner_status"] == "used-tool"
    assert restored["memory_store"]["pending_proactive_messages"][0]["message"] == "Check in tonight."
    assert restored["memory_store"]["recent_moods"] == ["stressed", "positive"]
    assert bot.pending_proactive_messages()[0]["message"] == "Check in tonight."
    assert bot.MEMORY_STORE["last_mood"] == "positive"
    assert bot.history[-1]["content"] == "I need help today."


def test_dadbot_runtime_state_attr_maps_read_and_write_through(bot):
    bot.history = [{"role": "system", "content": "Dad system prompt"}]
    bot.session_summary = "Mapped through runtime state"
    bot.session_summary_covered_messages = 3
    bot._pending_daily_checkin_context = True
    bot._last_turn_pipeline = {"path": "graph"}

    assert bot.runtime_state_manager.history == [{"role": "system", "content": "Dad system prompt"}]
    assert bot.runtime_state_manager.session_summary == "Mapped through runtime state"
    assert bot.runtime_state_manager.session_summary_covered_messages == 3
    assert bot.runtime_state_manager.pending_daily_checkin_context is True
    assert bot._internal_runtime.last_turn_pipeline == {"path": "graph"}

    bot.runtime_state_manager.active_thread_id = "thread-42"
    bot._internal_runtime.prompt_guard_stats = {"checked": 2}

    assert bot.active_thread_id == "thread-42"
    assert bot._prompt_guard_stats == {"checked": 2}
    assert "history" not in bot.__dict__
    assert "_last_turn_pipeline" not in bot.__dict__


def test_action_methods_are_mixin_owned_not_declared_on_dadbot():
    assert "record_relationship_history_point" not in DadBot.__dict__
    assert "soft_reset_session_context" not in DadBot.__dict__


def test_record_relationship_history_point_uses_memory_manager_surface(bot, monkeypatch):
    captured = {}

    monkeypatch.setattr(bot.memory, "relationship_history", lambda limit=180: [{"source": "older"}])

    def _capture_mutation(**changes):
        captured.update(changes)
        return changes

    monkeypatch.setattr(bot.memory, "mutate_memory_store", _capture_mutation)

    point = bot.record_relationship_history_point(trust_level=71, openness_level=63, source="turn")

    assert point["source"] == "turn"
    assert point["trust_level"] == 71
    assert point["openness_level"] == 63
    assert "relationship_history" in captured
    assert captured["save"] is True
    assert captured["relationship_history"][-1] == point


def test_soft_reset_session_context_syncs_snapshot_via_runtime_state_manager(bot, monkeypatch):
    calls = []
    bot.session_summary = "Keep this summary"

    monkeypatch.setattr(bot.runtime_state_manager, "sync_active_thread_snapshot", lambda: calls.append("synced"))

    result = bot.soft_reset_session_context(preserve_recent_summary=True)

    assert result["mode"] == "soft"
    assert calls == ["synced"]
    assert bot.session_summary == "Keep this summary"


def test_orchestrator_tracks_task_status_response_and_events():
    broker = FakeBroker()
    state_store = InMemoryStateStore()
    orchestrator = DadBotOrchestrator(
        broker,
        state_store=state_store,
        planner_debug_factory=planner_debug_factory,
    )

    request = ChatRequest(session_id="session-1", user_input="hello there")
    task = orchestrator.submit_chat(request)

    assert broker.enqueued[0].task_id == task.task_id
    assert task.execution_graph is not None
    assert task.execution_graph.edges == [("api", "queue"), ("queue", "worker"), ("worker", "state")]
    assert orchestrator.task_status(task.task_id)["status"] == "queued"

    broker.results.append(
        WorkerResult(
            task_id=task.task_id,
            session_id=request.session_id,
            request_id=request.request_id,
            status="completed",
            session_state={
                "history": [{"role": "system", "content": "Dad system prompt"}],
                "planner_debug": planner_debug_factory(),
                "memory_store": {
                    "pending_proactive_messages": [{"message": "Check in tonight.", "source": "life-pattern", "created_at": "2026-04-19T18:00:00"}],
                    "recent_moods": ["stressed"],
                    "last_mood": "stressed",
                },
            },
            response=ChatResponse(
                session_id=request.session_id,
                request_id=request.request_id,
                reply="Love you, buddy.",
                active_model="llama3.2",
            ),
        )
    )

    response = orchestrator.response_for_task(task.task_id)
    task_status = orchestrator.task_status(task.task_id)
    events = orchestrator.session_events(request.session_id)

    assert response["reply"] == "Love you, buddy."
    assert task_status["status"] == "completed"
    assert task_status["session_state"]["history"][0]["role"] == "system"
    assert task_status["session_state"]["memory_store"]["pending_proactive_messages"][0]["message"] == "Check in tonight."
    assert any(event["event_type"] == "response.ready" for event in events)


def test_orchestrator_isolates_session_events_by_tenant():
    broker = FakeBroker()
    state_store = InMemoryStateStore()
    orchestrator = DadBotOrchestrator(
        broker,
        state_store=state_store,
        planner_debug_factory=planner_debug_factory,
    )

    orchestrator.submit_chat(ChatRequest(session_id="shared-session", tenant_id="family-a", user_input="hello"))
    orchestrator.submit_chat(ChatRequest(session_id="shared-session", tenant_id="family-b", user_input="hello"))

    tenant_a_events = orchestrator.session_events("shared-session", tenant_id="family-a")
    tenant_b_events = orchestrator.session_events("shared-session", tenant_id="family-b")

    assert len(tenant_a_events) == 2
    assert len(tenant_b_events) == 2
    assert all(event["tenant_id"] == "family-a" for event in tenant_a_events)
    assert all(event["tenant_id"] == "family-b" for event in tenant_b_events)


def test_worker_processor_retries_and_returns_response_with_fake_bot():
    fake_bot = FakeBot()
    processor = DadBotTaskProcessor(ServiceConfig(), bot_factory=lambda model_name: fake_bot)
    request = ChatRequest(session_id="session-2", user_input="hello there")
    task = WorkerTask(
        session_id=request.session_id,
        request=request,
        session_state={"history": [{"role": "system", "content": "Dad system prompt"}]},
    )

    result = processor.process(task)

    assert result.status == "completed"
    assert result.response is not None
    assert result.response.reply == "Dad heard: hello there"
    assert fake_bot.async_calls == 2
    assert fake_bot.sync_calls == 0
    assert fake_bot.loaded_snapshot["history"][0]["role"] == "system"


def test_worker_processor_falls_back_to_sync_bot_entrypoint():
    class LegacySyncBot:
        def __init__(self):
            self.MODEL_NAME = "legacy-sync"
            self.ACTIVE_MODEL = "legacy-sync"
            self.calls = 0

        def load_session_state_snapshot(self, snapshot):
            return dict(snapshot or {})

        def process_user_message(self, user_input, attachments=None):
            self.calls += 1
            return f"Legacy heard: {user_input}", False

        def snapshot_session_state(self):
            return {"history": [{"role": "system", "content": "Dad system prompt"}]}

    legacy_bot = LegacySyncBot()
    processor = DadBotTaskProcessor(ServiceConfig(), bot_factory=lambda model_name: legacy_bot)
    request = ChatRequest(session_id="legacy-session", user_input="hello there")
    task = WorkerTask(session_id=request.session_id, request=request)

    result = processor.process(task)

    assert result.status == "completed"
    assert result.response is not None
    assert result.response.reply == "Legacy heard: hello there"
    assert legacy_bot.calls == 1


def test_service_config_from_environment_reads_external_state_backends(monkeypatch):
    monkeypatch.setenv("DADBOT_REDIS_URL", "redis://cache.example:6379/0")
    monkeypatch.setenv("DADBOT_POSTGRES_DSN", "postgresql://dad:secret@db.example:5432/dadbot")
    monkeypatch.setenv("DADBOT_API_WORKERS", "5")
    monkeypatch.setenv("DADBOT_OTEL_ENABLED", "true")

    config = ServiceConfig.from_environment()

    assert config.persistence.redis_url == "redis://cache.example:6379/0"
    assert config.persistence.postgres_dsn == "postgresql://dad:secret@db.example:5432/dadbot"
    assert config.workers.worker_count == 5
    assert config.telemetry.otel_enabled is True