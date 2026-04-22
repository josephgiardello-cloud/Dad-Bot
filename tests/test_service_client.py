from dadbot_system.client import DadServiceClient, ServiceChatResult, ServiceClientConfig


class StubServiceClient(DadServiceClient):
    def __init__(self):
        super().__init__()
        self.calls = []

    def ensure_service_running(self, preferred_model: str = "") -> dict:
        self.calls.append(("ensure", preferred_model))
        return {"status": "ok"}

    def _request(self, method: str, path: str, payload=None, timeout: float = 10.0):
        self.calls.append((method, path, payload))
        if path.startswith("/sessions/") and path.endswith("/turn"):
            return {
                "task": {
                    "task_id": "task-turn-1",
                    "request_id": "request-turn-1",
                    "status": "completed",
                    "session_state": {
                        "history": [{"role": "assistant", "content": "Direct turn reply"}],
                    },
                },
                "response": {
                    "request_id": "request-turn-1",
                    "reply": "Direct turn reply",
                    "should_end": False,
                    "active_model": "llama3.2",
                },
            }
        if path.startswith("/sessions/") and path.endswith("/stream"):
            return {
                "task_id": "task-stream-1",
                "request_id": "request-stream-1",
                "status": "queued",
                "stream": {
                    "type": "websocket",
                    "path": "/sessions/session-123/events/stream",
                    "tenant_id": "default",
                },
            }
        if path.startswith("/sessions/") and "/replay?" in path:
            return {
                "session_id": "session-123",
                "tenant_id": "default",
                "event_count": 1,
                "events": [{"event_type": "request.accepted", "event_id": "event-1"}],
            }
        if path.startswith("/sessions/") and path.endswith("/chat"):
            return {"task_id": "task-1", "request_id": "request-1", "status": "queued"}
        if path.startswith("/sessions/") and "/events?tenant_id=" in path:
            return [{"event_type": "request.accepted", "tenant_id": path.rsplit("=", 1)[-1]}]
        if path == "/tasks/task-1":
            return {
                "task": {
                    "task_id": "task-1",
                    "request_id": "request-1",
                    "status": "completed",
                    "session_state": {
                        "history": [{"role": "system", "content": "Dad system prompt"}],
                        "planner_debug": {},
                        "memory_store": {
                            "pending_proactive_messages": [],
                            "recent_moods": [],
                            "last_mood": "neutral",
                        },
                    },
                },
                "response": {
                    "request_id": "request-1",
                    "reply": "Love you, buddy.",
                    "should_end": False,
                    "active_model": "llama3.2",
                },
            }
        raise AssertionError(f"Unexpected request: {method} {path}")


def test_service_client_chat_submits_and_waits_for_result():
    client = StubServiceClient()

    result = client.chat(
        "session-123",
        user_input="hello",
        attachments=[{"type": "audio", "transcript": "hello"}],
        requested_model="llama3.2",
    )

    assert isinstance(result, ServiceChatResult)
    assert result.task_id == "task-1"
    assert result.reply == "Love you, buddy."
    assert result.active_model == "llama3.2"
    assert result.session_state["history"][0]["content"] == "Dad system prompt"
    assert client.calls[0] == ("ensure", "llama3.2")
    assert client.calls[1][2]["tenant_id"] == "default"


def test_service_client_includes_explicit_tenant_id_and_can_poll_tenant_events():
    client = StubServiceClient()

    client.chat("session-123", user_input="hello", tenant_id="family-a")
    events = client.session_events("session-123", tenant_id="family-a")

    assert client.calls[1][2]["tenant_id"] == "family-a"
    assert events[0]["tenant_id"] == "family-a"


def test_service_client_build_local_command_includes_production_persistence_flags():
    client = DadServiceClient(
        ServiceClientConfig(
            base_url="http://127.0.0.1:9123",
            auto_start_local=True,
            worker_count=4,
            redis_url="redis://cache.example:6379/0",
            postgres_dsn="postgresql://dad:secret@db.example:5432/dadbot",
            otel_enabled=True,
            python_executable="python",
            script_path="Dad.py",
        )
    )

    command = client.build_local_service_command(preferred_model="llama3.2")

    assert command[:3] == ["python", "Dad.py", "--serve-api"]
    assert "--worker-count" in command
    assert "4" in command
    assert "--redis-url" in command
    assert "redis://cache.example:6379/0" in command
    assert "--postgres-dsn" in command
    assert "postgresql://dad:secret@db.example:5432/dadbot" in command
    assert "--otel" in command


def test_service_client_builds_websocket_event_stream_url():
    client = DadServiceClient(ServiceClientConfig(base_url="https://dad.example/app", tenant_id="family-a"))

    url = client.session_event_stream_url("session-123")

    assert url == "wss://dad.example/app/sessions/session-123/events/stream?tenant_id=family-a"


def test_service_client_turn_returns_completed_response_payload():
    client = StubServiceClient()

    result = client.turn("session-123", user_input="hello")

    assert isinstance(result, ServiceChatResult)
    assert result.task_id == "task-turn-1"
    assert result.reply == "Direct turn reply"
    assert result.task_payload["status"] == "completed"


def test_service_client_submit_stream_returns_stream_metadata():
    client = StubServiceClient()

    payload = client.submit_stream("session-123", user_input="hello")

    assert payload["task_id"] == "task-stream-1"
    assert payload["stream"]["type"] == "websocket"


def test_service_client_replay_returns_event_log_payload():
    client = StubServiceClient()

    payload = client.replay("session-123", limit=10)

    assert payload["event_count"] == 1
    assert payload["events"][0]["event_type"] == "request.accepted"