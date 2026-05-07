import socket
import threading
import time

import pytest

pytestmark = pytest.mark.integration

uvicorn = pytest.importorskip("uvicorn")
pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")

from dadbot_system import (
    DadBotOrchestrator,
    DadServiceClient,
    InMemoryEventBus,
    InMemoryStateStore,
    ServiceClientConfig,
    ServiceConfig,
    create_api_app,
)
from dadbot_system.contracts import ChatResponse, WorkerResult
from dadbot_system.security import ServiceTokenManager


SERVICE_TOKEN_SECRET = "integration-service-token-secret"


def service_config(*, read_limit: int = 120, write_limit: int = 60, admin_limit: int = 20) -> ServiceConfig:
    return ServiceConfig.from_dict(
        {
            "security": {
                "auth_required": True,
                "token_secret": SERVICE_TOKEN_SECRET,
                "token_issuer": "dadbot",
                "max_requests_per_minute": read_limit,
                "max_write_requests_per_minute": write_limit,
                "max_admin_requests_per_minute": admin_limit,
            }
        }
    )


def auth_headers(*, tenant_id: str = "family-a", scopes: tuple[str, ...] = ("read", "write")) -> dict[str, str]:
    token = ServiceTokenManager(SERVICE_TOKEN_SECRET, issuer="dadbot").issue(
        subject="integration-client",
        tenant_id=tenant_id,
        scopes=scopes,
        allowed_tools=[],
        ttl_seconds=3600.0,
    )
    return {"Authorization": f"Bearer {token}"}


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


class ImmediateResultBroker:
    def __init__(self):
        self.results = []

    def enqueue(self, task):
        reply = f"Dad heard: {task.request.user_input}"
        self.results.append(
            WorkerResult(
                task_id=task.task_id,
                session_id=task.session_id,
                request_id=task.request.request_id,
                status="completed",
                session_state={
                    "history": [
                        {"role": "system", "content": "Dad system prompt"},
                        {"role": "assistant", "content": reply},
                    ],
                    "planner_debug": planner_debug_factory(),
                    "memory_store": {
                        "pending_proactive_messages": [],
                        "recent_moods": [],
                        "last_mood": "neutral",
                    },
                },
                response=ChatResponse(
                    session_id=task.session_id,
                    request_id=task.request.request_id,
                    reply=reply,
                    active_model=task.request.requested_model or "llama3.2",
                ),
            )
        )

    def get_result_nowait(self):
        if not self.results:
            return None
        return self.results.pop(0)


class FakeRuntimeBot:
    def __init__(self) -> None:
        self.MEMORY_STORE = {
            "last_continuous_learning_at": "2026-05-07T09:00:00Z",
            "learning_cycle_count": 3,
            "last_learning_turn": 42,
        }
        self._proactive_heartbeat_interval_seconds = 900
        self.maintenance_scheduler = type(
            "Maintenance",
            (),
            {"run_proactive_heartbeat": lambda _self, force=True: {"queued_total": 2, "forced": force}},
        )()
        self.long_term_signals = type(
            "LongTerm",
            (),
            {
                "schedule_continuous_learning": lambda _self: type("Task", (), {"dadbot_task_id": "learn-queued"})(),
                "perform_continuous_learning_cycle": lambda _self: {"cycle": 4, "hypotheses_updated": True},
            },
        )()
        self._mcp_running = False

    def maintenance_snapshot(self):
        return {"last_scheduled_proactive_at": "2026-05-07T08:45:00Z"}

    def local_mcp_status(self):
        return {"running": self._mcp_running, "tool_count": 18, "server_name": "dadbot-local-services"}

    def start_local_mcp_server_process(self, *, restart: bool = False):
        self._mcp_running = True
        return {"running": True, "restart": restart, "tool_count": 18}

    def stop_local_mcp_server_process(self):
        self._mcp_running = False
        return {"running": False, "tool_count": 18}

    def submit_background_task(self, _func, **_kwargs):
        return type("Task", (), {"dadbot_task_id": "learn-bg"})()

    def shutdown(self):
        return None


def reserve_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_live_server(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if getattr(server, "started", False):
            return server, thread
        time.sleep(0.05)

    server.should_exit = True
    thread.join(timeout=2)
    raise RuntimeError("Live API test server did not start in time")


def test_live_api_chat_roundtrip_over_http():
    broker = ImmediateResultBroker()
    orchestrator = DadBotOrchestrator(
        broker,
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    app = create_api_app(orchestrator, config=service_config())
    port = reserve_free_port()
    server, thread = start_live_server(app, port)

    try:
        client = DadServiceClient(
            ServiceClientConfig(
                base_url=f"http://127.0.0.1:{port}",
                token_secret=SERVICE_TOKEN_SECRET,
                auto_start_local=False,
                task_timeout_seconds=5.0,
                poll_interval_seconds=0.05,
            )
        )

        health = client.health()
        result = client.chat(
            "integration-session",
            user_input="hello from integration",
            requested_model="llama3.2",
            tenant_id="family-a",
        )

        assert health["status"] == "ok"
        assert result.reply == "Dad heard: hello from integration"
        assert result.active_model == "llama3.2"
        assert result.task_payload["status"] == "completed"
        assert result.task_payload["tenant_id"] == "family-a"
        assert result.session_state["history"][1]["content"] == "Dad heard: hello from integration"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_api_event_stream_websocket_replays_and_streams_tenant_events():
    from fastapi.testclient import TestClient

    broker = ImmediateResultBroker()
    event_bus = InMemoryEventBus()
    orchestrator = DadBotOrchestrator(
        broker,
        state_store=InMemoryStateStore(),
        event_bus=event_bus,
        planner_debug_factory=planner_debug_factory,
    )
    app = create_api_app(orchestrator, config=service_config())

    with TestClient(app) as client:
        websocket_token = ServiceTokenManager(SERVICE_TOKEN_SECRET, issuer="dadbot").issue(
            subject="integration-client",
            tenant_id="family-a",
            scopes={"read", "write"},
            ttl_seconds=3600.0,
        )
        with client.websocket_connect(f"/v1/sessions/socket-session/events/stream?access_token={websocket_token}") as websocket:
            response = client.post(
                "/v1/sessions/socket-session/chat",
                json={"user_input": "hello from websocket", "tenant_id": "family-a"},
                headers=auth_headers(),
            )

            accepted = websocket.receive_json()
            dispatched = websocket.receive_json()

    assert response.status_code == 200
    assert accepted["event_type"] == "request.accepted"
    assert accepted["tenant_id"] == "family-a"
    assert dispatched["event_type"] == "request.dispatched"
    assert dispatched["tenant_id"] == "family-a"


def test_api_turn_endpoint_blocks_until_response():
    from fastapi.testclient import TestClient

    broker = ImmediateResultBroker()
    orchestrator = DadBotOrchestrator(
        broker,
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    app = create_api_app(orchestrator, config=service_config())

    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/turn-session/turn",
            json={"user_input": "hello from turn endpoint", "tenant_id": "family-a", "timeout_seconds": 3},
            headers=auth_headers(),
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["task"]["status"] == "completed"
    assert payload["response"]["reply"] == "Dad heard: hello from turn endpoint"
    assert payload["task"]["tenant_id"] == "family-a"


def test_api_replay_endpoint_returns_session_event_log():
    from fastapi.testclient import TestClient

    broker = ImmediateResultBroker()
    orchestrator = DadBotOrchestrator(
        broker,
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    app = create_api_app(orchestrator, config=service_config())

    with TestClient(app) as client:
        post_response = client.post(
            "/v1/sessions/replay-session/chat",
            json={"user_input": "hello replay", "tenant_id": "family-a"},
            headers=auth_headers(),
        )
        assert post_response.status_code == 200

        replay_response = client.get(
            "/v1/sessions/replay-session/replay?tenant_id=family-a&event_type=request.accepted&limit=10",
            headers=auth_headers(),
        )

    payload = replay_response.json()
    assert replay_response.status_code == 200
    assert payload["session_id"] == "replay-session"
    assert payload["tenant_id"] == "family-a"
    assert payload["event_count"] >= 1
    assert all(event["event_type"] == "request.accepted" for event in payload["events"])


def test_api_channel_gateway_ingest_normalizes_channel_metadata():
    from fastapi.testclient import TestClient

    broker = ImmediateResultBroker()
    orchestrator = DadBotOrchestrator(
        broker,
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    app = create_api_app(orchestrator, config=service_config())

    with TestClient(app) as client:
        response = client.post(
            "/v1/channels/Slack Connect/sessions/gateway-session/ingest",
            json={
                "message": "hello from slack",
                "tenant_id": "family-a",
                "sender_id": "u-42",
                "sender_name": "Tony",
                "external_message_id": "msg-99",
            },
            headers=auth_headers(),
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["gateway"]["channel"] == "slack-connect"
    assert payload["gateway"]["user_id"] == "u-42"
    assert payload["gateway"]["message_id"] == "msg-99"
    assert payload["tenant_id"] == "family-a"


def test_api_automation_endpoints_expose_heartbeat_learning_and_browser_controls():
    from fastapi.testclient import TestClient

    broker = ImmediateResultBroker()
    orchestrator = DadBotOrchestrator(
        broker,
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    app = create_api_app(orchestrator, config=service_config(), runtime_bot=FakeRuntimeBot())

    with TestClient(app) as client:
        admin_headers = auth_headers(scopes=("read", "write", "admin"))
        status_response = client.get("/v1/automation/status", headers=admin_headers)
        heartbeat_response = client.post("/v1/automation/heartbeat", json={"force": False}, headers=admin_headers)
        learning_response = client.post(
            "/v1/automation/self-improvement",
            json={"force": True, "background": False},
            headers=admin_headers,
        )
        browser_start_response = client.post("/v1/automation/browser/start", json={"restart": True}, headers=admin_headers)
        browser_status_response = client.get("/v1/automation/browser/status", headers=admin_headers)
        browser_stop_response = client.post("/v1/automation/browser/stop", headers=admin_headers)

    assert status_response.status_code == 200
    assert status_response.json()["gateway"]["channels"][0] == "chat"
    assert heartbeat_response.json()["result"] == {"queued_total": 2, "forced": False}
    assert learning_response.json()["result"] == {"cycle": 4, "hypotheses_updated": True}
    assert browser_start_response.json()["running"] is True
    assert browser_status_response.json()["running"] is True
    assert browser_stop_response.json()["running"] is False


def test_api_rejects_missing_bearer_token():
    from fastapi.testclient import TestClient

    broker = ImmediateResultBroker()
    orchestrator = DadBotOrchestrator(
        broker,
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    app = create_api_app(orchestrator, config=service_config())

    with TestClient(app) as client:
        response = client.post("/v1/sessions/secure-session/chat", json={"user_input": "hello"})

    assert response.status_code == 401


def test_api_rejects_tenant_mismatch():
    from fastapi.testclient import TestClient

    broker = ImmediateResultBroker()
    orchestrator = DadBotOrchestrator(
        broker,
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    app = create_api_app(orchestrator, config=service_config())

    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/mismatch-session/chat",
            json={"user_input": "hello", "tenant_id": "family-b"},
            headers=auth_headers(tenant_id="family-a"),
        )

    assert response.status_code == 403


def test_api_rate_limits_repeated_write_requests():
    from fastapi.testclient import TestClient

    broker = ImmediateResultBroker()
    orchestrator = DadBotOrchestrator(
        broker,
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    app = create_api_app(orchestrator, config=service_config(write_limit=2))

    with TestClient(app) as client:
        first = client.post("/v1/sessions/rate-limit/chat", json={"user_input": "one"}, headers=auth_headers())
        second = client.post("/v1/sessions/rate-limit/chat", json={"user_input": "two"}, headers=auth_headers())
        third = client.post("/v1/sessions/rate-limit/chat", json={"user_input": "three"}, headers=auth_headers())

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
