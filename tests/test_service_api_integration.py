import socket
import threading
import time

import pytest

uvicorn = pytest.importorskip("uvicorn")
pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")

from dadbot_system import DadServiceClient, DadBotOrchestrator, InMemoryEventBus, InMemoryStateStore, ServiceClientConfig, ServiceConfig, create_api_app
from dadbot_system.contracts import ChatResponse, WorkerResult


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
    app = create_api_app(orchestrator, config=ServiceConfig())
    port = reserve_free_port()
    server, thread = start_live_server(app, port)

    try:
        client = DadServiceClient(
            ServiceClientConfig(
                base_url=f"http://127.0.0.1:{port}",
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
    app = create_api_app(orchestrator, config=ServiceConfig())

    with TestClient(app) as client:
        with client.websocket_connect("/sessions/socket-session/events/stream?tenant_id=family-a") as websocket:
            response = client.post(
                "/sessions/socket-session/chat",
                json={"user_input": "hello from websocket", "tenant_id": "family-a"},
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
    app = create_api_app(orchestrator, config=ServiceConfig())

    with TestClient(app) as client:
        response = client.post(
            "/sessions/turn-session/turn",
            json={"user_input": "hello from turn endpoint", "tenant_id": "family-a", "timeout_seconds": 3},
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
    app = create_api_app(orchestrator, config=ServiceConfig())

    with TestClient(app) as client:
        post_response = client.post(
            "/sessions/replay-session/chat",
            json={"user_input": "hello replay", "tenant_id": "family-a"},
        )
        assert post_response.status_code == 200

        replay_response = client.get(
            "/sessions/replay-session/replay?tenant_id=family-a&event_type=request.accepted&limit=10"
        )

    payload = replay_response.json()
    assert replay_response.status_code == 200
    assert payload["session_id"] == "replay-session"
    assert payload["tenant_id"] == "family-a"
    assert payload["event_count"] >= 1
    assert all(event["event_type"] == "request.accepted" for event in payload["events"])