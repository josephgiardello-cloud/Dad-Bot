import base64
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytestmark = pytest.mark.integration

uvicorn = pytest.importorskip("uvicorn")
pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")

# Keep optional service imports behind importorskip so missing dependencies skip collection cleanly.
from dadbot.core.execution_contract import TurnResponse  # noqa: E402
from dadbot_system import (  # noqa: E402
    DadBotOrchestrator,
    DadServiceClient,
    InMemoryEventBus,
    InMemoryStateStore,
    ServiceClientConfig,
    ServiceConfig,
    create_api_app,
)
from dadbot_system.contracts import ChatResponse, ToolCapability, WorkerResult  # noqa: E402
from dadbot_system.delegation import SQLiteDelegationStore  # noqa: E402
from dadbot_system.security import ServiceTokenManager  # noqa: E402
from dadbot_system.worker import DadBotTaskProcessor  # noqa: E402

SERVICE_TOKEN_SECRET = "integration-service-token-secret"


def service_config(*, read_limit: int = 120, write_limit: int = 60, admin_limit: int = 20) -> ServiceConfig:
    return ServiceConfig.from_dict(
        {
            "security": {
                "auth_required": True,
                "token_secret": SERVICE_TOKEN_SECRET,
                "token_issuer": "dadbot",
                "authority_sqlite_path": ImmediateResultBroker.authority_path,
                "max_requests_per_minute": read_limit,
                "max_write_requests_per_minute": write_limit,
                "max_admin_requests_per_minute": admin_limit,
            }
        }
    )


def auth_headers(
    *,
    tenant_id: str = "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa",
    scopes: tuple[str, ...] = ("read", "write"),
    dadbot_key: str = "",
) -> dict[str, str]:
    token = ServiceTokenManager(SERVICE_TOKEN_SECRET, issuer="dadbot").issue(
        subject="integration-client",
        tenant_id=tenant_id,
        scopes=scopes,
        allowed_tools=[],
        ttl_seconds=3600.0,
    )
    headers = {"Authorization": f"Bearer {token}"}
    if dadbot_key:
        headers["X-DADBOT-KEY"] = dadbot_key
    return headers


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
    authority_path = ""

    def __init__(self):
        self.results = []

    def append_executed(self, task, template):
        class Bot:
            def __init__(self, tenant_id):
                self.config = type("Config", (), {"tenant_id": tenant_id})()
                self.ACTIVE_MODEL = template.response.active_model

            def load_session_state_snapshot(self, snapshot):
                pass

            async def execute_turn(self, request):
                return TurnResponse(reply=template.response.reply, should_end=template.response.should_end)

            def snapshot_session_state(self):
                return template.session_state

        processor = DadBotTaskProcessor(
            service_config(), bot_factory=lambda *, model_name, tenant_id: Bot(tenant_id)
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(processor.process, task).result()
        self.results.append(result)

    def enqueue(self, task):
        reply = f"Dad heard: {task.request.user_input}"
        self.append_executed(task,
            WorkerResult(
                task_id=task.task_id,
                session_id=task.session_id,
                request_id=task.request.request_id,
                tenant_id=task.tenant_id,
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
                    tenant_id=task.tenant_id,
                    reply=reply,
                    active_model=task.request.requested_model or "llama3.2",
                ),
            )
        )

    def get_result_nowait(self):
        if not self.results:
            return None
        return self.results.pop(0)


class MandatoryHaltBroker(ImmediateResultBroker):
    def enqueue(self, task):
        self.append_executed(task,
            WorkerResult(
                task_id=task.task_id,
                session_id=task.session_id,
                request_id=task.request.request_id,
                tenant_id=task.tenant_id,
                status="completed",
                session_state={
                    "history": [{"role": "assistant", "content": "unsafe raw output"}],
                    "planner_debug": planner_debug_factory(),
                    "memory_store": {},
                    "goal_alignment_mandatory_halt": True,
                    "last_reflection_summary": {
                        "current_risk_level": "critical",
                        "predicted_drift_probability": 1.0,
                    },
                },
                response=ChatResponse(
                    session_id=task.session_id,
                    request_id=task.request.request_id,
                    tenant_id=task.tenant_id,
                    reply="unsafe raw output",
                    active_model=task.request.requested_model or "llama3.2",
                ),
            )
        )


class CountingBroker(ImmediateResultBroker):
    def __init__(self):
        super().__init__()
        self.enqueue_count = 0

    def enqueue(self, task):
        self.enqueue_count += 1
        super().enqueue(task)


class SparseStateBroker(ImmediateResultBroker):
    def enqueue(self, task):
        reply = f"Dad heard: {task.request.user_input}"
        self.append_executed(task,
            WorkerResult(
                task_id=task.task_id,
                session_id=task.session_id,
                request_id=task.request.request_id,
                tenant_id=task.tenant_id,
                status="completed",
                session_state={
                    "history": [{"role": "assistant", "content": reply}],
                },
                response=ChatResponse(
                    session_id=task.session_id,
                    request_id=task.request.request_id,
                    tenant_id=task.tenant_id,
                    reply=reply,
                    active_model=task.request.requested_model or "llama3.2",
                ),
            )
        )


@pytest.fixture(autouse=True)
def integration_authority(tmp_path, monkeypatch):
    path = str(tmp_path / "authority.sqlite")
    SQLiteDelegationStore.initialize(path)
    store = SQLiteDelegationStore(path, SERVICE_TOKEN_SECRET)
    for subject in ("integration-client", "dadbot-client"):
        for tenant in ("default", "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa", "t1_bbbbbbbbbbbbbbbbbbbbbbbbbb"):
            store.grant_membership(
                issuer="dadbot",
                subject=subject,
                tenant_id=tenant,
                scopes={"admin", "read", "write"},
            )
    monkeypatch.setattr(ImmediateResultBroker, "authority_path", path)


class FakeRuntimeBot:
    def __init__(self) -> None:
        self.config = type("Config", (), {"tenant_id": "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"})()
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


class RuntimeSafetyConfiguredBot(FakeRuntimeBot):
    def __init__(self) -> None:
        super().__init__()
        self.safety_support = type(
            "SafetySupport",
            (),
            {
                "settings": lambda _self: {"high_risk_phrases": ["custom crisis keyword"]},
                "has_negated_reassurance": lambda _self, text: "not custom crisis keyword" in str(text or ""),
            },
        )()


class SpatialRuntimeBot(FakeRuntimeBot):
    def __init__(self) -> None:
        super().__init__()
        self.tts_manager = type(
            "TTSManagerStub",
            (),
            {"get_tts_audio": lambda _self, _text: (b"RIFFTEST", "", "audio/wav")},
        )()

    def process_user_message(self, user_input: str, attachments=None):
        _ = user_input, attachments
        raise AssertionError("Authenticated spatial requests must run through delegated task execution")


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
            tenant_id="t1_aaaaaaaaaaaaaaaaaaaaaaaaaa",
        )

        assert health["status"] == "ok"
        assert result.reply == "Dad heard: hello from integration"
        assert result.active_model == "llama3.2"
        assert result.task_payload["status"] == "completed"
        assert result.task_payload["tenant_id"] == "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"
        assert result.session_state["history"][1]["content"] == "Dad heard: hello from integration"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_health_endpoint_is_exempt_from_auth_but_api_routes_are_not(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DADBOT_SECRET_KEY", "integration-shared-key")
    orchestrator = DadBotOrchestrator(
        ImmediateResultBroker(),
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    app = create_api_app(orchestrator, config=service_config())

    with TestClient(app) as client:
        health = client.get("/v1/health")
        protected = client.get(
            "/v1/gateway/channels",
            headers={"X-DADBOT-KEY": "integration-shared-key"},
        )

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert protected.status_code == 401


def test_readiness_endpoint_checks_worker_and_state_store_availability():
    from fastapi.testclient import TestClient

    class ReadyWorkers:
        worker_count = 1
        is_ready = True

        def shutdown(self):
            return None

    class UnavailableStateStore(InMemoryStateStore):
        def check_available(self):
            raise OSError("state backend unavailable")

    ready_orchestrator = DadBotOrchestrator(
        ImmediateResultBroker(),
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    ready_app = create_api_app(
        ready_orchestrator,
        worker_manager=ReadyWorkers(),
        config=service_config(),
    )

    unavailable_orchestrator = DadBotOrchestrator(
        ImmediateResultBroker(),
        state_store=UnavailableStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    unavailable_app = create_api_app(
        unavailable_orchestrator,
        worker_manager=ReadyWorkers(),
        config=service_config(),
    )

    with TestClient(ready_app) as ready_client, TestClient(unavailable_app) as unavailable_client:
        ready = ready_client.get("/v1/ready")
        unavailable = unavailable_client.get("/v1/ready")

    assert ready.status_code == 200
    assert ready.json() == {
        "status": "ready",
        "checks": {
            "dependencies": {"state_store": "available", "authority_store": "available"},
            "runtime": {"workers": "available"},
            "model": {"status": "unprobed", "probed_at": None, "error_type": None},
            "vector": {"status": "unprobed", "probed_at": None, "error_type": None},
            "memory": {"status": "unprobed", "probed_at": None, "error_type": None},
        },
    }
    assert unavailable.status_code == 503
    assert unavailable.json()["checks"]["dependencies"]["state_store"] == "unavailable"

    class QuarantinedWorkers:
        worker_count = 1
        readiness_status = "available"

        @property
        def is_ready(self):
            return self.readiness_status == "available"

        def shutdown(self):
            return None

    workers = QuarantinedWorkers()
    readiness_app = create_api_app(
        ready_orchestrator,
        worker_manager=workers,
        config=service_config(),
    )
    with TestClient(readiness_app) as readiness_client:
        for state, expected_status in (
            ("starting", 503),
            ("available", 200),
            ("quarantined", 503),
            ("unavailable", 503),
        ):
            workers.readiness_status = state
            response = readiness_client.get("/v1/ready")
            assert response.status_code == expected_status
            payload = response.json()
            assert payload["status"] == ("ready" if expected_status == 200 else "not_ready")
            assert payload["checks"]["runtime"]["workers"] == state


def test_readiness_model_probe_checks_endpoint_and_model_name(monkeypatch):
    from dadbot_system import api as api_module

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"models":[{"name":"test-model"}]}'

    requested_urls = []

    def fake_urlopen(request, timeout):
        requested_urls.append((request.full_url, timeout))
        return Response()

    monkeypatch.setattr(api_module, "urlopen", fake_urlopen)

    assert api_module._probe_model_dependency("http://ollama:11434", "test-model", 0.2) == ("ok", None)
    assert (
        api_module._probe_model_dependency("http://ollama:11434", "missing-model", 0.2)
        == ("degraded", "ModelNotFoundError")
    )
    assert requested_urls == [
        ("http://ollama:11434/api/tags", 0.2),
        ("http://ollama:11434/api/tags", 0.2),
    ]


def test_readiness_vector_probe_checks_http_endpoint(monkeypatch):
    from dadbot_system import api as api_module

    class Response:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request.full_url, request.get_method(), timeout))
        return Response()

    monkeypatch.setattr(api_module, "urlopen", fake_urlopen)

    assert api_module._probe_vector_dependency("https://vector.example.test", "", 0.2) == ("ok", None)
    assert requests == [("https://vector.example.test", "HEAD", 0.2)]


def test_readiness_vector_probe_checks_postgres_connection(monkeypatch):
    import psycopg

    from dadbot_system import api as api_module

    executed = []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, query):
            executed.append(query)

        def fetchone(self):
            return (1,)

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def cursor(self):
            return Cursor()

    connections = []

    def fake_connect(dsn, *, connect_timeout):
        connections.append((dsn, connect_timeout))
        return Connection()

    monkeypatch.setattr(psycopg, "connect", fake_connect)

    assert (
        api_module._probe_vector_dependency("postgresql://db.example.test/vectors", "", 0.2)
        == ("ok", None)
    )
    assert connections == [("postgresql://db.example.test/vectors", 1)]
    assert executed == ["SELECT 1"]


def test_readiness_memory_probe_checks_backend_connection():
    from contextlib import nullcontext
    from types import SimpleNamespace

    from dadbot_system.api import _probe_memory_dependency

    executed = []
    cursor = SimpleNamespace(
        execute=executed.append,
        fetchone=lambda: (1,),
    )
    index = SimpleNamespace(with_connection=lambda operation: operation(
        SimpleNamespace(cursor=lambda: nullcontext(cursor)),
    ))
    runtime_bot = SimpleNamespace(context_service=SimpleNamespace(semantic_index=index))

    assert _probe_memory_dependency(runtime_bot, 0.2) == ("ok", None)
    assert executed == ["SELECT 1"]
    assert _probe_memory_dependency(SimpleNamespace(), 0.2) == (
        "unavailable",
        "MemoryBackendUnavailableError",
    )


def test_readiness_probe_timeout_is_bounded_and_reports_error_type(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from fastapi.testclient import TestClient

    from dadbot_system import api as api_module

    calls = []

    def slow_model_probe(_endpoint, _model_name, _timeout):
        time.sleep(0.25)
        calls.append("finished")
        return "ok", None

    monkeypatch.setattr(api_module, "_probe_model_dependency", slow_model_probe)
    orchestrator = DadBotOrchestrator(
        ImmediateResultBroker(),
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    config = service_config()
    config.api.readiness_probes = ["model"]
    config.api.readiness_probe_timeout_ms = 25
    app = create_api_app(orchestrator, worker_manager=type(
        "ReadyWorkers",
        (),
        {"worker_count": 1, "is_ready": True, "readiness_status": "available", "shutdown": lambda self: None},
    )(), config=config)

    with TestClient(app) as client:
        started = time.monotonic()
        response = client.get("/v1/ready")
        elapsed = time.monotonic() - started

    assert response.status_code == 200
    assert elapsed < 0.15
    assert response.json()["checks"]["model"]["status"] == "unavailable"
    assert response.json()["checks"]["model"]["error_type"] == "TimeoutError"
    assert response.json()["checks"]["model"]["probed_at"]

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(time.sleep, 0.3).result()
    assert calls == ["finished"]


def test_readiness_probe_cache_serializes_concurrent_requests(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from fastapi.testclient import TestClient

    from dadbot_system import api as api_module

    call_count = 0

    def model_probe(_endpoint, _model_name, _timeout):
        nonlocal call_count
        call_count += 1
        time.sleep(0.05)
        return "ok", None

    monkeypatch.setattr(api_module, "_probe_model_dependency", model_probe)
    orchestrator = DadBotOrchestrator(
        ImmediateResultBroker(),
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    config = service_config()
    config.api.readiness_probes = ["model"]
    config.api.readiness_probe_cache_seconds = 30
    workers = type(
        "ReadyWorkers",
        (),
        {"worker_count": 1, "is_ready": True, "readiness_status": "available", "shutdown": lambda self: None},
    )()
    app = create_api_app(orchestrator, worker_manager=workers, config=config)

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=8) as executor:
        responses = list(executor.map(lambda _index: client.get("/v1/ready"), range(8)))

    assert all(response.status_code == 200 for response in responses)
    assert call_count == 1
    assert len({response.json()["checks"]["model"]["probed_at"] for response in responses}) == 1


def test_required_model_probe_transitions_readiness_up_down_up(monkeypatch):
    from fastapi.testclient import TestClient

    from dadbot_system import api as api_module

    probe_results = iter((("ok", None), ("unavailable", "ConnectionError"), ("ok", None)))
    monkeypatch.setattr(api_module, "_probe_model_dependency", lambda *_args: next(probe_results))
    orchestrator = DadBotOrchestrator(
        ImmediateResultBroker(),
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    config = service_config()
    config.api.readiness_probes = ["model"]
    config.api.readiness_required_probes = ["model"]
    config.api.readiness_probe_cache_seconds = 0
    workers = type(
        "ReadyWorkers",
        (),
        {"worker_count": 1, "is_ready": True, "readiness_status": "available", "shutdown": lambda self: None},
    )()
    app = create_api_app(orchestrator, worker_manager=workers, config=config)

    with TestClient(app) as client:
        responses = [client.get("/v1/ready") for _ in range(3)]

    assert [response.status_code for response in responses] == [200, 503, 200]
    assert [response.json()["checks"]["model"]["status"] for response in responses] == [
        "ok",
        "unavailable",
        "ok",
    ]
    assert all(response.json()["checks"]["vector"]["status"] == "unprobed" for response in responses)


def test_required_memory_probe_transitions_readiness_unprobed_ok_degraded(monkeypatch):
    from fastapi.testclient import TestClient

    from dadbot_system import api as api_module

    results = iter((("ok", None), ("degraded", "OSError")))
    monkeypatch.setattr(api_module, "_probe_memory_dependency", lambda *_args: next(results))
    orchestrator = DadBotOrchestrator(
        ImmediateResultBroker(),
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    workers = type(
        "ReadyWorkers",
        (),
        {"worker_count": 1, "is_ready": True, "readiness_status": "available", "shutdown": lambda self: None},
    )()

    default_app = create_api_app(orchestrator, worker_manager=workers, config=service_config())
    config = service_config()
    config.api.readiness_probes = ["memory"]
    config.api.readiness_required_probes = ["memory"]
    config.api.readiness_probe_cache_seconds = 0
    configured_app = create_api_app(
        orchestrator,
        worker_manager=workers,
        config=config,
        runtime_bot=type("RuntimeBot", (), {})(),
    )

    with TestClient(default_app) as default_client, TestClient(configured_app) as client:
        unprobed = default_client.get("/v1/ready")
        responses = [client.get("/v1/ready") for _ in range(2)]

    assert unprobed.status_code == 200
    assert unprobed.json()["checks"]["memory"]["status"] == "unprobed"
    assert [response.status_code for response in responses] == [200, 503]
    assert [response.json()["checks"]["memory"]["status"] for response in responses] == ["ok", "degraded"]


def test_readiness_probe_environment_configuration(monkeypatch):
    monkeypatch.setenv("DADBOT_READINESS_PROBES", "MODEL, vector, memory")
    monkeypatch.setenv("DADBOT_READINESS_REQUIRED", "model")
    monkeypatch.setenv("DADBOT_READINESS_PROBE_TIMEOUT_MS", "250")
    monkeypatch.setenv("DADBOT_READINESS_PROBE_CACHE_SECONDS", "3")

    config = ServiceConfig.from_environment()

    assert config.api.readiness_probes == ["model", "vector", "memory"]
    assert config.api.readiness_required_probes == ["model"]
    assert config.api.readiness_probe_timeout_ms == 250
    assert config.api.readiness_probe_cache_seconds == 3


def test_explain_session_actions_is_tenant_scoped_and_uses_persisted_audit_records(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DADBOT_SECRET_KEY", "integration-shared-key")
    orchestrator = DadBotOrchestrator(
        ImmediateResultBroker(),
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    orchestrator.session("shared-session", "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa").load_snapshot(
        {
            "last_tool_actions": [
                {
                    "tool_name": "set_reminder",
                    "status": "succeeded",
                    "authorization_decision": "verified",
                    "authority_mode": "delegated_task",
                    "trace_id": "trace-1",
                    "timestamp": "2026-11-09T12:00:00+00:00",
                    "input_hash": "a" * 64,
                    "output_hash": "b" * 64,
                }
            ]
        }
    )
    app = create_api_app(orchestrator, config=service_config())
    tenant_a_headers = auth_headers(
        tenant_id="t1_aaaaaaaaaaaaaaaaaaaaaaaaaa",
        dadbot_key="integration-shared-key",
    )
    tenant_b_headers = auth_headers(
        tenant_id="t1_bbbbbbbbbbbbbbbbbbbbbbbbbb",
        dadbot_key="integration-shared-key",
    )

    with TestClient(app) as client:
        response = client.get(
            "/v1/sessions/shared-session/actions/explain?limit=1",
            headers=tenant_a_headers,
        )
        mismatch = client.get(
            "/v1/sessions/shared-session/actions/explain?tenant_id=t1_aaaaaaaaaaaaaaaaaaaaaaaaaa",
            headers=tenant_b_headers,
        )

    assert response.status_code == 200
    assert response.json()["tenant_id"] == "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert response.json()["actions"][0]["tool_name"] == "set_reminder"
    assert response.json()["actions"][0]["explanation"] == (
        "set_reminder completed after verified delegated authority."
    )
    assert mismatch.status_code == 403


def test_doctor_is_admin_protected_and_reports_bounded_runtime_signals(monkeypatch):
    from collections import deque

    from fastapi.testclient import TestClient

    import dadbot.services.memory_service as memory_service_module
    from dadbot.services.memory_service import MemoryService

    monkeypatch.setenv("DADBOT_SECRET_KEY", "integration-shared-key")
    monkeypatch.setattr(memory_service_module, "_RECENT_RETRIEVAL_STATUSES", deque(maxlen=256))
    for status in ("ok", "partial", "degraded"):
        MemoryService.record_retrieval_status({}, status=status, errors=[], source="test")

    class ReadyWorkers:
        worker_count = 1
        is_ready = True

        def shutdown(self):
            return None

    class RuntimeBot:
        ACTIVE_MODEL = "local-model"

        def current_runtime_health_snapshot(self, **kwargs):
            assert kwargs == {"force": False, "log_warnings": False, "persist": False}
            return {
                "level": "green",
                "health_score": 98,
                "memory_context_ratio": 0.25,
                "prompt_guard_trim_count": 0,
                "recent_runtime_issue_count": 0,
                "private_detail": "must not be returned",
            }

    orchestrator = DadBotOrchestrator(
        ImmediateResultBroker(),
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    orchestrator.tool_registry().register(ToolCapability(name="test-tool", description="Test capability"))
    config = service_config()
    config.persistence.vector_store_url = "https://vector.example.test"
    app = create_api_app(
        orchestrator,
        worker_manager=ReadyWorkers(),
        config=config,
        runtime_bot=RuntimeBot(),
    )

    read_headers = auth_headers(dadbot_key="integration-shared-key")
    admin_headers = auth_headers(
        scopes=("admin", "read"),
        dadbot_key="integration-shared-key",
    )
    with TestClient(app) as client:
        denied = client.get("/v1/doctor", headers=read_headers)
        authorized = client.get("/v1/doctor", headers=admin_headers)

    assert denied.status_code == 403
    assert authorized.status_code == 200
    payload = authorized.json()
    assert payload["status"] == "ok"
    assert payload["checks"] == {
        "dependencies": {"state_store": "available", "authority_store": "available"},
        "runtime": {"workers": "available"},
        "model": {"status": "unprobed", "probed_at": None, "error_type": None},
        "vector": {"status": "unprobed", "probed_at": None, "error_type": None},
        "memory": {"status": "unprobed", "probed_at": None, "error_type": None},
    }
    assert payload["memory_retrieval_status_counts"] == {
        "window_size": 3,
        "ok": 1,
        "partial": 1,
        "degraded": 1,
    }
    assert payload["tool_registry"] == {"status": "available", "registered_count": 1}
    assert payload["model_runtime"] == {
        "status": "configured_not_probed",
        "active_model": "local-model",
    }
    assert payload["vector_store"] == {"status": "configured_not_probed"}
    assert payload["trace_backend"] == {"status": "disabled"}
    assert payload["runtime_health"]["health_score"] == 98
    assert payload["pending_approvals"] == {"status": "not_exposed"}
    assert "private_detail" not in str(payload)


def test_authenticated_worker_does_not_dequeue_while_authority_is_unavailable(monkeypatch):
    from dadbot_system import worker as worker_module

    order = []
    authority_status = [False]

    class Processor:
        def __init__(self, _config, **_kwargs):
            self.poll_count = 0

        def poll_session_revocations(self):
            self.poll_count += 1
            available = self.poll_count > 1
            order.append(f"authority-{available}")
            return available

        def process(self, task):
            raise AssertionError(f"Unexpected task execution: {task}")

    class RequestQueue:
        def get(self, **_kwargs):
            order.append("dequeue")
            return None

    class ResultQueue:
        pass

    monkeypatch.setattr(worker_module, "DadBotTaskProcessor", Processor)
    monkeypatch.setattr(worker_module, "configure_logging", lambda _settings: None)
    config = ServiceConfig.from_dict(
        {"security": {"auth_required": True}, "telemetry": {"otel_enabled": False}}
    )

    worker_module.worker_process_main(
        config.to_dict(),
        RequestQueue(),
        ResultQueue(),
        authority_status=authority_status,
    )

    assert order == ["authority-False", "authority-True", "dequeue"]
    assert authority_status == [True]

    class AliveProcess:
        exitcode = None

        def is_alive(self):
            return True

    manager = worker_module.WorkerProcessManager(None, config)
    worker_count = max(1, config.workers.worker_count)
    manager._processes = [AliveProcess() for _ in range(worker_count)]
    manager._authority_status = [False] * worker_count
    assert not manager.is_ready
    manager._authority_status = [True] * worker_count
    assert manager.is_ready


def test_authenticated_worker_quarantines_queued_task_until_authority_recovers(monkeypatch):
    from dadbot_system import worker as worker_module

    order = []
    authority_status = [worker_module.WORKER_AUTHORITY_STARTING]
    task = object()
    processed = []

    class Processor:
        def __init__(self, _config, **_kwargs):
            self.poll_count = 0

        def poll_session_revocations(self):
            self.poll_count += 1
            if self.poll_count == 1:
                order.append("authority-revoked")
                return False
            if self.poll_count == 2:
                assert authority_status[0] == worker_module.WORKER_AUTHORITY_QUARANTINED
                assert processed == []
                order.append("authority-restored")
                return True
            raise AssertionError("Worker polled authority after shutdown")

        def process(self, queued_task):
            assert queued_task is task
            order.append("effect-executed")
            processed.append(queued_task)
            stop_event.stopped = True
            return worker_module.WorkerResult(
                task_id="task",
                session_id="session",
                request_id="request",
                tenant_id="tenant",
                status="completed",
            )

    class RequestQueue:
        def get(self, **_kwargs):
            order.append("dequeue")
            return task

    class ResultQueue:
        def put(self, result):
            order.append(f"result-{result.status}")

    class StopEvent:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, _timeout):
            return None

    stop_event = StopEvent()
    monkeypatch.setattr(worker_module, "DadBotTaskProcessor", Processor)
    monkeypatch.setattr(worker_module, "configure_logging", lambda _settings: None)
    config = ServiceConfig.from_dict(
        {"security": {"auth_required": True}, "telemetry": {"otel_enabled": False}}
    )

    worker_module.worker_process_main(
        config.to_dict(),
        RequestQueue(),
        ResultQueue(),
        stop_event=stop_event,
        authority_status=authority_status,
    )

    assert order == [
        "authority-revoked",
        "authority-restored",
        "dequeue",
        "effect-executed",
        "result-completed",
    ]
    assert processed == [task]
    assert authority_status == [worker_module.WORKER_AUTHORITY_AVAILABLE]

    class AliveProcess:
        exitcode = None

        def is_alive(self):
            return True

    manager = worker_module.WorkerProcessManager(None, config)
    worker_count = max(1, config.workers.worker_count)
    manager._processes = [AliveProcess() for _ in range(worker_count)]
    manager._authority_status = [worker_module.WORKER_AUTHORITY_QUARANTINED] * worker_count
    assert manager.readiness_status == "quarantined"
    assert not manager.is_ready
    manager._authority_status = [worker_module.WORKER_AUTHORITY_AVAILABLE] * worker_count
    assert manager.readiness_status == "available"
    assert manager.is_ready


@pytest.mark.parametrize("effect_committed_before_retry", [False, True])
def test_authenticated_worker_holds_dequeued_task_across_authority_loss(
    monkeypatch, effect_committed_before_retry
):
    import asyncio

    from dadbot.core.control_plane import ExecutionControlPlane, InMemoryExecutionLedger, SessionRegistry
    from dadbot_system import worker as worker_module

    order = []
    effect_calls = []

    async def execute_effect(_session, _job):
        effect_calls.append("effect")
        return "done", False

    control_plane = ExecutionControlPlane(
        registry=SessionRegistry(),
        ledger=InMemoryExecutionLedger(),
        kernel_executor=execute_effect,
    )
    authority_status = [worker_module.WORKER_AUTHORITY_STARTING]
    task = object()
    processed = 0
    stop_event = type(
        "StopEvent",
        (),
        {
            "stopped": False,
            "is_set": lambda self: self.stopped,
            "wait": lambda self, _timeout: None,
        },
    )()

    class Processor:
        def __init__(self, _config, **_kwargs):
            self.poll_count = 0

        def poll_session_revocations(self):
            self.poll_count += 1
            available = self.poll_count >= 3
            order.append(f"poll-{available}")
            return available

        def process(self, queued_task):
            nonlocal processed
            assert queued_task is task
            processed += 1
            if not effect_committed_before_retry and processed == 1:
                return _retryable_authority_result(worker_module)
            asyncio.run(
                control_plane.submit_turn(
                    session_id="session",
                    user_input="same request",
                    metadata={"request_id": "request", "effect_id": "effect", "trace_id": "trace"},
                )
            )
            if effect_committed_before_retry and processed == 1:
                return _retryable_authority_result(worker_module)
            return worker_module.WorkerResult(
                task_id="task",
                session_id="session",
                request_id="request",
                tenant_id="tenant",
                status="completed",
            )

    class RequestQueue:
        def __init__(self):
            self.dequeued = False

        def get(self, **_kwargs):
            assert authority_status[0] == worker_module.WORKER_AUTHORITY_AVAILABLE
            assert not self.dequeued
            self.dequeued = True
            order.append("dequeue")
            return task

    class ResultQueue:
        def put(self, result):
            assert result.status == "completed"
            order.append("publish")
            stop_event.stopped = True

    monkeypatch.setattr(worker_module, "DadBotTaskProcessor", Processor)
    monkeypatch.setattr(worker_module, "configure_logging", lambda _settings: None)
    config = ServiceConfig.from_dict(
        {"security": {"auth_required": True}, "telemetry": {"otel_enabled": False}}
    )
    worker_module.worker_process_main(
        config.to_dict(),
        RequestQueue(),
        ResultQueue(),
        stop_event=stop_event,
        authority_status=authority_status,
    )

    assert processed == 2
    assert effect_calls == ["effect"]
    assert order.count("dequeue") == 1
    assert order.count("publish") == 1
    assert authority_status == [worker_module.WORKER_AUTHORITY_AVAILABLE]


def _retryable_authority_result(worker_module):
    return worker_module.WorkerResult(
        task_id="task",
        session_id="session",
        request_id="request",
        tenant_id="tenant",
        status="failed",
        error=worker_module.AUTHORITY_UNAVAILABLE,
        retryable=True,
    )


def test_authenticated_worker_uses_capped_backoff_and_recovers_from_authority_flap(monkeypatch):
    from dadbot_system import worker as worker_module

    authority_status = [worker_module.WORKER_AUTHORITY_STARTING] * 2
    poll_results = [False, False, False, False, False, True, False, False, False, False, False, True]
    backoff_delays = []
    status_history = []
    processed = []
    tasks = [object(), object()]
    expected_tasks = list(tasks)
    stop_event = type(
        "StopEvent",
        (),
        {
            "stopped": False,
            "is_set": lambda self: self.stopped,
            "wait": lambda self, timeout: (
                backoff_delays.append(timeout),
                status_history.append(authority_status[0]),
            ),
        },
    )()

    class Processor:
        def __init__(self, _config, **_kwargs):
            pass

        def poll_session_revocations(self):
            return poll_results.pop(0)

        def process(self, task):
            processed.append(task)
            return worker_module.WorkerResult(
                task_id=str(len(processed)),
                session_id="session",
                request_id=str(len(processed)),
                tenant_id="tenant",
                status="completed",
            )

    class RequestQueue:
        def get(self, **_kwargs):
            task = tasks.pop(0)
            assert authority_status[0] == worker_module.WORKER_AUTHORITY_AVAILABLE
            return task

    class ResultQueue:
        def put(self, _result):
            if len(processed) == 2:
                stop_event.stopped = True

    class AliveProcess:
        exitcode = None

        def is_alive(self):
            return True

    monkeypatch.setattr(worker_module, "DadBotTaskProcessor", Processor)
    monkeypatch.setattr(worker_module, "configure_logging", lambda _settings: None)
    config = ServiceConfig.from_dict(
        {
            "security": {"auth_required": True},
            "workers": {"worker_count": 2},
            "telemetry": {"otel_enabled": False},
        }
    )
    worker_module.worker_process_main(
        config.to_dict(),
        RequestQueue(),
        ResultQueue(),
        stop_event=stop_event,
        authority_status=authority_status,
        worker_index=0,
    )

    assert processed == expected_tasks
    assert backoff_delays == [
        0.1,
        0.2,
        0.4,
        0.8,
        1.6,
        0.1,
        0.2,
        0.4,
        0.8,
        1.6,
    ]
    assert status_history == [
        worker_module.WORKER_AUTHORITY_QUARANTINED,
        worker_module.WORKER_AUTHORITY_QUARANTINED,
        worker_module.WORKER_AUTHORITY_UNAVAILABLE,
        worker_module.WORKER_AUTHORITY_UNAVAILABLE,
        worker_module.WORKER_AUTHORITY_UNAVAILABLE,
        worker_module.WORKER_AUTHORITY_QUARANTINED,
        worker_module.WORKER_AUTHORITY_QUARANTINED,
        worker_module.WORKER_AUTHORITY_UNAVAILABLE,
        worker_module.WORKER_AUTHORITY_UNAVAILABLE,
        worker_module.WORKER_AUTHORITY_UNAVAILABLE,
    ]
    assert poll_results == []
    assert authority_status[0] == worker_module.WORKER_AUTHORITY_AVAILABLE

    manager = worker_module.WorkerProcessManager(None, config)
    manager._processes = [AliveProcess(), AliveProcess()]
    manager._authority_status = [worker_module.WORKER_AUTHORITY_AVAILABLE, worker_module.WORKER_AUTHORITY_QUARANTINED]
    assert manager.readiness_status == "quarantined"
    manager._authority_status = [
        worker_module.WORKER_AUTHORITY_AVAILABLE,
        worker_module.WORKER_AUTHORITY_UNAVAILABLE,
    ]
    assert manager.readiness_status == "unavailable"


def test_authenticated_worker_shutdown_interrupts_authority_wait(monkeypatch):
    from dadbot_system import worker as worker_module

    class Processor:
        def __init__(self, _config, **_kwargs):
            pass

        def poll_session_revocations(self):
            return False

    class StopEvent:
        waiting = False

        def is_set(self):
            return self.waiting

        def wait(self, _timeout):
            self.waiting = True

    class RequestQueue:
        def get(self, **_kwargs):
            raise AssertionError("Worker dequeued a task before authority recovered")

    class ResultQueue:
        pass

    monkeypatch.setattr(worker_module, "DadBotTaskProcessor", Processor)
    monkeypatch.setattr(worker_module, "configure_logging", lambda _settings: None)
    config = ServiceConfig.from_dict(
        {"security": {"auth_required": True}, "telemetry": {"otel_enabled": False}}
    )
    stop_event = StopEvent()

    worker_module.worker_process_main(
        config.to_dict(),
        RequestQueue(),
        ResultQueue(),
        stop_event=stop_event,
    )

    assert stop_event.waiting


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
        with client.websocket_connect("/v1/sessions/socket-session/events/stream", headers=auth_headers()) as websocket:
            response = client.post(
                "/v1/sessions/socket-session/chat",
                json={"user_input": "hello from websocket", "tenant_id": "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"},
                headers=auth_headers(),
            )

            accepted = websocket.receive_json()
            dispatched = websocket.receive_json()

    assert response.status_code == 200
    assert accepted["event_type"] == "request.accepted"
    assert accepted["tenant_id"] == "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert dispatched["event_type"] == "request.dispatched"
    assert dispatched["tenant_id"] == "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"


def test_api_pulse_websocket_streams_vitals_and_thinking_state():
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
        with client.websocket_connect("/v1/sessions/pulse-session/pulse/stream", headers=auth_headers()) as websocket:
            initial = websocket.receive_json()
            response = client.post(
                "/v1/sessions/pulse-session/chat",
                json={"user_input": "hello pulse", "tenant_id": "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"},
                headers=auth_headers(),
            )
            updated = websocket.receive_json()

    assert response.status_code == 200
    assert initial["event_type"] == "pulse.snapshot"
    assert initial["session_id"] == "pulse-session"
    assert updated["session_id"] == "pulse-session"
    assert updated["trust_meter"]["trust_level"] == 50


def test_api_pulse_websocket_does_not_read_another_tenants_session():
    from fastapi.testclient import TestClient

    orchestrator = DadBotOrchestrator(
        ImmediateResultBroker(),
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    family_a = orchestrator.session("shared-session", "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa")
    family_b = orchestrator.session("shared-session", "t1_bbbbbbbbbbbbbbbbbbbbbbbbbb")
    family_a.snapshot = lambda: {
        "cognition_stream": [{"thought_trace": "family-a thought"}],
        "last_reflection_summary": {"current_risk_level": "high", "marker": "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"},
    }
    family_b.snapshot = lambda: {
        "cognition_stream": [{"thought_trace": "family-b thought"}],
        "last_reflection_summary": {"current_risk_level": "low", "marker": "t1_bbbbbbbbbbbbbbbbbbbbbbbbbb"},
    }
    app = create_api_app(orchestrator, config=service_config())

    with TestClient(app) as client:
        payloads = {}
        for tenant_id in ("t1_aaaaaaaaaaaaaaaaaaaaaaaaaa", "t1_bbbbbbbbbbbbbbbbbbbbbbbbbb"):
            with client.websocket_connect(
                "/v1/sessions/shared-session/pulse/stream",
                headers=auth_headers(tenant_id=tenant_id),
            ) as websocket:
                payloads[tenant_id] = websocket.receive_json()

    assert payloads["t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"]["tenant_id"] == "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert payloads["t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"]["cognition_stream"][0]["thought_trace"] == "family-a thought"
    assert payloads["t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"]["reflection_summary"]["marker"] == "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert payloads["t1_bbbbbbbbbbbbbbbbbbbbbbbbbb"]["tenant_id"] == "t1_bbbbbbbbbbbbbbbbbbbbbbbbbb"
    assert payloads["t1_bbbbbbbbbbbbbbbbbbbbbbbbbb"]["cognition_stream"][0]["thought_trace"] == "family-b thought"
    assert payloads["t1_bbbbbbbbbbbbbbbbbbbbbbbbbb"]["reflection_summary"]["marker"] == "t1_bbbbbbbbbbbbbbbbbbbbbbbbbb"


def test_api_refuses_auth_off_in_production(monkeypatch):
    monkeypatch.setenv("DADBOT_ENVIRONMENT", "production")
    config = ServiceConfig.from_dict({"security": {"auth_required": False}})
    orchestrator = DadBotOrchestrator(
        ImmediateResultBroker(),
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )

    with pytest.raises(RuntimeError, match="cannot be disabled in production"):
        create_api_app(orchestrator, config=config)


def test_api_allows_auth_off_in_development(monkeypatch):
    monkeypatch.setenv("DADBOT_ENVIRONMENT", "development")
    config = ServiceConfig.from_dict({"security": {"auth_required": False}})
    orchestrator = DadBotOrchestrator(
        ImmediateResultBroker(),
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )

    app = create_api_app(orchestrator, config=config)

    assert app is not None


def test_api_spatial_websocket_relays_text_and_audio_payload():
    from fastapi.testclient import TestClient

    class SpatialBroker(ImmediateResultBroker):
        def enqueue(self, task):
            reply = f"Dad heard: {task.request.user_input}"
            template = WorkerResult(
                task_id=task.task_id,
                session_id=task.session_id,
                request_id=task.request.request_id,
                tenant_id=task.tenant_id,
                status="completed",
                response=ChatResponse(
                    session_id=task.session_id,
                    request_id=task.request.request_id,
                    tenant_id=task.tenant_id,
                    reply=reply,
                ),
            )
            self.append_executed(task, template)

    broker = SpatialBroker()
    orchestrator = DadBotOrchestrator(
        broker,
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    runtime_bot = SpatialRuntimeBot()
    app = create_api_app(orchestrator, config=service_config(), runtime_bot=runtime_bot)

    with TestClient(app) as client:
        with client.websocket_connect("/v1/spatial/ws", headers=auth_headers()) as websocket:
            websocket.send_json({"text": "hey dad"})
            response = websocket.receive_json()

    assert response["type"] == "response"
    assert response["user"]["text"] == "hey dad"
    assert response["text"] == "Dad heard: hey dad"
    assert response["audio_mime_type"] == "audio/wav"
    assert base64.b64decode(response["audio_base64"]) == b"RIFFTEST"


def test_spatial_websocket_uses_tenant_bound_tasks_for_same_session_id():
    from fastapi.testclient import TestClient

    class SpatialBroker(ImmediateResultBroker):
        def __init__(self):
            super().__init__()
            self.tasks = []

        def enqueue(self, task):
            self.tasks.append(task)
            reply = f"{task.tenant_id}:{task.request.user_input}"
            self.append_executed(
                task,
                WorkerResult(
                    task_id=task.task_id,
                    session_id=task.session_id,
                    request_id=task.request.request_id,
                    tenant_id=task.tenant_id,
                    status="completed",
                    response=ChatResponse(
                        session_id=task.session_id,
                        request_id=task.request.request_id,
                        tenant_id=task.tenant_id,
                        reply=reply,
                    ),
                ),
            )

    broker = SpatialBroker()
    orchestrator = DadBotOrchestrator(broker, state_store=InMemoryStateStore())
    app = create_api_app(orchestrator, config=service_config(), runtime_bot=SpatialRuntimeBot())

    with TestClient(app) as client:
        responses = []
        for tenant_id in (
            "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa",
            "t1_bbbbbbbbbbbbbbbbbbbbbbbbbb",
        ):
            with client.websocket_connect(
                f"/v1/spatial/ws?tenant_id={tenant_id}",
                headers=auth_headers(tenant_id=tenant_id),
            ) as websocket:
                websocket.send_json({"text": "same session"})
                responses.append(websocket.receive_json())

    assert [task.session_id for task in broker.tasks] == ["spatial-avatar", "spatial-avatar"]
    assert [task.tenant_id for task in broker.tasks] == [
        "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa",
        "t1_bbbbbbbbbbbbbbbbbbbbbbbbbb",
    ]
    assert [response["text"] for response in responses] == [
        "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa:same session",
        "t1_bbbbbbbbbbbbbbbbbbbbbbbbbb:same session",
    ]


def test_admin_runtime_effect_is_tenant_bound_and_rechecks_revocation():
    from fastapi.testclient import TestClient

    from dadbot.core.tool_executor import execute_tool
    from dadbot_system.delegation import SQLiteDelegationStore

    effects = []
    runtime_bot = FakeRuntimeBot()
    runtime_bot.maintenance_scheduler.run_proactive_heartbeat = (
        lambda force=True: execute_tool(
            tool_name="admin-effect",
            executor=lambda: effects.append("executed"),
        ).status
    )
    orchestrator = DadBotOrchestrator(ImmediateResultBroker(), state_store=InMemoryStateStore())
    app = create_api_app(orchestrator, config=service_config(), runtime_bot=runtime_bot)
    store = SQLiteDelegationStore(ImmediateResultBroker.authority_path, SERVICE_TOKEN_SECRET)

    with TestClient(app) as client:
        headers = auth_headers(scopes=("read", "write", "admin"))
        first = client.post("/v1/automation/heartbeat", headers=headers)
        other_tenant = client.post(
            "/v1/automation/heartbeat",
            headers=auth_headers(
                tenant_id="t1_bbbbbbbbbbbbbbbbbbbbbbbbbb",
                scopes=("read", "write", "admin"),
            ),
        )
        store.revoke_membership(
            issuer="dadbot",
            subject="integration-client",
            tenant_id="t1_aaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
        revoked = client.post("/v1/automation/heartbeat", headers=headers)

    assert first.status_code == 200
    assert effects == ["executed"]
    assert other_tenant.status_code == 403
    assert revoked.status_code == 403
    assert effects == ["executed"]


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
            json={"user_input": "hello from turn endpoint", "tenant_id": "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa", "timeout_seconds": 3},
            headers=auth_headers(),
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["task"]["status"] == "completed"
    assert payload["response"]["reply"] == "Dad heard: hello from turn endpoint"
    assert payload["task"]["tenant_id"] == "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert payload["turn"]["response_text"] == "Dad heard: hello from turn endpoint"
    assert payload["turn"]["integrity_status"]["merkle_check_passed"] is True
    assert isinstance(payload["turn"]["subconscious_metadata"], list)


def test_api_turn_endpoint_contract_defaults_when_optional_state_missing():
    from fastapi.testclient import TestClient

    broker = SparseStateBroker()
    orchestrator = DadBotOrchestrator(
        broker,
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    app = create_api_app(orchestrator, config=service_config())

    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/defaults-session/turn",
            json={"user_input": "defaults check", "tenant_id": "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa", "timeout_seconds": 3},
            headers=auth_headers(),
        )

    payload = response.json()
    turn = payload["turn"]
    assert response.status_code == 200
    assert turn["response_text"] == "Dad heard: defaults check"
    assert turn["trust_meter"]["trust_level"] == 50
    assert turn["trust_meter"]["openness_level"] == 50
    assert turn["drift_alarm"]["active"] is False
    assert turn["drift_alarm"]["current_risk_level"] == "low"
    assert turn["integrity_status"]["merkle_check_passed"] is True
    assert turn["integrity_status"]["reason"] == ""
    assert turn["subconscious_metadata"] == []
    assert turn["thinking_state"] == []


def test_api_turn_endpoint_veto_proxy_blocks_mandatory_halt_response_and_records_intervention():
    from fastapi.testclient import TestClient

    broker = MandatoryHaltBroker()
    orchestrator = DadBotOrchestrator(
        broker,
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    app = create_api_app(orchestrator, config=service_config())

    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/veto-session/turn",
            json={"user_input": "trigger veto", "tenant_id": "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa", "timeout_seconds": 3},
            headers=auth_headers(),
        )

        replay = client.get(
            "/v1/sessions/veto-session/replay?tenant_id=t1_aaaaaaaaaaaaaaaaaaaaaaaaaa&event_type=hard_stop.intervention&limit=10",
            headers=auth_headers(),
        )

    payload = response.json()
    replay_payload = replay.json()
    assert response.status_code == 200
    assert payload["response"]["status"] == "blocked"
    assert "HARD_STOP_INTERVENTION" in payload["response"]["reply"]
    assert payload["response"]["metadata"]["service_veto_proxy"]["blocked"] is True
    assert payload["response"]["metadata"]["service_intervention"]["reason"] == "goal_alignment_mandatory_halt"
    assert any(
        event["event_type"] == "hard_stop.intervention" for event in payload["task"]["session_state"]["event_log"]
    )
    assert replay.status_code == 200
    assert replay_payload["event_count"] == 1
    assert replay_payload["events"][0]["event_type"] == "hard_stop.intervention"


def test_api_turn_endpoint_sb243_safe_mode_short_circuits_crisis_registry_before_orchestration():
    from fastapi.testclient import TestClient

    broker = CountingBroker()
    orchestrator = DadBotOrchestrator(
        broker,
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    app = create_api_app(orchestrator, config=service_config())

    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/sb243-session/turn",
            json={
                "user_input": "I want to kill myself tonight.",
                "tenant_id": "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa",
                "timeout_seconds": 3,
            },
            headers=auth_headers(),
        )
        replay = client.get(
            "/v1/sessions/sb243-session/replay?tenant_id=t1_aaaaaaaaaaaaaaaaaaaaaaaaaa&event_type=hard_stop.intervention&limit=10",
            headers=auth_headers(),
        )

    payload = response.json()
    replay_payload = replay.json()
    assert response.status_code == 200
    assert payload["task"]["status"] == "blocked"
    assert payload["response"]["status"] == "blocked"
    assert payload["response"]["metadata"]["service_intervention"]["reason"] == "sb243_crisis_registry"
    assert payload["response"]["metadata"]["service_intervention"]["safe_mode"] == "sb243"
    assert "HARD_STOP_INTERVENTION" in payload["response"]["reply"]
    assert broker.enqueue_count == 0
    assert replay.status_code == 200
    assert replay_payload["event_count"] == 1
    assert replay_payload["events"][0]["event_type"] == "hard_stop.intervention"


def test_api_turn_endpoint_sb243_safe_mode_allows_negated_reassurance_and_executes_normally():
    from fastapi.testclient import TestClient

    broker = CountingBroker()
    orchestrator = DadBotOrchestrator(
        broker,
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    app = create_api_app(orchestrator, config=service_config())

    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/sb243-negated/turn",
            json={
                "user_input": "I'm not suicidal, just exhausted and overwhelmed.",
                "tenant_id": "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa",
                "timeout_seconds": 3,
            },
            headers=auth_headers(),
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["task"]["status"] == "completed"
    assert payload["response"]["status"] == "completed"
    assert payload["response"]["reply"] == "Dad heard: I'm not suicidal, just exhausted and overwhelmed."
    assert broker.enqueue_count == 1


def test_api_turn_endpoint_sb243_uses_runtime_safety_configured_crisis_terms():
    from fastapi.testclient import TestClient

    broker = CountingBroker()
    orchestrator = DadBotOrchestrator(
        broker,
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    app = create_api_app(orchestrator, config=service_config(), runtime_bot=RuntimeSafetyConfiguredBot())

    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/sb243-runtime/turn",
            json={
                "user_input": "custom crisis keyword",
                "tenant_id": "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa",
                "timeout_seconds": 3,
            },
            headers=auth_headers(),
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["task"]["status"] == "blocked"
    assert payload["response"]["metadata"]["service_intervention"]["reason"] == "sb243_crisis_registry"
    assert payload["response"]["metadata"]["service_intervention"]["matched_term"] == "custom crisis keyword"
    assert broker.enqueue_count == 0


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
            json={"user_input": "hello replay", "tenant_id": "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"},
            headers=auth_headers(),
        )
        assert post_response.status_code == 200

        replay_response = client.get(
            "/v1/sessions/replay-session/replay?event_type=request.accepted&limit=10",
            headers=auth_headers(),
        )
        events_response = client.get(
            "/v1/sessions/replay-session/events",
            headers=auth_headers(),
        )
        wrong_tenant_events = client.get(
            "/v1/sessions/replay-session/events?tenant_id=t1_bbbbbbbbbbbbbbbbbbbbbbbbbb",
            headers=auth_headers(),
        )
        wrong_tenant_replay = client.get(
            "/v1/sessions/replay-session/replay?tenant_id=t1_bbbbbbbbbbbbbbbbbbbbbbbbbb",
            headers=auth_headers(),
        )

    payload = replay_response.json()
    assert replay_response.status_code == 200
    assert payload["session_id"] == "replay-session"
    assert payload["tenant_id"] == "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert payload["event_count"] >= 1
    assert all(event["event_type"] == "request.accepted" for event in payload["events"])
    assert events_response.status_code == 200
    assert len(events_response.json()) >= 1
    assert wrong_tenant_events.status_code == 403
    assert wrong_tenant_replay.status_code == 403


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
                "tenant_id": "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa",
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
    assert payload["tenant_id"] == "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"


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
        browser_start_response = client.post(
            "/v1/automation/browser/start", json={"restart": True}, headers=admin_headers
        )
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


def test_api_rejects_missing_secret_handshake_when_enabled(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DADBOT_SECRET_KEY", "super-secret")
    broker = ImmediateResultBroker()
    orchestrator = DadBotOrchestrator(
        broker,
        state_store=InMemoryStateStore(),
        planner_debug_factory=planner_debug_factory,
    )
    app = create_api_app(orchestrator, config=service_config())

    with TestClient(app) as client:
        rejected = client.post(
            "/v1/sessions/secret-session/chat",
            json={"user_input": "hello", "tenant_id": "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"},
            headers=auth_headers(),
        )
        accepted = client.post(
            "/v1/sessions/secret-session/chat",
            json={"user_input": "hello", "tenant_id": "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"},
            headers=auth_headers(dadbot_key="super-secret"),
        )

    assert rejected.status_code == 401
    assert rejected.json()["detail"] == "Invalid or missing X-DADBOT-KEY"
    assert accepted.status_code == 200


def test_api_pulse_websocket_requires_secret_handshake_when_enabled(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DADBOT_SECRET_KEY", "super-secret")
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
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/v1/sessions/secret-pulse/pulse/stream", headers=auth_headers()):
                pass

        with client.websocket_connect(
            "/v1/sessions/secret-pulse/pulse/stream", headers=auth_headers(dadbot_key="super-secret")
        ) as websocket:
            payload = websocket.receive_json()

    assert payload["event_type"] == "pulse.snapshot"


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
            json={"user_input": "hello", "tenant_id": "t1_bbbbbbbbbbbbbbbbbbbbbbbbbb"},
            headers=auth_headers(tenant_id="t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"),
        )

    assert response.status_code == 403


def test_api_rejects_noncanonical_principal_before_submission():
    from fastapi.testclient import TestClient

    broker = ImmediateResultBroker()
    orchestrator = DadBotOrchestrator(broker, state_store=InMemoryStateStore())
    app = create_api_app(orchestrator, config=service_config())

    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            json={"user_input": "hello", "tenant_id": "family/a"},
            headers=auth_headers(tenant_id="family-a"),
        )

    assert response.status_code == 403
    assert not broker.results


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
