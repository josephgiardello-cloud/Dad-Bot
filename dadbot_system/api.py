from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from .contracts import DEFAULT_TENANT_ID, ChatRequest, HealthResponse, ServiceConfig, normalize_channel_name
from .kernel import ControlPlane, build_control_plane
from .security import (
    AuthenticationError,
    AuthorizationError,
    ServicePrincipal,
    ServiceTokenManager,
    SlidingWindowRateLimiter,
)

try:
    from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:
    APIRouter = None
    Depends = None
    FastAPI = None
    Header = None
    HTTPException = None
    Request = None
    CORSMiddleware = None

    class WebSocket:  # pragma: no cover - fallback only when FastAPI is missing.
        pass

    class WebSocketDisconnect(Exception):
        pass


def create_api_app(
    orchestrator,
    *,
    worker_manager=None,
    config: ServiceConfig | None = None,
    control_plane: ControlPlane | None = None,
    runtime_bot=None,
):
    if FastAPI is None or HTTPException is None or CORSMiddleware is None or APIRouter is None:
        exc = ImportError("FastAPI is not installed")
        raise RuntimeError(
            "FastAPI is not installed. Install the 'service' dependencies to run the API layer."
        ) from exc

    service_config = config or ServiceConfig()
    security_config = service_config.security
    if security_config.auth_required and not security_config.token_secret:
        raise RuntimeError("DADBOT_API_TOKEN_SECRET is required when API authentication is enabled")

    _control_plane = control_plane if control_plane is not None else build_control_plane()
    token_manager = (
        ServiceTokenManager(security_config.token_secret, issuer=security_config.token_issuer)
        if security_config.token_secret
        else None
    )
    rate_limiter = SlidingWindowRateLimiter(window_seconds=60.0)

    def _version_prefix() -> str:
        normalized = "/" + str(service_config.api.version_prefix or "").strip("/")
        return "" if normalized == "/" else normalized

    def _extract_bearer_token(authorization: str | None, x_dadbot_token: str | None = None) -> str:
        header_token = str(x_dadbot_token or "").strip()
        if header_token:
            return header_token
        raw_value = str(authorization or "").strip()
        if raw_value.lower().startswith("bearer "):
            return raw_value[7:].strip()
        return ""

    def _resolve_principal(token: str) -> ServicePrincipal:
        if not security_config.auth_required:
            return ServicePrincipal(
                subject="anonymous",
                tenant_id=DEFAULT_TENANT_ID,
                scopes=frozenset({"read", "write", "admin"}),
            )
        if token_manager is None:
            raise HTTPException(status_code=503, detail="API authentication is not configured")
        if not token:
            raise HTTPException(status_code=401, detail="Missing bearer token")
        try:
            return token_manager.verify(token)
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    def _enforce_scope(principal: ServicePrincipal, scope: str) -> None:
        try:
            principal.require_scope(scope)
        except AuthorizationError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    def _enforce_rate_limit(principal: ServicePrincipal, bucket: str) -> None:
        if bucket == "admin":
            limit = security_config.max_admin_requests_per_minute
        elif bucket == "write":
            limit = security_config.max_write_requests_per_minute
        else:
            limit = security_config.max_requests_per_minute
        retry_after = rate_limiter.enforce(key=f"{principal.rate_limit_key}:{bucket}", limit=limit)
        if retry_after > 0.0:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(int(retry_after))},
            )

    def _resolve_tenant(principal: ServicePrincipal, requested_tenant_id: str | None = None) -> str:
        effective_tenant_id = str(principal.tenant_id or DEFAULT_TENANT_ID)
        requested = str(requested_tenant_id or "").strip()
        if requested and requested != effective_tenant_id:
            raise HTTPException(status_code=403, detail="tenant_id does not match the authenticated principal")
        return effective_tenant_id

    def _apply_request_identity(request_payload: dict, principal: ServicePrincipal, *, tenant_id: str) -> dict:
        resolved_payload = dict(request_payload or {})
        resolved_payload["tenant_id"] = tenant_id
        metadata = dict(resolved_payload.get("metadata") or {})
        metadata["auth"] = principal.to_metadata()
        metadata["service_policy"] = {
            "allowed_tools": sorted(principal.allowed_tools),
            "source": "api_token",
        }
        resolved_payload["metadata"] = metadata
        return resolved_payload

    async def _http_principal(
        request: Request,
        authorization: str | None = Header(default=None),
        x_dadbot_token: str | None = Header(default=None),
    ) -> ServicePrincipal:
        principal = _resolve_principal(_extract_bearer_token(authorization, x_dadbot_token))
        request.state.service_principal = principal
        return principal

    @asynccontextmanager
    async def lifespan(_app):
        async def _kernel_execute(session_entry: dict, payload: dict) -> dict:
            _ = session_entry
            _session_id = str(payload.get("_session_id") or "")
            request = ChatRequest.from_dict(payload, session_id=_session_id)
            _timeout = float(payload.get("timeout_seconds") or 60.0)
            task = orchestrator.submit_chat(request)
            task_status, response = await _wait_for_task_completion(task.task_id, timeout_seconds=_timeout)
            return {
                "task": task_status,
                "response": response,
                "execution_graph": task.execution_graph.to_dict() if task.execution_graph is not None else None,
            }

        _scheduler_task = asyncio.create_task(_control_plane.scheduler.run(_kernel_execute))
        try:
            yield
        finally:
            _control_plane.scheduler.stop()
            _scheduler_task.cancel()
            try:
                await _scheduler_task
            except asyncio.CancelledError:
                pass
            if worker_manager is not None:
                worker_manager.shutdown()
            if runtime_bot is not None and callable(getattr(runtime_bot, "shutdown", None)):
                runtime_bot.shutdown()

    app = FastAPI(title=service_config.api.title, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=service_config.api.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Dadbot-Token"],
    )

    @app.middleware("http")
    async def _security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    async def _wait_for_task_completion(task_id: str, *, timeout_seconds: float = 60.0):
        deadline = asyncio.get_running_loop().time() + max(1.0, float(timeout_seconds or 60.0))
        while asyncio.get_running_loop().time() < deadline:
            status = orchestrator.task_status(task_id)
            if status is None:
                return None, None
            response = orchestrator.response_for_task(task_id)
            if response is not None:
                return status, response
            if str(status.get("status") or "").strip().lower() == "failed":
                return status, None
            await asyncio.sleep(0.05)
        return orchestrator.task_status(task_id), orchestrator.response_for_task(task_id)

    def _resolve_runtime_bot():
        bot = runtime_bot if runtime_bot is not None else getattr(orchestrator, "runtime_bot", None)
        if bot is None:
            raise HTTPException(status_code=503, detail="Runtime bot is not available for automation controls")
        return bot

    def _gateway_capabilities() -> dict:
        version_prefix = _version_prefix()
        return {
            "channels": ["chat", "sms", "email", "discord", "slack", "webhook"],
            "ingest_path": f"{version_prefix}/channels/{{channel_name}}/sessions/{{session_id}}/ingest",
            "supports_metadata": [
                "message_id",
                "user_id",
                "user_name",
                "conversation_id",
                "received_at",
                "delivery_mode",
            ],
        }

    router = APIRouter()

    @router.get("/health")
    async def health_check(principal: ServicePrincipal = Depends(_http_principal)):
        _enforce_rate_limit(principal, "read")
        health = HealthResponse(
            status="ok",
            workers=worker_manager.worker_count if worker_manager is not None else 0,
            queue_backend=service_config.queue.backend,
            state_backend=type(orchestrator.state_store).__name__,
            service_name=service_config.telemetry.service_name,
        )
        return health.to_dict()

    @router.post("/sessions/{session_id}/chat")
    async def submit_chat(session_id: str, payload: dict, principal: ServicePrincipal = Depends(_http_principal)):
        _enforce_scope(principal, "write")
        _enforce_rate_limit(principal, "write")
        tenant_id = _resolve_tenant(principal, payload.get("tenant_id"))
        request = ChatRequest.from_dict(_apply_request_identity(payload, principal, tenant_id=tenant_id), session_id=session_id)
        if not request.user_input and not request.attachments:
            raise HTTPException(status_code=400, detail="user_input or attachments are required")
        task = orchestrator.submit_chat(request)
        return {
            "task_id": task.task_id,
            "request_id": request.request_id,
            "status": "queued",
            "execution_graph": task.execution_graph.to_dict() if task.execution_graph is not None else None,
        }

    @router.get("/gateway/channels")
    async def gateway_channels(principal: ServicePrincipal = Depends(_http_principal)):
        _enforce_rate_limit(principal, "read")
        return _gateway_capabilities()

    @router.post("/channels/{channel_name}/sessions/{session_id}/ingest")
    async def ingest_channel_message(
        channel_name: str,
        session_id: str,
        payload: dict,
        principal: ServicePrincipal = Depends(_http_principal),
    ):
        _enforce_scope(principal, "write")
        _enforce_rate_limit(principal, "write")
        tenant_id = _resolve_tenant(principal, payload.get("tenant_id"))
        request_payload = _apply_request_identity(
            {**dict(payload or {}), "channel": normalize_channel_name(channel_name)},
            principal,
            tenant_id=tenant_id,
        )
        request = ChatRequest.from_dict(request_payload, session_id=session_id)
        if not request.user_input and not request.attachments:
            raise HTTPException(status_code=400, detail="user_input or attachments are required")
        task = orchestrator.submit_chat(request)
        gateway = dict(request.metadata.get("gateway") or {})
        return {
            "task_id": task.task_id,
            "request_id": request.request_id,
            "status": "queued",
            "session_id": session_id,
            "tenant_id": request.tenant_id,
            "gateway": gateway,
            "execution_graph": task.execution_graph.to_dict() if task.execution_graph is not None else None,
        }

    @router.post("/sessions/{session_id}/turn")
    async def execute_turn(session_id: str, payload: dict, principal: ServicePrincipal = Depends(_http_principal)):
        _enforce_scope(principal, "write")
        _enforce_rate_limit(principal, "write")
        tenant_id = _resolve_tenant(principal, payload.get("tenant_id"))
        resolved_payload = _apply_request_identity(payload, principal, tenant_id=tenant_id)
        check_request = ChatRequest.from_dict(resolved_payload, session_id=session_id)
        if not check_request.user_input and not check_request.attachments:
            raise HTTPException(status_code=400, detail="user_input or attachments are required")

        timeout_seconds = float(resolved_payload.get("timeout_seconds") or 60.0)
        job_payload = {**resolved_payload, "_session_id": session_id}
        result_future = await _control_plane.submit_turn(session_id, job_payload)

        try:
            result = await asyncio.wait_for(result_future, timeout=timeout_seconds)
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Timed out waiting for turn completion") from exc

        task_status = result.get("task")
        response = result.get("response")

        if task_status is None:
            raise HTTPException(status_code=404, detail="Task not found")
        if response is None and str((task_status or {}).get("status") or "").strip().lower() != "failed":
            raise HTTPException(status_code=504, detail="Timed out waiting for turn completion")

        return result

    @router.post("/sessions/{session_id}/stream")
    async def submit_stream(session_id: str, payload: dict, principal: ServicePrincipal = Depends(_http_principal)):
        _enforce_scope(principal, "write")
        _enforce_rate_limit(principal, "write")
        tenant_id = _resolve_tenant(principal, payload.get("tenant_id"))
        request = ChatRequest.from_dict(_apply_request_identity(payload, principal, tenant_id=tenant_id), session_id=session_id)
        if not request.user_input and not request.attachments:
            raise HTTPException(status_code=400, detail="user_input or attachments are required")

        task = orchestrator.submit_chat(request)
        return {
            "task_id": task.task_id,
            "request_id": request.request_id,
            "status": "queued",
            "stream": {
                "type": "websocket",
                "path": f"{_version_prefix()}/sessions/{session_id}/events/stream",
                "tenant_id": request.tenant_id or DEFAULT_TENANT_ID,
            },
            "execution_graph": task.execution_graph.to_dict() if task.execution_graph is not None else None,
        }

    @router.get("/tasks/{task_id}")
    async def get_task_status(task_id: str, principal: ServicePrincipal = Depends(_http_principal)):
        _enforce_scope(principal, "read")
        _enforce_rate_limit(principal, "read")
        status = orchestrator.task_status(task_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Task not found")
        _resolve_tenant(principal, status.get("tenant_id"))
        response = orchestrator.response_for_task(task_id)
        return {"task": status, "response": response}

    @router.get("/automation/status")
    async def automation_status(principal: ServicePrincipal = Depends(_http_principal)):
        _enforce_scope(principal, "admin")
        _enforce_rate_limit(principal, "admin")
        bot = _resolve_runtime_bot()
        maintenance_snapshot = {}
        if callable(getattr(bot, "maintenance_snapshot", None)):
            maintenance_snapshot = dict(bot.maintenance_snapshot() or {})
        local_mcp = {}
        if callable(getattr(bot, "local_mcp_status", None)):
            local_mcp = dict(bot.local_mcp_status() or {})
        memory_store = dict(getattr(bot, "MEMORY_STORE", {}) or {})
        return {
            "heartbeat_interval_seconds": int(getattr(bot, "_proactive_heartbeat_interval_seconds", 0) or 0),
            "maintenance": maintenance_snapshot,
            "continuous_learning": {
                "last_run_at": memory_store.get("last_continuous_learning_at"),
                "cycle_count": int(memory_store.get("learning_cycle_count", 0) or 0),
                "last_turn": int(memory_store.get("last_learning_turn", 0) or 0),
            },
            "local_mcp": local_mcp,
            "gateway": _gateway_capabilities(),
        }

    @router.post("/automation/heartbeat")
    async def trigger_heartbeat(payload: dict | None = None, principal: ServicePrincipal = Depends(_http_principal)):
        _enforce_scope(principal, "admin")
        _enforce_rate_limit(principal, "admin")
        bot = _resolve_runtime_bot()
        maintenance = getattr(bot, "maintenance_scheduler", None)
        if maintenance is None or not callable(getattr(maintenance, "run_proactive_heartbeat", None)):
            raise HTTPException(status_code=503, detail="Proactive heartbeat manager is not available")
        options = dict(payload or {})
        result = maintenance.run_proactive_heartbeat(force=bool(options.get("force", True)))
        return {"status": "completed", "result": result}

    @router.post("/automation/self-improvement")
    async def trigger_self_improvement(
        payload: dict | None = None,
        principal: ServicePrincipal = Depends(_http_principal),
    ):
        _enforce_scope(principal, "admin")
        _enforce_rate_limit(principal, "admin")
        bot = _resolve_runtime_bot()
        long_term = getattr(bot, "long_term_signals", None)
        if long_term is None:
            raise HTTPException(status_code=503, detail="Continuous learning manager is not available")
        options = dict(payload or {})
        background = bool(options.get("background", True))
        force = bool(options.get("force", True))
        if background and not force and callable(getattr(long_term, "schedule_continuous_learning", None)):
            task = long_term.schedule_continuous_learning()
            task_id = str(getattr(task, "dadbot_task_id", "") or "") if task is not None else ""
            return {"status": "queued" if task_id else "skipped", "task_id": task_id}
        perform_cycle = getattr(long_term, "perform_continuous_learning_cycle", None)
        if not callable(perform_cycle):
            raise HTTPException(status_code=503, detail="Continuous learning execution is not available")
        if background and callable(getattr(bot, "submit_background_task", None)):
            task = bot.submit_background_task(
                perform_cycle,
                task_kind="continuous-learning",
                metadata={"api_surface": "service", "forced": force},
            )
            return {"status": "queued", "task_id": str(getattr(task, "dadbot_task_id", "") or "")}
        return {"status": "completed", "result": perform_cycle()}

    @router.get("/automation/browser/status")
    async def browser_status(principal: ServicePrincipal = Depends(_http_principal)):
        _enforce_scope(principal, "admin")
        _enforce_rate_limit(principal, "admin")
        bot = _resolve_runtime_bot()
        if not callable(getattr(bot, "local_mcp_status", None)):
            raise HTTPException(status_code=503, detail="Local MCP controls are not available")
        return dict(bot.local_mcp_status() or {})

    @router.post("/automation/browser/start")
    async def browser_start(payload: dict | None = None, principal: ServicePrincipal = Depends(_http_principal)):
        _enforce_scope(principal, "admin")
        _enforce_rate_limit(principal, "admin")
        bot = _resolve_runtime_bot()
        if not callable(getattr(bot, "start_local_mcp_server_process", None)):
            raise HTTPException(status_code=503, detail="Local MCP controls are not available")
        options = dict(payload or {})
        return dict(bot.start_local_mcp_server_process(restart=bool(options.get("restart", False))) or {})

    @router.post("/automation/browser/stop")
    async def browser_stop(principal: ServicePrincipal = Depends(_http_principal)):
        _enforce_scope(principal, "admin")
        _enforce_rate_limit(principal, "admin")
        bot = _resolve_runtime_bot()
        if not callable(getattr(bot, "stop_local_mcp_server_process", None)):
            raise HTTPException(status_code=503, detail="Local MCP controls are not available")
        return dict(bot.stop_local_mcp_server_process() or {})

    @router.get("/sessions/{session_id}/events")
    async def get_session_events(
        session_id: str,
        tenant_id: str = DEFAULT_TENANT_ID,
        principal: ServicePrincipal = Depends(_http_principal),
    ):
        _enforce_scope(principal, "read")
        _enforce_rate_limit(principal, "read")
        effective_tenant_id = _resolve_tenant(principal, tenant_id)
        return orchestrator.session_events(session_id, tenant_id=effective_tenant_id)

    @router.get("/sessions/{session_id}/replay")
    async def replay_session_events(
        session_id: str,
        tenant_id: str = DEFAULT_TENANT_ID,
        event_type: str = "",
        since_event_id: str = "",
        limit: int = 200,
        principal: ServicePrincipal = Depends(_http_principal),
    ):
        _enforce_scope(principal, "read")
        _enforce_rate_limit(principal, "read")
        effective_tenant_id = _resolve_tenant(principal, tenant_id)
        events = list(orchestrator.session_events(session_id, tenant_id=effective_tenant_id))

        normalized_event_type = str(event_type or "").strip().lower()
        if normalized_event_type:
            events = [
                event for event in events if str(event.get("event_type") or "").strip().lower() == normalized_event_type
            ]

        normalized_since_event_id = str(since_event_id or "").strip()
        if normalized_since_event_id:
            replay_from_index = 0
            for index, event in enumerate(events):
                if str(event.get("event_id") or "") == normalized_since_event_id:
                    replay_from_index = index + 1
                    break
            events = events[replay_from_index:]

        safe_limit = max(1, min(int(limit or 200), 2000))
        if len(events) > safe_limit:
            events = events[-safe_limit:]

        return {
            "session_id": session_id,
            "tenant_id": effective_tenant_id,
            "event_count": len(events),
            "events": events,
        }

    @router.websocket("/sessions/{session_id}/events/stream")
    async def stream_session_events(websocket: WebSocket, session_id: str, tenant_id: str = ""):
        try:
            principal = _resolve_principal(
                _extract_bearer_token(
                    websocket.headers.get("authorization"),
                    websocket.query_params.get("access_token"),
                )
            )
            _enforce_scope(principal, "read")
            _enforce_rate_limit(principal, "read")
            effective_tenant_id = _resolve_tenant(principal, tenant_id)
        except HTTPException:
            await websocket.close(code=4401)
            return

        await websocket.accept()
        event_bus = getattr(orchestrator, "event_bus", None)
        subscriber = event_bus.subscribe() if event_bus is not None and hasattr(event_bus, "subscribe") else None
        try:
            for event in orchestrator.session_events(session_id, tenant_id=effective_tenant_id):
                await websocket.send_json(event)

            if subscriber is None:
                while True:
                    await asyncio.sleep(15)
                    await websocket.send_json(
                        {"event_type": "heartbeat", "session_id": session_id, "tenant_id": effective_tenant_id}
                    )

            while True:
                event = await asyncio.to_thread(subscriber.get, True, 15.0)
                payload = event.to_dict() if hasattr(event, "to_dict") else dict(event)
                if payload.get("session_id") != session_id or payload.get("tenant_id") != effective_tenant_id:
                    continue
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            return
        finally:
            if subscriber is not None and event_bus is not None and hasattr(event_bus, "unsubscribe"):
                event_bus.unsubscribe(subscriber)

    version_prefix = _version_prefix()
    if version_prefix:
        app.include_router(router, prefix=version_prefix)
        if service_config.api.legacy_routes_enabled:
            app.include_router(router)
    else:
        app.include_router(router)

    return app
