from __future__ import annotations

# ruff: noqa: B008, C901, PLR0913, PLR0915
import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request as URLRequest
from urllib.request import urlopen

from dadbot.api_models import build_pulse_envelope, build_turn_envelope
from dadbot.core.execution_contract import ExecutionMode, SovereignContext
from dadbot.core.execution_ledger import IntegrityBreachError
from dadbot.managers.tts import TTSManager
from dadbot.runtime.env_loader import is_production_environment

from .contracts import (
    DEFAULT_TENANT_ID,
    ChatRequest,
    EventEnvelope,
    EventType,
    HealthResponse,
    ServiceConfig,
    normalize_channel_name,
)
from .execution_identity import require_tenant_id
from .kernel import ControlPlane, build_control_plane
from .mcp_security import (
    MCP_CONNECT_SCOPE,
    MCP_TOKEN_MAX_TTL_SECONDS,
    MCP_TOOL_BASE_SCOPES,
    approved_mcp_tools,
    mcp_resource_url,
    mcp_tool_scope,
)
from .runtime_signals import configure_tracing, is_trace_provider_available, shutdown_tracing
from .security import (
    AuthenticationError,
    AuthorizationError,
    RedisWebSocketTicketStore,
    ServicePrincipal,
    ServiceTokenManager,
    SlidingWindowRateLimiter,
    TicketStoreUnavailableError,
)
from .state import TenantOwnedStateStoreView

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from fastapi import Request
else:
    try:
        from fastapi import Request
    except ImportError:
        Request = None

try:
    from fastapi import (  # type: ignore[reportMissingImports]
        APIRouter,
        Depends,
        FastAPI,
        Header,
        HTTPException,
        WebSocket,
        WebSocketDisconnect,
    )
    from fastapi.middleware.cors import CORSMiddleware  # type: ignore[reportMissingImports]
    from fastapi.responses import JSONResponse  # type: ignore[reportMissingImports]
except ImportError:
    APIRouter = cast(Any, None)
    Depends = cast(Any, None)
    FastAPI = cast(Any, None)
    Header = cast(Any, None)
    HTTPException = cast(Any, None)
    CORSMiddleware = cast(Any, None)
    JSONResponse = cast(Any, None)

    class WebSocket:  # pragma: no cover - fallback only when FastAPI is missing.
        headers: dict[str, str]
        query_params: dict[str, str]

        async def close(self, *, code: int = 1000) -> None:
            _ = code

        async def accept(self) -> None:
            return

        async def send_json(self, payload: dict[str, Any]) -> None:
            _ = payload

    class WebSocketDisconnectError(Exception):
        pass

    WebSocketDisconnect = WebSocketDisconnectError


def _probe_model_dependency(endpoint: str, model_name: str, timeout_seconds: float) -> tuple[str, str | None]:
    if not model_name:
        return "degraded", "ModelConfigurationError"
    base_url = endpoint.rstrip("/")
    tags_url = base_url if base_url.endswith("/api/tags") else f"{base_url}/api/tags"
    request = URLRequest(tags_url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read())
    models = payload.get("models") if isinstance(payload, dict) else None
    names = {
        str(item.get("name") or "").strip()
        for item in models or []
        if isinstance(item, dict)
    }
    if model_name in names:
        return "ok", None
    return "degraded", "ModelNotFoundError"


def _probe_vector_dependency(vector_url: str, _model_name: str, timeout_seconds: float) -> tuple[str, str | None]:
    if not vector_url:
        return "degraded", "ConfigurationError"
    parsed = urlparse(vector_url)
    if parsed.scheme in {"postgres", "postgresql"}:
        import psycopg

        with psycopg.connect(
            vector_url,
            connect_timeout=max(1, int(timeout_seconds + 0.999)),
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
        return "ok", None
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "unavailable", "ConfigurationError"

    request = URLRequest(vector_url, method="HEAD")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status_code = int(response.status)
    except HTTPError as exc:
        if exc.code in {405, 501}:
            request = URLRequest(vector_url, headers={"Range": "bytes=0-0"})
            with urlopen(request, timeout=timeout_seconds) as response:
                status_code = int(response.status)
        else:
            status_code = int(exc.code)
    if status_code < 400:
        return "ok", None
    if status_code < 500:
        return "degraded", "HTTPError"
    return "unavailable", "HTTPError"


def _probe_memory_dependency(runtime_bot: Any, _timeout_seconds: float) -> tuple[str, str | None]:
    context_service = getattr(runtime_bot, "context_service", None)
    semantic_index = getattr(context_service, "semantic_index", None)
    with_connection = getattr(semantic_index, "with_connection", None)
    if not callable(with_connection):
        return "unavailable", "MemoryBackendUnavailableError"

    def check_connection(connection: Any) -> None:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()

    with_connection(check_connection)
    return "ok", None


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
    probe_names = {"model", "vector", "memory"}
    configured_probes = {
        str(name).strip().lower() for name in service_config.api.readiness_probes if str(name).strip()
    }
    required_probes = {
        str(name).strip().lower()
        for name in service_config.api.readiness_required_probes
        if str(name).strip()
    }
    unsupported_probes = (configured_probes | required_probes) - probe_names
    if unsupported_probes:
        raise ValueError(f"Unsupported readiness probes: {', '.join(sorted(unsupported_probes))}")
    configured_probes.update(required_probes)
    probe_cache: dict[str, tuple[float, dict[str, Any]]] = {}
    probe_locks = {name: asyncio.Lock() for name in probe_names}
    from .delegation import (
        AuthorityUnavailableError,
        SQLiteDelegationStore,
        require_live_effect_authority,
    )
    from .delegation import (
        runtime_authority_scope as bind_runtime_authority,
    )

    delegation_store = None
    if not security_config.auth_required and is_production_environment():
        raise RuntimeError("API authentication cannot be disabled in production")
    veto_proxy_enabled = str(os.environ.get("DADBOT_VETO_PROXY_ENABLED", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    try:
        veto_drift_threshold = float(os.environ.get("DADBOT_VETO_DRIFT_THRESHOLD", "0.98"))
    except ValueError:
        veto_drift_threshold = 0.98
    veto_drift_threshold = max(0.0, min(veto_drift_threshold, 1.0))
    sb243_safe_mode_enabled = str(os.environ.get("DADBOT_SB243_SAFE_MODE_ENABLED", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    sb243_crisis_terms_raw = str(
        os.environ.get(
            "DADBOT_SB243_CRISIS_TERMS",
            "kill myself,killing myself,end my life,want to die,don't want to live,do not want to live,"
            "hurt myself,harm myself,suicidal,suicide,self harm,self-harm,overdose",
        )
    )
    sb243_crisis_terms = [term.strip().lower() for term in sb243_crisis_terms_raw.split(",") if term.strip()]
    dadbot_secret_key = str(os.environ.get("DADBOT_SECRET_KEY") or "").strip()
    sb243_negation_patterns: tuple[str, ...] = (
        r"\bnot suicidal\b",
        r"\bi am not suicidal\b",
        r"\bi'm not suicidal\b",
        r"\bnot going to hurt myself\b",
        r"\bnot going to harm myself\b",
        r"\bi won't hurt myself\b",
        r"\bi wont hurt myself\b",
        r"\bdon't want to hurt myself\b",
        r"\bdo not want to hurt myself\b",
    )
    if security_config.auth_required and not security_config.token_secret:
        raise RuntimeError("DADBOT_API_TOKEN_SECRET is required when API authentication is enabled")
    if security_config.auth_required:
        require_authenticated_execution = getattr(orchestrator, "require_authenticated_execution", None)
        if not callable(require_authenticated_execution):
            raise RuntimeError("Authenticated service requires an authority-enforcing orchestrator")
        if not security_config.authority_sqlite_path:
            raise RuntimeError("Authenticated service requires a configured durable authority store")
        if security_config.authority_sqlite_path:
            from .state import SQLiteStateStore

            authority_path = Path(security_config.authority_sqlite_path).resolve()
            state_store = getattr(orchestrator, "state_store", None)
            configured_state_path = service_config.persistence.sqlite_path
            actual_state_path = state_store.path if isinstance(state_store, SQLiteStateStore) else ""
            state_paths = (configured_state_path, actual_state_path)
            if any(
                path
                and (
                    authority_path == Path(path).resolve()
                    or (authority_path.exists() and Path(path).exists() and authority_path.samefile(path))
                )
                for path in state_paths
            ):
                raise RuntimeError("Authority and service state require separate SQLite database files")
            delegation_store = SQLiteDelegationStore(
                security_config.authority_sqlite_path,
                security_config.token_secret,
                issuer=security_config.token_issuer,
            )
            delegation_store.check_available()
            orchestrator.configure_delegation_store(delegation_store)
        require_authenticated_execution()

    _control_plane = control_plane if control_plane is not None else build_control_plane()
    token_manager = (
        ServiceTokenManager(security_config.token_secret, issuer=security_config.token_issuer)
        if security_config.token_secret
        else None
    )
    rate_limiter = SlidingWindowRateLimiter(window_seconds=60.0)
    ticket_namespace = hashlib.sha256(
        json.dumps([security_config.token_issuer, security_config.token_secret, dadbot_secret_key]).encode("utf-8")
    ).hexdigest()
    websocket_tickets = (
        RedisWebSocketTicketStore(service_config.persistence.redis_url, namespace=ticket_namespace)
        if service_config.persistence.redis_url
        else None
    )

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
            principal = token_manager.verify(token)
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        if principal.audience not in {"", "dadbot-api"}:
            raise HTTPException(status_code=401, detail="Bearer token audience is not valid for this API")
        return principal

    def _enforce_scope(principal: ServicePrincipal, scope: str) -> None:
        try:
            principal.require_scope(scope)
            if security_config.auth_required:
                if delegation_store is None:
                    raise HTTPException(status_code=503, detail="Delegation authority is unavailable")
                delegation_store.require_membership(
                    issuer=principal.issuer,
                    subject=principal.subject,
                    tenant_id=principal.tenant_id,
                    scope=scope,
                )
        except AuthorizationError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except AuthorityUnavailableError as exc:
            logger.error("Delegation authority unavailable during request authorization")
            raise HTTPException(status_code=503, detail="Delegation authority unavailable") from exc

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
        try:
            require_tenant_id(
                effective_tenant_id,
                issuer=principal.issuer or ("auth-off" if principal.subject == "anonymous" else "unbound"),
                subject=principal.subject,
                acceptance_path="api._resolve_tenant",
            )
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="Invalid principal tenant identity") from exc
        requested = str(requested_tenant_id or "").strip()
        if requested and requested != effective_tenant_id:
            raise HTTPException(status_code=403, detail="tenant_id does not match the authenticated principal")
        return effective_tenant_id

    def _is_valid_secret_key(provided_secret: str) -> bool:
        if not dadbot_secret_key:
            return True
        candidate = str(provided_secret or "").strip()
        if not candidate:
            return False
        return hmac.compare_digest(candidate, dadbot_secret_key)

    async def _websocket_principal(
        websocket: WebSocket, *, path: str, scope: str, tenant_id: str = ""
    ) -> tuple[ServicePrincipal, str]:
        if {"access_token", "dadbot_key", "x_dadbot_key"}.intersection(websocket.query_params):
            raise HTTPException(status_code=401, detail="Use a WebSocket ticket, not query credentials")
        if "ticket" in websocket.query_params:
            if websocket_tickets is None:
                raise HTTPException(status_code=503, detail="WebSocket tickets require DADBOT_REDIS_URL")
            try:
                principal = await asyncio.to_thread(
                    websocket_tickets.consume, websocket.query_params["ticket"], path=path
                )
            except AuthenticationError as exc:
                raise HTTPException(status_code=401, detail=str(exc)) from exc
            except TicketStoreUnavailableError as exc:
                logger.error("WebSocket ticket store unavailable during handshake")
                raise HTTPException(status_code=503, detail="WebSocket ticket store unavailable") from exc
        else:
            if not _is_valid_secret_key(websocket.headers.get("x-dadbot-key", "")):
                raise HTTPException(status_code=401, detail="Invalid or missing X-DADBOT-KEY")
            principal = _resolve_principal(
                _extract_bearer_token(websocket.headers.get("authorization"), websocket.headers.get("x-dadbot-token"))
            )
        _enforce_scope(principal, scope)
        _enforce_rate_limit(principal, scope)
        return principal, _resolve_tenant(principal, tenant_id)

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

    def _submit_chat_request(request: ChatRequest, http_request: Request):
        if not (security_config.auth_required and security_config.authority_sqlite_path):
            return orchestrator.submit_chat(request)
        from .delegation import AuthorityUnavailableError

        token = _extract_bearer_token(
            http_request.headers.get("authorization"), http_request.headers.get("x-dadbot-token")
        )
        try:
            return orchestrator.submit_chat(request, bearer_token=token)
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except AuthorizationError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except AuthorityUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def _websocket_tenant_dependency(path: str, scope: str):
        async def bind(
            websocket: WebSocket,
            tenant_id: str = "",
            session_id: str = "",
        ):
            try:
                principal, effective_tenant_id = await _websocket_principal(
                    websocket,
                    path=path.format(session_id=session_id),
                    scope=scope,
                    tenant_id=tenant_id,
                )
            except HTTPException as exc:
                from fastapi import WebSocketException

                raise WebSocketException(
                    code=1013 if exc.status_code == 503 else 4401
                ) from exc

            from dadbot_system.tenant_context import require_request_context

            websocket.state.service_principal = principal
            with require_live_effect_authority(required=security_config.auth_required):
                with _request_tenant_context_scope(effective_tenant_id):
                    websocket.state.tenant_context = require_request_context()
                    yield effective_tenant_id

        return bind


    def _request_tenant_context_scope(tenant_id: str):
        import os

        from dadbot.runtime.env_loader import runtime_environment
        from dadbot_system.tenant_context import (
            TenantContext,
            local_default_context_scope,
            tenant_context_scope,
        )

        if tenant_id == DEFAULT_TENANT_ID:
            enabled = (
                runtime_environment() != "production"
                and os.environ.get("DADBOT_ALLOW_LEGACY_DEFAULT_TENANT", "")
                .strip()
                .lower()
                in {"1", "true", "yes", "on"}
            )
            if not enabled:
                raise HTTPException(
                    status_code=403,
                    detail="Legacy default tenant compatibility is disabled",
                )
            return local_default_context_scope(enabled=True)

        try:
            context = TenantContext(tenant_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=403,
                detail="Tenant context requires an issued canonical tenant ID",
            ) from exc
        return tenant_context_scope(context)


    async def _http_principal(
        request: Request,
        authorization: str | None = Header(default=None),
        x_dadbot_token: str | None = Header(default=None),
    ):
        if request.url.path.rstrip("/") in {
            "/health",
            "/ready",
            f"{_version_prefix()}/health",
            f"{_version_prefix()}/ready",
        }:
            principal = ServicePrincipal(
                subject="health-check",
                tenant_id=DEFAULT_TENANT_ID,
                scopes=frozenset({"read"}),
            )
            request.state.service_principal = principal
            yield principal
            return

        principal = _resolve_principal(
            _extract_bearer_token(authorization, x_dadbot_token)
        )
        request.state.service_principal = principal

        body = await request.body()
        payload = json.loads(body) if body else {}
        requested_tenant = (
            payload.get("tenant_id", request.query_params.get("tenant_id"))
            if isinstance(payload, dict)
            else request.query_params.get("tenant_id")
        )
        tenant_id = _resolve_tenant(principal, requested_tenant)
        from dadbot_system.tenant_context import require_request_context

        with require_live_effect_authority(required=security_config.auth_required):
            with _request_tenant_context_scope(tenant_id):
                request.state.tenant_context = require_request_context()
                yield principal

    @asynccontextmanager
    async def lifespan(_app):
        if service_config.telemetry.otel_enabled:
            configure_tracing(service_config.telemetry)

        recovered_tasks = orchestrator.recover_pending_tasks()
        if recovered_tasks:
            logger.warning("Requeued %d incomplete tasks from durable storage", recovered_tasks)

        async def _kernel_execute(session_entry: dict, payload: dict) -> dict:
            _ = session_entry
            _session_id = str(payload.get("_session_id") or "")
            request = ChatRequest.from_dict(payload, session_id=_session_id)
            _timeout = float(payload.get("timeout_seconds") or 60.0)
            task_id = str(payload.get("_submitted_task_id") or "")
            if task_id:
                task_status = orchestrator.task_status(
                    task_id,
                    tenant_id=request.tenant_id,
                )
                if (
                    not task_status
                    or task_status.get("tenant_id") != request.tenant_id
                    or task_status.get("session_id") != request.session_id
                    or task_status.get("request_id") != request.request_id
                ):
                    raise RuntimeError("Scheduled task identity mismatch")
                execution_graph = task_status.get("execution_graph")
            else:
                task = orchestrator.submit_chat(request)
                task_id = task.task_id
                execution_graph = task.execution_graph.to_dict() if task.execution_graph is not None else None
            task_status, response = await _wait_for_task_completion(
                task_id,
                tenant_id=request.tenant_id,
                timeout_seconds=_timeout,
            )
            return {
                "task": task_status,
                "response": response,
                "execution_graph": execution_graph,
            }

        _scheduler_task = _control_plane.task_manager.register(
            name="scheduler.run",
            coro=_control_plane.scheduler.run(_kernel_execute),
        )
        try:
            yield
        finally:
            _control_plane.scheduler.stop()
            if not _scheduler_task.done():
                _scheduler_task.cancel()
            await _control_plane.task_manager.shutdown(cancel_pending=True)
            if worker_manager is not None:
                worker_manager.shutdown()
            if runtime_bot is not None and callable(getattr(runtime_bot, "shutdown", None)):
                runtime_bot.shutdown()
            if websocket_tickets is not None:
                await asyncio.to_thread(websocket_tickets.close)
            if service_config.telemetry.otel_enabled:
                shutdown_tracing()

    app = FastAPI(title=service_config.api.title, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=service_config.api.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Dadbot-Token", "X-DADBOT-KEY"],
    )

    @app.middleware("http")
    async def _security_headers(request: Any, call_next):
        is_health_check = request.url.path.rstrip("/") in {"/health", f"{_version_prefix()}/health"}
        is_preflight = (
            request.method == "OPTIONS"
            and "origin" in request.headers
            and "access-control-request-method" in request.headers
        )
        if dadbot_secret_key and not is_preflight and not is_health_check:
            provided_secret = request.headers.get("x-dadbot-key")
            if not _is_valid_secret_key(str(provided_secret or "")):
                return JSONResponse(status_code=401, content={"detail": "Invalid or missing X-DADBOT-KEY"})
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    async def _wait_for_task_completion(
        task_id: str,
        *,
        tenant_id: str,
        timeout_seconds: float = 60.0,
    ):
        from .worker import AUTHORITY_UNAVAILABLE

        deadline = asyncio.get_running_loop().time() + max(1.0, float(timeout_seconds or 60.0))
        while asyncio.get_running_loop().time() < deadline:
            status = orchestrator.task_status(task_id, tenant_id=tenant_id)
            if status is None:
                return None, None
            if status.get("error") == AUTHORITY_UNAVAILABLE:
                raise HTTPException(status_code=503, detail=AUTHORITY_UNAVAILABLE)
            response = orchestrator.response_for_task(task_id, tenant_id=tenant_id)
            if response is not None:
                return status, response
            if str(status.get("status") or "").strip().lower() == "failed":
                return status, None
            await asyncio.sleep(0.05)
        return (
            orchestrator.task_status(task_id, tenant_id=tenant_id),
            orchestrator.response_for_task(task_id, tenant_id=tenant_id),
        )

    def _resolve_runtime_bot():
        bot = runtime_bot if runtime_bot is not None else getattr(orchestrator, "runtime_bot", None)
        if bot is None:
            raise HTTPException(status_code=503, detail="Runtime bot is not available for automation controls")
        return bot

    @contextmanager
    def _runtime_authority(principal: ServicePrincipal, scope: str):
        bot = _resolve_runtime_bot()
        if not security_config.auth_required:
            yield bot
            return
        tenant_id = _resolve_tenant(principal)
        bot_tenant_id = str(getattr(getattr(bot, "config", None), "tenant_id", "") or "")
        if bot_tenant_id != tenant_id:
            raise HTTPException(
                status_code=403,
                detail="Runtime bot is not bound to the authenticated tenant",
            )
        if delegation_store is None:
            raise HTTPException(status_code=503, detail="Delegation authority is unavailable")
        try:
            with bind_runtime_authority(
                delegation_store,
                principal,
                scope=scope,
                tenant_id=tenant_id,
            ):
                yield bot
        except AuthorizationError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except AuthorityUnavailableError as exc:
            logger.error("Delegation authority unavailable during runtime operation")
            raise HTTPException(status_code=503, detail="Delegation authority unavailable") from exc

    def _with_runtime_authority(scope: str):
        def decorate(handler):
            @wraps(handler)
            async def wrapped(*args, **kwargs):
                principal = kwargs.get("principal")
                if not isinstance(principal, ServicePrincipal):
                    raise RuntimeError("Runtime operation requires an authenticated principal")
                _enforce_scope(principal, scope)
                with _runtime_authority(principal, scope):
                    return await handler(*args, **kwargs)

            return wrapped

        return decorate

    def _extract_spatial_text(payload: dict[str, Any]) -> str:
        for key in ("text", "transcript", "message", "prompt", "user_input"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        user = payload.get("user")
        if isinstance(user, dict):
            for key in ("text", "transcript", "message"):
                value = user.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ""

    def _encode_spatial_audio(bot: Any, reply_text: str) -> tuple[str, str, str]:
        if not reply_text.strip():
            return "", "", ""
        manager = getattr(bot, "tts_manager", None)
        if manager is None or not callable(getattr(manager, "get_tts_audio", None)):
            manager = TTSManager(bot)
        try:
            audio_bytes, audio_error, audio_mime = manager.get_tts_audio(reply_text)
        except Exception as exc:
            return "", f"tts_unavailable: {exc}", ""
        if audio_error or not audio_bytes:
            return "", str(audio_error or ""), ""
        encoded = base64.b64encode(audio_bytes).decode("ascii")
        return encoded, "", str(audio_mime or "audio/wav")

    def _publish_service_intervention_event(
        *,
        session_id: str,
        tenant_id: str,
        task_id: str,
        request_id: str,
        intervention: dict,
    ) -> dict:
        envelope = EventEnvelope(
            session_id=session_id,
            event_type=EventType.HARD_STOP_INTERVENTION,
            tenant_id=tenant_id,
            payload={
                "task_id": task_id,
                "request_id": request_id,
                "intervention": dict(intervention or {}),
            },
        )
        scoped_store = TenantOwnedStateStoreView(
            orchestrator.state_store, tenant_id, orchestrator.task_store
        )
        serialized = envelope.to_dict()
        scoped_store.append_event(session_id, serialized)
        event_bus = getattr(orchestrator, "event_bus", None)
        if event_bus is not None and callable(getattr(event_bus, "publish", None)):
            event_bus.publish(envelope)
        return serialized

    def _build_blocked_turn_payload(
        *,
        task_payload: dict,
        response_payload: dict,
        intervention_payload: dict,
        blocked_reply: str,
        task_error: str = "",
        enforce_mandatory_halt: bool = False,
    ) -> tuple[dict, dict, dict]:
        task_data = dict(task_payload or {})
        response_data = dict(response_payload or {})

        task_id = str(task_data.get("task_id") or "")
        session_id = str(task_data.get("session_id") or response_data.get("session_id") or "")
        request_id = str(response_data.get("request_id") or task_data.get("request_id") or "")
        task_tenant_id = str(task_data.get("tenant_id") or "").strip()
        response_tenant_id = str(response_data.get("tenant_id") or "").strip()
        if task_tenant_id and response_tenant_id and task_tenant_id != response_tenant_id:
            raise HTTPException(status_code=403, detail="Tenant context mismatch")
        tenant_id = task_tenant_id or response_tenant_id
        if not tenant_id:
            raise HTTPException(status_code=403, detail="Tenant context is required")
        task_data["tenant_id"] = tenant_id
        response_data["tenant_id"] = tenant_id

        intervention_event = _publish_service_intervention_event(
            session_id=session_id,
            tenant_id=tenant_id,
            task_id=task_id,
            request_id=request_id,
            intervention=intervention_payload,
        )

        session_state = dict(task_data.get("session_state") or {})
        if enforce_mandatory_halt:
            session_state["goal_alignment_mandatory_halt"] = True

        intervention_log = list(session_state.get("service_interventions") or [])
        intervention_log.append(intervention_event)
        session_state["service_interventions"] = intervention_log

        event_log = list(session_state.get("event_log") or [])
        event_log.append(intervention_event)
        session_state["event_log"] = event_log

        task_data["session_state"] = session_state
        task_data["status"] = "blocked"
        if task_error:
            task_data["error"] = task_error

        response_metadata = dict(response_data.get("metadata") or {})
        response_metadata["service_veto_proxy"] = {
            "blocked": True,
            "event_id": intervention_event.get("event_id"),
            "reason": intervention_payload.get("reason"),
            "severity": intervention_payload.get("severity"),
        }
        response_metadata["service_intervention"] = intervention_payload

        response_data = {
            **response_data,
            "session_id": session_id,
            "request_id": request_id,
            "tenant_id": tenant_id,
            "reply": blocked_reply,
            "should_end": True,
            "status": "blocked",
            "metadata": response_metadata,
        }

        if task_id:
            orchestrator.task_store.save_task(tenant_id, task_id, task_data)
            orchestrator.task_store.save_task_result(
                tenant_id,
                task_id,
                task_data,
                response_data,
            )

        return task_data, response_data, intervention_event

    def _evaluate_veto_proxy(*, task_status: dict | None, response: dict | None) -> dict | None:
        if not veto_proxy_enabled:
            return None

        task_payload = dict(task_status or {})
        response_payload = dict(response or {})
        session_state = dict(task_payload.get("session_state") or {})
        reflection = dict(session_state.get("last_reflection_summary") or {})

        if bool(session_state.get("goal_alignment_mandatory_halt", False)):
            return {
                "reason": "goal_alignment_mandatory_halt",
                "severity": "critical",
                "recommended_action": "HARD_STOP_INTERVENTION",
            }

        risk_level = str(reflection.get("current_risk_level") or "").strip().lower()
        if risk_level in {"critical", "severe"}:
            return {
                "reason": "reflection_risk_level",
                "severity": risk_level,
                "recommended_action": "HARD_STOP_INTERVENTION",
                "risk_level": risk_level,
            }

        drift_probability = float(reflection.get("predicted_drift_probability") or 0.0)
        if drift_probability >= veto_drift_threshold:
            return {
                "reason": "predicted_drift_probability",
                "severity": "high",
                "recommended_action": "HARD_STOP_INTERVENTION",
                "drift_probability": drift_probability,
                "threshold": veto_drift_threshold,
            }

        response_metadata = dict(response_payload.get("metadata") or {})
        if bool(response_metadata.get("service_hard_stop_required", False)):
            return {
                "reason": "response_metadata.service_hard_stop_required",
                "severity": "critical",
                "recommended_action": "HARD_STOP_INTERVENTION",
            }
        return None

    def _apply_veto_proxy_to_turn_payload(result: dict) -> dict:
        task_status = dict(result.get("task") or {})
        response = dict(result.get("response") or {})
        intervention = _evaluate_veto_proxy(task_status=task_status, response=response)
        if intervention is None or not task_status or not response:
            return result

        intervention_payload = {
            "type": "HARD_STOP_INTERVENTION",
            "source": "dadbot_system.api.veto_proxy",
            **dict(intervention),
        }
        task_status, response, _ = _build_blocked_turn_payload(
            task_payload=task_status,
            response_payload=response,
            intervention_payload=intervention_payload,
            blocked_reply="HARD_STOP_INTERVENTION: output withheld by service veto proxy.",
        )

        return {
            **result,
            "task": task_status,
            "response": response,
            "interventions": [*(result.get("interventions") or []), intervention_payload],
        }

    def _detect_sb243_crisis_signal(user_input: str) -> str:
        def _runtime_crisis_terms() -> list[str]:
            bot = runtime_bot if runtime_bot is not None else getattr(orchestrator, "runtime_bot", None)
            safety = getattr(bot, "safety_support", None) if bot is not None else None
            settings = safety.settings() if safety is not None and callable(getattr(safety, "settings", None)) else {}
            values = settings.get("high_risk_phrases") if isinstance(settings, dict) else []
            if not isinstance(values, list):
                return []
            return [str(item).strip().lower() for item in values if str(item).strip()]

        def _has_negated_reassurance(normalized_text: str) -> bool:
            bot = runtime_bot if runtime_bot is not None else getattr(orchestrator, "runtime_bot", None)
            safety = getattr(bot, "safety_support", None) if bot is not None else None
            check = getattr(safety, "has_negated_reassurance", None) if safety is not None else None
            if callable(check):
                try:
                    return bool(check(normalized_text))
                except Exception:  # noqa: BLE001 - runtime safety adapters may raise domain-specific exceptions.
                    pass
            return any(re.search(pattern, normalized_text) for pattern in sb243_negation_patterns)

        normalized = str(user_input or "").strip().lower()
        if not normalized:
            return ""
        if _has_negated_reassurance(normalized):
            return ""
        terms = _runtime_crisis_terms() or list(sb243_crisis_terms)
        for term in terms:
            if term and term in normalized:
                return term
        return ""

    def _build_sb243_safe_mode_turn_payload(
        *,
        session_id: str,
        tenant_id: str,
        request_id: str,
        matched_term: str,
    ) -> dict:
        task_id = f"sb243-{request_id}"
        intervention_payload = {
            "type": "HARD_STOP_INTERVENTION",
            "source": "dadbot_system.api.sb243",
            "reason": "sb243_crisis_registry",
            "severity": "critical",
            "recommended_action": "SAFE_MODE_LEGAL_BYPASS",
            "safe_mode": "sb243",
            "matched_term": matched_term,
        }
        task_payload = {
            "task_id": task_id,
            "request_id": request_id,
            "session_id": session_id,
            "tenant_id": tenant_id,
            "status": "blocked",
            "session_state": {},
            "error": "SB243 safe-mode bypass activated",
        }
        response_payload = {
            "session_id": session_id,
            "request_id": request_id,
            "tenant_id": tenant_id,
            "reply": (
                "HARD_STOP_INTERVENTION: I cannot continue this turn normally. "
                "If you might hurt yourself or you're in immediate danger, call or text 988 right now "
                "if you're in the U.S. or Canada, or call your local emergency number now."
            ),
            "should_end": True,
            "active_model": "",
            "mood": "critical",
            "status": "blocked",
            "error": "",
            "metadata": {},
        }
        task_payload, response_payload, _ = _build_blocked_turn_payload(
            task_payload=task_payload,
            response_payload=response_payload,
            intervention_payload=intervention_payload,
            blocked_reply=response_payload["reply"],
            task_error="SB243 safe-mode bypass activated",
            enforce_mandatory_halt=True,
        )
        return {
            "task": task_payload,
            "response": response_payload,
            "execution_graph": None,
            "interventions": [intervention_payload],
        }

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

    def _session_state_snapshot(session_id: str, tenant_id: str) -> dict[str, Any]:
        get_session = getattr(orchestrator, "session", None)
        if not callable(get_session):
            return {}
        session = get_session(session_id, tenant_id)
        snapshot = getattr(session, "snapshot", None)
        if not callable(snapshot):
            return {}
        state = snapshot()
        return dict(state) if isinstance(state, dict) else {}

    def _mean(values: list[float]) -> float:
        clean = [float(value) for value in values if value is not None]
        if not clean:
            return 0.0
        return float(sum(clean) / float(len(clean)))

    def _dominant_signal(influence_share: dict[str, Any]) -> str:
        cleaned = {
            "safety": float(influence_share.get("safety", 0.0) or 0.0),
            "tools": float(influence_share.get("tools", 0.0) or 0.0),
            "memory": float(influence_share.get("memory", 0.0) or 0.0),
            "coherence": float(influence_share.get("coherence", 0.0) or 0.0),
        }
        return max(cleaned, key=cleaned.get)

    def _confidence_trend(history: list[dict[str, Any]]) -> float:
        if len(history) < 4:
            return 0.0
        half = max(1, len(history) // 2)
        recent = history[-half:]
        prior = history[:-half]
        if not prior:
            return 0.0
        recent_avg = _mean([float(item.get("decision_confidence", 0.0) or 0.0) for item in recent])
        prior_avg = _mean([float(item.get("decision_confidence", 0.0) or 0.0) for item in prior])
        return float(recent_avg - prior_avg)

    def _stability_classification(*, confidence_avg: float, confidence_trend: float, anomaly_count: int) -> str:
        if confidence_avg < 0.12:
            return "unstable"
        if anomaly_count > 1 or abs(confidence_trend) >= 0.05:
            return "drifting"
        return "stable"

    def _compact_response_engine_diagnostics(session_state: dict[str, Any]) -> dict[str, Any] | None:
        monitor = dict(session_state.get("response_engine_drift_monitor") or {})
        report = dict(session_state.get("response_engine_decision_report") or {})
        selected = dict(report.get("selected") or {})
        if not monitor and not report and not selected:
            return None

        history = list(monitor.get("history") or [])
        window = history[-50:]
        influence_share = dict(selected.get("influence_share") or {})
        if not influence_share:
            influence_share = {
                "safety": abs(float(dict(selected.get("components") or {}).get("safety_weight", 0.0) or 0.0)),
                "tools": abs(float(dict(selected.get("components") or {}).get("tool_weight", 0.0) or 0.0)),
                "memory": abs(float(dict(selected.get("components") or {}).get("memory_weight", 0.0) or 0.0)),
                "coherence": abs(float(dict(selected.get("components") or {}).get("coherence_weight", 0.0) or 0.0)),
            }
            total = float(sum(float(value) for value in influence_share.values()))
            if total > 1e-9:
                influence_share = {key: float(value) / total for key, value in influence_share.items()}
            else:
                influence_share = {"safety": 0.0, "tools": 0.0, "memory": 0.0, "coherence": 0.0}

        confidence_values = [float(item.get("decision_confidence", 0.0) or 0.0) for item in window]
        confidence_avg = _mean(confidence_values)
        if confidence_avg <= 0.0:
            confidence_avg = float(selected.get("decision_confidence", report.get("decision_confidence", 0.0)) or 0.0)
        confidence_trend = _confidence_trend(window)

        anomalies = list(monitor.get("anomalies") or [])
        dominant_signal = _dominant_signal(influence_share)
        last_turn = {
            "confidence": float(
                selected.get("decision_confidence", report.get("decision_confidence", confidence_avg)) or confidence_avg
            ),
            "dominant_signal": dominant_signal,
        }
        return {
            "confidence_avg": round(float(confidence_avg), 3),
            "confidence_trend": round(float(confidence_trend), 3),
            "dominant_signal": dominant_signal,
            "anomalies": anomalies[:8],
            "last_turn": {
                "confidence": round(float(last_turn["confidence"]), 3),
                "dominant_signal": last_turn["dominant_signal"],
            },
            "stability": _stability_classification(
                confidence_avg=float(confidence_avg),
                confidence_trend=float(confidence_trend),
                anomaly_count=len(anomalies),
            ),
        }

    def _latest_response_engine_diagnostics() -> dict[str, Any] | None:
        registry = getattr(orchestrator, "session_registry", None) or getattr(_control_plane, "registry", None)
        sessions = getattr(registry, "_sessions", None)
        if not isinstance(sessions, dict) or not sessions:
            return None
        for session in reversed(list(sessions.values())):
            state = session.get("state") if isinstance(session, dict) else None
            if isinstance(state, dict):
                snapshot = _compact_response_engine_diagnostics(state)
                if snapshot is not None:
                    return snapshot
        return None

    async def _cached_probe(name: str) -> dict[str, Any]:
        if name not in configured_probes:
            return {"status": "unprobed", "probed_at": None, "error_type": None}

        timeout_seconds = min(
            5.0,
            max(0.001, service_config.api.readiness_probe_timeout_ms / 1000.0),
        )
        cache_seconds = max(0, service_config.api.readiness_probe_cache_seconds)
        async with probe_locks[name]:
            now = time.monotonic()
            cached = probe_cache.get(name)
            if cached is not None and cache_seconds > 0 and now - cached[0] < cache_seconds:
                return dict(cached[1])

            model_name = str(
                getattr(runtime_bot, "ACTIVE_MODEL", "") or service_config.default_model
            ).strip()
            endpoint = str(os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").strip()
            vector_url = str(service_config.persistence.vector_store_url or "").strip()
            probe_fn = {
                "model": lambda: _probe_model_dependency(endpoint, model_name, timeout_seconds),
                "vector": lambda: _probe_vector_dependency(vector_url, "", timeout_seconds),
                "memory": lambda: _probe_memory_dependency(runtime_bot, timeout_seconds),
            }[name]
            try:
                status, error_type = await asyncio.wait_for(
                    asyncio.to_thread(probe_fn),
                    timeout=timeout_seconds,
                )
            except TimeoutError:
                status, error_type = "unavailable", "TimeoutError"
            except Exception as exc:
                status, error_type = "unavailable", type(exc).__name__

            result = {
                "status": status,
                "probed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": error_type,
            }
            if cache_seconds > 0:
                probe_cache[name] = (time.monotonic(), result)
            return dict(result)

    async def _readiness_checks() -> tuple[dict[str, dict[str, Any]], bool]:
        dependencies: dict[str, str] = {}
        runtime: dict[str, str] = {}
        ready = True

        state_store_check = getattr(orchestrator.state_store, "check_available", None)
        try:
            if not callable(state_store_check):
                raise RuntimeError("State store does not support availability checks")
            state_store_check()
            dependencies["state_store"] = "available"
        except Exception as exc:
            logger.error("State store readiness check failed: %s", type(exc).__name__)
            dependencies["state_store"] = "unavailable"
            ready = False

        if security_config.auth_required:
            try:
                if delegation_store is None:
                    raise RuntimeError("Delegation authority store is not configured")
                delegation_store.check_available()
                dependencies["authority_store"] = "available"
            except Exception as exc:
                logger.error("Authority store readiness check failed: %s", type(exc).__name__)
                dependencies["authority_store"] = "unavailable"
                ready = False
        else:
            dependencies["authority_store"] = "disabled"

        worker_status = getattr(worker_manager, "readiness_status", None) if worker_manager is not None else None
        if not isinstance(worker_status, str) or worker_status not in (
            "available",
            "starting",
            "quarantined",
            "unavailable",
        ):
            worker_status = "available" if worker_manager is not None and worker_manager.is_ready else "unavailable"
        runtime["workers"] = worker_status
        ready = ready and worker_status == "available"

        model_check, vector_check, memory_check = await asyncio.gather(
            _cached_probe("model"),
            _cached_probe("vector"),
            _cached_probe("memory"),
        )
        ready = ready and all(
            name not in required_probes or result["status"] == "ok"
            for name, result in (
                ("model", model_check),
                ("vector", vector_check),
                ("memory", memory_check),
            )
        )
        return {
            "dependencies": dependencies,
            "runtime": runtime,
            "model": model_check,
            "vector": vector_check,
            "memory": memory_check,
        }, ready

    async def _doctor_diagnostics() -> dict[str, Any]:
        checks, ready = await _readiness_checks()
        optional_probe_degraded = any(
            checks[name]["status"] in {"degraded", "unavailable"}
            for name in ("model", "vector", "memory")
        )
        diagnostics: dict[str, Any] = {
            "status": "ok" if ready and not optional_probe_degraded else "degraded",
            "checks": checks,
        }
        from dadbot.services.memory_service import MemoryService

        diagnostics["memory_retrieval_status_counts"] = MemoryService.recent_retrieval_status_counts()

        tool_registry = getattr(orchestrator, "tool_registry", None)
        if callable(tool_registry):
            tool_registry = tool_registry()
        list_capabilities = getattr(tool_registry, "list_capabilities", None)
        if callable(list_capabilities):
            try:
                tool_count = len(list_capabilities())
                diagnostics["tool_registry"] = {"status": "available", "registered_count": tool_count}
            except Exception as exc:
                logger.error("Tool registry diagnostic failed: %s", type(exc).__name__)
                diagnostics["tool_registry"] = {"status": "unavailable"}
                diagnostics["status"] = "degraded"
        else:
            diagnostics["tool_registry"] = {"status": "not_exposed"}

        if service_config.telemetry.otel_enabled:
            try:
                trace_enabled = is_trace_provider_available()
            except ImportError:
                trace_enabled = False
            diagnostics["trace_backend"] = {
                "status": "available" if trace_enabled else "unavailable",
            }
            if not trace_enabled:
                diagnostics["status"] = "degraded"
        else:
            diagnostics["trace_backend"] = {"status": "disabled"}

        model_name = str(getattr(runtime_bot, "ACTIVE_MODEL", "") or "").strip()
        model_probe_status = checks["model"]["status"]
        diagnostics["model_runtime"] = {
            "status": (
                "configured_not_probed" if model_name else "not_probed"
            ) if model_probe_status == "unprobed" else model_probe_status,
            **({"active_model": model_name[:128]} if model_name else {}),
        }
        vector_configured = bool(str(service_config.persistence.vector_store_url or "").strip())
        vector_probe_status = checks["vector"]["status"]
        diagnostics["vector_store"] = {
            "status": (
                "configured_not_probed" if vector_configured else "not_configured"
            ) if vector_probe_status == "unprobed" else vector_probe_status,
        }

        health_snapshot = getattr(runtime_bot, "current_runtime_health_snapshot", None)
        if callable(health_snapshot):
            try:
                snapshot = health_snapshot(force=False, log_warnings=False, persist=False)
                if isinstance(snapshot, dict):
                    diagnostics["runtime_health"] = {
                        key: snapshot[key]
                        for key in (
                            "level",
                            "health_score",
                            "memory_context_ratio",
                            "prompt_guard_trim_count",
                            "recent_runtime_issue_count",
                        )
                        if key in snapshot
                    }
                    if str(snapshot.get("level") or "").lower() in {"yellow", "red"}:
                        diagnostics["status"] = "degraded"
                else:
                    raise TypeError("Runtime health snapshot must be a mapping")
            except Exception as exc:
                logger.error("Runtime health diagnostic failed: %s", type(exc).__name__)
                diagnostics["runtime_health"] = {"status": "unavailable"}
                diagnostics["status"] = "degraded"
        else:
            diagnostics["runtime_health"] = {"status": "not_exposed"}

        diagnostics["pending_approvals"] = {"status": "not_exposed"}
        return diagnostics

    def _integrity_metadata_from_state(session_state: dict[str, Any]) -> dict[str, Any]:
        last_integrity_status = dict(session_state.get("last_integrity_status") or {})
        return {
            "integrity_failure": not bool(last_integrity_status.get("merkle_check_passed", True)),
            "integrity_failure_reason": str(last_integrity_status.get("reason") or ""),
            "integrity_failure_diagnostics": dict(last_integrity_status.get("diagnostics") or {}),
        }

    def _attach_turn_contract(
        payload: dict[str, Any], *, session_id: str, tenant_id: str, request_id: str
    ) -> dict[str, Any]:
        task_payload = dict(payload.get("task") or {})
        response_payload = dict(payload.get("response") or {})
        session_state = dict(task_payload.get("session_state") or {})
        turn_envelope = build_turn_envelope(
            session_id=session_id,
            request_id=request_id,
            tenant_id=tenant_id,
            response_payload=response_payload,
            session_state=session_state,
            session_metadata=_integrity_metadata_from_state(session_state),
        )
        resolved_payload = dict(payload)
        resolved_payload["turn"] = turn_envelope.model_dump(mode="json")
        return resolved_payload

    def _build_pulse_payload(session_id: str, tenant_id: str, *, event_type: str = "pulse.snapshot") -> dict[str, Any]:
        session_state = _session_state_snapshot(session_id, tenant_id)
        pulse_envelope = build_pulse_envelope(
            session_id=session_id,
            tenant_id=tenant_id,
            session_state=session_state,
            session_metadata=_integrity_metadata_from_state(session_state),
            event_type=event_type,
        )
        return pulse_envelope.model_dump(mode="json")

    router = APIRouter()

    @router.post("/mcp/token")
    async def issue_mcp_access_token(payload: dict, principal: ServicePrincipal = Depends(_http_principal)):
        if not security_config.auth_required or token_manager is None or delegation_store is None:
            raise HTTPException(status_code=503, detail="Authenticated MCP token exchange is unavailable")
        _enforce_rate_limit(principal, "write")
        requested_tools = payload.get("tools")
        if (
            not isinstance(requested_tools, list)
            or not requested_tools
            or len(requested_tools) > len(MCP_TOOL_BASE_SCOPES)
            or any(not isinstance(tool, str) or not tool.strip() for tool in requested_tools)
            or len(set(requested_tools)) != len(requested_tools)
        ):
            raise HTTPException(status_code=422, detail="tools must be a non-empty list of unique tool names")
        try:
            manifest = approved_mcp_tools()
            resource_url = mcp_resource_url()
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error("MCP token exchange is unavailable because its manifest or resource URL is invalid")
            raise HTTPException(status_code=503, detail="MCP tool policy is unavailable") from exc
        if any(tool not in manifest or tool not in MCP_TOOL_BASE_SCOPES for tool in requested_tools):
            raise HTTPException(status_code=422, detail="tools contains an unknown or unapproved MCP tool")
        if principal.allowed_tools and not set(requested_tools).issubset(principal.allowed_tools):
            raise HTTPException(status_code=403, detail="Requested MCP tools exceed the principal's tool grant")

        base_scopes = {MCP_TOOL_BASE_SCOPES[tool] for tool in requested_tools}
        for scope in base_scopes:
            _enforce_scope(principal, scope)

        ttl_seconds = payload.get("ttl_seconds", 120)
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, int)
            or not 1 <= ttl_seconds <= MCP_TOKEN_MAX_TTL_SECONDS
        ):
            raise HTTPException(
                status_code=422,
                detail=f"ttl_seconds must be an integer from 1 to {MCP_TOKEN_MAX_TTL_SECONDS}",
            )
        _resolve_tenant(principal, None)
        scopes = {
            MCP_CONNECT_SCOPE,
            *(mcp_tool_scope(tool) for tool in requested_tools),
            *base_scopes,
        }
        try:
            token = token_manager.issue(
                subject=principal.subject,
                tenant_id=principal.tenant_id,
                scopes=scopes,
                allowed_tools=requested_tools,
                ttl_seconds=ttl_seconds,
                audience=resource_url,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=403, detail="Unable to issue an MCP token for this principal") from exc
        return {
            "access_token": token,
            "token_type": "Bearer",
            "audience": resource_url,
            "expires_in": ttl_seconds,
            "scopes": sorted(scopes),
            "allowed_tools": requested_tools,
        }

    @router.post("/websocket/ticket")
    async def issue_websocket_ticket(payload: dict, principal: ServicePrincipal = Depends(_http_principal)):
        channel = payload.get("channel")
        if channel not in ("events", "pulse", "spatial"):
            raise HTTPException(status_code=422, detail="channel must be events, pulse or spatial")
        _enforce_scope(principal, "write" if channel == "spatial" else "read")
        _enforce_rate_limit(principal, "write" if channel == "spatial" else "read")
        _resolve_tenant(principal, payload.get("tenant_id"))
        path = "/spatial/ws"
        if channel != "spatial":
            session_id = payload.get("session_id")
            if (
                not isinstance(session_id, str)
                or not session_id.strip()
                or any(character in session_id for character in "/?#")
            ):
                raise HTTPException(status_code=422, detail="A valid session_id is required")
            path = f"/sessions/{session_id}/{channel}/stream"
        if websocket_tickets is None:
            raise HTTPException(status_code=503, detail="WebSocket tickets require DADBOT_REDIS_URL")
        try:
            ticket, expires_at = await asyncio.to_thread(websocket_tickets.issue, principal, path=path)
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except TicketStoreUnavailableError as exc:
            logger.error("WebSocket ticket store unavailable during issuance")
            raise HTTPException(status_code=503, detail="WebSocket ticket store unavailable") from exc
        return {"ticket": ticket, "expires_at": expires_at, "expires_in": max(0.0, expires_at - time.time())}

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
        payload = health.to_dict()
        diagnostics = _latest_response_engine_diagnostics()
        if diagnostics is not None:
            payload["response_engine_diagnostics"] = diagnostics
        return payload

    @router.get("/ready")
    async def readiness_check():
        check_groups, ready = await _readiness_checks()

        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "status": "ready" if ready else "not_ready",
                "checks": check_groups,
            },
        )

    @router.get("/doctor")
    async def doctor_check(principal: ServicePrincipal = Depends(_http_principal)):
        _enforce_scope(principal, "admin")
        _enforce_rate_limit(principal, "admin")
        diagnostics = await _doctor_diagnostics()
        return JSONResponse(
            status_code=200 if diagnostics["status"] == "ok" else 503,
            content=diagnostics,
        )

    @router.post("/sessions/{session_id}/chat")
    async def submit_chat(
        session_id: str, payload: dict, http_request: Request, principal: ServicePrincipal = Depends(_http_principal)
    ):
        _enforce_scope(principal, "write")
        _enforce_rate_limit(principal, "write")
        tenant_id = _resolve_tenant(principal, payload.get("tenant_id"))
        request_payload = _apply_request_identity(payload, principal, tenant_id=tenant_id)
        request = ChatRequest.from_dict(request_payload, session_id=session_id)
        if not request.user_input and not request.attachments:
            raise HTTPException(status_code=400, detail="user_input or attachments are required")
        task = _submit_chat_request(request, http_request)
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
        http_request: Request,
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
        task = _submit_chat_request(request, http_request)
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
    async def execute_turn(
        session_id: str, payload: dict, http_request: Request, principal: ServicePrincipal = Depends(_http_principal)
    ):
        _enforce_scope(principal, "write")
        _enforce_rate_limit(principal, "write")
        tenant_id = _resolve_tenant(principal, payload.get("tenant_id"))
        resolved_payload = _apply_request_identity(payload, principal, tenant_id=tenant_id)
        check_request = ChatRequest.from_dict(resolved_payload, session_id=session_id)
        if not check_request.user_input and not check_request.attachments:
            raise HTTPException(status_code=400, detail="user_input or attachments are required")

        sovereign_context = SovereignContext(
            session_id=str(session_id or "default"),
            tenant_id=str(tenant_id or DEFAULT_TENANT_ID),
            trace_id=str(resolved_payload.get("trace_id") or check_request.request_id or ""),
            request_id=str(check_request.request_id or ""),
            execution_mode=ExecutionMode.LIVE,
            policy_scope="api.turn",
        )
        resolved_payload["trace_id"] = sovereign_context.trace_id
        metadata = dict(resolved_payload.get("metadata") or {})
        metadata["sovereign_context"] = sovereign_context.to_dict()
        resolved_payload["metadata"] = metadata

        if sb243_safe_mode_enabled:
            matched_term = _detect_sb243_crisis_signal(check_request.user_input)
            if matched_term:
                return _attach_turn_contract(
                    _build_sb243_safe_mode_turn_payload(
                        session_id=session_id,
                        tenant_id=tenant_id,
                        request_id=check_request.request_id,
                        matched_term=matched_term,
                    ),
                    session_id=session_id,
                    tenant_id=tenant_id,
                    request_id=check_request.request_id,
                )

        timeout_seconds = float(resolved_payload.get("timeout_seconds") or 60.0)
        resolved_payload["request_id"] = check_request.request_id
        job_payload = {**resolved_payload, "_session_id": session_id}
        job_payload.pop("_submitted_task_id", None)
        if security_config.auth_required and security_config.authority_sqlite_path:
            request = ChatRequest.from_dict(resolved_payload, session_id=session_id)
            task = _submit_chat_request(request, http_request)
            job_payload["_submitted_task_id"] = task.task_id
        result_future = await _control_plane.submit_turn(session_id, job_payload)

        try:
            result = await asyncio.wait_for(result_future, timeout=timeout_seconds)
        except IntegrityBreachError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"Integrity breach hard-abort: {exc}",
            ) from exc
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Timed out waiting for turn completion") from exc

        task_outcomes = await _control_plane.task_manager.await_session(session_id)
        for outcome in task_outcomes:
            if isinstance(outcome, IntegrityBreachError):
                raise HTTPException(
                    status_code=409,
                    detail=f"Integrity breach hard-abort: {outcome}",
                )
            if isinstance(outcome, Exception):
                raise HTTPException(status_code=500, detail=f"Kernel background task failure: {outcome}")

        task_status = result.get("task")
        response = result.get("response")

        if task_status is None:
            raise HTTPException(status_code=404, detail="Task not found")
        if response is None and str((task_status or {}).get("status") or "").strip().lower() != "failed":
            raise HTTPException(status_code=504, detail="Timed out waiting for turn completion")

        return _attach_turn_contract(
            _apply_veto_proxy_to_turn_payload(result),
            session_id=session_id,
            tenant_id=tenant_id,
            request_id=check_request.request_id,
        )

    @router.post("/sessions/{session_id}/stream")
    async def submit_stream(
        session_id: str, payload: dict, http_request: Request, principal: ServicePrincipal = Depends(_http_principal)
    ):
        _enforce_scope(principal, "write")
        _enforce_rate_limit(principal, "write")
        tenant_id = _resolve_tenant(principal, payload.get("tenant_id"))
        request_payload = _apply_request_identity(payload, principal, tenant_id=tenant_id)
        request = ChatRequest.from_dict(request_payload, session_id=session_id)
        if not request.user_input and not request.attachments:
            raise HTTPException(status_code=400, detail="user_input or attachments are required")

        task = _submit_chat_request(request, http_request)
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
        tenant_id = _resolve_tenant(principal, None)
        status = orchestrator.task_status(task_id, tenant_id=tenant_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Task not found")
        response = orchestrator.response_for_task(task_id, tenant_id=tenant_id)
        return {"task": status, "response": response}

    @router.get("/automation/status")
    @_with_runtime_authority("admin")
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
    @_with_runtime_authority("admin")
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
    @_with_runtime_authority("admin")
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
    @_with_runtime_authority("admin")
    async def browser_status(principal: ServicePrincipal = Depends(_http_principal)):
        _enforce_scope(principal, "admin")
        _enforce_rate_limit(principal, "admin")
        bot = _resolve_runtime_bot()
        if not callable(getattr(bot, "local_mcp_status", None)):
            raise HTTPException(status_code=503, detail="Local MCP controls are not available")
        return dict(bot.local_mcp_status() or {})

    @router.post("/automation/browser/start")
    @_with_runtime_authority("admin")
    async def browser_start(payload: dict | None = None, principal: ServicePrincipal = Depends(_http_principal)):
        _enforce_scope(principal, "admin")
        _enforce_rate_limit(principal, "admin")
        bot = _resolve_runtime_bot()
        if not callable(getattr(bot, "start_local_mcp_server_process", None)):
            raise HTTPException(status_code=503, detail="Local MCP controls are not available")
        options = dict(payload or {})
        return dict(bot.start_local_mcp_server_process(restart=bool(options.get("restart", False))) or {})

    @router.post("/automation/browser/stop")
    @_with_runtime_authority("admin")
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
        tenant_id: str | None = None,
        principal: ServicePrincipal = Depends(_http_principal),
    ):
        _enforce_scope(principal, "read")
        _enforce_rate_limit(principal, "read")
        effective_tenant_id = _resolve_tenant(principal, tenant_id)
        return orchestrator.session_events(session_id, tenant_id=effective_tenant_id)

    @router.get("/sessions/{session_id}/actions/explain")
    async def explain_session_actions(
        session_id: str,
        tenant_id: str | None = None,
        limit: int = 10,
        principal: ServicePrincipal = Depends(_http_principal),
    ):
        _enforce_scope(principal, "read")
        _enforce_rate_limit(principal, "read")
        effective_tenant_id = _resolve_tenant(principal, tenant_id)
        from .action_explanation import explain_last_actions

        session_state = _session_state_snapshot(session_id, effective_tenant_id)
        return {
            "session_id": session_id,
            "tenant_id": effective_tenant_id,
            **explain_last_actions(session_state.get("last_tool_actions"), limit=limit),
        }

    @router.get("/sessions/{session_id}/replay")
    async def replay_session_events(
        session_id: str,
        tenant_id: str | None = None,
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
    async def stream_session_events(
        websocket: WebSocket,
        session_id: str,
        tenant_id: str = Depends(
            _websocket_tenant_dependency(
                "/sessions/{session_id}/events/stream",
                "read",
            )
        ),
    ):
        effective_tenant_id = tenant_id

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

    @router.websocket("/sessions/{session_id}/pulse/stream")
    async def stream_session_pulse(
        websocket: WebSocket,
        session_id: str,
        tenant_id: str = Depends(
            _websocket_tenant_dependency(
                "/sessions/{session_id}/pulse/stream",
                "read",
            )
        ),
    ):
        effective_tenant_id = tenant_id

        await websocket.accept()
        last_signature = ""
        try:
            while True:
                payload = _build_pulse_payload(session_id, effective_tenant_id)
                signature = json.dumps(payload, sort_keys=True, default=str)
                if signature != last_signature:
                    await websocket.send_json(payload)
                    last_signature = signature
                else:
                    await websocket.send_json(
                        {
                            "event_type": "pulse.heartbeat",
                            "session_id": session_id,
                            "tenant_id": effective_tenant_id,
                            "status": "alive",
                        }
                    )
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            return

    @router.websocket("/spatial/ws")
    async def spatial_avatar_bridge(
        websocket: WebSocket,
        tenant_id: str = Depends(
            _websocket_tenant_dependency("/spatial/ws", "write")
        ),
    ):

        await websocket.accept()
        principal = websocket.state.service_principal
        while True:
            try:
                raw_message = await websocket.receive_text()
            except WebSocketDisconnect:
                return

            try:
                payload = json.loads(raw_message)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "error": "invalid_json"})
                continue

            if not isinstance(payload, dict):
                await websocket.send_json({"type": "error", "error": "invalid_payload"})
                continue

            user_text = _extract_spatial_text(payload)
            if not user_text:
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": "text_required",
                        "message": "Provide text/transcript in the websocket payload.",
                    }
                )
                continue

            try:
                request_payload = _apply_request_identity(
                    {"user_input": user_text, "channel": "spatial"},
                    principal,
                    tenant_id=tenant_id,
                )
                request = ChatRequest.from_dict(request_payload, session_id="spatial-avatar")
                task = orchestrator.submit_chat(request, execution_principal=principal)
                task_status, response = await _wait_for_task_completion(
                    task.task_id,
                    tenant_id=tenant_id,
                    timeout_seconds=45.0,
                )
                if response is None:
                    detail = str((task_status or {}).get("error") or "Worker did not return a response")
                    raise HTTPException(status_code=503, detail=detail)
                reply_text = str(response.get("reply") or "")
            except Exception as exc:
                message = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": "runtime_failure",
                        "message": message,
                    }
                )
                continue

            bot = runtime_bot if runtime_bot is not None else getattr(orchestrator, "runtime_bot", None)
            audio_base64, audio_error, audio_mime = _encode_spatial_audio(bot, reply_text)
            response_payload = {
                "type": "response",
                "user": {"text": user_text},
                "text": reply_text,
                "audio_base64": audio_base64,
                "audio_mime": audio_mime or "audio/wav",
                "audio_mime_type": audio_mime or "audio/wav",
            }
            if audio_error:
                response_payload["audio_error"] = audio_error
            await websocket.send_json(response_payload)

    version_prefix = _version_prefix()
    if version_prefix:
        app.include_router(router, prefix=version_prefix)
        if service_config.api.legacy_routes_enabled:
            app.include_router(router)
    else:
        app.include_router(router)

    return app
