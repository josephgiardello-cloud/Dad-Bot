from __future__ import annotations

# ruff: noqa: B008, C901, PLR0913, PLR0915
import asyncio
import hmac
import json
import os
import re
from contextlib import asynccontextmanager
from typing import Any, cast

from dadbot.core.execution_contract import ExecutionMode, SovereignContext
from dadbot.core.execution_ledger import IntegrityBreachError
from dadbot.api_models import build_pulse_envelope, build_turn_envelope

from .contracts import (
    DEFAULT_TENANT_ID,
    ChatRequest,
    EventEnvelope,
    EventType,
    HealthResponse,
    ServiceConfig,
    normalize_channel_name,
)
from .kernel import ControlPlane, build_control_plane
from .security import (
    AuthenticationError,
    AuthorizationError,
    ServicePrincipal,
    ServiceTokenManager,
    SlidingWindowRateLimiter,
)
from .state import NamespacedStateStore

try:
    from fastapi import (  # type: ignore[reportMissingImports]
        APIRouter,
        Depends,
        FastAPI,
        Header,
        HTTPException,
        Request,
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
    Request = cast(Any, None)
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

    def _is_valid_secret_key(provided_secret: str) -> bool:
        if not dadbot_secret_key:
            return True
        candidate = str(provided_secret or "").strip()
        if not candidate:
            return False
        return hmac.compare_digest(candidate, dadbot_secret_key)

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
        if dadbot_secret_key:
            provided_secret = request.headers.get("x-dadbot-key")
            if not _is_valid_secret_key(str(provided_secret or "")):
                return JSONResponse(status_code=401, content={"detail": "Invalid or missing X-DADBOT-KEY"})
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
        scoped_store = NamespacedStateStore(orchestrator.state_store, orchestrator.tenant_namespace(tenant_id))
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
        tenant_id = str(task_data.get("tenant_id") or response_data.get("tenant_id") or DEFAULT_TENANT_ID)

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
            orchestrator.state_store.save_task(task_id, task_data)
            orchestrator.state_store.save_response(task_id, response_data)

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

    def _session_state_snapshot(session_id: str) -> dict[str, Any]:
        registry = getattr(orchestrator, "session_registry", None) or getattr(_control_plane, "registry", None)
        session = registry.get(session_id) if registry is not None and callable(getattr(registry, "get", None)) else None
        if not isinstance(session, dict):
            return {}
        state = session.get("state")
        return dict(state) if isinstance(state, dict) else {}

    def _integrity_metadata_from_state(session_state: dict[str, Any]) -> dict[str, Any]:
        last_integrity_status = dict(session_state.get("last_integrity_status") or {})
        return {
            "integrity_failure": not bool(last_integrity_status.get("merkle_check_passed", True)),
            "integrity_failure_reason": str(last_integrity_status.get("reason") or ""),
            "integrity_failure_diagnostics": dict(last_integrity_status.get("diagnostics") or {}),
        }

    def _attach_turn_contract(payload: dict[str, Any], *, session_id: str, tenant_id: str, request_id: str) -> dict[str, Any]:
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
        session_state = _session_state_snapshot(session_id)
        pulse_envelope = build_pulse_envelope(
            session_id=session_id,
            tenant_id=tenant_id,
            session_state=session_state,
            session_metadata=_integrity_metadata_from_state(session_state),
            event_type=event_type,
        )
        return pulse_envelope.model_dump(mode="json")

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
        request_payload = _apply_request_identity(payload, principal, tenant_id=tenant_id)
        request = ChatRequest.from_dict(request_payload, session_id=session_id)
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
        job_payload = {**resolved_payload, "_session_id": session_id}
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
    async def submit_stream(session_id: str, payload: dict, principal: ServicePrincipal = Depends(_http_principal)):
        _enforce_scope(principal, "write")
        _enforce_rate_limit(principal, "write")
        tenant_id = _resolve_tenant(principal, payload.get("tenant_id"))
        request_payload = _apply_request_identity(payload, principal, tenant_id=tenant_id)
        request = ChatRequest.from_dict(request_payload, session_id=session_id)
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
            provided_secret = str(
                websocket.query_params.get("dadbot_key")
                or websocket.query_params.get("x_dadbot_key")
                or websocket.headers.get("x-dadbot-key")
                or ""
            )
            if not _is_valid_secret_key(provided_secret):
                raise HTTPException(status_code=401, detail="Invalid or missing X-DADBOT-KEY")
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

    @router.websocket("/sessions/{session_id}/pulse/stream")
    async def stream_session_pulse(websocket: WebSocket, session_id: str, tenant_id: str = ""):
        try:
            provided_secret = str(
                websocket.query_params.get("dadbot_key")
                or websocket.query_params.get("x_dadbot_key")
                or websocket.headers.get("x-dadbot-key")
                or ""
            )
            if not _is_valid_secret_key(provided_secret):
                raise HTTPException(status_code=401, detail="Invalid or missing X-DADBOT-KEY")
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

    version_prefix = _version_prefix()
    if version_prefix:
        app.include_router(router, prefix=version_prefix)
        if service_config.api.legacy_routes_enabled:
            app.include_router(router)
    else:
        app.include_router(router)

    return app
