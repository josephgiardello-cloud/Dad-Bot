"""Production-grade external tool runtime primitives.

This module closes key operational gaps between deterministic local tool execution
and real-world connector execution:

- HTTP/auth abstraction boundaries
- Retry/backoff and network failure modeling
- Rate limiting and throttling
- Dynamic tool capability discovery + version negotiation
- Cost/latency/reliability-aware routing
- Partial success / degraded-mode / fallback semantics
- Execution isolation profiles with resource-limit prechecks
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from dadbot.core.kernel_locks import KernelToolIdempotencyRegistry

_IDEMPOTENCY_POLICY_KEYS: frozenset[str] = frozenset(
    {
        "approval_granted",
        "enforce_hitl",
        "enforce_permissions",
        "session_permissions",
    },
)


class ToolExecutionStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    DEGRADED = "degraded"
    SKIPPED = "skipped"


class NetworkFailureKind(str, Enum):
    DNS = "dns"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    RATE_LIMIT = "rate_limit"
    SERVER = "server"
    CLIENT = "client"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class HttpRequest:
    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    json_body: dict[str, Any] | None = None
    timeout_seconds: float = 10.0


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    headers: dict[str, str] = field(default_factory=dict)
    body: Any = None
    elapsed_ms: float = 0.0

    @property
    def is_success(self) -> bool:
        return 200 <= int(self.status_code) < 300

    @property
    def is_partial(self) -> bool:
        return int(self.status_code) in {206, 207}


class HttpTransport(Protocol):
    """Transport boundary so runtime logic is decoupled from requests/httpx."""

    def send(self, request: HttpRequest) -> HttpResponse: ...


class AuthStrategy(Protocol):
    def apply(self, headers: dict[str, str]) -> dict[str, str]: ...


@dataclass(frozen=True)
class NoAuth:
    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        return dict(headers)


@dataclass(frozen=True)
class BearerTokenAuth:
    token: str

    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        merged = dict(headers)
        merged["Authorization"] = f"Bearer {self.token}"
        return merged


@dataclass(frozen=True)
class ApiKeyAuth:
    api_key: str
    header_name: str = "X-API-Key"
    prefix: str = ""

    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        merged = dict(headers)
        value = f"{self.prefix}{self.api_key}" if self.prefix else self.api_key
        merged[self.header_name] = value
        return merged


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_ms: float = 80.0
    max_delay_ms: float = 1_500.0
    backoff_factor: float = 2.0
    jitter_ratio: float = 0.2
    retryable_statuses: frozenset[int] = frozenset(
        {408, 409, 425, 429, 500, 502, 503, 504},
    )

    def delay_for_attempt(
        self,
        attempt: int,
        rng: Callable[[], float] | None = None,
    ) -> float:
        """Return delay seconds for attempt number (1-indexed)."""
        # Deterministic default jitter source for core/runtime safety.
        rng_fn = rng or (lambda: 0.5)
        exponential_ms = self.base_delay_ms * (self.backoff_factor ** max(0, int(attempt) - 1))
        bounded_ms = min(self.max_delay_ms, exponential_ms)
        jitter = bounded_ms * self.jitter_ratio * rng_fn()
        return max(0.0, (bounded_ms + jitter) / 1000.0)

    def should_retry(
        self,
        *,
        attempt: int,
        status_code: int | None = None,
        failure_kind: NetworkFailureKind | None = None,
    ) -> bool:
        if int(attempt) >= max(1, int(self.max_attempts)):
            return False
        if failure_kind in {
            NetworkFailureKind.DNS,
            NetworkFailureKind.TIMEOUT,
            NetworkFailureKind.CONNECTION,
            NetworkFailureKind.SERVER,
            NetworkFailureKind.RATE_LIMIT,
        }:
            return True
        if status_code is not None and int(status_code) in self.retryable_statuses:
            return True
        return False


@dataclass(frozen=True)
class RateLimitPolicy:
    max_requests: int
    window_seconds: float


class SlidingWindowRateLimiter:
    """Lightweight throttling guard for tool invocations."""

    def __init__(
        self,
        policy: RateLimitPolicy,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._policy = policy
        self._clock = clock or time.monotonic
        self._events: deque[float] = deque()

    def _evict(self, now: float) -> None:
        cutoff = now - max(0.001, float(self._policy.window_seconds))
        while self._events and self._events[0] <= cutoff:
            self._events.popleft()

    def allow(self) -> bool:
        now = self._clock()
        self._evict(now)
        if len(self._events) >= max(1, int(self._policy.max_requests)):
            return False
        self._events.append(now)
        return True

    def remaining(self) -> int:
        now = self._clock()
        self._evict(now)
        return max(0, int(self._policy.max_requests) - len(self._events))


@dataclass(frozen=True)
class ResourceLimits:
    max_cpu_ms: float = 2_000.0
    max_memory_mb: float = 256.0
    max_io_ops: int = 200
    max_network_calls: int = 10


@dataclass(frozen=True)
class IsolationProfile:
    tool_name: str
    compartment: str
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    allow_network: bool = True
    allow_filesystem: bool = False


@dataclass(frozen=True)
class ResourceEstimate:
    cpu_ms: float = 0.0
    memory_mb: float = 0.0
    io_ops: int = 0
    network_calls: int = 0


class IsolationGuard:
    def __init__(self) -> None:
        self._profiles: dict[str, IsolationProfile] = {}

    def register(self, profile: IsolationProfile) -> None:
        self._profiles[str(profile.tool_name).strip().lower()] = profile

    def profile_for(self, tool_name: str) -> IsolationProfile | None:
        return self._profiles.get(str(tool_name).strip().lower())

    def validate(
        self,
        tool_name: str,
        estimate: ResourceEstimate | None,
    ) -> tuple[bool, str]:
        profile = self.profile_for(tool_name)
        if profile is None or estimate is None:
            return True, "ok"
        limits = profile.limits
        if estimate.cpu_ms > limits.max_cpu_ms:
            return False, "cpu_limit_exceeded"
        if estimate.memory_mb > limits.max_memory_mb:
            return False, "memory_limit_exceeded"
        if estimate.io_ops > limits.max_io_ops:
            return False, "io_limit_exceeded"
        if estimate.network_calls > limits.max_network_calls:
            return False, "network_limit_exceeded"
        if not profile.allow_network and estimate.network_calls > 0:
            return False, "network_disallowed"
        return True, "ok"


@dataclass(frozen=True)
class ToolCapability:
    name: str
    version: str
    intents: tuple[str, ...]
    cost_units: float
    avg_latency_ms: float
    reliability: float = 0.9
    supports_partial: bool = True
    tags: tuple[str, ...] = field(default_factory=tuple)

    def supports_intent(self, intent: str) -> bool:
        needle = str(intent or "").strip().lower()
        return needle in {item.strip().lower() for item in self.intents}


@dataclass
class ToolExecutionResult:
    tool_name: str
    status: ToolExecutionStatus
    output: Any = None
    error: str = ""
    attempts: int = 1
    latency_ms: float = 0.0
    confidence: float = 1.0
    degraded_reason: str = ""
    fallback_used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def usable(self, *, min_confidence: float = 0.5) -> bool:
        return self.status in {
            ToolExecutionStatus.OK,
            ToolExecutionStatus.PARTIAL,
            ToolExecutionStatus.DEGRADED,
        } and float(self.confidence) >= float(min_confidence)


ToolHandler = Callable[[dict[str, Any]], ToolExecutionResult]
PermissionEnforcer = Callable[[str, dict[str, Any]], bool | tuple[bool, str]]
MemorySink = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class ToolExecutionTrace:
    tool_name: str
    status: str
    attempts: int
    latency_ms: float
    failure_kind: str
    error_code: str
    degraded_reason: str
    fallback_used: bool
    timestamp_utc_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "status": self.status,
            "attempts": self.attempts,
            "latency_ms": self.latency_ms,
            "failure_kind": self.failure_kind,
            "error_code": self.error_code,
            "degraded_reason": self.degraded_reason,
            "fallback_used": self.fallback_used,
            "timestamp_utc_ms": self.timestamp_utc_ms,
        }


class DynamicToolRegistry:
    """Runtime tool registration + capability introspection/negotiation."""

    def __init__(self) -> None:
        self._capabilities: dict[str, list[ToolCapability]] = {}
        self._handlers: dict[tuple[str, str], ToolHandler] = {}

    def register(self, capability: ToolCapability, handler: ToolHandler) -> None:
        key = str(capability.name).strip().lower()
        self._capabilities.setdefault(key, []).append(capability)
        self._capabilities[key].sort(
            key=lambda item: _parse_version(item.version),
            reverse=True,
        )
        self._handlers[(key, capability.version)] = handler

    def discover(
        self,
        *,
        intent: str | None = None,
        tag: str | None = None,
    ) -> list[ToolCapability]:
        found: list[ToolCapability] = []
        for versions in self._capabilities.values():
            for cap in versions:
                if intent and not cap.supports_intent(intent):
                    continue
                if tag and str(tag).strip().lower() not in {t.lower() for t in cap.tags}:
                    continue
                found.append(cap)
        return sorted(
            found,
            key=lambda item: (item.name, _parse_version(item.version)),
            reverse=True,
        )

    def negotiate(
        self,
        name: str,
        *,
        required_version: str | None = None,
    ) -> ToolCapability | None:
        candidates = list(self._capabilities.get(str(name).strip().lower(), []))
        if not candidates:
            return None
        if required_version is None:
            return candidates[0]
        required_major = _parse_version(required_version)[0]
        compatible = [item for item in candidates if _parse_version(item.version)[0] == required_major]
        return compatible[0] if compatible else None

    def handler_for(
        self,
        name: str,
        *,
        required_version: str | None = None,
    ) -> tuple[ToolCapability, ToolHandler] | None:
        capability = self.negotiate(name, required_version=required_version)
        if capability is None:
            return None
        key = (str(capability.name).strip().lower(), capability.version)
        handler = self._handlers.get(key)
        if handler is None:
            return None
        return capability, handler


@dataclass(frozen=True)
class RouterWeights:
    reliability_weight: float = 1.4
    latency_weight: float = 0.002
    cost_weight: float = 0.4
    load_weight: float = 0.25


class CostAwareToolRouter:
    """Select a tool capability using cost, latency, reliability, and load."""

    def __init__(self, *, weights: RouterWeights | None = None) -> None:
        self._weights = weights or RouterWeights()

    def route(
        self,
        registry: DynamicToolRegistry,
        *,
        intent: str,
        load_by_tool: dict[str, float] | None = None,
    ) -> ToolCapability | None:
        candidates = registry.discover(intent=intent)
        if not candidates:
            return None
        loads = {str(k).strip().lower(): float(v) for k, v in dict(load_by_tool or {}).items()}

        def score(cap: ToolCapability) -> float:
            load = max(0.0, loads.get(cap.name.lower(), 0.0))
            return (
                self._weights.reliability_weight * max(0.0, min(1.0, float(cap.reliability)))
                - self._weights.latency_weight * max(0.0, float(cap.avg_latency_ms))
                - self._weights.cost_weight * max(0.0, float(cap.cost_units))
                - self._weights.load_weight * load
            )

        return max(candidates, key=score)


def classify_network_failure(
    *,
    exc: Exception | None = None,
    status_code: int | None = None,
) -> NetworkFailureKind:
    if status_code is not None:
        code = int(status_code)
        if code == 429:
            return NetworkFailureKind.RATE_LIMIT
        if 500 <= code <= 599:
            return NetworkFailureKind.SERVER
        if 400 <= code <= 499:
            return NetworkFailureKind.CLIENT
    if exc is None:
        return NetworkFailureKind.UNKNOWN
    message = str(exc).lower()
    name = type(exc).__name__.lower()
    if "timeout" in message or "timeout" in name:
        return NetworkFailureKind.TIMEOUT
    if "dns" in message or "name or service not known" in message:
        return NetworkFailureKind.DNS
    if "connection" in message or "connection" in name or "connect" in message:
        return NetworkFailureKind.CONNECTION
    return NetworkFailureKind.UNKNOWN


def _extract_idempotency_policy_context(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract policy-relevant context so idempotency cannot cross policy boundaries."""
    source = dict(payload or {})
    context: dict[str, Any] = {}
    for key in _IDEMPOTENCY_POLICY_KEYS:
        if key in source:
            context[key] = source.get(key)

    nested = source.get("policy_context")
    if not isinstance(nested, dict):
        nested = source.get("_policy_context")
    if isinstance(nested, dict):
        for key in _IDEMPOTENCY_POLICY_KEYS:
            if key in nested and key not in context:
                context[key] = nested.get(key)

    permissions = context.get("session_permissions")
    if isinstance(permissions, (list, tuple, set)):
        context["session_permissions"] = sorted(str(item) for item in permissions)
    return context


class ExternalToolRuntime:
    """Failure-aware runtime executor for external tools.

    Includes retry/backoff, throttling, isolation checks, partial-result semantics,
    and fallback tool chaining.
    """

    def __init__(
        self,
        registry: DynamicToolRegistry,
        *,
        retry_policy: RetryPolicy | None = None,
        rate_limiter: SlidingWindowRateLimiter | None = None,
        isolation_guard: IsolationGuard | None = None,
        sleeper: Callable[[float], None] | None = None,
        event_tap: Any | None = None,
        permission_enforcer: PermissionEnforcer | None = None,
        hard_timeout_seconds: float | None = None,
        latency_spike_ms: float = 2_500.0,
        memory_sink: MemorySink | None = None,
        retain_traces: int = 2_000,
    ) -> None:
        self._registry = registry
        self._retry_policy = retry_policy or RetryPolicy()
        self._rate_limiter = rate_limiter
        self._isolation_guard = isolation_guard or IsolationGuard()
        self._sleeper = sleeper or time.sleep
        self._event_tap = event_tap
        self._permission_enforcer = permission_enforcer
        self._hard_timeout_seconds = (
            max(0.01, float(hard_timeout_seconds)) if hard_timeout_seconds is not None else None
        )
        self._latency_spike_ms = max(0.0, float(latency_spike_ms))
        self._memory_sink = memory_sink
        self._traces: deque[ToolExecutionTrace] = deque(maxlen=max(10, int(retain_traces)))

    @staticmethod
    def _normalized_error_code(raw: str) -> str:
        token = str(raw or "").strip().lower()
        if not token:
            return "unknown"
        safe = []
        for char in token:
            if char.isalnum() or char in {"_", ":"}:
                safe.append(char)
            elif char in {" ", "-", ".", "/"}:
                safe.append("_")
        value = "".join(safe).strip("_")
        if not value:
            return "unknown"
        return value.split(":", 1)[0]

    def _build_normalized_error(
        self,
        *,
        error: str,
        status: ToolExecutionStatus,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        local_meta = dict(metadata or {})
        status_code = local_meta.get("http_status")
        failure_kind = classify_network_failure(status_code=status_code).value
        if status == ToolExecutionStatus.TIMEOUT:
            failure_kind = NetworkFailureKind.TIMEOUT.value
        if not error and status == ToolExecutionStatus.DEGRADED:
            failure_kind = NetworkFailureKind.UNKNOWN.value
        retryable = self._retry_policy.should_retry(
            attempt=1,
            status_code=status_code,
            failure_kind=classify_network_failure(status_code=status_code),
        )
        code = self._normalized_error_code(error)
        return {
            "code": code,
            "message": str(error or "unknown_error"),
            "status": status.value,
            "failure_kind": failure_kind,
            "status_code": status_code,
            "retryable": bool(retryable),
        }

    def _normalize_failure_surface(self, result: ToolExecutionResult) -> ToolExecutionResult:
        result.metadata = dict(result.metadata or {})
        if result.status in {ToolExecutionStatus.OK, ToolExecutionStatus.PARTIAL}:
            return result
        normalized = self._build_normalized_error(
            error=str(result.error or ""),
            status=result.status,
            metadata=result.metadata,
        )
        result.metadata["normalized_error"] = normalized
        if not result.error:
            result.error = str(normalized.get("message") or "unknown_error")
        return result

    def _apply_latency_spike_model(self, result: ToolExecutionResult) -> ToolExecutionResult:
        if self._latency_spike_ms <= 0:
            return result
        if result.status != ToolExecutionStatus.OK:
            return result
        if float(result.latency_ms) <= self._latency_spike_ms:
            return result
        result.status = ToolExecutionStatus.DEGRADED
        result.degraded_reason = "latency_spike"
        result.confidence = min(float(result.confidence), 0.49)
        result.metadata = dict(result.metadata or {})
        result.metadata["latency_spike"] = True
        result.metadata["latency_threshold_ms"] = self._latency_spike_ms
        return result

    def _record_trace(self, result: ToolExecutionResult) -> None:
        metadata = dict(result.metadata or {})
        normalized = dict(metadata.get("normalized_error") or {})
        self._traces.append(
            ToolExecutionTrace(
                tool_name=str(result.tool_name or ""),
                status=result.status.value,
                attempts=max(0, int(result.attempts)),
                latency_ms=max(0.0, float(result.latency_ms)),
                failure_kind=str(normalized.get("failure_kind") or ""),
                error_code=str(normalized.get("code") or ""),
                degraded_reason=str(result.degraded_reason or ""),
                fallback_used=bool(result.fallback_used),
                timestamp_utc_ms=int(time.time() * 1000),
            )
        )

    def _ingest_tool_result_memory(self, *, payload: dict[str, Any], result: ToolExecutionResult) -> None:
        if not callable(self._memory_sink):
            return
        output_preview = result.output
        if isinstance(output_preview, dict):
            output_preview = dict(list(output_preview.items())[:8])
        event = {
            "kind": "tool_result",
            "tool_name": str(result.tool_name or ""),
            "status": result.status.value,
            "attempts": int(result.attempts),
            "latency_ms": float(result.latency_ms),
            "degraded_reason": str(result.degraded_reason or ""),
            "error": str(result.error or ""),
            "input": dict(payload or {}),
            "output_preview": output_preview,
            "metadata": dict(result.metadata or {}),
            "captured_at_ms": int(time.time() * 1000),
        }
        try:
            self._memory_sink(event)
        except Exception:
            # Ingestion is non-critical and must never break tool execution.
            return

    @staticmethod
    def _permission_decision(
        enforcer: PermissionEnforcer | None,
        tool_name: str,
        payload: dict[str, Any],
    ) -> tuple[bool, str]:
        if not callable(enforcer):
            return True, "allowed"
        decision = enforcer(tool_name, dict(payload or {}))
        if isinstance(decision, tuple):
            allowed = bool(decision[0])
            reason = str(decision[1] if len(decision) > 1 else "permission_denied")
            return allowed, reason
        if bool(decision):
            return True, "allowed"
        return False, "permission_denied"

    @staticmethod
    def _run_handler_with_timeout(
        handler: ToolHandler,
        payload: dict[str, Any],
        timeout_seconds: float | None,
    ) -> ToolExecutionResult:
        if timeout_seconds is None:
            return handler(dict(payload))
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(handler, dict(payload))
            try:
                return future.result(timeout=max(0.01, float(timeout_seconds)))
            except FutureTimeoutError as exc:
                future.cancel()
                raise TimeoutError("tool_timeout_exceeded") from exc

    def tool_execution_stats(self) -> dict[str, Any]:
        traces = list(self._traces)
        total = len(traces)
        failures = [
            t
            for t in traces
            if t.status in {
                ToolExecutionStatus.ERROR.value,
                ToolExecutionStatus.TIMEOUT.value,
                ToolExecutionStatus.DEGRADED.value,
            }
        ]
        latencies = sorted(t.latency_ms for t in traces)
        p95_index = min(len(latencies) - 1, int(round((len(latencies) - 1) * 0.95))) if latencies else 0
        failure_classes: dict[str, int] = {}
        for item in failures:
            key = item.failure_kind or item.error_code or "unknown"
            failure_classes[key] = int(failure_classes.get(key, 0)) + 1
        return {
            "total": total,
            "errors": len([t for t in traces if t.status in {ToolExecutionStatus.ERROR.value, ToolExecutionStatus.TIMEOUT.value}]),
            "timeouts": len([t for t in traces if t.status == ToolExecutionStatus.TIMEOUT.value]),
            "degraded": len([t for t in traces if t.status == ToolExecutionStatus.DEGRADED.value]),
            "partial": len([t for t in traces if t.status == ToolExecutionStatus.PARTIAL.value]),
            "error_rate": (len(failures) / total) if total > 0 else 0.0,
            "p95_latency_ms": latencies[p95_index] if latencies else 0.0,
            "failure_classes": failure_classes,
        }

    def recent_tool_traces(self, *, limit: int = 100) -> list[dict[str, Any]]:
        bounded = max(1, int(limit))
        return [trace.to_dict() for trace in list(self._traces)[-bounded:]]

    def _emit_tool_event(self, event_type: str, **payload: Any) -> None:
        emit = getattr(self._event_tap, "emit", None)
        if not callable(emit):
            return
        run_id = str(payload.pop("run_id", "") or "").strip()
        if not run_id:
            run_id = str(getattr(self._event_tap, "run_id", "") or "").strip()
        if not run_id:
            return
        safe_payload = {
            str(k): (
                v
                if isinstance(v, (str, int, float, bool, type(None)))
                else str(v)
            )
            for k, v in payload.items()
        }
        emit(event_type, run_id=run_id, **safe_payload)

    def execute(
        self,
        tool_name: str,
        payload: dict[str, Any],
        *,
        required_version: str | None = None,
        min_confidence: float = 0.5,
        estimate: ResourceEstimate | None = None,
        fallback_tools: list[str] | None = None,
    ) -> ToolExecutionResult:
        chain = [str(tool_name)] + [str(name) for name in list(fallback_tools or [])]
        failures: list[dict[str, Any]] = []

        for index, candidate in enumerate(chain):
            result = self._execute_single(
                candidate,
                dict(payload),
                required_version=required_version if index == 0 else None,
                estimate=estimate,
            )
            if result.usable(min_confidence=min_confidence):
                result.fallback_used = index > 0
                if failures:
                    result.metadata.setdefault("fallback_failures", failures)
                return result

            # Single-tool execution must preserve native terminal status instead
            # of collapsing into a generic degraded envelope.
            if len(chain) == 1:
                return result

            failures.append(
                {
                    "tool_name": result.tool_name,
                    "status": result.status.value,
                    "error": result.error,
                    "attempts": result.attempts,
                },
            )

        # If all fallback candidates are unusable, return an explicit degraded
        # fallback-exhausted result carrying full prior failure evidence.
        return ToolExecutionResult(
            tool_name=str(chain[-1]).strip().lower() if chain else str(tool_name).strip().lower(),
            status=ToolExecutionStatus.DEGRADED,
            output=None,
            error="all_tool_candidates_unusable",
            attempts=max(
                1,
                sum(int(item.get("attempts", 1)) for item in failures) if failures else 1,
            ),
            confidence=0.0,
            degraded_reason="fallback_chain_exhausted",
            fallback_used=len(chain) > 1,
            metadata={"fallback_failures": failures},
        )

    def execute_for_intent(
        self,
        *,
        intent: str,
        payload: dict[str, Any],
        router: CostAwareToolRouter,
        load_by_tool: dict[str, float] | None = None,
        min_confidence: float = 0.5,
    ) -> ToolExecutionResult:
        selected = router.route(
            self._registry,
            intent=intent,
            load_by_tool=load_by_tool,
        )
        if selected is None:
            return ToolExecutionResult(
                tool_name="",
                status=ToolExecutionStatus.ERROR,
                output=None,
                error=f"no_tool_capability_for_intent:{intent}",
                confidence=0.0,
            )
        return self.execute(
            selected.name,
            payload,
            required_version=selected.version,
            min_confidence=min_confidence,
        )

    @staticmethod
    def _resolve_idempotency_key(tool_name: str, payload: dict[str, Any]) -> str:
        idempotency_key = str(payload.get("_idempotency_key") or "").strip()
        if idempotency_key:
            return idempotency_key
        keyed_payload = dict(payload)
        policy_context = _extract_idempotency_policy_context(payload)
        if policy_context:
            keyed_payload["_policy_context_fingerprint"] = policy_context
        return KernelToolIdempotencyRegistry.deterministic_key(
            tool_name=tool_name,
            payload=keyed_payload,
            scope="external_tool_runtime",
        )

    def _emit_tool_call_end(
        self,
        *,
        tool_name: str,
        idempotency_key: str,
        result: ToolExecutionResult,
    ) -> None:
        self._emit_tool_event(
            "TOOL_CALL_END",
            tool_name=tool_name,
            idempotency_key=idempotency_key,
            status=result.status.value,
            attempts=int(result.attempts),
            latency_ms=float(result.latency_ms),
            error=str(result.error or ""),
        )

    def _emit_tool_call_end_basic(
        self,
        *,
        tool_name: str,
        idempotency_key: str,
        result: ToolExecutionResult,
    ) -> None:
        self._emit_tool_event(
            "TOOL_CALL_END",
            tool_name=tool_name,
            idempotency_key=idempotency_key,
            status=result.status.value,
            error=result.error,
        )

    @staticmethod
    def _attach_idempotency_metadata(result: ToolExecutionResult, *, idempotency_key: str) -> ToolExecutionResult:
        result.metadata["idempotency_key"] = idempotency_key
        return result

    def _idempotent_replay_result(
        self,
        *,
        normalized_name: str,
        cached: ToolExecutionResult,
        idempotency_key: str,
    ) -> ToolExecutionResult:
        replay = ToolExecutionResult(
            tool_name=normalized_name,
            status=ToolExecutionStatus.SKIPPED,
            output=cached.output,
            error="idempotent_replay",
            attempts=0,
            latency_ms=0.0,
            confidence=float(cached.confidence),
            degraded_reason="kernel_idempotency",
            fallback_used=bool(cached.fallback_used),
            metadata=dict(cached.metadata or {}),
        )
        replay.metadata["idempotency_key"] = idempotency_key
        replay.metadata["idempotent_replay"] = True
        self._emit_tool_call_end_basic(
            tool_name=normalized_name,
            idempotency_key=idempotency_key,
            result=replay,
        )
        return replay

    def _skipped_result(
        self,
        *,
        normalized_name: str,
        idempotency_key: str,
        error: str,
        degraded_reason: str,
    ) -> ToolExecutionResult:
        result = ToolExecutionResult(
            tool_name=normalized_name,
            status=ToolExecutionStatus.SKIPPED,
            output=None,
            error=error,
            confidence=0.0,
            degraded_reason=degraded_reason,
        )
        self._attach_idempotency_metadata(result, idempotency_key=idempotency_key)
        self._emit_tool_call_end_basic(
            tool_name=normalized_name,
            idempotency_key=idempotency_key,
            result=result,
        )
        return result

    @staticmethod
    def _should_retry_tool_result(
        result: ToolExecutionResult,
        *,
        attempt: int,
        http_status: Any,
        retry_policy: RetryPolicy,
    ) -> bool:
        failure_kind = classify_network_failure(status_code=http_status)
        return bool(
            result.status in {ToolExecutionStatus.ERROR, ToolExecutionStatus.TIMEOUT}
            and retry_policy.should_retry(
                attempt=attempt,
                status_code=http_status,
                failure_kind=failure_kind,
            )
        )

    def _finalize_tool_result(
        self,
        *,
        normalized_name: str,
        idempotency_key: str,
        result: ToolExecutionResult,
        attempt: int,
        start: float,
    ) -> ToolExecutionResult:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        result.attempts = attempt
        result.latency_ms = round(elapsed_ms, 3)
        result.tool_name = normalized_name
        result = self._apply_latency_spike_model(result)
        result = self._normalize_failure_surface(result)
        self._attach_idempotency_metadata(result, idempotency_key=idempotency_key)
        if result.status in {
            ToolExecutionStatus.OK,
            ToolExecutionStatus.PARTIAL,
            ToolExecutionStatus.DEGRADED,
        }:
            KernelToolIdempotencyRegistry.put(idempotency_key, result)
        self._record_trace(result)
        self._emit_tool_call_end(
            tool_name=normalized_name,
            idempotency_key=idempotency_key,
            result=result,
        )
        return result

    def _build_retry_exhausted_result(
        self,
        *,
        normalized_name: str,
        idempotency_key: str,
        start: float,
    ) -> ToolExecutionResult:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        result = ToolExecutionResult(
            tool_name=normalized_name,
            status=ToolExecutionStatus.TIMEOUT,
            output=None,
            error="retry_exhausted",
            attempts=max(1, int(self._retry_policy.max_attempts)),
            latency_ms=round(elapsed_ms, 3),
            confidence=0.0,
        )
        result = self._normalize_failure_surface(result)
        self._attach_idempotency_metadata(result, idempotency_key=idempotency_key)
        self._record_trace(result)
        self._emit_tool_call_end(
            tool_name=normalized_name,
            idempotency_key=idempotency_key,
            result=result,
        )
        return result

    def _execute_single(
        self,
        tool_name: str,
        payload: dict[str, Any],
        *,
        required_version: str | None,
        estimate: ResourceEstimate | None,
    ) -> ToolExecutionResult:
        normalized_name = str(tool_name).strip().lower()
        idempotency_key = self._resolve_idempotency_key(normalized_name, payload)
        cached = KernelToolIdempotencyRegistry.get(idempotency_key)
        if isinstance(cached, ToolExecutionResult):
            return self._idempotent_replay_result(
                normalized_name=normalized_name,
                cached=cached,
                idempotency_key=idempotency_key,
            )
        self._emit_tool_event(
            "TOOL_CALL_START",
            tool_name=normalized_name,
            idempotency_key=idempotency_key,
        )

        allowed, reason = self._permission_decision(
            self._permission_enforcer,
            normalized_name,
            payload,
        )
        if not allowed:
            return self._skipped_result(
                normalized_name=normalized_name,
                idempotency_key=idempotency_key,
                error=f"permission_denied:{reason}",
                degraded_reason="permission_rejected",
            )

        if self._rate_limiter is not None and not self._rate_limiter.allow():
            return self._skipped_result(
                normalized_name=normalized_name,
                idempotency_key=idempotency_key,
                error="rate_limited",
                degraded_reason="throttled",
            )

        allowed, reason = self._isolation_guard.validate(normalized_name, estimate)
        if not allowed:
            return self._skipped_result(
                normalized_name=normalized_name,
                idempotency_key=idempotency_key,
                error=f"isolation_violation:{reason}",
                degraded_reason="isolation_rejected",
            )

        capability_and_handler = self._registry.handler_for(
            normalized_name,
            required_version=required_version,
        )
        if capability_and_handler is None:
            result = ToolExecutionResult(
                tool_name=normalized_name,
                status=ToolExecutionStatus.ERROR,
                output=None,
                error="tool_not_registered_or_incompatible_version",
                confidence=0.0,
            )
            self._attach_idempotency_metadata(result, idempotency_key=idempotency_key)
            self._emit_tool_call_end_basic(
                tool_name=normalized_name,
                idempotency_key=idempotency_key,
                result=result,
            )
            return result

        capability, handler = capability_and_handler
        start = time.perf_counter()
        return self._execute_with_handler(
            normalized_name=normalized_name,
            idempotency_key=idempotency_key,
            capability=capability,
            handler=handler,
            payload=payload,
            start=start,
        )

    def _execute_with_handler(
        self,
        *,
        normalized_name: str,
        idempotency_key: str,
        capability: ToolCapability,
        handler,
        payload: dict[str, Any],
        start: float,
    ) -> ToolExecutionResult:
        timeout_seconds = payload.get("_timeout_seconds")
        if timeout_seconds is None:
            timeout_seconds = payload.get("timeout_seconds")
        if timeout_seconds is None:
            timeout_seconds = self._hard_timeout_seconds
        resolved_timeout = max(0.01, float(timeout_seconds)) if timeout_seconds is not None else None

        for attempt in range(1, max(1, int(self._retry_policy.max_attempts)) + 1):
            try:
                result = self._run_handler_with_timeout(handler, dict(payload), resolved_timeout)
            except Exception as exc:  # Failure modeled as runtime event, not uncaught exception.  # noqa: BLE001
                failure_kind = classify_network_failure(exc=exc)
                if self._retry_policy.should_retry(attempt=attempt, failure_kind=failure_kind):
                    self._sleeper(self._retry_policy.delay_for_attempt(attempt))
                    continue
                status = (
                    ToolExecutionStatus.TIMEOUT
                    if failure_kind == NetworkFailureKind.TIMEOUT
                    else ToolExecutionStatus.ERROR
                )
                result = ToolExecutionResult(
                    tool_name=normalized_name,
                    status=status,
                    output=None,
                    error=str(exc),
                    attempts=attempt,
                    confidence=0.0,
                    metadata={"failure_kind": failure_kind.value},
                )
                finalized = self._finalize_tool_result(
                    normalized_name=normalized_name,
                    idempotency_key=idempotency_key,
                    result=result,
                    attempt=attempt,
                    start=start,
                )
                self._ingest_tool_result_memory(payload=payload, result=finalized)
                return finalized

            if result.status == ToolExecutionStatus.PARTIAL and not capability.supports_partial:
                result = ToolExecutionResult(
                    tool_name=normalized_name,
                    status=ToolExecutionStatus.DEGRADED,
                    output=result.output,
                    error="partial_output_not_supported_by_capability",
                    attempts=attempt,
                    confidence=min(result.confidence, 0.45),
                    degraded_reason="unsupported_partial",
                    metadata=dict(result.metadata),
                )

            http_status = None
            if isinstance(result.metadata, dict):
                http_status = result.metadata.get("http_status")
            if self._should_retry_tool_result(
                result,
                attempt=attempt,
                http_status=http_status,
                retry_policy=self._retry_policy,
            ):
                self._sleeper(self._retry_policy.delay_for_attempt(attempt))
                continue

            finalized = self._finalize_tool_result(
                normalized_name=normalized_name,
                idempotency_key=idempotency_key,
                result=result,
                attempt=attempt,
                start=start,
            )
            self._ingest_tool_result_memory(payload=payload, result=finalized)
            return finalized

        exhausted = self._build_retry_exhausted_result(
            normalized_name=normalized_name,
            idempotency_key=idempotency_key,
            start=start,
        )
        self._ingest_tool_result_memory(payload=payload, result=exhausted)
        return exhausted
def _parse_version(version: str) -> tuple[int, int, int]:
    """Parse semver-ish strings; non-numeric segments are ignored.

    Examples:
        "1.2.3" -> (1, 2, 3)
        "2.0" -> (2, 0, 0)
        "3" -> (3, 0, 0)

    """
    cleaned = str(version or "0").strip()
    parts = cleaned.split(".")
    values: list[int] = []
    for part in parts[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        values.append(int(digits) if digits else 0)
    while len(values) < 3:
        values.append(0)
    return tuple(values)


__all__ = [
    "ApiKeyAuth",
    "AuthStrategy",
    "BearerTokenAuth",
    "CostAwareToolRouter",
    "DynamicToolRegistry",
    "ExternalToolRuntime",
    "HttpRequest",
    "HttpResponse",
    "HttpTransport",
    "IsolationGuard",
    "IsolationProfile",
    "NetworkFailureKind",
    "NoAuth",
    "RateLimitPolicy",
    "ResourceEstimate",
    "ResourceLimits",
    "RetryPolicy",
    "RouterWeights",
    "SlidingWindowRateLimiter",
    "ToolCapability",
    "ToolExecutionResult",
    "ToolExecutionStatus",
    "classify_network_failure",
]
