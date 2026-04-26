from __future__ import annotations

from dataclasses import dataclass

from dadbot.core.external_tool_runtime import (
    CostAwareToolRouter,
    DynamicToolRegistry,
    ExternalToolRuntime,
    IsolationGuard,
    IsolationProfile,
    NetworkFailureKind,
    RateLimitPolicy,
    ResourceEstimate,
    ResourceLimits,
    RetryPolicy,
    SlidingWindowRateLimiter,
    ToolCapability,
    ToolExecutionResult,
    ToolExecutionStatus,
    classify_network_failure,
)


@dataclass
class FakeClock:
    now: float = 0.0

    def tick(self, seconds: float) -> None:
        self.now += float(seconds)

    def __call__(self) -> float:
        return self.now


def _ok_result(tool_name: str, payload: dict) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_name=tool_name,
        status=ToolExecutionStatus.OK,
        output={"echo": dict(payload)},
        confidence=0.95,
    )


def _register_tool(
    registry: DynamicToolRegistry,
    *,
    name: str,
    version: str,
    intent: str,
    cost: float,
    latency: float,
    reliability: float,
    supports_partial: bool = True,
    handler=None,
) -> None:
    capability = ToolCapability(
        name=name,
        version=version,
        intents=(intent,),
        cost_units=cost,
        avg_latency_ms=latency,
        reliability=reliability,
        supports_partial=supports_partial,
        tags=("external",),
    )
    registry.register(capability, handler or (lambda payload: _ok_result(name, payload)))


def test_retry_policy_recovers_from_transient_exception():
    registry = DynamicToolRegistry()
    calls = {"count": 0}

    def flaky(payload: dict) -> ToolExecutionResult:
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimeoutError("upstream timeout")
        return ToolExecutionResult(
            tool_name="weather_api",
            status=ToolExecutionStatus.OK,
            output={"temp": 72},
            confidence=0.9,
        )

    _register_tool(
        registry,
        name="weather_api",
        version="1.0.0",
        intent="weather_lookup",
        cost=1.0,
        latency=120,
        reliability=0.8,
        handler=flaky,
    )

    runtime = ExternalToolRuntime(
        registry,
        retry_policy=RetryPolicy(max_attempts=3, base_delay_ms=1.0, jitter_ratio=0.0),
        sleeper=lambda _seconds: None,
    )

    result = runtime.execute("weather_api", {"city": "austin"})

    assert result.status == ToolExecutionStatus.OK
    assert result.attempts == 2
    assert calls["count"] == 2


def test_retry_policy_does_not_retry_non_retryable_client_error():
    registry = DynamicToolRegistry()
    calls = {"count": 0}

    def bad_request(_payload: dict) -> ToolExecutionResult:
        calls["count"] += 1
        return ToolExecutionResult(
            tool_name="crm_api",
            status=ToolExecutionStatus.ERROR,
            output=None,
            error="bad request",
            confidence=0.0,
            metadata={"http_status": 400},
        )

    _register_tool(
        registry,
        name="crm_api",
        version="1.2.0",
        intent="contact_lookup",
        cost=1.0,
        latency=80,
        reliability=0.95,
        handler=bad_request,
    )

    runtime = ExternalToolRuntime(
        registry,
        retry_policy=RetryPolicy(max_attempts=4, base_delay_ms=1.0, jitter_ratio=0.0),
        sleeper=lambda _seconds: None,
    )
    result = runtime.execute("crm_api", {"email": "x@y.com"})

    assert result.status == ToolExecutionStatus.ERROR
    assert result.attempts == 1
    assert calls["count"] == 1


def test_rate_limiter_throttles_excess_invocations():
    registry = DynamicToolRegistry()
    _register_tool(
        registry,
        name="calendar_api",
        version="1.0.0",
        intent="calendar_lookup",
        cost=1.0,
        latency=50,
        reliability=0.9,
    )

    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(RateLimitPolicy(max_requests=2, window_seconds=5.0), clock=clock)
    runtime = ExternalToolRuntime(registry, rate_limiter=limiter)

    first = runtime.execute("calendar_api", {"day": "monday"})
    second = runtime.execute("calendar_api", {"day": "tuesday"})
    third = runtime.execute("calendar_api", {"day": "wednesday"})

    assert first.status == ToolExecutionStatus.OK
    assert second.status == ToolExecutionStatus.OK
    assert third.status == ToolExecutionStatus.SKIPPED
    assert "rate_limited" in third.error

    clock.tick(6.0)
    fourth = runtime.execute("calendar_api", {"day": "thursday"})
    assert fourth.status == ToolExecutionStatus.OK


def test_isolation_guard_rejects_over_budget_execution():
    registry = DynamicToolRegistry()
    _register_tool(
        registry,
        name="reporting_api",
        version="2.0.0",
        intent="report_generate",
        cost=2.0,
        latency=220,
        reliability=0.88,
    )

    guard = IsolationGuard()
    guard.register(
        IsolationProfile(
            tool_name="reporting_api",
            compartment="external/reporting",
            limits=ResourceLimits(max_cpu_ms=100.0, max_memory_mb=128.0, max_io_ops=50, max_network_calls=2),
        )
    )

    runtime = ExternalToolRuntime(registry, isolation_guard=guard)
    result = runtime.execute(
        "reporting_api",
        {"range": "last_30_days"},
        estimate=ResourceEstimate(cpu_ms=150.0, memory_mb=64.0, io_ops=10, network_calls=1),
    )

    assert result.status == ToolExecutionStatus.SKIPPED
    assert "isolation_violation:cpu_limit_exceeded" in result.error


def test_fallback_chain_used_when_primary_partial_is_unusable():
    registry = DynamicToolRegistry()

    def weak_partial(_payload: dict) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_name="primary_search",
            status=ToolExecutionStatus.PARTIAL,
            output={"rows": []},
            confidence=0.2,
            degraded_reason="sparse_index",
        )

    def fallback_ok(_payload: dict) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_name="fallback_search",
            status=ToolExecutionStatus.OK,
            output={"rows": [{"id": 1}]},
            confidence=0.91,
        )

    _register_tool(
        registry,
        name="primary_search",
        version="1.0.0",
        intent="search",
        cost=0.8,
        latency=40,
        reliability=0.7,
        handler=weak_partial,
    )
    _register_tool(
        registry,
        name="fallback_search",
        version="1.1.0",
        intent="search",
        cost=1.1,
        latency=90,
        reliability=0.92,
        handler=fallback_ok,
    )

    runtime = ExternalToolRuntime(registry)
    result = runtime.execute(
        "primary_search",
        {"q": "dadbot"},
        min_confidence=0.5,
        fallback_tools=["fallback_search"],
    )

    assert result.status == ToolExecutionStatus.OK
    assert result.fallback_used is True
    assert len(result.metadata.get("fallback_failures", [])) == 1


def test_dynamic_registry_version_negotiation_uses_compatible_major():
    registry = DynamicToolRegistry()
    _register_tool(
        registry,
        name="geo_api",
        version="1.2.0",
        intent="geo_lookup",
        cost=1.0,
        latency=80,
        reliability=0.8,
    )
    _register_tool(
        registry,
        name="geo_api",
        version="2.1.0",
        intent="geo_lookup",
        cost=1.3,
        latency=70,
        reliability=0.85,
    )

    cap_v1 = registry.negotiate("geo_api", required_version="1.0.0")
    cap_v2 = registry.negotiate("geo_api", required_version="2.0.0")

    assert cap_v1 is not None
    assert cap_v1.version.startswith("1.")
    assert cap_v2 is not None
    assert cap_v2.version.startswith("2.")


def test_cost_aware_router_prefers_better_reliability_latency_cost_tradeoff():
    registry = DynamicToolRegistry()
    _register_tool(
        registry,
        name="search_fast_low_quality",
        version="1.0.0",
        intent="search",
        cost=0.3,
        latency=20,
        reliability=0.45,
    )
    _register_tool(
        registry,
        name="search_balanced",
        version="1.0.0",
        intent="search",
        cost=0.8,
        latency=60,
        reliability=0.93,
    )

    router = CostAwareToolRouter()
    selected = router.route(registry, intent="search", load_by_tool={"search_balanced": 0.1})

    assert selected is not None
    assert selected.name == "search_balanced"


def test_execute_for_intent_uses_router_selected_tool():
    registry = DynamicToolRegistry()

    _register_tool(
        registry,
        name="weather_primary",
        version="1.0.0",
        intent="weather_lookup",
        cost=1.2,
        latency=100,
        reliability=0.9,
    )
    _register_tool(
        registry,
        name="weather_backup",
        version="1.0.0",
        intent="weather_lookup",
        cost=2.0,
        latency=150,
        reliability=0.8,
    )

    runtime = ExternalToolRuntime(registry)
    router = CostAwareToolRouter()
    result = runtime.execute_for_intent(intent="weather_lookup", payload={"city": "Austin"}, router=router)

    assert result.status == ToolExecutionStatus.OK
    assert result.tool_name == "weather_primary"


def test_network_failure_classifier_maps_statuses_and_exceptions():
    assert classify_network_failure(status_code=429) == NetworkFailureKind.RATE_LIMIT
    assert classify_network_failure(status_code=503) == NetworkFailureKind.SERVER
    assert classify_network_failure(status_code=404) == NetworkFailureKind.CLIENT
    assert classify_network_failure(exc=TimeoutError("timed out")) == NetworkFailureKind.TIMEOUT
    assert classify_network_failure(exc=ConnectionError("connection reset")) == NetworkFailureKind.CONNECTION


def test_retry_policy_delay_increases_monotonically_without_jitter():
    policy = RetryPolicy(base_delay_ms=10.0, backoff_factor=2.0, jitter_ratio=0.0, max_delay_ms=1_000.0)
    delays = [policy.delay_for_attempt(1), policy.delay_for_attempt(2), policy.delay_for_attempt(3)]
    assert delays[0] < delays[1] < delays[2]


def test_partial_output_degraded_when_capability_disables_partial():
    registry = DynamicToolRegistry()

    def partial_handler(_payload: dict) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_name="strict_tool",
            status=ToolExecutionStatus.PARTIAL,
            output={"items": [1]},
            confidence=0.9,
        )

    _register_tool(
        registry,
        name="strict_tool",
        version="1.0.0",
        intent="strict_intent",
        cost=1.0,
        latency=100,
        reliability=0.99,
        supports_partial=False,
        handler=partial_handler,
    )

    runtime = ExternalToolRuntime(registry)
    result = runtime.execute("strict_tool", {"x": 1}, min_confidence=0.1)

    assert result.status == ToolExecutionStatus.DEGRADED
    assert result.degraded_reason == "unsupported_partial"


def test_fallback_chain_exhaustion_returns_degraded_result():
    registry = DynamicToolRegistry()

    def always_fail(_payload: dict) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_name="broken",
            status=ToolExecutionStatus.ERROR,
            output=None,
            error="upstream_500",
            confidence=0.0,
            metadata={"http_status": 500},
        )

    _register_tool(
        registry,
        name="broken_a",
        version="1.0.0",
        intent="lookup",
        cost=1.0,
        latency=50,
        reliability=0.1,
        handler=always_fail,
    )
    _register_tool(
        registry,
        name="broken_b",
        version="1.0.0",
        intent="lookup",
        cost=1.0,
        latency=50,
        reliability=0.1,
        handler=always_fail,
    )

    runtime = ExternalToolRuntime(
        registry,
        retry_policy=RetryPolicy(max_attempts=1),
        sleeper=lambda _seconds: None,
    )
    result = runtime.execute("broken_a", {"k": "v"}, fallback_tools=["broken_b"])

    assert result.status == ToolExecutionStatus.DEGRADED
    assert result.degraded_reason == "fallback_chain_exhausted"
    assert result.fallback_used is True
    assert len(result.metadata.get("fallback_failures", [])) == 2
