"""Integration tests for http_fetch_tool.

These tests use a local mock transport (no real network I/O) to verify:
  - Correct HTTP GET/POST/HEAD round-trips
  - Retry-on-error semantics
  - URL validation and method allowlist
  - Auth header injection
  - Response body parsing (JSON vs plain text)
  - Fallback chain behaviour in ExternalToolRuntime

A separate real-network smoke test class is marked integration so it only
runs in CI lanes that have outbound network access.
"""

from __future__ import annotations

import pytest

from dadbot.core.external_tool_runtime import (
    DynamicToolRegistry,
    ExternalToolRuntime,
    HttpRequest,
    HttpResponse,
    RetryPolicy,
    ToolExecutionStatus,
)
from dadbot.tools.http_fetch_tool import (
    UrllibHttpTransport,
    build_bearer_auth,
    build_http_fetch_tool,
)

pytestmark = pytest.mark.unit


# ── Mock transport ────────────────────────────────────────────────────────────

class _MockTransport:
    """Deterministic transport stub for unit tests."""

    def __init__(self, responses: list[HttpResponse]) -> None:
        self._queue = list(responses)
        self.calls: list[HttpRequest] = []

    def send(self, request: HttpRequest) -> HttpResponse:
        self.calls.append(request)
        if self._queue:
            return self._queue.pop(0)
        return HttpResponse(status_code=500, body={"error": "no_more_responses"})


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestHttpFetchToolBasic:
    def test_successful_get(self):
        transport = _MockTransport([HttpResponse(status_code=200, body={"ok": True}, elapsed_ms=12.0)])
        cap, handler = build_http_fetch_tool(transport=transport, retry_policy=RetryPolicy(max_attempts=1))
        result = handler({"method": "GET", "url": "https://example.com/api"})
        assert result.status == ToolExecutionStatus.OK
        assert result.output["status_code"] == 200
        assert result.output["body"] == {"ok": True}
        assert result.confidence == 1.0
        assert len(transport.calls) == 1
        assert transport.calls[0].method == "GET"
        assert transport.calls[0].url == "https://example.com/api"

    def test_empty_url_returns_error(self):
        transport = _MockTransport([])
        cap, handler = build_http_fetch_tool(transport=transport)
        result = handler({"method": "GET", "url": ""})
        assert result.status == ToolExecutionStatus.ERROR
        assert result.error == "url_required"

    def test_disallowed_method_returns_error(self):
        transport = _MockTransport([])
        cap, handler = build_http_fetch_tool(transport=transport, allow_mutation=False)
        result = handler({"method": "POST", "url": "https://example.com/"})
        assert result.status == ToolExecutionStatus.ERROR
        assert "method_not_allowed" in result.error

    def test_post_allowed_with_mutation_flag(self):
        transport = _MockTransport([HttpResponse(status_code=201, body={"created": True})])
        cap, handler = build_http_fetch_tool(transport=transport, allow_mutation=True)
        result = handler({"method": "POST", "url": "https://api.example.com/items", "json_body": {"x": 1}})
        assert result.status == ToolExecutionStatus.OK
        assert transport.calls[0].method == "POST"
        assert transport.calls[0].json_body == {"x": 1}

    def test_retry_on_server_error(self):
        responses = [
            HttpResponse(status_code=503, body={"error": "unavailable"}),
            HttpResponse(status_code=503, body={"error": "unavailable"}),
            HttpResponse(status_code=200, body={"data": "ok"}),
        ]
        transport = _MockTransport(responses)
        retry = RetryPolicy(max_attempts=3, base_delay_ms=0.0)
        cap, handler = build_http_fetch_tool(transport=transport, retry_policy=retry)
        # Patch sleeper: RetryPolicy.delay_for_attempt will return 0 because base_delay_ms=0
        result = handler({"method": "GET", "url": "https://slow.example.com/"})
        assert result.status == ToolExecutionStatus.OK
        assert result.attempts == 3
        assert len(transport.calls) == 3

    def test_exhausted_retries_returns_error(self):
        transport = _MockTransport([
            HttpResponse(status_code=500),
            HttpResponse(status_code=500),
            HttpResponse(status_code=500),
        ])
        retry = RetryPolicy(max_attempts=3, base_delay_ms=0.0)
        cap, handler = build_http_fetch_tool(transport=transport, retry_policy=retry)
        result = handler({"method": "GET", "url": "https://broken.example.com/"})
        assert result.status == ToolExecutionStatus.ERROR
        assert "http_error:500" in result.error

    def test_bearer_auth_injected_in_headers(self):
        transport = _MockTransport([HttpResponse(status_code=200, body={})])
        auth = build_bearer_auth("my-secret-token")
        cap, handler = build_http_fetch_tool(transport=transport, auth=auth, retry_policy=RetryPolicy(max_attempts=1))
        handler({"method": "GET", "url": "https://secure.example.com/"})
        assert transport.calls[0].headers.get("Authorization") == "Bearer my-secret-token"

    def test_partial_response_returns_partial_status(self):
        transport = _MockTransport([HttpResponse(status_code=206, body={"part": "one"})])
        cap, handler = build_http_fetch_tool(transport=transport, retry_policy=RetryPolicy(max_attempts=1))
        result = handler({"method": "GET", "url": "https://streaming.example.com/"})
        assert result.status == ToolExecutionStatus.PARTIAL


class TestHttpFetchInExternalToolRuntime:
    """Verify the tool plug-in round-trip through DynamicToolRegistry / ExternalToolRuntime."""

    def test_registered_and_executable(self):
        transport = _MockTransport([HttpResponse(status_code=200, body={"value": 42})])
        cap, handler = build_http_fetch_tool(transport=transport, retry_policy=RetryPolicy(max_attempts=1))
        registry = DynamicToolRegistry()
        registry.register(cap, handler)
        runtime = ExternalToolRuntime(registry, sleeper=lambda _: None)
        result = runtime.execute("http_fetch", {"method": "GET", "url": "https://example.com/"})
        assert result.status == ToolExecutionStatus.OK
        assert result.output["body"] == {"value": 42}

    def test_fallback_chain_activates_on_failure(self):
        failing_transport = _MockTransport([HttpResponse(status_code=500)])
        ok_transport = _MockTransport([HttpResponse(status_code=200, body={"fallback": True})])
        cap_v1, handler_v1 = build_http_fetch_tool(transport=failing_transport, retry_policy=RetryPolicy(max_attempts=1))
        # Make a second capability with a different name to act as fallback
        from dadbot.core.external_tool_runtime import ToolCapability
        cap_v2 = ToolCapability(
            name="http_fetch_fallback",
            version="1.0.0",
            intents=("http_fetch_fallback",),
            cost_units=2.0,
            avg_latency_ms=500.0,
        )
        registry = DynamicToolRegistry()
        registry.register(cap_v1, handler_v1)
        cap_fb, handler_fb = build_http_fetch_tool(transport=ok_transport, retry_policy=RetryPolicy(max_attempts=1))
        registry.register(cap_v2, lambda p: handler_fb(p))
        runtime = ExternalToolRuntime(registry, sleeper=lambda _: None)
        result = runtime.execute("http_fetch", {"method": "GET", "url": "https://example.com/"}, fallback_tools=["http_fetch_fallback"])
        assert result.status == ToolExecutionStatus.OK
        assert result.fallback_used is True


class TestUrllibTransportSchemeCheck:
    """Unit tests for UrllibHttpTransport without real network I/O."""

    def test_http_scheme_blocked_by_default(self):
        t = UrllibHttpTransport(allow_http=False)
        resp = t.send(HttpRequest(method="GET", url="http://insecure.example.com/"))
        assert resp.status_code == 400
        assert "http_scheme_not_allowed" in str(resp.body)

    def test_empty_url_returns_400(self):
        t = UrllibHttpTransport()
        resp = t.send(HttpRequest(method="GET", url=""))
        assert resp.status_code == 400


@pytest.mark.integration
class TestHttpFetchRealNetwork:
    """Real outbound network smoke tests — only run in the INTEGRATION lane."""

    def test_get_httpbin_status_200(self):
        """GET a well-known endpoint; expect HTTP 200 and valid JSON body."""
        from dadbot.core.external_tool_runtime import RetryPolicy
        from dadbot.tools.http_fetch_tool import build_http_fetch_tool

        capability, handler = build_http_fetch_tool(
            transport=None,
            auth=None,
            retry_policy=RetryPolicy(max_attempts=1),
            allow_mutation=False,
            allow_http=False,
            max_response_bytes=65536,
            timeout_seconds=10,
        )
        result = handler({"method": "GET", "url": "https://httpbin.org/get"})
        assert result.status == ToolExecutionStatus.OK, f"Expected OK, got {result.status}: {result.error}"
        body = result.output or {}
        assert "url" in body or "headers" in body

    def test_https_required_rejects_http_url(self):
        """Plaintext HTTP URL must be rejected without a real network call."""
        from dadbot.tools.http_fetch_tool import build_http_fetch_tool

        capability, handler = build_http_fetch_tool(allow_http=False)
        result = handler({"method": "GET", "url": "http://httpbin.org/get"})
        assert result.status == ToolExecutionStatus.ERROR
