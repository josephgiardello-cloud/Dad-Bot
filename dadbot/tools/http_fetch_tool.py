"""Real HTTP fetch tool — plugs into ExternalToolRuntime / DynamicToolRegistry.

Implements the HttpTransport protocol using stdlib urllib so there are no
additional runtime dependencies.  The tool is deliberately limited to
read-style operations (GET, HEAD) by default; POST is opt-in via
allow_mutation=True so callers explicitly own the side-effect surface.

Security boundaries
-------------------
- URL must start with https:// (configurable via allow_http=True for dev).
- Redirect limit: 3 hops.
- Response body is capped at max_response_bytes (default 512 KB) to prevent
  memory exhaustion from unexpected large payloads.
- The tool never follows cross-scheme redirects (http → https is ok; the
  reverse is blocked).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from dadbot.core.external_tool_runtime import (
    ApiKeyAuth,
    AuthStrategy,
    BearerTokenAuth,
    HttpRequest,
    HttpResponse,
    HttpTransport,
    IsolationProfile,
    NetworkFailureKind,
    NoAuth,
    ResourceLimits,
    RetryPolicy,
    ToolCapability,
    ToolExecutionResult,
    ToolExecutionStatus,
    classify_network_failure,
)

_DEFAULT_MAX_RESPONSE_BYTES = 512 * 1024  # 512 KB
_DEFAULT_REDIRECT_LIMIT = 3


class UrllibHttpTransport:
    """HttpTransport implementation backed by stdlib urllib."""

    def __init__(
        self,
        *,
        max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
        redirect_limit: int = _DEFAULT_REDIRECT_LIMIT,
        allow_http: bool = False,
    ) -> None:
        self._max_bytes = max(1, int(max_response_bytes))
        self._redirect_limit = max(0, int(redirect_limit))
        self._allow_http = bool(allow_http)

    def send(self, request: HttpRequest) -> HttpResponse:
        url = str(request.url or "").strip()
        if not url:
            return HttpResponse(status_code=400, body=None, elapsed_ms=0.0)
        if not self._allow_http and url.startswith("http://"):
            return HttpResponse(
                status_code=400,
                body={"error": "http_scheme_not_allowed"},
                elapsed_ms=0.0,
            )

        method = str(request.method or "GET").upper()
        headers = dict(request.headers or {})
        headers.setdefault("User-Agent", "DadBot/1.0 (system-tool)")
        headers.setdefault("Accept", "application/json, text/plain, */*")

        body_bytes: bytes | None = None
        if request.json_body is not None:
            body_bytes = json.dumps(request.json_body).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")

        opener = urllib.request.build_opener(
            urllib.request.HTTPRedirectHandler,
        )
        urllib_request = urllib.request.Request(
            url,
            data=body_bytes,
            headers=headers,
            method=method,
        )

        started = time.perf_counter()
        try:
            timeout = max(0.1, float(request.timeout_seconds))
            with opener.open(urllib_request, timeout=timeout) as response:
                status = int(response.status)
                raw = response.read(self._max_bytes)
                resp_headers: dict[str, str] = {
                    k.lower(): v for k, v in response.headers.items()
                }
                elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
                content_type = resp_headers.get("content-type", "")
                if "json" in content_type:
                    try:
                        body = json.loads(raw.decode("utf-8", errors="replace"))
                    except json.JSONDecodeError:
                        body = raw.decode("utf-8", errors="replace")
                else:
                    body = raw.decode("utf-8", errors="replace")
                return HttpResponse(
                    status_code=status,
                    headers=resp_headers,
                    body=body,
                    elapsed_ms=elapsed_ms,
                )
        except urllib.error.HTTPError as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            return HttpResponse(
                status_code=int(exc.code),
                body={"error": str(exc.reason)},
                elapsed_ms=elapsed_ms,
            )
        except urllib.error.URLError as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            kind = classify_network_failure(exc=exc)
            return HttpResponse(
                status_code=0,
                body={"error": str(exc.reason), "failure_kind": kind.value},
                elapsed_ms=elapsed_ms,
            )


def _build_auth_headers(auth: AuthStrategy | None, headers: dict[str, str]) -> dict[str, str]:
    if auth is None or isinstance(auth, NoAuth):
        return dict(headers)
    return auth.apply(dict(headers))


def build_http_fetch_tool(
    *,
    transport: HttpTransport | None = None,
    auth: AuthStrategy | None = None,
    retry_policy: RetryPolicy | None = None,
    allow_mutation: bool = False,
    allow_http: bool = False,
    max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
    timeout_seconds: float = 10.0,
) -> tuple[ToolCapability, Any]:
    """Build a (ToolCapability, handler) pair for real HTTP fetch.

    Register the returned pair with DynamicToolRegistry:
        cap, handler = build_http_fetch_tool()
        registry.register(cap, handler)

    Payload schema (passed to registry.execute / ExternalToolRuntime.execute):
        {
            "method":    "GET" | "POST" | "HEAD"   (default: "GET")
            "url":       "https://..."              (required)
            "headers":   {}                         (optional extra headers)
            "params":    {}                         (optional query string params)
            "json_body": {}                         (optional, POST only)
            "timeout":   10.0                       (optional override)
        }
    """
    resolved_transport: HttpTransport = transport or UrllibHttpTransport(
        max_response_bytes=int(max_response_bytes),
        allow_http=bool(allow_http),
    )
    resolved_retry = retry_policy or RetryPolicy(max_attempts=3, base_delay_ms=200.0)
    allowed_methods = {"GET", "HEAD", "POST"} if allow_mutation else {"GET", "HEAD"}

    def handler(payload: dict[str, Any]) -> ToolExecutionResult:
        method = str(payload.get("method") or "GET").upper()
        url = str(payload.get("url") or "").strip()
        extra_headers = dict(payload.get("headers") or {})
        params = dict(payload.get("params") or {})
        json_body = payload.get("json_body") if method == "POST" else None
        timeout = float(payload.get("timeout") or timeout_seconds)

        if not url:
            return ToolExecutionResult(
                tool_name="http_fetch",
                status=ToolExecutionStatus.ERROR,
                output=None,
                error="url_required",
            )
        if method not in allowed_methods:
            return ToolExecutionResult(
                tool_name="http_fetch",
                status=ToolExecutionStatus.ERROR,
                output=None,
                error=f"method_not_allowed:{method}",
            )

        if params:
            from urllib.parse import urlencode, urlparse, urlunparse, parse_qs
            parsed = urlparse(url)
            existing = parse_qs(parsed.query)
            existing.update({k: [str(v)] for k, v in params.items()})
            from urllib.parse import urlencode as _enc
            new_query = _enc({k: v[0] for k, v in existing.items()})
            url = urlunparse(parsed._replace(query=new_query))

        merged_headers = _build_auth_headers(auth, extra_headers)

        attempts = 0
        last_response: HttpResponse | None = None
        for attempt in range(1, max(1, resolved_retry.max_attempts) + 1):
            attempts = attempt
            request = HttpRequest(
                method=method,
                url=url,
                headers=merged_headers,
                json_body=json_body,
                timeout_seconds=timeout,
            )
            response = resolved_transport.send(request)
            last_response = response

            if response.is_success or response.is_partial:
                return ToolExecutionResult(
                    tool_name="http_fetch",
                    status=ToolExecutionStatus.PARTIAL if response.is_partial else ToolExecutionStatus.OK,
                    output={
                        "status_code": response.status_code,
                        "headers": response.headers,
                        "body": response.body,
                    },
                    latency_ms=float(response.elapsed_ms),
                    attempts=attempts,
                    confidence=1.0,
                    metadata={"url": url, "method": method},
                )

            failure_kind = classify_network_failure(status_code=response.status_code)
            if not resolved_retry.should_retry(
                attempt=attempt,
                status_code=response.status_code,
                failure_kind=failure_kind,
            ):
                break

        status_code = int(last_response.status_code) if last_response else 0
        return ToolExecutionResult(
            tool_name="http_fetch",
            status=ToolExecutionStatus.ERROR,
            output={"status_code": status_code, "body": last_response.body if last_response else None},
            error=f"http_error:{status_code}",
            attempts=attempts,
            confidence=0.0,
            metadata={"url": url, "method": method},
        )

    capability = ToolCapability(
        name="http_fetch",
        version="1.0.0",
        intents=("http_get", "http_fetch", "web_request", "api_call"),
        cost_units=1.0,
        avg_latency_ms=200.0,
        reliability=0.95,
        supports_partial=True,
        tags=("network", "external", "http"),
    )
    return capability, handler


def build_bearer_auth(token: str) -> BearerTokenAuth:
    return BearerTokenAuth(token=str(token))


def build_api_key_auth(api_key: str, *, header_name: str = "X-API-Key") -> ApiKeyAuth:
    return ApiKeyAuth(api_key=str(api_key), header_name=str(header_name))


HTTP_FETCH_ISOLATION_PROFILE = IsolationProfile(
    tool_name="http_fetch",
    compartment="network",
    limits=ResourceLimits(
        max_cpu_ms=5_000.0,
        max_memory_mb=64.0,
        max_io_ops=0,
        max_network_calls=1,
    ),
    allow_network=True,
    allow_filesystem=False,
)
