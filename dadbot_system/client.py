from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from .contracts import DEFAULT_TENANT_ID, normalize_tenant_id


def env_flag(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default


def repo_root_path() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class ServiceClientConfig:
    base_url: str = field(default_factory=lambda: os.environ.get("DADBOT_API_URL", "http://127.0.0.1:8010"))
    startup_timeout_seconds: float = 25.0
    task_timeout_seconds: float = 90.0
    poll_interval_seconds: float = 0.25
    auto_start_local: bool = field(default_factory=lambda: env_flag("DADBOT_AUTO_START_API", True))
    worker_count: int = field(default_factory=lambda: max(1, env_int("DADBOT_API_WORKERS", 2)))
    redis_url: str = field(default_factory=lambda: str(os.environ.get("DADBOT_REDIS_URL") or "").strip())
    postgres_dsn: str = field(default_factory=lambda: str(os.environ.get("DADBOT_POSTGRES_DSN") or "").strip())
    tenant_id: str = field(
        default_factory=lambda: normalize_tenant_id(os.environ.get("DADBOT_TENANT_ID") or DEFAULT_TENANT_ID)
    )
    otel_enabled: bool = field(default_factory=lambda: env_flag("DADBOT_OTEL_ENABLED", False))
    python_executable: str = field(default_factory=lambda: sys.executable)
    script_path: str = field(default_factory=lambda: str(repo_root_path() / "Dad.py"))


@dataclass(slots=True)
class ServiceChatResult:
    task_id: str
    request_id: str
    reply: str
    should_end: bool
    active_model: str
    session_state: dict[str, Any]
    task_payload: dict[str, Any]
    response_payload: dict[str, Any]


class DadServiceClient:
    def __init__(self, config: ServiceClientConfig | None = None):
        self.config = config or ServiceClientConfig()
        self._spawned_process = None

    @property
    def base_url(self) -> str:
        return self.config.base_url.rstrip("/")

    def _build_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def session_event_stream_url(self, session_id: str, *, tenant_id: str = "") -> str:
        resolved_tenant_id = normalize_tenant_id(tenant_id or self.config.tenant_id)
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        base_path = parsed.path.rstrip("/")
        query = urlencode({"tenant_id": resolved_tenant_id})
        return f"{scheme}://{parsed.netloc}{base_path}/sessions/{session_id}/events/stream?{query}"

    def stream_url(self, session_id: str, *, tenant_id: str = "") -> str:
        return self.session_event_stream_url(session_id, tenant_id=tenant_id)

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 10.0
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(self._build_url(path), data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Dad Bot API returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Dad Bot API is unavailable: {exc}") from exc

        if not raw.strip():
            return {}
        return json.loads(raw)

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health", timeout=3.0)

    def is_healthy(self) -> bool:
        try:
            payload = self.health()
        except RuntimeError:
            return False
        return payload.get("status") == "ok"

    def _default_host_port(self) -> tuple[str, int]:
        parsed = urlparse(self.base_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return host, port

    def _port_is_open(self) -> bool:
        host, port = self._default_host_port()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.75)
            return sock.connect_ex((host, port)) == 0

    def build_local_service_command(self, preferred_model: str = "") -> list[str]:
        host, port = self._default_host_port()
        command = [
            self.config.python_executable,
            self.config.script_path,
            "--serve-api",
            "--api-host",
            host,
            "--api-port",
            str(port),
            "--worker-count",
            str(max(1, self.config.worker_count)),
        ]
        if preferred_model:
            command.extend(["--model", preferred_model])
        if self.config.redis_url:
            command.extend(["--redis-url", self.config.redis_url])
        if self.config.postgres_dsn:
            command.extend(["--postgres-dsn", self.config.postgres_dsn])
        if self.config.otel_enabled:
            command.append("--otel")
        return command

    def start_local_service(self, preferred_model: str = "") -> None:
        if not self.config.auto_start_local:
            raise RuntimeError("Dad Bot API is not healthy and auto-start is disabled")
        command = self.build_local_service_command(preferred_model=preferred_model)

        creationflags = 0
        for flag_name in ("CREATE_NEW_PROCESS_GROUP", "DETACHED_PROCESS", "CREATE_NO_WINDOW"):
            creationflags |= int(getattr(subprocess, flag_name, 0))

        self._spawned_process = subprocess.Popen(
            command,
            cwd=str(repo_root_path()),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )

    def ensure_service_running(self, preferred_model: str = "") -> dict[str, Any]:
        if self.is_healthy():
            return self.health()

        if not self._port_is_open():
            self.start_local_service(preferred_model=preferred_model)

        deadline = time.monotonic() + self.config.startup_timeout_seconds
        last_error = None
        while time.monotonic() < deadline:
            try:
                payload = self.health()
                if payload.get("status") == "ok":
                    return payload
            except RuntimeError as exc:
                last_error = exc
            time.sleep(self.config.poll_interval_seconds)

        if last_error is not None:
            raise RuntimeError(f"Dad Bot API did not become healthy: {last_error}")
        raise RuntimeError("Dad Bot API did not become healthy before timeout")

    def submit_chat(
        self,
        session_id: str,
        *,
        user_input: str,
        attachments: list[dict[str, Any]] | None = None,
        requested_model: str = "",
        tenant_id: str = "",
    ) -> dict[str, Any]:
        self.ensure_service_running(preferred_model=requested_model)
        resolved_tenant_id = normalize_tenant_id(tenant_id or self.config.tenant_id)
        payload = {
            "user_input": user_input,
            "attachments": list(attachments or []),
            "requested_model": requested_model,
            "tenant_id": resolved_tenant_id,
        }
        return self._request("POST", f"/sessions/{session_id}/chat", payload=payload, timeout=15.0)

    def submit_stream(
        self,
        session_id: str,
        *,
        user_input: str,
        attachments: list[dict[str, Any]] | None = None,
        requested_model: str = "",
        tenant_id: str = "",
    ) -> dict[str, Any]:
        self.ensure_service_running(preferred_model=requested_model)
        resolved_tenant_id = normalize_tenant_id(tenant_id or self.config.tenant_id)
        payload = {
            "user_input": user_input,
            "attachments": list(attachments or []),
            "requested_model": requested_model,
            "tenant_id": resolved_tenant_id,
        }
        return self._request("POST", f"/sessions/{session_id}/stream", payload=payload, timeout=15.0)

    def turn(
        self,
        session_id: str,
        *,
        user_input: str,
        attachments: list[dict[str, Any]] | None = None,
        requested_model: str = "",
        tenant_id: str = "",
        timeout_seconds: float = 60.0,
    ) -> ServiceChatResult:
        self.ensure_service_running(preferred_model=requested_model)
        resolved_tenant_id = normalize_tenant_id(tenant_id or self.config.tenant_id)
        payload = {
            "user_input": user_input,
            "attachments": list(attachments or []),
            "requested_model": requested_model,
            "tenant_id": resolved_tenant_id,
            "timeout_seconds": float(timeout_seconds),
        }
        result_payload = self._request(
            "POST", f"/sessions/{session_id}/turn", payload=payload, timeout=max(15.0, float(timeout_seconds) + 5.0)
        )

        task = dict(result_payload.get("task") or {})
        response = dict(result_payload.get("response") or {})
        if not response:
            error = str(task.get("error") or "Dad Bot API turn did not return a response")
            raise RuntimeError(error)

        return ServiceChatResult(
            task_id=str(task.get("task_id") or ""),
            request_id=str(response.get("request_id") or task.get("request_id") or ""),
            reply=str(response.get("reply") or ""),
            should_end=bool(response.get("should_end")),
            active_model=str(response.get("active_model") or ""),
            session_state=dict(task.get("session_state") or {}),
            task_payload=task,
            response_payload=response,
        )

    def session_events(self, session_id: str, *, tenant_id: str = "") -> list[dict[str, Any]]:
        resolved_tenant_id = normalize_tenant_id(tenant_id or self.config.tenant_id)
        result = self._request(
            "GET", f"/sessions/{session_id}/events?{urlencode({'tenant_id': resolved_tenant_id})}", timeout=15.0
        )
        if isinstance(result, list):
            return [r for r in result if isinstance(r, dict)]
        return [result] if isinstance(result, dict) else []

    def replay(
        self, session_id: str, *, tenant_id: str = "", event_type: str = "", since_event_id: str = "", limit: int = 200
    ) -> dict[str, Any]:
        resolved_tenant_id = normalize_tenant_id(tenant_id or self.config.tenant_id)
        query = urlencode(
            {
                "tenant_id": resolved_tenant_id,
                "event_type": str(event_type or ""),
                "since_event_id": str(since_event_id or ""),
                "limit": max(1, int(limit or 200)),
            }
        )
        return self._request("GET", f"/sessions/{session_id}/replay?{query}", timeout=15.0)

    def task_status(self, task_id: str) -> dict[str, Any]:
        return self._request("GET", f"/tasks/{task_id}", timeout=15.0)

    def wait_for_chat_result(self, task_id: str) -> ServiceChatResult:
        deadline = time.monotonic() + self.config.task_timeout_seconds
        last_status = None
        while time.monotonic() < deadline:
            payload = self.task_status(task_id)
            task = dict(payload.get("task") or {})
            response = dict(payload.get("response") or {})
            status = str(task.get("status") or "").strip().lower()
            last_status = status or last_status
            if response:
                return ServiceChatResult(
                    task_id=str(task.get("task_id") or task_id),
                    request_id=str(response.get("request_id") or task.get("request_id") or ""),
                    reply=str(response.get("reply") or ""),
                    should_end=bool(response.get("should_end")),
                    active_model=str(response.get("active_model") or ""),
                    session_state=dict(task.get("session_state") or {}),
                    task_payload=task,
                    response_payload=response,
                )
            if status == "failed":
                raise RuntimeError(str(task.get("error") or "Dad Bot API task failed"))
            time.sleep(self.config.poll_interval_seconds)

        raise TimeoutError(
            f"Dad Bot API task {task_id} timed out after {self.config.task_timeout_seconds:.1f}s (last status: {last_status})"
        )

    def chat(
        self,
        session_id: str,
        *,
        user_input: str,
        attachments: list[dict[str, Any]] | None = None,
        requested_model: str = "",
        tenant_id: str = "",
    ) -> ServiceChatResult:
        task_payload = self.submit_chat(
            session_id,
            user_input=user_input,
            attachments=attachments,
            requested_model=requested_model,
            tenant_id=tenant_id,
        )
        return self.wait_for_chat_result(str(task_payload.get("task_id") or ""))
