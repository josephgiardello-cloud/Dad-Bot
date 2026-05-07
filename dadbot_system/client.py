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
from .security import ServiceTokenManager


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
    api_version_prefix: str = field(default_factory=lambda: str(os.environ.get("DADBOT_API_VERSION_PREFIX") or "/v1"))
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
    auth_token: str = field(default_factory=lambda: str(os.environ.get("DADBOT_API_AUTH_TOKEN") or "").strip())
    token_secret: str = field(default_factory=lambda: str(os.environ.get("DADBOT_API_TOKEN_SECRET") or "").strip())
    token_issuer: str = field(default_factory=lambda: str(os.environ.get("DADBOT_API_TOKEN_ISSUER") or "dadbot"))
    token_subject: str = field(default_factory=lambda: str(os.environ.get("DADBOT_API_TOKEN_SUBJECT") or "dadbot-client"))
    token_ttl_seconds: int = field(default_factory=lambda: env_int("DADBOT_API_TOKEN_TTL_SECONDS", 3600))
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
        self._session_snapshot_history: dict[str, list[dict[str, Any]]] = {}

    @property
    def base_url(self) -> str:
        return self.config.base_url.rstrip("/")

    def _build_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _api_path(self, path: str) -> str:
        normalized_prefix = "/" + str(self.config.api_version_prefix or "").strip("/")
        normalized_path = "/" + str(path or "").lstrip("/")
        if normalized_prefix == "/":
            return normalized_path
        if normalized_path == normalized_prefix or normalized_path.startswith(normalized_prefix + "/"):
            return normalized_path
        return f"{normalized_prefix}{normalized_path}"

    def _auth_token(self, *, tenant_id: str) -> str:
        if self.config.auth_token:
            return self.config.auth_token
        if not self.config.token_secret:
            return ""
        manager = ServiceTokenManager(self.config.token_secret, issuer=self.config.token_issuer)
        return manager.issue(
            subject=self.config.token_subject,
            tenant_id=tenant_id,
            scopes={"read", "write", "admin"},
            allowed_tools=[],
            ttl_seconds=float(self.config.token_ttl_seconds),
        )

    def session_event_stream_url(self, session_id: str, *, tenant_id: str = "") -> str:
        resolved_tenant_id = normalize_tenant_id(tenant_id or self.config.tenant_id)
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        base_path = parsed.path.rstrip("/")
        query = urlencode({"access_token": self._auth_token(tenant_id=resolved_tenant_id)})
        return f"{scheme}://{parsed.netloc}{base_path}{self._api_path(f'/sessions/{session_id}/events/stream')}?{query}"

    def stream_url(self, session_id: str, *, tenant_id: str = "") -> str:
        return self.session_event_stream_url(session_id, tenant_id=tenant_id)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: float = 10.0,
        auth_tenant_id: str = "",
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        resolved_tenant_id = normalize_tenant_id(
            str(auth_tenant_id or (payload or {}).get("tenant_id") or self.config.tenant_id or DEFAULT_TENANT_ID)
        )
        auth_token = self._auth_token(tenant_id=resolved_tenant_id)
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(self._build_url(self._api_path(path)), data=body, headers=headers, method=method)
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

        self._record_session_snapshot(
            str(task.get("session_id") or session_id),
            dict(task.get("session_state") or {}),
            source="turn",
        )

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

    def task_status(self, task_id: str, *, tenant_id: str = "") -> dict[str, Any]:
        return self._request(
            "GET",
            f"/tasks/{task_id}",
            timeout=15.0,
            auth_tenant_id=tenant_id,
        )

    def wait_for_chat_result(self, task_id: str, *, tenant_id: str = "") -> ServiceChatResult:
        deadline = time.monotonic() + self.config.task_timeout_seconds
        last_status = None
        while time.monotonic() < deadline:
            payload = self.task_status(task_id, tenant_id=tenant_id)
            task = dict(payload.get("task") or {})
            response = dict(payload.get("response") or {})
            status = str(task.get("status") or "").strip().lower()
            last_status = status or last_status
            if response:
                self._record_session_snapshot(
                    str(task.get("session_id") or ""),
                    dict(task.get("session_state") or {}),
                    source="chat",
                )
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
        resolved_tenant_id = normalize_tenant_id(tenant_id or self.config.tenant_id)
        task_payload = self.submit_chat(
            session_id,
            user_input=user_input,
            attachments=attachments,
            requested_model=requested_model,
            tenant_id=resolved_tenant_id,
        )
        return self.wait_for_chat_result(
            str(task_payload.get("task_id") or ""),
            tenant_id=resolved_tenant_id,
        )

    # ------------------------------------------------------------------
    # Cognitive navigation and longitudinal analytics
    # ------------------------------------------------------------------

    def _record_session_snapshot(self, session_id: str, state: dict[str, Any] | None, *, source: str = "") -> None:
        normalized_session_id = str(session_id or "").strip()
        if not normalized_session_id or not isinstance(state, dict):
            return
        history = self._session_snapshot_history.setdefault(normalized_session_id, [])
        snapshot = {
            "captured_at": time.time(),
            "source": str(source or ""),
            "state": dict(state),
        }
        history.append(snapshot)
        if len(history) > 256:
            del history[0 : len(history) - 256]

    @staticmethod
    def _build_graph_indexes(graph_payload: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], set[str]]:
        edges = [edge for edge in list(graph_payload.get("edges") or []) if isinstance(edge, dict)]
        out_index: dict[str, list[dict[str, Any]]] = {}
        in_index: dict[str, list[dict[str, Any]]] = {}
        nodes: set[str] = set()
        for edge in edges:
            source = str(edge.get("source") or "").strip()
            target = str(edge.get("target") or "").strip()
            if not source or not target:
                continue
            nodes.add(source)
            nodes.add(target)
            out_index.setdefault(source, []).append(edge)
            in_index.setdefault(target, []).append(edge)
        return out_index, in_index, nodes

    def _latest_state_for_session(self, session_id: str, *, tenant_id: str = "") -> dict[str, Any]:
        normalized_session_id = str(session_id or "").strip()
        if not normalized_session_id:
            return {}

        history = self._session_snapshot_history.get(normalized_session_id) or []
        if history:
            latest = history[-1]
            if isinstance(latest.get("state"), dict):
                return dict(latest["state"])

        resolved_tenant_id = normalize_tenant_id(tenant_id or self.config.tenant_id)
        events = self.session_events(normalized_session_id, tenant_id=resolved_tenant_id)
        for event in reversed(events):
            payload = dict(event.get("payload") or {}) if isinstance(event, dict) else {}
            event_state = payload.get("session_state")
            if isinstance(event_state, dict):
                self._record_session_snapshot(normalized_session_id, event_state, source="events")
                return dict(event_state)
        return {}

    def cognition_graph(self, session_id: str, *, tenant_id: str = "") -> dict[str, Any]:
        state = self._latest_state_for_session(session_id, tenant_id=tenant_id)
        reflection = dict(state.get("last_reflection_summary") or {})
        evidence_graph = dict(reflection.get("evidence_graph") or {})
        edges = [edge for edge in list(evidence_graph.get("edges") or []) if isinstance(edge, dict)]
        out_index, in_index, nodes = self._build_graph_indexes({"edges": edges})
        return {
            "session_id": str(session_id or "").strip(),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": sorted(nodes),
            "edges": edges,
            "out_index": out_index,
            "in_index": in_index,
            "reflection_summary": reflection,
            "friction_analysis": dict(state.get("last_friction_analysis") or {}),
            "goal_resynthesis": dict(state.get("last_goal_resynthesis") or {}),
            "goal_alignment_mandatory_halt": bool(state.get("goal_alignment_mandatory_halt", False)),
        }

    def cognition_neighbors(
        self,
        session_id: str,
        node_id: str,
        *,
        direction: str = "out",
        tenant_id: str = "",
    ) -> list[dict[str, Any]]:
        graph = self.cognition_graph(session_id, tenant_id=tenant_id)
        normalized_node = str(node_id or "").strip()
        if not normalized_node:
            return []
        direction_norm = str(direction or "out").strip().lower()
        if direction_norm == "in":
            return list(graph.get("in_index", {}).get(normalized_node) or [])
        if direction_norm == "both":
            return list(graph.get("out_index", {}).get(normalized_node) or []) + list(
                graph.get("in_index", {}).get(normalized_node) or []
            )
        return list(graph.get("out_index", {}).get(normalized_node) or [])

    def cognition_paths(
        self,
        session_id: str,
        source_node: str,
        target_node: str,
        *,
        max_depth: int = 6,
        max_paths: int = 24,
        tenant_id: str = "",
    ) -> list[list[dict[str, Any]]]:
        graph = self.cognition_graph(session_id, tenant_id=tenant_id)
        out_index: dict[str, list[dict[str, Any]]] = dict(graph.get("out_index") or {})
        start = str(source_node or "").strip()
        goal = str(target_node or "").strip()
        if not start or not goal:
            return []

        depth_limit = max(1, int(max_depth or 1))
        path_limit = max(1, int(max_paths or 1))
        paths: list[list[dict[str, Any]]] = []

        def _dfs(current: str, path_edges: list[dict[str, Any]], visited: set[str]) -> None:
            if len(paths) >= path_limit:
                return
            if len(path_edges) > depth_limit:
                return
            if current == goal and path_edges:
                paths.append(list(path_edges))
                return
            for edge in list(out_index.get(current) or []):
                target = str(edge.get("target") or "").strip()
                if not target or target in visited:
                    continue
                path_edges.append(edge)
                visited.add(target)
                _dfs(target, path_edges, visited)
                visited.remove(target)
                path_edges.pop()

        _dfs(start, [], {start})
        return paths

    def cognition_compare_branches(
        self,
        session_id: str,
        origin_node: str,
        *,
        depth: int = 3,
        tenant_id: str = "",
    ) -> dict[str, Any]:
        graph = self.cognition_graph(session_id, tenant_id=tenant_id)
        origin = str(origin_node or "").strip()
        if not origin:
            return {"origin": "", "branches": []}

        max_depth = max(1, int(depth or 1))
        branches: list[dict[str, Any]] = []
        for edge in list(graph.get("out_index", {}).get(origin) or []):
            first_target = str(edge.get("target") or "").strip()
            if not first_target:
                continue
            branch_paths = self.cognition_paths(
                session_id,
                source_node=first_target,
                target_node="outcome:unrecovered",
                max_depth=max_depth,
                max_paths=16,
                tenant_id=tenant_id,
            )
            recovered_paths = self.cognition_paths(
                session_id,
                source_node=first_target,
                target_node="outcome:recovered",
                max_depth=max_depth,
                max_paths=16,
                tenant_id=tenant_id,
            )
            branches.append(
                {
                    "edge": edge,
                    "to_unrecovered_paths": len(branch_paths),
                    "to_recovered_paths": len(recovered_paths),
                    "branch_weight": float(edge.get("weight") or 0.0),
                }
            )
        branches.sort(key=lambda item: (item["to_unrecovered_paths"], item["branch_weight"]), reverse=True)
        return {"origin": origin, "branches": branches}

    def cognition_counterfactual(
        self,
        session_id: str,
        *,
        block_nodes: list[str] | None = None,
        block_edges: list[tuple[str, str]] | None = None,
        tenant_id: str = "",
    ) -> dict[str, Any]:
        graph = self.cognition_graph(session_id, tenant_id=tenant_id)
        blocked_nodes = {str(node or "").strip() for node in list(block_nodes or []) if str(node or "").strip()}
        blocked_edges = {
            (str(src or "").strip(), str(dst or "").strip())
            for src, dst in list(block_edges or [])
            if str(src or "").strip() and str(dst or "").strip()
        }

        start_nodes = [
            node
            for node in list(graph.get("nodes") or [])
            if str(node).startswith("event:") and node not in blocked_nodes
        ]

        def _reachable(target_node: str) -> bool:
            queue = list(start_nodes)
            seen = set(queue)
            out_index: dict[str, list[dict[str, Any]]] = dict(graph.get("out_index") or {})
            while queue:
                node = queue.pop(0)
                if node == target_node:
                    return True
                for edge in list(out_index.get(node) or []):
                    src = str(edge.get("source") or "").strip()
                    dst = str(edge.get("target") or "").strip()
                    if not dst or dst in seen:
                        continue
                    if dst in blocked_nodes or src in blocked_nodes:
                        continue
                    if (src, dst) in blocked_edges:
                        continue
                    seen.add(dst)
                    queue.append(dst)
            return False

        recovered_reachable = _reachable("outcome:recovered")
        unrecovered_reachable = _reachable("outcome:unrecovered")
        return {
            "session_id": str(session_id or "").strip(),
            "blocked_nodes": sorted(blocked_nodes),
            "blocked_edges": sorted(blocked_edges),
            "reachable": {
                "outcome:recovered": recovered_reachable,
                "outcome:unrecovered": unrecovered_reachable,
            },
            "predicted_halt_risk": 1.0 if unrecovered_reachable and not recovered_reachable else 0.35,
        }

    def cognition_timeline(self, session_id: str, *, window: int = 64) -> dict[str, Any]:
        normalized_session_id = str(session_id or "").strip()
        history = list(self._session_snapshot_history.get(normalized_session_id) or [])
        if not history:
            return {
                "session_id": normalized_session_id,
                "points": [],
                "trend": "unknown",
                "stabilization_ratio": 0.0,
            }

        safe_window = max(1, int(window or 1))
        points: list[dict[str, Any]] = []
        for index, item in enumerate(history[-safe_window:]):
            state = dict(item.get("state") or {})
            reflection = dict(state.get("last_reflection_summary") or {})
            friction = dict(state.get("last_friction_analysis") or {})
            points.append(
                {
                    "index": index,
                    "captured_at": float(item.get("captured_at") or 0.0),
                    "risk_level": str(reflection.get("current_risk_level") or "unknown"),
                    "drift_probability": float(reflection.get("predicted_drift_probability") or 0.0),
                    "friction_score": float(friction.get("composite_score") or 0.0),
                    "mandatory_halt": bool(state.get("goal_alignment_mandatory_halt", False)),
                }
            )

        if len(points) >= 2:
            first = points[0]
            last = points[-1]
            drift_delta = float(last["drift_probability"]) - float(first["drift_probability"])
            trend = "stabilizing" if drift_delta < -0.05 else "escalating" if drift_delta > 0.05 else "flat"
        else:
            trend = "flat"

        if points:
            stable_points = sum(1 for point in points if point["risk_level"] in {"low", "moderate"} and not point["mandatory_halt"])
            stabilization_ratio = stable_points / float(len(points))
        else:
            stabilization_ratio = 0.0

        return {
            "session_id": normalized_session_id,
            "points": points,
            "trend": trend,
            "stabilization_ratio": round(stabilization_ratio, 3),
        }

    def cognition_influence_map(self, session_id: str, *, tenant_id: str = "") -> dict[str, Any]:
        state = self._latest_state_for_session(session_id, tenant_id=tenant_id)
        reflection = dict(state.get("last_reflection_summary") or {})
        friction = dict(state.get("last_friction_analysis") or {})
        resynthesis = dict(state.get("last_goal_resynthesis") or {})
        graph = self.cognition_graph(session_id, tenant_id=tenant_id)
        edges = list(graph.get("edges") or [])

        top_causes = sorted(
            [edge for edge in edges if isinstance(edge, dict)],
            key=lambda edge: float(edge.get("weight") or 0.0),
            reverse=True,
        )[:5]

        actions: list[dict[str, Any]] = []
        if friction.get("recommended_intervention"):
            actions.append(
                {
                    "action": str(friction.get("recommended_intervention") or ""),
                    "reason": "friction_analysis.recommended_intervention",
                    "triggered_by": str(friction.get("primary_friction_factor") or ""),
                }
            )
        if reflection.get("recommended_intervention"):
            actions.append(
                {
                    "action": str(reflection.get("recommended_intervention") or ""),
                    "reason": "reflection_summary.recommended_intervention",
                    "triggered_by": str(reflection.get("likely_trigger_category") or ""),
                }
            )
        if bool(state.get("goal_alignment_mandatory_halt", False)):
            actions.append(
                {
                    "action": "MANDATORY_HALT",
                    "reason": "goal_alignment_mandatory_halt",
                    "triggered_by": "alignment_guard",
                }
            )

        for proposal in list(resynthesis.get("proposals") or []):
            if not isinstance(proposal, dict):
                continue
            actions.append(
                {
                    "action": str(proposal.get("adjustment_type") or "goal_adjustment"),
                    "reason": "goal_resynthesis.proposal",
                    "triggered_by": str(proposal.get("rationale") or ""),
                }
            )

        return {
            "session_id": str(session_id or "").strip(),
            "top_causes": top_causes,
            "actions": actions,
            "attribution_ready": bool(actions and top_causes),
        }
