from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from dadbot.core.kernel_mutation_gate import apply_event, emit_event

from .contracts import (
    DEFAULT_TENANT_ID,
    ChatRequest,
    ChatResponse,
    EventEnvelope,
    EventType,
    ExecutionGraph,
    ExecutionNode,
    ToolCapability,
    WorkerResult,
    WorkerTask,
    normalize_tenant_id,
)
from .execution_identity import ExecutionIdentity, require_identifier, require_tenant_id
from .runtime_signals import start_span
from .state import AppStateContainer, InMemoryStateStore, NamespacedStateStore, StateStore

if TYPE_CHECKING:
    from .delegation import SQLiteDelegationStore

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolCapability] = {}

    def register(self, capability: ToolCapability) -> None:
        self._tools[capability.name] = capability

    def list_capabilities(self) -> list[ToolCapability]:
        return list(self._tools.values())


class DadBotOrchestrator:
    def __init__(self, broker, *, state_store: StateStore | None = None, event_bus=None, planner_debug_factory=None):
        self.broker = broker
        self.state_store = state_store or InMemoryStateStore()
        self.event_bus = event_bus
        self._planner_debug_factory = planner_debug_factory or (lambda: {})
        self._sessions: dict[str, AppStateContainer] = {}
        self._tool_registry = ToolRegistry()
        self._recovered_task_ids: set[str] = set()
        self._authenticated_execution = False
        self._delegation_store: SQLiteDelegationStore | None = None

    def require_authenticated_execution(self) -> None:
        self._authenticated_execution = True

    def configure_delegation_store(self, store) -> None:
        self._delegation_store = store

    @staticmethod
    def tenant_namespace(tenant_id: str) -> str:
        return f"tenant:{require_tenant_id(tenant_id)}"

    def session(self, session_id: str, tenant_id: str = DEFAULT_TENANT_ID) -> AppStateContainer:
        normalized_tenant_id = require_tenant_id(tenant_id)
        container_key = f"{normalized_tenant_id}:{session_id}"
        container = self._sessions.get(container_key)
        if container is None:
            container = AppStateContainer(
                session_id,
                self._planner_debug_factory,
                tenant_id=normalized_tenant_id,
                store=NamespacedStateStore(self.state_store, self.tenant_namespace(normalized_tenant_id)),
                event_bus=self.event_bus,
            )
            self._sessions[container_key] = container
        return container

    def tool_registry(self) -> ToolRegistry:
        return self._tool_registry

    def build_execution_graph(self, request: ChatRequest) -> ExecutionGraph:
        return ExecutionGraph(
            nodes=[
                ExecutionNode(
                    node_id="api", layer="api", status="completed", metadata={"request_id": request.request_id}
                ),
                ExecutionNode(
                    node_id="queue",
                    layer="orchestration",
                    status="queued",
                    metadata={"session_id": request.session_id, "tenant_id": normalize_tenant_id(request.tenant_id)},
                ),
                ExecutionNode(node_id="worker", layer="worker", status="pending"),
                ExecutionNode(node_id="state", layer="state", status="pending"),
            ],
            edges=[("api", "queue"), ("queue", "worker"), ("worker", "state")],
        )

    def submit_chat(self, request: ChatRequest, *, bearer_token: str = "") -> WorkerTask:
        delegation_store = self._delegation_store
        if self._authenticated_execution and delegation_store is None:
            from .delegation import AuthorityUnavailableError

            raise AuthorityUnavailableError("Delegated execution authority is unavailable")
        require_tenant_id(request.tenant_id)
        require_identifier(request.session_id, "session")
        require_identifier(request.request_id, "request")
        tenant_event = emit_event(
            "MUTATION_EVENT",
            {
                "op": "normalize_tenant_id",
                "value": normalize_tenant_id(request.tenant_id),
            },
            source="DadBotOrchestrator.submit_chat",
        )
        normalized_tenant = apply_event(
            tenant_event,
            {"tenant_id": request.tenant_id},
            lambda state, evt: {
                **state,
                "tenant_id": str(evt.payload.get("value") or state.get("tenant_id") or ""),
            },
        )["tenant_id"]
        request = replace(request, tenant_id=str(normalized_tenant))
        container = self.session(request.session_id, request.tenant_id)
        graph = self.build_execution_graph(request)

        with start_span("orchestrator.submit_chat", session_id=request.session_id, request_id=request.request_id):
            task = WorkerTask(
                session_id=request.session_id,
                request=request,
                tenant_id=request.tenant_id,
                session_state=container.snapshot(),
                execution_graph=graph,
            )
            if self._authenticated_execution and delegation_store is not None:
                task.delegation = delegation_store.issue(bearer_token, task)
            self.state_store.save_task(
                task.task_id,
                {
                    "task_id": task.task_id,
                    "request_id": request.request_id,
                    "session_id": request.session_id,
                    "tenant_id": request.tenant_id,
                    "status": "queued",
                    "request": request.to_dict(),
                    "session_state": task.session_state,
                    "delegated_session_state": task.session_state if self._authenticated_execution else None,
                    "execution_graph": graph.to_dict(),
                    "accepted_at": task.accepted_at,
                    "task_kind": task.task_kind,
                    "attempt": task.attempt,
                    "delegation": task.delegation,
                },
            )
            self._publish(
                EventEnvelope(
                    session_id=request.session_id,
                    event_type=EventType.REQUEST_ACCEPTED,
                    tenant_id=request.tenant_id,
                    payload={"task_id": task.task_id, "request_id": request.request_id},
                )
            )
            self.broker.enqueue(task)
            self._publish(
                EventEnvelope(
                    session_id=request.session_id,
                    event_type=EventType.REQUEST_DISPATCHED,
                    tenant_id=request.tenant_id,
                    payload={"task_id": task.task_id, "request_id": request.request_id},
                )
            )
            logger.info(
                "Queued chat request",
                extra={
                    "task_id": task.task_id,
                    "request_id": request.request_id,
                    "session_id": request.session_id,
                    "tenant_id": request.tenant_id,
                },
            )
            return task

    def recover_pending_tasks(self, *, limit: int = 1000) -> int:
        if self._authenticated_execution and self._delegation_store is None:
            logger.error("Incomplete tasks remain queued: delegated execution authority is unavailable")
            return 0
        recovered = 0
        for payload in self.state_store.list_incomplete_tasks(limit=limit):
            task_id = str(payload.get("task_id") or "").strip()
            if task_id in self._recovered_task_ids:
                continue
            accepted_at = str(payload.get("accepted_at") or "").strip()
            if not accepted_at:
                logger.warning("Leaving an incomplete legacy task queued because it lacks accepted_at metadata")
                continue
            session_id = str(payload.get("session_id") or "").strip()
            request_payload = payload.get("request")
            if not task_id or not session_id or not isinstance(request_payload, dict):
                raise RuntimeError("Cannot recover persisted task with incomplete request metadata")

            tenant_id = require_tenant_id(payload.get("tenant_id"))
            identity = ExecutionIdentity(
                tenant_id=tenant_id,
                session_id=session_id,
                request_id=require_identifier(payload.get("request_id"), "request"),
                task_id=task_id,
            )
            identity.require_match(
                tenant_id=request_payload.get("tenant_id"),
                session_id=request_payload.get("session_id"),
                request_id=request_payload.get("request_id"),
            )
            request = ChatRequest.from_dict(request_payload)
            session_state = payload.get("session_state")
            if not isinstance(session_state, dict):
                session_state = self.session(session_id, tenant_id).snapshot()
            task = WorkerTask(
                session_id=session_id,
                request=request,
                tenant_id=tenant_id,
                session_state=session_state,
                execution_graph=self.build_execution_graph(request),
                accepted_at=accepted_at,
                task_id=task_id,
                task_kind=str(payload.get("task_kind") or "chat"),
                attempt=max(1, int(payload.get("attempt") or 1) + 1),
                delegation=str(payload.get("delegation") or ""),
            )
            if self._authenticated_execution:
                from .delegation import AuthorityUnavailableError
                from .security import AuthorizationError

                store = self._delegation_store
                if store is None:
                    raise RuntimeError("Delegation authority became unavailable during recovery")
                try:
                    store.verify(task.delegation, task, operation="execute")
                except (AuthorizationError, AuthorityUnavailableError) as exc:
                    logger.error("Task %s remains queued: authority verification failed: %s", task_id, exc)
                    continue
            self.broker.enqueue(task)
            self._recovered_task_ids.add(task_id)
            recovered += 1
        return recovered

    def drain_results(self, limit: int = 100) -> list[WorkerResult]:
        drained: list[WorkerResult] = []
        for _ in range(limit):
            result = self.broker.get_result_nowait()
            if result is None:
                break
            self.apply_worker_result(result)
            drained.append(result)
        return drained

    def apply_worker_result(self, result: WorkerResult) -> ChatResponse | None:
        if self._authenticated_execution and self._delegation_store is not None:
            with self._delegation_store.publication_fence():
                return self._apply_worker_result(result)
        return self._apply_worker_result(result)

    def _apply_worker_result(self, result: WorkerResult) -> ChatResponse | None:
        task_payload = self.state_store.load_task(result.task_id)
        if not task_payload:
            raise ValueError("worker result identity mismatch: task not found")
        identity = ExecutionIdentity(
            tenant_id=require_tenant_id(task_payload.get("tenant_id")),
            session_id=require_identifier(task_payload.get("session_id"), "session"),
            request_id=require_identifier(task_payload.get("request_id"), "request"),
            task_id=require_identifier(task_payload.get("task_id"), "task"),
        )
        if identity.task_id != result.task_id:
            raise ValueError("worker result task identity mismatch")
        request_payload = task_payload.get("request")
        if not isinstance(request_payload, dict):
            raise ValueError("worker result request identity mismatch")
        identity.require_match(
            tenant_id=request_payload.get("tenant_id"),
            session_id=request_payload.get("session_id"),
            request_id=request_payload.get("request_id"),
        )
        identity.require_match(tenant_id=result.tenant_id, session_id=result.session_id, request_id=result.request_id)
        if result.response is not None:
            identity.require_match(
                tenant_id=result.response.tenant_id,
                session_id=result.response.session_id,
                request_id=result.response.request_id,
            )
        from .worker import AUTHORITY_UNAVAILABLE

        authority_error = result.status == "failed" and result.error == AUTHORITY_UNAVAILABLE
        if self._authenticated_execution and not authority_error:
            from .delegation import AuthorityUnavailableError
            from .security import AuthorizationError

            store = self._delegation_store
            try:
                if store is None:
                    raise AuthorityUnavailableError(AUTHORITY_UNAVAILABLE)
                delegated_state = task_payload.get("delegated_session_state")
                if delegated_state is None:
                    delegated_state = task_payload.get("session_state") or {}
                if not isinstance(delegated_state, dict):
                    raise ValueError("Invalid delegated session state")
                task = WorkerTask(
                    task_id=identity.task_id,
                    session_id=identity.session_id,
                    tenant_id=identity.tenant_id,
                    request=ChatRequest.from_dict(request_payload),
                    session_state=dict(delegated_state),
                    delegation=str(task_payload.get("delegation") or ""),
                )
                store.verify_result(task, result)
            except (AuthorizationError, AuthorityUnavailableError) as exc:
                logger.warning("Task %s result rejected: %s", result.task_id, exc)
                authority_error = True
        if authority_error:
            if str(task_payload.get("status") or "").lower() not in {"queued", "running"}:
                raise ValueError("authority rejection does not match an incomplete task")
            logger.error("Task %s remains queued: %s", result.task_id, AUTHORITY_UNAVAILABLE)
            task_payload["error"] = AUTHORITY_UNAVAILABLE
            self.state_store.save_task(result.task_id, task_payload)
            return None
        if str(task_payload.get("status") or "").lower() not in {"queued", "running"}:
            response = dict(result.response.to_dict()) if result.response is not None else None
            if (
                task_payload.get("status") != result.status
                or task_payload.get("error") != result.error
                or task_payload.get("completed_at") != result.completed_at
                or task_payload.get("session_state") != dict(result.session_state or {})
                or self.state_store.load_response(result.task_id) != response
            ):
                raise ValueError("worker result conflicts with an already completed task")
            return result.response
        tenant_id = identity.tenant_id
        container = self.session(result.session_id, tenant_id)
        if result.session_state:
            container.load_snapshot(result.session_state)

        payload_event = emit_event(
            "MUTATION_EVENT",
            {
                "op": "dict_update",
                "updates": {
                    "status": result.status,
                    "error": result.error,
                    "completed_at": result.completed_at,
                    "tenant_id": tenant_id,
                    "session_state": dict(result.session_state or {}),
                },
            },
            source="DadBotOrchestrator.apply_worker_result",
        )
        task_payload = apply_event(
            payload_event,
            task_payload,
            lambda state, evt: {**state, **dict(evt.payload.get("updates") or {})},
        )
        if result.response is not None:
            self.state_store.save_task_result(
                result.task_id,
                task_payload,
                dict(result.response.to_dict()),
            )
            container.record_response(result.task_id, result.response)
            return result.response

        self.state_store.save_task(result.task_id, task_payload)
        self._publish(
            EventEnvelope(
                session_id=result.session_id,
                event_type=EventType.REQUEST_FAILED,
                tenant_id=tenant_id,
                payload={"task_id": result.task_id, "request_id": result.request_id, "error": result.error},
            )
        )
        return None

    def task_status(self, task_id: str) -> dict[str, Any] | None:
        self.drain_results()
        return self.state_store.load_task(task_id)

    def response_for_task(self, task_id: str) -> dict[str, Any] | None:
        self.drain_results()
        return self.state_store.load_response(task_id)

    def session_events(self, session_id: str, tenant_id: str = DEFAULT_TENANT_ID) -> list[dict[str, Any]]:
        self.drain_results()
        scoped_store = NamespacedStateStore(self.state_store, self.tenant_namespace(tenant_id))
        return scoped_store.list_events(session_id)

    def _publish(self, event: EventEnvelope) -> None:
        scoped_store = NamespacedStateStore(self.state_store, self.tenant_namespace(event.tenant_id))
        scoped_store.append_event(event.session_id, event.to_dict())
        if self.event_bus is not None:
            self.event_bus.publish(event)
