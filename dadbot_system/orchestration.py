from __future__ import annotations

import logging
from dataclasses import replace
from itertools import islice
from typing import TYPE_CHECKING, Any

from dadbot.core.kernel_mutation_gate import apply_event, emit_event

from .contracts import (
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
from .owned_tasks import OwnedTaskStore, build_owned_task_store
from .runtime_signals import start_span
from .state import (
    AppStateContainer,
    InMemoryStateStore,
    SQLiteStateStore,
    StateStore,
    TenantOwnedStateStoreView,
)

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
    def __init__(
        self, broker, *, state_store: StateStore | None = None,
        task_store: OwnedTaskStore | None = None, event_bus=None, planner_debug_factory=None,
    ):
        self.broker = broker
        self.state_store = state_store or InMemoryStateStore()
        if task_store is not None:
            self.task_store = task_store
        elif isinstance(self.state_store, SQLiteStateStore):
            self.task_store = build_owned_task_store(sqlite_path=self.state_store.path)
        elif isinstance(self.state_store, InMemoryStateStore):
            self.task_store = build_owned_task_store(in_memory_store=self.state_store.owned_tasks)
        elif getattr(self.state_store, "task_store", None) is not None:
            self.task_store = self.state_store.task_store
        else:
            raise RuntimeError("An explicit tenant-qualified task store is required for this state backend")
        self.event_bus = event_bus
        self._planner_debug_factory = planner_debug_factory or (lambda: {})
        self._sessions: dict[tuple[str, str], AppStateContainer] = {}
        self._tool_registry = ToolRegistry()
        self._recovered_task_ids: set[tuple[str, str]] = set()
        self._authenticated_execution = False
        self._delegation_store: SQLiteDelegationStore | None = None

    def require_authenticated_execution(self) -> None:
        self._authenticated_execution = True

    def configure_delegation_store(self, store) -> None:
        self._delegation_store = store

    @staticmethod
    def tenant_namespace(tenant_id: str) -> str:
        return f"tenant:{require_tenant_id(tenant_id)}"

    def session(self, session_id: str, tenant_id: str) -> AppStateContainer:
        normalized_tenant_id = require_tenant_id(tenant_id)
        require_identifier(session_id, "session")
        container_key = (normalized_tenant_id, session_id)
        container = self._sessions.get(container_key)
        if container is None:
            container = AppStateContainer(
                session_id,
                self._planner_debug_factory,
                tenant_id=normalized_tenant_id,
                store=TenantOwnedStateStoreView(self.state_store, normalized_tenant_id, self.task_store),
                event_bus=self.event_bus,
            )
            self._sessions[container_key] = container
        return container

    def list_sessions(self, tenant_id: str) -> list[str]:
        normalized_tenant_id = require_tenant_id(tenant_id)
        store = TenantOwnedStateStoreView(self.state_store, normalized_tenant_id, self.task_store)
        return store.list_sessions()

    def terminate_session(self, session_id: str, tenant_id: str) -> bool:
        normalized_tenant_id = require_tenant_id(tenant_id)
        require_identifier(session_id, "session")
        store = TenantOwnedStateStoreView(self.state_store, normalized_tenant_id, self.task_store)
        if self._authenticated_execution:
            if self._delegation_store is None:
                from .delegation import AuthorityUnavailableError

                raise AuthorityUnavailableError("Delegated execution authority is unavailable")
            self._delegation_store.revoke_session(
                tenant_id=normalized_tenant_id,
                session_id=session_id,
            )
        deleted = store.delete_session(session_id)
        self._sessions.pop((normalized_tenant_id, session_id), None)
        return deleted

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

    def submit_chat(
        self,
        request: ChatRequest,
        *,
        bearer_token: str = "",
        execution_principal=None,
    ) -> WorkerTask:
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
                if execution_principal is not None:
                    task.delegation = delegation_store.issue_for_principal(execution_principal, task)
                else:
                    task.delegation = delegation_store.issue(bearer_token, task)
            self.task_store.save_task(
                request.tenant_id,
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
        self.task_store.reject_legacy_recovery()
        pending = (
            payload
            for tenant_id in self.task_store.list_task_tenants()
            for payload in self.task_store.list_incomplete_tasks(tenant_id, limit=limit)
        )
        for payload in islice(pending, max(1, int(limit))):
            task_id = str(payload.get("task_id") or "").strip()
            tenant_id = require_tenant_id(payload.get("tenant_id"))
            if (tenant_id, task_id) in self._recovered_task_ids:
                continue
            accepted_at = str(payload.get("accepted_at") or "").strip()
            if not accepted_at:
                logger.warning("Leaving an incomplete legacy task queued because it lacks accepted_at metadata")
                continue
            session_id = str(payload.get("session_id") or "").strip()
            request_payload = payload.get("request")
            if not task_id or not session_id or not isinstance(request_payload, dict):
                raise RuntimeError("Cannot recover persisted task with incomplete request metadata")

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
            self._recovered_task_ids.add((tenant_id, task_id))
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
        task_payload = self.task_store.load_task(require_tenant_id(result.tenant_id), result.task_id)
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
            self.task_store.save_task(identity.tenant_id, result.task_id, task_payload)
            return None
        if str(task_payload.get("status") or "").lower() not in {"queued", "running"}:
            response = dict(result.response.to_dict()) if result.response is not None else None
            if (
                task_payload.get("status") != result.status
                or task_payload.get("error") != result.error
                or task_payload.get("completed_at") != result.completed_at
                or task_payload.get("session_state") != dict(result.session_state or {})
                or self.task_store.load_response(identity.tenant_id, result.task_id) != response
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
            self.task_store.save_task_result(
                tenant_id,
                result.task_id,
                task_payload,
                dict(result.response.to_dict()),
            )
            container.record_event(
                EventType.RESPONSE_READY,
                {
                    "task_id": result.task_id,
                    "request_id": result.response.request_id,
                    "status": result.response.status,
                },
            )
            return result.response

        self.task_store.save_task(tenant_id, result.task_id, task_payload)
        self._publish(
            EventEnvelope(
                session_id=result.session_id,
                event_type=EventType.REQUEST_FAILED,
                tenant_id=tenant_id,
                payload={"task_id": result.task_id, "request_id": result.request_id, "error": result.error},
            )
        )
        return None

    def task_status(self, task_id: str, *, tenant_id: str) -> dict[str, Any] | None:
        require_tenant_id(tenant_id)
        require_identifier(task_id, "task")
        self.drain_results()
        return self.task_store.load_task(tenant_id, task_id)

    def response_for_task(self, task_id: str, *, tenant_id: str) -> dict[str, Any] | None:
        require_tenant_id(tenant_id)
        require_identifier(task_id, "task")
        self.drain_results()
        return self.task_store.load_response(tenant_id, task_id)

    def save_task(self, task_id: str, payload: dict[str, Any], *, tenant_id: str) -> None:
        self.task_store.save_task(tenant_id, task_id, payload)

    def session_events(self, session_id: str, tenant_id: str) -> list[dict[str, Any]]:
        self.drain_results()
        scoped_store = TenantOwnedStateStoreView(self.state_store, tenant_id, self.task_store)
        return scoped_store.list_events(session_id)

    def _publish(self, event: EventEnvelope) -> None:
        scoped_store = TenantOwnedStateStoreView(self.state_store, event.tenant_id, self.task_store)
        scoped_store.append_event(event.session_id, event.to_dict())
        if self.event_bus is not None:
            self.event_bus.publish(event)
