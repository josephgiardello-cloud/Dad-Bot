from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dadbot.contracts import AttachmentList, FinalizedTurnResult
from dadbot.core.semantic_primitives import hash as semantic_hash
from dadbot.kernel_boundary import KernelBoundary

if TYPE_CHECKING:
    from dadbot.core.control_plane import ExecutionControlPlane

@dataclass(frozen=True)
class KernelTrace:
    trace_id: str
    semantic_input_hash: str
    schedule_projection_hash: str
    policy_projection_hash: str
    mutation_projection_hash: str
    ledger_projection_hash: str

    def to_dict(self) -> dict[str, str]:
        return {
            "trace_id": self.trace_id,
            "semantic_input_hash": self.semantic_input_hash,
            "schedule_projection_hash": self.schedule_projection_hash,
            "policy_projection_hash": self.policy_projection_hash,
            "mutation_projection_hash": self.mutation_projection_hash,
            "ledger_projection_hash": self.ledger_projection_hash,
        }


class KernelGateway:
    """Hard kernel boundary for all semantic execution.

    All runtime semantic operations must execute under this scope so that
    ledger/scheduler/policy interactions cannot bypass kernel invariants.
    """

    def __init__(self, control_plane: ExecutionControlPlane) -> None:
        self._control_plane = control_plane

    @staticmethod
    def in_scope() -> bool:
        return KernelBoundary.in_scope()

    @staticmethod
    def assert_scope(operation: str) -> None:
        KernelBoundary.assert_scope(operation)

    @staticmethod
    def open_scope() -> AbstractContextManager[None]:
        return KernelBoundary.open_scope()

    def _scope(self) -> AbstractContextManager[None]:
        return KernelBoundary.open_scope()

    def persist_agent_loop_state(
        self, *, session_id: str, tenant_id: str, trace_token: str, state_patch: dict[str, Any],
    ) -> dict[str, Any]:
        from dadbot.core.agent_loop_progress import (
            read_agent_loop_goal_blackboard,
            reconcile_committed_agent_loop_progress,
        )
        from dadbot.core.agent_loop_recovery import AgentLoopCursor
        from dadbot_system.delegation import tenant_mutation_fence

        if not state_patch or set(state_patch) - {"agent_loop_progress", "agent_loop_cursor"}:
            raise ValueError("Kernel agent-loop persistence accepts only canonical loop state")
        if "agent_loop_progress" in state_patch:
            read_agent_loop_goal_blackboard(state_patch, tenant_id=tenant_id, session_id=session_id)
            reconcile_committed_agent_loop_progress(
                state_patch,
                completed_entries=self._control_plane._ledger_index.completed_agent_loop_entries(
                    session_id=session_id, tenant_id=tenant_id,
                ),
                tenant_id=tenant_id, session_id=session_id,
            )
        if "agent_loop_cursor" in state_patch:
            value = state_patch["agent_loop_cursor"]
            if type(value) is not dict or type(value.get("policy")) is not dict:
                raise ValueError("Invalid canonical agent-loop cursor")
            AgentLoopCursor.read(
                value, tenant_id=tenant_id, session_id=session_id,
                initial_input=None, policy=value["policy"],
            )
        with tenant_mutation_fence(
            tenant_id, required=self._control_plane.ledger._authenticated_authority_required,
        ), self._scope():
            return self._control_plane._merge_session_state(
                session_id=session_id, tenant_id=tenant_id,
                trace_token=trace_token, state_patch=state_patch,
            )

    @staticmethod
    def _derive_kernel_trace(
        *,
        session_id: str,
        user_input: str,
        attachments: AttachmentList | None,
        metadata: dict[str, Any],
    ) -> KernelTrace:
        sovereign_context = dict(metadata.get("sovereign_context") or {})
        semantic_input = {
            "tenant_id": str(metadata.get("tenant_id") or sovereign_context.get("tenant_id") or ""),
            "session_id": str(session_id or "default"),
            "user_input": str(user_input or ""),
            "attachments": list(attachments or []),
            "confluence_key": str(metadata.get("confluence_key") or ""),
            "confluence_mode": str(metadata.get("confluence_mode") or ""),
            "request_id": str(metadata.get("request_id") or ""),
            "context_session_id": str(sovereign_context.get("session_id") or ""),
            "context_tenant_id": str(sovereign_context.get("tenant_id") or ""),
            "context_trace_id": str(sovereign_context.get("trace_id") or ""),
            "context_request_id": str(sovereign_context.get("request_id") or ""),
        }
        semantic_input_hash = semantic_hash(semantic_input)
        return KernelTrace(
            trace_id=f"ktr-{semantic_input_hash[:20]}",
            semantic_input_hash=semantic_input_hash,
            schedule_projection_hash=semantic_hash({"kind": "schedule", "seed": semantic_input_hash}),
            policy_projection_hash=semantic_hash({"kind": "policy", "seed": semantic_input_hash}),
            mutation_projection_hash=semantic_hash({"kind": "mutation", "seed": semantic_input_hash}),
            ledger_projection_hash=semantic_hash({"kind": "ledger", "seed": semantic_input_hash}),
        )

    async def submit_turn(
        self,
        *,
        session_id: str,
        user_input: str,
        attachments: AttachmentList | None = None,
        metadata: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> FinalizedTurnResult:
        md = dict(metadata or {})
        trace = self._derive_kernel_trace(
            session_id=session_id,
            user_input=user_input,
            attachments=attachments,
            metadata=md,
        )
        md.setdefault("confluence_key", f"kgw:{trace.semantic_input_hash[:24]}")
        md.setdefault("trace_id", trace.trace_id)
        md["kernel_trace"] = trace.to_dict()
        self._control_plane._topology_begin_turn(
            trace_id=trace.trace_id,
            session_id=str(session_id or "default"),
        )
        self._control_plane._topology_record_node(
            node_id="control_plane.submit_turn",
            metadata={"source": "kernel_gateway.submit_turn"},
        )
        self._control_plane._topology_record_node(
            node_id="kernel_gateway.submit_turn",
            metadata={"kernel_trace": trace.to_dict()},
        )
        self._control_plane._topology_record_node(
            node_id="orchestrator.handle_turn",
            metadata={"session_id": str(session_id or "default")},
        )
        with self._scope():
            try:
                return await self._control_plane._submit_turn_kernel(
                    session_id=session_id,
                    user_input=user_input,
                    attachments=attachments,
                    metadata=md,
                    timeout_seconds=timeout_seconds,
                )
            finally:
                self._control_plane._topology_end_turn()


__all__ = ["KernelGateway", "KernelTrace"]
