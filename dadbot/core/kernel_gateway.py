from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dadbot.contracts import AttachmentList, FinalizedTurnResult
from dadbot.kernel_boundary import KernelBoundary
from dadbot.core.semantic_primitives import hash as semantic_hash

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
    def open_scope():
        return KernelBoundary.open_scope()

    def _scope(self):
        return KernelBoundary.open_scope()

    @staticmethod
    def _derive_kernel_trace(
        *,
        session_id: str,
        user_input: str,
        attachments: AttachmentList | None,
        metadata: dict[str, Any],
    ) -> KernelTrace:
        semantic_input = {
            "session_id": str(session_id or "default"),
            "user_input": str(user_input or ""),
            "attachments": list(attachments or []),
            "confluence_key": str(metadata.get("confluence_key") or ""),
            "confluence_mode": str(metadata.get("confluence_mode") or ""),
            "request_id": str(metadata.get("request_id") or ""),
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
        with self._scope():
            return await self._control_plane._submit_turn_kernel(
                session_id=session_id,
                user_input=user_input,
                attachments=attachments,
                metadata=md,
                timeout_seconds=timeout_seconds,
            )


__all__ = ["KernelGateway", "KernelTrace"]
