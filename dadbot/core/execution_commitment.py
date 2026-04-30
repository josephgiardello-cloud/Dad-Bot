from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


def _stable_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


@dataclass(frozen=True)
class ExecutionCommitment:
    """Canonical lock-hash primitive for turn execution determinism.

    Contract:
    - Same canonical payload => same lock_hash and lock_id.
    - Any mutation to payload dimensions (input, model identity, memory/tool fingerprints,
      blackboard seed/fingerprint, or manifest hash) => different lock_hash.
    - Model adapters consume this lock via determinism_context to normalize generation output
      under the same commitment boundary.
    """

    user_input: str
    attachments: list[Any]
    llm_provider: str
    llm_model: str
    state_machine: str
    agent_blackboard_seed: dict[str, Any]
    agent_blackboard_seed_fingerprint: str
    agent_blackboard: dict[str, Any]
    agent_blackboard_fingerprint: str
    memory_fingerprint: str
    determinism_manifest_hash: str
    tool_trace_hash: str
    lock_version: int = 3

    def payload(self) -> dict[str, Any]:
        return {
            "user_input": str(self.user_input or ""),
            "attachments": list(self.attachments or []),
            "llm_provider": str(self.llm_provider or ""),
            "llm_model": str(self.llm_model or ""),
            "state_machine": str(self.state_machine or "PLAN_ACT_OBSERVE_RESPOND"),
            "agent_blackboard_seed": dict(self.agent_blackboard_seed or {}),
            "agent_blackboard_seed_fingerprint": str(
                self.agent_blackboard_seed_fingerprint or "",
            ),
            "agent_blackboard": dict(self.agent_blackboard or {}),
            "agent_blackboard_fingerprint": str(
                self.agent_blackboard_fingerprint or "",
            ),
            "memory_fingerprint": str(self.memory_fingerprint or ""),
            "determinism_manifest_hash": str(self.determinism_manifest_hash or ""),
            "tool_trace_hash": str(self.tool_trace_hash or ""),
        }

    @property
    def lock_hash(self) -> str:
        return _stable_sha256(self.payload())

    @property
    def lock_id(self) -> str:
        return f"det-{self.lock_hash[:16]}"


__all__ = ["ExecutionCommitment"]
