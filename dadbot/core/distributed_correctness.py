from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from dadbot.core.runtime_errors import AuthorityViolation


class NodeRole(Enum):
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class NodeAuthority:
    node_id: str
    epoch: int
    lease_until_ms: int
    role: NodeRole
    state_hash: str = ""

    def is_active(self, now_ms: int) -> bool:
        return now_ms <= self.lease_until_ms


@dataclass(frozen=True)
class ReconciliationPlan:
    authoritative_node: str
    authoritative_hash: str
    divergent_nodes: list[str]
    converged: bool


class DistributedCorrectnessModel:
    """Single-process model of cluster correctness primitives.

    This module encodes authority, split-brain detection, and reconciliation
    semantics independent of transport/runtime platform.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, NodeAuthority] = {}

    def register_node(
        self,
        *,
        node_id: str,
        epoch: int,
        lease_until_ms: int,
        role: NodeRole,
        state_hash: str = "",
    ) -> None:
        self._nodes[node_id] = NodeAuthority(
            node_id=node_id,
            epoch=int(epoch),
            lease_until_ms=int(lease_until_ms),
            role=role,
            state_hash=str(state_hash or ""),
        )

    def leaders(self, *, now_ms: int) -> list[NodeAuthority]:
        active = [n for n in self._nodes.values() if n.is_active(now_ms)]
        return [n for n in active if n.role == NodeRole.LEADER]

    def current_authority(self, *, now_ms: int) -> NodeAuthority | None:
        leaders = self.leaders(now_ms=now_ms)
        if not leaders:
            return None
        # Deterministic tie-break: max epoch, then lexical node_id.
        return sorted(leaders, key=lambda x: (x.epoch, x.node_id), reverse=True)[0]

    def detect_split_brain(self, *, now_ms: int) -> list[str]:
        leaders = self.leaders(now_ms=now_ms)
        if len(leaders) <= 1:
            return []
        highest_epoch = max(n.epoch for n in leaders)
        conflicting = [n.node_id for n in leaders if n.epoch == highest_epoch]
        return sorted(conflicting)

    def enforce_no_split_brain(self, *, now_ms: int) -> None:
        conflicting = self.detect_split_brain(now_ms=now_ms)
        if conflicting:
            raise AuthorityViolation(
                "Distributed correctness violation: split-brain detected",
                context={"leaders": conflicting, "now_ms": now_ms},
            )

    def validate_authority(self, *, node_id: str, now_ms: int) -> bool:
        authority = self.current_authority(now_ms=now_ms)
        return authority is not None and authority.node_id == node_id

    def reconcile(self, *, now_ms: int) -> ReconciliationPlan:
        authority = self.current_authority(now_ms=now_ms)
        if authority is None:
            return ReconciliationPlan(
                authoritative_node="",
                authoritative_hash="",
                divergent_nodes=sorted(self._nodes.keys()),
                converged=False,
            )
        divergent = [
            node.node_id
            for node in self._nodes.values()
            if node.node_id != authority.node_id and node.state_hash != authority.state_hash
        ]
        return ReconciliationPlan(
            authoritative_node=authority.node_id,
            authoritative_hash=authority.state_hash,
            divergent_nodes=sorted(divergent),
            converged=len(divergent) == 0,
        )


__all__ = [
    "NodeRole",
    "NodeAuthority",
    "ReconciliationPlan",
    "DistributedCorrectnessModel",
]
