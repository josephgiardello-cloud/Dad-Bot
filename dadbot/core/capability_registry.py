"""Capability Registry — per-pipeline-stage capability enforcement.

Bridges the existing session-level authorization system (dadbot.core.authorization)
to the per-node execution layer.  At node entry, TurnGraph checks that the
turn's session has the capabilities required by the stage being entered.

Design
------
CapabilityRegistry    — maps stage names → NodeCapabilityRequirement.
NodeCapabilityRequirement — frozen spec: which capabilities are required and
                            what happens on violation (enforce / warn / skip).
CapabilityViolationError  — raised when a stage cannot be entered.
CapabilitySnapshot        — immutable record of capabilities at turn start.
                            Compared on resume to detect privilege escalation.
enforce_node_entry        — the enforcement function called by TurnGraph.

Why this matters with durable execution
----------------------------------------
Long-running turns that survive a crash and resume can experience a changed
security context (e.g., a different session ID is used to restart).  The
``CapabilitySnapshot`` frozen at turn start is stored in TurnContext.state
and verified at every resume so a resumed turn can NEVER gain capabilities
it did not have at the start.

Architecture role
-----------------
This module imports ONLY from dadbot.core.authorization.  TurnGraph imports
this module; the graph executor has no authorization logic of its own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from dadbot.core.authorization import (
    Capability,
    CapabilitySet,
    SessionAuthorizationPolicy,
)

logger = logging.getLogger(__name__)

# State key where the capability snapshot is stored during a turn.
_CAP_SNAPSHOT_KEY = "_capability_snapshot"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CapabilityViolationError(RuntimeError):
    """Raised when a pipeline stage cannot be entered due to missing capability."""


# ---------------------------------------------------------------------------
# Enforcement mode
# ---------------------------------------------------------------------------


class EnforcementMode(StrEnum):
    """What the registry does when a required capability is missing."""

    ENFORCE = "enforce"  # Raise CapabilityViolationError — turn is aborted
    WARN = "warn"  # Log warning and continue (dev/test mode)
    SKIP = "skip"  # Skip the stage silently (soft degradation)


# ---------------------------------------------------------------------------
# Per-node requirement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeCapabilityRequirement:
    """Declarative capability requirement for a pipeline stage.

    Attributes
    ----------
    stage:
        Stage name this requirement applies to.  Use ``"*"`` for a global
        default applied to all stages not explicitly registered.
    required_capabilities:
        Set of ``Capability`` values the session must hold.  All listed
        capabilities must be present (AND semantics).
    mode:
        Enforcement action when requirements are not met.
    reason:
        Human-readable description of why this capability is required.

    """

    stage: str
    required_capabilities: frozenset[Capability] = field(default_factory=frozenset)
    mode: EnforcementMode = EnforcementMode.ENFORCE
    reason: str = ""

    def is_satisfied_by(self, caps: CapabilitySet) -> bool:
        """Return True if *caps* satisfies all required capabilities."""
        return all(caps.has(cap) for cap in self.required_capabilities)


# ---------------------------------------------------------------------------
# Capability snapshot (frozen at turn start)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapabilitySnapshot:
    """Immutable capability record frozen at turn-start.

    Stored in ``TurnContext.state[_CAP_SNAPSHOT_KEY]`` so that resumed turns
    can be verified against the original capability set, preventing privilege
    escalation across crashes.
    """

    session_id: str
    granted: tuple[str, ...]  # sorted capability value strings

    @classmethod
    def from_policy(
        cls,
        session_id: str,
        policy: SessionAuthorizationPolicy,
    ) -> CapabilitySnapshot:
        caps = policy.caps_for(session_id)
        granted = tuple(sorted(cap.value for cap in Capability if caps.has(cap)))
        return cls(session_id=session_id, granted=granted)

    def to_dict(self) -> dict[str, Any]:
        return {"session_id": self.session_id, "granted": list(self.granted)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CapabilitySnapshot:
        return cls(
            session_id=str(d.get("session_id") or ""),
            granted=tuple(d.get("granted") or []),
        )

    def has(self, capability: Capability) -> bool:
        return capability.value in self.granted

    def is_equivalent_to(self, other: CapabilitySnapshot) -> bool:
        """Return True if both snapshots represent the same capability set.

        Session ID is NOT compared — a resumed session may have a different ID
        but must have the same (or lesser) capability set.
        """
        return set(self.granted) == set(other.granted)

    def is_escalation_of(self, original: CapabilitySnapshot) -> bool:
        """Return True if this snapshot has MORE capabilities than *original*.

        Used on resume: if the new session has capabilities not present at
        turn start, it is a privilege escalation and must be rejected.
        """
        return bool(set(self.granted) - set(original.granted))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class CapabilityRegistry:
    """Maps pipeline stage names to NodeCapabilityRequirement.

    Usage::

        registry = CapabilityRegistry()
        # Require EXECUTE for inference; warn (don't abort) for save.
        registry.register("inference", NodeCapabilityRequirement(
            stage="inference",
            required_capabilities=frozenset({Capability.EXECUTE}),
            mode=EnforcementMode.ENFORCE,
            reason="Inference calls the LLM — requires EXECUTE capability",
        ))
        registry.register("save", NodeCapabilityRequirement(
            stage="save",
            required_capabilities=frozenset({Capability.WRITE}),
            mode=EnforcementMode.ENFORCE,
            reason="SaveNode persists turn state",
        ))

    The default requirement (stage ``"*"``) requires no capabilities, which is
    backward-compatible with existing turns that don't use the registry.
    """

    def __init__(self) -> None:
        self._requirements: dict[str, NodeCapabilityRequirement] = {
            "*": NodeCapabilityRequirement(stage="*"),  # no-op default
        }

    def register(
        self,
        stage: str,
        requirement: NodeCapabilityRequirement,
    ) -> None:
        """Register a capability requirement for *stage*."""
        self._requirements[str(stage or "*").strip().lower() or "*"] = requirement

    def requirement_for(self, stage: str) -> NodeCapabilityRequirement:
        """Return the requirement for *stage*, falling back to the global ``"*"``."""
        key = str(stage or "").strip().lower()
        return self._requirements.get(
            key,
            self._requirements.get("*", NodeCapabilityRequirement(stage="*")),
        )

    def all_requirements(self) -> dict[str, NodeCapabilityRequirement]:
        return dict(self._requirements)


# ---------------------------------------------------------------------------
# Enforcement function (called by TurnGraph._execute_stage)
# ---------------------------------------------------------------------------


def enforce_node_entry(
    stage: str,
    *,
    registry: CapabilityRegistry,
    caps: CapabilitySet | None,
    session_id: str = "",
) -> EnforcementMode:
    """Enforce capability requirement at node entry.

    Parameters
    ----------
    stage:
        Name of the pipeline stage being entered.
    registry:
        The CapabilityRegistry to look up requirements from.
    caps:
        The CapabilitySet for the current session.  When None (no policy),
        enforcement is skipped entirely (backward compatible).
    session_id:
        For logging only.

    Returns
    -------
    EnforcementMode
        The mode that was applied (ENFORCE/WARN/SKIP/no-op).

    Raises
    ------
    CapabilityViolationError
        When mode is ENFORCE and the requirement is not satisfied.

    """
    if caps is None:
        return EnforcementMode.WARN  # no-op; compatible with legacy turns

    req = registry.requirement_for(stage)
    if not req.required_capabilities:
        return EnforcementMode.WARN  # no requirements registered for this stage

    satisfied = req.is_satisfied_by(caps)
    if satisfied:
        return req.mode

    msg = (
        f"CapabilityViolation: stage {stage!r} requires {sorted(c.value for c in req.required_capabilities)!r}"
        f" but session {session_id!r} grants {sorted(v for v in dir(Capability) if not v.startswith('_'))!r}."
        f" Reason: {req.reason}"
    )

    if req.mode == EnforcementMode.ENFORCE:
        raise CapabilityViolationError(msg)
    if req.mode == EnforcementMode.WARN:
        logger.warning("CAPABILITY_WARN: %s", msg)
    elif req.mode == EnforcementMode.SKIP:
        logger.info("CAPABILITY_SKIP: %s (stage will be skipped)", stage)
    return req.mode


# ---------------------------------------------------------------------------
# Snapshot helpers (called by TurnGraph.execute() at turn boundaries)
# ---------------------------------------------------------------------------


def freeze_capabilities(
    turn_context: Any,
    *,
    policy: SessionAuthorizationPolicy,
    session_id: str,
) -> CapabilitySnapshot:
    """Freeze the session's capability set into TurnContext.state.

    Call once at turn start.  Returns the snapshot (also stored in state).
    """
    snapshot = CapabilitySnapshot.from_policy(session_id, policy)
    state = getattr(turn_context, "state", None)
    if isinstance(state, dict):
        state[_CAP_SNAPSHOT_KEY] = snapshot.to_dict()
    return snapshot


def verify_capability_freeze(
    turn_context: Any,
    *,
    policy: SessionAuthorizationPolicy,
    session_id: str,
) -> None:
    """Verify that the current session has not gained capabilities since turn start.

    Call at turn resume (after restoring state from a ResumePoint).  Raises
    CapabilityViolationError if the new session has MORE capabilities than the
    original.

    When no snapshot is stored in state (legacy / non-capability turns), this
    is a no-op.
    """
    state = getattr(turn_context, "state", None)
    if not isinstance(state, dict):
        return
    raw = state.get(_CAP_SNAPSHOT_KEY)
    if not raw:
        return  # no snapshot — backward compatible no-op

    original = CapabilitySnapshot.from_dict(raw)
    current = CapabilitySnapshot.from_policy(session_id, policy)
    if current.is_escalation_of(original):
        escalated = sorted(set(current.granted) - set(original.granted))
        raise CapabilityViolationError(
            f"Capability escalation detected on resume: session {session_id!r} "
            f"has new capabilities {escalated!r} not present at turn start.  "
            "The resumed turn is rejected.",
        )


def get_capability_snapshot(turn_context: Any) -> CapabilitySnapshot | None:
    """Return the frozen capability snapshot from TurnContext.state, or None."""
    state = getattr(turn_context, "state", None)
    if not isinstance(state, dict):
        return None
    raw = state.get(_CAP_SNAPSHOT_KEY)
    if not raw:
        return None
    return CapabilitySnapshot.from_dict(raw)
