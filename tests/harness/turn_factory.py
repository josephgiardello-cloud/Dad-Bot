"""TurnFactory — deterministic TurnContext generator for tests.

Same seed always produces:
  - Same user_input
  - Same trace_id (SHA-256 derived)
  - Same temporal axis (via from_lock_hash)
  - Same VirtualClock base epoch
  - Same mutation ordering (when mutations are passed)
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from dadbot.core.graph import (
    MutationIntent,
    TurnContext,
    TurnTemporalAxis,
    VirtualClock,
)

_USER_INPUTS = [
    "What's for dinner tonight?",
    "Tell me about my week.",
    "How are you feeling today?",
    "Remember when we talked about saving money?",
    "What did I do last weekend?",
    "I'm feeling stressed about work.",
    "Can you help me plan my day?",
    "What's the weather like today?",
    "Tell me something interesting.",
    "I need some advice about a decision.",
    "Did you remember what I told you yesterday?",
    "What's our budget looking like?",
]


def _seed_to_trace_id(seed: int) -> str:
    return hashlib.sha256(f"dadbot-turn-seed-{seed}".encode()).hexdigest()[:32]


def _seed_to_lock_hash(seed: int) -> str:
    return hashlib.sha256(f"temporal-lock-seed-{seed}".encode()).hexdigest()[:16]


class TurnFactory:
    """Deterministic TurnContext generator.

    Usage::

        factory = TurnFactory()
        ctx = factory.build_turn(seed=42)
        # Same seed always produces same trace_id, user_input, temporal axis.

        ctx_a, ctx_b = factory.build_pair(seed=42)
        assert ctx_a.trace_id == ctx_b.trace_id  # guaranteed identical
    """

    def build_turn(
        self,
        seed: int,
        *,
        mutations: list[MutationIntent] | None = None,
        enable_virtual_clock: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> TurnContext:
        rng = random.Random(seed)
        user_input = _USER_INPUTS[rng.randint(0, len(_USER_INPUTS) - 1)]
        trace_id = _seed_to_trace_id(seed)
        temporal = TurnTemporalAxis.from_lock_hash(_seed_to_lock_hash(seed))

        ctx = TurnContext(
            user_input=user_input,
            trace_id=trace_id,
            temporal=temporal,
            metadata=dict(metadata or {}),
        )

        if enable_virtual_clock:
            ctx.virtual_clock = VirtualClock(
                base_epoch=1_700_000_000.0 + seed * 3600.0,
                step_size_seconds=30.0,
            )

        if mutations:
            for intent in mutations:
                ctx.mutation_queue.queue(intent)

        return ctx

    def build_pair(
        self,
        seed: int,
        **kwargs: Any,
    ) -> tuple[TurnContext, TurnContext]:
        """Return two independently constructed but identical turns from the same seed."""
        return self.build_turn(seed, **kwargs), self.build_turn(seed, **kwargs)

    def context_snapshot_hash(self, ctx: TurnContext) -> str:
        """Stable hash of deterministic serialized snapshot scope only.

        Uses serialized context and a normalized execution-trace projection
        (sequence/type/stage/phase) and explicitly excludes volatile timings.
        """
        import json

        raw_trace = list(getattr(ctx, "state", {}).get("execution_trace") or [])
        normalized_trace = [
            {
                "sequence": int(item.get("sequence", 0) or 0),
                "event_type": str(item.get("event_type", "") or ""),
                "stage": str(item.get("stage", "") or ""),
                "phase": str(item.get("phase", "") or ""),
            }
            for item in raw_trace
            if isinstance(item, dict)
        ]

        snapshot = {
            "schema_version": "1",
            "context": {
                "trace_id": str(getattr(ctx, "trace_id", "") or ""),
                "user_input": str(getattr(ctx, "user_input", "") or ""),
                "temporal_wall_time": str(getattr(getattr(ctx, "temporal", None), "wall_time", "") or ""),
                "temporal_wall_date": str(getattr(getattr(ctx, "temporal", None), "wall_date", "") or ""),
                "phase": str(getattr(getattr(ctx, "phase", None), "value", "") or ""),
            },
            "trace": normalized_trace,
        }
        serialized = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()[:24]
