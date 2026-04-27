"""MutationFuzzer — seeded adversarial generator for MutationIntent objects.

Produces a mix of valid and invalid mutations with controllable ratios.
The same seed always produces the same sequence of mutations, enabling
deterministic adversarial replay.
"""
from __future__ import annotations

import random
from typing import Any

from dadbot.core.graph import MutationIntent

_TEMPORAL_VALID = {"wall_time": "2026-01-01T00:00:00+00:00", "wall_date": "2026-01-01"}

_VALID_OPS: dict[str, list[str]] = {
    "memory": ["save_mood_state"],
    "relationship": ["update"],
    "ledger": [
        "append_history",
        "record_turn_state",
        "sync_thread_snapshot",
        "clear_turn_context",
        "schedule_maintenance",
        "health_snapshot",
        "capability_audit_event",
    ],
    "goal": ["upsert_goal", "complete_goal", "abandon_goal"],
    "graph": [],
}

# Payloads designed to trigger validation failures
_MALFORMED_PAYLOADS: list[dict[str, Any]] = [
    {},                                                                 # no temporal at all
    {"temporal": "not-a-dict"},                                        # wrong type
    {"temporal": {"wall_time": "", "wall_date": "2026-01-01"}},        # empty wall_time
    {"temporal": {"wall_time": "2026-01-01T00:00:00", "wall_date": ""}},  # empty wall_date
    {"temporal": {"wall_date": "2026-01-01"}},                         # missing wall_time key
]


class MutationFuzzer:
    """Seeded generator for MutationIntent instances.

    Usage::

        fuzzer = MutationFuzzer()
        valid_mutations = fuzzer.generate_valid(seed=42, count=20)
        mixed = fuzzer.generate(seed=1337, count=50, include_invalid=True)
    """

    def generate(
        self,
        seed: int,
        count: int = 50,
        *,
        include_invalid: bool = False,
        invalid_ratio: float = 0.15,
    ) -> list[MutationIntent]:
        """Return a list of MutationIntents (valid by default, optionally mixed)."""
        rng = random.Random(seed)
        results: list[MutationIntent] = []
        for _ in range(count):
            if include_invalid and rng.random() < invalid_ratio:
                # Attempt invalid construction — silently skip if it raises
                results.extend(self._try_invalid(rng))
            else:
                results.append(self._valid(rng))
        return results

    def generate_valid(self, seed: int, count: int = 50) -> list[MutationIntent]:
        return self.generate(seed, count, include_invalid=False)

    def generate_invalid_payloads(self, seed: int, count: int = 10) -> list[dict[str, Any]]:
        """Return raw malformed payload dicts for negative-path validation tests."""
        rng = random.Random(seed)
        return [dict(rng.choice(_MALFORMED_PAYLOADS)) for _ in range(count)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _valid(self, rng: random.Random) -> MutationIntent:
        kind = rng.choice(list(_VALID_OPS))
        ops = _VALID_OPS[kind]
        payload: dict[str, Any] = {"temporal": dict(_TEMPORAL_VALID)}
        if ops:
            payload["op"] = rng.choice(ops)
        return MutationIntent(
            type=kind,
            payload=payload,
            priority=rng.randint(1, 300),
            requires_temporal=True,
        )

    def _try_invalid(self, rng: random.Random) -> list[MutationIntent]:
        """Attempt to construct an invalid intent; return empty list if it raises (expected)."""
        kind = rng.choice(["memory", "ledger"])
        bad_payload = dict(rng.choice(_MALFORMED_PAYLOADS))
        try:
            return [MutationIntent(type=kind, payload=bad_payload)]
        except (RuntimeError, ValueError):
            return []

    def priority_sorted(self, intents: list[MutationIntent]) -> list[MutationIntent]:
        """Return intents in canonical drain order (priority, turn_index, sequence_id)."""
        return sorted(
            intents,
            key=lambda m: (m.priority, m.turn_index, m.sequence_id),
        )
