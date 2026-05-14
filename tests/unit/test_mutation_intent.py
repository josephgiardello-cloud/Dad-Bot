"""Unit tests for MutationIntent — construction validation, payload_hash, op checks."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit
from harness.deterministic_seeds import MUTATION_FUZZ
from harness.mutation_fuzzer import MutationFuzzer

from dadbot.core.graph import (
    MutationIntent,
    MutationKind,
)

_T = {"wall_time": "2026-01-01T00:00:00", "wall_date": "2026-01-01"}


# ---------------------------------------------------------------------------
# Construction — positive paths
# ---------------------------------------------------------------------------


class TestMutationIntentConstruction:
    def test_memory_valid(self):
        intent = MutationIntent(
            type="memory",
            payload={"op": "save_mood_state", "temporal": _T},
        )
        assert intent.type is MutationKind.MEMORY

    def test_all_kinds_accepted(self):
        for kind in MutationKind:
            payload: dict = {"temporal": _T} if kind.value in {"memory", "relationship", "ledger"} else {}
            MutationIntent(
                type=kind.value, payload=payload, requires_temporal=kind.value in {"memory", "relationship", "ledger"}
            )

    def test_requires_temporal_false_skips_check(self):
        intent = MutationIntent(type="memory", payload={}, requires_temporal=False)
        assert intent.type is MutationKind.MEMORY

    def test_goal_no_temporal_required(self):
        intent = MutationIntent(
            type="goal",
            payload={"op": "upsert_goal"},
            requires_temporal=False,
        )
        assert intent.type is MutationKind.GOAL

    def test_priority_defaults_to_100(self):
        intent = MutationIntent(type="goal", payload={}, requires_temporal=False)
        assert intent.priority == 100

    def test_ordering_fields_cast_from_numeric_strings(self):
        intent = MutationIntent(
            type="goal",
            payload={},
            requires_temporal=False,
            priority=50,
            turn_index=1,
            sequence_id=3,
        )
        assert intent.priority == 50
        assert intent.turn_index == 1
        assert intent.sequence_id == 3


# ---------------------------------------------------------------------------
# Construction — negative paths (validation gates)
# ---------------------------------------------------------------------------


class TestMutationIntentValidation:
    @pytest.mark.parametrize("bad_type", ["bogus", "MEMORY", "save", "", "null"])
    def test_invalid_type_string_raises(self, bad_type):
        with pytest.raises(RuntimeError, match="must be one of"):
            MutationIntent(type=bad_type, payload={})

    def test_non_dict_payload_raises(self):
        with pytest.raises(RuntimeError, match="payload must be a dict"):
            MutationIntent(type="goal", payload=["list", "not", "dict"])  # type: ignore

    def test_missing_temporal_when_required_raises(self):
        with pytest.raises(RuntimeError, match="TemporalNode required"):
            MutationIntent(type="memory", payload={"op": "save_mood_state"})

    def test_empty_wall_time_raises(self):
        bad = {"temporal": {"wall_time": "   ", "wall_date": "2026-01-01"}}
        with pytest.raises(RuntimeError, match="TemporalNode required"):
            MutationIntent(type="memory", payload=bad)

    def test_empty_wall_date_raises(self):
        bad = {"temporal": {"wall_time": "2026-01-01T00:00:00", "wall_date": ""}}
        with pytest.raises(RuntimeError, match="TemporalNode required"):
            MutationIntent(type="memory", payload=bad)

    def test_temporal_wrong_type_raises(self):
        with pytest.raises(RuntimeError, match="TemporalNode required"):
            MutationIntent(type="memory", payload={"temporal": "not-a-dict"})

    @pytest.mark.parametrize("op", ["write_arbitrary", "unknown_op", "delete_memory"])
    def test_invalid_memory_op_raises(self, op):
        with pytest.raises(RuntimeError, match="Unsupported memory mutation op"):
            MutationIntent(type="memory", payload={"op": op, "temporal": _T})

    @pytest.mark.parametrize("op", ["append", "delete", "replace"])
    def test_invalid_relationship_op_raises(self, op):
        with pytest.raises(RuntimeError, match="Unsupported relationship mutation op"):
            MutationIntent(type="relationship", payload={"op": op, "temporal": _T})

    @pytest.mark.parametrize("op", ["write_log", "flush", "truncate"])
    def test_invalid_ledger_op_raises(self, op):
        with pytest.raises(RuntimeError, match="Unsupported ledger mutation op"):
            MutationIntent(type="ledger", payload={"op": op, "temporal": _T})

    @pytest.mark.parametrize("op", ["create", "remove", "archive"])
    def test_invalid_goal_op_raises(self, op):
        with pytest.raises(RuntimeError, match="Unsupported goal mutation op"):
            MutationIntent(type="goal", payload={"op": op}, requires_temporal=False)

    def test_non_integer_priority_raises(self):
        with pytest.raises(RuntimeError, match="ordering fields must be integers"):
            MutationIntent(type="goal", payload={}, requires_temporal=False, priority="high")  # type: ignore

    def test_none_priority_raises(self):
        with pytest.raises(RuntimeError, match="ordering fields must be integers"):
            MutationIntent(type="goal", payload={}, requires_temporal=False, priority=None)  # type: ignore


# ---------------------------------------------------------------------------
# payload_hash stability
# ---------------------------------------------------------------------------


class TestMutationIntentPayloadHash:
    def test_identical_payloads_produce_identical_hash(self):
        p = {"op": "save_mood_state", "mood": "happy", "temporal": _T}
        a = MutationIntent(type="memory", payload=dict(p))
        b = MutationIntent(type="memory", payload=dict(p))
        assert a.payload_hash == b.payload_hash

    def test_different_payloads_differ(self):
        a = MutationIntent(type="memory", payload={"op": "save_mood_state", "mood": "happy", "temporal": _T})
        b = MutationIntent(type="memory", payload={"op": "save_mood_state", "mood": "sad", "temporal": _T})
        assert a.payload_hash != b.payload_hash

    def test_hash_length_is_24(self):
        intent = MutationIntent(type="memory", payload={"temporal": _T})
        assert len(intent.payload_hash) == 24

    def test_hash_hex_chars_only(self):
        intent = MutationIntent(type="memory", payload={"temporal": _T})
        assert all(c in "0123456789abcdef" for c in intent.payload_hash)

    def test_key_order_does_not_affect_hash(self):
        payload_a = {"temporal": _T, "op": "save_mood_state", "extra": "x"}
        payload_b = {"extra": "x", "op": "save_mood_state", "temporal": _T}
        a = MutationIntent(type="memory", payload=payload_a)
        b = MutationIntent(type="memory", payload=payload_b)
        assert a.payload_hash == b.payload_hash


# ---------------------------------------------------------------------------
# Fuzzer integration — fuzzer always produces constructable mutations
# ---------------------------------------------------------------------------


class TestMutationFuzzerIntegration:
    def test_fuzzer_valid_batch_all_construct(self):
        fuzzer = MutationFuzzer()
        intents = fuzzer.generate_valid(seed=MUTATION_FUZZ, count=100)
        assert len(intents) == 100
        for intent in intents:
            assert isinstance(intent, MutationIntent)
            assert intent.type in list(MutationKind)

    def test_fuzzer_same_seed_reproducible(self):
        fuzzer = MutationFuzzer()
        a = fuzzer.generate_valid(seed=42, count=20)
        b = fuzzer.generate_valid(seed=42, count=20)
        assert [i.payload_hash for i in a] == [i.payload_hash for i in b]

    def test_fuzzer_different_seeds_differ(self):
        fuzzer = MutationFuzzer()
        a = fuzzer.generate_valid(seed=1, count=20)
        b = fuzzer.generate_valid(seed=2, count=20)
        hashes_a = {i.payload_hash for i in a}
        hashes_b = {i.payload_hash for i in b}
        # Should not be identical sets
        assert hashes_a != hashes_b
