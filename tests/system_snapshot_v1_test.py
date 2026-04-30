"""Tests — Phase 0: System Snapshot V1.

Test suite Q: System snapshot reproducibility, golden behavior set, schema registry,
               and restore capability.

Coverage:
    Q1–Q5:   SchemaRegistry and FileTreeHasher
    Q6–Q10:  GoldenBehaviorRecord construction and replay
    Q11–Q15: GoldenBehaviorSet (default, replay_all, set_hash stability)
    Q16–Q20: SystemSnapshotV1 (build, hash stability, serialization, load)
    Q21–Q25: SnapshotRestoreValidator (validate, per-record replay)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dadbot.core.system_snapshot import (
    DAG_SCHEMA_VERSION,
    EVENT_LOG_SCHEMA_VERSION,
    SNAPSHOT_VERSION,
    TOOL_IR_SCHEMA_VERSION,
    FileTreeHasher,
    GoldenBehaviorRecord,
    GoldenBehaviorSet,
    SchemaRegistry,
    SnapshotRestoreValidator,
    SystemSnapshotV1,
)

# ---------------------------------------------------------------------------
# Workspace root (used for snapshot builds)
# ---------------------------------------------------------------------------

_WORKSPACE_ROOT = Path(__file__).parent.parent


# ===========================================================================
# Q1–Q5: SchemaRegistry and FileTreeHasher
# ===========================================================================


class TestSchemaRegistry:
    def test_default_versions(self):
        r = SchemaRegistry()
        assert r.tool_ir_schema_version == TOOL_IR_SCHEMA_VERSION
        assert r.dag_schema_version == DAG_SCHEMA_VERSION
        assert r.event_log_schema_version == EVENT_LOG_SCHEMA_VERSION
        assert r.snapshot_version == SNAPSHOT_VERSION

    def test_to_dict_has_all_keys(self):
        r = SchemaRegistry()
        d = r.to_dict()
        assert "tool_ir_schema_version" in d
        assert "dag_schema_version" in d
        assert "event_log_schema_version" in d
        assert "snapshot_version" in d

    def test_frozen_immutable(self):
        r = SchemaRegistry()
        with pytest.raises((AttributeError, TypeError)):
            r.snapshot_version = "V99"  # type: ignore[misc]

    def test_file_tree_hasher_returns_hex(self):
        h = FileTreeHasher.hash_directory(_WORKSPACE_ROOT / "dadbot")
        assert isinstance(h, str)
        assert len(h) == 64  # sha256 hex

    def test_file_tree_hasher_stable(self):
        h1 = FileTreeHasher.hash_directory(_WORKSPACE_ROOT / "dadbot")
        h2 = FileTreeHasher.hash_directory(_WORKSPACE_ROOT / "dadbot")
        assert h1 == h2


# ===========================================================================
# Q6–Q10: GoldenBehaviorRecord
# ===========================================================================


class TestGoldenBehaviorRecord:
    def _make(self, intent="question", strategy="fact_seeking", tools=None):
        return GoldenBehaviorRecord.build(
            prompt="What are my goals?",
            intent_type=intent,
            strategy=strategy,
            tool_plan=tools or ["memory_lookup"],
        )

    def test_build_creates_record(self):
        r = self._make()
        assert r.intent_type == "question"
        assert r.strategy == "fact_seeking"
        assert r.tool_plan == ("memory_lookup",)

    def test_tool_trace_hash_stable(self):
        r1 = self._make()
        r2 = self._make()
        assert r1.tool_trace_hash == r2.tool_trace_hash

    def test_envelope_hash_stable(self):
        r1 = self._make()
        r2 = self._make()
        assert r1.envelope_hash == r2.envelope_hash

    def test_envelope_hash_changes_with_intent(self):
        r_question = self._make(intent="question")
        r_emotional = self._make(intent="emotional")
        assert r_question.envelope_hash != r_emotional.envelope_hash

    def test_replay_is_identical(self):
        r = self._make()
        replayed = r.replay()
        assert replayed.envelope_hash == r.envelope_hash
        assert replayed.tool_trace_hash == r.tool_trace_hash
        assert replayed.plan_class_hash == r.plan_class_hash

    def test_verify_replay_true(self):
        r = self._make()
        assert r.verify_replay() is True

    def test_no_tools_produces_valid_record(self):
        r = GoldenBehaviorRecord.build("Hey!", "casual", "casual_reply", [])
        assert r.tool_plan == ()
        assert r.verify_replay() is True

    def test_to_dict_round_trip(self):
        r = self._make()
        d = r.to_dict()
        assert d["intent_type"] == "question"
        assert d["tool_plan"] == ["memory_lookup"]
        assert len(d["envelope_hash"]) == 64

    def test_plan_class_hash_ignores_specific_tools(self):
        """Same intent/strategy/count but different tools → same plan_class_hash."""
        r1 = GoldenBehaviorRecord.build("X", "question", "fact_seeking", ["memory_lookup"])
        r2 = GoldenBehaviorRecord.build("X", "question", "fact_seeking", ["goal_lookup"])
        assert r1.plan_class_hash == r2.plan_class_hash

    def test_tool_trace_hash_differs_for_different_tools(self):
        r1 = GoldenBehaviorRecord.build("X", "question", "fact_seeking", ["memory_lookup"])
        r2 = GoldenBehaviorRecord.build("X", "question", "fact_seeking", ["goal_lookup"])
        assert r1.tool_trace_hash != r2.tool_trace_hash


# ===========================================================================
# Q11–Q15: GoldenBehaviorSet
# ===========================================================================


class TestGoldenBehaviorSet:
    def test_default_has_25_records(self):
        gs = GoldenBehaviorSet.default()
        assert len(gs.records) == 25

    def test_set_hash_stable(self):
        gs1 = GoldenBehaviorSet.default()
        gs2 = GoldenBehaviorSet.default()
        assert gs1.set_hash == gs2.set_hash

    def test_replay_all_passes(self):
        gs = GoldenBehaviorSet.default()
        result = gs.replay_all()
        assert result["all_passed"] is True
        assert result["passed"] == result["total"]

    def test_get_by_intent_filters_correctly(self):
        gs = GoldenBehaviorSet.default()
        emotional = gs.get_by_intent("emotional")
        assert len(emotional) > 0
        assert all(r.intent_type == "emotional" for r in emotional)

    def test_to_dict_round_trip(self):
        gs = GoldenBehaviorSet.default()
        d = gs.to_dict()
        assert d["record_count"] == 25
        assert "set_hash" in d
        assert len(d["records"]) == 25


# ===========================================================================
# Q16–Q20: SystemSnapshotV1
# ===========================================================================


class TestSystemSnapshotV1:
    def test_build_returns_snapshot(self):
        snap = SystemSnapshotV1.build(_WORKSPACE_ROOT)
        assert snap.snapshot_version == SNAPSHOT_VERSION
        assert isinstance(snap.snapshot_hash, str)
        assert len(snap.snapshot_hash) == 64

    def test_snapshot_hash_stable(self):
        """Rebuilding the snapshot with same code always gives same file_tree_hash."""
        snap1 = SystemSnapshotV1.build(_WORKSPACE_ROOT)
        snap2 = SystemSnapshotV1.build(_WORKSPACE_ROOT)
        assert snap1.file_tree_hash == snap2.file_tree_hash

    def test_golden_set_hash_stable_in_snapshot(self):
        snap1 = SystemSnapshotV1.build(_WORKSPACE_ROOT)
        snap2 = SystemSnapshotV1.build(_WORKSPACE_ROOT)
        assert snap1.golden_set.set_hash == snap2.golden_set.set_hash

    def test_to_dict_contains_all_keys(self):
        snap = SystemSnapshotV1.build(_WORKSPACE_ROOT)
        d = snap.to_dict()
        assert "snapshot_version" in d
        assert "git_hash" in d
        assert "file_tree_hash" in d
        assert "schema_registry" in d
        assert "golden_set" in d
        assert "snapshot_hash" in d

    def test_write_and_load_round_trip(self, tmp_path):
        snap = SystemSnapshotV1.build(_WORKSPACE_ROOT)
        out = tmp_path / "SYSTEM_SNAPSHOT_TEST.json"
        snap.write(out)

        loaded = SystemSnapshotV1.load(out)
        assert loaded.snapshot_version == snap.snapshot_version
        assert loaded.file_tree_hash == snap.file_tree_hash
        assert loaded.golden_set.set_hash == snap.golden_set.set_hash
        assert loaded.snapshot_hash == snap.snapshot_hash


# ===========================================================================
# Q21–Q25: SnapshotRestoreValidator
# ===========================================================================


class TestSnapshotRestoreValidator:
    def test_validate_golden_replay_passes(self):
        snap = SystemSnapshotV1.build(_WORKSPACE_ROOT)
        validator = SnapshotRestoreValidator()
        result = validator.validate(snap, _WORKSPACE_ROOT)
        assert result.golden_replay_passed is True
        assert result.golden_passed == result.golden_total

    def test_validate_ok_when_golden_passes(self):
        snap = SystemSnapshotV1.build(_WORKSPACE_ROOT)
        validator = SnapshotRestoreValidator()
        result = validator.validate(snap, _WORKSPACE_ROOT)
        assert result.ok is True

    def test_validate_result_has_version(self):
        snap = SystemSnapshotV1.build(_WORKSPACE_ROOT)
        validator = SnapshotRestoreValidator()
        result = validator.validate(snap, _WORKSPACE_ROOT)
        assert result.snapshot_version == SNAPSHOT_VERSION

    def test_validate_file_tree_match(self):
        snap = SystemSnapshotV1.build(_WORKSPACE_ROOT)
        validator = SnapshotRestoreValidator()
        result = validator.validate(snap, _WORKSPACE_ROOT)
        # Same code → file tree must match.
        assert result.file_tree_match is True

    def test_validate_single_golden_record(self):
        gs = GoldenBehaviorSet.default()
        validator = SnapshotRestoreValidator()
        for record in gs.records:
            assert validator.validate_golden_record(record) is True

    def test_to_dict_has_required_keys(self):
        snap = SystemSnapshotV1.build(_WORKSPACE_ROOT)
        validator = SnapshotRestoreValidator()
        result = validator.validate(snap, _WORKSPACE_ROOT)
        d = result.to_dict()
        assert "ok" in d
        assert "golden_replay_passed" in d
        assert "golden_total" in d
        assert "golden_passed" in d
