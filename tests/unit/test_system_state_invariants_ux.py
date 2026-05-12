"""Tests for the 3 system architecture gap modules.

Gap 1 — system_state_model.py     (SystemStateSnapshot, Builder, History)
Gap 2 — system_invariants.py      (InvariantViolation, SystemInvariantSet, defaults)
Gap 3 — ux_behavioral_tuning.py   (PersonalityProfile, PersonalitySmoother, directives)
"""

from __future__ import annotations

import pytest

from dadbot.core.memory_feedback_policy import ToolMemoryProfile

# Gap 1
from dadbot.core.system_state_model import (
    SystemHealthStatus,
    SystemStateBuilder,
    SystemStateHistory,
    SystemStateSnapshot,
)

# Gap 2
from dadbot.core.system_invariants import (
    InvariantSeverity,
    InvariantViolation,
    SystemInvariant,
    SystemInvariantSet,
    build_default_invariant_set,
)

# Gap 3
from dadbot.core.ux_behavioral_tuning import (
    ConversationMood,
    PersonalityProfile,
    PersonalitySmoother,
    ResponseShapingDirective,
    TurnEmotionalContext,
    build_dad_personality,
    build_default_smoother,
)

NOW_MS = 1_700_000_000_000
pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _healthy_profile(tool_name: str, n: int = 10) -> ToolMemoryProfile:
    p = ToolMemoryProfile(tool_name=tool_name)
    p.total_executions = n
    p.success_count = n
    p.failure_count = 0
    p.abort_count = 0
    p.escalation_count = 0
    return p


def _failing_profile(
    tool_name: str,
    *,
    n_ok: int = 1,
    n_abort: int = 8,
    n_escalate: int = 1,
) -> ToolMemoryProfile:
    total = n_ok + n_abort + n_escalate
    p = ToolMemoryProfile(tool_name=tool_name)
    p.total_executions = total
    p.success_count = n_ok
    p.failure_count = total - n_ok
    p.abort_count = n_abort
    p.escalation_count = n_escalate
    return p


def _raw_snapshot(
    *,
    profiles: dict[str, ToolMemoryProfile] | None = None,
    health: SystemHealthStatus = SystemHealthStatus.HEALTHY,
    fault_count: int = 0,
    posture: str = "moderate",
    causal_graph=None,
) -> SystemStateSnapshot:
    """Build a snapshot directly (bypasses builder), for invariant violation tests."""
    return SystemStateSnapshot(
        timestamp_ms=NOW_MS,
        tool_profiles=profiles or {},
        overall_health=health,
        active_fault_count=fault_count,
        policy_posture=posture,
        causal_graph=causal_graph,
    )


# ===========================================================================
# GAP 1 — SystemStateModel
# ===========================================================================


class TestSystemHealthStatus:
    def test_enum_members_exist(self):
        assert SystemHealthStatus.HEALTHY.value == "healthy"
        assert SystemHealthStatus.DEGRADED.value == "degraded"
        assert SystemHealthStatus.CRITICAL.value == "critical"
        assert SystemHealthStatus.UNKNOWN.value == "unknown"


class TestSystemStateSnapshot:
    def test_is_healthy_true_for_healthy(self):
        s = _raw_snapshot(health=SystemHealthStatus.HEALTHY)
        assert s.is_healthy is True

    def test_is_healthy_false_for_degraded(self):
        s = _raw_snapshot(health=SystemHealthStatus.DEGRADED)
        assert s.is_healthy is False

    def test_is_operational_false_for_critical(self):
        s = _raw_snapshot(health=SystemHealthStatus.CRITICAL)
        assert s.is_operational is False

    def test_is_operational_true_for_degraded(self):
        s = _raw_snapshot(health=SystemHealthStatus.DEGRADED)
        assert s.is_operational is True

    def test_is_operational_true_for_unknown(self):
        s = _raw_snapshot(health=SystemHealthStatus.UNKNOWN)
        assert s.is_operational is True

    def test_degraded_tools_returns_unhealthy_names(self):
        profiles = {
            "good": _healthy_profile("good"),
            "bad": _failing_profile("bad"),
        }
        s = _raw_snapshot(profiles=profiles)
        assert "bad" in s.degraded_tools()
        assert "good" not in s.degraded_tools()

    def test_degraded_tools_empty_when_all_healthy(self):
        profiles = {"a": _healthy_profile("a"), "b": _healthy_profile("b")}
        s = _raw_snapshot(profiles=profiles)
        assert s.degraded_tools() == []

    def test_healthy_tool_count(self):
        profiles = {
            "a": _healthy_profile("a"),
            "b": _healthy_profile("b"),
            "c": _failing_profile("c"),
        }
        s = _raw_snapshot(profiles=profiles)
        assert s.healthy_tool_count() == 2

    def test_total_tool_count(self):
        profiles = {f"t{i}": _healthy_profile(f"t{i}") for i in range(5)}
        assert _raw_snapshot(profiles=profiles).total_tool_count() == 5

    def test_memory_entry_count_none_memory(self):
        assert _raw_snapshot().memory_entry_count() == 0

    def test_causal_node_count_none_graph(self):
        assert _raw_snapshot().causal_node_count() == 0

    def test_snapshot_summary_contains_required_keys(self):
        s = _raw_snapshot(profiles={"t": _healthy_profile("t")})
        summary = s.snapshot_summary()
        for key in (
            "overall_health", "tool_count", "healthy_tools",
            "degraded_tools", "active_fault_count", "policy_posture",
            "memory_entry_count", "causal_node_count",
        ):
            assert key in summary, f"missing key: {key}"

    def test_snapshot_summary_values(self):
        profiles = {"t": _healthy_profile("t")}
        s = _raw_snapshot(profiles=profiles, health=SystemHealthStatus.HEALTHY, fault_count=0)
        summary = s.snapshot_summary()
        assert summary["overall_health"] == "healthy"
        assert summary["tool_count"] == 1
        assert summary["healthy_tools"] == 1
        assert summary["degraded_tools"] == []


class TestSystemStateBuilder:
    def setup_method(self):
        self.builder = SystemStateBuilder()

    def test_all_healthy_produces_healthy_snapshot(self):
        profiles = {"a": _healthy_profile("a"), "b": _healthy_profile("b")}
        snap = self.builder.build(profiles, now_ms=NOW_MS)
        assert snap.overall_health == SystemHealthStatus.HEALTHY
        assert snap.is_healthy

    def test_one_failing_produces_degraded(self):
        profiles = {"ok": _healthy_profile("ok"), "bad": _failing_profile("bad")}
        snap = self.builder.build(profiles, now_ms=NOW_MS)
        assert snap.overall_health == SystemHealthStatus.DEGRADED

    def test_majority_failing_produces_critical(self):
        # 4 failing + 1 healthy = 80% unhealthy → CRITICAL
        profiles = {f"bad{i}": _failing_profile(f"bad{i}") for i in range(4)}
        profiles["ok"] = _healthy_profile("ok")
        snap = self.builder.build(profiles, now_ms=NOW_MS)
        assert snap.overall_health == SystemHealthStatus.CRITICAL

    def test_exactly_60_percent_unhealthy_is_critical(self):
        # 3 failing + 2 healthy = 60% → CRITICAL (threshold is >=0.6)
        profiles = {f"bad{i}": _failing_profile(f"bad{i}") for i in range(3)}
        profiles["ok1"] = _healthy_profile("ok1")
        profiles["ok2"] = _healthy_profile("ok2")
        snap = self.builder.build(profiles, now_ms=NOW_MS)
        assert snap.overall_health == SystemHealthStatus.CRITICAL

    def test_just_under_60_percent_is_degraded(self):
        # 2 failing + 4 healthy = 33% unhealthy.  Use minimal faults so the
        # 10-fault CRITICAL floor is not triggered.
        def _low_fault(name: str) -> ToolMemoryProfile:
            p = ToolMemoryProfile(tool_name=name)
            p.total_executions = 10
            p.success_count = 1
            p.failure_count = 9
            p.abort_count = 1
            p.escalation_count = 1
            return p

        profiles = {f"bad{i}": _low_fault(f"bad{i}") for i in range(2)}
        profiles.update({f"ok{i}": _healthy_profile(f"ok{i}") for i in range(4)})
        snap = self.builder.build(profiles, now_ms=NOW_MS)
        assert snap.overall_health == SystemHealthStatus.DEGRADED

    def test_empty_profiles_is_unknown(self):
        snap = self.builder.build({}, now_ms=NOW_MS)
        assert snap.overall_health == SystemHealthStatus.UNKNOWN

    def test_computes_active_fault_count(self):
        profiles = {"a": _failing_profile("a", n_abort=5, n_escalate=3)}
        snap = self.builder.build(profiles, now_ms=NOW_MS)
        assert snap.active_fault_count == 8  # 5 aborts + 3 escalations

    def test_lenient_posture_for_all_healthy_tools(self):
        # All healthy with many executions → high reliability → lenient
        profiles = {f"t{i}": _healthy_profile(f"t{i}", n=30) for i in range(4)}
        snap = self.builder.build(profiles, now_ms=NOW_MS)
        assert snap.policy_posture == "lenient"

    def test_aggressive_posture_for_high_abort_rate(self):
        profiles: dict[str, ToolMemoryProfile] = {}
        for i in range(3):
            p = ToolMemoryProfile(tool_name=f"t{i}")
            p.total_executions = 10
            p.success_count = 0
            p.failure_count = 10
            p.abort_count = 8  # 80% abort rate
            profiles[f"t{i}"] = p
        snap = self.builder.build(profiles, now_ms=NOW_MS)
        assert snap.policy_posture == "aggressive"

    def test_timestamp_uses_provided_now_ms(self):
        snap = self.builder.build({}, now_ms=12345)
        assert snap.timestamp_ms == 12345

    def test_metadata_propagated(self):
        snap = self.builder.build({}, now_ms=NOW_MS, metadata={"env": "test"})
        assert snap.metadata["env"] == "test"

    def test_ten_fault_floor_triggers_critical(self):
        # >=10 active faults should trigger CRITICAL regardless of fraction
        p = ToolMemoryProfile(tool_name="t")
        p.total_executions = 20
        p.success_count = 10
        p.failure_count = 10
        p.abort_count = 6
        p.escalation_count = 5  # 6+5=11 faults
        snap = self.builder.build({"t": p}, now_ms=NOW_MS)
        assert snap.overall_health == SystemHealthStatus.CRITICAL


class TestSystemStateHistory:
    def test_push_and_latest(self):
        history = SystemStateHistory()
        s1 = _raw_snapshot(health=SystemHealthStatus.HEALTHY)
        s2 = _raw_snapshot(health=SystemHealthStatus.DEGRADED)
        history.push(s1)
        history.push(s2)
        assert history.latest() is s2

    def test_latest_is_none_on_empty(self):
        assert SystemStateHistory().latest() is None

    def test_max_snapshots_evicts_oldest(self):
        history = SystemStateHistory(max_snapshots=3)
        snaps = [_raw_snapshot() for _ in range(5)]
        for s in snaps:
            history.push(s)
        kept = history.all_snapshots()
        assert len(kept) == 3
        assert kept[0] is snaps[2]  # First two evicted

    def test_all_snapshots_returns_copy(self):
        history = SystemStateHistory()
        s = _raw_snapshot()
        history.push(s)
        result = history.all_snapshots()
        result.clear()
        assert len(history.all_snapshots()) == 1

    def test_health_timeline_values(self):
        history = SystemStateHistory()
        history.push(_raw_snapshot(health=SystemHealthStatus.HEALTHY))
        history.push(_raw_snapshot(health=SystemHealthStatus.DEGRADED))
        timeline = history.health_timeline()
        assert len(timeline) == 2
        assert timeline[0][1] == "healthy"
        assert timeline[1][1] == "degraded"

    def test_fault_trend(self):
        history = SystemStateHistory()
        for count in (0, 2, 5, 10):
            history.push(_raw_snapshot(fault_count=count))
        assert history.fault_trend() == [0, 2, 5, 10]

    def test_is_degrading_true(self):
        history = SystemStateHistory()
        history.push(_raw_snapshot(health=SystemHealthStatus.HEALTHY))
        history.push(_raw_snapshot(health=SystemHealthStatus.DEGRADED))
        history.push(_raw_snapshot(health=SystemHealthStatus.CRITICAL))
        assert history.is_degrading() is True

    def test_is_degrading_false_when_stable(self):
        history = SystemStateHistory()
        for _ in range(3):
            history.push(_raw_snapshot(health=SystemHealthStatus.HEALTHY))
        assert history.is_degrading() is False

    def test_is_degrading_false_when_improving(self):
        history = SystemStateHistory()
        history.push(_raw_snapshot(health=SystemHealthStatus.CRITICAL))
        history.push(_raw_snapshot(health=SystemHealthStatus.DEGRADED))
        history.push(_raw_snapshot(health=SystemHealthStatus.HEALTHY))
        assert history.is_degrading() is False

    def test_is_degrading_needs_at_least_3(self):
        history = SystemStateHistory()
        history.push(_raw_snapshot(health=SystemHealthStatus.CRITICAL))
        history.push(_raw_snapshot(health=SystemHealthStatus.UNKNOWN))
        assert history.is_degrading() is False

    def test_invalid_max_snapshots_raises(self):
        with pytest.raises(ValueError):
            SystemStateHistory(max_snapshots=0)


# ===========================================================================
# GAP 2 — SystemInvariants
# ===========================================================================


class TestInvariantViolation:
    def test_is_fatal_for_fatal_severity(self):
        v = InvariantViolation(
            name="X", description="X", severity=InvariantSeverity.FATAL,
            snapshot_timestamp_ms=NOW_MS,
        )
        assert v.is_fatal is True

    def test_is_not_fatal_for_error(self):
        v = InvariantViolation(
            name="X", description="X", severity=InvariantSeverity.ERROR,
            snapshot_timestamp_ms=NOW_MS,
        )
        assert v.is_fatal is False

    def test_is_error_or_above_for_fatal(self):
        v = InvariantViolation(
            name="X", description="X", severity=InvariantSeverity.FATAL,
            snapshot_timestamp_ms=NOW_MS,
        )
        assert v.is_error_or_above is True

    def test_is_error_or_above_for_error(self):
        v = InvariantViolation(
            name="X", description="X", severity=InvariantSeverity.ERROR,
            snapshot_timestamp_ms=NOW_MS,
        )
        assert v.is_error_or_above is True

    def test_warning_is_not_error_or_above(self):
        v = InvariantViolation(
            name="X", description="X", severity=InvariantSeverity.WARNING,
            snapshot_timestamp_ms=NOW_MS,
        )
        assert v.is_error_or_above is False

    def test_frozen_dataclass(self):
        v = InvariantViolation(
            name="X", description="X", severity=InvariantSeverity.ERROR,
            snapshot_timestamp_ms=NOW_MS,
        )
        with pytest.raises((AttributeError, TypeError)):
            v.name = "Y"  # type: ignore[misc]


class TestSystemInvariant:
    def test_passing_predicate_returns_none(self):
        inv = SystemInvariant(
            name="always_ok", description="", severity=InvariantSeverity.ERROR,
            predicate=lambda s: (True, ""),
        )
        assert inv.check(_raw_snapshot()) is None

    def test_failing_predicate_returns_violation(self):
        inv = SystemInvariant(
            name="always_fail", description="desc", severity=InvariantSeverity.ERROR,
            predicate=lambda s: (False, "detail text"),
        )
        v = inv.check(_raw_snapshot())
        assert v is not None
        assert v.name == "always_fail"
        assert v.severity == InvariantSeverity.ERROR
        assert v.detail == "detail text"
        assert v.snapshot_timestamp_ms == NOW_MS

    def test_predicate_exception_reports_error_severity(self):
        def broken(s):
            raise RuntimeError("kaboom")

        inv = SystemInvariant(
            name="broken", description="", severity=InvariantSeverity.FATAL,
            predicate=broken,
        )
        v = inv.check(_raw_snapshot())
        assert v is not None
        # Exception in predicate → reported as ERROR, not FATAL
        assert v.severity == InvariantSeverity.ERROR
        assert "exception" in v.detail
        assert "kaboom" in v.detail


class TestSystemInvariantSet:
    def test_empty_set_is_valid(self):
        inv_set = SystemInvariantSet()
        assert inv_set.check(_raw_snapshot()) == []
        assert inv_set.is_valid(_raw_snapshot())

    def test_len_reflects_registered_count(self):
        inv_set = SystemInvariantSet()
        for i in range(4):
            inv_set.add(SystemInvariant(
                name=f"inv_{i}", description="", severity=InvariantSeverity.WARNING,
                predicate=lambda s: (True, ""),
            ))
        assert len(inv_set) == 4

    def test_all_failing_returns_all_violations(self):
        inv_set = SystemInvariantSet()
        for i in range(3):
            inv_set.add(SystemInvariant(
                name=f"fail_{i}", description="", severity=InvariantSeverity.WARNING,
                predicate=lambda s: (False, "bad"),
            ))
        assert len(inv_set.check(_raw_snapshot())) == 3
        assert not inv_set.is_valid(_raw_snapshot())

    def test_fatal_violations_only_fatal(self):
        inv_set = SystemInvariantSet()
        inv_set.add(SystemInvariant(
            name="warn", description="", severity=InvariantSeverity.WARNING,
            predicate=lambda s: (False, ""),
        ))
        inv_set.add(SystemInvariant(
            name="fatal", description="", severity=InvariantSeverity.FATAL,
            predicate=lambda s: (False, ""),
        ))
        fatals = inv_set.fatal_violations(_raw_snapshot())
        assert len(fatals) == 1
        assert fatals[0].name == "fatal"

    def test_error_or_above_excludes_warnings(self):
        inv_set = SystemInvariantSet()
        inv_set.add(SystemInvariant(
            name="w", description="", severity=InvariantSeverity.WARNING,
            predicate=lambda s: (False, ""),
        ))
        inv_set.add(SystemInvariant(
            name="e", description="", severity=InvariantSeverity.ERROR,
            predicate=lambda s: (False, ""),
        ))
        errors = inv_set.error_or_above(_raw_snapshot())
        assert len(errors) == 1
        assert errors[0].name == "e"


class TestDefaultInvariantSet:
    def test_default_set_has_7_invariants(self):
        assert len(build_default_invariant_set()) == 7

    def test_valid_snapshot_passes_all(self):
        inv_set = build_default_invariant_set()
        snap = SystemStateBuilder().build(
            {"t": _healthy_profile("t")}, now_ms=NOW_MS
        )
        violations = inv_set.check(snap)
        assert violations == [], [v.name for v in violations]

    def test_policy_posture_valid_triggered_on_bad_posture(self):
        inv_set = build_default_invariant_set()
        snap = _raw_snapshot(posture="turbo_mode")
        names = {v.name for v in inv_set.check(snap)}
        assert "POLICY_POSTURE_VALID" in names

    def test_policy_posture_valid_all_known_pass(self):
        inv_set = build_default_invariant_set()
        for posture in ("aggressive", "moderate", "lenient"):
            snap = _raw_snapshot(posture=posture)
            names = {v.name for v in inv_set.check(snap)}
            assert "POLICY_POSTURE_VALID" not in names, f"posture={posture} falsely flagged"

    def test_active_fault_count_negative_triggers_fatal(self):
        inv_set = build_default_invariant_set()
        snap = _raw_snapshot(fault_count=-5)
        violations = inv_set.check(snap)
        fatal_names = {v.name for v in violations if v.is_fatal}
        assert "ACTIVE_FAULT_COUNT_NON_NEGATIVE" in fatal_names

    def test_healthy_no_degraded_tools_triggered(self):
        inv_set = build_default_invariant_set()
        # Manually set HEALTHY health but include a failing tool profile
        snap = _raw_snapshot(
            profiles={"bad": _failing_profile("bad")},
            health=SystemHealthStatus.HEALTHY,
            fault_count=0,
        )
        names = {v.name for v in inv_set.check(snap)}
        assert "HEALTHY_NO_DEGRADED_TOOLS" in names

    def test_tool_health_abort_consistency_triggered(self):
        inv_set = build_default_invariant_set()

        # Build a profile that has is_healthy=True but abort_rate >= 0.5.
        # The standard is_healthy property prevents this naturally, so we
        # subclass to force is_healthy=True with contrived abort stats.
        class _ForceHealthy(ToolMemoryProfile):
            @property
            def is_healthy(self) -> bool:  # type: ignore[override]
                return True

        p = _ForceHealthy(tool_name="rigged")
        p.total_executions = 10
        p.abort_count = 6   # 60% abort rate
        p.escalation_count = 0
        p.success_count = 4
        p.failure_count = 6

        snap = _raw_snapshot(profiles={"rigged": p})
        names = {v.name for v in inv_set.check(snap)}
        assert "TOOL_HEALTH_ABORT_CONSISTENCY" in names

    def test_no_unknown_health_with_tools_triggered(self):
        inv_set = build_default_invariant_set()
        snap = _raw_snapshot(
            profiles={"t": _healthy_profile("t")},
            health=SystemHealthStatus.UNKNOWN,
        )
        names = {v.name for v in inv_set.check(snap)}
        assert "NO_UNKNOWN_HEALTH_WITH_TOOLS" in names

    def test_causal_graph_no_cycles_triggered(self):
        from dadbot.core.causal_dependency_graph import CausalDepGraph
        from dadbot.core.tool_memory_causal_contract import CausalMemoryEntry

        graph = CausalDepGraph()
        ea = CausalMemoryEntry(
            tool_name="a", contract_version="1", attempt=1,
            status="ok", causal_key="a", timestamp_ms=NOW_MS, latency_ms=1.0,
        )
        eb = CausalMemoryEntry(
            tool_name="b", contract_version="1", attempt=1,
            status="ok", causal_key="b", timestamp_ms=NOW_MS, latency_ms=1.0,
        )
        graph.add_node(ea)
        graph.add_node(eb)
        graph.add_edge("a", "b")
        graph.add_edge("b", "a")  # Creates a cycle

        snap = _raw_snapshot(causal_graph=graph)
        inv_set = build_default_invariant_set()
        violations = inv_set.check(snap)
        fatal_names = {v.name for v in violations if v.is_fatal}
        assert "CAUSAL_GRAPH_NO_CYCLES" in fatal_names

    def test_causal_graph_no_cycles_passes_when_no_graph(self):
        inv_set = build_default_invariant_set()
        snap = _raw_snapshot(causal_graph=None)
        names = {v.name for v in inv_set.check(snap)}
        assert "CAUSAL_GRAPH_NO_CYCLES" not in names


# ===========================================================================
# GAP 3 — UX Behavioral Tuning
# ===========================================================================


class TestPersonalityProfile:
    def test_default_construction(self):
        p = PersonalityProfile()
        for attr in ("warmth", "humor", "directness", "formality", "patience"):
            v = getattr(p, attr)
            assert 0.0 <= v <= 1.0

    def test_explicit_construction(self):
        p = PersonalityProfile(warmth=0.9, humor=0.3, directness=0.6, formality=0.1, patience=0.8)
        assert p.warmth == 0.9
        assert p.humor == 0.3

    def test_out_of_range_warmth_raises(self):
        with pytest.raises(ValueError, match="warmth"):
            PersonalityProfile(warmth=1.5)

    def test_negative_humor_raises(self):
        with pytest.raises(ValueError, match="humor"):
            PersonalityProfile(humor=-0.01)

    def test_blend_zero_weight_returns_self(self):
        p1 = PersonalityProfile(warmth=0.8, humor=0.6)
        p2 = PersonalityProfile(warmth=0.2, humor=0.1)
        blended = p1.blend(p2, 0.0)
        assert abs(blended.warmth - p1.warmth) < 1e-9
        assert abs(blended.humor - p1.humor) < 1e-9

    def test_blend_full_weight_returns_other(self):
        p1 = PersonalityProfile(warmth=0.8)
        p2 = PersonalityProfile(warmth=0.2)
        blended = p1.blend(p2, 1.0)
        assert abs(blended.warmth - p2.warmth) < 1e-9

    def test_blend_midpoint(self):
        p1 = PersonalityProfile(warmth=0.0)
        p2 = PersonalityProfile(warmth=1.0)
        blended = p1.blend(p2, 0.5)
        assert abs(blended.warmth - 0.5) < 1e-9

    def test_blend_result_always_in_range(self):
        p1 = PersonalityProfile(warmth=0.95, humor=0.9, patience=0.95)
        p2 = PersonalityProfile(warmth=0.1, humor=0.05, patience=0.1)
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            blended = p1.blend(p2, w)
            for attr in ("warmth", "humor", "directness", "formality", "patience"):
                v = getattr(blended, attr)
                assert 0.0 <= v <= 1.0, f"{attr}={v} out of range at weight={w}"

    def test_blend_clamps_overrange_weight(self):
        # weight > 1 should clamp to 1 (returns other)
        p1 = PersonalityProfile(warmth=0.8)
        p2 = PersonalityProfile(warmth=0.2)
        blended = p1.blend(p2, 5.0)
        assert abs(blended.warmth - 0.2) < 1e-9


class TestTurnEmotionalContext:
    def test_frustrated_sentiment_raises_stress(self):
        ctx = TurnEmotionalContext(user_sentiment="frustrated")
        assert ctx.combined_stress > 0.3

    def test_positive_healthy_low_stress(self):
        ctx = TurnEmotionalContext(
            user_sentiment="positive",
            system_health=SystemHealthStatus.HEALTHY,
        )
        assert ctx.combined_stress < 0.25

    def test_critical_health_higher_stress_than_healthy(self):
        base = TurnEmotionalContext(system_health=SystemHealthStatus.HEALTHY)
        critical = TurnEmotionalContext(system_health=SystemHealthStatus.CRITICAL)
        assert critical.combined_stress > base.combined_stress

    def test_tool_failure_adds_stress(self):
        ok = TurnEmotionalContext(last_tool_failed=False)
        fail = TurnEmotionalContext(last_tool_failed=True)
        assert fail.combined_stress > ok.combined_stress

    def test_combined_stress_clamped_0_1(self):
        ctx = TurnEmotionalContext(
            user_sentiment="angry",
            topic_stress_level=1.0,
            system_health=SystemHealthStatus.CRITICAL,
            last_tool_failed=True,
        )
        assert 0.0 <= ctx.combined_stress <= 1.0

    def test_combined_stress_zero_for_calm(self):
        ctx = TurnEmotionalContext(
            user_sentiment="positive",
            topic_stress_level=0.0,
            system_health=SystemHealthStatus.HEALTHY,
            last_tool_failed=False,
        )
        assert ctx.combined_stress < 0.1


class TestPersonalitySmoother:
    def test_starts_at_baseline(self):
        baseline = build_dad_personality()
        smoother = PersonalitySmoother(baseline)
        assert abs(smoother.current_personality.warmth - baseline.warmth) < 1e-9
        assert smoother.current_mood == ConversationMood.CALM

    def test_invalid_smoothing_factor_raises(self):
        with pytest.raises(ValueError):
            PersonalitySmoother(build_dad_personality(), smoothing_factor=1.5)

    def test_negative_smoothing_factor_raises(self):
        with pytest.raises(ValueError):
            PersonalitySmoother(build_dad_personality(), smoothing_factor=-0.1)

    def test_frustrated_user_produces_empathetic_tone(self):
        smoother = build_default_smoother()
        directive = smoother.observe_turn(TurnEmotionalContext(user_sentiment="frustrated"))
        assert directive.tone == "empathetic"

    def test_frustrated_user_mood_is_frustrated(self):
        smoother = build_default_smoother(smoothing_factor=1.0)  # fully reactive
        directive = smoother.observe_turn(TurnEmotionalContext(user_sentiment="frustrated"))
        assert directive.current_mood == ConversationMood.FRUSTRATED

    def test_frustrated_user_lowers_humor(self):
        smoother = build_default_smoother(smoothing_factor=1.0)
        directive = smoother.observe_turn(TurnEmotionalContext(
            user_sentiment="frustrated",
            topic_stress_level=0.9,
        ))
        assert directive.humor_level < 0.4

    def test_degraded_system_triggers_acknowledge(self):
        smoother = build_default_smoother()
        directive = smoother.observe_turn(TurnEmotionalContext(
            system_health=SystemHealthStatus.CRITICAL,
            topic_stress_level=0.8,
        ))
        assert directive.acknowledge_system_state is True

    def test_tool_failure_triggers_acknowledge(self):
        smoother = build_default_smoother()
        directive = smoother.observe_turn(TurnEmotionalContext(last_tool_failed=True))
        assert directive.acknowledge_system_state is True

    def test_healthy_positive_no_acknowledge(self):
        smoother = build_default_smoother()
        directive = smoother.observe_turn(TurnEmotionalContext(
            user_sentiment="positive",
            system_health=SystemHealthStatus.HEALTHY,
        ))
        assert directive.acknowledge_system_state is False

    def test_personality_changes_gradually_with_smoothing(self):
        baseline = build_dad_personality()
        smoother = PersonalitySmoother(baseline, smoothing_factor=0.15)
        initial_humor = smoother.current_personality.humor
        smoother.observe_turn(TurnEmotionalContext(
            user_sentiment="frustrated",
            topic_stress_level=1.0,
        ))
        after = smoother.current_personality.humor
        # Should decrease but not hit zero in a single turn
        assert after < initial_humor
        assert after > 0.0

    def test_personality_shifts_fully_when_smoothing_zero(self):
        smoother = PersonalitySmoother(build_dad_personality(), smoothing_factor=1.0)
        before = smoother.current_personality.humor
        smoother.observe_turn(TurnEmotionalContext(
            user_sentiment="frustrated",
            topic_stress_level=1.0,
        ))
        after = smoother.current_personality.humor
        # With smoothing=1.0, current tracks target immediately
        assert after < before

    def test_playful_mood_emerges_with_positive_deep_interaction(self):
        smoother = build_default_smoother()
        for i in range(4):
            smoother.observe_turn(TurnEmotionalContext(
                user_sentiment="positive",
                interaction_depth=i,
                system_health=SystemHealthStatus.HEALTHY,
            ))
        directive = smoother.observe_turn(TurnEmotionalContext(
            user_sentiment="positive",
            interaction_depth=5,
            system_health=SystemHealthStatus.HEALTHY,
        ))
        assert directive.current_mood in {ConversationMood.PLAYFUL, ConversationMood.ENGAGED}

    def test_reset_restores_baseline(self):
        smoother = build_default_smoother()
        baseline_humor = smoother.baseline.humor
        for _ in range(10):
            smoother.observe_turn(TurnEmotionalContext(
                user_sentiment="frustrated", topic_stress_level=1.0
            ))
        smoother.reset()
        assert abs(smoother.current_personality.warmth - smoother.baseline.warmth) < 1e-9
        assert abs(smoother.current_personality.humor - baseline_humor) < 1e-9
        assert smoother.current_mood == ConversationMood.CALM

    def test_directive_is_response_shaping_directive(self):
        smoother = build_default_smoother()
        directive = smoother.observe_turn(TurnEmotionalContext())
        assert isinstance(directive, ResponseShapingDirective)

    def test_directive_fields_in_valid_range(self):
        smoother = build_default_smoother()
        directive = smoother.observe_turn(TurnEmotionalContext())
        assert directive.tone in {"warm", "empathetic", "playful", "cautious", "matter-of-fact"}
        assert directive.verbosity in {"brief", "normal", "detailed"}
        assert 0.0 <= directive.humor_level <= 1.0
        assert 0.0 <= directive.hedge_level <= 1.0
        assert 0.0 <= directive.warmth_factor <= 1.0

    def test_multiple_turns_mood_tracked(self):
        smoother = build_default_smoother()
        d1 = smoother.observe_turn(TurnEmotionalContext(user_sentiment="positive"))
        d2 = smoother.observe_turn(TurnEmotionalContext(user_sentiment="frustrated"))
        assert d1.current_mood != d2.current_mood

    def test_baseline_property(self):
        baseline = build_dad_personality()
        smoother = PersonalitySmoother(baseline)
        assert smoother.baseline is baseline


class TestResponseShapingDirective:
    def _make(self, **kwargs) -> ResponseShapingDirective:
        defaults = dict(
            tone="warm", verbosity="normal", humor_level=0.5,
            hedge_level=0.2, acknowledge_system_state=False,
            warmth_factor=0.8, current_mood=ConversationMood.CALM,
        )
        defaults.update(kwargs)
        return ResponseShapingDirective(**defaults)

    def test_frozen(self):
        d = self._make()
        with pytest.raises((AttributeError, TypeError)):
            d.tone = "cold"  # type: ignore[misc]

    def test_to_prompt_hints_includes_tone(self):
        d = self._make(tone="empathetic")
        hints = d.to_prompt_hints()
        assert any("empathetic" in h for h in hints)

    def test_to_prompt_hints_low_humor_note(self):
        d = self._make(humor_level=0.1)
        hints = d.to_prompt_hints()
        assert any("humor" in h.lower() for h in hints)

    def test_to_prompt_hints_acknowledge_note(self):
        d = self._make(acknowledge_system_state=True)
        hints = d.to_prompt_hints()
        assert any("tool" in h.lower() or "acknowledge" in h.lower() for h in hints)

    def test_to_prompt_hints_all_strings(self):
        d = self._make(verbosity="brief", humor_level=0.8, hedge_level=0.7,
                       acknowledge_system_state=True)
        hints = d.to_prompt_hints()
        assert all(isinstance(h, str) for h in hints)
        assert len(hints) >= 1


class TestBuildDadPersonality:
    def test_returns_valid_profile(self):
        p = build_dad_personality()
        for attr in ("warmth", "humor", "directness", "formality", "patience"):
            assert 0.0 <= getattr(p, attr) <= 1.0

    def test_dad_is_warm(self):
        assert build_dad_personality().warmth >= 0.75

    def test_dad_is_casual(self):
        assert build_dad_personality().formality <= 0.3

    def test_dad_has_humor(self):
        assert build_dad_personality().humor >= 0.5

    def test_build_default_smoother_returns_smoother(self):
        smoother = build_default_smoother()
        assert isinstance(smoother, PersonalitySmoother)
        assert smoother.baseline == build_dad_personality()
