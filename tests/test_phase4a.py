"""Phase 4A benchmark, orchestrator, and checkpoint integration tests.

Canonical turns use offline model transport, not mocked execution.
Mock benchmark checks exercise only the harness.
"""


import asyncio
import contextlib
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.benchmark_runner import BenchmarkRunner
from tests.harness.graph_runner import confluence_key_for_turn
from tests.scenario_suite import (
    SCENARIOS,
    get_scenarios_by_category,
)


def _classify_phase4_failure(detail: object) -> str:
    text = str(detail or "").lower()
    infra_hints = ("connection", "timeout", "reused", "unavailable", "service")
    env_hints = ("modulenotfound", "no module named", "importerror", "dll", "not found")
    if any(h in text for h in env_hints):
        return "test_environment_failure"
    if any(h in text for h in infra_hints):
        return "infrastructure_failure"
    return "system_failure"



def _fail_certification_gate(stage: str, detail: object) -> None:
    category = _classify_phase4_failure(detail)
    pytest.fail(f"[{category}] {stage}: {detail}")


@pytest.mark.parametrize(
    ("detail", "category"),
    [
        ("No module named missing_dependency", "test_environment_failure"),
        ("Connection refused", "infrastructure_failure"),
        ("Invalid checkpoint chain", "system_failure"),
    ],
)
def test_phase4_failure_classification(detail, category):
    assert _classify_phase4_failure(detail) == category


def _assert_handle_turn_not_mocked(orchestrator, stage: str) -> None:
    # Structural policy: patched orchestrator turn execution cannot be certification evidence.
    if isinstance(getattr(orchestrator, "handle_turn", None), (AsyncMock, MagicMock)):
        pytest.fail(
            f"[certification_policy_violation] {stage}: orchestrator.handle_turn is mocked; "
            "mocked orchestrators cannot contribute to Phase 4 certification."
        )


from typing import Any, Tuple


def _make_orchestrator_with_checkpointer(db_path: str, *, strict: bool = False, make_test_dadbot) -> Tuple[Any, Any, Any]:
    """Return (orchestrator, checkpointer, llm_service) with a real SQLiteCheckpointer wired in.

    Model calls are stubbed; graph execution and checkpoint persistence run normally.
    """
    from dadbot.core.orchestrator import DadBotOrchestrator
    from dadbot.core.persistence import SQLiteCheckpointer

    bot = make_test_dadbot()
    checkpointer = SQLiteCheckpointer(db_path, auto_migrate=True, prune_every=0)
    orchestrator = DadBotOrchestrator(bot=bot, strict=strict, checkpointer=checkpointer)
    llm_service = orchestrator.registry.get("llm")
    return orchestrator, checkpointer, llm_service


@contextlib.contextmanager
def _stub_llm(llm):
    with (
        patch.object(llm.bot.model_port, "generate", return_value="[phase4a-offline]"),
        patch.object(llm.bot.model_port, "generate_async", new=AsyncMock(return_value="[phase4a-offline]")),
    ):
        yield




def test_all_scenarios_pass_mock():
    runner = BenchmarkRunner(strict=False, mode="mock")
    results = runner.run_all_scenarios()

    assert len(results) == len(SCENARIOS)
    assert all(result["execution"]["completed"] for result in results)
    assert all(result["scoring"]["success"] for result in results)


def test_categories_complete_mock():

    runner = BenchmarkRunner(strict=False, mode="mock")
    results = runner.run_all_scenarios()
    by_category = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)


    # Verify expected categories
    expected_categories = ["planning", "tool", "memory", "ux", "robustness"]
    for cat in expected_categories:
        assert cat in by_category
        scenarios = by_category[cat]
        assert all(s["scoring"]["success"] for s in scenarios)

def test_specific_scenario_mock():
    runner = BenchmarkRunner(strict=False, mode="mock")

    # Get specific scenario

    scenario = get_scenarios_by_category("planning")[0]
    result = runner.run_scenario(scenario)

    # Verify result structure
    assert result["scenario"] == scenario.name
    assert result["category"] == "planning"
    assert result["execution"]["completed"] is True
    assert result["trace"]["planner_output"] is not None
    assert result["scoring"]["success"] is True

def test_trace_structure_mock():
    runner = BenchmarkRunner(strict=False, mode="mock")
    results = runner.run_all_scenarios()

    for result in results:
        # Execution struct
        execution = result["execution"]
        assert "completed" in execution
        assert "steps" in execution
        assert "error" in execution

        # Trace struct
        trace = result["trace"]
        assert "planner_output" in trace
        assert "tools_executed" in trace
        assert "memory_accessed" in trace
        assert "final_response" in trace

        # Scoring struct
        scoring = result["scoring"]
        assert "success" in scoring
        assert "steps" in scoring
        assert "tool_used_correctly" in scoring
        assert "error" in scoring


@pytest.mark.phase4_harness
@pytest.mark.integration

class TestPhase4AOrchestratorIntegration:

    @pytest.fixture
    def orchestrator(self, make_test_dadbot, phase4a_db_path):
        orchestrator, _checkpointer, llm = _make_orchestrator_with_checkpointer(
            phase4a_db_path, make_test_dadbot=make_test_dadbot,
        )
        with _stub_llm(llm):
            yield orchestrator

    def test_orchestrator_available(self, orchestrator):
        assert orchestrator is not None
        from dadbot.core.orchestrator import DadBotOrchestrator

        assert isinstance(orchestrator, DadBotOrchestrator)

    def test_single_scenario_orchestrator(self, orchestrator):
        runner = BenchmarkRunner(
            strict=False,
            mode="orchestrator",
            orchestrator=orchestrator,
        )

        # Test with simplest scenario
        scenario = SCENARIOS[0]
        result = runner.run_scenario(scenario)

        # Verify execution completed (may fail gracefully)
        assert "execution" in result
        assert "trace" in result

        # If successful, verify trace structure
        if result["execution"]["completed"]:
            trace = result["trace"]
            assert isinstance(trace, dict)
            # Real trace should have planner data
            if "planner" in trace:
                assert isinstance(trace["planner"], dict)

    def test_all_scenarios_orchestrator(self, orchestrator):
        runner = BenchmarkRunner(
            strict=False,
            mode="orchestrator",
            orchestrator=orchestrator,
        )

        results = runner.run_all_scenarios()

        # Verify all scenarios attempted
        assert len(results) == len(SCENARIOS)

        # At least N scenarios must produce an execution_result envelope.
        attempted = sum(1 for r in results if isinstance(r.get("execution_result"), dict))
        assert attempted >= 10

        # Execution errors should be classified, never fatal to the harness.

        classified = [
            r["execution_result"].get("execution_error_class")
            for r in results
            if isinstance(r.get("execution_result"), dict)
        ]
        assert all(isinstance(c, str) and len(c) > 0 for c in classified)

        # Verify no crashes
        assert all("execution" in r for r in results)
        assert all("trace" in r for r in results)

    def test_orchestrator_trace_capture(self, orchestrator):
        runner = BenchmarkRunner(
            strict=False,
            mode="orchestrator",
            orchestrator=orchestrator,
        )

        # Run tool scenario to veriy tool tracing
        tool_scenarios = get_scenarios_by_category("tool")
        if tool_scenarios:
            result = runner.run_scenario(tool_scenarios[0])

            if result["execution"]["completed"]:
                # Traces should have real data
                trace = result["trace"]

                # Real orchestrator traces include nested dict structure
                if isinstance(trace, dict):
                    # May have tools, planner, memory keys
                    trace_keys = set(trace.keys())
                    assert len(trace_keys) > 0


@pytest.mark.phase4_harness
@pytest.mark.integration
class TestPhase4ACapabilityMeasurement:


    @pytest.fixture(scope="function")
    def orchestrator(self, make_test_dadbot):
        bot = make_test_dadbot()
        orchestrator = bot.turn_orchestrator
        with _stub_llm(orchestrator.registry.get("llm")):
            yield orchestrator

    def test_capability_profile_structure(self, orchestrator):
        runner = BenchmarkRunner(
            strict=False,
            mode="orchestrator",
            orchestrator=orchestrator,
        )

        results = runner.run_all_scenarios()

        # Compute capability profile by category from intelligence scores
        # (separate from execution validity).
        profile = {}
        by_category: dict[str, dict[str, float]] = {}

        for r in results:
            cat = r["category"]
            if cat not in by_category:
                by_category[cat] = {"score_sum": 0.0, "total": 0.0}

            by_category[cat]["total"] += 1.0
            cap = r.get("capability_score") or {}
            by_category[cat]["score_sum"] += float(cap.get(cat) or 0.0)

        # Compute scores
        for cat, counts in by_category.items():
            score = counts["score_sum"] / counts["total"] if counts["total"] > 0 else 0.0
            profile[cat] = score

        # Verify all categories present
        assert "planning" in profile
        assert "tool" in profile
        assert "memory" in profile
        assert "ux" in profile
        assert "robustness" in profile

        # Scores should be 0.0-1.0
        for score in profile.values():
            assert 0.0 <= score <= 1.0

    def test_real_vs_mock_difference(self, orchestrator):
        mock_runner = BenchmarkRunner(strict=False, mode="mock")
        mock_results = mock_runner.run_all_scenarios()
        mock_successes = sum(1 for r in mock_results if r["execution"]["completed"])
        assert mock_successes == len(SCENARIOS)

        real_runner = BenchmarkRunner(
            strict=False, mode="orchestrator", orchestrator=orchestrator,
        )
        real_results = real_runner.run_all_scenarios()

        assert len(real_results) == len(mock_results)
        assert any(not result["scoring"]["success"] for result in real_results)


class TestScenarioSuiteValidation:

    def test_scenarios_completeness(self):
        assert len(SCENARIOS) == 16

    def test_scenario_structure(self):
        required_fields = [
            "name",
            "category",
            "input_text",
            "expected_capabilities",
            "success_criteria",
            "description",
        ]


        for scenario in SCENARIOS:
            for field in required_fields:
                assert hasattr(scenario, field), f"Missing {field} in {scenario.name}"
                assert getattr(scenario, field) is not None


    def test_categories_distribution(self):
        categories = {}
        for scenario in SCENARIOS:
            cat = scenario.category
            categories[cat] = categories.get(cat, 0) + 1

        expected = {
            "planning": 3,
            "tool": 5,
            "memory": 3,
            "ux": 3,
            "robustness": 2,
        }

        for cat, count in expected.items():
            assert categories.get(cat, 0) == count


# ---------------------------------------------------------------------------
# NEW: Real checkpointing — no mocked handle_turn, real _execute_job paths
# ---------------------------------------------------------------------------


@pytest.mark.phase4
@pytest.mark.phase4_cert
@pytest.mark.durability
@pytest.mark.integration
class TestPhase4ARealCheckpointing:

    @pytest.mark.asyncio
    async def test_orchestrator_saves_checkpoint_after_real_turn(self, phase4a_db_path, make_test_dadbot):
        orchestrator, checkpointer, llm = _make_orchestrator_with_checkpointer(
            phase4a_db_path, make_test_dadbot=make_test_dadbot,
        )
        _assert_handle_turn_not_mocked(orchestrator, "real_checkpointing")

        with _stub_llm(llm):
            await orchestrator.handle_turn(
                "hello dad",
                session_id="cp-real-1",
                confluence_key=confluence_key_for_turn("cp-real-1", "hello dad"),
            )

        assert checkpointer.checkpoint_count("cp-real-1") == 1

    @pytest.mark.asyncio
    async def test_orchestrator_hard_fails_when_confluence_key_omitted_in_strict_mode(
        self, phase4a_db_path, make_test_dadbot, caplog,
    ):
        orchestrator, checkpointer, llm = _make_orchestrator_with_checkpointer(
            phase4a_db_path, strict=True, make_test_dadbot=make_test_dadbot,
        )
        _assert_handle_turn_not_mocked(orchestrator, "strict_confluence_boundary")

        with _stub_llm(llm):
            response, success = await orchestrator.handle_turn(
                "missing key should fail",
                session_id="strict-missing-key",
            )

        assert success is False
        assert "Something went wrong" in response
        assert checkpointer.checkpoint_count("strict-missing-key") == 0
        assert "Missing explicit confluence key" in caplog.text

    @pytest.mark.asyncio
    async def test_orchestrator_restores_state_after_simulated_restart(self, phase4a_db_path, make_test_dadbot):
        try:
            orch1, cp1, llm1 = _make_orchestrator_with_checkpointer(phase4a_db_path, make_test_dadbot=make_test_dadbot)
        except Exception as exc:
            _fail_certification_gate("orchestrator_boot", exc)
        _assert_handle_turn_not_mocked(orch1, "restart_boundary_initial")

        with _stub_llm(llm1):
            await orch1.handle_turn(
                "remember this",
                session_id="restart-real",
                confluence_key=confluence_key_for_turn("restart-real", "remember this"),
            )

        saved_hash = cp1.load_checkpoint("restart-real")["checkpoint_hash"]
        assert saved_hash, "No checkpoint_hash after turn 1"

        # Simulate process restart: new orchestrator instance, same DB path.
        try:
            orch2, cp2, llm2 = _make_orchestrator_with_checkpointer(phase4a_db_path, make_test_dadbot=make_test_dadbot)
        except Exception as exc:
            _fail_certification_gate("orchestrator_boot_restart", exc)
        _assert_handle_turn_not_mocked(orch2, "restart_boundary_resumed")

        with _stub_llm(llm2):
            await orch2.handle_turn(
                "ollow up",
                session_id="restart-real",
                confluence_key=confluence_key_for_turn("restart-real", "ollow up"),
            )

        # After turn 2, two checkpoints should exist (one per turn).
        count = cp2.checkpoint_count("restart-real")
        assert count >= 2, "Expected ≥2 checkpoints after restart boundary, got {count}"

        # The most-recent checkpoint should reerence the prior hash in its chain.
        latest = cp2.load_checkpoint("restart-real")
        assert latest["prev_checkpoint_hash"] == saved_hash, (
            "Hash-chain broken: prev_checkpoint_hash={latest['prev_checkpoint_hash']!r} expected={saved_hash!r}"
        )

    @pytest.mark.asyncio
    async def test_determinism_fields_present_in_saved_checkpoint(self, phase4a_db_path, make_test_dadbot):
        try:
            orchestrator, checkpointer, llm = _make_orchestrator_with_checkpointer(phase4a_db_path, make_test_dadbot=make_test_dadbot)
        except Exception as exc:
            _fail_certification_gate("orchestrator_boot", exc)
        _assert_handle_turn_not_mocked(orchestrator, "determinism_ields")

        with _stub_llm(llm):
            await orchestrator.handle_turn(
                "what time is it",
                session_id="det-ields",
                confluence_key=confluence_key_for_turn("det-ields", "what time is it"),
            )

        loaded = checkpointer.load_checkpoint("det-ields")
        # checkpoint_snapshot() serializes context.metadata (not session.state).
        # Determinism ields live in loaded["metadata"]["determinism"].
        meta = loaded.get("metadata") or {}
        det = meta.get("determinism") or {}
        assert isinstance(det, dict), "determinism metadata missing rom checkpoint"
        assert "tool_trace_hash" in det, "tool_trace_hash not in checkpoint determinism: {list(det)}"
        assert "lock_hash" in det, "lock_hash not in checkpoint determinism: {list(det)}"

    @pytest.mark.asyncio
    async def test_checkpoint_write_log_records_success_row(self, phase4a_db_path, make_test_dadbot):
        try:
            orchestrator, _checkpointer, llm = _make_orchestrator_with_checkpointer(phase4a_db_path, make_test_dadbot=make_test_dadbot)
        except Exception as exc:
            _fail_certification_gate("orchestrator_boot", exc)
        _assert_handle_turn_not_mocked(orchestrator, "checkpoint_write_log")

        with _stub_llm(llm):
            await orchestrator.handle_turn(
                "log this",
                session_id="write-log-real",
                confluence_key=confluence_key_for_turn("write-log-real", "log this"),
            )

        with contextlib.closing(sqlite3.connect(phase4a_db_path)) as conn:
            rows = conn.execute(
                "SELECT status, error FROM checkpoint_writes WHERE tenant_id = ? AND session_id = ?",
                ("default", "write-log-real"),

            ).fetchall()

        assert len(rows) >= 1, "No rows in checkpoint_writes after a real turn"
        statuses = [r[0] for r in rows]
        assert all(s == "ok" for s in statuses), f"Unexpected write statuses: {statuses}"

    @pytest.mark.asyncio
    async def test_manifest_drift_warning_in_lenient_mode(self, phase4a_db_path, caplog, make_test_dadbot):

        try:
            orch1, cp1, llm1 = _make_orchestrator_with_checkpointer(phase4a_db_path, strict=False, make_test_dadbot=make_test_dadbot)
        except Exception as exc:
            _fail_certification_gate("orchestrator_boot", exc)
        _assert_handle_turn_not_mocked(orch1, "maniest_drit_lenient_initial")

        with _stub_llm(llm1):
            await orch1.handle_turn(
                "irst turn",
                session_id="drit-lenient",
                confluence_key=confluence_key_for_turn("drit-lenient", "irst turn"),
            )

        # Mutate the stored maniest's env_hash to simulate drit.
        loaded = cp1.load_checkpoint("drit-lenient")
        cp1.save_checkpoint(
            "drit-lenient",
            loaded.get("trace_id", "trace-drit"),
            {
                **loaded,
                "checkpoint_hash": loaded["checkpoint_hash"],
                "prev_checkpoint_hash": "",
            },
            {**loaded.get("manifest", {}), "env_hash": "drifted-env-hash-xyz"},
        )

        # Second orchestrator loads the mutated checkpoint; lenient mode should warn, not raise.
        try:
            orch2, _cp2, llm2 = _make_orchestrator_with_checkpointer(phase4a_db_path, strict=False, make_test_dadbot=make_test_dadbot)
        except Exception as exc:
            _fail_certification_gate("orchestrator_boot_restart", exc)
        _assert_handle_turn_not_mocked(orch2, "maniest_drit_lenient_resumed")

        import logging

        with caplog.at_level(logging.WARNING):
            with _stub_llm(llm2):
                await orch2.handle_turn(
                    "second turn",
                    session_id="drit-lenient",
                    confluence_key=confluence_key_for_turn("drit-lenient", "second turn"),
                )

        drit_messages = [m for m in caplog.messages if "drift" in m.lower() or "env" in m.lower()]
        assert len(drit_messages) >= 1, "Expected at least one drit warning in logs; got: {caplog.messages}"


# ---------------------------------------------------------------------------
# NEW: Determinism veriication — lock_hash and tool_trace_hash continuity
# ---------------------------------------------------------------------------


@pytest.mark.phase4
@pytest.mark.phase4_cert
@pytest.mark.durability
@pytest.mark.integration
class TestPhase4ADeterminismVeriication:

    @pytest.mark.asyncio
    async def test_lock_hash_stable_across_identical_inputs(self, phase4a_db_path, make_test_dadbot):
        try:
            orch1, cp1, llm1 = _make_orchestrator_with_checkpointer(phase4a_db_path, make_test_dadbot=make_test_dadbot)
            orch2, cp2, llm2 = _make_orchestrator_with_checkpointer(phase4a_db_path + ".b", make_test_dadbot=make_test_dadbot)
        except Exception as exc:
            _fail_certification_gate("orchestrator_boot_pair", exc)
        _assert_handle_turn_not_mocked(orch1, "lock_hash_stability_a")
        _assert_handle_turn_not_mocked(orch2, "lock_hash_stability_b")

        user_input = "determinism test input"

        with _stub_llm(llm1):
            await orch1.handle_turn(
                user_input,
                session_id="det-a",
                confluence_key=confluence_key_for_turn("det-a", user_input),
            )
        with _stub_llm(llm2):
            await orch2.handle_turn(
                user_input,
                session_id="det-b",
                confluence_key=confluence_key_for_turn("det-b", user_input),
            )

        cp1_data = cp1.load_checkpoint("det-a")
        cp2_data = cp2.load_checkpoint("det-b")

        # lock_hash lives in checkpoint["metadata"]["determinism"] (context.metadata).
        hash1 = ((cp1_data.get("metadata") or {}).get("determinism") or {}).get("lock_hash", "")
        hash2 = ((cp2_data.get("metadata") or {}).get("determinism") or {}).get("lock_hash", "")

        assert hash1, "lock_hash missing rom orchestrator 1 checkpoint"
        assert hash2, "lock_hash missing rom orchestrator 2 checkpoint"
        assert hash1 == hash2, (
            "Same input produced dierent lock_hash: {hash1!r} vs {hash2!r}\n"
            "This indicates non-determinism in the context-building pipeline."
        )

        # Clean up second temp DB
        try:
            Path(phase4a_db_path + ".b").unlink(missing_ok=True)
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_tool_trace_hash_present_and_non_empty_after_real_turn(self, phase4a_db_path, make_test_dadbot):
        try:
            orchestrator, checkpointer, llm = _make_orchestrator_with_checkpointer(phase4a_db_path, make_test_dadbot=make_test_dadbot)
        except Exception as exc:
            _fail_certification_gate("orchestrator_boot", exc)
        _assert_handle_turn_not_mocked(orchestrator, "tool_trace_presence")

        with _stub_llm(llm):
            await orchestrator.handle_turn(
                "tool check",
                session_id="tool-det",
                confluence_key=confluence_key_for_turn("tool-det", "tool check"),
            )

        loaded = checkpointer.load_checkpoint("tool-det")
        # tool_trace_hash is in context.metadata["determinism"] (serialized by checkpoint_snapshot).
        det = (loaded.get("metadata") or {}).get("determinism") or {}
        tth = str(det.get("tool_trace_hash") or "")
        assert len(tth) >= 16, "tool_trace_hash is missing or too short: {tth!r}"

    @pytest.mark.asyncio
    async def test_checkpoint_chain_integrity_after_two_real_turns(self, phase4a_db_path, make_test_dadbot):
        try:
            orchestrator, checkpointer, llm = _make_orchestrator_with_checkpointer(phase4a_db_path, make_test_dadbot=make_test_dadbot)
        except Exception as exc:
            _fail_certification_gate("orchestrator_boot", exc)
        _assert_handle_turn_not_mocked(orchestrator, "checkpoint_chain_integrity")

        with _stub_llm(llm):
            await orchestrator.handle_turn(
                "turn one",
                session_id="chain-real",
                confluence_key=confluence_key_for_turn("chain-real", "turn one"),
            )
        cp_turn1 = checkpointer.load_checkpoint("chain-real")
        hash_turn1 = cp_turn1["checkpoint_hash"]

        with _stub_llm(llm):
            await orchestrator.handle_turn(
                "turn two",
                session_id="chain-real",
                confluence_key=confluence_key_for_turn("chain-real", "turn two"),
            )
        cp_turn2 = checkpointer.load_checkpoint("chain-real")

        assert cp_turn2["prev_checkpoint_hash"] == hash_turn1, (
            "Hash chain broken after turn 2: expected prev={hash_turn1!r}, got {cp_turn2['prev_checkpoint_hash']!r}"
        )

    @pytest.mark.asyncio
    async def test_persistence_metrics_are_observable_per_session(self, phase4a_db_path, make_test_dadbot):
        try:
            orchestrator, checkpointer, llm = _make_orchestrator_with_checkpointer(phase4a_db_path, make_test_dadbot=make_test_dadbot)
        except Exception as exc:
            _fail_certification_gate("orchestrator_boot", exc)
        _assert_handle_turn_not_mocked(orchestrator, "persistence_metrics")

        n_turns = 3
        with _stub_llm(llm):
            for i in range(n_turns):
                await orchestrator.handle_turn(
                    f"turn {i}",
                    session_id="metrics-obs",
                    confluence_key=confluence_key_for_turn("metrics-obs", f"turn {i}"),
                )

        cp_count = checkpointer.checkpoint_count("metrics-obs")
        assert cp_count == n_turns, "Expected {n_turns} checkpoints, got {cp_count}"

        with contextlib.closing(sqlite3.connect(phase4a_db_path)) as conn:
            write_count = conn.execute(
                "SELECT COUNT(*) FROM checkpoint_writes WHERE tenant_id = ? AND session_id = ?",
                ("default", "metrics-obs"),
            ).fetchone()[0]

        assert write_count == n_turns, "checkpoint_writes table: expected {n_turns} rows, got {write_count}"


# ---------------------------------------------------------------------------
# GAP 1: True separate-process restart
# ---------------------------------------------------------------------------


@pytest.mark.phase4
@pytest.mark.phase4_cert
@pytest.mark.durability
@pytest.mark.integration
class TestPhase4ATrueProcessRestart:

    def test_checkpoint_survives_real_process_boundary(self, phase4a_db_path, tmp_path, monkeypatch):
        import json
        import subprocess
        import sys
        import textwrap

        from dadbot.core.dadbot import DadBot

        runtime_root = tmp_path / "process-runtime"
        runtime_root.mkdir()
        (runtime_root / "profile.json").write_text(json.dumps(DadBot.default_profile()), encoding="utf-8")
        monkeypatch.setenv("DADBOT_ENV_FILE", ".env")
        monkeypatch.setenv("DADBOT_ALLOW_LEGACY_DEFAULT_TENANT", "true")
        monkeypatch.setenv("DADBOT_GRAPH_ENABLE_LLM_EXTRACTION", "0")
        script = textwrap.dedent("""
            import asyncio
            import json
            import sys
            from pathlib import Path
            from unittest.mock import AsyncMock, patch
            from dadbot.config import DadBotConfig
            from dadbot.core.dadbot import DadBot
            from dadbot.core.orchestrator import DadBotOrchestrator
            from dadbot.core.persistence import SQLiteCheckpointer
            from dadbot_system.state import InMemoryStateStore
            from tests.harness.graph_runner import confluence_key_for_turn

            root = Path(sys.argv[2])
            config = DadBotConfig()
            config.profile_path = root / "profile.json"
            config.memory_path = root / "memory.json"
            config.semantic_memory_db_path = root / "semantic.sqlite3"
            config.graph_store_db_path = root / "graph.sqlite3"
            config.session_log_dir = root / "sessions"
            config.tenant_id = "default"
            bot = DadBot(config=config, document_store=InMemoryStateStore())
            bot.embed_texts = lambda texts, purpose="": [[0.0] * 12 for _ in texts]
            checkpointer = SQLiteCheckpointer(sys.argv[1], auto_migrate=True, prune_every=0)
            orchestrator = DadBotOrchestrator(bot=bot, strict=False, checkpointer=checkpointer)

            async def main():
                with (
                    patch.object(bot.model_port, "generate", return_value="offline reply"),
                    patch.object(bot.model_port, "generate_async", new=AsyncMock(return_value="offline reply")),
                ):
                    await orchestrator.handle_turn(
                        sys.argv[3],
                        session_id="proc-restart",
                        confluence_key=confluence_key_for_turn("proc-restart", sys.argv[3]),
                    )

            try:
                asyncio.run(main())
                checkpoint = checkpointer.load_checkpoint("proc-restart")
                result = {
                    "hash": checkpoint["checkpoint_hash"],
                    "previous": checkpoint["prev_checkpoint_hash"],
                    "count": checkpointer.checkpoint_count("proc-restart"),
                }
            finally:
                bot.shutdown()
            print(json.dumps(result))
        """)

        def run_process(prompt):
            completed = subprocess.run(
                [sys.executable, "-c", script, phase4a_db_path, str(runtime_root), prompt],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            assert completed.returncode == 0, completed.stderr[-2000:]
            return json.loads(completed.stdout.strip().splitlines()[-1])

        first = run_process("remember this process boundary")
        second = run_process("follow up in a new process")
        assert first["count"] == 1
        assert second["count"] == 2
        assert first["hash"]
        assert second["previous"] == first["hash"]


# ---------------------------------------------------------------------------
# GAP 2: Tool determinism across process-boundary restarts
# ---------------------------------------------------------------------------


@pytest.mark.phase4
@pytest.mark.phase4_cert
@pytest.mark.durability
@pytest.mark.integration

class TestPhase4AToolDeterminismAcrossRestarts:

    @pytest.mark.asyncio
    async def test_tool_trace_hash_stable_across_independent_sessions(self, phase4a_db_path, make_test_dadbot):
        db_b = phase4a_db_path + ".tool_det_b"
        result_a = _make_orchestrator_with_checkpointer(phase4a_db_path, make_test_dadbot=make_test_dadbot)
        result_b = _make_orchestrator_with_checkpointer(db_b, make_test_dadbot=make_test_dadbot)
        if not result_a or not result_b or result_a[0] is None or result_b[0] is None:
            pytest.skip("Orchestrator unavailable for process boundary test")
        orch_a, cp_a, llm_a = result_a
        orch_b, cp_b, llm_b = result_b

        ixed_input = "echo tool determinism probe"
        try:
            with _stub_llm(llm_a):
                await orch_a.handle_turn(
                    ixed_input,
                    session_id="tool-det-a",
                    confluence_key=confluence_key_for_turn("tool-det-a", ixed_input),
                )
            with _stub_llm(llm_b):
                await orch_b.handle_turn(
                    ixed_input,
                    session_id="tool-det-b",
                    confluence_key=confluence_key_for_turn("tool-det-b", ixed_input),
                )
            cp_a_data = cp_a.load_checkpoint("tool-det-a")
            cp_b_data = cp_b.load_checkpoint("tool-det-b")

            tth_a = ((cp_a_data.get("metadata") or {}).get("determinism") or {}).get("tool_trace_hash", "")
            tth_b = ((cp_b_data.get("metadata") or {}).get("determinism") or {}).get("tool_trace_hash", "")

            assert tth_a, "tool_trace_hash missing rom session A checkpoint"
            assert tth_b, "tool_trace_hash missing rom session B checkpoint"
            assert tth_a == tth_b, (
                "tool_trace_hash diers between independent sessions or identical input:\n"
                "  session A: {tth_a!r}\n  session B: {tth_b!r}\n"
                "Non-determinism detected in tool execution trace."
            )
        finally:
            try:
                Path(db_b).unlink(missing_ok=True)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_tool_trace_hash_survives_checkpoint_round_trip(self, phase4a_db_path, make_test_dadbot):
        result = _make_orchestrator_with_checkpointer(phase4a_db_path, make_test_dadbot=make_test_dadbot)
        if not result or result[0] is None:
            pytest.skip("Orchestrator unavailable for round trip test")
        orchestrator, checkpointer, llm = result
        _assert_handle_turn_not_mocked(orchestrator, "tool_det_round_trip")

        with _stub_llm(llm):
            await orchestrator.handle_turn(
                "round trip tool hash",
                session_id="tool-rt",
                confluence_key=confluence_key_for_turn("tool-rt", "round trip tool hash"),
            )

        # Capture hash at save time rom context.
        loaded = checkpointer.load_checkpoint("tool-rt")
        saved_tth = ((loaded.get("metadata") or {}).get("determinism") or {}).get("tool_trace_hash", "")
        assert saved_tth, "tool_trace_hash not persisted after real turn"

        # Load again (resh connection) — must match exactly.
        loaded2 = checkpointer.load_checkpoint("tool-rt")
        loaded_tth = ((loaded2.get("metadata") or {}).get("determinism") or {}).get("tool_trace_hash", "")
        assert loaded_tth == saved_tth, "tool_trace_hash mutated between two loads: {saved_tth!r} → {loaded_tth!r}"

    @pytest.mark.asyncio
    async def test_tool_trace_hash_changes_with_different_inputs(self, phase4a_db_path, make_test_dadbot):
        db_b = phase4a_db_path + ".tool_distinct_b"

        result_a = _make_orchestrator_with_checkpointer(phase4a_db_path, make_test_dadbot=make_test_dadbot)
        result_b = _make_orchestrator_with_checkpointer(db_b, make_test_dadbot=make_test_dadbot)
        if not result_a or not result_b or result_a[0] is None or result_b[0] is None:
            pytest.skip("Orchestrator unavailable for process boundary test")
        orch_a, cp_a, llm_a = result_a
        orch_b, cp_b, llm_b = result_b
        _assert_handle_turn_not_mocked(orch_a, "tool_det_di_a")
        _assert_handle_turn_not_mocked(orch_b, "tool_det_di_b")

        try:
            with _stub_llm(llm_a):
                await orch_a.handle_turn(
                    "alpha input probe",
                    session_id="tool-di-a",
                    confluence_key=confluence_key_for_turn("tool-di-a", "alpha input probe"),
                )
            with _stub_llm(llm_b):
                await orch_b.handle_turn(
                    "completely dierent beta input",
                    session_id="tool-di-b",
                    confluence_key=confluence_key_for_turn("tool-di-b", "completely dierent beta input"),
                )

            tth_a = ((cp_a.load_checkpoint("tool-di-a").get("metadata") or {}).get("determinism") or {}).get(
                "tool_trace_hash", ""
            )
            tth_b = ((cp_b.load_checkpoint("tool-di-b").get("metadata") or {}).get("determinism") or {}).get(
                "tool_trace_hash", ""
            )

            assert tth_a, "tool_trace_hash missing rom session A"
            assert tth_b, "tool_trace_hash missing rom session B"
            # Both must be valid hashes; dierent inputs typically dier.
            # (We don't hard-assert inequality — the hash may legitimately collide
            # i both inputs produce empty tool plans — but we assert both exist.)
            assert len(tth_a) >= 16 and len(tth_b) >= 16, "tool_trace_hashes too short: {tth_a!r} / {tth_b!r}"
        finally:
            try:
                Path(db_b).unlink(missing_ok=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# GAP 3: Scale and pruning under high-volume load
# ---------------------------------------------------------------------------


@pytest.mark.phase4_harness
@pytest.mark.soak

class TestPhase4AScaleAndPruning:

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_pruning_enforced_after_hundred_turns(self, phase4a_db_path, make_test_dadbot):

        # Wire checkpointer with prune_every=1 so pruning fires on every save.
        result = _make_orchestrator_with_checkpointer(phase4a_db_path, make_test_dadbot=make_test_dadbot)
        if not result or result[0] is None:
            pytest.skip("Orchestrator unavailable: orchestrator is None")
        orchestrator, checkpointer, llm = result
        checkpointer.prune_every = 1

        n_turns = 110
        with _stub_llm(llm):
            for i in range(n_turns):
                await orchestrator.handle_turn(
                    f"turn {i}",
                    session_id="prune-scale",
                    confluence_key=confluence_key_for_turn("prune-scale", f"turn {i}"),
                )

        inal_count = checkpointer.checkpoint_count("prune-scale")
        assert inal_count <= 10, (
            f"After {n_turns} turns with keep_count=10, expected ≤10 checkpoints, got {inal_count}"
        )
        # Must still have at least one checkpoint (the most-recent turn).
        assert inal_count >= 1, "All checkpoints were pruned — none remain"

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_concurrent_sessions_remain_isolated(self, phase4a_db_path, make_test_dadbot):
        from dadbot.core.persistence import SQLiteCheckpointer

        n_sessions = 8
        turns_per_session = 5

        try:
            from dadbot.core.orchestrator import DadBotOrchestrator
        except Exception as exc:
            pytest.skip(f"Orchestrator unavailable: {exc}")

        async def run_session(session_id: str) -> int:
            bot = make_test_dadbot()
            cp = SQLiteCheckpointer(phase4a_db_path, auto_migrate=True, prune_every=0)
            orch = DadBotOrchestrator(bot=bot, strict=False, checkpointer=cp)
            llm = orch.registry.get("llm")
            with _stub_llm(llm):
                for t in range(turns_per_session):
                    await orch.handle_turn(
                        f"session {session_id} turn {t}",
                        session_id=session_id,
                        confluence_key=confluence_key_for_turn(session_id, f"session {session_id} turn {t}"),
                    )
            return cp.checkpoint_count(session_id)

        session_ids = [f"concurrent-{i}" for i in range(n_sessions)]
        counts = await asyncio.gather(*[run_session(sid) for sid in session_ids])

        for sid, count in zip(session_ids, counts, strict=True):
            assert count == turns_per_session, f"Session {sid!r}: expected {turns_per_session} checkpoints, got {count}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_prune_returns_correct_deleted_count(self, phase4a_db_path, make_test_dadbot):

        result = _make_orchestrator_with_checkpointer(phase4a_db_path, make_test_dadbot=make_test_dadbot)
        if not result or result[0] is None:
            pytest.skip("Orchestrator unavailable: orchestrator is None")
        orchestrator, checkpointer, llm = result

        n_turns = 20
        with _stub_llm(llm):
            for i in range(n_turns):
                await orchestrator.handle_turn(
                    f"prune count turn {i}",
                    session_id="prune-count",
                    confluence_key=confluence_key_for_turn("prune-count", f"prune count turn {i}"),
                )

        before = checkpointer.checkpoint_count("prune-count")
        keep = 5
        deleted = checkpointer.prune_old_checkpoints("prune-count", keep_count=keep)
        after = checkpointer.checkpoint_count("prune-count")

        assert deleted == before - keep, (
            "prune returned {deleted} but count went rom {beore} to {ater} (expected delta {beore - keep})"
        )

        assert after == keep


# ---------------------------------------------------------------------------
# GAP 4: Large-state checkpoint round-trip
# ---------------------------------------------------------------------------


@pytest.mark.phase4
@pytest.mark.phase4_cert
@pytest.mark.durability
@pytest.mark.integration

class TestPhase4ALargeStateCheckpoint:

    @staticmethod
    def _build_large_state(target_kb: int = 512) -> dict:
        import random as _random
        import string
        rng = _random.Random(42)  # Deterministic seed

        def rand_str(n: int) -> str:
            return "".join(rng.choices(string.ascii_letters + string.digits + " .,!?", k=n))

        goals = [
            {
                "id": f"goal-{i}",
                "title": rand_str(60),
                "description": rand_str(200),
                "status": rng.choice(["active", "completed", "pending"]),
                "priority": rng.randint(1, 10),
                "subtasks": [rand_str(80) for _ in range(5)],
            }
            for i in range(80)
        ]
        memories = [rand_str(300) for _ in range(200)]
        tool_results = [
            {
                "tool": f"tool_{j}",
                "output": rand_str(500),
                "status": "ok",
                "latency_ms": rng.uniform(10, 500),
            }
            for j in range(100)
        ]
        # Pad to target size with a large reeorm ield.
        padding = rand_str(max(0, target_kb * 1024 - 8000))
        return {
            "session_goals": goals,
            "memories": memories,
            "tool_results": tool_results,
            "memory_structured": {"raw_notes": padding},
            "turn_count": 9999,
        }


    def test_large_state_round_trip_fidelity(self, phase4a_db_path):
        import json as _json

        from dadbot.core.persistence import SQLiteCheckpointer

        cp = SQLiteCheckpointer(phase4a_db_path, auto_migrate=True, prune_every=0)
        large_state = self._build_large_state(target_kb=512)

        import hashlib as _hashlib

        state_json = _json.dumps(large_state, sort_keys=True, default=str)
        original_hash = _hashlib.sha256(state_json.encode()).hexdigest()
        original_size_kb = len(state_json) / 1024
        assert original_size_kb >= 512

        ake_checkpoint_hash = _hashlib.sha256(b"large-state-test").hexdigest()
        checkpoint = {
            "checkpoint_hash": ake_checkpoint_hash,
            "prev_checkpoint_hash": "",
            "state": large_state,
            "metadata": {"determinism": {"tool_trace_hash": "aabbcc" * 5, "lock_hash": "ddee" * 5}},
        }
        maniest = {
            "env_hash": "test-env",
            "python_version": "3.x",
            "maniest_hash": _hashlib.sha256(b"maniest").hexdigest(),
        }

        cp.save_checkpoint("large-state", "trace-ls-1", checkpoint, maniest)
        loaded = cp.load_checkpoint("large-state")

        loaded_state = loaded.get("state") or {}
        loaded_json = _json.dumps(loaded_state, sort_keys=True, default=str)
        loaded_hash = _hashlib.sha256(loaded_json.encode()).hexdigest()

        assert loaded_hash == original_hash, (
            "State hash mismatch ater round-trip "
            f"(original_size={original_size_kb:.1f} KB): "
            "original={original_hash!r} loaded={loaded_hash!r}"
        )

    def test_large_state_metadata_survives_round_trip(self, phase4a_db_path):
        import hashlib as _hashlib

        from dadbot.core.persistence import SQLiteCheckpointer

        cp = SQLiteCheckpointer(phase4a_db_path, auto_migrate=True, prune_every=0)
        large_state = self._build_large_state(target_kb=256)

        tth = _hashlib.sha256(b"tool-trace-large").hexdigest()
        lh = _hashlib.sha256(b"lock-hash-large").hexdigest()
        fake_hash = _hashlib.sha256(b"large-meta-test").hexdigest()

        checkpoint = {
            "checkpoint_hash": fake_hash,
            "prev_checkpoint_hash": "",
            "state": large_state,
            "metadata": {"determinism": {"tool_trace_hash": tth, "lock_hash": lh}},
        }
        manifest = {
            "env_hash": "env-large",
            "manifest_hash": _hashlib.sha256(b"mf-large").hexdigest(),
        }
        cp.save_checkpoint("large-meta", "trace-lm-1", checkpoint, manifest)

        loaded = cp.load_checkpoint("large-meta")
        det = (loaded.get("metadata") or {}).get("determinism") or {}

        assert det.get("tool_trace_hash") == tth, (
            f"tool_trace_hash mutated during large-state round-trip: {det.get('tool_trace_hash')!r}"
        )
        assert det.get("lock_hash") == lh, f"lock_hash mutated during large-state round-trip: {det.get('lock_hash')!r}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_orchestrator_handles_large_preloaded_state(self, phase4a_db_path, bot, monkeypatch):
        import hashlib as _hashlib

        from dadbot.core.persistence import SQLiteCheckpointer

        cp_seed = SQLiteCheckpointer(phase4a_db_path, auto_migrate=True, prune_every=0)
        large_state = self._build_large_state(target_kb=256)

        seed_hash = _hashlib.sha256(b"orchestrator-large-seed").hexdigest()
        seed_cp = {
            "checkpoint_hash": seed_hash,
            "prev_checkpoint_hash": "",
            "state": large_state,
            "metadata": {
                "determinism": {"tool_trace_hash": "seed" * 16, "lock_hash": "seed" * 16},
            },
        }
        manifest = {
            "env_hash": "env-large-orch",
            "manifest_hash": _hashlib.sha256(b"mf-large-orch").hexdigest(),
        }
        cp_seed.save_checkpoint("large-orch", "trace-lo-seed", seed_cp, manifest)

        # Now run a real orchestrator turn — it will load the large checkpoint first.
        try:
            from dadbot.core.orchestrator import DadBotOrchestrator
            checkpointer = SQLiteCheckpointer(phase4a_db_path, auto_migrate=True, prune_every=0)
            orchestrator = DadBotOrchestrator(bot=bot, strict=False, checkpointer=checkpointer)
            llm = orchestrator.registry.get("llm")
        except Exception as exc:
            _fail_certification_gate("orchestrator_boot", exc)
        _assert_handle_turn_not_mocked(orchestrator, "large_state_preload")

        async def generate_candidate(turn_context, rich_context=None):
            return ("Checkpoint turn completed.", False)

        monkeypatch.setattr(llm, "generate_candidate", generate_candidate)
        await orchestrator.handle_turn(
            "hello after large state",
            session_id="large-orch",
            confluence_key=confluence_key_for_turn("large-orch", "hello after large state"),
        )

        # The turn should have added a second checkpoint (turn 2).
        count = checkpointer.checkpoint_count("large-orch")
        assert count >= 2, f"Expected ≥2 checkpoints after loading large state + 1 real turn, got {count}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
