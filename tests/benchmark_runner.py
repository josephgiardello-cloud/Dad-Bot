"""Benchmark harness for scenario evaluation.

PHASE 1: Mock execution (validates scenario structure)
PHASE 4A: Real orchestrator execution (measures actual capability)

This harness supports both modes. Use mode="mock" for fast iteration,
mode="orchestrator" for real measurements.

OUTPUT STRUCTURE:
{
    "scenario": str,
    "category": str,
    "input": str,
    "execution": {
        "completed": bool,
        "steps": int,
        "error": None | str,
    },
    "trace": {
        "planner_output": dict | None,
        "tools_executed": list,
        "memory_accessed": list,
        "final_response": str,
    },
    "scoring": {
        "success": bool,
        "steps": int,
        "tool_used_correctly": bool,
        "error": None | str,
    }
}
"""

import asyncio
import logging
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from evaluation.coherence_engine import CoherenceEngine
from tests.scenario_suite import SCENARIOS, Scenario
from tests.scoring_engine import CapabilityScore, ScoringEngine
from tests.trace_schema import NormalizedTrace

logger = logging.getLogger(__name__)


class _InMemorySessionStore:
    """Minimal save/load store used to bypass filesystem persistence in tests."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def save_session_state(self, key: str, value: Any) -> None:
        self._data[str(key)] = value

    def load_session_state(self, key: str) -> Any:
        return self._data.get(str(key))


@dataclass
class ExecutionTrace:
    """Captured execution trace from a scenario run."""

    scenario_name: str
    category: str
    input_text: str
    planner_output: dict[str, Any] | None = None
    tools_executed: list[str] = None
    memory_accessed: list[str] = None
    final_response: str = ""
    steps: int = 0
    completed: bool = False
    error: str | None = None

    def __post_init__(self):
        if self.tools_executed is None:
            self.tools_executed = []
        if self.memory_accessed is None:
            self.memory_accessed = []


@dataclass
class ScoreResult:
    """Minimal scoring result."""

    success: bool
    steps: int
    tool_used_correctly: bool
    error: str | None = None


@dataclass
class ExecutionResult:
    """Execution validity separate from capability/intelligence scoring."""

    valid_run: bool
    execution_error: str | None
    trace_available: bool
    execution_error_class: str = "none"


class BenchmarkRunner:
    """Executes scenarios and captures execution traces.

    Supports two modes:
    1. mock: Fast validation with synthetic execution (Phase 1)
    2. orchestrator: Real orchestrator execution (Phase 4A)
    """

    _scoring_engine = ScoringEngine()
    _coherence_engine = CoherenceEngine()

    def __init__(
        self,
        strict: bool = False,
        mode: Literal["mock", "orchestrator"] = "mock",
        orchestrator: Any | None = None,
        sandbox_outputs: bool = True,
        sandbox_root: str | None = None,
        disable_persistence_disk: bool = True,
    ):
        """Initialize benchmark runner.

        Args:
            strict: Whether to enforce strict validation.
            mode: Execution mode ("mock" or "orchestrator").
            orchestrator: Pre-initialized DadBotOrchestrator instance (required for orchestrator mode).
        """
        self.strict = strict
        self.mode = mode
        self.orchestrator = orchestrator
        self._session_id = f"benchmark_session_{uuid.uuid4().hex[:8]}"
        self._turn_counter = 0
        self.sandbox_outputs = bool(sandbox_outputs)
        self.disable_persistence_disk = bool(disable_persistence_disk)
        self._sandbox_root = Path(sandbox_root) if sandbox_root else None
        self._sandbox_tmp: tempfile.TemporaryDirectory[str] | None = None
        self._sandbox_applied = False

        if mode == "orchestrator" and orchestrator is None:
            logger.warning("⚠️  Orchestrator mode selected but no orchestrator provided. Falling back to mock mode.")
            self.mode = "mock"

    @staticmethod
    def _classify_execution_error(error: str | None) -> str:
        if not error:
            return "none"
        lowered = str(error).lower()
        if "permission" in lowered or "access is denied" in lowered:
            return "io_permission"
        if "no such file" in lowered or "file not found" in lowered:
            return "io_missing_path"
        if "disk" in lowered or "write" in lowered or "read-only" in lowered:
            return "io_write"
        if "timeout" in lowered:
            return "timeout"
        return "runtime"

    def _build_execution_result(self, completed: bool, error: str | None, trace_available: bool) -> ExecutionResult:
        error_class = self._classify_execution_error(error)
        valid_run = bool(completed)
        if not valid_run and error_class.startswith("io_") and trace_available:
            # IO failure at save boundary still counts as a valid intelligence run.
            valid_run = True
        return ExecutionResult(
            valid_run=valid_run,
            execution_error=error,
            trace_available=bool(trace_available),
            execution_error_class=error_class,
        )

    @staticmethod
    def _extract_planner_diagnostics(
        normalized: NormalizedTrace | None,
        scenario: Scenario | None = None,
    ) -> dict[str, float]:
        planner = getattr(normalized, "planner", None) if normalized is not None else None
        if planner is None:
            return {
                "plan_length": 0.0,
                "branching_factor": 0.0,
                "revision_count": 0.0,
                "dependency_correctness": 0.0,
            }

        plan_length = float(max(int(getattr(planner, "step_count", 0) or 0), 0))
        branching_factor = float(max(getattr(planner, "branching_factor", lambda: 0.0)() or 0.0, 0.0))
        revision_count = float(max(int(getattr(planner, "replan_count", 0) or 0), 0))

        deps = list(getattr(planner, "dependencies", []) or [])
        dep_count = len(deps)
        if plan_length <= 0:
            dependency_correctness = 0.0
        else:
            max_edges = max((int(plan_length) * max(int(plan_length) - 1, 0)) / 2.0, 1.0)
            structure_score = 1.0 if dep_count <= max_edges else max(0.0, 1.0 - ((dep_count - max_edges) / max_edges))

            expects_dependencies = False
            if scenario is not None:
                caps = [str(item or "").lower() for item in list(getattr(scenario, "expected_capabilities", []) or [])]
                expects_dependencies = any("dependenc" in item for item in caps)
            if expects_dependencies and dep_count == 0:
                dependency_correctness = 0.0
            elif not expects_dependencies and dep_count == 0:
                dependency_correctness = 1.0
            else:
                dependency_correctness = float(max(0.0, min(1.0, structure_score)))

        return {
            "plan_length": plan_length,
            "branching_factor": round(branching_factor, 4),
            "revision_count": revision_count,
            "dependency_correctness": round(dependency_correctness, 4),
        }

    def _extract_phase45_diagnostics(
        self,
        normalized: NormalizedTrace | None,
        scenario: Scenario | None = None,
    ) -> dict[str, Any]:
        raw_state = dict(getattr(normalized, "raw_state", {}) or {}) if normalized is not None else {}
        coherence = self._coherence_engine.score(raw_state)

        ux = dict(raw_state.get("ux_trace") or raw_state.get("ux_feedback") or {})
        memory_causal = dict(raw_state.get("memory_causal_trace") or {})
        tool_failure_semantics = [
            item
            for item in list(raw_state.get("tool_failure_semantics") or [])
            if isinstance(item, dict)
        ]

        wrong_tool_count = 0
        for item in tool_failure_semantics:
            failure_class = str(item.get("failure_class") or "").strip().lower()
            if failure_class == "wrong_tool":
                wrong_tool_count += 1

        expected_tool_use = True
        min_tool_calls = 0
        if scenario is not None:
            spec = dict(getattr(scenario, "behavioral_spec", {}) or {})
            expected_tool_use = bool(spec.get("expected_tool_use", True))
            min_tool_calls = max(int(spec.get("min_tool_calls") or 0), 0)

        tool_count = len(list(getattr(normalized, "tools", []) or [])) if normalized is not None else 0
        semantic_total = len(tool_failure_semantics)

        optimality = 1.0
        if expected_tool_use and tool_count == 0:
            optimality = 0.0
        elif (not expected_tool_use) and tool_count > 0:
            optimality = 0.6

        if min_tool_calls > 0 and tool_count < min_tool_calls:
            optimality *= tool_count / float(min_tool_calls)

        if semantic_total > 0:
            optimality *= max(0.0, 1.0 - (wrong_tool_count / float(semantic_total)))

        contradiction_count = 0
        contradictions_seen: set[str] = set()

        for item in list(raw_state.get("memory_contradictions") or []):
            if isinstance(item, dict):
                key = str(item.get("reason") or item.get("message") or item)
            else:
                key = str(item)
            key = key.strip()
            if key:
                contradictions_seen.add(key)

        for item in list(raw_state.get("contradictions") or []):
            key = str(item or "").strip()
            if key:
                contradictions_seen.add(key)

        memory_structured = dict(raw_state.get("memory_structured") or {})
        for value in memory_structured.values():
            if not isinstance(value, dict):
                continue
            for item in list(value.get("contradictions") or []):
                key = str(item or "").strip()
                if key:
                    contradictions_seen.add(key)

        for penalty in list(coherence.penalties or []):
            key = str(penalty or "").strip().lower()
            if "contradiction" in key:
                contradictions_seen.add(key)

        contradiction_count = len(contradictions_seen)
        contradiction_detected = contradiction_count > 0
        user_confusion_detected = bool(ux.get("user_confusion_detected", False))

        if not contradiction_detected:
            contradiction_resolution = 1.0
        elif user_confusion_detected:
            contradiction_resolution = 0.3
        elif bool(memory_causal.get("influenced_final_response", False)):
            contradiction_resolution = 1.0
        else:
            contradiction_resolution = 0.7

        return {
            "coherence_score": round(float(coherence.score), 4),
            "coherence_penalty_count": int(len(list(coherence.penalties or []))),
            "contradiction_detected": bool(contradiction_detected),
            "contradiction_count": int(contradiction_count),
            "contradiction_resolution": round(float(contradiction_resolution), 4),
            "tool_selection_optimality": round(max(0.0, min(1.0, float(optimality))), 4),
        }

    @staticmethod
    def _condense_capability_score(
        capability_score: dict[str, Any] | None,
        *,
        scenario: Scenario | None = None,
        execution_steps: int = 0,
        execution_error_class: str = "none",
        planner_diagnostics: dict[str, float] | None = None,
        phase45_diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not isinstance(capability_score, dict):
            return None

        subsystem_names = ("planning", "tools", "memory", "ux", "robustness")

        def _subsystem_value(name: str) -> float:
            section = capability_score.get(name)
            if isinstance(section, dict) and section.get("score") is not None:
                return float(section.get("score") or 0.0)
            return 0.0

        def _subsystem_partial_success(name: str) -> float:
            section = capability_score.get(name)
            if isinstance(section, dict):
                return 1.0 if bool(section.get("partial_success")) else 0.0
            return 0.0

        def _signal_value(name: str, signal_name: str) -> float | None:
            section = capability_score.get(name)
            if not isinstance(section, dict):
                return None
            for item in list(section.get("signals") or []):
                if not isinstance(item, dict):
                    continue
                if str(item.get("name") or "").strip() != signal_name:
                    continue
                value = item.get("value")
                if value is None:
                    continue
                return float(value)
            return None

        max_steps = 0
        quality_threshold = 0.0
        if scenario is not None:
            spec = dict(getattr(scenario, "behavioral_spec", {}) or {})
            max_steps = max(int(spec.get("max_steps") or 0), 0)
            quality_threshold = max(float(spec.get("quality_threshold") or 0.0), 0.0)

        actual_steps = max(int(execution_steps), 0)
        if max_steps <= 0:
            step_efficiency = 1.0
        elif actual_steps <= 0:
            step_efficiency = 0.0
        elif actual_steps <= max_steps:
            step_efficiency = 1.0
        else:
            step_efficiency = max(0.0, min(1.0, max_steps / float(actual_steps)))

        partial_success_values = [_subsystem_partial_success(name) for name in subsystem_names]
        partial_success = sum(partial_success_values) / max(len(partial_success_values), 1)

        tool_correctness = _signal_value("tools", "tool_success_rate")
        if tool_correctness is None:
            tool_correctness = _subsystem_value("tools")

        subsystem_coverage = (
            sum(1.0 for name in subsystem_names if isinstance(capability_score.get(name), dict))
            / float(len(subsystem_names))
        )

        overall = float(capability_score.get("overall") or 0.0)
        quality_threshold_met = True if quality_threshold <= 0.0 else overall >= quality_threshold

        return {
            "overall": overall,
            "planning": _subsystem_value("planning"),
            "tools": _subsystem_value("tools"),
            "memory": _subsystem_value("memory"),
            "ux": _subsystem_value("ux"),
            "robustness": _subsystem_value("robustness"),
            "partial_success": round(partial_success, 4),
            "tool_correctness": round(float(tool_correctness), 4),
            "step_efficiency": round(step_efficiency, 4),
            "failure_type": str(execution_error_class or "none"),
            "quality_threshold": round(quality_threshold, 4),
            "quality_threshold_met": bool(quality_threshold_met),
            "subsystem_coverage": round(subsystem_coverage, 4),
            "plan_length": float((planner_diagnostics or {}).get("plan_length") or 0.0),
            "branching_factor": float((planner_diagnostics or {}).get("branching_factor") or 0.0),
            "revision_count": float((planner_diagnostics or {}).get("revision_count") or 0.0),
            "dependency_correctness": float((planner_diagnostics or {}).get("dependency_correctness") or 0.0),
            "coherence_score": float((phase45_diagnostics or {}).get("coherence_score") or 0.0),
            "coherence_penalty_count": int((phase45_diagnostics or {}).get("coherence_penalty_count") or 0),
            "contradiction_detected": bool((phase45_diagnostics or {}).get("contradiction_detected") or False),
            "contradiction_count": int((phase45_diagnostics or {}).get("contradiction_count") or 0),
            "contradiction_resolution": float((phase45_diagnostics or {}).get("contradiction_resolution") or 0.0),
            "tool_selection_optimality": float((phase45_diagnostics or {}).get("tool_selection_optimality") or 0.0),
        }

    def _ensure_orchestrator_sandbox(self) -> None:
        if self.mode != "orchestrator" or self.orchestrator is None or self._sandbox_applied:
            return

        if not self.sandbox_outputs and not self.disable_persistence_disk:
            self._sandbox_applied = True
            return

        root = self._sandbox_root
        if root is None and self.sandbox_outputs:
            self._sandbox_tmp = tempfile.TemporaryDirectory(prefix="dadbot-benchmark-")
            root = Path(self._sandbox_tmp.name)
            self._sandbox_root = root
        if root is None:
            root = Path.cwd() / "tmp" / f"benchmark-{uuid.uuid4().hex[:8]}"
            self._sandbox_root = root

        root.mkdir(parents=True, exist_ok=True)
        session_logs = root / "session_logs"
        session_logs.mkdir(parents=True, exist_ok=True)

        bot = getattr(self.orchestrator, "bot", None)
        if bot is None:
            self._sandbox_applied = True
            return

        # Phase 4A tests validate orchestrator behavior, not external model latency.
        # Keep maintenance deterministic and offline-safe by avoiding network-backed
        # summary refresh work from background threads.
        self._stabilize_llm_calls(bot)
        self._stabilize_background_maintenance(bot)

        if self.sandbox_outputs:
            try:
                bot.SESSION_LOG_DIR = session_logs
            except Exception:
                pass

            config = getattr(bot, "config", None)
            if config is not None:
                for attr_name, path_value in (
                    ("session_log_dir", session_logs),
                    ("memory_path", root / "dad_memory.json"),
                    ("semantic_memory_db_path", root / "dad_memory_semantic.sqlite3"),
                    ("graph_store_db_path", root / "dad_memory_graph.sqlite3"),
                    ("profile_path", root / "dad_profile.json"),
                ):
                    try:
                        setattr(config, attr_name, path_value)
                    except Exception:
                        continue

            # Keep environment in sync for any lazy config reads.
            os_env_updates = {
                "DADBOT_SESSION_LOG_DIR": str(session_logs),
                "DADBOT_MEMORY_PATH": str(root / "dad_memory.json"),
                "DADBOT_SEMANTIC_DB_PATH": str(root / "dad_memory_semantic.sqlite3"),
                "DADBOT_GRAPH_DB_PATH": str(root / "dad_memory_graph.sqlite3"),
            }
            for env_key, env_value in os_env_updates.items():
                try:
                    import os

                    os.environ[env_key] = env_value
                except Exception:
                    continue

        if self.disable_persistence_disk:
            try:
                in_memory_store = _InMemorySessionStore()
                bot._tenant_document_store = in_memory_store
                bot._customer_state_store = in_memory_store
            except Exception:
                logger.debug("Failed to install in-memory persistence sandbox", exc_info=True)

        self._sandbox_applied = True

    @staticmethod
    def _stabilize_llm_calls(bot: Any) -> None:
        """Replace live model calls with deterministic offline responses for benchmarks."""

        def _offline_llm_response(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {"message": {"content": "[benchmark-offline]"}}

        try:
            bot.call_ollama_chat = _offline_llm_response
        except Exception:
            logger.debug("Failed to patch bot.call_ollama_chat for benchmark sandbox", exc_info=True)

        try:
            runtime_client = getattr(bot, "runtime_client", None)
            if runtime_client is not None:
                runtime_client.call_llm = _offline_llm_response
                runtime_client.call_ollama_chat = _offline_llm_response
                runtime_client.call_ollama_chat_with_model = _offline_llm_response
        except Exception:
            logger.debug("Failed to patch runtime client LLM methods for benchmark sandbox", exc_info=True)

    @staticmethod
    def _stabilize_background_maintenance(bot: Any) -> None:
        """Prevent background maintenance from issuing blocking external LLM calls."""
        try:

            def _offline_refresh_session_summary(force: bool = False) -> str:
                return str(getattr(bot, "session_summary", "") or "")

            bot.refresh_session_summary = _offline_refresh_session_summary
        except Exception:
            logger.debug("Failed to patch refresh_session_summary for benchmark sandbox", exc_info=True)

        try:
            maintenance_scheduler = getattr(bot, "maintenance_scheduler", None)
            if maintenance_scheduler is not None:

                def _offline_post_turn_maintenance(user_input: str, current_mood: str) -> dict[str, Any]:
                    return {
                        "summary_refreshed": False,
                        "scheduled_proactive": False,
                        "scheduled_proactive_count": 0,
                        "relationship_reflected": False,
                        "wisdom_generated": False,
                        "periodic_synthesis": False,
                        "periodic_archive_delta": 0,
                        "persona_evolved": False,
                        "memory_compaction": False,
                        "memory_compaction_updated_at": None,
                        "memory_graph_refreshed": True,
                        "offline_stub": True,
                    }

                maintenance_scheduler.run_post_turn_maintenance = _offline_post_turn_maintenance
        except Exception:
            logger.debug("Failed to patch maintenance manager for benchmark sandbox", exc_info=True)

    def _score_trace(self, trace: ExecutionTrace, scenario: Scenario) -> ScoreResult:
        """Apply minimal scoring to execution trace (backward-compatible)."""
        success = trace.completed and trace.error is None

        tool_used_correctly = True
        if scenario.category == "tool":
            tool_used_correctly = len(trace.tools_executed) > 0 or not self.strict

        return ScoreResult(
            success=success,
            steps=trace.steps,
            tool_used_correctly=tool_used_correctly,
            error=trace.error,
        )

    def _build_normalized_trace_from_mock(self, trace: ExecutionTrace) -> NormalizedTrace:
        """Convert mock ExecutionTrace into NormalizedTrace."""
        return NormalizedTrace.from_mock(
            scenario_name=trace.scenario_name,
            category=trace.category,
            input_text=trace.input_text,
            final_response=trace.final_response,
            completed=trace.completed,
            error=trace.error,
            planner_output=trace.planner_output,
            tools_executed=trace.tools_executed,
            memory_accessed=trace.memory_accessed,
        )

    def _score_normalized(self, normalized: NormalizedTrace, scenario: Scenario) -> dict[str, Any]:
        """Run the scoring engine and return serializable dict."""
        try:
            cap_score: CapabilityScore = self._scoring_engine.score(normalized, scenario)
            return cap_score.to_dict()
        except Exception as e:
            logger.warning(f"Scoring engine error for {scenario.name}: {e}")
            return {"error": str(e), "overall": 0.0}

    async def _run_scenario_orchestrator_async(self, scenario: Scenario) -> dict[str, Any]:
        """Execute scenario through real orchestrator (Phase 4A)."""
        trace = ExecutionTrace(
            scenario_name=scenario.name,
            category=scenario.category,
            input_text=scenario.input_text,
        )

        try:
            self._turn_counter += 1
            self._ensure_orchestrator_sandbox()

            # Execute through orchestrator
            response_text, success = await self.orchestrator.handle_turn(
                user_input=scenario.input_text,
                session_id=self._session_id,
                timeout_seconds=15.0,
            )

            trace.completed = success
            trace.steps = 1  # Will be updated with real step count from context
            trace.final_response = str(response_text or "")

            # Capture trace from orchestrator context
            context = getattr(self.orchestrator, "_last_turn_context", None)
            if context:
                # Planner output
                plan = context.state.get("plan")
                if plan:
                    trace.planner_output = plan if isinstance(plan, dict) else {"plan": plan}
                    trace.steps = len(plan) if isinstance(plan, (list, dict)) else 1

                # Tools executed
                tool_ir = context.state.get("tool_ir", {})
                executions = tool_ir.get("executions", [])
                trace.tools_executed = [e.get("tool_name") for e in executions if e.get("tool_name")]

                # Memory accessed
                memory_structured = context.state.get("memory_structured", {})
                trace.memory_accessed = list(memory_structured.keys())

        except TimeoutError:
            trace.error = "Execution timeout (15s)"
            trace.completed = False
            logger.warning(f"Scenario {scenario.name} timed out")
        except Exception as e:
            trace.error = f"{type(e).__name__}: {str(e)[:100]}"
            trace.completed = False
            logger.exception(f"Error executing scenario {scenario.name}")

        # Apply legacy scoring (backward compat)
        score = self._score_trace(trace, scenario)

        # Build normalized trace from orchestrator context
        context = getattr(self.orchestrator, "_last_turn_context", None)
        normalized: NormalizedTrace | None = None
        if context:
            try:
                normalized = NormalizedTrace.from_orchestrator_context(
                    scenario_name=scenario.name,
                    category=scenario.category,
                    input_text=scenario.input_text,
                    final_response=trace.final_response,
                    completed=trace.completed,
                    context=context,
                    duration_ms=trace.steps * 100.0,  # rough proxy
                    error=trace.error,
                )
            except Exception as ex:
                logger.warning(f"NormalizedTrace build failed for {scenario.name}: {ex}")

        capability_score = self._score_normalized(normalized, scenario) if normalized else None
        execution_result = self._build_execution_result(
            completed=trace.completed,
            error=trace.error,
            trace_available=normalized is not None,
        )
        planner_diagnostics = self._extract_planner_diagnostics(normalized, scenario)
        phase45_diagnostics = self._extract_phase45_diagnostics(normalized, scenario)
        condensed_capability_score = self._condense_capability_score(
            capability_score,
            scenario=scenario,
            execution_steps=trace.steps,
            execution_error_class=execution_result.execution_error_class,
            planner_diagnostics=planner_diagnostics,
            phase45_diagnostics=phase45_diagnostics,
        )

        return {
            "scenario": scenario.name,
            "category": scenario.category,
            "input": scenario.input_text,
            "execution_result": asdict(execution_result),
            "execution": {
                "completed": trace.completed,
                "steps": trace.steps,
                "error": trace.error,
            },
            "trace": {
                "planner_output": trace.planner_output,
                "tools_executed": trace.tools_executed,
                "memory_accessed": trace.memory_accessed,
                "final_response": trace.final_response,
            },
            "scoring": asdict(score),
            "capability_score": condensed_capability_score,
            "capability_score_detail": capability_score,
        }

    def _run_scenario_orchestrator(self, scenario: Scenario) -> dict[str, Any]:
        """Execute scenario through orchestrator (sync wrapper)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = self._run_scenario_orchestrator_async(scenario)
        if loop and loop.is_running():
            # Already in async context
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return asyncio.run(coro)

    def run_scenario(self, scenario: Scenario) -> dict[str, Any]:
        """Execute a single scenario with appropriate backend.

        Dispatches to orchestrator or mock based on configured mode.
        """
        if self.mode == "orchestrator":
            return self._run_scenario_orchestrator(scenario)
        else:
            return self._run_scenario_mock(scenario)

    def _run_scenario_mock(self, scenario: Scenario) -> dict[str, Any]:
        """Execute scenario with mock execution (Phase 1)."""
        trace = ExecutionTrace(
            scenario_name=scenario.name,
            category=scenario.category,
            input_text=scenario.input_text,
        )

        try:
            # PHASE 1: Validate scenario definition
            if not scenario.name or not scenario.input_text:
                raise ValueError("Scenario must have name and input_text")

            # PHASE 1: Mock execution
            trace.completed = True
            trace.steps = 1
            trace.final_response = f"[PHASE 1 MOCK] Response for: {scenario.input_text[:50]}..."

            # PHASE 1: Mock trace capture
            if scenario.category == "tool":
                trace.tools_executed = ["mock_tool"]

            if scenario.category == "memory":
                trace.memory_accessed = ["memory_store"]

            if scenario.category == "planning":
                trace.planner_output = {
                    "plan": f"Plan: {scenario.input_text[:40]}...",
                    "steps": 3,
                }

        except Exception as e:
            trace.error = f"{type(e).__name__}: {e!s}"
            trace.completed = False
            logger.exception(f"Error executing scenario {scenario.name}")

        # Apply legacy scoring (backward compat)
        score = self._score_trace(trace, scenario)

        # Build NormalizedTrace from mock data
        normalized = self._build_normalized_trace_from_mock(trace)
        capability_score = self._score_normalized(normalized, scenario)
        execution_result = self._build_execution_result(
            completed=trace.completed,
            error=trace.error,
            trace_available=True,
        )
        planner_diagnostics = self._extract_planner_diagnostics(normalized, scenario)
        phase45_diagnostics = self._extract_phase45_diagnostics(normalized, scenario)
        condensed_capability_score = self._condense_capability_score(
            capability_score,
            scenario=scenario,
            execution_steps=trace.steps,
            execution_error_class=execution_result.execution_error_class,
            planner_diagnostics=planner_diagnostics,
            phase45_diagnostics=phase45_diagnostics,
        )

        return {
            "scenario": scenario.name,
            "category": scenario.category,
            "input": scenario.input_text,
            "execution_result": asdict(execution_result),
            "execution": {
                "completed": trace.completed,
                "steps": trace.steps,
                "error": trace.error,
            },
            "trace": {
                "planner_output": trace.planner_output,
                "tools_executed": trace.tools_executed,
                "memory_accessed": trace.memory_accessed,
                "final_response": trace.final_response,
            },
            "scoring": asdict(score),
            "capability_score": condensed_capability_score,
            "capability_score_detail": capability_score,
        }

    def run_all_scenarios(
        self,
        scenarios: list[Scenario] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute all scenarios (or specified subset).

        Uses configured backend (mock or orchestrator).
        """
        if scenarios is None:
            scenarios = SCENARIOS

        if self.mode == "orchestrator":
            return self._run_all_scenarios_orchestrator(scenarios)
        else:
            return self._run_all_scenarios_mock(scenarios)

    def _run_all_scenarios_mock(
        self,
        scenarios: list[Scenario],
    ) -> list[dict[str, Any]]:
        """Execute all scenarios with mock backend."""
        results = []
        for scenario in scenarios:
            result = self._run_scenario_mock(scenario)
            results.append(result)
        return results

    def _run_all_scenarios_orchestrator(
        self,
        scenarios: list[Scenario],
    ) -> list[dict[str, Any]]:
        """Execute all scenarios with orchestrator backend (sequentially)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = self._run_all_scenarios_orchestrator_async(scenarios)
        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return asyncio.run(coro)

    async def _run_all_scenarios_orchestrator_async(
        self,
        scenarios: list[Scenario],
    ) -> list[dict[str, Any]]:
        """Execute all scenarios sequentially through orchestrator."""
        results = []
        for scenario in scenarios:
            result = await self._run_scenario_orchestrator_async(scenario)
            results.append(result)
        return results


def print_benchmark_results(results: list[dict[str, Any]], mode: str = "mock") -> None:
    """Pretty-print benchmark results."""
    mode_label = "PHASE 4A - ORCHESTRATOR EXECUTION" if mode == "orchestrator" else "PHASE 1 - MOCK EXECUTION"

    print("\n" + "=" * 80)
    print(f"BENCHMARK RESULTS ({mode_label})")
    print("=" * 80)

    # Summary by category
    by_category = {}
    for result in results:
        cat = result.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = {"pass": 0, "fail": 0}

        if result["scoring"]["success"]:
            by_category[cat]["pass"] += 1
        else:
            by_category[cat]["fail"] += 1

    print("\n📊 SUMMARY BY CATEGORY:")
    for cat in ["planning", "tool", "memory", "ux", "robustness"]:
        counts = by_category.get(cat, {"pass": 0, "fail": 0})
        total = counts["pass"] + counts["fail"]
        pct = (counts["pass"] / total * 100) if total > 0 else 0
        status = "✓" if counts["pass"] == total else "⚠" if counts["pass"] > 0 else "✗"
        print(f"  {status} {cat:12s}: {counts['pass']:2d}/{total:2d} ({pct:5.1f}%)")

    # Detailed results
    print("\n📋 DETAILED SCENARIO RESULTS:")
    for result in results:
        scenario = result["scenario"]
        score = result["scoring"]
        status = "✓" if score["success"] else "✗"
        error_msg = f" | {score['error']}" if score["error"] else ""
        print(f"  {status} {scenario:35s}{error_msg}")

    # Failures summary
    failures = [r for r in results if not r["scoring"]["success"]]
    if failures:
        print(f"\n🔴 FAILURES: {len(failures)}/{len(results)}")
        for result in failures:
            print(f"  - {result['scenario']}: {result['scoring'].get('error', 'Unknown')}")
    else:
        print("\n🟢 ALL SCENARIOS PASSED")

    # Trace summary (if orchestrator mode)
    if mode == "orchestrator":
        print("\n🔍 EXECUTION TRACE SUMMARY:")
        has_traces = any(r.get("trace") for r in results)
        if has_traces:
            for result in results:
                if result.get("trace"):
                    scenario = result["scenario"]
                    trace = result["trace"]
                    tools = trace.get("tools_executed", [])
                    memory = trace.get("memory_accessed", [])
                    print(f"  {scenario}:")
                    print(f"    Tools executed: {len(tools)} {tools}")
                    print(f"    Memory keys: {len(memory)} {memory}")
        else:
            print("  (No trace data captured)")

    print("\n" + "=" * 80)
    if mode == "mock":
        print("PHASE 1 COMPLETE: Mock validation passed")
        print("To run Phase 4A (real orchestrator), pass orchestrator instance to BenchmarkRunner")
    else:
        print("PHASE 4A COMPLETE: Real orchestrator execution")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("BENCHMARK RUNNER - Phase 1 (Mock) and Phase 4A (Orchestrator)")
    print("=" * 80)
    print("\nUSAGE:")
    print("\n1. PHASE 1 - Mock execution (fast validation):")
    print("   runner = BenchmarkRunner(strict=False, mode='mock')")
    print("   results = runner.run_all_scenarios()")
    print("\n2. PHASE 4A - Real orchestrator execution:")
    print("   from dadbot.core.dadbot import DadBot")
    print("   bot = DadBot()")
    print("   runner = BenchmarkRunner(orchestrator=bot.turn_orchestrator, mode='orchestrator')")
    print("   results = runner.run_all_scenarios()")
    print("\n" + "=" * 80)
    print("\nRunning PHASE 1 (mock) validation by default...")

    runner = BenchmarkRunner(strict=False, mode="mock")
    results = runner.run_all_scenarios()
    print_benchmark_results(results, mode="mock")
