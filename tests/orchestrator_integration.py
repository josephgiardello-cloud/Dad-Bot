"""Phase 4A: Orchestrator Integration - Real Execution Layer

Injects scenarios through DadBotOrchestrator and captures real traces.
This is where synthetic metrics become real intelligence profiles.

Key difference from Phase 1:
- Phase 1: Mock execution (validates scenario structure)
- Phase 4A: Real orchestrator execution (measures agent capability)
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple

from dadbot.core.orchestrator import DadBotOrchestrator
from dadbot.core.graph import TurnContext
from tests.scenario_suite import Scenario, SCENARIOS

logger = logging.getLogger(__name__)


class OrchestratorIntegrationLayer:
    """Wraps DadBotOrchestrator for scenario-based testing.
    
    Responsibilities:
    1. Inject scenario inputs through orchestrator
    2. Capture execution traces from TurnContext
    3. Extract capability signals (planner, tools, memory)
    4. Return structured evaluation data
    """

    def __init__(self, orchestrator: Optional[DadBotOrchestrator] = None):
        """Initialize integration layer.
        
        Args:
            orchestrator: Pre-built DadBotOrchestrator instance.
                        If None, will be bootstrapped on first execution.
        """
        self.orchestrator = orchestrator
        self._session_id = "benchmark_session"
        self._turn_counter = 0

    def _try_bootstrap_orchestrator(self) -> bool:
        """Attempt to bootstrap orchestrator from bot runtime.
        
        Returns:
            True if successful, False if orchestrator unavailable
        """
        if self.orchestrator is not None:
            return True
        
        try:
            # Try to get orchestrator from currently running bot
            # This relies on the bot being already initialized elsewhere
            from dadbot.core.dadbot import DadBot
            
            # Check if DadBot singleton is already running
            # (This is implementation-dependent; adjust based on actual bot architecture)
            logger.warning(
                "Orchestrator not provided. Attempting bootstrap from runtime. "
                "For best results, pass a pre-initialized orchestrator."
            )
            return False
        except Exception as e:
            logger.warning(f"Bootstrap failed: {e}")
            return False

    def _capture_planner_data(self, context: TurnContext) -> Dict[str, Any]:
        """Extract planner output and diagnostics from context.
        
        Captures:
        - Raw plan structure
        - Detected goals
        - Task decomposition
        - Decision points
        """
        plan_data = {}
        
        # Raw plan
        plan = context.state.get("plan")
        if plan:
            plan_data["plan"] = plan
            if isinstance(plan, (list, dict)):
                plan_data["plan_length"] = len(plan) if isinstance(plan, list) else 1
        
        # Goals
        goals = context.state.get("detected_goals", [])
        plan_data["detected_goals"] = goals
        plan_data["goal_count"] = len(goals)
        
        # Task decomposition
        decomposition = context.state.get("task_decomposition", {})
        plan_data["decomposition"] = decomposition
        if decomposition:
            plan_data["task_count"] = len(decomposition.get("tasks", []))
            plan_data["dependencies"] = decomposition.get("dependencies", [])
        
        return plan_data

    def _capture_tool_trace(self, context: TurnContext) -> Dict[str, Any]:
        """Extract tool execution trace from context.
        
        Captures:
        - Tools requested vs executed
        - Tool success rates
        - Failures and retries
        - Execution sequence
        """
        tool_data = {}
        
        # Tool IR (Intermediate Representation)
        tool_ir = context.state.get("tool_ir", {})
        
        # Execution plan
        execution_plan = tool_ir.get("execution_plan", [])
        tool_data["tools_planned"] = [t.get("tool_name") for t in execution_plan]
        tool_data["plan_count"] = len(execution_plan)
        
        # Actual executions
        executions = tool_ir.get("executions", [])
        tool_data["tools_executed"] = [e.get("tool_name") for e in executions]
        tool_data["execution_count"] = len(executions)
        
        # Results
        results = context.state.get("tool_results", [])
        tool_data["results_captured"] = len(results)
        
        # Success rate
        successful = sum(1 for r in results if r.get("status") == "success")
        tool_data["success_count"] = successful
        tool_data["success_rate"] = (
            successful / len(results) if results else 0.0
        )
        
        return tool_data

    def _capture_memory_state(self, context: TurnContext) -> Dict[str, Any]:
        """Extract memory access and state from context.
        
        Captures:
        - Memory keys accessed
        - Memory size
        - Retrieval patterns
        """
        memory_data = {}
        
        # Structured memory
        memory_structured = context.state.get("memory_structured", {})
        memory_data["memory_accessed"] = list(memory_structured.keys())
        memory_data["memory_key_count"] = len(memory_structured)
        
        # Full history ID
        memory_data["history_id"] = context.state.get("memory_full_history_id", "")
        
        # Session goals
        session_goals = context.state.get("session_goals", [])
        memory_data["session_goals"] = len(session_goals)
        
        return memory_data

    def _capture_safety_check(self, context: TurnContext) -> Dict[str, Any]:
        """Extract safety/policy check results.
        
        Captures:
        - Safety decision
        - Policy violations (if any)
        """
        safety_data = {}
        
        # Safety state
        safety_state = context.state.get("safety_check_result", {})
        safety_data["safe"] = safety_state.get("safe", True)
        safety_data["violations"] = safety_state.get("violations", [])
        
        return safety_data

    def _capture_full_trace(self, context: TurnContext, scenario: Scenario) -> Dict[str, Any]:
        """Aggregate all execution signals into structured trace."""
        return {
            "planner": self._capture_planner_data(context),
            "tools": self._capture_tool_trace(context),
            "memory": self._capture_memory_state(context),
            "safety": self._capture_safety_check(context),
            "metadata": {
                "trace_id": context.trace_id,
                "phase": str(getattr(context.phase, "value", context.phase) or "unknown"),
            },
        }

    async def execute_scenario_async(self, scenario: Scenario) -> Dict[str, Any]:
        """Execute a single scenario through real orchestrator.
        
        Returns comprehensive execution result with trace.
        """
        if not self._try_bootstrap_orchestrator():
            return {
                "scenario": scenario.name,
                "category": scenario.category,
                "input": scenario.input_text,
                "execution": {
                    "completed": False,
                    "error": "Orchestrator not available",
                },
                "trace": None,
                "response": "",
            }

        self._turn_counter += 1
        try:
            # Execute through orchestrator
            response_text, success = await self.orchestrator.handle_turn(
                user_input=scenario.input_text,
                session_id=self._session_id,
                timeout_seconds=15.0,
            )

            # Capture execution trace
            context = getattr(self.orchestrator, "_last_turn_context", None)
            trace = None
            if context:
                trace = self._capture_full_trace(context, scenario)

            return {
                "scenario": scenario.name,
                "category": scenario.category,
                "input": scenario.input_text,
                "execution": {
                    "completed": success,
                    "error": None,
                    "turn_number": self._turn_counter,
                },
                "trace": trace,
                "response": str(response_text or ""),
            }

        except asyncio.TimeoutError:
            return {
                "scenario": scenario.name,
                "category": scenario.category,
                "input": scenario.input_text,
                "execution": {
                    "completed": False,
                    "error": "Execution timeout",
                },
                "trace": None,
                "response": "",
            }
        except Exception as e:
            logger.exception(f"Error executing scenario {scenario.name}")
            return {
                "scenario": scenario.name,
                "category": scenario.category,
                "input": scenario.input_text,
                "execution": {
                    "completed": False,
                    "error": f"{type(e).__name__}: {str(e)[:100]}",
                },
                "trace": None,
                "response": "",
            }

    def execute_scenario(self, scenario: Scenario) -> Dict[str, Any]:
        """Execute scenario synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = self.execute_scenario_async(scenario)
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return asyncio.run(coro)

    async def execute_all_scenarios_async(
        self,
        scenarios: Optional[List[Scenario]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute all scenarios sequentially through orchestrator."""
        if scenarios is None:
            scenarios = SCENARIOS

        results = []
        for scenario in scenarios:
            result = await self.execute_scenario_async(scenario)
            results.append(result)
        return results

    def execute_all_scenarios(
        self,
        scenarios: Optional[List[Scenario]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute all scenarios synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = self.execute_all_scenarios_async(scenarios)
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return asyncio.run(coro)


def print_orchestrator_integration_results(results: List[Dict[str, Any]]) -> None:
    """Print orchestrator execution results with trace summaries."""
    print("\n" + "=" * 80)
    print("PHASE 4A: ORCHESTRATOR INTEGRATION RESULTS")
    print("=" * 80)

    by_category = {}
    for result in results:
        cat = result.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = {"pass": 0, "fail": 0}
        
        if result["execution"].get("completed"):
            by_category[cat]["pass"] += 1
        else:
            by_category[cat]["fail"] += 1

    print("\n📊 EXECUTION SUMMARY:")
    for cat in ["planning", "tool", "memory", "ux", "robustness"]:
        counts = by_category.get(cat, {"pass": 0, "fail": 0})
        total = counts["pass"] + counts["fail"]
        pct = (counts["pass"] / total * 100) if total > 0 else 0
        status = "✓" if counts["pass"] == total else "⚠" if counts["pass"] > 0 else "✗"
        print(f"  {status} {cat:12s}: {counts['pass']:2d}/{total:2d} ({pct:5.1f}%)")

    print("\n📋 EXECUTION DETAILS:")
    for result in results:
        scenario = result["scenario"]
        error = result["execution"].get("error")
        status = "✓" if result["execution"].get("completed") else "✗"
        error_msg = f" | {error}" if error else ""
        print(f"  {status} {scenario:35s}{error_msg}")

    print("\n🔍 TRACE SUMMARY:")
    for result in results:
        if result["trace"]:
            scenario = result["scenario"]
            trace = result["trace"]
            planner = trace.get("planner", {})
            tools = trace.get("tools", {})
            memory = trace.get("memory", {})
            
            print(f"\n  {scenario}:")
            print(f"    Planner: {planner.get('plan_length', 0)} steps, "
                  f"{planner.get('goal_count', 0)} goals")
            print(f"    Tools: {tools.get('execution_count', 0)} executed, "
                  f"{tools.get('success_rate', 0):.1%} success")
            print(f"    Memory: {memory.get('memory_key_count', 0)} keys accessed")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    logger.error(
        "⚠️  PHASE 4A REQUIRES PRE-INITIALIZED ORCHESTRATOR\n"
        "Usage: Pass DadBotOrchestrator instance to OrchestratorIntegrationLayer\n"
        "Example:\n"
        "  from dadbot.core.dadbot import DadBot\n"
        "  bot = DadBot()\n"
        "  orchestrator = bot.turn_orchestrator\n"
        "  layer = OrchestratorIntegrationLayer(orchestrator)\n"
        "  results = layer.execute_all_scenarios()\n"
    )
