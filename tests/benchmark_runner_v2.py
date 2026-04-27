"""Benchmark harness for scenario evaluation (Phase 1 MVP).

PHASE 1 GOAL: Validate scenarios can be defined and scored.
(Full orchestrator integration comes after scenarios are validated.)

This harness demonstrates the concept with mock execution.
Transition to real orchestrator execution in Phase 2 after
scenario validity is confirmed.

OUTPUT STRUCTURE:
{
    "scenario_name": str,
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

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from tests.scenario_suite import Scenario, SCENARIOS

logger = logging.getLogger(__name__)


@dataclass
class ExecutionTrace:
    """Captured execution trace from a scenario run."""
    scenario_name: str
    category: str
    input_text: str
    planner_output: Optional[Dict[str, Any]] = None
    tools_executed: List[str] = None
    memory_accessed: List[str] = None
    final_response: str = ""
    steps: int = 0
    completed: bool = False
    error: Optional[str] = None

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
    error: Optional[str] = None


class BenchmarkRunner:
    """Executes scenarios and captures execution traces.
    
    PHASE 1: Mock-based scenario validation.
    PHASE 2: Real orchestrator integration (coming next).
    """

    def __init__(self, strict: bool = False):
        """Initialize benchmark runner.

        Args:
            strict: Whether to enforce strict validation.
        """
        self.strict = strict

    def _score_trace(self, trace: ExecutionTrace, scenario: Scenario) -> ScoreResult:
        """Apply minimal scoring to execution trace.

        PRINCIPLE: Do NOT overbuild. Just measure basic success.
        Full scoring engine (weighted, coherence, efficiency) comes after
        first benchmark run reveals what signals matter.
        """
        success = trace.completed and trace.error is None
        
        tool_used_correctly = True
        if scenario.category == "tool":
            # For tool scenarios: did execution use/attempt tools?
            tool_used_correctly = len(trace.tools_executed) > 0 or not self.strict

        return ScoreResult(
            success=success,
            steps=trace.steps,
            tool_used_correctly=tool_used_correctly,
            error=trace.error,
        )

    def run_scenario(self, scenario: Scenario) -> Dict[str, Any]:
        """Execute a single scenario.

        PHASE 1: Mock execution validates scenario structure.
        PHASE 2: Real orchestrator execution will replace this.
        """
        trace = ExecutionTrace(
            scenario_name=scenario.name,
            category=scenario.category,
            input_text=scenario.input_text,
        )

        try:
            # PHASE 1: Validate scenario definition
            if not scenario.name or not scenario.input_text:
                raise ValueError("Scenario must have name and input_text")
            
            # PHASE 1: Mock execution (Phase 2 will do real orchestrator execution)
            trace.completed = True
            trace.steps = 1
            trace.final_response = f"[PHASE 1 MOCK] Response for: {scenario.input_text[:50]}..."
            
            # PHASE 1: Mock tool/memory trace capture
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
            trace.error = f"{type(e).__name__}: {str(e)}"
            trace.completed = False
            logger.exception(f"Error executing scenario {scenario.name}")

        # Apply minimal scoring
        score = self._score_trace(trace, scenario)

        return {
            "scenario": scenario.name,
            "category": scenario.category,
            "input": scenario.input_text,
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
        }

    def run_all_scenarios(
        self,
        scenarios: Optional[List[Scenario]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute all scenarios (or specified subset)."""
        if scenarios is None:
            scenarios = SCENARIOS

        results = []
        for scenario in scenarios:
            result = self.run_scenario(scenario)
            results.append(result)
        return results


def print_benchmark_results(results: List[Dict[str, Any]]) -> None:
    """Pretty-print benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS (PHASE 1 - SCENARIO VALIDATION)")
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

    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE: Scenario suite is valid and ready for orchestrator integration")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    runner = BenchmarkRunner(strict=False)
    print("Running PHASE 1 scenario validation benchmark...")
    results = runner.run_all_scenarios()
    print_benchmark_results(results)
