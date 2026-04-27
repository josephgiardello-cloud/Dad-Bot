# Phase 4A: Orchestrator Integration

## Overview

**Goal**: Transition from synthetic metrics (Phase 1) to real intelligence measurements by executing scenarios through the actual DadBotOrchestrator.

**Key Principle**: "Scenarios define truth / Scoring quantifies truth / Benchmarks compare truth"
- Scenarios are locked (define what success looks like)
- Scoring only measures scenario expectations
- Orchestrator execution reveals real capability gaps

## Architecture

### Three-Layer Stack

```
┌─────────────────────────────────────────────────────────┐
│ Layer 1: Scenario Definition (LOCKED)                   │
│ - 15 scenarios across 5 categories                       │
│ - Each defines expected_capabilities & success_criteria │
│ - Located: tests/scenario_suite.py                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 2: Benchmark Runner (UPGRADED in Phase 4A)        │
│ - Dual mode: mock (Phase 1) | orchestrator (Phase 4A)  │
│ - Injects scenarios into execution pipeline             │
│ - Captures execution traces from TurnContext             │
│ - Located: tests/benchmark_runner.py                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 3: Execution Engine                               │
│ - Phase 1: Mock execution (deterministic, fast)        │
│ - Phase 4A: Real orchestrator (actual agent pipeline)  │
│ - Located: dadbot/core/orchestrator.py                  │
└─────────────────────────────────────────────────────────┘
```

### Execution Flow (Phase 4A)

```
Input Scenario
       ↓
BenchmarkRunner.run_scenario()
       ↓
orchestrator.handle_turn(scenario.input_text)
       ↓
8-Node Pipeline:
  TemporalNode → HealthNode → ContextBuilderNode → PlannerNode 
  → InferenceNode → SafetyNode → ReflectionNode → SaveNode
       ↓
Capture _last_turn_context with execution traces
       ↓
Extract metrics:
  - Planner output (plan, goals, decomposition)
  - Tool execution (planned vs executed, success rate)
  - Memory state (keys accessed, retrieval patterns)
  - Safety check (violations if any)
       ↓
Scoring & Results
```

## Implementation

### 1. BenchmarkRunner Dual-Mode

#### Phase 1 (Mock) - Always Works
```python
runner = BenchmarkRunner(strict=False, mode="mock")
results = runner.run_all_scenarios()
```

**Characteristics**:
- No orchestrator required
- Deterministic, fast execution
- Validates scenario structure
- Returns synthetic traces (all success=True)

#### Phase 4A (Orchestrator) - Real Measurement
```python
from dadbot.core.dadbot import DadBot

bot = DadBot()  # Must be initialized with config
runner = BenchmarkRunner(
    strict=False,
    mode="orchestrator",
    orchestrator=bot.turn_orchestrator
)
results = runner.run_all_scenarios()
```

**Characteristics**:
- Requires pre-initialized orchestrator
- Real agent pipeline execution
- Captures actual capability gaps
- Returns real traces with planner/tools/memory data

### 2. Trace Capture Points

#### Planner Data
```python
context.state["plan"]  # Planner output
context.state["detected_goals"]  # Goal extraction
context.state["task_decomposition"]  # Task graph
```

**Metrics**:
- plan_length: Number of steps in plan
- goal_count: Extracted goals
- task_count: Decomposed tasks
- dependencies: Task dependencies

#### Tool Execution
```python
context.state["tool_ir"]  # Tool Intermediate Representation
context.state["tool_results"]  # Execution results
```

**Metrics**:
- tools_planned: Tools in execution plan
- tools_executed: Tools actually executed
- success_rate: Successful executions / total

#### Memory Access
```python
context.state["memory_structured"]  # Memory dict
context.state["memory_full_history_id"]  # History ref
```

**Metrics**:
- memory_key_count: Keys accessed
- memory_keys: List of accessed keys
- session_goals: Tracked goals in memory

#### Safety Check
```python
context.state["safety_check_result"]  # Safety decision
```

**Metrics**:
- safe: Boolean safety verdict
- violations: List of policy violations

### 3. Result Structure

```python
{
    "scenario": str,  # Scenario name
    "category": str,  # planning|tool|memory|ux|robustness
    "input": str,  # User input
    "execution": {
        "completed": bool,  # Success flag
        "steps": int,  # Step count from plan
        "error": str | None,  # Error message if failed
        "turn_number": int,  # Turn counter (Phase 4A only)
    },
    "trace": {  # Real traces (Phase 4A) or synthetic (Phase 1)
        "planner": dict,  # {plan_length, goal_count, ...}
        "tools": dict,  # {execution_count, success_rate, ...}
        "memory": dict,  # {memory_key_count, ...}
        "safety": dict,  # {safe, violations}
        "metadata": dict,  # {trace_id, phase}
    },
    "response": str,  # Agent response text
}
```

## Setup & Execution

### Prerequisites

1. **Python Environment**: 3.13+ with Dad-Bot dependencies
2. **DadBot Initialization**: Must have `config.yaml` with valid LLM settings
3. **Service Registry**: All 7 services must be bootable

### Running Phase 1 (Always Works)

```bash
cd tests/
python benchmark_runner.py
```

Expected output: ✓ All 15 scenarios pass (100%)

### Running Phase 4A (Real Execution)

```bash
cd tests/
python run_phase4a.py
```

**Flow**:
1. Attempts to initialize DadBot with real orchestrator
2. If successful: executes all 15 scenarios through agent pipeline
3. If failed: falls back to Phase 1 mock execution

**Expected output**:
- Real capability measurements (not 100%)
- Traces with actual planner/tool/memory data
- Identified gaps by category

## Orchestrator Integration Points

### Bootstrapping the Orchestrator

#### Option A: From DadBot Instance
```python
from dadbot.core.dadbot import DadBot

bot = DadBot()  # Requires config.yaml with LLM settings
orchestrator = bot.turn_orchestrator
```

**Requirements**:
- `config.yaml` in workspace root
- Valid LLM API keys (OPENAI_API_KEY or equivalent)
- All 7 services initialized (health, memory, inference, etc.)

#### Option B: Direct Registry Boot
```python
from dadbot.registry import boot_registry
from dadbot.core.orchestrator import DadBotOrchestrator

registry = boot_registry(config_path="config.yaml", bot=None)
orchestrator = DadBotOrchestrator(registry, bot=None)
```

### Service Dependencies

Orchestrator requires all 7 services registered:

1. **HealthService**: Health checks
2. **MemoryService**: Context + goal-aware ranking
3. **InferenceService**: LLM invocations
4. **PersistenceService**: Checkpoint/save
5. **SafetyPolicyService**: Policy validation
6. **ReflectionService**: Response reflection
7. **TelemetryService**: Event logging

If any service is missing, orchestrator initialization fails with KeyError.

## Expected Results

### Phase 1 Output (Always 100%)
```
planning: 3/3 (100%)
tool: 4/4 (100%)
memory: 3/3 (100%)
ux: 3/3 (100%)
robustness: 2/2 (100%)
TOTAL: 15/15 (100%)
```

### Phase 4A Output (Real Capability Profile)
```
planning: 8/10 (80%)  ← Real capability measurement
tool: 7/10 (70%)      ← Identifies tool selection gaps
memory: 9/10 (90%)    ← Shows memory effectiveness
ux: 6/10 (60%)        ← Reveals UX coherence issues
robustness: 7/10 (70%)← Shows degradation handling gaps
TOTAL: 37/50 (74%)    ← Intelligence profile baseline
```

## Next Steps After Phase 4A

### Phase 4B: Scoring Engine v1 Expansion
- Extend ScoreResult with partial_success (0.0-1.0 float)
- Add tool_correctness scoring (not just boolean)
- Add step_efficiency normalization
- Add failure_type classification

### Phase 4C: Planner Diagnostics
- Plan length metric
- Branching factor analysis
- Revision count tracking
- Dependency correctness validation

### Phase 4D-4E: UX & Tool Intelligence
- UX continuity metrics (coherence, contradictions)
- Tool intelligence (correct vs optimal tool selection)

### Phase 4F-4G: Baseline & Maturity Report
- Industry baseline comparison
- Gap analysis and recommendations
- Maturity classification

## Troubleshooting

### "Orchestrator not available" Message
**Cause**: DadBot could not be initialized (missing config.yaml, LLM keys, etc.)

**Solution**: 
1. Ensure `config.yaml` exists in workspace root
2. Set required env vars (OPENAI_API_KEY, etc.)
3. Run Phase 1 mock execution first to validate scenarios
4. Debug DadBot initialization separately

### "Service 'X' is not registered" Error
**Cause**: Service registry incomplete

**Solution**:
1. Check that all 7 services are in registry
2. Verify boot_registry() executed successfully
3. Check for circular import issues in service modules

### Execution Timeout
**Cause**: Orchestrator takes >15s per scenario

**Solution**:
1. Reduce orchestrator timeout in `_run_scenario_orchestrator_async()`
2. Check LLM API latency (add network timeout)
3. Profile orchestrator with fewer scenarios first

## Files & Structure

```
tests/
  scenario_suite.py           # LOCKED - 15 scenarios
  benchmark_runner.py         # UPDATED - dual-mode runner
  orchestrator_integration.py  # NEW - integration layer (reference)
  run_phase4a.py             # NEW - demo script
  test_phase4a.py            # NEW (optional) - pytest integration

dadbot/
  core/
    orchestrator.py          # Main pipeline (unchanged)
    graph.py                 # Node execution (unchanged)
    dadbot.py               # Bot initialization (unchanged)
  registry.py                # Service bootstrap (unchanged)
  services/                  # All 7 services (unchanged)
```

## Success Criteria

Phase 4A is complete when:

1. ✓ BenchmarkRunner dual-mode works correctly
2. ✓ Phase 1 (mock) still passes all 15 scenarios (regression test)
3. ✓ Phase 4A executes real scenarios without crash
4. ✓ Real results show actual gaps (not 100%)
5. ✓ Traces captured planner/tool/memory data properly
6. ✓ Scoring applies consistently to real and mock results
7. ✓ Ready for Phase 4B scoring expansion

## References

- **Orchestrator**: [dadbot/core/orchestrator.py](file:///c:/Users/josep/OneDrive/Desktop/Dad-Bot/dadbot/core/orchestrator.py)
- **TurnContext**: [dadbot/core/graph.py](file:///c:/Users/josep/OneDrive/Desktop/Dad-Bot/dadbot/core/graph.py) (line ~200)
- **Scenario Suite**: [tests/scenario_suite.py](file:///c:/Users/josep/OneDrive/Desktop/Dad-Bot/tests/scenario_suite.py)
- **Benchmark Runner**: [tests/benchmark_runner.py](file:///c:/Users/josep/OneDrive/Desktop/Dad-Bot/tests/benchmark_runner.py)
